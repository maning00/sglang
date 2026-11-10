"""CPU geometry for provider index tensors; never allocation/reuse authority.

Storage weak refs make ordinary views (also those returned through tch) inherit
geometry without keeping device allocations alive. Copies/cats explicitly
register their CPU result. Request rows keep only their populated CPU prefix.
"""

from itertools import islice

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from sglang.srt.runtime_context import get_resources


def _entries():
    return get_resources().kv_provider_index_mirrors


def _key(tensor):
    return tensor.untyped_storage()._cdata, tensor.dtype


class _Dense:
    def __init__(self, tensor, cpu):
        self.offset = tensor.storage_offset()
        # Device copies are snapshots too. A request-row mirror is mutable;
        # sharing its CPU storage here would silently change cloned geometry.
        self.cpu = cpu.reshape(-1).clone()

    def read(self, tensor):
        offset = tensor.storage_offset() - self.offset
        end = offset + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
        if offset < 0 or end >= self.cpu.numel():
            return None
        return self.cpu.as_strided(
            tensor.shape, tensor.stride(), self.cpu.storage_offset() + offset
        )


class RowIndexMirror:
    def __init__(self, table):
        self.width = table.shape[1]
        self.offset = table.storage_offset()
        self.rows = {}
        self.key = _key(table)
        self.ensure_registered(table)

    def ensure_registered(self, table):
        if _key(table) != self.key:
            raise RuntimeError("provider request table storage changed")
        table._provider_row_mirror = self
        root = table
        while getattr(root, "_base", None) is not None:
            root = root._base
        if _key(root) == self.key:
            root._provider_row_mirror = self
        entry = _entries().get(self.key)
        if entry is None or entry[0].expired() or entry[1] is not self:
            _register(table, self)

    def read(self, tensor):
        if tensor.ndim != 1 or tensor.stride(0) != 1:
            return None
        row, col = divmod(tensor.storage_offset() - self.offset, self.width)
        cpu = self.rows.get(row)
        end = col + tensor.numel()
        return None if cpu is None or end > cpu.numel() else cpu[col:end]

    def write(self, row, prefix, tail):
        self.rows[row] = torch.cat((index_cpu(prefix), index_cpu(tail)))

    def write_slice(self, indices, values):
        row, span = indices
        if not isinstance(row, int) or not isinstance(span, slice):
            raise RuntimeError("provider request row writes require CPU row/range")
        cpu = index_cpu(values).reshape(-1)
        start = span.start or 0
        end = start + cpu.numel()
        old = self.rows.get(row, torch.empty(0, dtype=torch.int64, device="cpu"))
        if span.step not in (None, 1) or start > old.numel():
            raise RuntimeError("provider CPU request row has an unknown gap")
        if end > old.numel():
            old = torch.cat(
                (old, torch.empty(end - old.numel(), dtype=torch.int64, device="cpu"))
            )
            self.rows[row] = old
        old[start:end].copy_(cpu)


def _register(tensor, mirror):
    entries = _entries()
    root = tensor
    while getattr(root, "_base", None) is not None:
        root = root._base
    row_mirror = getattr(root, "_provider_row_mirror", None)
    if isinstance(row_mirror, RowIndexMirror) and row_mirror is not mirror:
        raise RuntimeError("provider mutable row mirror cannot be replaced")
    old = entries.get(_key(tensor))
    if (
        old is not None
        and not old[0].expired()
        and isinstance(old[1], RowIndexMirror)
        and old[1] is not mirror
    ):
        raise RuntimeError("provider mutable row mirror cannot be replaced")
    # A bounded sweep, including live entries, prevents a long-lived root
    # from hiding expired temporary storage behind it.
    for key in list(islice(entries, 8)):
        weak, old = entries.pop(key)
        if not weak.expired():
            entries[key] = (weak, old)
    entries[_key(tensor)] = (StorageWeakRef(tensor.untyped_storage()), mirror)


def find_index_cpu(tensor):
    if not tensor.numel():
        return torch.empty(tensor.shape, dtype=torch.int64, device="cpu")
    entry = _entries().get(_key(tensor))
    if entry is None or entry[0].expired():
        root = tensor
        while getattr(root, "_base", None) is not None:
            root = root._base
        mirror = getattr(root, "_provider_row_mirror", None)
        if isinstance(mirror, RowIndexMirror):
            mirror.ensure_registered(root)
            entry = _entries().get(_key(tensor))
    if entry is None or entry[0].expired():
        return None
    return entry[1].read(tensor)


def index_cpu(tensor):
    cpu = find_index_cpu(tensor)
    if cpu is not None:
        return cpu
    if tensor.device.type == "cpu":
        return tensor.detach().to(dtype=torch.int64)
    # Exceptional caller-provided indices with no CPU provenance. Enqueue a
    # pinned copy, then wait only when exact geometry is actually required.
    cpu = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
    cpu.copy_(tensor.detach(), non_blocking=True)
    torch.cuda.current_stream(tensor.device).record_event().synchronize()
    # A partial/2D miss on a mutable row mirror must not freeze that storage
    # into a dense snapshot. The same rule preserves any existing provenance.
    if tensor.is_contiguous() and not any(
        key[0] == _key(tensor)[0] and not entry[0].expired()
        for key, entry in _entries().items()
    ):
        attach_index_cpu(tensor, cpu)
    return cpu


def attach_index_cpu(tensor, cpu):
    if cpu.device.type != "cpu":
        raise ValueError("provider index mirror must use CPU storage")
    if not tensor.is_contiguous() or tensor.numel() != cpu.numel():
        raise ValueError("provider index mirror needs matching contiguous geometry")
    entry = _entries().get(_key(tensor))
    if entry is not None and not entry[0].expired():
        known = entry[1].read(tensor)
        if known is not None:
            if not torch.equal(known, cpu.to(dtype=torch.int64).reshape(tensor.shape)):
                raise ValueError(
                    "provider CPU index mirror disagrees with supplied geometry"
                )
            return tensor
        if isinstance(entry[1], RowIndexMirror):
            return tensor
    if entry is None or entry[0].expired() or entry[1].read(tensor) is None:
        _register(tensor, _Dense(tensor, cpu.to(dtype=torch.int64)))
    return tensor


def ensure_row_mirror(pool):
    mirror = getattr(pool, "_provider_row_mirror", None)
    if mirror is None:
        mirror = RowIndexMirror(pool.req_to_token)
        pool._provider_row_mirror = mirror
    else:
        mirror.ensure_registered(pool.req_to_token)
    return mirror


def reject_unmirrored_write(table, operation):
    root = table
    while getattr(root, "_base", None) is not None:
        root = root._base
    entry = _entries().get(_key(table))
    if isinstance(getattr(root, "_provider_row_mirror", None), RowIndexMirror) or (
        entry is not None
        and not entry[0].expired()
        and isinstance(entry[1], RowIndexMirror)
    ):
        raise RuntimeError(
            f"provider rejects unmirrored request-table writer: {operation}"
        )


def attach_page_ids(tensor, pages, page_size):
    pages = tuple(pages)
    cpu = (
        torch.tensor(pages, dtype=torch.int64, device="cpu")[:, None] * page_size
        + torch.arange(page_size, device="cpu")
    ).reshape(-1)
    if cpu.numel() != tensor.numel():
        raise ValueError("provider page geometry does not cover index tensor")
    attach_index_cpu(tensor, cpu)
    tensor._kv_provider_page_ids = pages
    return tensor


def copy_index_metadata(source, target):
    cpu = find_index_cpu(source)
    if cpu is not None:
        attach_index_cpu(target, cpu)
    ready = getattr(source, "_kv_provider_release_ready", None)
    if ready is not None:
        target._kv_provider_release_ready = ready
    return target


def cat_indices(tensors):
    tensors = list(tensors)
    out = torch.cat(tensors)
    cpus = [find_index_cpu(tensor) for tensor in tensors]
    if all(cpu is not None for cpu in cpus):
        attach_index_cpu(out, torch.cat(cpus))
    ready = [getattr(tensor, "_kv_provider_release_ready", None) for tensor in tensors]
    if any(r is not None for r in ready):
        from sglang.srt.mem_cache.kv_memory_provider import JoinedReceipt

        out._kv_provider_release_ready = JoinedReceipt(
            tuple(r for r in ready if r is not None)
        )
    return out
