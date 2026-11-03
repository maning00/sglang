"""Pages whose DMA ownership cannot be revoked are retained until process exit."""

import threading

import msgspec
import torch


class DevicePages(msgspec.Struct):
    ids: torch.Tensor
    released: torch.Tensor
    ready: object = None


class QuarantinedPages:
    def __init__(self, *, page_size, owner):
        self.page_size, self.owner = page_size, owner
        self.pages = set()
        self._ordered = []
        self._devices = {}
        self.keepalive = []
        self._lock = threading.Lock()

    def mark(self, pages, keepalive):
        from sglang.srt.runtime_context import get_resources

        with self._lock:
            if not self.keepalive:
                get_resources().rdma_quarantines.append(self)
            before = len(self.pages)
            added = set(map(int, pages)) - self.pages
            self._ordered.extend(sorted(added))
            self.pages.update(added)
            self.keepalive.append(keepalive)
            return len(self.pages) - before

    @property
    def withheld(self):
        with self._lock:
            result = set()
            for state in self._devices.values():
                if state.ready is not None:
                    state.ready.synchronize()
                result.update(
                    p
                    for p, released in zip(self._ordered, state.released.cpu().tolist())
                    if released
                )
            return result

    @property
    def retained_tokens(self):
        return len(self.withheld) * self.page_size

    def filter_released(self, page_ids):
        if not self.pages:
            return page_ids
        with self._lock:
            key = (page_ids.device, page_ids.dtype)
            state = self._devices.get(key)
            if state is not None and state.ready is not None:
                torch.cuda.current_stream(page_ids.device).wait_event(state.ready)
            if state is None or state.ids.numel() != len(self._ordered):
                ids = page_ids.new_tensor(self._ordered)
                released = torch.zeros_like(ids, dtype=torch.bool)
                if state is not None:
                    released[: state.ids.numel()].copy_(state.released)
                state = self._devices[key] = DevicePages(ids, released)
            state.released.logical_or_(torch.isin(state.ids, page_ids))
            healthy = page_ids[~torch.isin(page_ids, state.ids)]
            # Only stats/invariant observation copies the fixed-size membership
            # mask to the host. Dynamic device compaction can still synchronize.
            if page_ids.is_cuda:
                state.ready = torch.cuda.Event()
                state.ready.record(torch.cuda.current_stream(page_ids.device))
            return healthy


def full_page_quarantine(allocator, *, required=True):
    from .paged import PagedTokenToKVPoolAllocator
    from .swa import SWATokenToKVPoolAllocator

    full = (
        allocator.full_attn_allocator
        if isinstance(allocator, SWATokenToKVPoolAllocator)
        else allocator
    )
    if not isinstance(full, PagedTokenToKVPoolAllocator):
        if required:
            raise ValueError("RDMA quarantine requires a paged full allocator")
        return None
    return full.quarantine
