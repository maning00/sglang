"""Reservation-backed FULL PageSource; reuse the existing paged kernels/frees."""

from contextlib import contextmanager
from dataclasses import replace

import torch

from sglang.srt.mem_cache.allocator.paged import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.mem_cache.kv_memory_provider import (
    EventReceipt,
    JoinedReceipt,
    ProviderFailStop,
    indices_ready,
    refund_request_reservation,
    reservation_request_id,
)
from sglang.srt.mem_cache.provider_indices import (
    attach_index_cpu,
    attach_page_ids,
    cat_indices,
    copy_index_metadata,
    index_cpu,
)
from sglang.srt.mem_cache.utils import get_num_new_pages
from sglang.srt.utils.device import async_h2d


class ProviderPageSource(PagedTokenToKVPoolAllocator):
    def __init__(self, *args, provider, **kwargs):
        self.provider = provider
        self._initializing = True
        self._active = ()
        self._reset_lease = None
        super().__init__(*args, **kwargs)
        self._initializing = False

    def available_size(self):
        if self._active:
            return sum(r.remaining(p) for r, p in self._active) * self.page_size
        return self.provider.accounting().cached * self.page_size

    def get_all_free_pages(self):
        # Device free_pages is only the active allocation's authorized prefix.
        return self.free_pages

    def merge_and_sort_free(self):
        pass

    @contextmanager
    def reservation_scope(self, reservation, purpose):
        if reservation is None:
            raise RuntimeError("FULL allocation requires a request reservation")
        previous = self._active
        self._active = ((reservation, purpose),)
        try:
            yield
        finally:
            self._active = previous

    @contextmanager
    def batch_reservation_scope(self, reqs, prefix_lens, seq_lens):
        if self._active:
            raise RuntimeError("nested FULL batch allocation")
        try:
            for req, prefix, end in zip(
                reqs, prefix_lens.tolist(), seq_lens.tolist(), strict=True
            ):
                reservation = req.kv.reservation
                n = (end + self.page_size - 1) // self.page_size - (
                    prefix + self.page_size - 1
                ) // self.page_size
                if reservation is None:
                    raise RuntimeError("batch extend lacks a KV reservation")
                expected = replace(
                    reservation.shape,
                    request_id=reservation_request_id(req),
                    revision=req.kv.reservation_revision,
                    prefix_tokens=prefix,
                    end_tokens=end,
                    extend_pages=n,
                )
                if (
                    not reservation.covers(expected)
                    or reservation.remaining("extend") != n
                    or reservation.remaining("load")
                ):
                    raise RuntimeError("batch extend does not match its KV reservation")
            self._active = tuple((req.kv.reservation, "extend") for req in reqs)
            yield
        finally:
            self._active = ()
            for req in reqs:
                refund_request_reservation(req)

    @contextmanager
    def _materialize(self, n):
        if not self._active:
            raise RuntimeError("FULL allocation outside a reservation scope")
        if sum(r.remaining(p) for r, p in self._active) < n:
            raise RuntimeError("FULL reservation exhausted")
        handles, left = [], n
        for reservation, purpose in self._active:
            count = min(left, reservation.remaining(purpose))
            handles.extend(reservation.take(purpose, count))
            left -= count
        page_ids = [h.page_id for h in handles]
        if (
            len(page_ids) != n
            or len(set(page_ids)) != n
            or any(p < 1 or p > self.num_pages for p in page_ids)
        ):
            raise ProviderFailStop("provider issued invalid or duplicate FULL pages")
        self.free_pages = async_h2d(page_ids, dtype=torch.int64, device=self.device)
        try:
            yield tuple(page_ids)
            if self.free_pages.numel():
                raise RuntimeError("paged kernel did not consume its authorized prefix")
        except BaseException:
            self.provider.release_pages(handles, ready=indices_ready(self.free_pages))
            raise
        finally:
            self.free_pages = torch.empty(0, dtype=torch.int64, device=self.device)

    def alloc(self, need_size):
        if need_size % self.page_size:
            raise ValueError("FULL load must allocate whole pages")
        with self._materialize(need_size // self.page_size) as pages:
            out = super().alloc(need_size)
            return attach_page_ids(out, pages, self.page_size)

    def alloc_extend(
        self,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        num_new_pages=None,
    ):
        n = get_num_new_pages(
            seq_lens=seq_lens_cpu, page_size=self.page_size, prefix_lens=prefix_lens_cpu
        )
        if num_new_pages is not None and num_new_pages != n:
            raise RuntimeError("incorrect FULL page demand")
        with self._materialize(n) as pages:
            out = super().alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages=n,
            )
            prefixes, ends = prefix_lens_cpu.tolist(), seq_lens_cpu.tolist()
            tails = None
            if any(p % self.page_size and p < e for p, e in zip(prefixes, ends)):
                tails = index_cpu(last_loc).reshape(-1).tolist()
            # Reuse the CPU-vectorized reference allocator rather than building
            # a Python integer/range for every token of every chunk.
            cpu = torch.empty((extend_num_tokens,), dtype=torch.int64, device="cpu")
            cpu_last = (
                index_cpu(last_loc)
                if tails is not None
                else torch.full((len(prefixes),), -1, dtype=torch.int64, device="cpu")
            )
            alloc_extend_naive(
                prefix_lens_cpu,
                seq_lens_cpu,
                cpu_last,
                torch.tensor(pages, dtype=torch.int64, device="cpu"),
                cpu,
                self.page_size,
                "cpu",
            )
            used, cursor = [], 0
            for i, (prefix, end) in enumerate(zip(prefixes, ends)):
                if prefix % self.page_size and prefix < end:
                    used.append(tails[i] // self.page_size)
                count = (end + self.page_size - 1) // self.page_size - (
                    prefix + self.page_size - 1
                ) // self.page_size
                used.extend(pages[cursor : cursor + count])
                cursor += count
            attach_index_cpu(out, cpu)
            out._kv_provider_page_ids = tuple(dict.fromkeys(used))
            return out

    def alloc_decode(self, *args, **kwargs):
        raise RuntimeError("KV page provider supports prefill only")

    def _pages_for_free(self, indices, *, segment=False):
        cpu = index_cpu(indices)
        pages = (
            cpu[:: self.page_size] // self.page_size
            if segment
            else torch.unique(cpu // self.page_size)
        )
        pages._kv_provider_release_ready = indices_ready(indices)
        return pages

    def free_page_ids(self, page_ids):
        cpu = index_cpu(page_ids)
        cpu._kv_provider_release_ready = getattr(
            page_ids, "_kv_provider_release_ready", None
        ) or indices_ready(page_ids)
        return super().free_page_ids(cpu)

    @staticmethod
    def _cat_free_group(indices):
        return cat_indices(indices)

    @staticmethod
    def _copy_for_free_group(indices):
        return copy_index_metadata(indices, indices.clone())

    def _release_page_ids(self, *page_ids):
        ids = torch.cat(page_ids) if len(page_ids) != 1 else page_ids[0]
        if not ids.numel():
            return
        page_list = index_cpu(ids).reshape(-1).tolist()
        if len(set(page_list)) != len(page_list):
            raise RuntimeError("duplicate FULL physical return")
        # The provider owns PENDING_FREE and all retire ACKs. Never stage pages
        # in the base allocator before they have authoritative reuse permission.
        guard = JoinedReceipt(
            tuple(
                getattr(part, "_kv_provider_release_ready", None) or indices_ready(part)
                for part in page_ids
            )
            + (
                (EventReceipt(self._reuse_guard_event),)
                if self._reuse_guard_event is not None
                else ()
            )
        )
        self.provider.release_page_ids(page_list, ready=guard)

    def authorize_reset(self, lease):
        self.provider.validate_reset(lease)
        approved = lease.approved_pages
        if len(set(approved)) != len(approved) or any(
            not 1 <= p <= self.num_pages for p in approved
        ):
            raise RuntimeError("invalid provider reset approval set")
        self._reset_lease = lease

    def validate_clear(self):
        if not self._initializing:
            if self._reset_lease is None:
                raise RuntimeError("provider clear requires a quiesced reset lease")
            self.provider.validate_reset(self._reset_lease)

    def clear(self, *, reset_lease=None):
        if reset_lease is not None:
            self.authorize_reset(reset_lease)
        self.validate_clear()
        # The provider's approved set is rebuilt by commit_reset; there is no
        # independent arange/free list that could revive quarantined pages.
        self.free_pages = torch.empty(0, dtype=torch.int64, device=self.device)
        self.free_group = None
        self.free_page_ids_group = []
        self.staged_pages = []
        self.num_staged_pages = 0
        self._reuse_guard_event = self._staged_guard_event = None

    def finish_reset(self):
        self._reset_lease = None

    def resize(self, config):
        raise RuntimeError("KV page provider does not support resize")
