"""The scheduler is the sole owner of allocator page movement."""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.utils import get_num_new_pages

from .pool import LendPool


class LendablePagedAllocator(PagedTokenToKVPoolAllocator):
    @classmethod
    def install(cls, allocator, pool: LendPool, *, forward_stream=None):
        if type(allocator) is not PagedTokenToKVPoolAllocator:
            raise ValueError("decode lend requires the paged full-attention allocator")
        if torch.device(allocator.device).type != "cpu" and forward_stream is None:
            raise ValueError("decode lend requires the forward stream before lending")
        # Preserve allocations made during model initialization.
        result = cls.__new__(cls)
        result.__dict__.update(allocator.__dict__)
        result.lend_pool = pool
        result.lend_forward_stream = forward_stream
        # One batch stays allocator-owned until the id copy and forward finish.
        result._pending_lend = None
        result._pending_lend_returned = False
        result._lend_host = (
            torch.empty(256, dtype=torch.int64, pin_memory=True)
            if torch.device(allocator.device).type != "cpu"
            else None
        )
        return result

    def _pending_pages(self):
        if self._pending_lend is None or self._pending_lend_returned:
            return 0
        return len(self._pending_lend[0])

    def _return_pending_lend(self):
        if self._pending_pages():
            self.free_pages = torch.cat((self.free_pages, self._pending_lend[0]))
            self._pending_lend_returned = True
        # Keep the source and pinned destination alive until the async D2H ends.

    def _publish_ready_lend(self):
        if self._pending_lend is None:
            return
        _, host, copied, forwarded = self._pending_lend
        if not copied.query() or not forwarded.query():
            return
        if not self._pending_lend_returned:
            self.lend_pool.add_free(host.tolist())
        self._pending_lend = None
        self._pending_lend_returned = False

    def get_all_free_pages(self):
        pages = super().get_all_free_pages()
        if self._pending_pages():
            pages = torch.cat((pages, self._pending_lend[0]))
        return pages

    def merge_and_sort_free(self):
        self._return_pending_lend()
        return super().merge_and_sort_free()

    def _append_pages(self, pages):
        # A pinned source makes this H2D asynchronous on the scheduler stream.
        host = torch.tensor(
            pages, dtype=torch.int64, pin_memory=torch.device(self.device).type != "cpu"
        )
        self.free_pages = torch.cat(
            (self.free_pages, host.to(self.device, non_blocking=True))
        )

    def clear(self):
        pool = getattr(self, "lend_pool", None)
        if pool is None:
            return super().clear()
        self._return_pending_lend()
        with pool.lock:
            if pool.writing or pool.pinned:
                raise RuntimeError("Cannot clear decode KV while lent pages are in use")
            draining, pool.draining = pool.draining, True
            pool.reclaim(len(pool.pages))
        try:
            super().clear()
        finally:
            with pool.lock:
                pool.decode_used = 0
                pool.draining = draining

    def available_size(self):
        return (
            super().available_size()
            + (self._pending_pages() + self.lend_pool.assured_reclaimable)
            * self.page_size
        )

    def _ensure_pages(self, count):
        if count <= len(self.free_pages):
            return
        self.merge_and_sort_free()
        if count > len(self.free_pages):
            pages = self.lend_pool.reclaim(count - len(self.free_pages))
            if pages:
                self._append_pages(pages)

    def _prepare(self, count):
        self._ensure_pages(count)
        # Publish a conservative snapshot before kernels can release the GIL.
        # A large allocation closes the read/write gate in this iteration.
        self._update_used(count)

    def _update_used(self, pending=0):
        with self.lend_pool.lock:
            real_free = (
                super().available_size() // self.page_size + self._pending_pages()
            )
            self.lend_pool.decode_used = (
                self.num_pages - real_free - len(self.lend_pool.pages) + pending
            )

    def alloc(self, need_size):
        self._prepare(need_size // self.page_size)
        try:
            return super().alloc(need_size)
        finally:
            self._update_used()

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
        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        self._prepare(num_new_pages)
        try:
            return super().alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens,
                seq_lens_cpu,
                last_loc,
                extend_num_tokens,
                num_new_pages,
            )
        finally:
            self._update_used()

    def alloc_decode(self, seq_lens, seq_lens_cpu, last_loc):
        self._prepare(
            get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=self.page_size, decode=True
            )
        )
        try:
            return super().alloc_decode(seq_lens, seq_lens_cpu, last_loc)
        finally:
            self._update_used()

    def rebalance(self):
        pool = self.lend_pool
        pool.expire()
        self._publish_ready_lend()
        real_free = super().available_size() // self.page_size + self._pending_pages()
        reserve = max(1, int(self.num_pages * pool.idle))
        self._update_used()
        with pool.lock:
            returned = pool.reclaim(min(max(0, reserve - real_free), len(pool.free)))
            draining = pool.draining
            pool.request_demotions(
                max(0, reserve - real_free - len(returned) - len(pool.demoting))
            )
        if returned:
            self._append_pages(returned)
            real_free += len(returned)
        if not draining and self._pending_lend is None and real_free > reserve + 256:
            self.merge_and_sort_free()
            count = min(256, len(self.free_pages) - reserve)
            pages = self.free_pages[:count]
            self.free_pages = self.free_pages[count:]
            if pages.device.type == "cpu":
                pool.add_free(pages.tolist())
            else:
                # Finished requests can still have a redundant forward in flight.
                # A D2H on the scheduler stream alone does not order those writes.
                forwarded = torch.cuda.Event()
                forwarded.record(self.lend_forward_stream)
                host = self._lend_host[:count]
                host.copy_(pages, non_blocking=True)
                copied = torch.cuda.Event()
                copied.record()
                self._pending_lend = (pages, host, copied, forwarded)
        with pool.lock:
            target = max(1, int(len(pool.pages) * pool.watermark))
            pool.request_demotions(max(0, target - len(pool.free) - len(pool.demoting)))
