"""Pages whose DMA ownership cannot be revoked are retained until process exit."""

import threading

import torch


class QuarantinedPages:
    def __init__(self, *, page_size, owner):
        self.page_size, self.owner = page_size, owner
        self.pages = set()
        self.withheld = set()
        self.keepalive = []
        self._lock = threading.Lock()

    def mark(self, pages, keepalive):
        from sglang.srt.runtime_context import get_resources

        with self._lock:
            if not self.keepalive:
                get_resources().rdma_quarantines.append(self)
            before = len(self.pages)
            self.pages.update(map(int, pages))
            self.keepalive.append(keepalive)
            return len(self.pages) - before

    @property
    def retained_tokens(self):
        return len(self.withheld) * self.page_size

    def filter_released(self, page_ids):
        if not self.pages:
            return page_ids
        # Only the failure path pays for host filtering; marking happens before
        # the load counter resolves, so request release sees the complete set.
        values = page_ids.cpu().tolist()
        with self._lock:
            self.withheld.update(p for p in values if p in self.pages)
            healthy = [p for p in values if p not in self.pages]
        return torch.tensor(healthy, dtype=page_ids.dtype, device=page_ids.device)


def full_page_quarantine(allocator):
    from .paged import PagedTokenToKVPoolAllocator
    from .swa import SWATokenToKVPoolAllocator

    full = (
        allocator.full_attn_allocator
        if isinstance(allocator, SWATokenToKVPoolAllocator)
        else allocator
    )
    if not isinstance(full, PagedTokenToKVPoolAllocator):
        raise ValueError("RDMA quarantine requires a paged full allocator")
    return full.quarantine
