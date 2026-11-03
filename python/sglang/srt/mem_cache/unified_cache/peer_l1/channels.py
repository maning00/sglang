"""Adapt native and lend owners without changing their cleanup mechanisms."""

from __future__ import annotations

import time

import numpy as np

from .session import Grant


class NativeGrant(Grant, frozen=True, kw_only=True):
    cleanup: int


class NativeOwnerChannel:
    def __init__(self, *, peer, worker_id):
        self.peer, self.worker_id = peer, worker_id

    def lookup(self, keys):
        return self.peer.lookup(keys)

    def contains(self, keys):
        return self.peer.contains(keys)

    def acquire_exact(self, keys):
        started = time.monotonic()
        result = self.peer.acquire(keys)
        if result is None:
            return None
        rows, cleanup = result
        return NativeGrant(
            owner=self.worker_id,
            epoch=self.peer.epoch,
            grant_id=self.peer.seq,
            rows=rows,
            acquired_before=started,
            lease_s=self.peer.lease_s,
            cleanup=cleanup,
        )

    def cancel_unsubmitted(self, seq):
        self.peer.cleanup.enqueue(
            self.peer.cleanup.prepare(self.peer, seq, cancel=True)
        )

    def release(self, grant):
        self.peer.cleanup.enqueue(grant.cleanup)


class LendOwnerChannel:
    def __init__(self, peer):
        self.peer = peer

    def acquire_exact(self, keys):
        started = time.monotonic()
        reply = self.peer.call("ACQUIRE", keys)
        lease = reply.data[0] if reply is not None and reply.data else 0
        if not lease:
            return None
        records = reply.data[1]
        if len(records) != len(keys):
            self.peer.cleanup("UNPIN", lease)
            return None
        return Grant(
            owner=self.peer.worker_id,
            epoch=self.peer.epoch,
            grant_id=lease,
            rows=np.array([r[0] for r in records], dtype=np.int64),
            acquired_before=started,
            lease_s=self.peer.lease_s,
        )

    def cancel_unsubmitted(self, seq):
        self.peer.cleanup("CANCEL_ACQUIRE", seq)

    def release(self, grant):
        self.peer.cleanup("UNPIN", grant.grant_id)
