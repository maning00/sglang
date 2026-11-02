"""Native owner RPCs and a callback-safe, acknowledged cleanup queue."""

from __future__ import annotations

import ctypes
import math
import time
import uuid
from enum import IntEnum
from functools import lru_cache

import numpy as np

from .directory import _native


# Keep aligned with native/owner.h::Operation.
class Operation(IntEnum):
    LOOKUP = 1
    ACQUIRE = 2
    RELEASE = 3
    CANCEL = 4
    CONTAINS = 5
    CLEANUPS = 6
    ACQUIRE_EXACT = 7


@lru_cache(maxsize=1)
def native():
    lib = _native()
    ptr, u64, size = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t
    string, integer = ctypes.c_char_p, ctypes.c_int
    signatures = {
        "owner_start": [ptr, string, u64, u64, integer],
        "owner_stop": [ptr],
        "owner_publish": [ptr, ptr, ptr, size],
        "owner_retire": [ptr, ptr, ptr, size],
        "owner_rpc": [string, u64, u64, u64, u64, ptr, ptr, size, integer],
        "cleanup_start": [integer],
        "cleanup_stop": [ptr],
        "cleanup_pending": [ptr],
        "cleanup_create": [ptr, string, u64, u64, u64, u64],
        "cleanup_enqueue": [ptr],
    }
    for name, args in signatures.items():
        function = getattr(lib, f"l1_{name}")
        function.argtypes = args
        function.restype = ctypes.c_int64
    lib.l1_owner_error.restype = ctypes.c_char_p
    lib.l1_cleanup_enqueue.restype = None
    return lib


def checked(result):
    if result < 0:
        raise RuntimeError(native().l1_owner_error().decode())
    return result


def full_keys(keys):
    value = bytes.fromhex("".join(keys))
    if len(value) != len(keys) * 32:
        raise ValueError("Peer L1 ownership requires complete SHA-256 keys")
    return np.frombuffer(value, dtype="<u8").reshape(-1, 4)


class OwnerService:
    def __init__(self, directory, *, lease_s=30.0):
        self.directory = directory
        self.endpoint = "sglang-l1-" + uuid.uuid4().hex
        self.epoch = uuid.uuid4().int & ((1 << 64) - 1)
        self.address = checked(
            native().l1_owner_start(
                directory.address,
                self.endpoint.encode(),
                self.epoch,
                directory.pin_cap,
                math.ceil(lease_s * 1000),
            )
        )

    def publish(self, keys, rows):
        keys, rows = full_keys(keys), np.ascontiguousarray(rows, dtype=np.int64)
        if len(keys) != len(rows):
            raise ValueError("Peer L1 publication needs one key per row")
        checked(
            native().l1_owner_publish(
                self.address, keys.ctypes.data, rows.ctypes.data, len(keys)
            )
        )

    def retire(self, keys, *, aligned=False):
        keys = full_keys(keys)
        rows = np.empty(len(keys), dtype=np.int64)
        checked(
            native().l1_owner_retire(
                self.address, keys.ctypes.data, rows.ctypes.data, len(keys)
            )
        )
        return rows if aligned else rows[rows >= 0]

    def close(self):
        if self.address:
            native().l1_owner_stop(self.address)
            self.address = 0


class CleanupQueue:
    def __init__(self, *, timeout_s):
        self.address = checked(native().l1_cleanup_start(math.ceil(timeout_s * 1000)))

    def prepare(self, peer, seq, *, cancel=False):
        return checked(
            native().l1_cleanup_create(
                self.address,
                peer.endpoint,
                Operation.CANCEL if cancel else Operation.RELEASE,
                peer.epoch,
                peer.reader,
                seq,
            )
        )

    def enqueue(self, item):
        native().l1_cleanup_enqueue(item)

    @property
    def pending(self):
        return native().l1_cleanup_pending(self.address)

    def close(self):
        native().l1_cleanup_stop(self.address)


class OwnerPeer:
    def __init__(self, *, endpoint, epoch, pin_cap, cleanup, timeout_s, cooldown_s=1.0):
        self.endpoint, self.epoch, self.pin_cap = endpoint.encode(), epoch, pin_cap
        self.cleanup = cleanup
        self.timeout_ms = math.ceil(timeout_s * 1000)
        self.cooldown_s = cooldown_s
        self.reader = uuid.uuid4().int & ((1 << 64) - 1)
        self.seq = 0
        self.retry_after = 0.0

    def request(self, op, keys, *, seq=0):
        rows = np.empty(len(keys), dtype=np.int64)
        n = native().l1_owner_rpc(
            self.endpoint,
            op,
            self.epoch,
            self.reader,
            seq,
            keys.ctypes.data,
            rows.ctypes.data,
            len(keys),
            self.timeout_ms,
        )
        return None if n < 0 else rows[:n]

    def available(self):
        return time.monotonic() >= self.retry_after

    def lookup(self, keys):
        if not self.available():
            return 0
        rows = self.request(Operation.LOOKUP, keys[: self.pin_cap])
        if rows is None:
            self.retry_after = time.monotonic() + self.cooldown_s
            return 0
        return len(rows)

    def acquire(self, keys):
        if not self.available():
            return None
        self.seq += 1
        rows = self.request(Operation.ACQUIRE_EXACT, keys, seq=self.seq)
        if rows is None:
            self.cleanup.enqueue(self.cleanup.prepare(self, self.seq, cancel=True))
            self.retry_after = time.monotonic() + self.cooldown_s
            return None
        if len(rows) != len(keys):
            return None
        return rows, self.cleanup.prepare(self, self.seq)

    def contains(self, keys):
        found = self.request(Operation.CONTAINS, keys)
        return None if found is None else found.astype(bool)
