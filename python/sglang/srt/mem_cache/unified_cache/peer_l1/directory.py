"""Batched native operations on a peer's process-shared directory.

The host-only library deliberately does not initialize a GPU runtime. This lets
the identical pin protocol run in CPU tests and in the HIP stream callback.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import mmap
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _native():
    source = Path(__file__).resolve().parents[4] / "kernels/jit/csrc/peer_l1"
    key = hashlib.sha256(
        (source / "directory.cpp").read_bytes() + (source / "directory.h").read_bytes()
    ).hexdigest()[:20]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    build = cache / "sglang/peer_l1" / key
    build.mkdir(parents=True, exist_ok=True)
    library = build / "directory.so"
    with (build / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            temporary = build / f"directory.{os.getpid()}.so"
            try:
                subprocess.run(
                    [
                        os.environ.get("CXX", "c++"),
                        "-std=c++17",
                        "-O3",
                        "-shared",
                        "-fPIC",
                        str(source / "directory.cpp"),
                        "-o",
                        str(temporary),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                temporary.replace(library)
            finally:
                temporary.unlink(missing_ok=True)
    native = ctypes.CDLL(str(library))
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    signatures = {
        "bytes": [ctypes.c_uint32],
        "init": [ptr, ctypes.c_uint32, ctypes.c_uint64],
        "publish": [ptr, ptr, ptr, size],
        "retire": [ptr, ptr, ptr, size],
        "probe": [ptr, ptr, ptr, ptr, size],
        "pin": [ptr, ptr, ptr, size],
        "unpin": [ptr, ptr, size],
        "states": [ptr, ptr, ptr, ptr, size],
        **{name: [ptr] for name in ("refs", "freeze", "thaw", "rebuild")},
    }
    native.l1_error.restype = ctypes.c_char_p

    def check(result, function, args):
        if result < 0:
            raise RuntimeError(native.l1_error().decode())
        return result

    for name, args in signatures.items():
        function = getattr(native, f"l1_{name}")
        function.argtypes = args
        function.restype = ctypes.c_int64
        function.errcheck = check
    return native


def hash_words(keys) -> np.ndarray:
    """The first 128 SHA-256 bits, with the same byte order on every worker."""
    return np.frombuffer(
        b"".join(bytes.fromhex(key[:32]) for key in keys), dtype="<u8"
    ).reshape(-1, 2)


class L1Directory:
    def __init__(self, path: str, rows: int, pin_cap: int, *, create: bool):
        self.path, self.num_rows, self.pin_cap = path, rows, pin_cap
        self._native = _native()
        self._owner = create
        self._retired = 0
        length = self._native.l1_bytes(rows)
        fd = os.open(path, os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0), 0o600)
        try:
            if create:
                os.ftruncate(fd, length)
            if os.fstat(fd).st_size != length:
                raise ValueError("Peer L1 directory has an incompatible size")
            self._mapping = mmap.mmap(fd, length)
        finally:
            os.close(fd)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        if create:
            self._native.l1_init(self.address, rows, pin_cap)
        else:
            self._native.l1_refs(self.address)  # Validate the header before use.

    def probe(self, hashes: np.ndarray):
        count = min(len(hashes), self.pin_cap)
        rows, gens = np.empty(count, dtype=np.int64), np.empty(count, dtype=np.int64)
        count = self._native.l1_probe(
            self.address, hashes.ctypes.data, rows.ctypes.data, gens.ctypes.data, count
        )
        return rows[:count], gens[:count]

    def pin(self, rows: np.ndarray, gens: np.ndarray) -> bool:
        return bool(
            self._native.l1_pin(
                self.address, rows.ctypes.data, gens.ctypes.data, len(rows)
            )
        )

    def unpin(self, rows: np.ndarray):
        self._native.l1_unpin(self.address, rows.ctypes.data, len(rows))

    def publish(self, keys, rows):
        hashes, rows = hash_words(keys), np.ascontiguousarray(rows, dtype=np.int64)
        if len(hashes) != len(rows):
            raise ValueError("Peer L1 publication needs one hash per row")
        self._native.l1_publish(
            self.address, hashes.ctypes.data, rows.ctypes.data, len(rows)
        )

    def retire(self, keys) -> np.ndarray:
        hashes = hash_words(keys)
        rows = np.empty(len(hashes), dtype=np.int64)
        self._native.l1_retire(
            self.address, hashes.ctypes.data, rows.ctypes.data, len(rows)
        )
        self._retired += len(rows)
        if self._retired >= self.num_rows:
            self._native.l1_rebuild(self.address)
            self._retired = 0
        return rows[rows >= 0]

    def states(self, rows):
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        pins, live = np.empty_like(rows), np.empty_like(rows)
        self._native.l1_states(
            self.address,
            rows.ctypes.data,
            pins.ctypes.data,
            live.ctypes.data,
            len(rows),
        )
        return pins, live

    @property
    def pinned_refs(self) -> int:
        return self._native.l1_refs(self.address)

    def freeze(self) -> bool:
        return bool(self._native.l1_freeze(self.address))

    def thaw(self):
        self._native.l1_thaw(self.address)

    def close(self):
        if self._mapping is None:
            return
        self._mapping.close()
        self._mapping = None
        if self._owner:
            Path(self.path).unlink(missing_ok=True)
