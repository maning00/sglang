"""A batched native hash table in /dev/shm shared by the prefill ranks."""

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
    source = (
        Path(__file__).resolve().parents[5]
        / "kernels/jit/csrc/peer_l1/lend_directory.cpp"
    )
    key = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    build = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "sglang/peer_l1"
        / key
    )
    build.mkdir(parents=True, exist_ok=True)
    library = build / "lend_directory.so"
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
                        "-pthread",
                        str(source),
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
    lib = ctypes.CDLL(str(library))
    ptr, size = ctypes.c_void_p, ctypes.c_uint64
    for name, args, result in (
        ("bytes", [size], size),
        ("init", [ptr, size], ctypes.c_int),
        ("query", [ptr, ptr, ptr, size], None),
        ("put", [ptr, ptr, ptr, size], ctypes.c_int),
        ("remove", [ptr, ptr, ptr, size], None),
    ):
        func = getattr(lib, f"lend_{name}")
        func.argtypes, func.restype = args, result
    return lib


class LendDirectory:
    def __init__(self, path, capacity, *, create=False):
        self.path, self.owner = Path(path), create
        self.native = _native()
        size = self.native.lend_bytes(capacity)
        fd = os.open(path, os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0), 0o600)
        try:
            if create:
                os.ftruncate(fd, size)
            self.memory = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        self.buffer = (ctypes.c_char * size).from_buffer(self.memory)
        self.ptr = ctypes.addressof(self.buffer)
        if create and self.native.lend_init(self.ptr, capacity):
            raise RuntimeError("Unable to initialize decode lend shared directory")

    @staticmethod
    def _keys(keys):
        raw = b"".join(bytes.fromhex(k) for k in keys)
        if len(raw) != len(keys) * 32:
            raise ValueError("Decode lend directory keys must be SHA-256 hashes")
        return np.frombuffer(raw, dtype=np.uint8)

    def query(self, keys):
        raw = self._keys(keys)
        values = np.empty((len(keys), 3), dtype=np.int64)
        self.native.lend_query(self.ptr, raw.ctypes.data, values.ctypes.data, len(keys))
        return values.tolist()

    def put(self, rank, records):
        if not records:
            return 0
        raw = self._keys([r[2] for r in records])
        values = np.array([[rank, r[0], r[1]] for r in records], dtype=np.int64)
        return self.native.lend_put(
            self.ptr, raw.ctypes.data, values.ctypes.data, len(records)
        )

    def remove(self, rank, records):
        if not records:
            return
        raw = self._keys([r[2] for r in records])
        values = np.array([[rank, r[0], r[1]] for r in records], dtype=np.int64)
        self.native.lend_remove(
            self.ptr, raw.ctypes.data, values.ctypes.data, len(records)
        )

    def close(self):
        del self.buffer
        self.memory.close()
        if self.owner:
            self.path.unlink(missing_ok=True)
