"""Private owner state; remote access is mediated by the native owner service."""

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
    owner = Path(__file__).resolve().parent / "native"
    key = hashlib.sha256(
        b"".join(
            path.read_bytes()
            for path in (
                source / "directory.cpp",
                source / "directory.h",
                owner / "owner.cpp",
                owner / "owner.h",
            )
        )
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
                        "-pthread",
                        "-I",
                        str(source),
                        str(source / "directory.cpp"),
                        str(owner / "owner.cpp"),
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
        "states": [ptr, ptr, ptr, ptr, size],
        **{name: [ptr] for name in ("refs", "freeze", "thaw")},
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


class L1Directory:
    def __init__(self, rows: int, pin_cap: int):
        self.num_rows, self.pin_cap = rows, pin_cap
        self._native = _native()
        self._mapping = mmap.mmap(
            -1, self._native.l1_bytes(rows), flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        self._native.l1_init(self.address, rows, pin_cap)

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
