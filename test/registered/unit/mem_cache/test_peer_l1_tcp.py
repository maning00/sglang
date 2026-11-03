"""TCP peers must not serialize behind an incomplete frame or a blocked reply."""

import hashlib
import json
import socket
import struct
import threading
import time
from urllib.parse import urlsplit

import pytest

from sglang.srt.mem_cache.unified_cache.peer_l1.control import (
    Operation,
    OwnerPeer,
    OwnerService,
    full_keys,
)
from sglang.srt.mem_cache.unified_cache.peer_l1.directory import L1Directory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def receive(sock, size):
    result = b""
    while len(result) < size:
        data = sock.recv(size - len(result))
        if not data:
            raise EOFError
        result += data
    return result


def test_client_reuses_connection_and_drops_truncated_reply():
    """A truncated reply must not become the next request's frame after reconnect."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.settimeout(2)
    accepted = []
    errors = []

    def server():
        try:
            for count in (3, 1):
                conn, _ = listener.accept()
                accepted.append(conn)
                with conn:
                    conn.settimeout(2)
                    for i in range(count):
                        header = struct.unpack("=6Q", receive(conn, 48))
                        receive(conn, header[-1] * 32)
                        payload = struct.pack("=2Qq", header[2], 1, 7)
                        conn.sendall(payload[:8] if count == 3 and i == 2 else payload)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=server)
    worker.start()
    peer = OwnerPeer(
        endpoint=f"tcp://127.0.0.1:{listener.getsockname()[1]}",
        epoch=42,
        pin_cap=1,
        cleanup=None,
        timeout_s=0.2,
    )
    keys = full_keys([hashlib.sha256(b"key").hexdigest()])
    try:
        assert peer.request(Operation.LOOKUP, keys).tolist() == [7]
        assert peer.request(Operation.LOOKUP, keys).tolist() == [7]
        assert peer.request(Operation.LOOKUP, keys) is None
        assert peer.request(Operation.LOOKUP, keys).tolist() == [7]
    finally:
        worker.join(5)
        listener.close()
    assert not errors and len(accepted) == 2 and not worker.is_alive()


@pytest.mark.parametrize("blocked_reply", [False, True])
def test_slow_tcp_peer_does_not_block_other_readers(blocked_reply):
    """A reader that stops mid-frame or before reading cannot consume others' deadline."""
    directory = L1Directory(16, 8)
    service = OwnerService(directory, endpoint="tcp://127.0.0.1:0")
    endpoint = service.endpoint
    key = hashlib.sha256(b"key").hexdigest()
    keys = full_keys([key])
    service.publish([key], [3])
    host = urlsplit(endpoint)
    slow = socket.socket()
    slow.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    slow.settimeout(2)
    slow.connect((host.hostname, host.port))
    try:
        n = 1 << 20 if blocked_reply else 0
        header = struct.pack(
            "=6Q", 0x4C314F574E455231, Operation.CONTAINS, service.epoch, 1, 1, n
        )
        if blocked_reply:
            slow.sendall(header + keys.tobytes() * n)
            # Peek waits until dispatch has produced a reply, without draining it.
            assert slow.recv(1, socket.MSG_PEEK)
        else:
            slow.sendall(header[:1])
            time.sleep(0.01)
        for url in (endpoint,):
            peer = OwnerPeer(
                endpoint=url,
                epoch=service.epoch,
                pin_cap=8,
                cleanup=None,
                timeout_s=0.05,
            )
            samples = []
            for _ in range(32):
                started = time.perf_counter_ns()
                assert peer.request(Operation.LOOKUP, keys).tolist() == [3]
                samples.append((time.perf_counter_ns() - started) / 1000)
            print(
                json.dumps(
                    {
                        "blocked_reply": blocked_reply,
                        "samples": len(samples),
                        "max_us": max(samples),
                        "median_us": sorted(samples)[len(samples) // 2],
                    }
                )
            )
    finally:
        slow.close()
        service.close()
        directory.close()


def test_unix_and_tcp_listeners_share_owner_authority():
    directory = L1Directory(16, 8)
    service = OwnerService(directory)
    endpoint = service.listen_tcp("tcp://127.0.0.1:0")
    key = hashlib.sha256(b"shared").hexdigest()
    keys = full_keys([key])
    peer = OwnerPeer(
        endpoint=endpoint, epoch=service.epoch, pin_cap=8, cleanup=None, timeout_s=0.05
    )
    unix = OwnerPeer(
        endpoint=service.endpoint,
        epoch=service.epoch,
        pin_cap=8,
        cleanup=None,
        timeout_s=0.05,
    )
    try:
        service.publish([key], [3])
        assert peer.request(Operation.ACQUIRE_EXACT, keys, seq=1).tolist() == [3]
        service.retire([key])
        assert unix.lookup(keys) == 0 and not directory.freeze()
        assert peer.request(Operation.RELEASE, full_keys([]), seq=1).tolist() == []
        assert directory.freeze()
    finally:
        service.close()
        directory.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
