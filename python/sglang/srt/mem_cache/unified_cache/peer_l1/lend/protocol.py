"""Small, batched control messages; no GPU work runs in the service thread."""

from __future__ import annotations

import logging
import threading
import time
import uuid

import msgspec
import zmq

logger = logging.getLogger(__name__)


class Message(msgspec.Struct, array_like=True):
    seq: int
    op: str
    epoch: str = ""
    data: list = msgspec.field(default_factory=list)


encode = msgspec.msgpack.encode
decoder = msgspec.msgpack.Decoder(Message)


class LendService:
    def __init__(
        self, pool, address, *, layout=None, worker_id="", compatibility_id=""
    ):
        self.pool = pool
        self.address = address
        self.layout = layout
        self.worker_id = worker_id
        self.compatibility_id = compatibility_id
        self.port = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="decode-lend", daemon=True
        )
        self._thread.start()
        self._ready.wait()
        if self._error:
            raise self._error

    def _run(self):
        sock = zmq.Context.instance().socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        try:
            host = self.address
            if ":" in host:
                sock.setsockopt(zmq.IPV6, 1)
                host = f"[{host}]" if not host.startswith("[") else host
            self.port = sock.bind_to_random_port(f"tcp://{host}")
        except zmq.ZMQError as exc:
            self._error = exc
            self._ready.set()
            sock.close()
            return
        self._ready.set()
        peers = set()
        try:
            while not self._stop.is_set():
                if sock.poll(20):
                    identity, payload = sock.recv_multipart()
                    try:
                        request = decoder.decode(payload)
                        writer = identity.decode("ascii")
                        data = []
                        if request.op == "HELLO":
                            peers.add(identity)
                            data = [
                                len(self.pool.free),
                                self.pool.lease_s,
                                self.layout,
                                self.worker_id,
                                self.compatibility_id,
                            ]
                        elif identity in peers and request.epoch == self.pool.epoch:
                            op, args = request.op, request.data
                            if op == "ALLOC":
                                data = self.pool.allocate(writer, args[0], request.seq)
                            elif op == "CANCEL_ALLOC":
                                self.pool.cancel_allocation(writer, args[0])
                            elif op == "COMMIT":
                                data = self.pool.commit(writer, args)
                            elif op == "ABORT":
                                self.pool.abort(writer, args)
                            elif op == "ACQUIRE":
                                data = self.pool.acquire(writer, request.seq, args)
                            elif op == "CANCEL_ACQUIRE":
                                self.pool.cancel_acquire(writer, args[0])
                            elif op == "UNPIN":
                                self.pool.unpin(writer, args[0])
                            elif op == "DEMOTED":
                                self.pool.demoted(writer, args)
                            else:
                                raise ValueError(f"Unknown lend request {op}")
                        sock.send_multipart(
                            [
                                identity,
                                encode(
                                    Message(
                                        request.seq, request.op, self.pool.epoch, data
                                    )
                                ),
                            ]
                        )
                    except Exception:
                        logger.exception("decode lend rejected a control message")
                self.pool.expire()
                with self.pool.lock:
                    events = list(self.pool.events)
                    self.pool.events.clear()
                for op, writer, records in events:
                    sock.send_multipart(
                        [
                            writer.encode("ascii"),
                            encode(Message(0, op, self.pool.epoch, records)),
                        ]
                    )
        finally:
            sock.close()

    def close(self):
        self._stop.set()
        self._thread.join()
        if self.pool.publisher:
            self.pool.publisher.close()


class LendPeer:
    """Owned and used exclusively by the prefill I/O thread."""

    def __init__(
        self, info, on_event, *, timeout_s=2.0, layout=None, compatibility_id=""
    ):
        self.info, self.on_event, self.timeout_s = info, on_event, timeout_s
        self.socket = zmq.Context.instance().socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, uuid.uuid4().hex.encode("ascii"))
        self.socket.setsockopt(zmq.LINGER, 0)
        host = info.endpoint
        if ":" in host:
            self.socket.setsockopt(zmq.IPV6, 1)
            host = f"[{host}]" if not host.startswith("[") else host
        self.socket.connect(f"tcp://{host}:{info.lend_port}")
        self.seq, self.epoch, self.lease_s = 0, "", 30.0
        self.worker_id = ""
        self.session = None
        self._cleanup = {}
        self._cleanup_seqs = {}
        reply = self.call("HELLO")
        if reply is not None:
            self.epoch = reply.epoch
            self.lease_s = reply.data[1]
            self.worker_id = reply.data[3]
            if compatibility_id and (
                len(reply.data) < 5 or reply.data[4] != compatibility_id
            ):
                self.close()
                raise ValueError("Decode lend peers have incompatible KV identities")
            if layout is not None and reply.data[2] != layout:
                self.close()
                raise ValueError("Decode lend peers have different KV region layouts")

    def poll(self, timeout=0):
        if not self.socket.poll(timeout):
            return None
        reply = decoder.decode(self.socket.recv())
        cleanup = self._cleanup_seqs.pop(reply.seq, None)
        if cleanup is not None:
            self._cleanup.pop(cleanup, None)
            return None
        if reply.seq == 0:
            if reply.epoch == self.epoch:
                self.on_event(self, reply)
            return None
        return reply

    def call(self, op, data=None):
        self.seq += 1
        seq = self.seq
        self.socket.send(encode(Message(seq, op, self.epoch, data or [])))
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            reply = self.poll(
                min(20, max(0, int((deadline - time.monotonic()) * 1000)))
            )
            if reply is not None and reply.seq == seq:
                if op != "HELLO" and reply.epoch != self.epoch:
                    return None
                return reply
        if op == "ACQUIRE":
            self.cleanup("CANCEL_ACQUIRE", seq)
        if op == "ALLOC":
            # Same DEALER connection preserves ordering after ALLOC. No RDMA was
            # posted for an unseen grant. The cancellation is safe even if its
            # late ALLOC reply is discarded while waiting for this ACK.
            self.call("CANCEL_ALLOC", [seq])
        return None

    def cleanup(self, op, identifier):
        key = (op, identifier, self.epoch)
        if key not in self._cleanup:
            self.seq += 1
            self._cleanup[key] = (self.seq, 0.0, 0.01)
            self._cleanup_seqs[self.seq] = key
        self.flush_cleanup()

    def flush_cleanup(self):
        now = time.monotonic()
        for key, (seq, deadline, delay) in list(self._cleanup.items()):
            if now < deadline:
                continue
            op, identifier, epoch = key
            try:
                self.socket.send(
                    encode(Message(seq, op, epoch, [identifier])), zmq.DONTWAIT
                )
            except zmq.ZMQError:
                pass
            self._cleanup[key] = (seq, now + delay, min(1.0, delay * 2))

    def close(self):
        self.socket.close()
