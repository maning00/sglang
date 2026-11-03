"""MoRI submission ownership. Failed terminal statuses do not prove quiescence."""

import msgspec


class Submission(msgspec.Struct):
    engine: object
    local: list
    remote: list
    owner: object = None
    statuses: list = msgspec.field(default_factory=list)
    arguments: list = msgspec.field(default_factory=list)
    submitted: bool = False


class UnsafeTransfer(RuntimeError):
    def __init__(self, submission):
        super().__init__(
            "RDMA failed after possible submission; targets require quarantine"
        )
        self.submission = submission


def new_status():
    from mori import cpp

    return cpp.TransferStatus()


def submit_batches(submission, *, local_offsets, remote_offsets, sizes, write):
    operation = (
        submission.engine._engine.BatchWrite
        if write
        else submission.engine._engine.BatchRead
    )
    for start in range(0, len(submission.local), 16):
        end = start + 16
        statuses = [new_status() for _ in submission.local[start:end]]
        ids = [submission.engine.allocate_transfer_uid() for _ in statuses]
        args = (
            submission.local[start:end],
            local_offsets[start:end],
            submission.remote[start:end],
            remote_offsets[start:end],
            sizes[start:end],
            statuses,
            ids,
        )
        # Keep statuses even if the native batch throws after a partial post.
        submission.statuses.extend(statuses)
        submission.arguments.append(args)
        submission.submitted = True
        operation(*args)


def execute(submission, *, local_offsets, remote_offsets, sizes, write, wait):
    try:
        submit_batches(
            submission,
            local_offsets=local_offsets,
            remote_offsets=remote_offsets,
            sizes=sizes,
            write=write,
        )
        if not wait(submission.statuses) or not all(
            s.Succeeded() for s in submission.statuses
        ):
            raise RuntimeError("RDMA status failed or did not complete")
        return True
    except BaseException as exc:
        if not submission.submitted:
            raise
        from sglang.srt.runtime_context import get_resources

        get_resources().rdma_quarantines.append(submission)
        raise UnsafeTransfer(submission) from exc
