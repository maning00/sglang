"""Read-only instance owner discovery, independent of PD and its transport."""

from fastapi import APIRouter, Depends, Response

router = APIRouter()


def owner_snapshot():
    from sglang.srt.entrypoints.http_server import _global_state

    return _global_state.scheduler_info.get("peer_l1_owners")


@router.get("/peer_l1/owners")
async def owners(snapshot: bytes | None = Depends(owner_snapshot)):
    if snapshot is None:
        return Response(status_code=503)
    return Response(content=snapshot, media_type="application/json")
