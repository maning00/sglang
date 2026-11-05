"""IPC eligibility and backend-local slots; global ranks never index a GPU kernel."""

import socket
from uuid import UUID

import msgspec
import torch


class DeviceInfo(msgspec.Struct, frozen=True):
    host: str
    gpu: str
    visible: tuple[str, ...]


def _device_uuid(ordinal):
    # PyTorch exposes the runtime's visible-device identity. Its _CUuuid string
    # has a GPU- prefix; uuid.UUID also accepts the simulator's UUID object.
    value = UUID(str(torch.cuda.get_device_properties(ordinal).uuid).removeprefix("GPU-"))
    if not value.int:
        raise RuntimeError("PyTorch returned an empty peer GPU UUID")
    return value.hex


def exchange_devices(group, device):
    torch.cuda.current_device()
    identities = [_device_uuid(ordinal) for ordinal in range(torch.cuda.device_count())]
    local = DeviceInfo(
        socket.gethostname(), identities[device.index], tuple(identities)
    )
    peers = [None] * torch.distributed.get_world_size(group)
    torch.distributed.all_gather_object(peers, local, group=group)
    if len({p.gpu for p in peers}) != len(peers):
        raise ValueError("peer_l1 requires distinct physical GPUs within an instance")
    return peers


def ipc_sets(devices):
    return [
        tuple(
            j
            for j, other in enumerate(devices)
            if j != i
            and own.host == other.host
            and other.gpu in own.visible
            and own.gpu in other.visible
        )[:7]
        for i, own in enumerate(devices)
    ]
