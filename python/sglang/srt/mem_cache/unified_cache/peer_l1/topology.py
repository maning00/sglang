"""IPC eligibility and backend-local slots; global ranks never index a GPU kernel."""

import ctypes
import socket

import msgspec
import torch


class DeviceInfo(msgspec.Struct, frozen=True):
    host: str
    gpu: str
    visible: tuple[str, ...]


def exchange_devices(group, device):
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    hip.hipDeviceGetUuid.restype = ctypes.c_int
    identities = []
    for ordinal in range(torch.cuda.device_count()):
        uuid = ctypes.create_string_buffer(16)
        if hip.hipDeviceGetUuid(uuid, ordinal):
            raise RuntimeError("Unable to identify peer GPU")
        identities.append(uuid.raw.hex())
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
