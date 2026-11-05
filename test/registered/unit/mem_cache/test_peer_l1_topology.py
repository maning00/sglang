"""GPU identity normalization and visible-ordinal exchange, without a GPU."""

from types import SimpleNamespace
from uuid import UUID

import pytest

from sglang.srt.mem_cache.unified_cache.peer_l1 import topology
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("prefix", ["", "GPU-"])
def test_uuid_matches_runtime_bytes(monkeypatch, prefix):
    value = UUID("00112233-4455-6677-8899-aabbccddeeff")
    monkeypatch.setattr(topology.torch.cuda, "get_device_properties",
                        lambda ordinal: SimpleNamespace(uuid=prefix + str(value)))
    assert bytes.fromhex(topology._device_uuid(0)) == value.bytes


def test_uuid_object_and_zero_rejection(monkeypatch):
    values = [UUID(int=1), UUID(int=0)]
    monkeypatch.setattr(topology.torch.cuda, "get_device_properties",
                        lambda ordinal: SimpleNamespace(uuid=values[ordinal]))
    assert topology._device_uuid(0) == values[0].hex
    with pytest.raises(RuntimeError, match="empty peer GPU UUID"):
        topology._device_uuid(1)


def test_exchange_preserves_visible_ordinals_and_duplicate_guard(monkeypatch):
    assert topology.exchange_devices.__module__ == topology.__name__
    identities = [UUID(int=7), UUID(int=3)]
    monkeypatch.setattr(topology.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(topology.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(topology.torch.cuda, "get_device_properties",
                        lambda i: SimpleNamespace(uuid=identities[i]))
    monkeypatch.setattr(topology.torch.distributed, "get_world_size", lambda group: 2)
    duplicate = [False]

    def gather(peers, local, group):
        assert local.gpu == identities[1].hex
        assert local.visible == tuple(u.hex for u in identities)
        peers[:] = [local, topology.DeviceInfo(local.host,
                    local.gpu if duplicate[0] else identities[0].hex, local.visible)]

    monkeypatch.setattr(topology.torch.distributed, "all_gather_object", gather)
    assert len(topology.exchange_devices(None, SimpleNamespace(index=1))) == 2
    duplicate[0] = True
    with pytest.raises(ValueError, match="distinct physical GPUs"):
        topology.exchange_devices(None, SimpleNamespace(index=1))
