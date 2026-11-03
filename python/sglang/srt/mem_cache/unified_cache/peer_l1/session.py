"""Reader-side authorization and completion contracts, independent of transport."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

import msgspec
import numpy as np


class Grant(msgspec.Struct, frozen=True, kw_only=True):
    owner: str
    epoch: int | str
    grant_id: int
    rows: np.ndarray
    acquired_before: float
    lease_s: float | None


class OwnerChannel(Protocol):
    def acquire_exact(self, keys) -> Grant | None: ...
    def cancel_unsubmitted(self, seq: int) -> None: ...
    def release(self, grant: Grant) -> None: ...


class Transfer(Protocol):
    """Layer resolution permits forward; failure is judged after forward and drain."""

    def resolved_layers(self) -> Sequence[int]: ...
    def failed_requests(self) -> set[str]: ...
    def quiesced(self) -> bool: ...


class PeerSession(msgspec.Struct, frozen=True, kw_only=True):
    endpoint: str
    owner: OwnerChannel
    mover: Any
    mapping: Any
    cost: Literal["local", "fabric", "rdma"]
