from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MoriKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)


class MoriKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        pass

    def poll(self) -> KVPoll:
        pass

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class MoriKVReceiver(CommonKVReceiver):

    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)


class MoriKVBootstrapServer(CommonKVBootstrapServer):
    pass
