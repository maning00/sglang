"""Debug-only checks for the interval between reservation and copy submission."""

import sys


class CollectiveGuard:
    # Inspect the real Python function body, so aliases imported before this
    # guard was enabled are checked too. Metadata queries are not collectives.
    _collectives = frozenset(
        "all_reduce all_gather all_gather_into_tensor all_gather_single "
        "all_gather_object all_to_all all_to_all_single broadcast "
        "broadcast_object_list reduce reduce_scatter reduce_scatter_tensor "
        "reduce_scatter_single gather gather_object scatter scatter_object_list "
        "barrier monitored_barrier".split()
    )

    def __init__(self):
        self.enabled = False

    def enable(self):
        if self.enabled:
            return
        if sys.getprofile() is not None:
            raise RuntimeError(
                "peer_l1 check_hooks needs an unused Python profile hook"
            )
        self.enabled = True
        sys.setprofile(self._check)

    def disable(self):
        if self.enabled:
            self.enabled = False
            sys.setprofile(None)

    def _check(self, frame, event, arg):
        if (
            event == "call"
            and frame.f_code.co_name in self._collectives
            and frame.f_globals.get("__name__", "").startswith("torch.distributed.")
        ):
            self.disable()
            raise AssertionError(
                f"peer_l1: {frame.f_code.co_name} called with an unsubmitted reservation"
            )
