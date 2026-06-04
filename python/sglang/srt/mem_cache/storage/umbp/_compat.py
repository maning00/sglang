from __future__ import annotations

from typing import Any, Optional


def get_umbp_client_or_none(cache_controller: Any) -> Optional[Any]:
    storage_backend = getattr(cache_controller, "storage_backend", None)
    if storage_backend is None:
        return None
    client = getattr(storage_backend, "client", None)
    if client is None or not hasattr(client, "report_external_kv_blocks"):
        return None
    return client
