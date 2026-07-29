"""Stable Chaquopy facade for health checks and mobile graph construction."""

from daxigua_mobile import build_mobile_graph_json as _build_mobile_graph_json


def healthcheck() -> str:
    """Return a stable value used by the Android startup diagnostic."""

    return "ready"


def build_mobile_graph_json(scene_json: str) -> str:
    """Convert an Android scene snapshot to the final model's JSON tensors."""

    return _build_mobile_graph_json(scene_json)
