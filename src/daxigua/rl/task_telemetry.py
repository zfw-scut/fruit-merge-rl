"""Generic, low-overhead task telemetry for the local project portal.

Producers mirror a small bounded JSON snapshot into ``runs/task_telemetry``.
The dashboard scans only that directory, so arbitrary output trees never become
part of the request hot path and a new task type does not require portal code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from hashlib import sha1
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TELEMETRY_ROOT = PROJECT_ROOT / "runs" / "task_telemetry"
_VALID_STATUSES = {"pending", "running", "completed", "failed", "cancelled"}
_SAFE_ID = re.compile(r"[^a-zA-Z0-9._-]+")


def make_task_id(prefix: str, output_dir: str | Path | None = None) -> str:
    """Return a readable stable id without exposing the full output path."""

    clean_prefix = _SAFE_ID.sub("-", prefix.strip()).strip("-._") or "task"
    if output_dir is None:
        return clean_prefix
    digest = sha1(str(Path(output_dir)).encode("utf-8")).hexdigest()[:10]
    return f"{clean_prefix}-{digest}"


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _safe_mapping(values: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in (values or {}).items():
        clean_key = str(key)[:80]
        number = _finite_number(value)
        if number is not None:
            result[clean_key] = number
        elif isinstance(value, (str, bool)) or value is None:
            result[clean_key] = value
    return result


def _normalise_descriptors(items: list[Mapping[str, Any]] | None) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for item in items or []:
        key = str(item.get("key", "")).strip()[:80]
        if not key:
            continue
        result.append(
            {
                "key": key,
                "label": str(item.get("label", key))[:120],
                "format": str(item.get("format", "number"))[:32],
                "unit": str(item.get("unit", ""))[:32],
            }
        )
    return result


@dataclass
class TaskTelemetryPublisher:
    """Atomically publish one bounded task snapshot for portal consumption."""

    task_id: str
    task_type: str
    name: str
    output_dir: str | Path | None = None
    telemetry_root: str | Path = DEFAULT_TELEMETRY_ROOT
    stale_after_seconds: float = 120.0
    history_limit: int = 240
    identity: Mapping[str, Any] | None = None
    metric_schema: list[Mapping[str, Any]] | None = None
    series_schema: list[Mapping[str, Any]] | None = None
    _created_at: float = field(init=False, default_factory=time.time)
    _history: deque[dict[str, Any]] = field(init=False)
    _last: dict[str, Any] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.task_id = _SAFE_ID.sub("-", self.task_id.strip()).strip("-._")
        if not self.task_id:
            raise ValueError("task_id must contain at least one safe character")
        if not self.task_type.strip() or not self.name.strip():
            raise ValueError("task_type and name must not be empty")
        self.telemetry_root = Path(self.telemetry_root)
        self._history = deque(maxlen=max(1, int(self.history_limit)))

    @property
    def path(self) -> Path:
        return Path(self.telemetry_root) / f"{self.task_id}.json"

    def update(
        self,
        *,
        status: str = "running",
        phase: str = "",
        message: str = "",
        current: float | int | None = None,
        total: float | int | None = None,
        unit: str = "",
        metrics: Mapping[str, Any] | None = None,
        history_step: float | int | None = None,
        record_history: bool = True,
    ) -> dict[str, Any]:
        if status not in _VALID_STATUSES:
            raise ValueError(f"unsupported task status: {status}")
        now = time.time()
        clean_metrics = _safe_mapping(metrics)
        clean_current = _finite_number(current)
        clean_total = _finite_number(total)
        fraction = None
        if clean_current is not None and clean_total is not None and float(clean_total) > 0:
            fraction = max(0.0, min(1.0, float(clean_current) / float(clean_total)))

        if record_history and clean_metrics:
            row: dict[str, Any] = {"timestamp": now, "metrics": clean_metrics}
            clean_step = _finite_number(history_step)
            if clean_step is not None:
                row["step"] = clean_step
            self._history.append(row)

        payload: dict[str, Any] = {
            "format_version": 1,
            "id": self.task_id,
            "task_type": self.task_type[:80],
            "name": self.name[:160],
            "status": status,
            "phase": phase[:120],
            "message": message[:500],
            "progress": {
                "current": clean_current,
                "total": clean_total,
                "unit": unit[:40],
                "fraction": fraction,
            },
            "metrics": clean_metrics,
            "metric_schema": _normalise_descriptors(self.metric_schema),
            "series_schema": _normalise_descriptors(self.series_schema),
            "history": list(self._history),
            "identity": _safe_mapping(self.identity),
            "output_dir": str(self.output_dir) if self.output_dir is not None else None,
            "created_at": self._created_at,
            "updated_at": now,
            "stale_after_seconds": max(1.0, float(self.stale_after_seconds)),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f".json.{os.getpid()}.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temp_path, self.path)
        self._last = payload
        return payload

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        return self.update(status="completed", **kwargs)

    def fail(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self.update(status="failed", message=message, **kwargs)


def scan_task_telemetry(
    project_root: str | Path = PROJECT_ROOT,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Read validated task snapshots without traversing task output folders."""

    root = Path(project_root) / "runs" / "task_telemetry"
    timestamp = time.time() if now is None else float(now)
    tasks: list[dict[str, Any]] = []
    if root.is_dir():
        for path in root.glob("*.json"):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("format_version") != 1:
                continue
            if not isinstance(payload.get("id"), str) or not isinstance(payload.get("task_type"), str):
                continue
            updated_at = _finite_number(payload.get("updated_at"))
            stale_after = _finite_number(payload.get("stale_after_seconds"))
            payload["stale"] = bool(
                payload.get("status") == "running"
                and updated_at is not None
                and timestamp - float(updated_at) > float(stale_after or 120.0)
            )
            tasks.append(payload)

    tasks.sort(key=lambda item: float(item.get("updated_at") or 0.0), reverse=True)
    current = next(
        (item for item in tasks if item.get("status") in {"running", "pending"}),
        tasks[0] if tasks else None,
    )
    return {"available": bool(tasks), "current": current, "tasks": tasks}
