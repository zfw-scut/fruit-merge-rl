"""Small multi-source telemetry registry and read-only aggregator."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_REGISTRY = (
    PROJECT_ROOT / "runs" / "portal_processes" / "telemetry_sources.json"
)


def _normalise_source(source: Mapping[str, Any]) -> dict[str, str] | None:
    source_id = str(source.get("id", "")).strip()[:80]
    name = str(source.get("name", source_id)).strip()[:120]
    role = str(source.get("role", "训练遥测")).strip()[:120]
    url = str(source.get("url", "")).rstrip("/")
    parsed = urlparse(url)
    if (
        not source_id
        or not name
        or parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port is None
    ):
        return None
    return {"id": source_id, "name": name, "role": role, "url": url}


def load_telemetry_sources(
    path: str | Path = DEFAULT_SOURCE_REGISTRY,
) -> list[dict[str, str]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    raw_sources = payload.get("sources", []) if isinstance(payload, dict) else []
    if not isinstance(raw_sources, list):
        return []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in raw_sources:
        source = _normalise_source(raw) if isinstance(raw, dict) else None
        if source is None or source["id"] in seen:
            continue
        seen.add(source["id"])
        result.append(source)
    return result


def write_telemetry_sources(
    sources: Iterable[Mapping[str, Any]],
    path: str | Path = DEFAULT_SOURCE_REGISTRY,
) -> Path:
    target = Path(path)
    normalised = [source for raw in sources if (source := _normalise_source(raw))]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"format_version": 1, "updated_at": time.time(), "sources": normalised},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _fetch_source(source: Mapping[str, str], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        **source,
        "available": False,
        "payload": None,
        "error": None,
        "latency_ms": None,
    }
    try:
        with urlopen(source["url"] + "/api/status", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("遥测响应不是JSON对象")
        result["available"] = True
        result["payload"] = payload
    except (OSError, URLError, TimeoutError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["latency_ms"] = (time.perf_counter() - started) * 1000.0
    return result


def aggregate_telemetry_sources(
    sources: Iterable[Mapping[str, Any]],
    *,
    timeout: float = 1.5,
) -> dict[str, Any]:
    normalised = [source for raw in sources if (source := _normalise_source(raw))]
    if not normalised:
        return {"available": False, "payload": None, "sources": []}
    by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(normalised))) as executor:
        futures = {
            executor.submit(_fetch_source, source, timeout): source["id"]
            for source in normalised
        }
        for future in as_completed(futures):
            by_id[futures[future]] = future.result()
    results = [by_id[source["id"]] for source in normalised]
    selected = next(
        (
            item
            for item in results
            if item["available"]
            and isinstance(item.get("payload"), dict)
            and item["payload"].get("training")
        ),
        next((item for item in results if item["available"]), None),
    )
    return {
        "available": selected is not None,
        "payload": selected["payload"] if selected is not None else None,
        "selected_source_id": selected["id"] if selected is not None else None,
        "sources": results,
    }
