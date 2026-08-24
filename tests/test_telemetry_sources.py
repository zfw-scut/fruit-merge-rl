from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from daxigua.portal.telemetry_sources import (
    aggregate_telemetry_sources,
    load_telemetry_sources,
    write_telemetry_sources,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class TelemetrySourceTests(unittest.TestCase):
    def test_registry_round_trip_keeps_only_loopback_sources(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "sources.json"
            write_telemetry_sources(
                [
                    {
                        "id": "server-7",
                        "name": "7号实例",
                        "role": "风险模型",
                        "url": "http://127.0.0.1:18765",
                    },
                    {
                        "id": "unsafe",
                        "name": "外部",
                        "url": "https://example.com/status",
                    },
                ],
                path,
            )

            result = load_telemetry_sources(path)

            self.assertEqual([item["id"] for item in result], ["server-7"])

    def test_aggregator_preserves_source_order_and_selects_training(self):
        sources = [
            {
                "id": "task",
                "name": "任务服务器",
                "role": "采集",
                "url": "http://127.0.0.1:18765",
            },
            {
                "id": "train",
                "name": "训练服务器",
                "role": "基线",
                "url": "http://127.0.0.1:18766",
            },
        ]

        def response(url, timeout):
            if "18765" in url:
                return _Response({"tasks": {"available": True}})
            return _Response({"training": {"progress": 123}})

        with patch("daxigua.portal.telemetry_sources.urlopen", side_effect=response):
            result = aggregate_telemetry_sources(sources)

        self.assertTrue(result["available"])
        self.assertEqual(result["selected_source_id"], "train")
        self.assertEqual(
            [item["id"] for item in result["sources"]], ["task", "train"]
        )


if __name__ == "__main__":
    unittest.main()
