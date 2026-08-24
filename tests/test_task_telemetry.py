from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from daxigua.rl.task_telemetry import (
    TaskTelemetryPublisher,
    make_task_id,
    scan_task_telemetry,
)


class TaskTelemetryTests(unittest.TestCase):
    def test_publisher_writes_bounded_generic_snapshot(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            publisher = TaskTelemetryPublisher(
                task_id="pair-risk-train",
                task_type="supervised_training",
                name="堵塞风险预测器",
                output_dir=root / "external-output",
                telemetry_root=root / "runs" / "task_telemetry",
                history_limit=2,
                metric_schema=[{"key": "loss", "label": "验证损失"}],
            )
            for epoch in range(3):
                publisher.update(
                    phase="训练",
                    current=epoch + 1,
                    total=3,
                    unit="epoch",
                    metrics={"loss": 0.3 - epoch * 0.1, "invalid": float("nan")},
                    history_step=epoch + 1,
                )
            publisher.complete(
                phase="完成",
                current=3,
                total=3,
                unit="epoch",
                metrics={"loss": 0.1},
                record_history=False,
            )

            result = scan_task_telemetry(root)

            self.assertTrue(result["available"])
            task = result["current"]
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["progress"]["fraction"], 1.0)
            self.assertEqual(len(task["history"]), 2)
            self.assertNotIn("invalid", task["history"][-1]["metrics"])
            self.assertFalse(task["stale"])

    def test_scanner_marks_stale_and_ignores_invalid_files(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            telemetry = root / "runs" / "task_telemetry"
            telemetry.mkdir(parents=True)
            (telemetry / "valid.json").write_text(
                json.dumps(
                    {
                        "format_version": 1,
                        "id": "valid",
                        "task_type": "collection",
                        "status": "running",
                        "updated_at": 10.0,
                        "stale_after_seconds": 5.0,
                    }
                ),
                encoding="utf-8",
            )
            (telemetry / "broken.json").write_text("not json", encoding="utf-8")

            result = scan_task_telemetry(root, now=20.0)

            self.assertEqual(len(result["tasks"]), 1)
            self.assertTrue(result["tasks"][0]["stale"])

    def test_task_id_hides_full_output_path(self):
        task_id = make_task_id("pair risk", "C:/secret/path/output")
        self.assertTrue(task_id.startswith("pair-risk-"))
        self.assertNotIn("secret", task_id)


if __name__ == "__main__":
    unittest.main()
