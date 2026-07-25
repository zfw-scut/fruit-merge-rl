"""训练实验目录导出器测试。"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.export_training_catalog import export_catalog


class TrainingCatalogTest(unittest.TestCase):
    """验证训练摘要的统计结果和迁移边界。"""

    def test_export_catalog_summarizes_run_without_copying_large_artifacts(self):
        """导出器应统计原始产物，但不把 checkpoint 或 replay 复制进文档。"""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs_dir = root / 'runs'
            output_dir = root / 'docs' / 'training_runs'
            run_dir = runs_dir / 'sample_run'
            checkpoint_dir = run_dir / 'checkpoints'
            replay_dir = run_dir / 'replay_cold'
            checkpoint_dir.mkdir(parents=True)
            replay_dir.mkdir()

            config = {
                'created_at': '2026-07-26T00:00:00',
                'args': {
                    'total_updates': 2,
                    'device': 'cpu',
                    'learning_rate': 0.0001,
                },
            }
            (run_dir / 'config.json').write_text(
                json.dumps(config),
                encoding='utf-8',
            )

            metric_fields = (
                'update_step',
                'env_steps',
                'collect_steps',
                'collect_mean_reward_total',
                'collect_mean_score_reward',
                'updates_per_second',
                'env_steps_per_second',
            )
            with (run_dir / 'metrics.csv').open(
                    'w',
                    encoding='utf-8',
                    newline='') as file:
                writer = csv.DictWriter(file, fieldnames=metric_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        'update_step': 1,
                        'env_steps': 4,
                        'collect_steps': 2,
                        'collect_mean_reward_total': 1,
                        'collect_mean_score_reward': 2,
                        'updates_per_second': 3,
                        'env_steps_per_second': 12,
                    }
                )
                writer.writerow(
                    {
                        'update_step': 2,
                        'env_steps': 8,
                        'collect_steps': 6,
                        'collect_mean_reward_total': 3,
                        'collect_mean_score_reward': 4,
                        'updates_per_second': 4,
                        'env_steps_per_second': 16,
                    }
                )

            episode_fields = (
                'episode_index',
                'score',
                'episode_reward',
                'episode_length',
                'terminated',
                'truncated',
            )
            with (run_dir / 'episode_metrics.csv').open(
                    'w',
                    encoding='utf-8',
                    newline='') as file:
                writer = csv.DictWriter(file, fieldnames=episode_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        'episode_index': 1,
                        'score': 100,
                        'episode_reward': 80,
                        'episode_length': 10,
                        'terminated': 1,
                        'truncated': 0,
                    }
                )
                writer.writerow(
                    {
                        'episode_index': 2,
                        'score': 300,
                        'episode_reward': 240,
                        'episode_length': 20,
                        'terminated': 0,
                        'truncated': 1,
                    }
                )

            (checkpoint_dir / 'best.pt').write_bytes(b'checkpoint-data')
            (replay_dir / 'segment_00000000.pt').write_bytes(b'replay-data')

            summaries = export_catalog(runs_dir, output_dir)

            self.assertEqual(len(summaries), 1)
            summary = summaries[0]
            self.assertEqual(summary.status, '已完成')
            self.assertEqual(summary.episode_stats['score_mean'], 200)
            self.assertEqual(summary.episode_stats['score_median'], 200)
            self.assertEqual(summary.episode_stats['terminated_count'], 1)
            self.assertEqual(summary.episode_stats['truncated_count'], 1)

            # reward 均值按 collect_steps 加权：(1*2 + 3*6) / 8 = 2.5。
            self.assertEqual(summary.reward_stats['total'], 2.5)
            self.assertEqual(summary.reward_stats['score_reward'], 3.5)

            generated_run = output_dir / 'runs' / 'sample_run'
            self.assertTrue((generated_run / 'summary.md').is_file())
            self.assertTrue((generated_run / 'metrics_summary.json').is_file())
            self.assertFalse(any(generated_run.rglob('*.pt')))

            summary_text = (generated_run / 'summary.md').read_text(encoding='utf-8')
            self.assertIn('1.000e-04', summary_text)
            self.assertIn('| `score_mean` | 200 |', summary_text)

            artifact_text = (generated_run / 'artifacts.md').read_text(encoding='utf-8')
            self.assertIn('checkpoint', artifact_text)
            self.assertIn('replay', artifact_text)


if __name__ == '__main__':
    unittest.main()
