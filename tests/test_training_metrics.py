"""训练指标和 episode 日志测试。"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from daxigua_rl.models import GNNQNetwork
from daxigua_rl.scripts.train_dqn import EpisodeLogger, evaluate_policy
from daxigua_rl.training.collector import RolloutStats


class TrainingMetricsTest(unittest.TestCase):
    """验证训练脚本新增的评估和 episode 指标。"""

    def test_episode_logger_writes_one_row_per_finished_episode(self):
        """EpisodeLogger 应把每个已结束 episode 单独写入 CSV。"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'episode_metrics.csv'
            logger = EpisodeLogger(path)
            try:
                stats = RolloutStats(
                    steps=10,
                    episodes=2,
                    total_reward=12.0,
                    episode_rewards=(5.0, 7.0),
                    episode_lengths=(3, 4),
                    episode_scores=(30, 50),
                    episode_end_offsets=(3, 8),
                    episode_terminated_flags=(True, False),
                    episode_truncated_flags=(False, True),
                    buffer_size=10,
                )
                written = logger.log_collect_stats(
                    stats,
                    phase='train',
                    update_step=12,
                    start_env_steps=100,
                    epsilon=0.25,
                )
            finally:
                logger.close()

            self.assertEqual(written, 2)
            with path.open(newline='', encoding='utf-8') as file_obj:
                rows = list(csv.DictReader(file_obj))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['episode_index'], '1')
        self.assertEqual(rows[0]['env_steps'], '103')
        self.assertEqual(rows[0]['score'], '30.0')
        self.assertEqual(rows[1]['episode_index'], '2')
        self.assertEqual(rows[1]['env_steps'], '108')
        self.assertEqual(rows[1]['truncated'], '1')

    def test_evaluate_policy_returns_score_extremes(self):
        """evaluate_policy 应返回本次评估最高分和最低分。"""

        args = SimpleNamespace(
            action_count=5,
            max_physics_frames=120,
            stable_frames=4,
            score_scale=1.0,
            survival_bonus=0.05,
            height_delta_weight=0.02,
            danger_height_weight=1.0,
            terminal_penalty=-100.0,
            eval_episodes=2,
            eval_max_steps=3,
            seed=0,
        )
        model = GNNQNetwork(hidden_dim=32, message_layers=2)
        stats = evaluate_policy(model, args, device='cpu')

        self.assertEqual(stats['eval_episodes'], 2)
        self.assertIn('eval_score_max', stats)
        self.assertIn('eval_score_min', stats)
        self.assertGreaterEqual(stats['eval_score_max'], stats['eval_score_min'])


if __name__ == '__main__':
    unittest.main()
