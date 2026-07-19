"""训练指标和 episode 日志测试。"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from daxigua_rl.models import GNNQNetwork
from daxigua_rl.scripts.train_dqn import (
    EpisodeLogger,
    build_metric_row,
    evaluate_policy,
    load_config_defaults,
    parse_args,
)
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

    def test_metric_row_includes_reward_breakdown_means(self):
        """metrics.csv 行应包含 reward breakdown 的窗口均值。"""

        collect_stats = RolloutStats(
            steps=4,
            episodes=0,
            total_reward=10.0,
            reward_breakdown_totals=(
                ('total', 10.0),
                ('score_reward', 8.0),
                ('survival_bonus', 0.2),
                ('height_delta_reward', -0.1),
                ('danger_penalty', -1.0),
                ('terminal_penalty', 0.0),
                ('previous_height_ratio', 1.2),
                ('next_height_ratio', 1.4),
                ('height_delta_ratio', 0.2),
            ),
            buffer_size=32,
        )
        train_stats = SimpleNamespace(
            loss=1.0,
            mean_q=2.0,
            mean_target=3.0,
            mean_reward=4.0,
            mean_abs_td_error=5.0,
            bootstrap_count=6,
            grad_norm=7.0,
            target_synced=False,
        )

        row = build_metric_row(
            update_step=10,
            env_steps=20,
            epsilon=0.5,
            train_stats=train_stats,
            collect_stats=collect_stats,
            eval_stats=None,
            best_eval_score=float('-inf'),
            best_eval_update=0,
            timing={'elapsed': 2.0},
        )

        self.assertEqual(row['collect_mean_reward_total'], 2.5)
        self.assertEqual(row['collect_mean_score_reward'], 2.0)
        self.assertEqual(row['collect_mean_danger_penalty'], -0.25)
        self.assertAlmostEqual(row['collect_mean_next_height_ratio'], 0.35)

    def test_toml_config_loads_defaults_and_cli_can_override(self):
        """TOML 配置应能提供默认参数，命令行显式参数应优先。"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / 'train.toml'
            config_path.write_text(
                '\n'.join((
                    '[runtime]',
                    'run_dir = "runs/from_config"',
                    'device = "cuda"',
                    '',
                    '[training]',
                    'total_updates = 100',
                    'batch_size = 64',
                    '',
                    '[parallel]',
                    'num_envs = 4',
                    'async_rollout = true',
                )),
                encoding='utf-8',
            )

            args = parse_args((
                '--config',
                str(config_path),
                '--total-updates',
                '20',
                '--no-async-rollout',
            ))

        self.assertEqual(args.config, str(config_path))
        self.assertEqual(args.run_dir, 'runs/from_config')
        self.assertEqual(args.device, 'cuda')
        self.assertEqual(args.total_updates, 20)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.num_envs, 4)
        self.assertFalse(args.async_rollout)

    def test_toml_config_rejects_unknown_keys(self):
        """TOML 配置里写错字段名时应直接报错。"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / 'bad.toml'
            config_path.write_text(
                '[training]\nunknown_option = 1\n',
                encoding='utf-8',
            )

            with self.assertRaises(ValueError):
                load_config_defaults(config_path)


if __name__ == '__main__':
    unittest.main()
