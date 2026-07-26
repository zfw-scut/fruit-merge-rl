"""训练指标和 episode 日志测试。"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

from daxigua_rl.models import GNNQNetwork
from daxigua_rl.scripts.train_dqn import (
    EpisodeLogger,
    build_env_config,
    build_metric_row,
    close_training_resources,
    evaluate_policy,
    load_config_defaults,
    parse_args,
    validate_args,
    write_attribution_warmup_summary,
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
            physics_fps=30,
            max_physics_frames=120,
            stable_frames=4,
            space_iterations=8,
            gamma=0.99,
            lambda_phi=0.5,
            capacity_weight=0.6,
            recoverability_weight=0.3,
            chain_readiness_weight=0.1,
            terminal_penalty=0.0,
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
                ('task_reward', 8.0),
                ('potential_shaping_reward', 1.6),
                ('terminal_penalty', 0.4),
                ('previous_potential', 1.2),
                ('next_potential', 1.4),
                ('potential_delta', 0.2),
                ('previous_top_connected_capacity', 2.0),
                ('next_top_connected_capacity', 2.4),
                ('previous_recoverability', 2.8),
                ('next_recoverability', 2.4),
                ('previous_chain_readiness', 0.4),
                ('next_chain_readiness', 0.8),
                ('merge_event_count', 4.0),
            ),
            potential_shaping_abs_values=(0.1, 0.2, 0.3, 0.4),
            state_analysis_calls=6,
            state_analysis_seconds=0.06,
            state_analysis_cache_hits=2,
            state_analysis_degraded_count=1,
            attribution_tracker_calls=4,
            attribution_tracker_seconds=0.02,
            attribution_events_created=5,
            attribution_events_confirmed=2,
            attribution_events_cancelled=1,
            attribution_events_interrupted=1,
            attribution_pending_event_count=3,
            attribution_lineage_merge_count=6,
            attribution_chain_merge_count=2,
            attribution_max_chain_depth=3,
            attribution_event_status_counts=(
                ('BORN_BURIED', 'cancelled', 1),
                ('MERGE_LINEAGE', 'confirmed', 2),
                ('REACHABILITY_SEALED', 'pending', 2),
            ),
            attribution_delays=(1, 2, 4, 8),
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
        self.assertEqual(row['collect_mean_task_reward'], 2.0)
        self.assertEqual(row['collect_mean_potential_shaping_reward'], 0.4)
        self.assertAlmostEqual(row['collect_mean_next_potential'], 0.35)
        self.assertAlmostEqual(
            row['collect_p95_abs_potential_shaping_reward'],
            0.385,
        )
        self.assertEqual(row['collect_state_analysis_calls'], 6)
        self.assertEqual(row['collect_state_analysis_cache_hits'], 2)
        self.assertEqual(row['collect_state_analysis_degraded_count'], 1)
        self.assertAlmostEqual(
            row['collect_state_analysis_cache_hit_rate'],
            0.5,
        )
        self.assertAlmostEqual(
            row['collect_state_analysis_degraded_rate'],
            1 / 6,
        )
        self.assertAlmostEqual(
            row['collect_mean_state_analysis_seconds'],
            0.01,
        )
        self.assertEqual(row['collect_attribution_tracker_calls'], 4)
        self.assertAlmostEqual(
            row['collect_mean_attribution_tracker_seconds'],
            0.005,
        )
        self.assertEqual(row['collect_attribution_events_created'], 5)
        self.assertEqual(row['collect_attribution_events_confirmed'], 2)
        self.assertEqual(row['collect_attribution_events_cancelled'], 1)
        self.assertEqual(row['collect_attribution_events_interrupted'], 1)
        self.assertEqual(row['collect_attribution_pending_event_count'], 3)
        self.assertEqual(row['collect_attribution_lineage_merge_count'], 6)
        self.assertEqual(row['collect_attribution_chain_merge_count'], 2)
        self.assertEqual(row['collect_attribution_max_chain_depth'], 3)
        self.assertAlmostEqual(
            row['collect_mean_attribution_delay'],
            3.75,
        )
        self.assertAlmostEqual(
            row['collect_p95_attribution_delay'],
            7.4,
        )
        self.assertEqual(
            row['collect_attribution_event_status_counts'],
            '{"BORN_BURIED":{"cancelled":1},'
            '"MERGE_LINEAGE":{"confirmed":2},'
            '"REACHABILITY_SEALED":{"pending":2}}',
        )

    def test_warmup_attribution_summary_is_kept_separate(self):
        stats = RolloutStats(
            steps=8,
            episodes=0,
            total_reward=0.0,
            attribution_tracker_calls=8,
            attribution_tracker_seconds=0.04,
            attribution_events_created=3,
            attribution_events_confirmed=1,
            attribution_events_cancelled=1,
            attribution_pending_event_count=1,
            attribution_event_status_counts=(
                ('MERGE_LINEAGE', 'confirmed', 1),
                ('REACHABILITY_SEALED', 'pending', 1),
            ),
            attribution_delays=(2, 4),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            write_attribution_warmup_summary(run_dir, stats)
            payload = json.loads(
                (run_dir / 'attribution_warmup.json').read_text(
                    encoding='utf-8',
                )
            )

        self.assertEqual(payload['phase'], 'warmup')
        self.assertEqual(payload['steps'], 8)
        self.assertEqual(payload['events_created'], 3)
        self.assertEqual(payload['pending_event_count_at_end'], 1)
        self.assertEqual(payload['mean_attribution_delay'], 3.0)
        self.assertEqual(
            payload['event_status_counts'],
            {
                'MERGE_LINEAGE': {'confirmed': 1},
                'REACHABILITY_SEALED': {'pending': 1},
            },
        )

    def test_cleanup_still_closes_tracker_when_replay_flush_fails(self):
        calls = []

        def failing_flush():
            calls.append('flush')
            raise OSError('simulated flush failure')

        collector = SimpleNamespace(
            worker_id=0,
            close=lambda: calls.append('collector') or (),
        )
        replay = SimpleNamespace(flush=failing_flush)
        metrics = SimpleNamespace(
            close=lambda: calls.append('metrics')
        )
        episode_metrics = SimpleNamespace(
            close=lambda: calls.append('episodes')
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            with redirect_stderr(io.StringIO()):
                close_training_resources(
                    run_dir=Path(tmp_dir),
                    replay_buffer=replay,
                    collector=collector,
                    metrics=metrics,
                    episode_metrics=episode_metrics,
                    suppress_errors=True,
                )
            self.assertTrue(
                (Path(tmp_dir) / 'attribution_shutdown.json').exists()
            )

        self.assertEqual(
            calls,
            ['collector', 'flush', 'metrics', 'episodes'],
        )

    def test_reward_and_dqn_use_the_same_gamma(self):
        """训练入口只能从同一个 gamma 构造 TD target 和 Reward V2。"""

        args = parse_args((
            '--gamma',
            '0.87',
            '--lambda-phi',
            '0.4',
            '--capacity-weight',
            '0.5',
            '--recoverability-weight',
            '0.3',
            '--chain-readiness-weight',
            '0.2',
        ))

        env_config = build_env_config(args)

        self.assertEqual(args.gamma, 0.87)
        self.assertEqual(env_config.reward_config.gamma, args.gamma)
        self.assertEqual(env_config.reward_config.lambda_phi, 0.4)

    def test_training_rejects_noncanonical_action_count_for_attribution(self):
        args = parse_args(('--action-count', '7'))

        with self.assertRaisesRegex(
                ValueError,
                'full state attribution requires'):
            validate_args(args)

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
                    '[dqn]',
                    'gamma = 0.87',
                    '',
                    '[reward]',
                    'lambda_phi = 0.4',
                    'capacity_weight = 0.5',
                    'recoverability_weight = 0.3',
                    'chain_readiness_weight = 0.2',
                    'terminal_penalty = 0.0',
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
        self.assertEqual(args.gamma, 0.87)
        self.assertEqual(args.lambda_phi, 0.4)
        self.assertEqual(args.capacity_weight, 0.5)
        self.assertEqual(args.recoverability_weight, 0.3)
        self.assertEqual(args.chain_readiness_weight, 0.2)
        self.assertEqual(args.terminal_penalty, 0.0)
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
