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
    clear_active_failure_diagnostic,
    close_training_resources,
    evaluate_policy,
    load_config_defaults,
    parse_args,
    process_counterfactual_rollout,
    validate_args,
    write_attribution_warmup_summary,
)
from daxigua_rl.training.collector import RolloutStats


class TrainingMetricsTest(unittest.TestCase):
    """验证训练脚本新增的评估和 episode 指标。"""

    def test_successful_resume_clears_only_active_failure_pointer(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            latest = run_dir / 'failure_latest.json'
            history = run_dir / 'failure_20260727_000000_000000.json'
            latest.write_text('{}', encoding='utf-8')
            history.write_text('{}', encoding='utf-8')

            clear_active_failure_diagnostic(run_dir)
            clear_active_failure_diagnostic(run_dir)

            self.assertFalse(latest.exists())
            self.assertTrue(history.exists())

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

    def test_episode_logger_resume_backs_up_orphan_tail_and_continues_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'episode_metrics.csv'
            stats = RolloutStats(
                steps=1,
                episodes=1,
                total_reward=1.0,
                episode_rewards=(1.0,),
                episode_lengths=(1,),
                episode_scores=(2.0,),
                episode_end_offsets=(1,),
                episode_terminated_flags=(True,),
                episode_truncated_flags=(False,),
            )
            initial = EpisodeLogger(path)
            initial.log_collect_stats(
                stats,
                phase='train',
                update_step=1,
                start_env_steps=0,
                epsilon=1.0,
            )
            initial.log_collect_stats(
                stats,
                phase='train',
                update_step=4,
                start_env_steps=1,
                epsilon=0.5,
            )
            initial.close()

            resumed = EpisodeLogger(
                path,
                resume_update_step=2,
            )
            try:
                self.assertEqual(resumed.orphaned_row_count, 1)
                self.assertTrue(
                    resumed.orphaned_backup_path.is_file()
                )
                resumed.log_collect_stats(
                    stats,
                    phase='train',
                    update_step=3,
                    start_env_steps=1,
                    epsilon=0.75,
                )
            finally:
                resumed.close()

            with path.open(newline='', encoding='utf-8') as file_obj:
                rows = list(csv.DictReader(file_obj))
        self.assertEqual(
            [row['update_step'] for row in rows],
            ['1', '3'],
        )
        self.assertEqual(
            [row['episode_index'] for row in rows],
            ['1', '2'],
        )

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
            replay_transitions_emitted=3,
            n_step_pending_count=2,
            causal_rule_input_event_count=7,
            causal_rule_eligible_event_count=5,
            causal_rule_budget_count=3,
            causal_samples_emitted=2,
            causal_rule_skip_reason_counts=(
                ('missing_context', 1),
                ('confidence_tier_c', 2),
            ),
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
            td_loss=0.8,
            rule_rank_loss=0.5,
            weighted_rule_rank_loss=0.075,
            counterfactual_loss=0.25,
            weighted_counterfactual_loss=0.025,
            structural_loss=0.2,
            weighted_structural_loss=0.03,
            structural_valid_count=24,
            structural_sample_count=6,
            structural_mean_abs_error=0.35,
            causal_update_applied=True,
            causal_batch_size=6,
            rule_batch_size=4,
            counterfactual_batch_size=1,
            shapley_batch_size=1,
            rule_pair_accuracy=0.75,
            rule_margin_satisfaction_rate=0.5,
            counterfactual_sign_accuracy=1.0,
            counterfactual_mean_abs_error=0.4,
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
            timing={
                'elapsed': 2.0,
                'completed_updates': 3,
                'completed_env_steps': 8,
            },
            causal_replay_stats={
                'total_count': 9,
                'stratum_counts': {
                    'positive_setup': 4,
                    'negative_blocking': 3,
                    'counterfactual': 2,
                },
                'supervision_kind_counts': {
                    'rule': 7,
                    'counterfactual': 1,
                    'shapley': 1,
                },
                'estimated_unique_graph_bytes': 1234,
                'estimated_graph_sharing_saved_bytes': 567,
            },
            counterfactual_stats=SimpleNamespace(
                enabled=True,
                worker_count=2,
                pending_task_count=3,
                candidate_pool_capacity=256,
                candidate_pool_count=4,
                cumulative=SimpleNamespace(
                    candidate_offers=11,
                    candidate_pool_evictions=2,
                    candidate_dispatch_attempts=7,
                    candidate_dispatch_admitted=6,
                    candidate_close_dropped=1,
                    numeric_jitter_dropped=2,
                    semantic_divergence_dropped=1,
                    numeric_jitter_max_merge_event_position_error=0.11,
                    numeric_jitter_max_fruit_position_error=0.12,
                    numeric_jitter_max_linear_velocity_error=0.013,
                    numeric_jitter_max_orientation_error=0.0014,
                    numeric_jitter_max_angular_velocity_error=0.0015,
                ),
                scheduler=None,
            ),
            shapley_stats=SimpleNamespace(
                enabled=True,
                observed_event_count=20,
                selected_event_count=2,
                pending_task_count=0,
                cumulative=SimpleNamespace(
                    numeric_jitter_dropped=1,
                    semantic_divergence_dropped=0,
                    numeric_jitter_max_merge_event_position_error=0.21,
                    numeric_jitter_max_fruit_position_error=0.22,
                    numeric_jitter_max_linear_velocity_error=0.023,
                    numeric_jitter_max_orientation_error=0.0024,
                    numeric_jitter_max_angular_velocity_error=0.0025,
                ),
            ),
            actor_stats={
                'requests': 40,
                'batches': 5,
                'mean_batch_size': 8.0,
                'max_batch': 12,
                'seconds': 0.125,
            },
        )

        self.assertEqual(row['td_loss'], 0.8)
        self.assertEqual(row['rule_rank_loss'], 0.5)
        self.assertEqual(row['counterfactual_loss'], 0.25)
        self.assertEqual(row['structural_loss'], 0.2)
        self.assertEqual(row['weighted_structural_loss'], 0.03)
        self.assertEqual(row['structural_valid_count'], 24)
        self.assertEqual(row['structural_sample_count'], 6)
        self.assertEqual(row['structural_mean_abs_error'], 0.35)
        self.assertEqual(row['actor_inference_requests'], 40)
        self.assertEqual(row['actor_inference_batches'], 5)
        self.assertEqual(row['actor_inference_mean_batch_size'], 8.0)
        self.assertEqual(row['actor_inference_max_batch'], 12)
        self.assertEqual(row['actor_inference_seconds'], 0.125)
        self.assertEqual(row['causal_batch_size'], 6)
        self.assertEqual(row['collect_replay_transitions_emitted'], 3)
        self.assertEqual(row['collect_n_step_pending_count'], 2)
        self.assertEqual(row['collect_causal_samples_emitted'], 2)
        self.assertEqual(row['collect_rule_causal_budget_count'], 3)
        self.assertEqual(
            row['collect_rule_causal_skip_reasons'],
            '{"confidence_tier_c":2,"missing_context":1}',
        )
        self.assertEqual(row['causal_replay_size'], 9)
        self.assertEqual(row['causal_replay_positive_count'], 4)
        self.assertEqual(row['causal_replay_negative_count'], 3)
        self.assertEqual(row['causal_replay_counterfactual_count'], 2)
        self.assertEqual(row['causal_replay_shared_tensor_bytes'], 1234)
        self.assertEqual(row['causal_replay_saved_tensor_bytes'], 567)
        self.assertEqual(row['counterfactual_candidate_pool_capacity'], 256)
        self.assertEqual(row['counterfactual_candidate_pool_count'], 4)
        self.assertEqual(row['counterfactual_candidate_offers'], 11)
        self.assertEqual(row['counterfactual_candidate_pool_evictions'], 2)
        self.assertEqual(
            row['counterfactual_candidate_dispatch_attempts'],
            7,
        )
        self.assertEqual(
            row['counterfactual_candidate_dispatch_admitted'],
            6,
        )
        self.assertEqual(
            row['counterfactual_candidate_close_dropped'],
            1,
        )
        self.assertEqual(row['counterfactual_numeric_jitter_dropped'], 2)
        self.assertEqual(
            row['counterfactual_semantic_divergence_dropped'],
            1,
        )
        self.assertEqual(
            row[
                'counterfactual_numeric_jitter_max_'
                'linear_velocity_error'
            ],
            0.013,
        )
        self.assertEqual(row['shapley_numeric_jitter_dropped'], 1)
        self.assertEqual(
            row['shapley_semantic_divergence_dropped'],
            0,
        )
        self.assertEqual(
            row['shapley_numeric_jitter_max_orientation_error'],
            0.0024,
        )
        self.assertEqual(row['updates_per_second'], 1.5)
        self.assertEqual(row['env_steps_per_second'], 4.0)
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

    def test_counterfactual_rollout_routes_shapley_before_pool_offer(self):
        calls = []
        shapley_proposal = object()
        ordinary_proposal = object()

        collector = SimpleNamespace(
            drain_counterfactual_proposals=lambda: (
                calls.append(('drain',))
                or (shapley_proposal, ordinary_proposal)
            ),
        )
        coordinator = SimpleNamespace(
            target_policy='target-policy',
            stats=SimpleNamespace(
                scheduler=SimpleNamespace(real_steps=77),
            ),
            record_real_steps=lambda steps, dispatch_candidates: calls.append((
                'record_real_steps',
                steps,
                dispatch_candidates,
            )),
            offer_many=lambda proposals: (
                calls.append(('offer_many', tuple(proposals)))
                or ('ordinary-submission',)
            ),
            poll=lambda: calls.append(('counterfactual_poll',)),
        )

        def consider(proposal, policy, *, created_real_step):
            calls.append((
                'shapley_consider',
                proposal,
                policy,
                created_real_step,
            ))
            return SimpleNamespace(
                skip_counterfactual=(
                    proposal is shapley_proposal
                ),
            )

        shapley = SimpleNamespace(
            retry_pending=lambda: calls.append(('shapley_retry',)),
            consider=consider,
            poll=lambda: calls.append(('shapley_poll',)),
        )

        submissions = process_counterfactual_rollout(
            collector,
            coordinator,
            SimpleNamespace(steps=12),
            shapley_coordinator=shapley,
        )

        self.assertEqual(submissions, ('ordinary-submission',))
        self.assertEqual(
            calls,
            [
                ('record_real_steps', 12, False),
                ('shapley_retry',),
                ('drain',),
                (
                    'shapley_consider',
                    shapley_proposal,
                    'target-policy',
                    77,
                ),
                (
                    'shapley_consider',
                    ordinary_proposal,
                    'target-policy',
                    77,
                ),
                ('offer_many', (ordinary_proposal,)),
                ('counterfactual_poll',),
                ('shapley_poll',),
            ],
        )

    def test_warmup_attribution_summary_is_kept_separate(self):
        stats = RolloutStats(
            steps=8,
            episodes=0,
            total_reward=0.0,
            counterfactual_snapshot_failures=0,
            state_analysis_calls=8,
            state_analysis_degraded_count=1,
            potential_shaping_abs_values=(
                0.1,
                0.2,
                0.3,
            ),
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
        self.assertEqual(payload['schema_version'], 1)
        self.assertEqual(payload['steps'], 8)
        self.assertEqual(
            payload['counterfactual_snapshot_failures'],
            0,
        )
        self.assertEqual(payload['state_analysis_calls'], 8)
        self.assertEqual(payload['state_analysis_degraded_count'], 1)
        self.assertEqual(payload['state_analysis_degraded_rate'], 0.125)
        self.assertAlmostEqual(
            payload['p95_abs_potential_shaping_reward'],
            0.29,
        )
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

    def test_central_actor_and_structural_cli_validation(self):
        args = parse_args((
            '--centralized-actor-inference',
            '--num-envs',
            '1',
        ))
        with self.assertRaisesRegex(
                ValueError,
                'centralized-actor-inference requires'):
            validate_args(args)

        args = parse_args((
            '--centralized-actor-inference',
            '--num-envs',
            '2',
            '--actor-batch-wait-ms',
            '-1',
        ))
        with self.assertRaisesRegex(
                ValueError,
                'actor-batch-wait-ms'):
            validate_args(args)

        args = parse_args(('--lambda-structural', '-0.1'))
        with self.assertRaisesRegex(
                ValueError,
                'lambda-structural'):
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

    def test_toml_config_extends_base_and_rejects_cycles(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            base = root / 'base.toml'
            child = root / 'child.toml'
            base.write_text(
                '[training]\ntotal_updates = 100\nbatch_size = 64\n',
                encoding='utf-8',
            )
            child.write_text(
                'extends = "base.toml"\n'
                '[training]\ntotal_updates = 5\n',
                encoding='utf-8',
            )
            defaults = load_config_defaults(child)
            self.assertEqual(defaults['total_updates'], 5)
            self.assertEqual(defaults['batch_size'], 64)

            base.write_text(
                'extends = "child.toml"\n',
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'cyclic'):
                load_config_defaults(child)


if __name__ == '__main__':
    unittest.main()
