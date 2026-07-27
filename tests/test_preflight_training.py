"""正式训练环境预检的配置契约测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.preflight_training import (
    PROJECT_ROOT,
    _centralized_actor_rollout_audit,
    _cgroup_memory_status,
    _formal_config_audit,
    _require_full_structural_optimizer_stats,
    _require_full_structural_target,
    parse_training_args,
)
from daxigua_rl.scripts.train_dqn import scheduled_epsilon
from daxigua_rl.training.structural_targets import (
    STRUCTURAL_TARGET_FULL_VALID_MASK,
    StructuralTarget,
)


class PreflightTrainingTest(unittest.TestCase):
    def _config_args(self, name):
        return parse_training_args((
            '--config',
            str(PROJECT_ROOT / 'configs' / name),
        ))

    def test_formal_250k_config_satisfies_frozen_contract(self):
        audit = _formal_config_audit(
            self._config_args('train_dqn_causal_250k.toml')
        )

        self.assertTrue(audit['passed'], audit['mismatches'])
        self.assertEqual(audit['mismatches'], {})
        self.assertEqual(
            audit['value_contract']['run_dir'],
            'runs/dqn_causal_structure_h256_l4_n3_250k',
        )

    def test_smoke_config_cannot_replace_formal_preflight_target(self):
        audit = _formal_config_audit(
            self._config_args(
                'train_dqn_causal_smoke_5k.toml'
            )
        )

        self.assertFalse(audit['passed'])
        self.assertIn('total_updates', audit['mismatches'])

    def test_disabled_attribution_fails_formal_contract(self):
        args = self._config_args(
            'train_dqn_causal_250k.toml'
        )
        args.counterfactual_enabled = False
        args.shapley_enabled = False
        args.lambda_cf = 0.0

        audit = _formal_config_audit(args)

        self.assertFalse(audit['passed'])
        self.assertIn(
            'counterfactual_enabled',
            audit['mismatches'],
        )
        self.assertIn('shapley_enabled', audit['mismatches'])
        self.assertIn('lambda_cf', audit['mismatches'])

    def test_optional_500k_config_cannot_replace_formal_preflight_target(self):
        audit = _formal_config_audit(
            self._config_args('train_dqn_causal_500k.toml')
        )

        self.assertFalse(audit['passed'])
        self.assertEqual(
            audit['mismatches']['total_updates'],
            {'actual': 500_000, 'expected': 250_000},
        )

    def test_formal_250k_config_uses_the_frozen_smooth_epsilon_horizon(self):
        args = self._config_args('train_dqn_causal_250k.toml')
        anchors = {
            75_000: 0.50,
            125_000: 0.20,
            175_000: 0.07,
            200_000: 0.05,
            250_000: 0.05,
        }

        for update_step, expected in anchors.items():
            with self.subTest(update_step=update_step):
                self.assertAlmostEqual(
                    scheduled_epsilon(
                        update_step,
                        env_steps=0,
                        args=args,
                        schedule_total_updates=args.total_updates,
                    ),
                    expected,
                )

    def test_cgroup_memory_uses_reclaimable_working_set_headroom(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / 'memory.max').write_text(
                str(25 * 1024 ** 3),
                encoding='ascii',
            )
            (root / 'memory.current').write_text(
                str(19 * 1024 ** 3),
                encoding='ascii',
            )
            (root / 'memory.stat').write_text(
                f'anon {1 * 1024 ** 3}\n'
                f'inactive_file {16 * 1024 ** 3}\n',
                encoding='ascii',
            )
            paths = ((
                root / 'memory.max',
                root / 'memory.current',
                root / 'memory.stat',
            ),)
            status = _cgroup_memory_status(paths)

        self.assertEqual(status['raw_available_bytes'], 6 * 1024 ** 3)
        self.assertEqual(
            status['effective_available_bytes'],
            22 * 1024 ** 3,
        )
        self.assertEqual(status['working_set_bytes'], 3 * 1024 ** 3)

    def test_cgroup_memory_missing_stat_remains_conservative(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / 'memory.max').write_text(
                str(25 * 1024 ** 3),
                encoding='ascii',
            )
            (root / 'memory.current').write_text(
                str(19 * 1024 ** 3),
                encoding='ascii',
            )
            paths = ((
                root / 'memory.max',
                root / 'memory.current',
                root / 'missing.stat',
            ),)
            status = _cgroup_memory_status(paths)

        self.assertEqual(
            status['effective_available_bytes'],
            6 * 1024 ** 3,
        )

    def test_actor_preflight_exercises_small_real_async_chain(self):
        args = self._config_args(
            'train_dqn_causal_250k.toml'
        )
        # 保留正式配置的 16-worker 目标，只把这个集成单测实际启动的 worker 数量
        # 和模型缩小；生产 preflight 不会传 override。
        args.device = 'cpu'
        args.hidden_dim = 16
        args.message_layers = 1
        args.n_step = 1
        args.physics_fps = 30
        args.max_physics_frames = 40
        args.stable_frames = 2
        args.space_iterations = 8
        args.actor_batch_size = 2
        args.actor_batch_wait_ms = 50.0
        args.actor_request_timeout_seconds = 15.0

        audit = _centralized_actor_rollout_audit(
            args,
            worker_count_override=2,
        )

        self.assertEqual(audit['configured_worker_count'], 16)
        self.assertEqual(audit['exercised_worker_count'], 2)
        self.assertEqual(audit['formal_worker_target'], 16)
        self.assertEqual(audit['sync_count'], 2)
        self.assertTrue(audit['async_start_finish_exercised'])
        self.assertEqual(audit['greedy_responses_verified'], 4)
        self.assertEqual(audit['actor_requests'], 4)
        self.assertGreater(audit['actor_batches'], 0)
        self.assertGreaterEqual(audit['actor_max_batch'], 1)
        self.assertLessEqual(audit['actor_max_batch'], 2)
        self.assertEqual(audit['replay_size_after_close'], 4)
        self.assertEqual(audit['worker_finalizations'], 2)
        self.assertTrue(audit['closed_cleanly'])

    def test_full_structural_gate_rejects_missing_analysis_dimension(self):
        target = StructuralTarget(
            values=(0.0,) * 6,
            valid_mask=STRUCTURAL_TARGET_FULL_VALID_MASK ^ 1,
        )

        with self.assertRaisesRegex(
                RuntimeError,
                'all six structural target dimensions'):
            _require_full_structural_target(target)

    def test_full_structural_gate_requires_six_dimensions_per_sample(self):
        batch_size = 8
        incomplete = SimpleNamespace(
            structural_sample_count=batch_size,
            structural_valid_count=batch_size * 6 - 1,
        )
        complete = SimpleNamespace(
            structural_sample_count=batch_size,
            structural_valid_count=batch_size * 6,
        )

        with self.assertRaisesRegex(
                RuntimeError,
                'did not consume all six'):
            _require_full_structural_optimizer_stats(
                incomplete,
                batch_size,
            )
        self.assertEqual(
            _require_full_structural_optimizer_stats(
                complete,
                batch_size,
            ),
            batch_size * 6,
        )


if __name__ == '__main__':
    unittest.main()
