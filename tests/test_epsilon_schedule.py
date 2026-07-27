"""epsilon 衰减曲线测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from daxigua_rl.scripts.train_dqn import (
    linear_epsilon,
    online_policy_version,
    scheduled_epsilon,
    should_sync_parallel_workers,
)


class EpsilonScheduleTest(unittest.TestCase):
    """验证训练脚本中的 epsilon schedule。"""

    def _args(self, schedule='smooth'):
        return SimpleNamespace(
            epsilon_schedule=schedule,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=50_000,
            total_updates=100,
        )

    def test_smooth_epsilon_matches_design_anchors(self):
        """smooth schedule 应大致符合 30/50/70/80 进度点。"""

        args = self._args('smooth')
        anchors = {
            0: 1.0,
            30: 0.5,
            50: 0.2,
            70: 0.07,
            80: 0.05,
            100: 0.05,
        }

        for update_step, expected_epsilon in anchors.items():
            with self.subTest(update_step=update_step):
                epsilon = scheduled_epsilon(update_step, env_steps=0, args=args)
                self.assertAlmostEqual(epsilon, expected_epsilon, places=6)

    def test_smooth_epsilon_is_monotonic(self):
        """smooth schedule 在整个训练过程中应单调不增。"""

        args = self._args('smooth')
        values = [scheduled_epsilon(step, env_steps=0, args=args) for step in range(0, 101)]
        for previous, current in zip(values, values[1:]):
            self.assertLessEqual(current, previous + 1e-9)

    def test_linear_epsilon_keeps_old_env_step_behavior(self):
        """linear schedule 继续按 env_steps 和 epsilon_decay_steps 衰减。"""

        args = self._args('linear')
        self.assertAlmostEqual(linear_epsilon(0, args), 1.0)
        self.assertAlmostEqual(linear_epsilon(25_000, args), 0.525)
        self.assertAlmostEqual(linear_epsilon(50_000, args), 0.05)
        self.assertAlmostEqual(scheduled_epsilon(30, env_steps=25_000, args=args), 0.525)

    def test_fresh_resumed_parallel_collector_syncs_off_cycle(self):
        """非周期边界恢复时，新 worker 也必须先收到一次模型。"""

        self.assertTrue(
            should_sync_parallel_workers(
                61,
                100,
                model_synced=False,
            )
        )
        self.assertFalse(
            should_sync_parallel_workers(
                61,
                100,
                model_synced=True,
            )
        )
        self.assertTrue(
            should_sync_parallel_workers(
                101,
                100,
                model_synced=True,
            )
        )

    def test_online_policy_version_is_absolute_across_resume(self):
        """恢复进程不能把不同权重重新命名为 parallel-sync-1。"""

        self.assertEqual(online_policy_version(0), 'online-update-00000000')
        self.assertEqual(
            online_policy_version(12_345),
            'online-update-00012345',
        )
        with self.assertRaises(ValueError):
            online_policy_version(-1)

    def test_extended_resume_does_not_reexpand_smooth_exploration(self):
        args = self._args()
        args.total_updates = 25_000

        self.assertAlmostEqual(
            scheduled_epsilon(
                10_000,
                env_steps=80_000,
                args=args,
                schedule_total_updates=10_000,
            ),
            0.05,
        )
        self.assertAlmostEqual(
            scheduled_epsilon(
                10_001,
                env_steps=80_008,
                args=args,
                schedule_total_updates=10_000,
            ),
            0.05,
        )


if __name__ == '__main__':
    unittest.main()
