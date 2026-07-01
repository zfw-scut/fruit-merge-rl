"""epsilon 衰减曲线测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from daxigua_rl.scripts.train_dqn import linear_epsilon, scheduled_epsilon


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


if __name__ == '__main__':
    unittest.main()
