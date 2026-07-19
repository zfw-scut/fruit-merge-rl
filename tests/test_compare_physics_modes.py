"""物理模式对比脚本测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from daxigua.config import FPS
from daxigua_rl import DaxiguaEnv, DaxiguaEnvConfig
from daxigua_rl.scripts.compare_physics_modes import (
    build_mode_specs,
    parse_fast_fps_values,
    run_episode,
)
from daxigua_rl.reward import RewardConfig


class ComparePhysicsModesTest(unittest.TestCase):
    """验证 accurate/fast 对比脚本的基础行为。"""

    def test_parse_fast_fps_values(self):
        """fast fps 列表应支持逗号分隔配置。"""

        self.assertEqual(parse_fast_fps_values('15,30,45'), (15, 30, 45))
        self.assertEqual(parse_fast_fps_values(' 30 '), (30,))

    def test_build_mode_specs_contains_accurate_and_fast_modes(self):
        """默认模式应包含 accurate 和用户关心的 fast fps 组合。"""

        args = SimpleNamespace(
            accurate_fps=FPS,
            accurate_max_physics_frames=720,
            accurate_stable_frames=15,
            accurate_space_iterations=32,
            fast_fps_values='15,30,45',
            fast_max_physics_frames=200,
            fast_stable_frames=6,
            fast_space_iterations=8,
        )

        modes = build_mode_specs(args)

        self.assertEqual(modes[0].name, 'accurate')
        self.assertEqual(tuple(mode.fps for mode in modes[1:]), (15, 30, 45))
        self.assertTrue(all(mode.space_iterations == 8 for mode in modes[1:]))

    def test_env_config_controls_headless_physics_parameters(self):
        """DaxiguaEnvConfig 应能控制 headless 物理 fps 和迭代次数。"""

        env = DaxiguaEnv(
            config=DaxiguaEnvConfig(
                action_count=5,
                physics_fps=30,
                max_physics_frames=120,
                stable_frames=4,
                space_iterations=8,
            )
        )

        self.assertEqual(env.game.fps, 30)
        self.assertEqual(env.game.space.iterations, 8)

    def test_run_episode_returns_basic_metrics(self):
        """随机策略下单局对比应返回速度、分数和物理帧统计。"""

        args = SimpleNamespace(seed=0, max_steps=2)
        mode = build_mode_specs(
            SimpleNamespace(
                accurate_fps=30,
                accurate_max_physics_frames=80,
                accurate_stable_frames=3,
                accurate_space_iterations=8,
                fast_fps_values='15',
                fast_max_physics_frames=80,
                fast_stable_frames=3,
                fast_space_iterations=8,
            )
        )[0]

        result = run_episode(
            mode=mode,
            args=args,
            action_count=5,
            reward_config=RewardConfig(),
            model=None,
            graph_builder=None,
            device='cpu',
            episode_index=0,
        )

        self.assertEqual(result.mode, 'accurate')
        self.assertGreaterEqual(result.episode_length, 1)
        self.assertGreaterEqual(result.mean_physics_frames, 0.0)
        self.assertGreaterEqual(result.env_steps_per_second, 0.0)


if __name__ == '__main__':
    unittest.main()
