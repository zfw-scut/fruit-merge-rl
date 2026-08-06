"""Reward V2 21列几何、补偿与队列对齐测试。"""

from dataclasses import replace
import unittest

import torch

from daxigua.simulator import (
    AccessibleSpaceCalculator,
    SimulatorConfig,
    SpatialRewardComputer,
    SpatialRewardConfig,
    TensorVectorSimulator,
    build_standard_compensation_table,
    diagnose_spatial_reward,
)
from daxigua.simulator.types import BatchStepResult


class SpatialRewardGeometryTest(unittest.TestCase):
    def setUp(self):
        self.config = SimulatorConfig(
            max_fruits=64,
            action_count=21,
            use_cuda_extension=False,
        )
        self.simulator = TensorVectorSimulator(
            2, config=self.config, device='cpu'
        )
        self.calculator = AccessibleSpaceCalculator(
            self.config, device='cpu'
        )

    def test_empty_board_has_full_normalized_area_for_each_size(self):
        levels = torch.tensor(((1, 3, 5), (5, 2, 4)))
        analysis = self.calculator.analyze(
            self.simulator.observe(), levels
        )

        self.assertEqual(analysis.drop_x.shape, (2, 3, 21))
        self.assertEqual(analysis.depths.shape, (2, 3, 21))
        self.assertTrue(torch.allclose(
            analysis.normalized_areas, torch.ones((2, 3))
        ))

    def test_spawn_overlap_makes_the_exact_column_unavailable(self):
        self.simulator.positions[0, 0] = torch.tensor((280.0, 252.0))
        self.simulator.physics_radii[0, 0] = 20.0
        self.simulator.levels[0, 0] = 1
        self.simulator.active[0, 0] = True

        analysis = self.calculator.analyze(
            self.simulator.observe(), torch.tensor(((1,),))
        )

        self.assertAlmostEqual(float(analysis.drop_x[0, 0, 10]), 280.0)
        self.assertEqual(float(analysis.depths[0, 0, 10]), 0.0)
        self.assertLess(float(analysis.normalized_areas[0, 0]), 1.0)

    def test_floor_obstacle_reduces_space_without_hiding_other_columns(self):
        self.simulator.positions[0, 0] = torch.tensor((280.0, 1080.0))
        self.simulator.physics_radii[0, 0] = 20.0
        self.simulator.levels[0, 0] = 1
        self.simulator.active[0, 0] = True

        analysis = self.calculator.analyze(
            self.simulator.observe(), torch.tensor(((1,),))
        )

        self.assertLess(
            float(analysis.depths[0, 0, 10]),
            float(analysis.depths[0, 0, 0]),
        )
        self.assertGreater(float(analysis.normalized_areas[0, 0]), 0.9)
        self.assertLess(float(analysis.normalized_areas[0, 0]), 1.0)

    def test_standard_compensation_is_small_positive_and_action_independent(self):
        table = build_standard_compensation_table(self.config)

        self.assertEqual(len(table), 12)
        self.assertEqual(table[0][0], 0.0)
        for drop_level in range(1, 6):
            for future_level in range(1, 6):
                self.assertGreater(table[drop_level][future_level], 0.0)
                self.assertLess(table[drop_level][future_level], 1.0)

    def test_reward_config_rejects_invalid_weights_and_scale(self):
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            SpatialRewardConfig(queue_weights=(0.5, 0.3, 0.3))
        with self.assertRaisesRegex(ValueError, 'positive'):
            SpatialRewardConfig(reward_scale=0.0)


class SpatialRewardTransitionTest(unittest.TestCase):
    def setUp(self):
        self.config = SimulatorConfig(
            max_fruits=64,
            action_count=21,
            physics_fps=30,
            max_physics_frames=180,
            stable_frames=4,
            use_cuda_extension=False,
        )
        self.simulator = TensorVectorSimulator(
            1, config=self.config, device='cpu'
        )
        self.simulator.fruit_queue[0] = torch.tensor((1, 2, 3, 4))

    def test_cached_reward_matches_full_diagnostic_recalculation(self):
        previous = self.simulator.observe().clone()
        computer = SpatialRewardComputer(
            self.config, device='cpu',
            reward_config=SpatialRewardConfig(reward_scale=7.0),
        )
        computer.initialize(previous)

        result = self.simulator.step(torch.tensor((10,)))
        cached = computer.step(result)
        diagnostic = diagnose_spatial_reward(
            computer.calculator, previous, result
        )

        self.assertTrue(torch.allclose(cached.reward, diagnostic.reward))
        self.assertTrue(torch.allclose(
            cached.previous_potential, diagnostic.previous_potential
        ))
        self.assertTrue(torch.allclose(
            cached.next_potential, diagnostic.next_potential
        ))
        self.assertEqual(
            result.drop.queue_before[0, 1:4].tolist(),
            result.drop.queue_after[0, 0:3].tolist(),
        )

    def test_terminal_forces_next_potential_to_zero(self):
        previous = self.simulator.observe().clone()
        result = self.simulator.step(torch.tensor((10,)))
        terminal_physics = replace(
            result.physics, done=torch.tensor((True,))
        )
        terminal_result = BatchStepResult(
            observation=result.observation,
            drop=result.drop,
            physics=terminal_physics,
        )
        calculator = AccessibleSpaceCalculator(
            self.config, device='cpu'
        )

        diagnostic = diagnose_spatial_reward(
            calculator, previous, terminal_result
        )

        self.assertEqual(float(diagnostic.next_potential[0]), 0.0)
        self.assertTrue(bool(diagnostic.terminal[0]))
        self.assertLess(float(diagnostic.raw_space_delta[0]), 0.0)

    def test_reset_rows_restores_empty_board_cached_potential(self):
        computer = SpatialRewardComputer(self.config, device='cpu')
        computer.initialize(self.simulator.observe())
        computer._previous_potential[0] = 0.25

        computer.reset_rows(torch.tensor((True,)))

        self.assertEqual(float(computer._previous_potential[0]), 1.0)


if __name__ == '__main__':
    unittest.main()
