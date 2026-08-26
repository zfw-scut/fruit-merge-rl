import os
import unittest

import torch
from torch import nn

from daxigua.core import merged_fruit_physics_radius
from daxigua.rl.mergeability import (
    ExternalSupplyEstimate,
    MERGEABILITY_SOURCE_INTERNAL,
    MERGEABILITY_SOURCE_NONE,
    MergeabilityCalculator,
    MergeabilityConfig,
)


def _scene(items, *, batch=1, slots=8, device='cpu'):
    positions = torch.zeros((batch, slots, 2), device=device)
    radii = torch.zeros((batch, slots), device=device)
    levels = torch.zeros((batch, slots), dtype=torch.int64, device=device)
    active = torch.zeros((batch, slots), dtype=torch.bool, device=device)
    for slot, level, x, y in items:
        active[:, slot] = True
        levels[:, slot] = level
        radii[:, slot] = float(merged_fruit_physics_radius(level))
        positions[:, slot, 0] = float(x)
        positions[:, slot, 1] = float(y)
    return positions, radii, levels, active


class _UnavailableExternalSupply(nn.Module):
    def estimate(self, positions, radii, levels, active):
        shape = levels.shape
        difficulty = torch.full(
            shape, float('inf'), dtype=positions.dtype, device=positions.device
        )
        zeros = torch.zeros_like(radii)
        zero_levels = torch.zeros_like(levels)
        return ExternalSupplyEstimate(
            difficulty, zeros, zero_levels, zero_levels
        )


class MergeabilityCalculatorTest(unittest.TestCase):
    def setUp(self):
        self.config = MergeabilityConfig(top_k=3)
        self.calculator = MergeabilityCalculator(self.config)

    def test_same_level_neighbor_is_direct_internal_terminal(self):
        tensors = _scene((
            (0, 7, 200.0, 900.0),
            (1, 7, 348.0, 900.0),
        ))
        result = self.calculator.compute(*tensors)
        self.assertEqual(int(result.source[0, 0]), MERGEABILITY_SOURCE_INTERNAL)
        self.assertEqual(int(result.primary_dependency_slot[0, 0]), 1)
        self.assertEqual(float(result.internal_difficulty[0, 0]), 0.0)
        self.assertEqual(float(result.score[0, 0]), 1.0)
        self.assertEqual(float(result.material_score[0, 0]), 1.0)
        self.assertEqual(float(result.spatial_score[0, 0]), 1.0)

    def test_material_score_ignores_geometry_but_spatial_score_does_not(self):
        calculator = MergeabilityCalculator(
            self.config, external_supply=_UnavailableExternalSupply()
        )
        tensors = _scene((
            (0, 7, 80.0, 900.0),
            (1, 7, 480.0, 900.0),
        ))
        result = calculator.compute(*tensors)
        self.assertEqual(float(result.material_score[0, 0]), 1.0)
        self.assertEqual(float(result.score[0, 0]), 0.0)
        self.assertEqual(float(result.spatial_score[0, 0]), 0.0)

    def test_actual_score_never_exceeds_material_score(self):
        tensors = _scene((
            (0, 9, 90.0, 900.0),
            (1, 8, 240.0, 900.0),
            (2, 7, 390.0, 760.0),
            (3, 5, 470.0, 540.0),
        ), batch=3)
        result = self.calculator.compute(*tensors)
        self.assertTrue(bool((result.score <= result.material_score + 1e-6).all()))
        self.assertTrue(bool((result.spatial_score >= 0.0).all()))
        self.assertTrue(bool((result.spatial_score <= 1.0).all()))

    def test_lower_level_candidate_propagates_without_same_level_cycle(self):
        tensors = _scene((
            (0, 7, 200.0, 900.0),
            (1, 6, 345.0, 900.0),
        ))
        result = self.calculator.compute(*tensors)
        self.assertEqual(int(result.source[0, 0]), MERGEABILITY_SOURCE_INTERNAL)
        self.assertEqual(int(result.primary_dependency_slot[0, 0]), 1)
        self.assertAlmostEqual(float(result.difficulty[0, 0]), 2.0)
        self.assertAlmostEqual(float(result.score[0, 0]), 2.0 / 3.0, places=5)
        self.assertNotEqual(
            int(result.primary_dependency_slot[0, 1]), 0,
            'lower level fruit must not depend upward on the L7 target',
        )

    def test_large_upper_obstacle_closes_all_vertical_probes(self):
        tensors = _scene((
            (0, 5, 280.0, 900.0),
            (1, 11, 280.0, 500.0),
        ))
        result = self.calculator.compute(*tensors)
        self.assertEqual(int(result.external_capacity_level[0, 0]), 0)
        self.assertEqual(float(result.external_score[0, 0]), 0.0)
        self.assertEqual(int(result.source[0, 0]), MERGEABILITY_SOURCE_NONE)
        self.assertEqual(float(result.score[0, 0]), 0.0)

    def test_top_k_retains_only_three_closest_same_level_candidates(self):
        tensors = _scene((
            (0, 8, 280.0, 850.0),
            (1, 8, 120.0, 850.0),
            (2, 8, 440.0, 850.0),
            (3, 8, 280.0, 690.0),
            (4, 8, 280.0, 1010.0),
        ))
        result = self.calculator.compute(*tensors)
        self.assertEqual(int(result.dependency_valid[0, 0].sum()), 3)
        selected = set(
            result.dependency_slots[0, 0][
                result.dependency_valid[0, 0]
            ].tolist()
        )
        self.assertEqual(len(selected), 3)
        self.assertTrue(selected.issubset({1, 2, 3, 4}))

    def test_inactive_slots_are_zero_and_have_no_dependency(self):
        tensors = _scene(((0, 3, 280.0, 900.0),))
        result = self.calculator.compute(*tensors)
        self.assertTrue(bool((result.score[0, 1:] == 0.0).all()))
        self.assertTrue(bool((result.source[0, 1:] == 0).all()))
        self.assertTrue(bool(
            (result.primary_dependency_slot[0, 1:] == -1).all()
        ))

    def test_external_strategy_can_be_replaced_without_changing_output(self):
        calculator = MergeabilityCalculator(
            self.config, external_supply=_UnavailableExternalSupply()
        )
        tensors = _scene((
            (0, 4, 220.0, 900.0),
            (1, 4, 310.0, 900.0),
        ))
        result = calculator.compute(*tensors)
        self.assertEqual(float(result.score[0, 0]), 1.0)
        self.assertEqual(int(result.source[0, 0]), MERGEABILITY_SOURCE_INTERNAL)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and os.environ.get('DAXIGUA_RUN_CUDA_TESTS') == '1',
        'explicit CUDA test is disabled on the local diagnostic machine',
    )
    def test_cpu_and_cuda_results_match(self):
        cpu_inputs = _scene((
            (0, 8, 280.0, 850.0),
            (1, 7, 135.0, 875.0),
            (2, 6, 420.0, 760.0),
            (3, 5, 285.0, 600.0),
        ), batch=4)
        cpu_result = self.calculator.compute(*cpu_inputs)
        cuda_calculator = MergeabilityCalculator(self.config).cuda()
        cuda_result = cuda_calculator.compute(*(
            tensor.cuda() for tensor in cpu_inputs
        ))
        for cpu_value, cuda_value in zip(cpu_result, cuda_result):
            self.assertTrue(torch.allclose(
                cpu_value.to(torch.float32),
                cuda_value.cpu().to(torch.float32),
                atol=1e-5,
                rtol=1e-5,
                equal_nan=True,
            ))


if __name__ == '__main__':
    unittest.main()
