import math
import tempfile
import unittest
from pathlib import Path

import torch

from daxigua.rl.mergeability import MergeabilityResult
from daxigua.rl.mergeability_rollout import (
    SCENE_VALUE_DTYPES,
    lineage_aligned_spatial_change,
    scene_mergeability_delta,
    scene_mergeability_values,
    scene_spatial_values,
)
from daxigua.rl.observations import TensorState
from tools.analyze_mergeability_rollout_stats import analyze


def _state():
    return TensorState(
        positions=torch.zeros((1, 3, 2)),
        velocities=torch.zeros((1, 3, 2)),
        angular_velocities=torch.zeros((1, 3)),
        levels=torch.tensor([[2, 3, 0]]),
        physics_radii=torch.tensor([[2.0, 3.0, 0.0]]),
        age_frames=torch.zeros((1, 3), dtype=torch.int64),
        active=torch.tensor([[True, True, False]]),
        fruit_queue=torch.ones((1, 4), dtype=torch.int64),
        danger_progress=torch.zeros(1),
        over_danger_line=torch.zeros(1, dtype=torch.bool),
        physics_fps=30.0,
    )


def _result():
    shape = (1, 3)
    pair_shape = (1, 3, 3)
    return MergeabilityResult(
        score=torch.tensor([[0.5, 0.25, 1.0]]),
        difficulty=torch.zeros(shape),
        material_score=torch.tensor([[0.5, 0.5, 0.0]]),
        material_difficulty=torch.zeros(shape),
        spatial_score=torch.tensor([[1.0, 0.5, 0.0]]),
        internal_score=torch.zeros(shape),
        internal_difficulty=torch.zeros(shape),
        external_score=torch.zeros(shape),
        external_difficulty=torch.zeros(shape),
        source=torch.zeros(shape, dtype=torch.int8),
        primary_dependency_slot=torch.full(shape, -1, dtype=torch.int64),
        dependency_slots=torch.full(pair_shape, -1, dtype=torch.int64),
        dependency_valid=torch.zeros(pair_shape, dtype=torch.bool),
        external_capacity_radius=torch.zeros(shape),
        external_capacity_level=torch.zeros(shape, dtype=torch.int64),
        external_entry_level=torch.zeros(shape, dtype=torch.int64),
    )


class MergeabilityRolloutTest(unittest.TestCase):
    def test_scene_value_uses_active_circle_area(self):
        value, area, mean = scene_mergeability_values(_state(), _result())
        self.assertAlmostEqual(float(area[0]), 13.0 * math.pi, places=5)
        self.assertAlmostEqual(float(value[0]), 4.25 * math.pi, places=5)
        self.assertAlmostEqual(float(mean[0]), 4.25 / 13.0, places=6)

    def test_scene_spatial_value_uses_merge_conserved_material_mass(self):
        score, mass = scene_spatial_values(_state(), _result())
        self.assertAlmostEqual(float(mass[0]), 6.0, places=6)
        self.assertAlmostEqual(float(score[0]), 2.0 / 3.0, places=6)

    def test_lineage_alignment_keeps_normal_merge_unchanged(self):
        change = lineage_aligned_spatial_change(
            before_fruit_ids=torch.tensor([[10, 11]]),
            before_levels=torch.tensor([[7, 7]]),
            before_active=torch.tensor([[True, True]]),
            before_spatial_score=torch.tensor([[1.0, 1.0]]),
            after_fruit_ids=torch.tensor([[12, 0]]),
            after_active=torch.tensor([[True, False]]),
            after_spatial_score=torch.tensor([[1.0, 0.0]]),
            merge_count=torch.tensor([1]),
            merge_source_ids=torch.tensor([[[10, 11]]]),
            merge_new_fruit_ids=torch.tensor([[12]]),
        )
        self.assertTrue(bool(change.valid[0]))
        self.assertAlmostEqual(float(change.coverage[0]), 1.0, places=6)
        self.assertAlmostEqual(float(change.delta[0]), 0.0, places=6)

    def test_lineage_alignment_reports_real_post_merge_spatial_loss(self):
        change = lineage_aligned_spatial_change(
            before_fruit_ids=torch.tensor([[10, 11]]),
            before_levels=torch.tensor([[7, 7]]),
            before_active=torch.tensor([[True, True]]),
            before_spatial_score=torch.tensor([[1.0, 1.0]]),
            after_fruit_ids=torch.tensor([[12, 0]]),
            after_active=torch.tensor([[True, False]]),
            after_spatial_score=torch.tensor([[0.25, 0.0]]),
            merge_count=torch.tensor([1]),
            merge_source_ids=torch.tensor([[[10, 11]]]),
            merge_new_fruit_ids=torch.tensor([[12]]),
        )
        self.assertAlmostEqual(float(change.delta[0]), -0.75, places=6)

    def test_lineage_alignment_follows_multi_event_chain(self):
        change = lineage_aligned_spatial_change(
            before_fruit_ids=torch.tensor([[10, 11]]),
            before_levels=torch.tensor([[6, 6]]),
            before_active=torch.tensor([[True, True]]),
            before_spatial_score=torch.tensor([[0.8, 0.8]]),
            after_fruit_ids=torch.tensor([[13, 0]]),
            after_active=torch.tensor([[True, False]]),
            after_spatial_score=torch.tensor([[0.8, 0.0]]),
            merge_count=torch.tensor([2]),
            merge_source_ids=torch.tensor([[[10, 11], [12, 99]]]),
            merge_new_fruit_ids=torch.tensor([[12, 13]]),
        )
        self.assertTrue(bool(change.valid[0]))
        self.assertAlmostEqual(float(change.delta[0]), 0.0, places=6)

    def test_lineage_alignment_marks_removed_material_invalid(self):
        change = lineage_aligned_spatial_change(
            before_fruit_ids=torch.tensor([[10, 11]]),
            before_levels=torch.tensor([[11, 11]]),
            before_active=torch.tensor([[True, True]]),
            before_spatial_score=torch.tensor([[1.0, 1.0]]),
            after_fruit_ids=torch.tensor([[0, 0]]),
            after_active=torch.tensor([[False, False]]),
            after_spatial_score=torch.tensor([[0.0, 0.0]]),
            merge_count=torch.tensor([1]),
            merge_source_ids=torch.tensor([[[10, 11]]]),
            merge_new_fruit_ids=torch.tensor([[0]]),
        )
        self.assertFalse(bool(change.valid[0]))
        self.assertTrue(math.isnan(float(change.delta[0])))

    def test_lineage_alignment_ignores_new_unmerged_drop(self):
        change = lineage_aligned_spatial_change(
            before_fruit_ids=torch.tensor([[10, 0]]),
            before_levels=torch.tensor([[7, 0]]),
            before_active=torch.tensor([[True, False]]),
            before_spatial_score=torch.tensor([[0.5, 0.0]]),
            after_fruit_ids=torch.tensor([[10, 99]]),
            after_active=torch.tensor([[True, True]]),
            after_spatial_score=torch.tensor([[0.5, 1.0]]),
            merge_count=torch.tensor([0]),
            merge_source_ids=torch.zeros((1, 1, 2), dtype=torch.int64),
            merge_new_fruit_ids=torch.zeros((1, 1), dtype=torch.int64),
        )
        self.assertTrue(bool(change.valid[0]))
        self.assertAlmostEqual(float(change.delta[0]), 0.0, places=6)

    def test_delta_marks_episode_boundary_invalid(self):
        current = torch.tensor([12.0, 5.0])
        previous = torch.tensor([10.0, 9.0])
        valid = torch.tensor([True, False])
        delta, delta_valid = scene_mergeability_delta(current, previous, valid)
        self.assertEqual(float(delta[0]), 2.0)
        self.assertTrue(math.isnan(float(delta[1])))
        self.assertTrue(torch.equal(delta_valid, valid))

    def test_analysis_writes_raw_and_structural_images(self):
        rows = 120
        delta = torch.linspace(-30.0, 40.0, rows)
        delta[::11] = 0.0
        valid = torch.ones(rows, dtype=torch.bool)
        valid[::30] = False
        delta[~valid] = float('nan')
        values = {
            'environment_id': torch.arange(rows) % 4,
            'episode_id': torch.arange(rows) // 30,
            'episode_seed': torch.arange(rows) // 30 + 100,
            'decision_step': torch.arange(rows) // 4 + 1,
            'episode_drop': torch.arange(rows) % 30 + 1,
            'score': torch.arange(rows) * 3,
            'fruit_count': torch.arange(rows) % 20 + 5,
            'max_level': torch.arange(rows) % 8 + 1,
            'scene_mergeability': 1000.0 + torch.arange(rows) * 2.0,
            'occupied_area': 5000.0 + torch.arange(rows) * 5.0,
            'area_weighted_mean': torch.linspace(0.1, 0.8, rows),
            'delta': delta,
            'delta_valid': valid,
            'spatial_scene_score': torch.linspace(0.2, 0.9, rows),
            'material_mass': torch.linspace(10.0, 100.0, rows),
            'spatial_delta': delta / 100.0,
            'spatial_delta_valid': valid,
            'lineage_coverage': torch.ones(rows),
            'merge_count': torch.arange(rows) % 5,
            'merge_score': torch.arange(rows) % 200,
            'done': torch.arange(rows) % 30 == 29,
        }
        columns = {
            name: values[name].to(dtype=dtype)
            for name, dtype in SCENE_VALUE_DTYPES.items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / 'dataset'
            raw = dataset / 'raw'
            raw.mkdir(parents=True)
            torch.save({
                'format_version': 1,
                'table': 'scene_values',
                'rows': rows,
                'columns': columns,
            }, raw / 'scene_values-000000.pt')
            output = dataset / 'analysis'
            summary = analyze(dataset, output_dir=output, sample_rows=50)
            images = tuple(output.glob('*.png'))
            self.assertEqual(len(images), 6)
            self.assertEqual(summary['total_rows'], rows)
            self.assertEqual(summary['valid_delta_rows'], int(valid.sum()))
            self.assertTrue((output / 'delta_quantiles.csv').is_file())
            self.assertTrue((output / 'stage_summary.csv').is_file())
            self.assertTrue((output / 'spatial_merge_groups.csv').is_file())
            self.assertIn('spatial', summary)


if __name__ == '__main__':
    unittest.main()
