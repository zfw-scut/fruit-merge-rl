import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from daxigua.rl.mergeability_transitions import (
    DEFAULT_NEGATIVE_SEVERITY_BANDS,
    PriorityReservoir,
    SPATIAL_NEGATIVE_SEVERITY_BANDS,
    clone_compact_scene_batch,
    compact_scene_row,
    negative_severity_codes,
    select_compact_scene_rows,
    severity_band_manifest,
)
from daxigua.simulator import SimulatorConfig, TensorVectorSimulator
from tools.render_mergeability_negative_gallery import render


class MergeabilityTransitionTest(unittest.TestCase):
    def test_negative_severity_boundaries(self):
        delta = torch.tensor([
            -1.0,
            -7_147.749,
            -7_147.75,
            -20_346.159,
            -20_346.16,
            -61_597.649,
            -61_597.65,
            0.0,
            1.0,
            float('nan'),
        ], dtype=torch.float64)
        valid = torch.ones_like(delta, dtype=torch.bool)
        codes = negative_severity_codes(delta, valid)
        self.assertEqual(
            codes.tolist(), [0, 0, 1, 1, 2, 2, 3, -1, -1, -1]
        )

    def test_priority_reservoir_retains_highest_priorities(self):
        reservoir = PriorityReservoir(2, 2)
        reservoir.add(0, 0.1, 'low')
        reservoir.add(0, 0.9, 'high')
        reservoir.add(0, 0.4, 'middle')
        reservoir.add(0, 0.2, 'discarded')
        self.assertEqual(reservoir.samples(0), ['high', 'middle'])
        self.assertAlmostEqual(reservoir.minimum_priority(0), 0.4)
        self.assertEqual(reservoir.selected_counts(), (2, 0))

    def test_spatial_negative_severity_boundaries(self):
        delta = torch.tensor([
            -0.001,
            -0.01918909,
            -0.0191891,
            -0.09908729,
            -0.0990873,
            -0.25479979,
            -0.2547998,
            0.0,
        ], dtype=torch.float64)
        valid = torch.ones_like(delta, dtype=torch.bool)
        codes = negative_severity_codes(
            delta, valid, SPATIAL_NEGATIVE_SEVERITY_BANDS
        )
        self.assertEqual(codes.tolist(), [0, 0, 1, 1, 2, 2, 3, -1])

    def test_compact_snapshot_and_renderer(self):
        config = SimulatorConfig.training_fast(
            max_fruits=8,
            action_count=21,
            queue_length=4,
            use_cuda_extension=False,
        )
        simulator = TensorVectorSimulator(1, config=config, device='cpu')
        observation = simulator.reset(seeds=7)
        score = torch.zeros_like(observation.positions[..., 0])
        compact = clone_compact_scene_batch(observation, score)
        selected = select_compact_scene_rows(compact, torch.tensor([0]))
        scene = compact_scene_row(selected, 0)
        self.assertEqual(tuple(scene['positions'].shape), (8, 2))

        samples = []
        deltas = (-100.0, -10_000.0, -30_000.0, -80_000.0)
        for band, delta in zip(DEFAULT_NEGATIVE_SEVERITY_BANDS, deltas):
            samples.append({
                'severity_code': band.code,
                'severity_key': band.key,
                'environment_id': 0,
                'episode_id': band.code,
                'episode_seed': 7 + band.code,
                'episode_drop': 20 + band.code,
                'decision_step': 20 + band.code,
                'action_index': 10,
                'drop_x': 280.0,
                'dropped_level': 1,
                'dropped_fruit_id': 100 + band.code,
                'score_delta': 0,
                'merge_count': 0,
                'terminal': False,
                'before_scene_value': 100_000.0,
                'after_scene_value': 100_000.0 + delta,
                'before_occupied_area': 0.0,
                'after_occupied_area': 0.0,
                'before_weighted_mean': 0.0,
                'after_weighted_mean': 0.0,
                'delta': delta,
                'priority': 0.5,
                'before': scene,
                'after': scene,
                'merge_events': {
                    'source_levels': torch.zeros(4, dtype=torch.int64),
                    'new_levels': torch.zeros(4, dtype=torch.int64),
                    'positions': torch.zeros((4, 2)),
                    'score_deltas': torch.zeros(4, dtype=torch.int64),
                    'source_ids': torch.zeros((4, 2), dtype=torch.int64),
                    'new_fruit_ids': torch.zeros(4, dtype=torch.int64),
                },
            })
        payload = {
            'format': 'daxigua_mergeability_negative_transition_dataset',
            'format_version': 1,
            'manifest': {
                'simulator_config': asdict(config),
                'scene_value_definition': (
                    'sum(mergeability * pi * physics_radius^2)'
                ),
            },
            'severity_bands': severity_band_manifest(),
            'samples': samples,
        }
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / 'samples.pt'
            output = Path(temporary) / 'gallery'
            torch.save(payload, dataset)
            report = render(dataset, output, dpi=50, progress_interval=4)
            self.assertEqual(report['image_count'], 4)
            self.assertTrue((output / 'overview.png').is_file())
            self.assertTrue((output / 'index.csv').is_file())
            self.assertTrue((output / 'severity_summary.csv').is_file())
            self.assertTrue((output / 'report.json').is_file())

            spatial_deltas = (-0.01, -0.05, -0.15, -0.30)
            for sample, delta in zip(samples, spatial_deltas):
                sample['metric_kind'] = 'spatial'
                sample['delta'] = delta
                sample['spatial_delta'] = delta
                sample['spatial_delta_valid'] = True
                sample['lineage_coverage'] = 1.0
                sample['before_scene_value'] = 0.8
                sample['after_scene_value'] = 0.8 + delta
                sample['before_current_spatial_score'] = 0.8
                sample['after_current_spatial_score'] = 0.8 + delta
            payload['manifest']['metric_kind'] = 'spatial'
            payload['manifest']['scene_value_definition'] = (
                'lineage-aligned material-mass-weighted spatial score'
            )
            payload['severity_bands'] = severity_band_manifest(
                SPATIAL_NEGATIVE_SEVERITY_BANDS
            )
            spatial_dataset = Path(temporary) / 'spatial_samples.pt'
            spatial_output = Path(temporary) / 'spatial_gallery'
            torch.save(payload, spatial_dataset)
            spatial_report = render(
                spatial_dataset, spatial_output, dpi=50, progress_interval=4
            )
            self.assertEqual(spatial_report['metric_kind'], 'spatial')
            self.assertEqual(spatial_report['image_count'], 4)
            self.assertEqual(len(spatial_report['severity_summary']), 4)
            self.assertTrue((spatial_output / 'overview.png').is_file())


if __name__ == '__main__':
    unittest.main()
