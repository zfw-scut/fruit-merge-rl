import csv
from pathlib import Path
import tempfile
import unittest

import torch

from daxigua.rl.merge_potential_stats import (
    END_DROP_LIMIT,
    END_NATURAL,
    EPISODE_DTYPES,
    FruitSnapshotSampler,
    MERGE_SOURCE_DTYPES,
    OUTCOME_CENSORED,
    OUTCOME_MERGED,
    OUTCOME_TERMINAL_UNMERGED,
    SAMPLE_FIRST_OBSERVATION,
    SAMPLE_PERIODIC,
    SNAPSHOT_DTYPES,
    ShardedTensorWriter,
    extract_merge_sources,
    extract_snapshot_features,
    summarize_dataset,
)
from daxigua.simulator import SimulatorConfig
from daxigua.simulator.types import BatchMergeEvents, BatchObservation


def _observation(
        *,
        positions,
        levels,
        fruit_ids,
        active,
        step_count,
        radii=None,
        empty_space_ratio=None):
    positions = torch.tensor(positions, dtype=torch.float32)
    levels = torch.tensor(levels, dtype=torch.int64)
    fruit_ids = torch.tensor(fruit_ids, dtype=torch.int64)
    active = torch.tensor(active, dtype=torch.bool)
    batch, fruits = levels.shape
    if radii is None:
        radii = torch.ones(batch, fruits, dtype=torch.float32) * 10.0
    else:
        radii = torch.tensor(radii, dtype=torch.float32)
    step_count = torch.tensor(step_count, dtype=torch.int64)
    fruit_count = active.sum(dim=1).to(torch.int64)
    max_level = torch.where(active, levels, torch.zeros_like(levels)).amax(1)
    if empty_space_ratio is None:
        empty_space_ratio = torch.full((batch,), 0.75)
    else:
        empty_space_ratio = torch.tensor(
            empty_space_ratio, dtype=torch.float32
        )
    float_fruits = torch.zeros(batch, fruits, dtype=torch.float32)
    int_fruits = torch.zeros(batch, fruits, dtype=torch.int64)
    return BatchObservation(
        positions=positions,
        velocities=torch.zeros(batch, fruits, 2),
        angles=float_fruits.clone(),
        angular_velocities=float_fruits.clone(),
        levels=levels,
        physics_radii=radii,
        fruit_ids=fruit_ids,
        age_frames=int_fruits.clone(),
        active=active,
        fruit_queue=torch.ones(batch, 4, dtype=torch.int64),
        score=torch.zeros(batch, dtype=torch.int64),
        last_score=torch.zeros(batch, dtype=torch.int64),
        step_count=step_count,
        physics_frame=torch.zeros(batch, dtype=torch.int64),
        done=torch.zeros(batch, dtype=torch.bool),
        fruit_count=fruit_count,
        max_level=max_level,
        max_height=torch.zeros(batch),
        empty_space_ratio=empty_space_ratio,
        danger_progress=torch.zeros(batch),
        over_danger_line=torch.zeros(batch, dtype=torch.bool),
    )


def _columns(dtypes, rows):
    result = {}
    for name, dtype in dtypes.items():
        values = rows[name]
        result[name] = torch.as_tensor(values, dtype=dtype)
    return result


class MergePotentialSamplingTests(unittest.TestCase):
    def test_sampler_keeps_first_and_caps_periodic_snapshots(self):
        observation = _observation(
            positions=[[[100, 800], [0, 0]]],
            levels=[[1, 0]],
            fruit_ids=[[10, 0]],
            active=[[True, False]],
            step_count=[1],
        )
        sampler = FruitSnapshotSampler(
            1, 2, device='cpu', snapshot_stride=2,
            max_snapshots_per_fruit=2, snapshots_per_scale=1,
        )
        first = sampler.select(observation)
        self.assertEqual(int(first[0, 0]), SAMPLE_FIRST_OBSERVATION)

        observation.step_count[0] = 3
        periodic = sampler.select(observation)
        self.assertEqual(int(periodic[0, 0]), SAMPLE_PERIODIC)
        observation.step_count[0] = 5
        capped = sampler.select(observation)
        self.assertEqual(int(capped[0, 0]), 0)

        observation.fruit_ids[0, 0] = 11
        replaced = sampler.select(observation)
        self.assertEqual(int(replaced[0, 0]), SAMPLE_FIRST_OBSERVATION)

    def test_snapshot_features_measure_nearest_same_level_peer(self):
        observation = _observation(
            positions=[[[100, 800], [140, 800], [300, 700]]],
            levels=[[2, 2, 3]],
            fruit_ids=[[1, 2, 3]],
            active=[[True, True, True]],
            step_count=[7],
            radii=[[10, 10, 15]],
            empty_space_ratio=[0.7],
        )
        kinds = torch.tensor([[
            SAMPLE_FIRST_OBSERVATION,
            SAMPLE_PERIODIC,
            SAMPLE_PERIODIC,
        ]], dtype=torch.uint8)
        result = extract_snapshot_features(
            observation,
            kinds,
            torch.tensor([5]),
            SimulatorConfig(use_cuda_extension=False),
            env_chunk_size=1,
        )
        self.assertEqual(result['fruit_id'].tolist(), [1, 2, 3])
        self.assertEqual(result['same_level_peer_count'].tolist(), [1, 1, 0])
        self.assertTrue(result['has_same_level_peer'][:2].all().item())
        self.assertFalse(bool(result['has_same_level_peer'][2]))
        self.assertAlmostEqual(
            float(result['nearest_same_level_center_distance'][0]), 40.0
        )
        self.assertAlmostEqual(
            float(result['nearest_same_level_surface_gap'][0]), 20.0
        )
        self.assertTrue(torch.isnan(
            result['nearest_same_level_center_distance'][2]
        ))
        self.assertAlmostEqual(
            float(result['scene_occupancy_ratio'][0]), 0.3, places=6
        )

    def test_sampler_expands_interval_for_long_lived_fruit(self):
        observation = _observation(
            positions=[[[100, 800]]],
            levels=[[1]],
            fruit_ids=[[10]],
            active=[[True]],
            step_count=[1],
        )
        sampler = FruitSnapshotSampler(
            1, 1, device='cpu', snapshot_stride=2,
            max_snapshots_per_fruit=4, snapshots_per_scale=1,
        )
        self.assertEqual(
            int(sampler.select(observation)[0, 0]),
            SAMPLE_FIRST_OBSERVATION,
        )
        observation.step_count[0] = 3
        self.assertEqual(
            int(sampler.select(observation)[0, 0]), SAMPLE_PERIODIC
        )
        observation.step_count[0] = 5
        self.assertEqual(
            int(sampler.select(observation)[0, 0]), SAMPLE_PERIODIC
        )
        observation.step_count[0] = 7
        self.assertEqual(int(sampler.select(observation)[0, 0]), 0)
        observation.step_count[0] = 9
        self.assertEqual(
            int(sampler.select(observation)[0, 0]), SAMPLE_PERIODIC
        )

    def test_merge_events_expand_to_two_source_rows(self):
        events = BatchMergeEvents(
            count=torch.tensor([1]),
            source_levels=torch.tensor([[4, 0]]),
            new_levels=torch.tensor([[5, 0]]),
            positions=torch.tensor([[[120.0, 700.0], [0.0, 0.0]]]),
            score_deltas=torch.tensor([[20, 0]]),
            source_ids=torch.tensor([[[8, 9], [0, 0]]]),
            new_fruit_ids=torch.tensor([[10, 0]]),
        )
        result = extract_merge_sources(
            events, torch.tensor([3]), torch.tensor([12])
        )
        self.assertEqual(result['episode_id'].tolist(), [3, 3])
        self.assertEqual(result['fruit_id'].tolist(), [8, 9])
        self.assertEqual(result['source_index'].tolist(), [0, 1])
        self.assertEqual(result['merge_drop'].tolist(), [12, 12])


class MergePotentialSummaryTests(unittest.TestCase):
    def test_summary_labels_merged_terminal_and_censored_fruits(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            raw = dataset / 'raw'
            snapshots = ShardedTensorWriter(
                raw, 'snapshots', SNAPSHOT_DTYPES,
                shard_rows=100, background=False,
            )
            snapshot_rows = {
                name: [0, 0, 0, 0] for name in SNAPSHOT_DTYPES
            }
            snapshot_rows.update({
                'episode_id': [0, 0, 0, 1],
                'fruit_id': [1, 1, 2, 1],
                'observed_drop': [1, 3, 2, 1],
                'sample_kind': [
                    SAMPLE_FIRST_OBSERVATION,
                    SAMPLE_PERIODIC,
                    SAMPLE_FIRST_OBSERVATION,
                    SAMPLE_FIRST_OBSERVATION,
                ],
                'slot': [0, 0, 1, 0],
                'level': [1, 1, 2, 1],
                'x': [100.0, 105.0, 200.0, 100.0],
                'y': [800.0, 790.0, 700.0, 800.0],
                'radius': [10.0, 10.0, 15.0, 10.0],
                'scene_occupancy_ratio': [0.2, 0.3, 0.4, 0.2],
                'center_height_normalized': [0.2, 0.3, 0.4, 0.2],
                'top_height_normalized': [0.21, 0.31, 0.42, 0.21],
                'same_level_peer_count': [1, 1, 0, 0],
                'has_same_level_peer': [True, True, False, False],
                'nearest_same_level_center_distance': [40.0, 35.0, float('nan'), float('nan')],
                'nearest_same_level_surface_gap': [20.0, 15.0, float('nan'), float('nan')],
                'nearest_same_level_center_distance_normalized': [0.1, 0.08, float('nan'), float('nan')],
                'nearest_same_level_surface_gap_normalized': [0.05, 0.03, float('nan'), float('nan')],
            })
            snapshots.append(_columns(SNAPSHOT_DTYPES, snapshot_rows))
            snapshots.close()

            merges = ShardedTensorWriter(
                raw, 'merge_sources', MERGE_SOURCE_DTYPES,
                shard_rows=100, background=False,
            )
            merge_rows = {
                name: [0] for name in MERGE_SOURCE_DTYPES
            }
            merge_rows.update({
                'episode_id': [0],
                'fruit_id': [1],
                'merge_drop': [5],
                'source_level': [1],
                'new_level': [2],
                'new_fruit_id': [3],
            })
            merges.append(_columns(MERGE_SOURCE_DTYPES, merge_rows))
            merges.close()

            episodes = ShardedTensorWriter(
                raw, 'episodes', EPISODE_DTYPES,
                shard_rows=100, background=False,
            )
            episodes.append(_columns(EPISODE_DTYPES, {
                'episode_id': [0, 1],
                'seed': [100, 101],
                'end_drop': [10, 3],
                'score': [1000, 100],
                'max_level': [4, 2],
                'final_fruit_count': [10, 4],
                'end_kind': [END_NATURAL, END_DROP_LIMIT],
            }))
            episodes.close()

            result = summarize_dataset(
                dataset, horizons=(2, 4), factor_bins=4,
                labeled_shard_rows=100,
            )
            self.assertEqual(result['episodes'], 2)
            self.assertEqual(result['unique_observed_fruits'], 3)
            labeled_path = next((dataset / 'analysis' / 'labeled').glob('*.pt'))
            labeled = torch.load(
                labeled_path, map_location='cpu', weights_only=False
            )['columns']
            self.assertEqual(labeled['t_merge'].tolist(), [4, 2, -1, -1])
            self.assertEqual(labeled['outcome'].tolist(), [
                OUTCOME_MERGED,
                OUTCOME_MERGED,
                OUTCOME_TERMINAL_UNMERGED,
                OUTCOME_CENSORED,
            ])
            self.assertEqual(
                labeled['fruit_snapshot_count'].tolist(), [2, 2, 1, 1]
            )
            self.assertEqual(
                labeled['fruit_weight'].tolist(), [0.5, 0.5, 1.0, 1.0]
            )

            with (dataset / 'analysis' / 'lifecycle_by_level.csv').open(
                    encoding='utf-8-sig', newline='') as handle:
                rows = {int(row['level']): row for row in csv.DictReader(handle)}
            self.assertEqual(int(rows[1]['fruits']), 2)
            self.assertEqual(int(rows[1]['merged_fruits']), 1)
            self.assertEqual(int(rows[1]['censored_fruits']), 1)
            self.assertEqual(int(rows[2]['terminal_unmerged_fruits']), 1)

            with (
                dataset / 'analysis' / 'horizon_probabilities_by_level.csv'
            ).open(encoding='utf-8-sig', newline='') as handle:
                horizon_rows = {
                    (int(row['level']), int(row['horizon_drops'])): row
                    for row in csv.DictReader(handle)
                }
            self.assertAlmostEqual(
                float(horizon_rows[(1, 2)]['snapshot_probability']), 0.25
            )
            self.assertTrue((
                dataset / 'analysis' / 'factor_interactions_by_level.csv'
            ).is_file())
            self.assertTrue((
                dataset / 'analysis' / 'analysis_manifest.json'
            ).is_file())


if __name__ == '__main__':
    unittest.main()
