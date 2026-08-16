from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest

import torch

from daxigua.rl.merge_distance import (
    MergeDistanceConfig,
    MergeDistancePredictor,
    cumulative_merge_probabilities,
    merge_distance_loss,
    merge_distance_targets,
)
from daxigua.rl.merge_distance_data import (
    LABELED_SCENE_DTYPES,
    SCENE_DTYPES,
    SceneSnapshotSampler,
    extract_scene_rows,
    label_scene_dataset,
    load_labeled_scene_shard,
)
from daxigua.rl.merge_potential_stats import (
    END_NATURAL,
    EPISODE_DTYPES,
    MERGE_SOURCE_DTYPES,
    OUTCOME_MERGED,
    OUTCOME_TERMINAL_UNMERGED,
    ShardedTensorWriter,
)
from daxigua.rl.observations import TensorState
from daxigua.simulator import SimulatorConfig, TensorVectorSimulator
from tools.train_merge_distance_predictor import (
    load_predictor_checkpoint,
    train,
)


def _small_config():
    return MergeDistanceConfig(
        hidden_dim=32,
        edge_hidden_dim=24,
        message_layers=1,
        queue_hidden_dim=16,
        level_embedding_dim=8,
        max_neighbors=4,
        nearest_neighbors=2,
        motion_neighbors=1,
        vertical_neighbors_per_direction=1,
        horizons=(1, 2, 4, 8),
    )


def _scene_columns(episode_id, observed_drop, fruit_ids, levels, active):
    rows = len(episode_id)
    provided_fruits = len(fruit_ids[0])
    fruits = 64
    positions = torch.zeros(rows, fruits, 2)
    positions[..., 0] = torch.arange(fruits).float() * 40.0 + 80.0
    positions[..., 1] = 900.0
    padded_ids = torch.zeros(rows, fruits, dtype=torch.int64)
    padded_levels = torch.zeros(rows, fruits, dtype=torch.int64)
    padded_active = torch.zeros(rows, fruits, dtype=torch.bool)
    padded_ids[:, :provided_fruits] = torch.tensor(fruit_ids)
    padded_levels[:, :provided_fruits] = torch.tensor(levels)
    padded_active[:, :provided_fruits] = torch.tensor(active)
    result = {
        'episode_id': torch.tensor(episode_id),
        'observed_drop': torch.tensor(observed_drop),
        'positions': positions,
        'velocities': torch.zeros(rows, fruits, 2),
        'angular_velocities': torch.zeros(rows, fruits),
        'levels': padded_levels,
        'physics_radii': torch.ones(rows, fruits) * 20.0,
        'age_frames': torch.zeros(rows, fruits, dtype=torch.int64),
        'active': padded_active,
        'fruit_ids': padded_ids,
        'fruit_queue': torch.ones(rows, 4, dtype=torch.int64),
        'danger_progress': torch.zeros(rows),
        'over_danger_line': torch.zeros(rows, dtype=torch.bool),
    }
    return {
        name: result[name].to(dtype=dtype)
        for name, dtype in SCENE_DTYPES.items()
    }


class MergeDistanceModelTests(unittest.TestCase):
    def test_targets_keep_time_tail_and_terminal_separate(self):
        outcomes = torch.tensor((OUTCOME_MERGED,) * 6 + (
            OUTCOME_TERMINAL_UNMERGED,
        ))
        t_merge = torch.tensor((1, 2, 3, 8, 9, 100, -1))
        targets, valid = merge_distance_targets(
            outcomes, t_merge, (1, 2, 4, 8)
        )
        self.assertEqual(targets.tolist(), [0, 1, 2, 3, 4, 4, 5])
        self.assertTrue(bool(valid.all()))

    def test_predictor_outputs_per_fruit_distribution_and_backpropagates(self):
        simulator = TensorVectorSimulator(
            2,
            config=SimulatorConfig(
                max_fruits=64, use_cuda_extension=False
            ),
            device='cpu',
        )
        simulator.active[:, :2] = True
        simulator.levels[:, :2] = torch.tensor((2, 2))
        simulator.physics_radii[:, :2] = 30.0
        simulator.positions[:, 0] = torch.tensor((160.0, 900.0))
        simulator.positions[:, 1] = torch.tensor((220.0, 900.0))
        state = TensorState.from_observation(
            simulator.observe(), physics_fps=30
        )
        model = MergeDistancePredictor(_small_config())
        output = model(state)
        self.assertEqual(output.logits.shape, (2, 64, 6))
        self.assertTrue(torch.allclose(
            output.probabilities[:, :2].sum(dim=-1),
            torch.ones(2, 2),
        ))
        outcomes = torch.full((2, 64), 2, dtype=torch.int8)
        outcomes[:, 0] = OUTCOME_MERGED
        outcomes[:, 1] = OUTCOME_TERMINAL_UNMERGED
        t_merge = torch.full((2, 64), -1, dtype=torch.int32)
        t_merge[:, 0] = 3
        loss = merge_distance_loss(
            output.logits,
            state.active,
            outcomes,
            t_merge,
            model.config.horizons,
        )
        loss.backward()
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(model.output_head.weight.grad)
        cdf = cumulative_merge_probabilities(
            output.probabilities, model.config.horizons
        )
        self.assertTrue(bool((cdf[..., 1:] >= cdf[..., :-1]).all()))


class MergeDistanceDataTests(unittest.TestCase):
    def test_scene_sampler_keeps_first_nonempty_and_stride(self):
        simulator = TensorVectorSimulator(
            1,
            config=SimulatorConfig(max_fruits=64, use_cuda_extension=False),
            device='cpu',
        )
        sampler = SceneSnapshotSampler(
            1, device='cpu', scene_stride=3, max_scenes_per_episode=2
        )
        empty = simulator.observe()
        self.assertFalse(bool(sampler.select(empty)[0]))
        simulator.active[0, 0] = True
        simulator.fruit_ids[0, 0] = 10
        simulator.levels[0, 0] = 1
        simulator.physics_radii[0, 0] = 20.0
        simulator.step_count[0] = 1
        observation = simulator.observe()
        self.assertTrue(bool(sampler.select(observation)[0]))
        simulator.step_count[0] = 2
        self.assertFalse(bool(sampler.select(simulator.observe())[0]))
        simulator.step_count[0] = 4
        selected = sampler.select(simulator.observe())
        self.assertTrue(bool(selected[0]))
        rows = extract_scene_rows(
            simulator.observe(), selected, torch.tensor((7,))
        )
        self.assertEqual(rows['positions'].shape, (1, 64, 2))
        self.assertEqual(int(rows['fruit_ids'][0, 0]), 10)
        simulator.step_count[0] = 7
        self.assertFalse(bool(sampler.select(simulator.observe())[0]))

    def test_labeler_matches_each_scene_slot_to_future_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            scenes = ShardedTensorWriter(
                raw, 'scenes', SCENE_DTYPES,
                shard_rows=10, background=False,
            )
            scenes.append(_scene_columns(
                episode_id=(0, 0),
                observed_drop=(1, 3),
                fruit_ids=((10, 20), (10, 20)),
                levels=((2, 3), (2, 3)),
                active=((True, True), (True, True)),
            ))
            scenes.close()
            merges = ShardedTensorWriter(
                raw, 'merge_sources', MERGE_SOURCE_DTYPES,
                shard_rows=10, background=False,
            )
            merges.append({
                'episode_id': torch.tensor((0,), dtype=torch.int64),
                'fruit_id': torch.tensor((10,), dtype=torch.int64),
                'merge_drop': torch.tensor((5,), dtype=torch.int32),
                'source_level': torch.tensor((2,), dtype=torch.int8),
                'new_level': torch.tensor((3,), dtype=torch.int8),
                'new_fruit_id': torch.tensor((30,), dtype=torch.int64),
                'event_index': torch.tensor((0,), dtype=torch.int16),
                'source_index': torch.tensor((0,), dtype=torch.int8),
                'x': torch.tensor((100.0,)),
                'y': torch.tensor((800.0,)),
                'score_delta': torch.tensor((3,), dtype=torch.int32),
            })
            merges.close()
            episodes = ShardedTensorWriter(
                raw, 'episodes', EPISODE_DTYPES,
                shard_rows=10, background=False,
            )
            episodes.append({
                'episode_id': torch.tensor((0,), dtype=torch.int64),
                'seed': torch.tensor((123,), dtype=torch.int64),
                'end_drop': torch.tensor((10,), dtype=torch.int32),
                'score': torch.tensor((3,), dtype=torch.int32),
                'max_level': torch.tensor((3,), dtype=torch.int8),
                'final_fruit_count': torch.tensor((1,), dtype=torch.int16),
                'end_kind': torch.tensor((END_NATURAL,), dtype=torch.int8),
            })
            episodes.close()
            manifest = label_scene_dataset(
                root, horizons=(1, 2, 4, 8), shard_rows=10
            )
            labeled_path = next(
                root.joinpath('predictor', 'labeled').glob('*.pt')
            )
            labeled = load_labeled_scene_shard(labeled_path)

        self.assertEqual(manifest['scene_rows'], 2)
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['completed_scene_shards'], 1)
        self.assertEqual(labeled['t_merge'][:, 0].tolist(), [4, 2])
        self.assertEqual(
            labeled['outcome'][:, 0].tolist(),
            [OUTCOME_MERGED, OUTCOME_MERGED],
        )
        self.assertEqual(
            labeled['outcome'][:, 1].tolist(),
            [OUTCOME_TERMINAL_UNMERGED, OUTCOME_TERMINAL_UNMERGED],
        )
        self.assertEqual(labeled['fruit_snapshot_count'][:, 0].tolist(), [2, 2])
        self.assertTrue(torch.allclose(
            labeled['fruit_weight'][:, 0], torch.tensor((0.5, 0.5))
        ))


class MergeDistanceTrainingTests(unittest.TestCase):
    def test_tiny_sharded_dataset_trains_and_saves_frozen_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'dataset'
            rows = 20
            episode_ids = torch.arange(rows, dtype=torch.int64)
            base = _scene_columns(
                episode_id=episode_ids.tolist(),
                observed_drop=[1] * rows,
                fruit_ids=[[100 + index, 200 + index] for index in range(rows)],
                levels=[[2, 9] for _ in range(rows)],
                active=[[True, True] for _ in range(rows)],
            )
            outcomes = torch.full(
                (rows, 64), 2, dtype=torch.int8
            )
            outcomes[:, 0] = OUTCOME_MERGED
            outcomes[:, 1] = OUTCOME_TERMINAL_UNMERGED
            t_merge = torch.full((rows, 64), -1, dtype=torch.int32)
            t_merge[:, 0] = 3
            counts = torch.zeros(rows, 64, dtype=torch.int32)
            counts[:, :2] = 1
            fruit_weight = torch.zeros(rows, 64)
            fruit_weight[:, :2] = 1.0
            labeled = dict(base)
            labeled.update({
                'outcome': outcomes,
                't_merge': t_merge,
                'fruit_snapshot_count': counts,
                'fruit_weight': fruit_weight,
                'episode_end_drop': torch.full(
                    (rows,), 20, dtype=torch.int32
                ),
                'episode_end_kind': torch.full(
                    (rows,), END_NATURAL, dtype=torch.int8
                ),
            })
            writer = ShardedTensorWriter(
                dataset / 'labeled',
                'scenes_labeled',
                LABELED_SCENE_DTYPES,
                shard_rows=64,
                background=False,
            )
            writer.append(labeled)
            writer.close()
            class_counts = torch.zeros(12, 6, dtype=torch.int64)
            class_counts[2, 2] = 10
            class_counts[9, 5] = 10
            manifest = {
                'format_version': 1,
                'purpose': 'merge_distance_predictor_dataset',
                'source_dataset': 'synthetic',
                'source_identity': {
                    'physics_fps': 30,
                    'simulator_config': {
                        'board_width': 560,
                        'board_height': 1120,
                        'spawn_y': 252,
                        'wall_width': 20,
                        'gravity_y': 1800.0,
                    },
                },
                'horizons': [1, 2, 4, 8],
                'class_counts_by_level': {
                    'train': class_counts.tolist(),
                    'validation': class_counts.tolist(),
                    'test': class_counts.tolist(),
                },
            }
            (dataset / 'dataset_manifest.json').write_text(
                json.dumps(manifest), encoding='utf-8'
            )
            output = root / 'run'
            args = SimpleNamespace(
                command='train', dataset_dir=dataset, output_dir=output,
                device='cpu', epochs=1, batch_size=4,
                learning_rate=3e-4, weight_decay=1e-5,
                grad_clip_norm=5.0, hidden_dim=16, edge_hidden_dim=16,
                message_layers=1, queue_hidden_dim=8,
                level_embedding_dim=4, balance_power=0.5,
                max_balance_weight=8.0, seed=123, num_workers=0,
                max_train_batches=1, max_eval_batches=1,
                early_stopping_patience=1, compile_model=False,
                compile_mode='reduce-overhead', autocast_bfloat16=False,
                telemetry_interval_seconds=10.0,
            )
            result = train(args)
            checkpoint = output / 'checkpoints' / 'final.pt'
            model, payload = load_predictor_checkpoint(checkpoint, 'cpu')

        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['phase'], 'completed')
        self.assertEqual(payload['purpose'], 'merge_distance_predictor')
        self.assertEqual(model.config.horizons, (1, 2, 4, 8))


if __name__ == '__main__':
    unittest.main()
