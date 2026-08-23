"""轻量水果对堵塞风险数据和模型测试。"""

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from daxigua.rl.merge_potential_stats import ShardedTensorWriter
from daxigua.rl.pair_risk import (
    END_NATURAL,
    EPISODE_DTYPES,
    EVENT_CONFIRMED,
    EVENT_DTYPES,
    EVENT_ENDED,
    EXPOSURE_DTYPES,
    LABELED_DTYPES,
    PairRiskModel,
    PairRiskModelConfig,
    extract_pair_exposures,
    finalize_pair_risk_dataset,
    risk_metrics,
    stratified_episode_splits,
)
from tools.train_pair_risk import (
    balanced_sample_weights,
    build_training_balance,
    train,
    training_balance_to,
)
from tools.benchmark_pair_risk_collection import _gpu_summary
from tools.render_pair_risk_gallery import (
    confusion_kind,
    select_unique_candidates,
    timeline_target_specs,
    warning_time_bucket,
)


def _model_columns(batch=3, fruits=6):
    positions = torch.zeros(batch, fruits, 2)
    positions[:, 0] = torch.tensor([120.0, 900.0])
    positions[:, 1] = torch.tensor([360.0, 900.0])
    positions[:, 2] = torch.tensor([240.0, 820.0])
    positions[:, 3] = torch.tensor([80.0, 760.0])
    levels = torch.zeros(batch, fruits, dtype=torch.int64)
    levels[:, 0:2] = 8
    levels[:, 2] = 4
    levels[:, 3] = 6
    radii = torch.zeros(batch, fruits)
    radii[:, 0:2] = 100.0
    radii[:, 2] = 40.0
    radii[:, 3] = 60.0
    active = levels.gt(0)
    return {
        'positions': positions,
        'levels': levels,
        'physics_radii': radii,
        'age_frames': torch.full((batch, fruits), 300, dtype=torch.int64),
        'active': active,
        'fruit_queue': torch.tensor([[1, 2, 3, 4]]).expand(batch, -1),
        'danger_progress': torch.linspace(0.1, 0.3, batch),
        'over_danger_line': torch.zeros(batch, dtype=torch.bool),
        'pair_slot_i': torch.zeros(batch, dtype=torch.int64),
        'pair_slot_j': torch.ones(batch, dtype=torch.int64),
    }


class PairRiskModelTests(unittest.TestCase):
    def test_model_is_pair_swap_invariant_and_differentiable(self):
        torch.manual_seed(7)
        model = PairRiskModel(PairRiskModelConfig(max_fruits=6))
        columns = _model_columns()
        first = model(columns)
        swapped = dict(columns)
        swapped['pair_slot_i'] = columns['pair_slot_j']
        swapped['pair_slot_j'] = columns['pair_slot_i']
        second = model(swapped)
        self.assertEqual(tuple(first.shape), (3,))
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
        first.sum().backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
        ))

    def test_exposure_extraction_keeps_only_target_levels_and_stride(self):
        observation = SimpleNamespace(**{
            **_model_columns(batch=2),
            'velocities': torch.zeros(2, 6, 2),
            'angular_velocities': torch.zeros(2, 6),
            'fruit_ids': torch.tensor([
                [10, 11, 12, 13, 0, 0],
                [20, 21, 22, 23, 0, 0],
            ]),
            'step_count': torch.tensor([4, 5]),
        })
        pair = torch.triu_indices(6, 6, offset=1)
        exposures = extract_pair_exposures(
            observation,
            torch.tensor([2, 3]),
            pair[0],
            pair[1],
            exposure_stride=4,
        )
        self.assertEqual(int(exposures['episode_id'].numel()), 1)
        self.assertEqual(int(exposures['level'][0]), 8)
        self.assertEqual(int(exposures['fruit_id_i'][0]), 10)
        self.assertEqual(int(exposures['fruit_id_j'][0]), 11)

    def test_risk_metrics_reports_rare_positive_quality(self):
        result = risk_metrics(
            torch.tensor([5.0, 2.0, -2.0, -5.0]),
            torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        )
        self.assertEqual(result['samples'], 4)
        self.assertAlmostEqual(result['recall'], 1.0)
        self.assertAlmostEqual(result['average_precision'], 1.0)


class PairRiskLabelTests(unittest.TestCase):
    def _exposure_columns(self):
        metadata = [
            (2, 4, 10, 11, 8),
            (2, 12, 10, 11, 8),
            (2, 36, 10, 11, 8),
            (2, 40, 10, 11, 8),
            (2, 4, 20, 21, 9),
            (2, 90, 20, 21, 9),
        ]
        rows = len(metadata)
        state = _model_columns(rows)
        result = {
            'episode_id': torch.tensor([row[0] for row in metadata]),
            'step': torch.tensor([row[1] for row in metadata]),
            'pair_slot_i': torch.zeros(rows, dtype=torch.int16),
            'pair_slot_j': torch.ones(rows, dtype=torch.int16),
            'fruit_id_i': torch.tensor([row[2] for row in metadata]),
            'fruit_id_j': torch.tensor([row[3] for row in metadata]),
            'level': torch.tensor([row[4] for row in metadata]),
            'positions': state['positions'],
            'velocities': torch.zeros(rows, 6, 2),
            'angular_velocities': torch.zeros(rows, 6),
            'levels': state['levels'],
            'fruit_ids': torch.arange(rows * 6).reshape(rows, 6) + 1,
            'physics_radii': state['physics_radii'],
            'age_frames': state['age_frames'],
            'active': state['active'],
            'fruit_queue': state['fruit_queue'],
            'danger_progress': state['danger_progress'],
            'over_danger_line': state['over_danger_line'],
        }
        return result

    def test_finalize_uses_future_event_and_censors_unknown_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            exposure_writer = ShardedTensorWriter(
                raw, 'pair_risk_exposures', EXPOSURE_DTYPES,
                shard_rows=100, background=False,
            )
            event_writer = ShardedTensorWriter(
                raw, 'pair_risk_events', EVENT_DTYPES,
                shard_rows=100, background=False,
            )
            episode_writer = ShardedTensorWriter(
                raw, 'pair_risk_episodes', EPISODE_DTYPES,
                shard_rows=100, background=False,
            )
            exposure_writer.append(self._exposure_columns())
            event_writer.append({
                'episode_id': torch.tensor([2, 2]),
                'event_kind': torch.tensor([
                    EVENT_CONFIRMED, EVENT_ENDED
                ]),
                'onset_step': torch.tensor([12, 12]),
                'event_step': torch.tensor([36, 50]),
                'fruit_id_i': torch.tensor([10, 10]),
                'fruit_id_j': torch.tensor([11, 11]),
                'level': torch.tensor([8, 8]),
            })
            episode_writer.append({
                'episode_id': torch.tensor([2]),
                'seed': torch.tensor([123]),
                'end_step': torch.tensor([100]),
                'end_kind': torch.tensor([END_NATURAL]),
            })
            exposure_writer.close()
            event_writer.close()
            episode_writer.close()

            result = finalize_pair_risk_dataset(
                root,
                forecast_horizon=8,
                confirmation_drops=24,
                shard_rows=100,
            )
            self.assertEqual(result['confirmed_events'], 1)
            self.assertEqual(result['labeled_rows'], 4)
            self.assertEqual(result['censored_rows'], 1)
            self.assertEqual(result['post_confirmation_rows_skipped'], 1)
            shard = torch.load(
                next((root / 'labeled').glob('pair_risk_samples-*.pt')),
                weights_only=False,
            )['columns']
            self.assertEqual(int(shard['label'].sum()), 3)
            self.assertEqual(int((~shard['label']).sum()), 1)
            self.assertTrue(shard['split'].eq(0).all())
            self.assertEqual(shard['lead_to_onset'][:3].tolist(), [8, 0, -24])
            manifest = json.loads(
                (root / 'labeled' / 'manifest.json').read_text('utf-8')
            )
            self.assertEqual(manifest['counts']['train']['8']['positive'], 3)
            self.assertEqual(manifest['confirmed_events_by_level']['8'], 1)
            self.assertEqual(manifest['episode_split_counts']['train'], 1)

    def test_event_episode_split_balances_rare_level_without_leakage(self):
        episodes = tuple(range(20))
        events = {
            (episode, episode * 2 + 1, episode * 2 + 2): [{
                'level': 11,
                'onset': 10,
                'confirmed': 34,
                'end': 50,
                'event_id': episode,
            }]
            for episode in range(10)
        }
        splits = stratified_episode_splits(episodes, events)
        event_splits = [splits[episode] for episode in range(10)]
        self.assertEqual(event_splits.count(0), 8)
        self.assertEqual(event_splits.count(1), 1)
        self.assertEqual(event_splits.count(2), 1)
        self.assertEqual(len(splits), len(episodes))


class PairRiskBalanceTests(unittest.TestCase):
    def test_level_label_and_event_weights_have_equal_total_mass(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pair_risk_samples-00000.pt'
            levels = torch.tensor(
                [7] * 7 + [8] * 6, dtype=torch.int8
            )
            labels = torch.tensor(
                [0, 0, 0, 0, 1, 1, 1,
                 0, 0, 1, 1, 1, 1],
                dtype=torch.bool,
            )
            event_ids = torch.tensor(
                [-1, -1, -1, -1, 0, 0, 1,
                 -1, -1, 2, 3, 3, 3],
                dtype=torch.int64,
            )
            columns = {
                'split': torch.zeros(13, dtype=torch.int8),
                'level': levels,
                'label': labels,
                'event_id': event_ids,
            }
            torch.save({
                'format_version': 1,
                'table': 'pair_risk_samples',
                'columns': columns,
            }, path)
            balance = training_balance_to(
                build_training_balance((path,)), torch.device('cpu')
            )
            weights = balanced_sample_weights(columns, balance)
            for level in (7, 8):
                for label in (False, True):
                    mask = levels.eq(level) & labels.eq(label)
                    self.assertAlmostEqual(
                        float(weights[mask].sum()), 13 / 4, places=5
                    )
            for event_id in range(4):
                mask = event_ids.eq(event_id)
                self.assertAlmostEqual(
                    float(weights[mask].sum()), 13 / 8, places=5
                )

    def test_collection_gpu_summary_uses_continuous_samples(self):
        summary = _gpu_summary([
            {
                'gpu_utilization_percent': 40,
                'memory_used_mib': 1000,
                'power_watts': 120,
            },
            {
                'gpu_utilization_percent': 80,
                'memory_used_mib': 2000,
                'power_watts': 180,
            },
            {
                'gpu_utilization_percent': 100,
                'memory_used_mib': 3000,
                'power_watts': 240,
            },
        ])
        self.assertEqual(summary['samples'], 3)
        self.assertAlmostEqual(
            summary['gpu_utilization_percent']['mean'], 220 / 3
        )
        self.assertEqual(
            summary['memory_used_mib']['median'], 2000
        )

    def test_gallery_classification_and_event_deduplication(self):
        self.assertEqual(confusion_kind(True, True), 'TP')
        self.assertEqual(confusion_kind(True, False), 'FP')
        self.assertEqual(confusion_kind(False, True), 'FN')
        self.assertEqual(confusion_kind(False, False), 'TN')
        candidates = [
            {
                'priority': 0.99,
                'event_id': 3,
                'episode_id': 10,
                'fruit_id_i': 1,
                'fruit_id_j': 2,
            },
            {
                'priority': 0.98,
                'event_id': 3,
                'episode_id': 10,
                'fruit_id_i': 1,
                'fruit_id_j': 2,
            },
            {
                'priority': 0.97,
                'event_id': 4,
                'episode_id': 11,
                'fruit_id_i': 5,
                'fruit_id_j': 6,
            },
        ]
        selected = select_unique_candidates(candidates, 2)
        self.assertEqual([row['event_id'] for row in selected], [3, 4])

    def test_gallery_warning_time_buckets(self):
        self.assertEqual(warning_time_bucket(False, 0), 'no_event')
        self.assertEqual(warning_time_bucket(True, -3), 'at_or_after_onset')
        self.assertEqual(warning_time_bucket(True, 0), 'at_or_after_onset')
        self.assertEqual(warning_time_bucket(True, 4), 'lead_1_4')
        self.assertEqual(warning_time_bucket(True, 12), 'lead_5_12')
        self.assertEqual(warning_time_bucket(True, 24), 'lead_13_24')

    def test_gallery_timeline_targets_follow_label_semantics(self):
        positive = {'step': 100, 'label': True, 'lead_to_onset': 20}
        self.assertEqual(timeline_target_specs(
            positive,
            forecast_horizon=24,
            confirmation_drops=24,
            confirmed_step=144,
        ), [
            {'role': 'prediction', 'target_step': 100},
            {'role': 'onset', 'target_step': 120},
            {'role': 'confirmation', 'target_step': 144},
        ])
        negative = {'step': 200, 'label': False, 'lead_to_onset': 0}
        self.assertEqual(timeline_target_specs(
            negative,
            forecast_horizon=24,
            confirmation_drops=24,
        ), [
            {'role': 'prediction', 'target_step': 200},
            {'role': 'forecast_midpoint', 'target_step': 212},
            {'role': 'forecast_end', 'target_step': 224},
        ])


class PairRiskTrainingSmokeTests(unittest.TestCase):
    def test_one_epoch_training_writes_frozen_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'dataset'
            labeled = root / 'labeled'
            root.mkdir(parents=True)
            (root / 'manifest.json').write_text(json.dumps({
                'simulator_config': {
                    'max_fruits': 6,
                    'queue_length': 4,
                    'board_width': 560,
                    'board_height': 1120,
                    'wall_width': 20,
                    'physics_fps': 30,
                }
            }), encoding='utf-8')
            counts = {
                split: {
                    str(level): {'positive': 0, 'negative': 0}
                    for level in range(7, 12)
                }
                for split in ('train', 'validation', 'test')
            }
            counts['train']['8'] = {'positive': 2, 'negative': 2}
            counts['validation']['8'] = {'positive': 1, 'negative': 1}
            counts['test']['8'] = {'positive': 1, 'negative': 1}
            labeled.mkdir()
            (labeled / 'manifest.json').write_text(json.dumps({
                'counts': counts,
                'forecast_horizon': 24,
                'confirmation_drops': 24,
            }), encoding='utf-8')
            writer = ShardedTensorWriter(
                labeled, 'pair_risk_samples', LABELED_DTYPES,
                shard_rows=100, background=False,
            )
            rows = 8
            state = _model_columns(rows)
            splits = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
            labels = torch.tensor(
                [1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bool
            )
            writer.append({
                'episode_id': torch.arange(rows),
                'step': torch.arange(rows) + 40,
                'pair_slot_i': state['pair_slot_i'],
                'pair_slot_j': state['pair_slot_j'],
                'fruit_id_i': torch.arange(rows) * 2 + 1,
                'fruit_id_j': torch.arange(rows) * 2 + 2,
                'level': torch.full((rows,), 8),
                'positions': state['positions'],
                'velocities': torch.zeros(rows, 6, 2),
                'angular_velocities': torch.zeros(rows, 6),
                'levels': state['levels'],
                'fruit_ids': torch.arange(rows * 6).reshape(rows, 6) + 1,
                'physics_radii': state['physics_radii'],
                'age_frames': state['age_frames'],
                'active': state['active'],
                'fruit_queue': state['fruit_queue'],
                'danger_progress': state['danger_progress'],
                'over_danger_line': state['over_danger_line'],
                'label': labels,
                'split': splits,
                'event_id': torch.where(
                    labels, torch.arange(rows), torch.full((rows,), -1)
                ),
                'lead_to_onset': torch.where(
                    labels, torch.full((rows,), 4), torch.zeros(rows)
                ),
            })
            writer.close()
            output = Path(temporary) / 'run'
            result = train(Namespace(
                dataset_dir=root,
                output_dir=output,
                device='cpu',
                epochs=1,
                batch_size=2,
                learning_rate=1e-3,
                weight_decay=1e-4,
                grad_clip_norm=5.0,
                seed=3,
                level_embedding_dim=8,
                context_hidden_dim=16,
                head_hidden_dim=16,
                autocast_bfloat16=False,
                early_stop_patience=2,
            ))
            self.assertEqual(result['status'], 'complete')
            self.assertTrue((output / 'best.pt').is_file())
            self.assertEqual(result['test']['samples'], 2)


if __name__ == '__main__':
    unittest.main()
