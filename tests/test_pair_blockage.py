"""当前几何水果对堵塞检测器测试。"""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from daxigua.rl.merge_potential_stats import ShardedTensorWriter
from daxigua.rl.pair_blockage import (
    BLOCKAGE_LABELED_DTYPES,
    BLOCKAGE_TABLE,
    PairBlockageModel,
    PairBlockageModelConfig,
    finalize_pair_blockage_dataset,
)
from daxigua.rl.pair_risk import (
    END_NATURAL,
    EPISODE_DTYPES,
    EVENT_CONFIRMED,
    EVENT_DTYPES,
    EVENT_ENDED,
    EXPOSURE_DTYPES,
)
from tools.train_pair_blockage import (
    balanced_sample_weights,
    build_training_balance,
    training_balance_to,
)


def _geometry_columns(batch=3, fruits=6):
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
    return {
        'positions': positions,
        'levels': levels,
        'physics_radii': radii,
        'active': levels.gt(0),
        'pair_slot_i': torch.zeros(batch, dtype=torch.int64),
        'pair_slot_j': torch.ones(batch, dtype=torch.int64),
    }


class PairBlockageModelTests(unittest.TestCase):
    def test_model_is_pair_swap_invariant_and_geometry_only(self):
        torch.manual_seed(17)
        model = PairBlockageModel(
            PairBlockageModelConfig(max_fruits=6)
        )
        columns = _geometry_columns()
        first = model(columns)
        swapped = dict(columns)
        swapped['pair_slot_i'] = columns['pair_slot_j']
        swapped['pair_slot_j'] = columns['pair_slot_i']
        second = model(swapped)
        self.assertEqual(tuple(first.shape), (3,))
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
        self.assertEqual(set(model.REQUIRED_COLUMNS), {
            'positions', 'levels', 'physics_radii', 'active',
            'pair_slot_i', 'pair_slot_j',
        })
        first.sum().backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.parameters()
        ))


class PairBlockageLabelTests(unittest.TestCase):
    def _exposures(self):
        steps = (4, 12, 36, 48, 52, 72, 80)
        rows = len(steps)
        state = _geometry_columns(rows)
        return {
            'episode_id': torch.full((rows,), 2, dtype=torch.int64),
            'step': torch.tensor(steps, dtype=torch.int32),
            'pair_slot_i': torch.zeros(rows, dtype=torch.int16),
            'pair_slot_j': torch.ones(rows, dtype=torch.int16),
            'fruit_id_i': torch.full((rows,), 10, dtype=torch.int64),
            'fruit_id_j': torch.full((rows,), 11, dtype=torch.int64),
            'level': torch.full((rows,), 8, dtype=torch.int8),
            'positions': state['positions'],
            'velocities': torch.zeros(rows, 6, 2),
            'angular_velocities': torch.zeros(rows, 6),
            'levels': state['levels'].to(torch.int8),
            'fruit_ids': torch.arange(rows * 6).reshape(rows, 6) + 1,
            'physics_radii': state['physics_radii'],
            'age_frames': torch.full((rows, 6), 300, dtype=torch.int32),
            'active': state['active'],
            'fruit_queue': torch.ones(rows, 4, dtype=torch.int8),
            'danger_progress': torch.zeros(rows),
            'over_danger_line': torch.zeros(rows, dtype=torch.bool),
        }

    def test_full_confirmed_interval_is_positive_and_tail_is_censored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'raw'
            exposures = ShardedTensorWriter(
                raw, 'pair_risk_exposures', EXPOSURE_DTYPES,
                shard_rows=100, background=False,
            )
            events = ShardedTensorWriter(
                raw, 'pair_risk_events', EVENT_DTYPES,
                shard_rows=100, background=False,
            )
            episodes = ShardedTensorWriter(
                raw, 'pair_risk_episodes', EPISODE_DTYPES,
                shard_rows=100, background=False,
            )
            exposures.append(self._exposures())
            events.append({
                'episode_id': torch.tensor([2, 2]),
                'event_kind': torch.tensor([EVENT_CONFIRMED, EVENT_ENDED]),
                'onset_step': torch.tensor([12, 12]),
                'event_step': torch.tensor([36, 50]),
                'fruit_id_i': torch.tensor([10, 10]),
                'fruit_id_j': torch.tensor([11, 11]),
                'level': torch.tensor([8, 8]),
            })
            episodes.append({
                'episode_id': torch.tensor([2]),
                'seed': torch.tensor([123]),
                'end_step': torch.tensor([100]),
                'end_kind': torch.tensor([END_NATURAL]),
            })
            exposures.close()
            events.close()
            episodes.close()

            result = finalize_pair_blockage_dataset(
                root, confirmation_drops=24, shard_rows=100
            )
            self.assertEqual(result['labeled_rows'], 6)
            self.assertEqual(result['censored_rows'], 1)
            shard = torch.load(
                next((root / 'blocked_now').glob(
                    f'{BLOCKAGE_TABLE}-*.pt'
                )),
                weights_only=False,
            )['columns']
            self.assertEqual(shard['step'].tolist(), [4, 12, 36, 48, 52, 72])
            self.assertEqual(
                shard['label'].tolist(), [False, True, True, True, False, False]
            )
            self.assertEqual(
                shard['offset_from_onset'].tolist(), [0, 0, 24, 36, 0, 0]
            )
            self.assertEqual(set(shard), set(BLOCKAGE_LABELED_DTYPES))
            self.assertNotIn('age_frames', shard)
            self.assertNotIn('fruit_queue', shard)
            manifest = json.loads(
                (root / 'blocked_now' / 'manifest.json').read_text('utf-8')
            )
            self.assertEqual(
                manifest['label_semantics'],
                'confirmed_event_onset_le_t_le_event_end',
            )


class PairBlockageBalanceTests(unittest.TestCase):
    def test_event_duration_does_not_increase_positive_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / f'{BLOCKAGE_TABLE}-00000.pt'
            columns = {
                'split': torch.zeros(8, dtype=torch.int8),
                'level': torch.tensor([7, 7, 7, 7, 8, 8, 8, 8]),
                'label': torch.tensor(
                    [False, False, True, True, False, False, True, True]
                ),
                'event_id': torch.tensor([-1, -1, 0, 0, -1, -1, 1, 1]),
            }
            torch.save({
                'format_version': 1,
                'table': BLOCKAGE_TABLE,
                'columns': columns,
            }, path)
            balance = training_balance_to(
                build_training_balance((path,)), torch.device('cpu')
            )
            weights = balanced_sample_weights(columns, balance)
            for level in (7, 8):
                for label in (False, True):
                    mask = columns['level'].eq(level) & columns['label'].eq(label)
                    self.assertAlmostEqual(
                        float(weights[mask].sum()), 2.0, places=5
                    )


if __name__ == '__main__':
    unittest.main()
