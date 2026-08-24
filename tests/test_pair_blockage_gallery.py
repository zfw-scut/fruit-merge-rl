"""当前几何堵塞人工画廊测试。"""

from pathlib import Path
import tempfile
import unittest

import torch

from tools.render_pair_blockage_gallery import (
    attach_positive_transitions,
    candidate_identity,
    confusion_kind,
    render_candidate,
    select_distinct_candidates,
)
from daxigua.rl.merge_potential_stats import ShardedTensorWriter
from daxigua.rl.pair_blockage import (
    BLOCKAGE_AREA,
    BLOCKAGE_LABELED_DTYPES,
    BLOCKAGE_TABLE,
    PairBlockageModel,
    PairBlockageModelConfig,
)


def _candidate(**updates):
    candidate = {
        'kind': 'TP',
        'probability': 0.9,
        'priority': 0.9,
        'label': True,
        'episode_id': 11,
        'step': 80,
        'pair_slot_i': 0,
        'pair_slot_j': 1,
        'fruit_id_i': 101,
        'fruit_id_j': 102,
        'level': 8,
        'positions': torch.tensor([
            [130.0, 850.0], [390.0, 850.0], [260.0, 980.0],
            [0.0, 0.0],
        ]),
        'levels': torch.tensor([8, 8, 5, 0]),
        'physics_radii': torch.tensor([100.0, 100.0, 60.0, 0.0]),
        'active': torch.tensor([True, True, True, False]),
        'event_id': 7,
        'offset_from_onset': 8,
        'offset_to_end': 16,
    }
    candidate.update(updates)
    return candidate


class PairBlockageGalleryTests(unittest.TestCase):
    def test_confusion_kind_uses_current_binary_semantics(self):
        self.assertEqual(confusion_kind(True, True), 'TP')
        self.assertEqual(confusion_kind(True, False), 'FP')
        self.assertEqual(confusion_kind(False, True), 'FN')
        self.assertEqual(confusion_kind(False, False), 'TN')

    def test_positive_samples_are_deduplicated_by_event(self):
        first = _candidate(priority=0.9)
        duplicate = _candidate(step=84, priority=0.8)
        second = _candidate(event_id=8, priority=0.7)
        selected = select_distinct_candidates(
            [duplicate, second, first], count=3
        )
        self.assertEqual([item['event_id'] for item in selected], [7, 8])
        self.assertEqual(candidate_identity(first), ('event', 7))

    def test_negative_samples_are_deduplicated_by_episode_and_pair(self):
        first = _candidate(label=False, event_id=-1, priority=0.9)
        duplicate = _candidate(
            label=False, event_id=-1, step=84, priority=0.8,
            fruit_id_i=102, fruit_id_j=101,
        )
        second = _candidate(
            label=False, event_id=-1, episode_id=12, priority=0.7,
        )
        selected = select_distinct_candidates(
            [duplicate, second, first], count=3
        )
        self.assertEqual(len(selected), 2)

    def test_single_frame_gallery_renders(self):
        board = {
            'board_width': 560.0,
            'board_height': 1120.0,
            'wall_width': 20.0,
            'spawn_y': 252.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'scene.png'
            render_candidate(_candidate(), board, 0.5, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 10_000)

    def test_positive_transition_gallery_renders_three_frames(self):
        board = {
            'board_width': 560.0,
            'board_height': 1120.0,
            'wall_width': 20.0,
            'spawn_y': 252.0,
        }
        candidate = _candidate(offset_from_onset=8)
        candidate['onset_step'] = 72
        before = _candidate(
            kind='TN', probability=0.1, label=False, event_id=-1,
            step=68, offset_from_onset=0, offset_to_end=0,
        )
        onset = _candidate(
            kind='FN', probability=0.3, step=72,
            offset_from_onset=0, offset_to_end=24,
        )
        candidate['transition'] = (
            {'role': 'before_onset', 'frame': before},
            {'role': 'onset', 'frame': onset},
            {'role': 'classified', 'frame': candidate},
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'transition.png'
            render_candidate(candidate, board, 0.5, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 20_000)

    def test_transition_frames_are_recovered_from_labeled_shard(self):
        before = _candidate(
            kind='TN', probability=0.1, label=False, event_id=-1,
            step=68, offset_from_onset=0, offset_to_end=0,
        )
        onset = _candidate(
            kind='FN', probability=0.3, step=72,
            offset_from_onset=0, offset_to_end=24,
        )
        current = _candidate(step=80, offset_from_onset=8)
        frames = (before, onset, current)
        columns = {}
        scalar_names = (
            'episode_id', 'step', 'pair_slot_i', 'pair_slot_j',
            'fruit_id_i', 'fruit_id_j', 'level', 'label', 'event_id',
            'offset_from_onset', 'offset_to_end',
        )
        tensor_names = ('positions', 'levels', 'physics_radii', 'active')
        for name in scalar_names:
            columns[name] = torch.tensor(
                [frame[name] for frame in frames],
                dtype=BLOCKAGE_LABELED_DTYPES[name],
            )
        columns['split'] = torch.full(
            (3,), 2, dtype=BLOCKAGE_LABELED_DTYPES['split']
        )
        for name in tensor_names:
            columns[name] = torch.stack([
                frame[name] for frame in frames
            ]).to(BLOCKAGE_LABELED_DTYPES[name])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = ShardedTensorWriter(
                root / BLOCKAGE_AREA, BLOCKAGE_TABLE,
                BLOCKAGE_LABELED_DTYPES,
                shard_rows=10, background=False,
            )
            writer.append(columns)
            writer.close()
            model = PairBlockageModel(
                PairBlockageModelConfig(max_fruits=4)
            )
            summary = attach_positive_transitions(
                root, {(8, 'TP'): [current]}, model,
                torch.device('cpu'), threshold=0.5,
                autocast_bfloat16=False,
            )
        self.assertEqual(summary, {
            'positive_cases': 1, 'complete_cases': 1,
        })
        self.assertEqual(current['before_onset_frame']['step'], 68)
        self.assertEqual(current['onset_frame']['step'], 72)


if __name__ == '__main__':
    unittest.main()
