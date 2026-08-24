"""当前几何堵塞人工画廊测试。"""

from pathlib import Path
import tempfile
import unittest

import torch

from tools.render_pair_blockage_gallery import (
    candidate_identity,
    confusion_kind,
    render_candidate,
    select_distinct_candidates,
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


if __name__ == '__main__':
    unittest.main()
