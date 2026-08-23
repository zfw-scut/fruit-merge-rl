"""同级高等级水果对长期停滞检测测试。"""

import unittest

import torch

from daxigua.rl.pair_failure import PairFailureConfig, PairFailureTracker


class PairFailureTrackerTests(unittest.TestCase):
    def setUp(self):
        self.config = PairFailureConfig(
            motion_window_drops=2,
            confirmation_drops=5,
            max_net_displacement_ratio=0.10,
            adjacent_surface_gap_ratio=1.0,
        )
        self.tracker = PairFailureTracker(
            1, 4, device='cpu', config=self.config
        )

    def _frame(self, step, *, level=9, distance=240.0, shift=0.0,
               second_id=2):
        positions = torch.zeros(1, 4, 2)
        positions[0, 0] = torch.tensor([140.0 + shift, 700.0])
        positions[0, 1] = torch.tensor([140.0 + distance, 700.0])
        levels = torch.zeros(1, 4, dtype=torch.int64)
        levels[0, :2] = level
        radii = torch.zeros(1, 4)
        radii[0, :2] = 100.0 if level == 8 else 118.0
        ids = torch.zeros(1, 4, dtype=torch.int64)
        ids[0, 0] = 1
        ids[0, 1] = second_id
        active = ids.gt(0)
        steps = torch.tensor([step], dtype=torch.int64)
        return positions, levels, radii, ids, active, steps

    def test_high_level_pair_backdates_and_confirms(self):
        updates = [self.tracker.update(*self._frame(step)) for step in range(6)]
        pair = 0
        self.assertTrue(bool(updates[2].started[0, pair]))
        self.assertEqual(int(updates[2].onset_steps[0, pair]), 0)
        self.assertFalse(bool(updates[3].confirmed[0, pair]))
        self.assertFalse(bool(updates[4].confirmed[0, pair]))
        self.assertTrue(bool(updates[5].confirmed[0, pair]))
        self.assertEqual(int(updates[5].duration_drops[0, pair]), 5)

    def test_meaningful_net_displacement_cancels_candidate(self):
        self.tracker.update(*self._frame(0))
        self.tracker.update(*self._frame(1))
        started = self.tracker.update(*self._frame(2))
        self.assertTrue(bool(started.started[0, 0]))
        ended = self.tracker.update(*self._frame(3, shift=30.0))
        self.assertTrue(bool(ended.ended[0, 0]))
        self.assertFalse(bool(ended.ended_after_confirmation[0, 0]))

    def test_returning_to_same_position_does_not_accumulate_motion(self):
        self.tracker.update(*self._frame(0))
        self.tracker.update(*self._frame(1, shift=30.0))
        returned = self.tracker.update(*self._frame(2))
        self.assertTrue(bool(returned.started[0, 0]))
        self.assertLessEqual(
            float(returned.net_displacement_ratio_i[0, 0]), 1e-6
        )

    def test_medium_level_requires_persistent_adjacency(self):
        far = [
            self.tracker.update(*self._frame(step, level=8, distance=450.0))
            for step in range(6)
        ]
        self.assertFalse(any(bool(update.started.any()) for update in far))

        self.tracker.reset()
        near = [
            self.tracker.update(*self._frame(step, level=8, distance=280.0))
            for step in range(6)
        ]
        self.assertTrue(bool(near[2].started[0, 0]))
        self.assertTrue(bool(near[5].confirmed[0, 0]))

    def test_low_levels_are_ignored(self):
        updates = [
            self.tracker.update(*self._frame(step, level=6))
            for step in range(8)
        ]
        self.assertFalse(any(bool(update.started.any()) for update in updates))

    def test_identity_change_ends_candidate(self):
        for step in range(6):
            update = self.tracker.update(*self._frame(step))
        self.assertTrue(bool(update.confirmed[0, 0]))
        ended = self.tracker.update(*self._frame(6, second_id=3))
        self.assertTrue(bool(ended.ended[0, 0]))
        self.assertTrue(bool(ended.ended_after_confirmation[0, 0]))
        self.assertEqual(int(ended.fruit_id_j[0, 0]), 2)


if __name__ == '__main__':
    unittest.main()
