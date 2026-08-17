import unittest

import torch

from daxigua.rl.merge_distance_key_nodes import (
    StableSwitchDetector,
    merge_difficulty,
)


class MergeDistanceKeyNodeTest(unittest.TestCase):
    def test_probability_distribution_maps_to_expected_class_index(self):
        probabilities = torch.tensor((
            (0.5, 0.5, 0.0),
            (0.0, 0.25, 0.75),
        ))
        self.assertTrue(torch.allclose(
            merge_difficulty(probabilities), torch.tensor((0.5, 1.75))
        ))

    def test_detects_stable_switch_and_preserves_boundaries(self):
        detector = StableSwitchDetector(
            window_size=4,
            stable_range=0.2,
            jump_threshold=0.75,
            transition_timeout=12,
        )
        for step, value in enumerate((2.0, 2.1, 2.0, 1.95), 10):
            update = detector.update(value, step)
        self.assertTrue(detector.is_stable)

        update = detector.update(3.2, 14)
        self.assertTrue(update.transition_started)
        for step, value in enumerate((3.3, 3.25, 3.3), 15):
            update = detector.update(value, step)
        self.assertIsNotNone(update.switch)
        self.assertEqual(update.switch.start_step, 14)
        self.assertEqual(update.switch.settled_start_step, 14)
        self.assertEqual(update.switch.confirmed_step, 17)
        self.assertEqual(update.switch.direction, 'worsened')

    def test_short_excursion_is_cancelled(self):
        detector = StableSwitchDetector(
            window_size=3,
            stable_range=0.15,
            jump_threshold=0.8,
            transition_timeout=9,
        )
        for step in range(3):
            detector.update(4.0, step)
        self.assertTrue(detector.update(5.0, 3).transition_started)
        cancelled = False
        for step in range(4, 7):
            cancelled |= detector.update(4.0, step).transition_cancelled
        self.assertTrue(cancelled)
        self.assertFalse(detector.in_transition)


if __name__ == '__main__':
    unittest.main()
