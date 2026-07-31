"""最小分支保留的领域规则和状态契约测试。"""

from dataclasses import FrozenInstanceError
from random import Random
import unittest

from daxigua.core import (
    ActionCandidate,
    BoardGeometry,
    FRUIT_QUEUE_LENGTH,
    MAX_FRUIT_LEVEL,
    MIN_FRUIT_LEVEL,
    FruitState,
    dropped_fruit_physics_radius,
    fruit_radius,
    merge_score,
    merge_target_level,
    merged_fruit_physics_radius,
    random_spawn_level,
)


class DomainRulesTest(unittest.TestCase):
    def test_level_and_queue_boundaries_are_stable(self):
        self.assertEqual((MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL), (1, 11))
        self.assertEqual(FRUIT_QUEUE_LENGTH, 4)
        self.assertEqual(merge_target_level(10), 11)
        self.assertIsNone(merge_target_level(11))

    def test_display_and_physics_radii_remain_distinct(self):
        self.assertEqual(fruit_radius(3), 42)
        self.assertEqual(dropped_fruit_physics_radius(3), 40)
        self.assertEqual(merged_fruit_physics_radius(3), 41)

    def test_merge_score_is_the_game_rule_not_an_rl_reward(self):
        self.assertEqual(merge_score(2), 2)
        self.assertEqual(merge_score(10), 10)
        self.assertEqual(merge_score(11), 100)
        self.assertEqual(merge_score(12), 0)

    def test_spawn_generation_stays_inside_the_published_range(self):
        rng = Random(0)
        generated = {random_spawn_level(rng) for _ in range(256)}
        self.assertEqual(generated, {1, 2, 3, 4})


class StateContractTest(unittest.TestCase):
    def _fruit(self, **overrides):
        values = {
            'fruit_id': 1,
            'level': 3,
            'radius': 42,
            'x': 100,
            'y': 200,
            'vx': 0,
            'vy': 0,
            'angle': 0,
            'angular_velocity': 0,
            'age_frames': 0,
            'stable': True,
            'distance_to_left_wall': 58,
            'distance_to_right_wall': 318,
            'distance_to_floor': 438,
            'distance_to_danger_line': 58,
        }
        values.update(overrides)
        return FruitState(**values)

    def test_legacy_radius_fallback_and_explicit_physics_radius(self):
        self.assertEqual(self._fruit().physics_radius, 42.0)
        self.assertEqual(self._fruit(physics_radius=40).physics_radius, 40.0)

        action = ActionCandidate(0, 100, 0.5, 3, 42, 40)
        self.assertEqual(action.current_radius, 42.0)
        self.assertEqual(action.current_physics_radius, 40.0)

    def test_non_positive_radii_are_rejected(self):
        with self.assertRaises(ValueError):
            self._fruit(radius=0)
        with self.assertRaises(ValueError):
            ActionCandidate(0, 100, 0.5, 3, 42, 0)

    def test_contract_objects_are_immutable(self):
        geometry = BoardGeometry(560, 1120, 252, 20, 1100)
        with self.assertRaises(FrozenInstanceError):
            geometry.width = 400


if __name__ == '__main__':
    unittest.main()
