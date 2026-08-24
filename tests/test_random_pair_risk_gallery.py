import math
from pathlib import Path
import tempfile
import unittest

from tools.render_random_pair_risk_gallery import (
    _eligible_pair_count,
    _simulator_config,
    generate_random_scene,
    render_scene,
    risk_color,
)


class RandomPairRiskGalleryTests(unittest.TestCase):
    def test_random_scene_is_non_overlapping_and_has_two_target_pairs(self):
        config = _simulator_config(30)
        scene = generate_random_scene(
            0, 202_608_240, config, 30, 35
        )

        self.assertIsNotNone(scene)
        self.assertEqual(35, len(scene['fruits']))
        generation = scene['_generation']
        for level in (
                generation['target_level'], generation['backup_level']):
            self.assertGreaterEqual(sum(
                fruit['level'] == level for fruit in scene['fruits']
            ), 2)
        for first, fruit_i in enumerate(scene['fruits']):
            for fruit_j in scene['fruits'][first + 1:]:
                distance = math.hypot(
                    fruit_i['x'] - fruit_j['x'],
                    fruit_i['y'] - fruit_j['y'],
                )
                self.assertGreater(
                    distance,
                    fruit_i['physics_radius']
                    + fruit_j['physics_radius'],
                )

    def test_render_includes_complete_pair_prediction(self):
        config = _simulator_config(30)
        scene = {
            'fps': 30,
            'queue': [1, 2, 3, 4],
            'score': 0,
            'step_count': 120,
            'danger_progress': 0.0,
            'over_danger_line': False,
            'fruits': [
                {
                    'id': 1, 'level': 7, 'physics_radius': 73.0,
                    'x': 140.0, 'y': 1020.0,
                },
                {
                    'id': 2, 'level': 7, 'physics_radius': 73.0,
                    'x': 410.0, 'y': 1020.0,
                },
            ],
            '_generation': {
                'index': 0,
                'accepted_seed': 123,
                'target_level': 7,
                'backup_level': 8,
                'initial_fruit_count': 2,
            },
        }
        prediction = {
            'forecast_horizon': 24,
            'inference_ms': 1.25,
            'pairs': [{
                'fruit_id_i': 1,
                'fruit_id_j': 2,
                'level': 7,
                'probability': 0.75,
            }],
        }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'scene.png'
            render_scene(scene, prediction, config, output, dpi=60)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)

        self.assertEqual(1, _eligible_pair_count(scene))
        self.assertEqual('#b55353', risk_color(0.75))


if __name__ == '__main__':
    unittest.main()
