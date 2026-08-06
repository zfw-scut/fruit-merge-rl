import tempfile
import unittest
from pathlib import Path

from daxigua.simulator.scenario_lab import fruit_specs, write_scenario_lab_html
from daxigua.simulator.scenario_lab_service import validate_scenario
from daxigua.simulator.scenario_lab_web import render_scenario_lab_document


class ScenarioLabFrontendTests(unittest.TestCase):
    def test_render_contains_editor_and_reward_backend_contract(self):
        html = render_scenario_lab_document(
            title='场景测试',
            fruit_specs_json='[]',
            textures_json='[null]',
        )

        self.assertIn('场景实验室', html)
        self.assertIn('Reward V2', html)
        self.assertIn('id="board"', html)
        self.assertIn('评估 21 个动作', html)
        self.assertIn('daxigua:scenario-request', html)
        self.assertIn('/api/evaluate', html)
        self.assertIn('effective_normalized_area', html)
        self.assertNotIn('文件(F)', html)

    def test_fruit_specs_follow_all_stable_levels(self):
        specs = fruit_specs()

        self.assertEqual(11, len(specs))
        self.assertEqual('樱桃', specs[0]['name'])
        self.assertEqual('西瓜', specs[-1]['name'])
        self.assertEqual(20, specs[0]['radius'])
        self.assertEqual(156, specs[-1]['radius'])
        self.assertGreater(specs[0]['dropped_physics_radius'], 0)
        self.assertGreater(specs[0]['merged_physics_radius'], 0)

    def test_writer_embeds_all_textures_and_is_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            output = write_scenario_lab_html(Path(directory) / 'lab.html')
            html = output.read_text(encoding='utf-8')

        self.assertIn('data:image/png;base64,', html)
        self.assertNotIn('<script src=', html)
        self.assertNotIn('<link rel=', html)


class ScenarioLabBackendContractTests(unittest.TestCase):
    def test_validate_scenario_normalizes_reward_v2_inputs(self):
        scene = validate_scenario({
            'name': '测试场景',
            'fps': 30,
            'queue': [1, 2, 3, 4],
            'probe_action': 7,
            'fruits': [{
                'id': 9,
                'level': 3,
                'x': 120.0,
                'y': 900.0,
            }],
        })

        self.assertEqual((1, 2, 3, 4), scene['queue'])
        self.assertEqual(7, scene['probe_action'])
        self.assertGreater(scene['fruits'][0]['physics_radius'], 0.0)

    def test_validate_scenario_rejects_unspawnable_queue_level(self):
        with self.assertRaises(ValueError):
            validate_scenario({
                'fps': 120,
                'queue': [1, 2, 3, 6],
                'fruits': [],
            })

    def test_validate_scenario_rejects_duplicate_fruit_ids(self):
        with self.assertRaises(ValueError):
            validate_scenario({
                'fps': 120,
                'queue': [1, 2, 3, 4],
                'fruits': [
                    {'id': 1, 'level': 1, 'x': 100.0, 'y': 900.0},
                    {'id': 1, 'level': 2, 'x': 200.0, 'y': 900.0},
                ],
            })


if __name__ == '__main__':
    unittest.main()
