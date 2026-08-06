import tempfile
import unittest
from pathlib import Path

from daxigua.simulator.scenario_lab import fruit_specs, write_scenario_lab_html
from daxigua.simulator.scenario_lab_web import render_scenario_lab_document


class ScenarioLabFrontendTests(unittest.TestCase):
    def test_render_contains_editor_and_explicit_frontend_boundary(self):
        html = render_scenario_lab_document(
            title='场景测试',
            fruit_specs_json='[]',
            textures_json='[null]',
        )

        self.assertIn('场景实验室', html)
        self.assertIn('前端预览', html)
        self.assertIn('id="board"', html)
        self.assertIn('评估 21 个动作', html)
        self.assertIn('不会展示模拟数据', html)
        self.assertIn('daxigua:scenario-request', html)
        self.assertNotIn('文件(F)', html)

    def test_fruit_specs_follow_all_stable_levels(self):
        specs = fruit_specs()

        self.assertEqual(11, len(specs))
        self.assertEqual('樱桃', specs[0]['name'])
        self.assertEqual('西瓜', specs[-1]['name'])
        self.assertEqual(20, specs[0]['radius'])
        self.assertEqual(156, specs[-1]['radius'])

    def test_writer_embeds_all_textures_and_is_self_contained(self):
        with tempfile.TemporaryDirectory() as directory:
            output = write_scenario_lab_html(Path(directory) / 'lab.html')
            html = output.read_text(encoding='utf-8')

        self.assertIn('data:image/png;base64,', html)
        self.assertNotIn('<script src=', html)
        self.assertNotIn('<link rel=', html)


if __name__ == '__main__':
    unittest.main()
