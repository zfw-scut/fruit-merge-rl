import base64
import gzip
import json
import re
import tempfile
from pathlib import Path
import unittest

import torch

from daxigua.simulator import (
    BatchSimulationTrace,
    SimulatorConfig,
    load_fruit_texture_data_urls,
    load_trace_archive,
    save_trace_archive,
    trace_to_payload,
    write_replay_catalog,
    write_replay_fragment,
    write_replay_html,
    write_replay_payload_html,
)


def sample_trace():
    positions = torch.tensor([[[[100.0, 80.0], [0.0, 0.0]],
                               [[100.0, 100.0], [130.0, 100.0]],
                               [[115.0, 140.0], [0.0, 0.0]]]])
    velocities = torch.tensor([[[[0.0, 10.0], [0.0, 0.0]],
                                [[5.0, 8.0], [-5.0, 8.0]],
                                [[0.0, 0.0], [0.0, 0.0]]]])
    return BatchSimulationTrace(
        env_indices=torch.tensor([7]),
        actions=torch.tensor([3]),
        record_counts=torch.tensor([3]),
        frame_numbers=torch.tensor([[0, 2, 3]]),
        positions=positions,
        velocities=velocities,
        angles=torch.tensor([[[0.0, 0.0], [0.1, -0.1], [0.25, 0.0]]]),
        angular_velocities=torch.tensor(
            [[[0.2, 0.0], [0.2, -0.2], [0.0, 0.0]]]
        ),
        levels=torch.tensor([[[1, 0], [1, 1], [2, 0]]]),
        physics_radii=torch.tensor(
            [[[20.0, 0.0], [20.0, 20.0], [30.0, 0.0]]]
        ),
        fruit_ids=torch.tensor([[[10, 0], [10, 11], [12, 0]]]),
        active=torch.tensor(
            [[[True, False], [True, True], [True, False]]]
        ),
        scores=torch.tensor([[0, 1, 1]]),
        merge_counts=torch.tensor([[0, 1, 1]]),
        stable=torch.tensor([True]),
        done=torch.tensor([False]),
        truncated=torch.tensor([False]),
        score_deltas=torch.tensor([1]),
        physics_fps=60,
        frame_stride=2,
        settle_timeout=torch.tensor([False]),
    )


class ReplayPayloadTest(unittest.TestCase):
    def test_readable_payload_remains_compatible(self):
        payload = trace_to_payload(sample_trace(), SimulatorConfig())

        self.assertFalse(payload['compact_records'])
        self.assertEqual(payload['clips'][0]['records'][0]['frame'], 0)
        self.assertTrue(payload['clips'][0]['records'][0]['drop_start'])
        self.assertEqual(payload['fruit_display_radii'][0], 20)

    def test_compact_payload_uses_documented_record_schema(self):
        payload = trace_to_payload(
            sample_trace(), SimulatorConfig(), compact=True
        )

        self.assertTrue(payload['compact_records'])
        self.assertEqual(
            payload['record_schema'],
            ['frame', 'local_frame', 'drop', 'score', 'merges', 'fruits'],
        )
        self.assertEqual(payload['clips'][0]['records'][2][:5], [3, 3, 1, 1, 1])

    def test_all_project_textures_are_embeddable_pngs(self):
        urls = load_fruit_texture_data_urls()

        self.assertEqual(len(urls), 12)
        self.assertIsNone(urls[0])
        self.assertTrue(all(
            value.startswith('data:image/png;base64,')
            for value in urls[1:]
        ))


class ReplayFileTest(unittest.TestCase):
    def test_html_contains_textures_navigation_and_debug_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_replay_html(
                Path(directory) / 'sample.html',
                sample_trace(),
                title='纹理 <回放>',
            )
            html = path.read_text(encoding='utf-8')

        match = re.search(
            r"decodePayload\('gzip-base64',(" + r'"[A-Za-z0-9+/=]+"' + r")\)",
            html,
        )
        self.assertIsNotNone(match)
        encoded = json.loads(match.group(1))
        payload = json.loads(gzip.decompress(base64.b64decode(encoded)))

        self.assertIn('<title>纹理 &lt;回放&gt;</title>', html)
        self.assertEqual(html.count('data:image/png;base64,'), 11)
        self.assertIn('智能浏览（压缩长等待）', html)
        self.assertIn('下一合成', html)
        self.assertIn('下一超时', html)
        self.assertIn('jumpTimeout', html)
        self.assertIn('线速度向量', html)
        self.assertIn('GNN-DQN 模型决策', html)
        self.assertIn('21 个动作 Q 值', html)
        self.assertTrue(payload['compact_records'])

    def test_extended_payload_can_keep_model_decision_metadata(self):
        payload = trace_to_payload(
            sample_trace(), SimulatorConfig(), compact=True
        )
        payload['model_viewer'] = {'checkpoint': 'best.pt'}
        payload['clips'][0]['drop_summaries'][0]['decision'] = {
            'drop': 1,
            'action': 3,
            'q_values': [float(index) for index in range(21)],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = write_replay_payload_html(
                Path(directory) / 'model.html', payload
            )
            html = path.read_text(encoding='utf-8')
        match = re.search(
            r"decodePayload\('gzip-base64',(" + r'"[A-Za-z0-9+/=]+"' + r")\)",
            html,
        )
        decoded = json.loads(gzip.decompress(base64.b64decode(
            json.loads(match.group(1))
        )))
        decision = decoded['clips'][0]['drop_summaries'][0]['decision']
        self.assertEqual(decision['action'], 3)
        self.assertEqual(len(decision['q_values']), 21)

    def test_fragment_uses_the_same_full_player(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_replay_fragment(
                Path(directory) / 'sample.html', sample_trace()
            )
            fragment = path.read_text(encoding='utf-8')

        self.assertNotIn('<!doctype html>', fragment.lower())
        self.assertIn('水果贴图', fragment)
        self.assertIn('data:image/png;base64,', fragment)

    def test_trace_archive_round_trip_supports_fast_gzip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_trace_archive(
                Path(directory) / 'sample.pt.gz', sample_trace()
            )
            restored = load_trace_archive(path)

        self.assertEqual(len(restored), 1)
        self.assertTrue(torch.equal(restored[0].positions, sample_trace().positions))
        self.assertEqual(restored[0].physics_fps, 60)

    def test_catalog_percent_encodes_paths_and_lazy_loads_one_iframe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_path = root / 'sample replay.html'
            replay_path.write_text('placeholder', encoding='utf-8')
            catalog = write_replay_catalog(
                root / 'index.html',
                [{
                    'env_index': 7,
                    'step_count': 123,
                    'score': 456,
                    'physics_frames_in_replay': 789,
                    'end_kind': 'terminated',
                    'settle_timeout_count': 12,
                    'settle_timeout_rate': 0.25,
                    'replay': replay_path,
                }],
                title='完整局目录',
            )
            html = catalog.read_text(encoding='utf-8')

        self.assertIn('sample%20replay.html', html)
        self.assertEqual(html.count('<iframe'), 1)
        self.assertIn('只加载一个长局回放', html)
        self.assertIn('超时次数：多到少', html)
        self.assertIn('超时 ${entry.settle_timeout_count} 次', html)
        self.assertIn('"settle_timeout_count":12', html)


if __name__ == '__main__':
    unittest.main()
