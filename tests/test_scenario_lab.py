import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from daxigua.simulator.scenario_lab import fruit_specs
from daxigua.simulator.scenario_lab_service import (
    ScenarioLabEvaluator,
    validate_scenario,
)
from daxigua.simulator.scenario_lab_live import ScenarioLabLiveSession
from daxigua.simulator.scenario_lab_comparison import (
    ScenarioLabComparisonSession,
)
from daxigua.simulator.scenario_lab_server import ScenarioLabServer
from daxigua.simulator.config import SimulatorConfig
from daxigua.simulator.voronoi import (
    ScenarioVoronoiEvaluator,
    WeightedVoronoiGraphBuilder,
)
from daxigua.rl.scenario_model_controller import (
    ScenarioComparisonModelController,
    ScenarioModelController,
)
from daxigua.rl.config import ModelConfig
from daxigua.rl.model import BaselineGnnDqn
from daxigua.rl.merge_distance import (
    MergeDistanceConfig,
    MergeDistancePredictor,
)
from daxigua.rl.scenario_merge_distance_evaluator import (
    ScenarioMergeDistanceEvaluator,
)
from daxigua.rl.scenario_model_evaluator import ScenarioModelEvaluator
from daxigua.rl.viewer import LoadedViewerModel


class ScenarioLabFrontendTests(unittest.TestCase):
    def test_fruit_specs_follow_all_stable_levels(self):
        specs = fruit_specs()

        self.assertEqual(11, len(specs))
        self.assertEqual('葡萄', specs[0]['name'])
        self.assertEqual('大西瓜', specs[-1]['name'])
        self.assertEqual(20, specs[0]['radius'])
        self.assertEqual(156, specs[-1]['radius'])
        self.assertGreater(specs[0]['dropped_physics_radius'], 0)
        self.assertGreater(specs[0]['merged_physics_radius'], 0)

class ScenarioLabBackendContractTests(unittest.TestCase):
    def test_merge_distance_evaluator_returns_prediction_for_each_fruit(self):
        config = MergeDistanceConfig(
            hidden_dim=16,
            edge_hidden_dim=16,
            message_layers=1,
            queue_hidden_dim=8,
            queue_layers=1,
            level_embedding_dim=6,
            max_neighbors=4,
            nearest_neighbors=2,
            motion_neighbors=1,
            vertical_neighbors_per_direction=1,
        )
        model = MergeDistancePredictor(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.output_head.bias[3] = 5.0
        with TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'predictor.pt'
            torch.save({
                'purpose': 'merge_distance_predictor',
                'model_config': config.to_dict(),
                'geometry_config': model.geometry_config,
                'model_state': model.state_dict(),
                'metrics': {'exact_bin_accuracy': 0.5},
            }, checkpoint)
            evaluator = ScenarioMergeDistanceEvaluator(
                checkpoint, device='cpu'
            )
            payload = evaluator.evaluate({
                'fps': 120,
                'queue': [1, 2, 3, 4],
                'fruits': [{
                    'id': 42,
                    'level': 3,
                    'x': 280.0,
                    'y': 900.0,
                }],
            })

        self.assertEqual(1, payload['fruit_count'])
        prediction = payload['predictions'][0]
        self.assertEqual(42, prediction['fruit_id'])
        self.assertEqual(3, prediction['class_index'])
        self.assertEqual('5–8 次', prediction['label'])
        self.assertEqual(5, prediction['min_steps'])
        self.assertEqual(8, prediction['max_steps'])
        self.assertEqual(13, len(prediction['probabilities']))

    def test_weighted_voronoi_builder_keeps_every_active_site(self):
        builder = WeightedVoronoiGraphBuilder(
            SimulatorConfig(), device='cpu', sample_spacing=8.0
        )
        positions = torch.tensor([
            [[160.0, 900.0], [400.0, 900.0], [280.0, 1030.0]],
            [[180.0, 800.0], [380.0, 850.0], [280.0, 980.0]],
        ])
        radii = torch.tensor([
            [40.0, 80.0, 30.0],
            [35.0, 55.0, 70.0],
        ])

        graphs = builder.build(positions, radii)

        self.assertEqual(2, len(graphs))
        for graph in graphs:
            self.assertEqual(6, graph.visible_site_samples.numel())
            self.assertGreater(graph.edge_start.shape[0], 0)
            self.assertEqual(graph.edge_start.shape, graph.edge_end.shape)
            self.assertTrue(torch.all(graph.edge_clearance >= 0.0))
            self.assertLessEqual(
                int(graph.edge_owners.max().item()), 5
            )

    def test_disk_weight_changes_bisector_location(self):
        builder = WeightedVoronoiGraphBuilder(
            SimulatorConfig(), device='cpu', sample_spacing=4.0
        )
        graph = builder.build(
            torch.tensor([[[160.0, 900.0], [400.0, 900.0]]]),
            torch.tensor([[40.0, 80.0]]),
        )[0]
        owner_pair = graph.edge_owners.sort(dim=-1).values
        fruit_ridge = (owner_pair[:, 0] == 0) & (owner_pair[:, 1] == 1)
        midpoints = (
            graph.edge_start[fruit_ridge] + graph.edge_end[fruit_ridge]
        ) * 0.5
        closest = torch.argmin((midpoints[:, 1] - 900.0).abs())

        self.assertAlmostEqual(260.0, float(midpoints[closest, 0]), delta=8.0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_weighted_voronoi_cuda_matches_cpu_graph(self):
        config = SimulatorConfig()
        positions = torch.tensor([[[143.0, 781.0], [391.0, 869.0]]])
        radii = torch.tensor([[43.0, 77.0]])
        cpu_graph = WeightedVoronoiGraphBuilder(
            config, device='cpu', sample_spacing=8.0
        ).build(positions, radii)[0]
        cuda_graph = WeightedVoronoiGraphBuilder(
            config, device='cuda', sample_spacing=8.0
        ).build(positions, radii)[0]

        self.assertLessEqual(int(torch.max(torch.abs(
            cpu_graph.visible_site_samples
            - cuda_graph.visible_site_samples.cpu()
        )).item()), 2)
        self.assertEqual(
            cpu_graph.edge_start.shape,
            cuda_graph.edge_start.shape,
        )
        self.assertTrue(torch.allclose(
            cpu_graph.edge_clearance.sort().values,
            cuda_graph.edge_clearance.cpu().sort().values,
            atol=1e-4,
        ))

    def test_scenario_voronoi_payload_is_compact_and_cached(self):
        evaluator = ScenarioVoronoiEvaluator(
            SimulatorConfig(), device='cpu', sample_spacing=8.0
        )
        scene = {
            'fps': 120,
            'queue': [1, 2, 3, 4],
            'fruits': [
                {'id': 4, 'level': 3, 'x': 200.0, 'y': 930.0},
                {'id': 7, 'level': 5, 'x': 350.0, 'y': 920.0},
            ],
        }

        first = evaluator.evaluate(scene)
        second = evaluator.evaluate(scene)

        self.assertEqual('full_disk_weighted_voronoi_raster_v1', first['algorithm'])
        self.assertEqual(5, len(first['sites']))
        self.assertTrue(first['edges'])
        self.assertNotIn('owner_grid', first)
        self.assertFalse(first['cache_hit'])
        self.assertTrue(second['cache_hit'])

    def test_comparison_model_controller_repeats_same_policy_action(self):
        class FakeEvaluator:
            @staticmethod
            def evaluate(scene, **_context):
                return {
                    'action': 10,
                    'drop_x': 280.0,
                    'selected_q': 1.0,
                    'q_values': [float(index) for index in range(21)],
                    'inference_ms': 0.1,
                    'queue': list(scene['queue']),
                }

        session = ScenarioLabComparisonSession(
            preset='backend_parity',
            accelerated_device='cpu',
            publish_fps=60,
        )
        controller = ScenarioComparisonModelController(
            session, FakeEvaluator(), max_decisions=2
        )
        session.start()
        controller.start_service()
        try:
            session.execute({'type': 'clear', 'queue': [1, 2, 3, 4]})
            controller.start()
            deadline = time.monotonic() + 15.0
            status = controller.status()
            while (
                    status['decision_count'] < 2
                    and time.monotonic() < deadline):
                time.sleep(0.02)
                status = controller.status()
            snapshot = session.snapshot()
        finally:
            controller.close()
            session.close()

        self.assertEqual(2, status['decision_count'])
        self.assertEqual('limit', status['phase'])
        self.assertEqual(2, snapshot['left']['step_count'])
        self.assertEqual(2, snapshot['right']['step_count'])
        self.assertFalse(snapshot['difference']['diverged'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_play_vs_training_stops_at_aligned_trace_endpoint(self):
        session = ScenarioLabComparisonSession(
            preset='play_vs_training',
            play_device='cuda',
            accelerated_device='cuda',
            publish_fps=30,
        ).start()
        try:
            session.execute({'type': 'clear', 'queue': [1, 2, 3, 4]})
            outcome = session.execute(
                {'type': 'drop_action', 'action': 10}, timeout=15.0
            )
            deadline = time.monotonic() + 20.0
            snapshot = session.snapshot()
            while not snapshot['paused'] and time.monotonic() < deadline:
                snapshot = session.wait_for_snapshot(
                    snapshot['sequence'], timeout=1.0
                )
        finally:
            session.close()

        self.assertEqual(0, outcome['right']['fast_forwarded_frames'])
        self.assertFalse(snapshot['profiles']['right']['drop_fast_forward'])
        self.assertTrue(snapshot['paused'])
        self.assertFalse(snapshot['action_in_progress'])
        self.assertTrue(snapshot['difference_comparable'])
        self.assertTrue(snapshot['right']['trace']['playback_complete'])

    def test_comparison_session_synchronizes_identical_tensor_lanes(self):
        session = ScenarioLabComparisonSession(
            preset='backend_parity',
            accelerated_device='cpu',
            publish_fps=30,
        ).start()
        try:
            session.execute({'type': 'clear', 'queue': [1, 2, 3, 4]})
            session.execute({'type': 'drop_action', 'action': 10})
            deadline = time.monotonic() + 1.0
            snapshot = session.snapshot()
            while (
                    snapshot['left']['physics_frame'] < 8
                    and time.monotonic() < deadline):
                snapshot = session.wait_for_snapshot(
                    snapshot['sequence'], timeout=0.1
                )
        finally:
            session.close()

        self.assertEqual('backend_parity', snapshot['preset'])
        self.assertEqual(
            snapshot['left']['physics_frame'],
            snapshot['right']['physics_frame'],
        )
        self.assertFalse(snapshot['difference']['diverged'])
        self.assertEqual(0.0, snapshot['difference']['max_position_delta'])

    def test_comparison_http_api_exposes_state_and_commands(self):
        comparison = ScenarioLabComparisonSession(
            preset='backend_parity',
            accelerated_device='cpu',
            publish_fps=30,
        )
        server = ScenarioLabServer(
            SimpleNamespace(device='cpu'),
            comparison_session=comparison,
            host='127.0.0.1',
            port=0,
        ).start()
        try:
            with urlopen(server.url + 'api/health', timeout=2.0) as response:
                health = json.loads(response.read())
            request = Request(
                server.url + 'api/comparison/command',
                data=json.dumps({
                    'command': {'type': 'drop_action', 'action': 10},
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urlopen(request, timeout=2.0) as response:
                accepted = json.loads(response.read())
            with urlopen(
                    server.url + 'api/comparison/state', timeout=2.0
            ) as response:
                state = json.loads(response.read())
        finally:
            server.close()

        self.assertTrue(health['comparison_available'])
        self.assertTrue(accepted['accepted'])
        self.assertEqual(1, state['left']['step_count'])
        self.assertEqual(1, state['right']['step_count'])
    def test_http_service_is_api_only_and_exposes_portal_config(self):
        server = ScenarioLabServer(
            SimpleNamespace(device='cpu'), host='127.0.0.1', port=0
        ).start()
        try:
            with self.assertRaises(HTTPError) as raised:
                urlopen(server.url, timeout=2.0)
            error_payload = json.loads(raised.exception.read())
            config_request = Request(
                server.url + 'api/config',
                headers={'Origin': 'http://127.0.0.1:3000'},
            )
            with urlopen(config_request, timeout=2.0) as response:
                config = json.loads(response.read())
                allow_origin = response.headers.get(
                    'Access-Control-Allow-Origin'
                )
        finally:
            server.close()

        self.assertEqual(410, raised.exception.code)
        self.assertEqual('standalone_scenario_lab_removed', error_payload['error'])
        self.assertEqual('http://127.0.0.1:3000/#lab', error_payload['portal'])
        self.assertEqual(11, len(config['fruit_specs']))
        self.assertTrue(config['textures'][1].startswith('data:image/png;base64,'))
        self.assertEqual(21, config['geometry']['action_count'])
        self.assertEqual(
            'full_disk_weighted_voronoi_raster_v1',
            config['voronoi']['algorithm'],
        )
        self.assertEqual('http://127.0.0.1:3000', allow_origin)

    def test_voronoi_http_api_exposes_scene_graph(self):
        server = ScenarioLabServer(
            SimpleNamespace(device='cpu'), host='127.0.0.1', port=0
        ).start()
        try:
            with urlopen(server.url + 'api/health', timeout=2.0) as response:
                health = json.loads(response.read())
            request = Request(
                server.url + 'api/voronoi/evaluate',
                data=json.dumps({
                    'scene': {
                        'fps': 120,
                        'queue': [1, 2, 3, 4],
                        'fruits': [{
                            'id': 1,
                            'level': 4,
                            'x': 280.0,
                            'y': 900.0,
                        }],
                    },
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urlopen(request, timeout=4.0) as response:
                graph = json.loads(response.read())
        finally:
            server.close()

        self.assertTrue(health['voronoi_available'])
        self.assertEqual('cpu', health['voronoi_device'])
        self.assertEqual(4, len(graph['sites']))
        self.assertGreater(graph['stats']['edge_count'], 0)
        self.assertGreater(graph['stats']['free_sample_ratio'], 0.0)

    def test_model_evaluator_returns_all_action_effect_predictions(self):
        config = ModelConfig(
            hidden_dim=32,
            edge_hidden_dim=32,
            message_layers=1,
            queue_hidden_dim=16,
            queue_layers=1,
            level_embedding_dim=8,
            max_neighbors=4,
            nearest_neighbors=2,
            motion_neighbors=1,
            vertical_neighbors_per_direction=1,
            action_key_fruits=2,
            action_effect_enabled=True,
            structured_contact_enabled=True,
        )
        model = BaselineGnnDqn(config)
        loaded = LoadedViewerModel(
            checkpoint_path=Path('synthetic.pt'),
            checkpoint_sha256='a' * 64,
            model=model.eval(),
            model_config=config,
            progress={'transitions': 123},
            device=torch.device('cpu'),
        )

        payload = ScenarioModelEvaluator(loaded).evaluate({
            'name': '辅助预测契约',
            'fps': 120,
            'queue': [1, 2, 3, 4],
            'probe_action': 10,
            'fruits': [],
        })

        self.assertEqual(2, payload['format_version'])
        self.assertTrue(payload['model']['action_effect_available'])
        self.assertEqual(21, len(payload['action_effect_predictions']))
        prediction = payload['action_effect_predictions'][10]
        self.assertIn('first_contact', prediction)
        self.assertIsNotNone(prediction['first_contact']['target'])
        self.assertEqual(3, len(prediction['generations']))
        self.assertIn('final', prediction['q0'])

    def test_model_http_api_exposes_read_only_policy_evaluation(self):
        class FakeModelEvaluator:
            identity = {
                'checkpoint': 'best.pt',
                'checkpoint_sha256': '123456789abc',
                'device': 'cpu',
                'training_transitions': 16_000_000,
            }

            @staticmethod
            def evaluate(scene):
                return {
                    'action': 7,
                    'drop_x': 210.0,
                    'selected_q': 1.25,
                    'inference_ms': 2.0,
                    'q_values': [float(index) for index in range(21)],
                    'fruit_count': len(scene['fruits']),
                }

        class FakeModelController:
            def __init__(self):
                self.running = False

            def start_service(self):
                return self

            def close(self):
                return None

            def status(self):
                return {
                    'available': True,
                    'running': self.running,
                    'decision_count': 0,
                }

            def start(self):
                self.running = True
                return self.status()

            def stop(self, **_kwargs):
                self.running = False
                return self.status()

        server = ScenarioLabServer(
            SimpleNamespace(device='cpu'),
            model_evaluator=FakeModelEvaluator(),
            model_controller=FakeModelController(),
            host='127.0.0.1',
            port=0,
        ).start()
        scene = {
            'fps': 120,
            'queue': [1, 2, 3, 4],
            'fruits': [],
        }
        try:
            with urlopen(server.url + 'api/health', timeout=2.0) as response:
                health = json.loads(response.read())
            request = Request(
                server.url + 'api/model/evaluate',
                data=json.dumps({'scene': scene}).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urlopen(request, timeout=2.0) as response:
                result = json.loads(response.read())
            control_request = Request(
                server.url + 'api/model/control',
                data=json.dumps({'command': 'start'}).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urlopen(control_request, timeout=2.0) as response:
                control = json.loads(response.read())
        finally:
            server.close()

        self.assertTrue(health['model_available'])
        self.assertTrue(health['model_continuous_available'])
        self.assertEqual('best.pt', health['model']['checkpoint'])
        self.assertEqual(7, result['action'])
        self.assertEqual(21, len(result['q_values']))
        self.assertTrue(control['running'])

    def test_merge_distance_http_api_exposes_read_only_evaluation(self):
        class FakeMergeDistanceEvaluator:
            identity = {
                'checkpoint': 'merge-distance.pt',
                'checkpoint_sha256': 'abcdef123456',
                'device': 'cpu',
                'class_count': 13,
            }

            @staticmethod
            def evaluate(scene, **_context):
                return {
                    'fruit_count': len(scene['fruits']),
                    'predictions': [{
                        'fruit_id': fruit['id'],
                        'label': '5–8 次',
                    } for fruit in scene['fruits']],
                }

        server = ScenarioLabServer(
            SimpleNamespace(device='cpu'),
            merge_distance_evaluator=FakeMergeDistanceEvaluator(),
            host='127.0.0.1',
            port=0,
        ).start()
        scene = {
            'fps': 120,
            'queue': [1, 2, 3, 4],
            'fruits': [{
                'id': 9, 'level': 2, 'x': 200.0, 'y': 800.0,
            }],
        }
        try:
            with urlopen(server.url + 'api/health', timeout=2.0) as response:
                health = json.loads(response.read())
            request = Request(
                server.url + 'api/merge-distance/evaluate',
                data=json.dumps({
                    'scene': scene,
                    'danger_progress': 0.25,
                    'over_danger_line': False,
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urlopen(request, timeout=2.0) as response:
                result = json.loads(response.read())
        finally:
            server.close()

        self.assertTrue(health['merge_distance_available'])
        self.assertEqual(
            'merge-distance.pt',
            health['merge_distance_model']['checkpoint'],
        )
        self.assertEqual(9, result['predictions'][0]['fruit_id'])

    def test_comparison_model_control_http_api(self):
        class FakeModelEvaluator:
            identity = {'checkpoint': 'best.pt'}

        class FakeController:
            def __init__(self):
                self.running = False

            def start_service(self):
                return self

            def close(self):
                return None

            def status(self):
                return {'running': self.running, 'decision_count': 0}

            def start(self):
                self.running = True
                return self.status()

            def stop(self, **_kwargs):
                self.running = False
                return self.status()

        comparison = ScenarioLabComparisonSession(
            preset='backend_parity', accelerated_device='cpu'
        )
        controller = FakeController()
        server = ScenarioLabServer(
            SimpleNamespace(device='cpu'),
            model_evaluator=FakeModelEvaluator(),
            comparison_session=comparison,
            comparison_model_controller=controller,
            host='127.0.0.1',
            port=0,
        ).start()
        try:
            request = Request(
                server.url + 'api/comparison/model/control',
                data=json.dumps({'command': 'start'}).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urlopen(request, timeout=2.0) as response:
                control = json.loads(response.read())
            with urlopen(
                    server.url + 'api/comparison/state', timeout=2.0
            ) as response:
                state = json.loads(response.read())
        finally:
            server.close()

        self.assertTrue(control['running'])
        self.assertTrue(state['model_continuous']['running'])

    def test_live_http_api_exposes_state_and_accepts_drop_command(self):
        server = ScenarioLabServer(
            SimpleNamespace(device='cpu'), host='127.0.0.1', port=0
        ).start()
        try:
            with urlopen(server.url + 'api/health', timeout=2.0) as response:
                health = json.loads(response.read())
            request = Request(
                server.url + 'api/live/command',
                data=json.dumps({
                    'command': {'type': 'drop', 'level': 1, 'x': 280.0}
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
                method='POST',
            )
            with urlopen(request, timeout=2.0) as response:
                accepted = json.loads(response.read())
            with urlopen(server.url + 'api/live/state', timeout=2.0) as response:
                state = json.loads(response.read())
        finally:
            server.close()

        self.assertTrue(health['live_physics'])
        self.assertEqual('tensor_cpu', health['live_physics_backend'])
        self.assertTrue(health['training_physics_equivalent'])
        self.assertTrue(accepted['accepted'])
        self.assertEqual('tensor_cpu', state['physics_backend'])
        self.assertEqual(1, state['step_count'])
        self.assertEqual(1, len(state['fruits']))

    def test_live_session_accepts_consecutive_drops_without_settling(self):
        session = ScenarioLabLiveSession(physics_fps=120, publish_fps=60)
        session.start()
        try:
            session.execute({'type': 'clear', 'queue': [1, 2, 3, 4]})
            first = session.execute({'type': 'drop', 'level': 1, 'x': 180})
            second = session.execute({'type': 'drop', 'level': 2, 'x': 380})
            snapshot = session.snapshot()
            sequence = snapshot['sequence']
            later = session.wait_for_snapshot(sequence, timeout=0.2)
        finally:
            session.close()

        self.assertTrue(first['accepted'])
        self.assertTrue(second['accepted'])
        self.assertEqual(2, snapshot['step_count'])
        self.assertEqual(2, len(snapshot['fruits']))
        self.assertGreater(later['physics_frame'], snapshot['physics_frame'])
        self.assertFalse(snapshot['paused'])

    def test_live_session_defaults_to_full_rate_publish(self):
        session = ScenarioLabLiveSession()
        snapshot = session.snapshot()

        self.assertEqual(120, snapshot['physics_fps'])
        self.assertEqual(120, snapshot['publish_fps'])

    def test_live_session_model_action_advances_natural_queue(self):
        session = ScenarioLabLiveSession(physics_fps=120, publish_fps=60)
        session.start()
        try:
            session.execute({'type': 'clear', 'queue': [1, 2, 3, 4]})
            dropped = session.execute({'type': 'drop_action', 'action': 10})
            snapshot = session.snapshot()
        finally:
            session.close()

        self.assertTrue(dropped['accepted'])
        self.assertEqual(1, dropped['dropped_level'])
        self.assertEqual([1, 2, 3, 4], dropped['queue_before'])
        self.assertEqual(2, dropped['queue_after'][0])
        self.assertEqual(dropped['queue_after'], snapshot['queue'])
        self.assertEqual(1, snapshot['step_count'])

    def test_model_controller_decides_after_continuous_stable_window(self):
        class FakeEvaluator:
            @staticmethod
            def evaluate(scene, **_context):
                values = [float(index) / 20.0 for index in range(21)]
                return {
                    'action': 10,
                    'drop_x': 280.0,
                    'selected_q': values[10],
                    'q_values': values,
                    'inference_ms': 0.1,
                    'model': {'checkpoint': 'fake.pt'},
                    'queue': list(scene['queue']),
                }

        session = ScenarioLabLiveSession(physics_fps=120, publish_fps=60)
        controller = ScenarioModelController(session, FakeEvaluator())
        session.start()
        controller.start_service()
        try:
            session.execute({'type': 'clear', 'queue': [1, 2, 3, 4]})
            controller.start()
            deadline = time.monotonic() + 2.0
            status = controller.status()
            while (
                    status['decision_count'] < 1
                    and time.monotonic() < deadline):
                time.sleep(0.02)
                status = controller.status()
            snapshot = session.snapshot()
            controller.stop()
        finally:
            controller.close()
            session.close()

        self.assertEqual(1, status['decision_count'])
        self.assertEqual(1, snapshot['step_count'])
        self.assertEqual(10, status['last_evaluation']['action'])
        self.assertEqual(2, snapshot['queue'][0])

    def test_live_session_removes_fruit_at_command_boundary(self):
        session = ScenarioLabLiveSession(physics_fps=120, publish_fps=60)
        session.start()
        try:
            dropped = session.execute({
                'type': 'drop', 'level': 1, 'x': 280,
            })
            removed = session.execute({
                'type': 'remove', 'fruit_id': dropped['fruit_id'],
            })
            snapshot = session.snapshot()
        finally:
            session.close()

        self.assertTrue(removed['accepted'])
        self.assertEqual(dropped['fruit_id'], removed['fruit_id'])
        self.assertEqual([], snapshot['fruits'])

    def test_removed_support_releases_upper_fruit(self):
        session = ScenarioLabLiveSession(physics_fps=120, publish_fps=60)
        session.start()
        try:
            scene = validate_scenario({
                'fps': 120,
                'queue': [1, 2, 3, 4],
                'fruits': [
                    {'id': 1, 'level': 3, 'x': 280.0, 'y': 1060.0},
                    {'id': 2, 'level': 4, 'x': 280.0, 'y': 971.0},
                ],
            })
            session.execute({
                'type': 'load_scene', 'scene': scene, 'paused': True,
            })
            removed = session.execute({'type': 'remove', 'fruit_id': 1})
            before = next(
                fruit for fruit in session.snapshot()['fruits']
                if fruit['id'] == 2
            )
            session.execute({'type': 'resume'})
            deadline = time.monotonic() + 0.3
            after = before
            while after['y'] <= before['y'] and time.monotonic() < deadline:
                snapshot = session.wait_for_snapshot(
                    session.snapshot()['sequence'], timeout=0.05
                )
                after = next(
                    fruit for fruit in snapshot['fruits']
                    if fruit['id'] == 2
                )
        finally:
            session.close()

        self.assertTrue(removed['accepted'])
        self.assertGreater(after['y'], before['y'])

    def test_live_session_pause_stops_physics_but_keeps_commands(self):
        session = ScenarioLabLiveSession(physics_fps=120, publish_fps=60)
        session.start()
        try:
            session.execute({'type': 'pause'})
            session.execute({'type': 'drop', 'level': 1, 'x': 280})
            paused = session.snapshot()
            time.sleep(0.04)
            still_paused = session.snapshot()
            session.execute({'type': 'resume'})
            moving = session.snapshot()
            deadline = time.monotonic() + 0.2
            while (
                    moving['physics_frame'] <= still_paused['physics_frame']
                    and time.monotonic() < deadline):
                moving = session.wait_for_snapshot(
                    moving['sequence'], timeout=0.05
                )
        finally:
            session.close()

        self.assertEqual(paused['physics_frame'], still_paused['physics_frame'])
        self.assertEqual(1, len(still_paused['fruits']))
        self.assertGreater(moving['physics_frame'], still_paused['physics_frame'])

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

    def test_cpu_evaluator_returns_real_results_for_all_actions(self):
        evaluator = ScenarioLabEvaluator(device='cpu')

        payload = evaluator.evaluate({
            'name': '空场动作检查',
            'fps': 30,
            'queue': [1, 2, 3, 4],
            'probe_action': 6,
            'fruits': [],
        }, mode='probe')

        self.assertEqual(4, payload['format_version'])
        self.assertEqual('spatial_v2_1', payload['reward_version'])
        self.assertEqual(21, len(payload['actions']))
        self.assertEqual(6, payload['selected_action'])
        self.assertEqual(3, len(payload['actions'][6]['space_slots']))
        self.assertIn('reference_loss', payload['actions'][6])
        self.assertIn('reference_action', payload['actions'][6])
        self.assertNotIn('compensation', payload['actions'][6])
        self.assertLessEqual(
            max(action['reward'] for action in payload['actions']), 1e-6
        )
        self.assertTrue(payload['actions'][6]['result_fruits'])
        effect = payload['actions'][6]['action_effect']
        self.assertEqual('floor', effect['first_contact']['primary'])
        self.assertTrue(effect['first_contact']['valid'])
        self.assertIn('q0', effect)
        self.assertIn('generations', effect)
        self.assertEqual(3, len(effect['generations']))
        self.assertIn('settle_duration_seconds', effect['outcome'])


if __name__ == '__main__':
    unittest.main()
