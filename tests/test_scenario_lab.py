import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

from daxigua.simulator.scenario_lab import fruit_specs, write_scenario_lab_html
from daxigua.simulator.scenario_lab_service import (
    ScenarioLabEvaluator,
    validate_scenario,
)
from daxigua.simulator.scenario_lab_web import render_scenario_lab_document
from daxigua.simulator.scenario_lab_live import ScenarioLabLiveSession
from daxigua.simulator.scenario_lab_server import ScenarioLabServer
from daxigua.rl.scenario_model_controller import ScenarioModelController


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
        self.assertIn('/api/live/command', html)
        self.assertIn('/api/live/events', html)
        self.assertIn('/api/model/evaluate', html)
        self.assertIn('/api/model/control', html)
        self.assertIn('评估当前场景的模型倾向', html)
        self.assertIn('启动模型持续决策', html)
        self.assertIn('model-action-results', html)
        self.assertIn('new EventSource', html)
        self.assertIn('dropPreviews', html)
        self.assertIn('interpolateLiveFruits', html)
        self.assertIn('requestAnimationFrame(renderLiveFrame)', html)
        self.assertIn("{type:'remove',fruit_id:fruitId}", html)
        self.assertIn('beginTransientEdit', html)
        self.assertIn('finishTransientEdit', html)
        self.assertIn('按住编辑 · 松手恢复', html)
        self.assertIn("await pushLiveScene(true);await sendLiveCommand({type:'resume'})", html)
        self.assertIn(
            "button.disabled=busy||viewLocked||(!editable&&button.dataset.tool==='erase')",
            html,
        )
        self.assertIn('物理 ${state.physicsFps||120} FPS · 显示同步', html)
        self.assertIn('暂停并进入编辑', html)
        self.assertIn('effective_normalized_area', html)
        self.assertIn('data-evaluation-phase="before"', html)
        self.assertIn('data-evaluation-phase="after"', html)
        self.assertIn('evaluationScene', html)
        self.assertIn('evaluationViewFruits', html)
        self.assertIn('result_fruits', html)
        self.assertIn('applyEvaluation(payload,detail.scene)', html)
        self.assertIn("if(activeSideTab!=='result')return", html)
        self.assertNotIn(
            "if(backendConnected&&!state.livePaused)return;const action=activeResult()",
            html,
        )
        self.assertNotIn('回放轨迹', html)
        self.assertNotIn('文件(F)', html)

    def test_fruit_specs_follow_all_stable_levels(self):
        specs = fruit_specs()

        self.assertEqual(11, len(specs))
        self.assertEqual('葡萄', specs[0]['name'])
        self.assertEqual('大西瓜', specs[-1]['name'])
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
        self.assertTrue(accepted['accepted'])
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

        self.assertEqual(3, payload['format_version'])
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


if __name__ == '__main__':
    unittest.main()
