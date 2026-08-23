"""使用标准库HTTP服务连接场景实验室前端和真实评估后端。"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from threading import Event, Thread

from .replay import load_fruit_texture_data_urls
from .scenario_lab import fruit_specs
from .scenario_lab_live import ScenarioLabLiveSession
from .scenario_lab_service import validate_scenario
from .voronoi import ScenarioVoronoiEvaluator


class ScenarioLabServer:
    def __init__(
            self,
            evaluator,
            *,
            model_evaluator=None,
            model_controller=None,
            comparison_model_controller=None,
            voronoi_evaluator=None,
            live_session=None,
            comparison_session=None,
            host='127.0.0.1',
            port=8769,
            title='合成大西瓜 · Reward V2.1场景实验室'):
        self.evaluator = evaluator
        self.model_evaluator = model_evaluator
        self.model_controller = model_controller
        self.comparison_model_controller = comparison_model_controller
        self.live_session = live_session or ScenarioLabLiveSession(
            device=evaluator.device
        )
        self.voronoi_evaluator = (
            voronoi_evaluator or ScenarioVoronoiEvaluator(
                self.live_session.config,
                device=self.live_session.device,
            )
        )
        self.comparison_session = comparison_session
        geometry = self.live_session.config
        self.ui_config = {
            'format_version': 1,
            'title': title,
            'fruit_specs': fruit_specs(),
            'textures': load_fruit_texture_data_urls(),
            'geometry': {
                'board_width': geometry.board_width,
                'board_height': geometry.board_height,
                'wall_width': geometry.wall_width,
                'spawn_y': geometry.spawn_y,
                'action_count': geometry.action_count,
                'queue_length': geometry.queue_length,
                'max_fruits': geometry.max_fruits,
            },
            'voronoi': {
                'algorithm': 'full_disk_weighted_voronoi_raster_v1',
                'sample_spacing': (
                    self.voronoi_evaluator.builder.sample_spacing
                ),
                'top_boundary': 'open',
                'obstacle_boundaries': [
                    'left_wall', 'right_wall', 'floor'
                ],
            },
        }
        self._closing = Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_arguments):
                return

            def _origin(self):
                origin = self.headers.get('Origin')
                if origin is None:
                    return None
                return origin if re.fullmatch(
                    r'http://(?:localhost|127\.0\.0\.1):\d{1,5}', origin
                ) else None

            def _cors(self):
                origin = self._origin()
                if origin:
                    self.send_header('Access-Control-Allow-Origin', origin)
                    self.send_header('Vary', 'Origin')

            def _json(self, status, payload):
                body = json.dumps(
                    payload, ensure_ascii=False, separators=(',', ':')
                ).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self._cors()
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                self.send_response(204)
                self._cors()
                self.send_header(
                    'Access-Control-Allow-Methods', 'GET, POST, OPTIONS'
                )
                self.send_header(
                    'Access-Control-Allow-Headers', 'Content-Type'
                )
                self.end_headers()

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self._json(410, {
                        'error': 'standalone_scenario_lab_removed',
                        'message': '场景实验室界面已迁入 Xigua Atlas 项目门户',
                        'portal': 'http://127.0.0.1:3000/#lab',
                        'health': '/api/health',
                    })
                    return
                if self.path == '/api/health':
                    model_identity = (
                        owner.model_evaluator.identity
                        if owner.model_evaluator is not None else None
                    )
                    self._json(200, {
                        'ready': True,
                        'reward_version': 'spatial_v2_1',
                        'device': str(owner.evaluator.device),
                        'live_physics': True,
                        'live_physics_backend': owner.live_session.backend,
                        'live_physics_device': str(owner.live_session.device),
                        'training_physics_equivalent': True,
                        'voronoi_available': True,
                        'voronoi_device': str(
                            owner.voronoi_evaluator.device
                        ),
                        'comparison_available': (
                            owner.comparison_session is not None
                        ),
                        'model_available': model_identity is not None,
                        'model': model_identity,
                        'model_continuous_available': (
                            owner.model_controller is not None
                        ),
                        'comparison_model_continuous_available': (
                            owner.comparison_model_controller is not None
                        ),
                    })
                    return
                if self.path == '/api/config':
                    self._json(200, owner.ui_config)
                    return
                if self.path == '/api/live/state':
                    self._json(200, owner._live_payload(
                        owner.live_session.snapshot()
                    ))
                    return
                if self.path == '/api/live/events':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Connection', 'keep-alive')
                    self._cors()
                    self.end_headers()
                    sequence = -1
                    try:
                        while not owner._closing.is_set():
                            payload = owner.live_session.wait_for_snapshot(
                                sequence, timeout=5.0
                            )
                            payload = owner._live_payload(payload)
                            next_sequence = int(payload['sequence'])
                            if next_sequence == sequence:
                                self.wfile.write(b': keep-alive\n\n')
                            else:
                                body = json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                    separators=(',', ':'),
                                ).encode('utf-8')
                                self.wfile.write(b'data: ' + body + b'\n\n')
                                sequence = next_sequence
                            self.wfile.flush()
                    except (
                            BrokenPipeError,
                            ConnectionAbortedError,
                            ConnectionResetError):
                        pass
                    return
                if self.path == '/api/comparison/state':
                    if owner.comparison_session is None:
                        self._json(404, {
                            'error': '双环境物理对照尚未启用',
                        })
                        return
                    self._json(
                        200, owner._comparison_payload(
                            owner.comparison_session.snapshot()
                        )
                    )
                    return
                if self.path == '/api/comparison/events':
                    if owner.comparison_session is None:
                        self._json(404, {
                            'error': '双环境物理对照尚未启用',
                        })
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Connection', 'keep-alive')
                    self._cors()
                    self.end_headers()
                    sequence = -1
                    try:
                        while not owner._closing.is_set():
                            payload = (
                                owner.comparison_session.wait_for_snapshot(
                                    sequence, timeout=5.0
                                )
                            )
                            payload = owner._comparison_payload(payload)
                            next_sequence = int(payload['sequence'])
                            if next_sequence == sequence:
                                self.wfile.write(b': keep-alive\n\n')
                            else:
                                body = json.dumps(
                                    payload,
                                    ensure_ascii=False,
                                    separators=(',', ':'),
                                ).encode('utf-8')
                                self.wfile.write(b'data: ' + body + b'\n\n')
                                sequence = next_sequence
                            self.wfile.flush()
                    except (
                            BrokenPipeError,
                            ConnectionAbortedError,
                            ConnectionResetError):
                        pass
                    return
                self._json(404, {'error': '路径不存在'})

            def do_POST(self):
                if self.path not in (
                        '/api/evaluate',
                        '/api/voronoi/evaluate',
                        '/api/model/evaluate',
                        '/api/model/control',
                        '/api/comparison/model/control',
                        '/api/live/command',
                        '/api/comparison/command'):
                    self._json(404, {'error': '路径不存在'})
                    return
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if length <= 0 or length > 2 * 1024 * 1024:
                        raise ValueError('请求体大小无效')
                    request = json.loads(self.rfile.read(length))
                    if self.path == '/api/live/command':
                        command = request.get('command')
                        if not isinstance(command, dict):
                            raise TypeError('command must be an object')
                        if (
                                owner.model_controller is not None
                                and owner.model_controller.running
                                and command.get('type') in (
                                    'drop', 'drop_action', 'remove',
                                    'clear', 'load_scene')):
                            owner.model_controller.stop(
                                reason='manual_control'
                            )
                        if command.get('type') == 'load_scene':
                            command = dict(command)
                            command['scene'] = validate_scenario(
                                command.get('scene')
                            )
                        payload = owner.live_session.execute(command)
                    elif self.path == '/api/comparison/command':
                        if owner.comparison_session is None:
                            raise RuntimeError('双环境物理对照尚未启用')
                        command = request.get('command')
                        if not isinstance(command, dict):
                            raise TypeError('command must be an object')
                        if (
                                owner.comparison_model_controller is not None
                                and owner.comparison_model_controller.running
                                and command.get('type') != 'refresh_state'):
                            owner.comparison_model_controller.stop(
                                reason='manual_control'
                            )
                        if command.get('type') == 'load_scene':
                            command = dict(command)
                            command['scene'] = validate_scenario(
                                command.get('scene')
                            )
                        payload = owner.comparison_session.execute(command)
                    elif self.path == '/api/comparison/model/control':
                        controller = owner.comparison_model_controller
                        if controller is None:
                            raise RuntimeError(
                                '双环境模型持续决策控制器尚未加载'
                            )
                        command = request.get('command')
                        if command == 'start':
                            payload = controller.start()
                        elif command == 'stop':
                            payload = controller.stop()
                        else:
                            raise ValueError(
                                'comparison model control command must be '
                                'start or stop'
                            )
                    elif self.path == '/api/model/control':
                        if owner.model_controller is None:
                            raise RuntimeError('模型持续决策控制器尚未加载')
                        command = request.get('command')
                        if command == 'start':
                            payload = owner.model_controller.start()
                        elif command == 'stop':
                            payload = owner.model_controller.stop()
                        else:
                            raise ValueError(
                                'model control command must be start or stop'
                            )
                    elif self.path == '/api/model/evaluate':
                        if owner.model_evaluator is None:
                            raise RuntimeError('模型 checkpoint 尚未加载')
                        payload = owner.model_evaluator.evaluate(
                            request.get('scene')
                        )
                    elif self.path == '/api/voronoi/evaluate':
                        payload = owner.voronoi_evaluator.evaluate(
                            request.get('scene')
                        )
                    else:
                        payload = owner.evaluator.evaluate(
                            request.get('scene'),
                            mode=request.get('mode', 'all'),
                        )
                except (TypeError, ValueError, KeyError) as error:
                    self._json(400, {'error': str(error)})
                    return
                except Exception as error:
                    self._json(500, {
                        'error': str(error),
                        'error_type': type(error).__name__,
                    })
                    return
                self._json(200, payload)

        self.httpd = ThreadingHTTPServer((host, int(port)), Handler)
        self.httpd.daemon_threads = True
        self.host, self.port = self.httpd.server_address
        self._thread = None

    @property
    def url(self):
        return f'http://{self.host}:{self.port}/'

    def _live_payload(self, payload):
        if self.model_controller is not None:
            payload['model_continuous'] = self.model_controller.status()
        return payload

    def _comparison_payload(self, payload):
        if self.comparison_model_controller is not None:
            payload['model_continuous'] = (
                self.comparison_model_controller.status()
            )
        return payload

    def start(self):
        if self._thread is None:
            self._closing.clear()
            self.live_session.start()
            if self.comparison_session is not None:
                self.comparison_session.start()
            if self.model_controller is not None:
                self.model_controller.start_service()
            if self.comparison_model_controller is not None:
                self.comparison_model_controller.start_service()
            self._thread = Thread(
                target=self.httpd.serve_forever,
                name='scenario-lab-server',
                daemon=True,
            )
            self._thread.start()
        return self

    def serve_forever(self):
        self._closing.clear()
        self.live_session.start()
        if self.comparison_session is not None:
            self.comparison_session.start()
        if self.model_controller is not None:
            self.model_controller.start_service()
        if self.comparison_model_controller is not None:
            self.comparison_model_controller.start_service()
        self.httpd.serve_forever()

    def close(self):
        self._closing.set()
        if self._thread is not None:
            self.httpd.shutdown()
        self.httpd.server_close()
        if self.model_controller is not None:
            self.model_controller.close()
        if self.comparison_model_controller is not None:
            self.comparison_model_controller.close()
        self.live_session.close()
        if self.comparison_session is not None:
            self.comparison_session.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


__all__ = ['ScenarioLabServer']
