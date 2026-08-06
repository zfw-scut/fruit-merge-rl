"""使用标准库HTTP服务连接场景实验室前端和真实评估后端。"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread

from .scenario_lab import render_scenario_lab_html
from .scenario_lab_live import ScenarioLabLiveSession
from .scenario_lab_service import validate_scenario


class ScenarioLabServer:
    def __init__(
            self,
            evaluator,
            *,
            live_session=None,
            host='127.0.0.1',
            port=8769,
            title='合成大西瓜 · Reward V2场景实验室'):
        self.evaluator = evaluator
        self.live_session = live_session or ScenarioLabLiveSession()
        self.html = render_scenario_lab_html(title=title).encode('utf-8')
        self._closing = Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_arguments):
                return

            def _json(self, status, payload):
                body = json.dumps(
                    payload, ensure_ascii=False, separators=(',', ':')
                ).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(owner.html)))
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(owner.html)
                    return
                if self.path == '/api/health':
                    self._json(200, {
                        'ready': True,
                        'reward_version': 'spatial_v2',
                        'device': str(owner.evaluator.device),
                        'live_physics': True,
                    })
                    return
                if self.path == '/api/live/state':
                    self._json(200, owner.live_session.snapshot())
                    return
                if self.path == '/api/live/events':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Connection', 'keep-alive')
                    self.end_headers()
                    sequence = -1
                    try:
                        while not owner._closing.is_set():
                            payload = owner.live_session.wait_for_snapshot(
                                sequence, timeout=5.0
                            )
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
                if self.path not in ('/api/evaluate', '/api/live/command'):
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
                        if command.get('type') == 'load_scene':
                            command = dict(command)
                            command['scene'] = validate_scenario(
                                command.get('scene')
                            )
                        payload = owner.live_session.execute(command)
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

    def start(self):
        if self._thread is None:
            self._closing.clear()
            self.live_session.start()
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
        self.httpd.serve_forever()

    def close(self):
        self._closing.set()
        if self._thread is not None:
            self.httpd.shutdown()
        self.httpd.server_close()
        self.live_session.close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


__all__ = ['ScenarioLabServer']
