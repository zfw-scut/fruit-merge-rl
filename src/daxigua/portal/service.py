"""只绑定回环地址的文档索引、工具白名单和训练面板代理。"""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = PROJECT_ROOT / 'docs'
RUNTIME_ROOT = PROJECT_ROOT / 'runs' / 'portal_processes'


TOOL_DEFINITIONS: dict[str, dict[str, Any]] = {
    'scenario_lab': {
        'id': 'scenario_lab',
        'name': '实时场景实验室',
        'eyebrow': 'PHYSICS LAB',
        'description': '编辑水果局面，实时比较21个动作的模型预测与真实物理结果。',
        'accent': 'cyan',
        'kind': 'service',
        'primary_action': '启动实验室',
        'parameters': [
            {'id': 'device', 'label': '物理设备', 'type': 'select', 'default': 'cuda',
             'options': ['cuda', 'cpu']},
            {'id': 'model_device', 'label': '模型设备', 'type': 'select', 'default': 'auto',
             'options': ['auto', 'cuda', 'cpu']},
            {'id': 'port', 'label': '服务端口', 'type': 'number', 'default': 8769,
             'min': 1024, 'max': 65535, 'step': 1},
            {'id': 'reward_scale', 'label': '空间奖励缩放', 'type': 'range',
             'default': 1.0, 'min': 0.1, 'max': 2.0, 'step': 0.1},
            {'id': 'checkpoint', 'label': '模型 Checkpoint', 'type': 'checkpoint',
             'default': '', 'optional': True},
        ],
    },
    'training_dashboard': {
        'id': 'training_dashboard',
        'name': '历史训练面板',
        'eyebrow': 'RUN MONITOR',
        'description': '重新打开已经完成或迁回本地的训练run，查看曲线、评估和GPU资源。',
        'accent': 'violet',
        'kind': 'service',
        'primary_action': '启动只读面板',
        'parameters': [
            {'id': 'run_dir', 'label': '训练 Run', 'type': 'run', 'default': ''},
            {'id': 'port', 'label': '服务端口', 'type': 'number', 'default': 8765,
             'min': 1024, 'max': 65535, 'step': 1},
        ],
    },
    'model_viewer': {
        'id': 'model_viewer',
        'name': '模型对局观看器',
        'eyebrow': 'MODEL REPLAY',
        'description': '让指定checkpoint完成真实物理对局，并生成可逐帧查看Q值的HTML页面。',
        'accent': 'amber',
        'kind': 'task',
        'primary_action': '生成观看页面',
        'parameters': [
            {'id': 'checkpoint', 'label': '模型 Checkpoint', 'type': 'checkpoint',
             'default': ''},
            {'id': 'device', 'label': '推理设备', 'type': 'select', 'default': 'auto',
             'options': ['auto', 'cuda', 'cpu']},
            {'id': 'physics_fps', 'label': '物理帧率', 'type': 'segmented',
             'default': 120, 'options': [30, 120]},
            {'id': 'episodes', 'label': '生成局数', 'type': 'range', 'default': 1,
             'min': 1, 'max': 8, 'step': 1},
            {'id': 'max_drops', 'label': '最大投放', 'type': 'number', 'default': 1000,
             'min': 16, 'max': 2000, 'step': 16},
        ],
    },
    'training_preflight': {
        'id': 'training_preflight',
        'name': 'CUDA训练门禁',
        'eyebrow': 'SAFETY GATE',
        'description': '执行短rollout、联合反向传播、checkpoint往返和双帧率短评估。',
        'accent': 'rose',
        'kind': 'task',
        'primary_action': '执行训练门禁',
        'confirmation': '该操作会占用GPU并写入runs/preflight，确认使用当前参数执行？',
        'parameters': [
            {'id': 'config', 'label': '训练配置', 'type': 'config',
             'default': 'configs/gnn_dqn_auxiliary_action_l5_128m.toml'},
            {'id': 'device', 'label': '计算设备', 'type': 'select', 'default': 'cuda',
             'options': ['cuda', 'cpu']},
            {'id': 'smoke_envs', 'label': '门禁环境数', 'type': 'range', 'default': 32,
             'min': 8, 'max': 512, 'step': 8},
            {'id': 'evaluation_episodes', 'label': '短评估局数', 'type': 'range',
             'default': 8, 'min': 2, 'max': 64, 'step': 2},
        ],
    },
}


def _title_from_markdown(path: Path, content: str) -> str:
    for line in content.splitlines():
        if line.startswith('# '):
            return line[2:].strip()
    return path.stem


def _document_category(relative: str) -> tuple[str, str]:
    if relative.startswith('docs/model_evaluations/'):
        return 'evaluations', '模型评估'
    if relative.startswith('docs/model/'):
        return 'model', '模型设计'
    if relative.startswith('docs/codex/'):
        return 'codex', '开发记录'
    return 'guide', '项目指南'


def _strip_markdown(value: str) -> str:
    value = re.sub(r'```.*?```', ' ', value, flags=re.S)
    value = re.sub(r'!\[[^]]*]\([^)]*\)', ' ', value)
    value = re.sub(r'\[([^]]+)]\([^)]*\)', r'\1', value)
    value = re.sub(r'[`*_>#|~-]+', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()


def scan_documents(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    docs_root = project_root / 'docs'
    documents: list[dict[str, Any]] = []
    for path in sorted(docs_root.rglob('*.md')):
        content = path.read_text(encoding='utf-8')
        relative = path.relative_to(project_root).as_posix()
        category, category_label = _document_category(relative)
        plain = _strip_markdown(content)
        stat = path.stat()
        documents.append({
            'id': relative,
            'path': relative,
            'title': _title_from_markdown(path, content),
            'category': category,
            'category_label': category_label,
            'content': content,
            'excerpt': plain[:180],
            'search_text': plain,
            'modified_at': stat.st_mtime,
            'word_count': len(plain),
            'is_evidence': category == 'evaluations',
            'is_history': category == 'codex',
        })
    return documents


def document_revision(project_root: Path = PROJECT_ROOT) -> str:
    """返回轻量变更标识，供前端在不重复传输正文时检测文档更新。"""
    paths = list((project_root / 'docs').rglob('*.md'))
    latest = max((path.stat().st_mtime_ns for path in paths), default=0)
    return f'{len(paths)}:{latest}'


def _inside(base: Path, raw: str, *, must_exist: bool = True) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    base = base.resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f'路径必须位于 {base}')
    if must_exist and not candidate.exists():
        raise ValueError(f'路径不存在：{candidate}')
    return candidate


def _int_value(params: dict[str, Any], key: str, low: int, high: int) -> int:
    value = int(params[key])
    if not low <= value <= high:
        raise ValueError(f'{key} 必须位于 {low}～{high}')
    return value


def _float_value(params: dict[str, Any], key: str, low: float, high: float) -> float:
    value = float(params[key])
    if not low <= value <= high:
        raise ValueError(f'{key} 必须位于 {low:g}～{high:g}')
    return value


def _choice(params: dict[str, Any], key: str, options: tuple[Any, ...]) -> Any:
    value = params[key]
    if value not in options:
        raise ValueError(f'{key} 只能是 {options}')
    return value


def build_tool_command(tool_id: str, params: dict[str, Any]) -> tuple[list[str], str | None]:
    python = sys.executable
    if tool_id == 'scenario_lab':
        port = _int_value(params, 'port', 1024, 65535)
        device = _choice(params, 'device', ('cuda', 'cpu'))
        model_device = _choice(params, 'model_device', ('auto', 'cuda', 'cpu'))
        reward_scale = _float_value(params, 'reward_scale', 0.1, 2.0)
        command = [python, 'tools/open_scenario_lab.py', '--serve', '--host',
                   '127.0.0.1', '--port', str(port), '--device', device,
                   '--model-device', model_device, '--reward-scale', str(reward_scale)]
        checkpoint = str(params.get('checkpoint') or '').strip()
        if checkpoint:
            command.extend(['--checkpoint', str(_inside(PROJECT_ROOT, checkpoint))])
        return command, f'http://127.0.0.1:{port}/'
    if tool_id == 'training_dashboard':
        port = _int_value(params, 'port', 1024, 65535)
        run_dir = _inside(PROJECT_ROOT / 'runs', str(params.get('run_dir') or ''))
        command = [python, 'tools/serve_training_dashboard.py', '--run-dir',
                   str(run_dir), '--host', '127.0.0.1', '--port', str(port)]
        return command, f'http://127.0.0.1:{port}/'
    if tool_id == 'model_viewer':
        checkpoint = _inside(PROJECT_ROOT, str(params.get('checkpoint') or ''))
        device = _choice(params, 'device', ('auto', 'cuda', 'cpu'))
        fps = int(_choice(params, 'physics_fps', (30, 120)))
        episodes = _int_value(params, 'episodes', 1, 8)
        max_drops = _int_value(params, 'max_drops', 16, 2000)
        command = [python, 'tools/watch_gnn_dqn.py', str(checkpoint), '--device',
                   device, '--physics-fps', str(fps), '--episodes', str(episodes),
                   '--max-drops', str(max_drops), '--open']
        return command, None
    if tool_id == 'training_preflight':
        config = _inside(PROJECT_ROOT / 'configs', str(params.get('config') or ''))
        device = _choice(params, 'device', ('cuda', 'cpu'))
        smoke_envs = _int_value(params, 'smoke_envs', 8, 512)
        episodes = _int_value(params, 'evaluation_episodes', 2, 64)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        output = PROJECT_ROOT / 'runs' / 'preflight' / f'portal_{stamp}.json'
        command = [python, 'tools/preflight_training.py', '--config', str(config),
                   '--device', device, '--smoke-envs', str(smoke_envs),
                   '--evaluation-episodes', str(episodes), '--output', str(output)]
        return command, None
    raise ValueError(f'未知工具：{tool_id}')


def _choices() -> dict[str, list[dict[str, str]]]:
    runs: list[dict[str, str]] = []
    runs_root = PROJECT_ROOT / 'runs'
    if runs_root.exists():
        candidates = [path for path in runs_root.iterdir() if path.is_dir()]
        for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[:80]:
            if any((path / name).exists() for name in ('run_status.json', 'metrics.jsonl', 'run_identity.json')):
                runs.append({'label': path.name, 'value': path.relative_to(PROJECT_ROOT).as_posix()})
    checkpoints: list[dict[str, str]] = []
    if runs_root.exists():
        candidates = sorted(runs_root.glob('**/checkpoints/*.pt'),
                            key=lambda item: item.stat().st_mtime, reverse=True)[:120]
        for path in candidates:
            checkpoints.append({'label': f'{path.parent.parent.name} / {path.name}',
                                'value': path.relative_to(PROJECT_ROOT).as_posix()})
    configs = [
        {'label': path.name, 'value': path.relative_to(PROJECT_ROOT).as_posix()}
        for path in sorted((PROJECT_ROOT / 'configs').glob('*.toml'))
    ]
    return {'runs': runs, 'checkpoints': checkpoints, 'configs': configs}


@dataclass
class ManagedProcess:
    tool_id: str
    process: subprocess.Popen[str]
    log_path: Path
    started_at: float
    url: str | None
    command: list[str]

    def snapshot(self) -> dict[str, Any]:
        exit_code = self.process.poll()
        return {
            'tool_id': self.tool_id,
            'pid': self.process.pid,
            'running': exit_code is None,
            'exit_code': exit_code,
            'started_at': self.started_at,
            'url': self.url,
            'command_preview': subprocess.list2cmdline(self.command),
            'log_tail': self._tail(),
        }

    def _tail(self, lines: int = 24) -> list[str]:
        if not self.log_path.exists():
            return []
        return self.log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:]


class ProcessRegistry:
    def __init__(self):
        self._items: dict[str, ManagedProcess] = {}
        self._lock = threading.Lock()

    def start(self, tool_id: str, params: dict[str, Any]) -> dict[str, Any]:
        if tool_id not in TOOL_DEFINITIONS:
            raise ValueError(f'未知工具：{tool_id}')
        command, url = build_tool_command(tool_id, params)
        with self._lock:
            current = self._items.get(tool_id)
            if current is not None and current.process.poll() is None:
                raise RuntimeError('该工具已经在运行')
            RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
            log_path = RUNTIME_ROOT / f'{tool_id}_{int(time.time())}.log'
            log_handle = log_path.open('w', encoding='utf-8')
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=flags,
            )
            log_handle.close()
            managed = ManagedProcess(tool_id, process, log_path, time.time(), url, command)
            self._items[tool_id] = managed
            return managed.snapshot()

    def stop(self, tool_id: str) -> dict[str, Any]:
        with self._lock:
            managed = self._items.get(tool_id)
            if managed is None:
                raise ValueError('该工具没有门户启动记录')
            if managed.process.poll() is None:
                managed.process.terminate()
                try:
                    managed.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    managed.process.kill()
            return managed.snapshot()

    def snapshots(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: item.snapshot() for key, item in self._items.items()}


class PortalServer:
    def __init__(self, host: str = '127.0.0.1', port: int = 4312):
        if host not in ('127.0.0.1', 'localhost'):
            raise ValueError('门户控制API只允许绑定回环地址')
        registry = ProcessRegistry()

        class Handler(BaseHTTPRequestHandler):
            server_version = 'DaxiguaPortal/1.0'

            def _origin(self) -> str | None:
                origin = self.headers.get('Origin')
                if origin is None:
                    return None
                match = re.fullmatch(
                    r'http://(?:localhost|127\.0\.0\.1):(\d{1,5})', origin
                )
                return origin if match and int(match.group(1)) <= 65535 else None

            def _send(self, status: int, payload: Any):
                body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
                self.send_response(status)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                origin = self._origin()
                if origin:
                    self.send_header('Access-Control-Allow-Origin', origin)
                    self.send_header('Vary', 'Origin')
                self.end_headers()
                self.wfile.write(body)

            def _send_file(self, path: Path):
                body = path.read_bytes()
                content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                origin = self._origin()
                if origin:
                    self.send_header('Access-Control-Allow-Origin', origin)
                    self.send_header('Vary', 'Origin')
                self.end_headers()
                self.wfile.write(body)

            def _payload(self) -> dict[str, Any]:
                length = int(self.headers.get('Content-Length', '0'))
                if length > 1_000_000:
                    raise ValueError('请求体过大')
                return json.loads(self.rfile.read(length) or b'{}')

            def do_OPTIONS(self):
                self.send_response(204)
                origin = self._origin()
                if origin:
                    self.send_header('Access-Control-Allow-Origin', origin)
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    self.send_header('Vary', 'Origin')
                self.end_headers()

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                try:
                    if path == '/api/health':
                        self._send(200, {'ok': True, 'documents': len(scan_documents()),
                                         'project_root': str(PROJECT_ROOT)})
                    elif path == '/api/documents':
                        self._send(200, {'documents': scan_documents(),
                                         'revision': document_revision(),
                                         'timestamp': time.time()})
                    elif path == '/api/documents/revision':
                        self._send(200, {'revision': document_revision()})
                    elif path == '/api/file':
                        raw_path = parse_qs(parsed.query).get('path', [''])[0]
                        asset = _inside(DOCS_ROOT, unquote(raw_path))
                        if asset.is_dir():
                            raise ValueError('不能读取目录')
                        self._send_file(asset)
                    elif path == '/api/tools':
                        tools = []
                        snapshots = registry.snapshots()
                        for tool_id, definition in TOOL_DEFINITIONS.items():
                            tools.append({**definition, 'process': snapshots.get(tool_id)})
                        self._send(200, {'tools': tools, 'choices': _choices()})
                    elif path == '/api/dashboard/status':
                        try:
                            with urlopen('http://127.0.0.1:8765/api/status', timeout=1.5) as response:
                                payload = json.loads(response.read().decode('utf-8'))
                            self._send(200, {'available': True, 'payload': payload})
                        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
                            self._send(200, {'available': False, 'payload': None})
                    else:
                        self._send(404, {'error': '接口不存在'})
                except Exception as exc:
                    self._send(500, {'error': f'{type(exc).__name__}: {exc}'})

            def do_POST(self):
                path = unquote(urlparse(self.path).path)
                match = re.fullmatch(r'/api/tools/([a-z_]+)/(start|stop)', path)
                if not match:
                    self._send(404, {'error': '接口不存在'})
                    return
                tool_id, action = match.groups()
                try:
                    result = (registry.start(tool_id, self._payload().get('params', {}))
                              if action == 'start' else registry.stop(tool_id))
                    self._send(200, {'process': result})
                except (ValueError, RuntimeError, KeyError) as exc:
                    self._send(400, {'error': str(exc)})

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer((host, int(port)), Handler)
        self.host = host
        self.port = self._server.server_port

    @property
    def url(self) -> str:
        return f'http://{self.host}:{self.port}'

    def serve_forever(self):
        self._server.serve_forever(poll_interval=0.25)

    def shutdown(self):
        self._server.shutdown()
        self._server.server_close()
