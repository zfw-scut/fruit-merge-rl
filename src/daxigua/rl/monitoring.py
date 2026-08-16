"""与训练热路径隔离的资源监控和只读 Web 面板。"""

from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import os
from pathlib import Path
import queue
import threading
import time
import math

from .curves import (
    CURVE_FILENAME,
    existing_curve_metadata,
    render_training_curve_snapshot,
)
from .event_analysis import EVENT_ANALYSIS_FILENAME
from .merge_distance_status import scan_merge_distance_runs
from .merge_potential_status import scan_merge_potential_runs
from .training_queue import load_training_queue


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True





def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class ResourceSampler:
    def __init__(self, target_pid):
        self.target_pid = int(target_pid)
        self.psutil = None
        self.process = None
        self.nvml = None
        self.gpu_handle = None
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process(self.target_pid)
            psutil.cpu_percent(interval=None)
        except (ImportError, OSError):
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except (ImportError, OSError, RuntimeError):
            pass

    def sample(self):
        result = {}
        if self.psutil is not None:
            memory = self.psutil.virtual_memory()
            result.update({
                'cpu_utilization': self.psutil.cpu_percent(interval=None),
                'memory_used_mb': memory.used / 1024 ** 2,
                'memory_total_mb': memory.total / 1024 ** 2,
            })
            if self.process is not None:
                try:
                    result['process_rss_mb'] = (
                        self.process.memory_info().rss / 1024 ** 2
                    )
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    pass
        if self.nvml is not None and self.gpu_handle is not None:
            try:
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(
                    self.gpu_handle
                )
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                result.update({
                    'gpu_utilization': float(utilization.gpu),
                    'gpu_memory_used_mb': memory.used / 1024 ** 2,
                    'gpu_memory_total_mb': memory.total / 1024 ** 2,
                    'gpu_temperature': float(
                        self.nvml.nvmlDeviceGetTemperature(
                            self.gpu_handle,
                            self.nvml.NVML_TEMPERATURE_GPU,
                        )
                    ),
                    'gpu_power_watts': (
                        self.nvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                        / 1000.0
                    ),
                })
            except self.nvml.NVMLError:
                pass
        return result

    def close(self):
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except self.nvml.NVMLError:
                pass


class _DashboardState:
    def __init__(self, history_size):
        self.lock = threading.Lock()
        self.training = {}
        self.resources = {}
        self.plots = {}
        self.events = deque(maxlen=100)
        self.history = deque(maxlen=history_size)
        self.resource_history = deque(maxlen=history_size)
        self.timestamp = time.time()

    def update_training(self, payload):
        with self.lock:
            self.training.update(payload)
            self.timestamp = time.time()
            if 'env_steps_per_second' in payload:
                history_entry = {
                    'timestamp': self.timestamp,
                    'transitions': payload.get(
                        'transitions', self.training.get('transitions')
                    ),
                    'env_steps_per_second': payload.get(
                        'env_steps_per_second', 0.0
                    ),
                    'updates_per_second': payload.get(
                        'updates_per_second', 0.0
                    ),
                }
                optional_names = (
                    'learner_samples_per_second',
                    'branch_steps_per_second',
                    'loss',
                    'dqn_loss',
                    'mean_abs_td_error',
                    'aux_loss_merge',
                    'aux_loss_q0_lineage',
                    'aux_loss_first_contact',
                    'aux_loss_generation',
                    'aux_loss_outcome',
                    'branch_dqn_loss',
                    'branch_aux_loss_total',
                    'branch_sample_fraction',
                    'training_window_mean_score',
                    'training_window_max_score',
                    'training_rolling_mean_score',
                    'training_rolling_max_score',
                    'best_training_score',
                    'last_fast_eval_score',
                    'last_accurate_eval_score',
                    'epsilon_explore_action_fraction',
                    'active_learning_action_fraction',
                    'active_learning_effective_action_fraction',
                    'active_learning_greedy_overlap_rate',
                    'active_selected_rank_correlation',
                    'eval_created_l7_per_1000',
                    'eval_created_l8_per_1000',
                    'eval_created_l9_per_1000',
                    'eval_created_l10_per_1000',
                    'eval_created_l11_per_1000',
                    'active_episode_drops_mean',
                    'active_episode_drops_p50',
                    'active_episode_drops_p90',
                    'active_episode_drops_p99',
                    'active_episode_drops_max',
                    'long_episode_fraction_1000',
                    'long_episode_fraction_2000',
                    'long_episode_fraction_5000',
                    'long_episode_fraction_10000',
                    'active_fruit_count_max',
                )
                for name in optional_names:
                    if name in payload:
                        history_entry[name] = payload[name]
                self.history.append(history_entry)

    def update_resources(self, payload):
        with self.lock:
            resource = dict(payload)
            timestamp = resource.get('timestamp', time.time())
            used = resource.get('gpu_memory_used_mb')
            total = resource.get('gpu_memory_total_mb')
            if used is not None and total:
                resource['gpu_memory_utilization'] = 100.0 * used / total
            self.resources = resource
            if (
                    'gpu_utilization' in resource
                    or 'gpu_memory_used_mb' in resource):
                self.resource_history.append({
                    'timestamp': timestamp,
                    **{
                        name: resource[name]
                        for name in (
                            'gpu_utilization',
                            'gpu_memory_used_mb',
                            'gpu_memory_total_mb',
                            'gpu_memory_utilization',
                        )
                        if name in resource
                    },
                })

    def add_event(self, payload):
        with self.lock:
            self.events.append(dict(payload))

    def update_plot(self, name, payload):
        with self.lock:
            self.plots[str(name)] = dict(payload)

    def snapshot(self):
        with self.lock:
            resource_history = list(self.resource_history)
            if len(resource_history) > 900:
                last = len(resource_history) - 1
                indices = {
                    round(index * last / 899) for index in range(900)
                }
                resource_history = [
                    resource_history[index] for index in sorted(indices)
                ]
            return {
                'timestamp': self.timestamp,
                'training': dict(self.training),
                'resources': dict(self.resources),
                'plots': {
                    name: dict(payload)
                    for name, payload in self.plots.items()
                },
                'events': list(self.events),
                'history': list(self.history),
                'resource_history': resource_history,
            }


def _dashboard_process_main(
        metric_queue,
        host,
        port,
        resource_interval,
        history_size,
        curve_snapshot_enabled,
        curve_snapshot_interval,
        plot_done_event,
        run_dir,
        target_pid):
    state = _DashboardState(history_size)
    stop_event = threading.Event()
    sampler = ResourceSampler(target_pid)
    run_dir = Path(run_dir)
    output_path = run_dir / 'monitoring.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_lock = threading.Lock()
    previous_plot = existing_curve_metadata(run_dir)
    if previous_plot is not None:
        state.update_plot('training_curves', previous_plot)

    def render_curves_once():
        if not curve_snapshot_enabled:
            return None
        with render_lock:
            try:
                metadata = render_training_curve_snapshot(run_dir)
            except Exception as error:
                fallback = existing_curve_metadata(run_dir) or {}
                state.update_plot('training_curves', {
                    **fallback,
                    'error': f'{type(error).__name__}: {error}',
                    'last_attempt_at': time.time(),
                })
                return None
            state.update_plot('training_curves', metadata)
            return metadata

    def consume():
        with output_path.open('a', encoding='utf-8', buffering=1) as log:
            while not stop_event.is_set():
                try:
                    payload = metric_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if payload is None:
                    stop_event.set()
                    break
                kind = payload.pop('_kind', 'training')
                if kind == 'render_curves':
                    render_curves_once()
                    plot_done_event.set()
                    continue
                payload['monitor_timestamp'] = time.time()
                if kind == 'event':
                    state.add_event(payload)
                elif kind == 'plot':
                    state.update_plot(payload.pop('name'), payload)
                else:
                    state.update_training(payload)
                log.write(json.dumps(
                    {'kind': kind, **payload}, ensure_ascii=False
                ) + '\n')

    def collect_resources():
        resource_path = run_dir / 'resources.jsonl'
        with resource_path.open('a', encoding='utf-8', buffering=1) as log:
            while not stop_event.wait(resource_interval):
                payload = {'timestamp': time.time(), **sampler.sample()}
                state.update_resources(payload)
                log.write(json.dumps(
                    payload,
                    ensure_ascii=False,
                ) + '\n')

    def update_curve_snapshot():
        delay = min(5.0, curve_snapshot_interval)
        while not stop_event.is_set():
            if stop_event.wait(delay):
                return
            metadata = render_curves_once()
            delay = (
                curve_snapshot_interval
                if metadata is not None
                else min(15.0, curve_snapshot_interval)
            )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request_path = self.path.partition('?')[0]
            if request_path == '/api/status':
                snapshot = state.snapshot()
                snapshot['queue'] = load_training_queue(
                    run_dir, training=snapshot.get('training')
                )
                snapshot['merge_potential'] = scan_merge_potential_runs()
                snapshot['merge_distance'] = scan_merge_distance_runs()
                body = json.dumps(snapshot, ensure_ascii=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
            elif request_path in ('/', '/index.html'):
                body = json.dumps({
                    'error': 'standalone_dashboard_removed',
                    'message': '训练可视化已迁入 Xigua Atlas 项目门户',
                    'portal': 'http://127.0.0.1:3000/#live',
                    'api': '/api/status',
                }, ensure_ascii=False).encode('utf-8')
                self.send_response(410)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
            elif request_path in (
                    f'/plots/{CURVE_FILENAME}',
                    f'/plots/{EVENT_ANALYSIS_FILENAME}'):
                plot_root = run_dir / 'plots'
                plot_path = plot_root / Path(request_path).name
                try:
                    valid = (
                        not plot_root.is_symlink()
                        and plot_root.is_dir()
                        and not plot_path.is_symlink()
                        and plot_path.is_file()
                        and plot_path.resolve(strict=True).parent
                        == plot_root.resolve(strict=True)
                    )
                    body = plot_path.read_bytes() if valid else b'not found'
                except OSError:
                    body = b'not found'
                    valid = False
                self.send_response(200 if valid else 404)
                self.send_header(
                    'Content-Type', 'image/png' if valid else 'text/plain'
                )
                self.send_header('Cache-Control', 'no-store, max-age=0')
            else:
                body = b'not found'
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    consumer = threading.Thread(target=consume, daemon=True)
    resource_thread = threading.Thread(target=collect_resources, daemon=True)
    curve_thread = threading.Thread(
        target=update_curve_snapshot,
        name='daxigua-curve-snapshot',
        daemon=True,
    )
    consumer.start()
    resource_thread.start()
    if curve_snapshot_enabled:
        curve_thread.start()
    server = _ReusableThreadingHTTPServer((host, port), Handler)
    server.timeout = 0.5
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        stop_event.set()
        server.server_close()
        consumer.join(2.0)
        resource_thread.join(2.0)
        if curve_snapshot_enabled:
            curve_thread.join(10.0)
        sampler.close()


class DashboardPublisher:
    """训练侧非阻塞发布器；任何面板故障都降级为丢弃指标。"""

    def __init__(self, config, run_dir):
        self.enabled = bool(config.enabled)
        self.curve_snapshot_enabled = bool(
            config.curve_snapshot_enabled
        )
        self.process = None
        self.queue = None
        self.plot_done_event = None
        self._last_curve_snapshot_request = 0.0
        self.dropped_messages = 0
        if not self.enabled:
            return
        context = multiprocessing.get_context('spawn')
        self.queue = context.Queue(maxsize=128)
        self.plot_done_event = context.Event()
        self.process = context.Process(
            target=_dashboard_process_main,
            args=(
                self.queue,
                config.host,
                config.port,
                config.resource_interval_seconds,
                config.history_size,
                config.curve_snapshot_enabled,
                config.curve_snapshot_interval_seconds,
                self.plot_done_event,
                str(run_dir),
                os.getpid(),
            ),
            name='daxigua-training-dashboard',
            daemon=True,
        )
        self.process.start()

    def publish(self, payload, *, kind='training'):
        if not self.enabled or self.queue is None:
            return False
        message = _json_safe({'_kind': kind, **payload})
        try:
            self.queue.put_nowait(message)
            return True
        except (queue.Full, OSError, ValueError):
            self.dropped_messages += 1
            return False

    def event(self, event_kind, message, **values):
        return self.publish(
            {'kind': event_kind, 'message': message, **values}, kind='event'
        )

    def plot(self, name, metadata):
        return self.publish(
            {'name': str(name), **dict(metadata)}, kind='plot'
        )

    def snapshot_curves(self, *, wait=False, timeout=30.0):
        """请求旁路进程立即更新曲线；正式收尾时可等待原子落盘。"""

        if (
                not self.enabled
                or not self.curve_snapshot_enabled
                or self.queue is None
                or self.plot_done_event is None):
            return False
        self.plot_done_event.clear()
        message = {'_kind': 'render_curves'}
        try:
            if wait:
                self.queue.put(message, timeout=min(1.0, timeout))
            else:
                self.queue.put_nowait(message)
        except (queue.Full, OSError, ValueError):
            self.dropped_messages += 1
            return False
        self._last_curve_snapshot_request = time.monotonic()
        if not wait:
            return True
        return self.plot_done_event.wait(timeout)

    def close(self, timeout=5.0):
        if not self.enabled or self.queue is None:
            return
        if (
                self.curve_snapshot_enabled
                and time.monotonic() - self._last_curve_snapshot_request > 5.0):
            self.snapshot_curves(wait=True, timeout=30.0)
        try:
            self.queue.put_nowait(None)
        except (queue.Full, OSError, ValueError):
            pass
        if self.process is not None:
            self.process.join(timeout)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(1.0)


def _read_jsonl(path):
    rows = []
    try:
        with Path(path).open('r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def _completed_dashboard_snapshot(run_dir):
    run_dir = Path(run_dir)
    state = _DashboardState(7200)
    for row in _read_jsonl(run_dir / 'monitoring.jsonl'):
        kind = row.get('kind')
        payload = {
            key: value for key, value in row.items()
            if key not in ('kind', 'monitor_timestamp')
        }
        if kind == 'event':
            state.add_event(row)
        elif kind == 'plot':
            name = payload.pop('name', 'plot')
            state.update_plot(name, payload)
        else:
            state.update_training(payload)
    resources = _read_jsonl(run_dir / 'resources.jsonl')
    for resource in resources:
        state.update_resources(resource)
    for name, filename in (
            ('training_curves', 'training_curves.json'),
            ('evaluation_event_analysis', EVENT_ANALYSIS_FILENAME.replace(
                '.png', '.json'))):
        try:
            metadata = json.loads(
                (run_dir / 'plots' / filename).read_text(encoding='utf-8')
            )
        except (OSError, json.JSONDecodeError):
            continue
        state.update_plot(name, metadata)
    try:
        run_status = json.loads(
            (run_dir / 'run_status.json').read_text(encoding='utf-8')
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        run_status = {'phase': 'completed'}
    state.update_training(run_status)
    return state


def serve_completed_dashboard(run_dir, *, host='127.0.0.1', port=8765):
    """训练进程退出后继续提供只读面板和最终图片。"""

    run_dir = Path(run_dir).resolve()
    state = _completed_dashboard_snapshot(run_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request_path = self.path.partition('?')[0]
            if request_path == '/api/status':
                snapshot = state.snapshot()
                snapshot['timestamp'] = time.time()
                snapshot['queue'] = load_training_queue(
                    run_dir, training=snapshot.get('training')
                )
                snapshot['merge_potential'] = scan_merge_potential_runs()
                snapshot['merge_distance'] = scan_merge_distance_runs()
                body = json.dumps(snapshot, ensure_ascii=False).encode('utf-8')
                status = 200
                content_type = 'application/json; charset=utf-8'
            elif request_path in ('/', '/index.html'):
                body = json.dumps({
                    'error': 'standalone_dashboard_removed',
                    'message': '训练可视化已迁入 Xigua Atlas 项目门户',
                    'portal': 'http://127.0.0.1:3000/#live',
                    'api': '/api/status',
                }, ensure_ascii=False).encode('utf-8')
                status = 410
                content_type = 'application/json; charset=utf-8'
            elif request_path in (
                    f'/plots/{CURVE_FILENAME}',
                    f'/plots/{EVENT_ANALYSIS_FILENAME}'):
                plot_path = run_dir / 'plots' / Path(request_path).name
                try:
                    valid = (
                        plot_path.is_file()
                        and not plot_path.is_symlink()
                        and plot_path.resolve(strict=True).parent
                        == (run_dir / 'plots').resolve(strict=True)
                    )
                    body = plot_path.read_bytes() if valid else b'not found'
                except OSError:
                    valid = False
                    body = b'not found'
                status = 200 if valid else 404
                content_type = 'image/png' if valid else 'text/plain'
            else:
                body = b'not found'
                status = 404
                content_type = 'text/plain'
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store, max-age=0')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = _ReusableThreadingHTTPServer((host, int(port)), Handler)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
