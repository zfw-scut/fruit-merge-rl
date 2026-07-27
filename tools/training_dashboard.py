#!/usr/bin/env python3
"""只读训练状态面板服务。

服务只读取训练和旁路监控已经落盘的 CSV/JSON/PNG，不 import PyTorch，也不会
向训练进程发送信号。默认仅监听回环地址，外部访问应通过 SSH 端口转发完成。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8765
DEFAULT_HISTORY_LIMIT = 600
MAX_HISTORY_LIMIT = 5000
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_CSV_TAIL_BYTES = 16 * 1024 * 1024
MAX_EVENT_TAIL_BYTES = 2 * 1024 * 1024
DATA_STALE_AFTER_SECONDS = 30.0

ALLOWED_PLOT_FILES = frozenset(
    {
        'training_curves.png',
        'reward_breakdown_curves.png',
        'structure_learning_curves.png',
        'physics_mode_comparison.png',
    }
)
PLOT_TITLES = {
    'training_curves.png': '训练主曲线',
    'reward_breakdown_curves.png': '奖励分解曲线',
    'structure_learning_curves.png': '结构学习曲线',
    'physics_mode_comparison.png': '物理模式对比',
}
STATIC_ROUTES = {
    '/': 'index.html',
    '/index.html': 'index.html',
    '/app.js': 'app.js',
    '/styles.css': 'styles.css',
}
SENSITIVE_KEY_PARTS = (
    'password',
    'passwd',
    'secret',
    'credential',
    'authorization',
    'private_key',
    'access_key',
)
SAFE_EVENT_DETAIL_KEYS = frozenset(
    {
        'duration',
        'pids',
        'previous_pids',
        'current_pids',
        'mem_available_mb',
        'swap_used_mb',
        'target_rss_mb',
        'gpu_memory_used_mb',
        'threshold_mb',
        'samples',
        'exit_code',
    }
)


def utc_now() -> str:
    """返回适合 API 的 UTC 时间。"""

    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _file_age_seconds(path: Path | None) -> float | None:
    """返回文件距离最近落盘的秒数；文件不存在或不可读时返回 None。"""

    if path is None:
        return None
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _finite_number(value: Any) -> float | None:
    """把 CSV 标量转成有限浮点数；空值、NaN 和 Inf 均视为缺失。"""

    if value is None or value == '':
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite_number(value)
    return None if number is None else int(number)


def _first_number(row: dict[str, Any], *names: str) -> float | None:
    for name in names:
        number = _finite_number(row.get(name))
        if number is not None:
            return number
    return None


def _first_integer(row: dict[str, Any], *names: str) -> int | None:
    for name in names:
        number = _integer(row.get(name))
        if number is not None:
            return number
    return None


def _safe_read_json(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None or not path.is_file():
        return {}, None
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            return {}, f'{path.name} 超过读取上限'
        value = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value, dict):
            return {}, f'{path.name} 顶层不是对象'
        return value, None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, f'{path.name} 暂时不可读：{type(exc).__name__}'


def _complete_file_tail(
    path: Path,
    *,
    max_bytes: int,
    keep_header: bool,
) -> tuple[bytes, str | None]:
    """读取文件头和有限尾部，并丢弃训练进程尚未写完的最后一行。"""

    try:
        size = path.stat().st_size
        if size <= 0:
            return b'', None
        with path.open('rb') as file_obj:
            header = file_obj.readline(256 * 1024) if keep_header else b''
            start = max(len(header), size - max_bytes) if keep_header else max(0, size - max_bytes)
            file_obj.seek(start)
            if start > (len(header) if keep_header else 0):
                file_obj.readline()
            body = file_obj.read(max_bytes + 256 * 1024)
        if body and not body.endswith((b'\n', b'\r')):
            split_at = max(body.rfind(b'\n'), body.rfind(b'\r'))
            body = body[: split_at + 1] if split_at >= 0 else b''
        return header + body, None
    except OSError as exc:
        return b'', f'{path.name} 暂时不可读：{type(exc).__name__}'


def read_csv_rows(path: Path | None, limit: int) -> tuple[list[dict[str, str]], str | None]:
    """读取 CSV 的完整行尾快照。

    训练进程可能正写到一行中间，因此没有换行结尾的尾行不会进入面板。解析失败的
    单行也会被跳过，下一轮轮询可在 flush 后自然恢复。
    """

    if path is None or not path.is_file():
        return [], None
    payload, error = _complete_file_tail(
        path,
        max_bytes=MAX_CSV_TAIL_BYTES,
        keep_header=True,
    )
    if error or not payload:
        return [], error
    try:
        text = payload.decode('utf-8-sig')
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames:
            return [], f'{path.name} 缺少表头'
        rows: deque[dict[str, str]] = deque(maxlen=limit)
        expected = len(reader.fieldnames)
        for row in reader:
            if None in row or len(row) != expected:
                continue
            rows.append({key: value for key, value in row.items() if key is not None})
        return list(rows), None
    except (csv.Error, UnicodeError) as exc:
        return [], f'{path.name} 暂时不可解析：{type(exc).__name__}'


def read_events(path: Path | None, limit: int = 40) -> tuple[list[dict[str, Any]], str | None]:
    """读取近期结构化事件，只返回安全且对运维有意义的字段。"""

    if path is None or not path.is_file():
        return [], None
    payload, error = _complete_file_tail(
        path,
        max_bytes=MAX_EVENT_TAIL_BYTES,
        keep_header=False,
    )
    if error or not payload:
        return [], error
    events: deque[dict[str, Any]] = deque(maxlen=limit)
    for line in payload.decode('utf-8', errors='replace').splitlines():
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        event_name = str(raw.get('event') or 'monitor_event')
        level = str(raw.get('level') or 'info').lower()
        details = raw.get('details') if isinstance(raw.get('details'), dict) else {}
        safe_details = {
            str(key): value
            for key, value in details.items()
            if str(key).lower() in SAFE_EVENT_DETAIL_KEYS
            and not any(part in str(key).lower() for part in SENSITIVE_KEY_PARTS)
            and isinstance(value, (str, int, float, bool, list, type(None)))
        }
        events.append(
            {
                'timestamp': raw.get('timestamp'),
                'level': level,
                'event': event_name,
                'message': event_name.replace('_', ' '),
                'details': safe_details,
            }
        )
    return list(events), None


def _nested(mapping: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = mapping
        for part in path:
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return None


def _latest_directory(root: Path, marker_names: Iterable[str]) -> Path | None:
    """在一层子目录中选取最近更新且包含 marker 的目录。"""

    if not root.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    try:
        children = list(root.iterdir())
    except OSError:
        return None
    for child in children:
        if child.is_symlink() or not child.is_dir():
            continue
        markers = [child / name for name in marker_names]
        present = [path for path in markers if path.is_file()]
        if not present:
            continue
        try:
            mtime = max(path.stat().st_mtime for path in present)
        except OSError:
            continue
        candidates.append((mtime, child.resolve()))
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def discover_directories(
    project_root: Path,
    run_dir: Path | None,
    monitor_dir: Path | None,
    control_dir: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    """补齐未显式指定的 run/monitor/control 目录。"""

    runs_root = project_root / 'runs'
    resolved_run = run_dir.expanduser().resolve() if run_dir is not None else None
    resolved_monitor = monitor_dir.expanduser().resolve() if monitor_dir is not None else None
    resolved_control = control_dir.expanduser().resolve() if control_dir is not None else None
    if resolved_run is None:
        resolved_run = _latest_directory(runs_root, ('config.json', 'metrics.csv'))
    if resolved_monitor is None:
        resolved_monitor = _latest_directory(
            runs_root / 'resource_monitor',
            ('system_metrics.csv', 'gpu_metrics.csv', 'events.jsonl'),
        )
    if resolved_control is None:
        control_candidates = [
            value
            for value in (
                _latest_directory(
                    runs_root / 'stage_control',
                    ('train.exit', 'status.json', 'binding.env', 'train_wrapper.log'),
                ),
                _latest_directory(
                    runs_root / 'cloud_stage_control',
                    ('train.exit', 'status.json', 'binding.env', 'train_wrapper.log'),
                ),
            )
            if value is not None
        ]
        resolved_control = max(
            control_candidates,
            default=None,
            key=lambda path: path.stat().st_mtime,
        )
    return resolved_run, resolved_monitor, resolved_control


def _config_total_updates(config: dict[str, Any]) -> int | None:
    value = _nested(
        config,
        ('args', 'total_updates'),
        ('run_manifest', 'config', 'total_updates'),
        ('resolved', 'training', 'total_updates'),
    )
    return _integer(value)


def _series(
    rows: list[dict[str, Any]],
    x_names: tuple[str, ...],
    y_names: tuple[str, ...],
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for row in rows:
        x_value = _first_number(row, *x_names)
        y_value = _first_number(row, *y_names)
        if x_value is not None and y_value is not None:
            points.append({'x': x_value, 'y': y_value})
    return points


def _recent_average(rows: list[dict[str, Any]], *names: str, count: int = 20) -> float | None:
    values = [
        number
        for row in rows[-count:]
        if (number := _first_number(row, *names)) is not None and number > 0
    ]
    return sum(values) / len(values) if values else None


def _read_exit_code(control_dir: Path | None) -> tuple[int | None, str | None]:
    if control_dir is None:
        return None, None
    path = control_dir / 'train.exit'
    if not path.is_file():
        return None, None
    try:
        text = path.read_text(encoding='utf-8').strip()
        return int(text), None
    except (OSError, UnicodeError, ValueError):
        return None, 'train.exit 暂时不可解析'


PROGRESS_RE = re.compile(
    r'phase=(?P<phase>[^\s|]+).*?'
    r'(?P<current>\d+)\s*/\s*(?P<total>\d+).*?'
    r'env_steps=(?P<env_steps>\d+).*?'
    r'buffer=(?P<buffer>\d+).*?'
    r'eps=(?P<epsilon>[0-9.eE+-]+).*?'
    r'speed=(?P<env_speed>[0-9.eE+-]+)\s+env_steps/s.*?'
    r'eta=(?P<eta_minutes>[0-9.eE+-]+)min'
    r'(?:.*?loss=(?P<loss>[0-9.eE+-]+))?'
)


def read_progress_heartbeat(control_dir: Path | None) -> dict[str, Any]:
    """从启动器日志读取比 metrics.csv 更密集的最后一条进度心跳。"""

    if control_dir is None:
        return {}
    path = control_dir / 'train_wrapper.log'
    if not path.is_file():
        return {}
    try:
        size = path.stat().st_size
        with path.open('rb') as file_obj:
            file_obj.seek(max(0, size - 512 * 1024))
            if size > 512 * 1024:
                file_obj.readline()
            text = file_obj.read(512 * 1024).decode('utf-8', errors='replace')
    except OSError:
        return {}
    for line in reversed(text.splitlines()):
        if '[progress' not in line:
            continue
        match = PROGRESS_RE.search(line)
        if match is None:
            continue
        values = match.groupdict()
        current = int(values['current'])
        total = int(values['total'])
        eta_seconds = float(values['eta_minutes']) * 60.0
        update_rate = (
            (total - current) / eta_seconds
            if eta_seconds > 0 and total >= current
            else None
        )
        return {
            'phase': values['phase'],
            'current_update': current,
            'target_updates': total,
            'env_steps': int(values['env_steps']),
            'buffer_size': int(values['buffer']),
            'epsilon': float(values['epsilon']),
            'env_steps_per_second': float(values['env_speed']),
            'updates_per_second': update_rate,
            'eta_seconds': eta_seconds,
            'loss': _finite_number(values.get('loss')),
        }
    return {}


def read_cgroup_cpu_capacity(
    path: Path = Path('/sys/fs/cgroup/cpu.max'),
) -> float | None:
    """读取当前容器可用 CPU 核配额；无限额或非 Linux 时返回 None。"""

    try:
        quota_text, period_text = path.read_text(encoding='ascii').split()[:2]
        if quota_text == 'max':
            return None
        quota = float(quota_text)
        period = float(period_text)
        return quota / period if quota > 0 and period > 0 else None
    except (OSError, UnicodeError, ValueError):
        return None


def _plot_inventory(run_dir: Path | None) -> list[dict[str, Any]]:
    if run_dir is None:
        return []
    plot_root = run_dir / 'plots'
    if not plot_root.is_dir() or plot_root.is_symlink():
        return []
    plots: list[dict[str, Any]] = []
    for name in sorted(ALLOWED_PLOT_FILES):
        path = plot_root / name
        try:
            if path.is_symlink() or not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        plots.append(
            {
                'name': name,
                'title': PLOT_TITLES.get(name, name),
                'url': f'/plots/{name}',
                'size_bytes': stat.st_size,
                'modified_at': datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat().replace('+00:00', 'Z'),
            }
        )
    return plots


def _safe_plot_path(run_dir: Path | None, name: str) -> Path | None:
    """解析固定名称曲线，并拒绝文件或父目录符号链接。"""

    if run_dir is None or name not in ALLOWED_PLOT_FILES:
        return None
    plot_root = run_dir / 'plots'
    path = plot_root / name
    try:
        if (
            plot_root.is_symlink()
            or not plot_root.is_dir()
            or path.is_symlink()
            or not path.is_file()
        ):
            return None
        resolved_root = plot_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        if resolved_path.parent != resolved_root:
            return None
    except OSError:
        return None
    return resolved_path


class DashboardStateBuilder:
    """把训练落盘产物投影为稳定、无凭据的面板状态。"""

    def __init__(
        self,
        *,
        project_root: Path,
        run_dir: Path | None = None,
        monitor_dir: Path | None = None,
        control_dir: Path | None = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
    ) -> None:
        self.project_root = project_root.resolve()
        self.run_dir, self.monitor_dir, self.control_dir = discover_directories(
            self.project_root,
            run_dir,
            monitor_dir,
            control_dir,
        )
        self.history_limit = max(10, min(int(history_limit), MAX_HISTORY_LIMIT))

    def build(self) -> dict[str, Any]:
        warnings: list[str] = []
        config, error = _safe_read_json(
            None if self.run_dir is None else self.run_dir / 'config.json'
        )
        if error:
            warnings.append(error)
        metrics, error = read_csv_rows(
            None if self.run_dir is None else self.run_dir / 'metrics.csv',
            self.history_limit,
        )
        if error:
            warnings.append(error)
        episodes, error = read_csv_rows(
            None if self.run_dir is None else self.run_dir / 'episode_metrics.csv',
            self.history_limit,
        )
        if error:
            warnings.append(error)
        system_rows, error = read_csv_rows(
            None if self.monitor_dir is None else self.monitor_dir / 'system_metrics.csv',
            self.history_limit,
        )
        if error:
            warnings.append(error)
        gpu_rows, error = read_csv_rows(
            None if self.monitor_dir is None else self.monitor_dir / 'gpu_metrics.csv',
            self.history_limit,
        )
        if error:
            warnings.append(error)
        cgroup_rows, error = read_csv_rows(
            None if self.monitor_dir is None else self.monitor_dir / 'cgroup_metrics.csv',
            self.history_limit,
        )
        if error:
            warnings.append(error)
        events, error = read_events(
            None if self.monitor_dir is None else self.monitor_dir / 'events.jsonl'
        )
        if error:
            warnings.append(error)
        exit_code, error = _read_exit_code(self.control_dir)
        if error:
            warnings.append(error)
        heartbeat = read_progress_heartbeat(self.control_dir)
        heartbeat_phase = str(heartbeat.get('phase') or '').lower()
        training_heartbeat = heartbeat_phase in {'train', 'training'}

        metrics_age = _file_age_seconds(
            None if self.run_dir is None else self.run_dir / 'metrics.csv'
        )
        heartbeat_age = _file_age_seconds(
            None
            if self.control_dir is None
            else self.control_dir / 'train_wrapper.log'
        )
        resource_age = _file_age_seconds(
            None
            if self.monitor_dir is None
            else self.monitor_dir / 'system_metrics.csv'
        )
        observed_ages = [
            age
            for age in (heartbeat_age, resource_age, metrics_age)
            if age is not None
        ]
        data_age = min(observed_ages) if observed_ages else None
        data_fresh = bool(
            data_age is not None and data_age <= DATA_STALE_AFTER_SECONDS
        )
        heartbeat_fresh = bool(
            heartbeat_age is not None
            and heartbeat_age <= DATA_STALE_AFTER_SECONDS
        )
        resource_fresh = bool(
            resource_age is not None
            and resource_age <= DATA_STALE_AFTER_SECONDS
        )

        latest = metrics[-1] if metrics else {}
        latest_system = system_rows[-1] if system_rows else {}
        latest_cgroup = cgroup_rows[-1] if cgroup_rows else {}
        target_updates = _config_total_updates(config)
        current_update = _first_integer(latest, 'update_step', 'update', 'step') or 0
        env_steps = _first_integer(latest, 'env_steps', 'environment_steps') or 0
        if (
            training_heartbeat
            and heartbeat.get('current_update', -1) >= current_update
        ):
            current_update = int(heartbeat['current_update'])
            env_steps = int(heartbeat.get('env_steps', env_steps))
            target_updates = int(
                heartbeat.get('target_updates') or target_updates or 0
            ) or None
        updates_per_second = _first_number(
            latest, 'updates_per_second', 'updates_per_sec', 'update_rate'
        )
        if training_heartbeat and heartbeat.get('updates_per_second'):
            updates_per_second = float(heartbeat['updates_per_second'])
        if updates_per_second is None or updates_per_second <= 0:
            updates_per_second = _recent_average(
                metrics, 'updates_per_second', 'updates_per_sec', 'update_rate'
            )
        env_steps_per_second = _first_number(
            latest, 'env_steps_per_second', 'env_steps_per_sec'
        )
        if training_heartbeat and heartbeat.get('env_steps_per_second') is not None:
            env_steps_per_second = float(heartbeat['env_steps_per_second'])
        direct_eta = _first_number(
            latest, 'eta_seconds', 'remaining_seconds', 'estimated_remaining_seconds'
        )
        if training_heartbeat and heartbeat.get('eta_seconds') is not None:
            direct_eta = float(heartbeat['eta_seconds'])
        eta_source = (
            'heartbeat'
            if training_heartbeat and heartbeat.get('eta_seconds') is not None
            else ('metrics' if direct_eta is not None else None)
        )
        eta_seconds = direct_eta
        if (
            eta_seconds is None
            and target_updates is not None
            and updates_per_second is not None
            and updates_per_second > 0
        ):
            eta_seconds = max(0.0, (target_updates - current_update) / updates_per_second)
            eta_source = 'updates_per_second'
        percent = (
            min(100.0, max(0.0, current_update / target_updates * 100.0))
            if target_updates and target_updates > 0
            else None
        )

        target_process_count = _first_integer(latest_system, 'target_process_count') or 0
        process_active_observed = target_process_count > 0
        process_active = bool(
            (process_active_observed and resource_fresh)
            or (training_heartbeat and heartbeat_fresh)
        )
        is_complete = bool(
            exit_code == 0
            or (
                target_updates is not None
                and target_updates > 0
                and current_update >= target_updates
            )
        )
        if exit_code is not None:
            status = 'completed' if exit_code == 0 else 'failed'
        elif is_complete:
            status = 'completed'
        elif process_active:
            status = 'running'
        elif metrics and not data_fresh:
            status = 'stale'
        elif metrics:
            status = 'waiting'
        elif config:
            status = 'starting'
        else:
            status = 'no_data'

        resources = self._build_resources(
            latest_system,
            latest_cgroup,
            gpu_rows,
            read_cgroup_cpu_capacity(),
        )
        training = self._build_training(latest, episodes)
        if training_heartbeat and heartbeat.get('loss') is not None:
            training['loss'] = heartbeat['loss']
        if training_heartbeat and heartbeat.get('buffer_size') is not None:
            training['buffer_size'] = heartbeat['buffer_size']
        actor = {
            'requests': _first_integer(latest, 'actor_inference_requests'),
            'batches': _first_integer(latest, 'actor_inference_batches'),
            'mean_batch_size': _first_number(
                latest, 'actor_inference_mean_batch_size'
            ),
            'max_batch_size': _first_number(latest, 'actor_inference_max_batch'),
            'inference_seconds': _first_number(latest, 'actor_inference_seconds'),
        }
        causal = self._build_causal(latest)
        alerts = [event for event in events if event.get('level') in {'warning', 'error', 'critical'}]
        if exit_code not in (None, 0):
            alerts.append(
                {
                    'timestamp': utc_now(),
                    'level': 'error',
                    'event': 'training_exit_nonzero',
                    'message': f'训练进程异常退出（code={exit_code}）',
                    'details': {'exit_code': exit_code},
                }
            )
        alerts.extend(
            {
                'timestamp': utc_now(),
                'level': 'warning',
                'event': 'data_read_warning',
                'message': message,
                'details': {},
            }
            for message in warnings
        )

        return {
            'schema_version': 1,
            'generated_at': utc_now(),
            'status': status,
            'identity': {
                'run': None if self.run_dir is None else self.run_dir.name,
                'monitor': (
                    None if self.monitor_dir is None else self.monitor_dir.name
                ),
                'control': (
                    None if self.control_dir is None else self.control_dir.name
                ),
            },
            'freshness': {
                'data_fresh': data_fresh,
                'data_age_seconds': data_age,
                'heartbeat_age_seconds': heartbeat_age,
                'resource_age_seconds': resource_age,
                'metrics_age_seconds': metrics_age,
                'stale_after_seconds': DATA_STALE_AFTER_SECONDS,
            },
            'progress': {
                'phase': (
                    'complete'
                    if is_complete
                    else str(
                        heartbeat.get('phase')
                        or ('training' if metrics else 'warmup')
                    )
                ),
                'current_update': current_update,
                'target_updates': target_updates,
                'percent': percent,
                'env_steps': env_steps,
                'epsilon': heartbeat.get('epsilon', _first_number(latest, 'epsilon')),
                'updates_per_second': updates_per_second,
                'env_steps_per_second': env_steps_per_second,
                'eta_seconds': eta_seconds,
                'eta_source': eta_source,
                'elapsed_seconds': _first_number(latest_system, 'elapsed_sec'),
                'is_complete': is_complete,
                'process_active': process_active,
                'process_active_observed': process_active_observed,
                'exit_code': exit_code,
            },
            'resources': resources,
            'training': training,
            'actor': actor,
            'causal': causal,
            'series': self._build_series(
                metrics,
                episodes,
                system_rows,
                gpu_rows,
                cgroup_rows,
                resources.get('cpu_count'),
            ),
            'plots': _plot_inventory(self.run_dir),
            'alerts': alerts[-40:],
        }

    @staticmethod
    def _build_resources(
        system: dict[str, Any],
        cgroup: dict[str, Any],
        gpu_rows: list[dict[str, Any]],
        cgroup_cpu_capacity: float | None,
    ) -> dict[str, Any]:
        host_cpu_count = _first_integer(system, 'cpu_count')
        cpu_count = cgroup_cpu_capacity or host_cpu_count
        target_cpu = _first_number(system, 'target_cpu_percent', 'cpu_percent')
        cpu_cores_used = None if target_cpu is None else target_cpu / 100.0
        cpu_util = (
            None
            if target_cpu is None or not cpu_count
            else min(100.0, max(0.0, target_cpu / cpu_count))
        )
        latest_sample = gpu_rows[-1].get('sample') if gpu_rows else None
        latest_gpus = [
            row for row in gpu_rows if latest_sample is None or row.get('sample') == latest_sample
        ]
        gpu_util_values = [
            value
            for row in latest_gpus
            if (value := _first_number(row, 'util_gpu_percent', 'gpu_util_percent')) is not None
        ]
        gpu_used_values = [
            value
            for row in latest_gpus
            if (value := _first_number(row, 'memory_used_mb', 'gpu_memory_used_mb')) is not None
        ]
        gpu_total_values = [
            value
            for row in latest_gpus
            if (value := _first_number(row, 'memory_total_mb', 'gpu_memory_total_mb')) is not None
        ]
        gpu_names = [
            str(row['name'])
            for row in latest_gpus
            if row.get('name')
        ]
        gpu_used = sum(gpu_used_values) if gpu_used_values else _first_number(
            system, 'gpu_memory_used_mb'
        )
        gpu_total = sum(gpu_total_values) if gpu_total_values else _first_number(
            system, 'gpu_memory_total_mb'
        )
        cgroup_total_bytes = _first_number(cgroup, 'memory_max_bytes')
        cgroup_used_bytes = _first_number(cgroup, 'memory_working_set_bytes')
        cgroup_available_bytes = _first_number(
            cgroup, 'memory_effective_available_bytes'
        )
        mem_total = (
            cgroup_total_bytes / (1024 * 1024)
            if cgroup_total_bytes is not None
            else _first_number(system, 'mem_total_mb')
        )
        mem_used = (
            cgroup_used_bytes / (1024 * 1024)
            if cgroup_used_bytes is not None
            else _first_number(system, 'mem_used_mb')
        )
        return {
            'timestamp': system.get('timestamp'),
            'cpu_count': cpu_count,
            'host_cpu_count': host_cpu_count,
            'cpu_capacity_source': (
                'cgroup_cpu_max' if cgroup_cpu_capacity is not None else 'host'
            ),
            'cpu_cores_used': cpu_cores_used,
            'cpu_util_percent': cpu_util,
            'target_cpu_percent_raw': target_cpu,
            'load_1m': _first_number(system, 'load1', 'load_1m'),
            'load_5m': _first_number(system, 'load5', 'load_5m'),
            'memory_total_mb': mem_total,
            'memory_used_mb': mem_used,
            'memory_available_mb': (
                cgroup_available_bytes / (1024 * 1024)
                if cgroup_available_bytes is not None
                else _first_number(system, 'mem_available_mb')
            ),
            'memory_source': (
                'cgroup_working_set' if cgroup_total_bytes is not None else 'host'
            ),
            'memory_used_percent': (
                mem_used / mem_total * 100.0 if mem_total and mem_used is not None else None
            ),
            'swap_used_mb': _first_number(system, 'swap_used_mb'),
            'target_process_count': _first_integer(system, 'target_process_count'),
            'target_rss_mb': _first_number(system, 'target_rss_mb'),
            'gpu_count': len(latest_gpus) or _first_integer(system, 'gpu_count'),
            'gpu_name': ', '.join(dict.fromkeys(gpu_names)) or None,
            'gpu_util_percent': max(gpu_util_values) if gpu_util_values else _first_number(
                system, 'gpu_util_percent_max'
            ),
            'gpu_memory_used_mb': gpu_used,
            'gpu_memory_total_mb': gpu_total,
            'gpu_memory_used_percent': (
                gpu_used / gpu_total * 100.0 if gpu_total and gpu_used is not None else None
            ),
            'gpu_temperature_c': max(
                (
                    value
                    for row in latest_gpus
                    if (value := _first_number(row, 'temperature_c')) is not None
                ),
                default=_first_number(system, 'gpu_temperature_c_max'),
            ),
            'gpu_power_draw_w': sum(
                value
                for row in latest_gpus
                if (value := _first_number(row, 'power_draw_w')) is not None
            ) if latest_gpus else _first_number(system, 'gpu_power_draw_w_sum'),
            'gpu_power_limit_w': sum(
                value
                for row in latest_gpus
                if (value := _first_number(row, 'power_limit_w')) is not None
            ) if latest_gpus else _first_number(system, 'gpu_power_limit_w_sum'),
        }

    @staticmethod
    def _build_training(
        latest: dict[str, Any],
        episodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        episode_scores = [
            score
            for row in episodes
            if (score := _first_number(row, 'score', 'episode_score')) is not None
        ]
        episode_indices = [
            index
            for row in episodes
            if (index := _first_integer(row, 'episode_index')) is not None
        ]
        return {
            'loss': _first_number(latest, 'loss'),
            'td_loss': _first_number(latest, 'td_loss'),
            'structural_loss': _first_number(latest, 'structural_loss'),
            'weighted_structural_loss': _first_number(
                latest, 'weighted_structural_loss'
            ),
            'mean_reward': _first_number(latest, 'mean_reward'),
            'mean_abs_td_error': _first_number(latest, 'mean_abs_td_error'),
            'grad_norm': _first_number(latest, 'grad_norm'),
            'buffer_size': _first_integer(latest, 'buffer_size'),
            'episode_count': (
                max(episode_indices) if episode_indices else len(episodes)
            ),
            'episode_history_count': len(episodes),
            'latest_episode_score': episode_scores[-1] if episode_scores else None,
            'recent_episode_score_mean': (
                sum(episode_scores[-20:]) / len(episode_scores[-20:])
                if episode_scores
                else None
            ),
            'episode_score_max': max(episode_scores) if episode_scores else None,
            'collect_mean_episode_score': _first_number(
                latest, 'collect_mean_episode_score'
            ),
            'eval_score_mean': _first_number(latest, 'eval_score_mean'),
            'eval_score_max': _first_number(latest, 'eval_score_max'),
            'best_eval_score': _first_number(latest, 'best_eval_score'),
            'best_eval_update': _first_integer(latest, 'best_eval_update'),
            'max_fruit_level': _first_integer(latest, 'collect_max_fruit_level'),
            'max_chain_depth': _first_integer(
                latest, 'collect_attribution_max_chain_depth'
            ),
            'train_step_seconds': _first_number(latest, 'train_step_seconds'),
            'collect_seconds': _first_number(latest, 'collect_seconds'),
            'structural_valid_count': _first_integer(
                latest, 'structural_valid_count'
            ),
            'structural_sample_count': _first_integer(
                latest, 'structural_sample_count'
            ),
        }

    @staticmethod
    def _build_causal(latest: dict[str, Any]) -> dict[str, Any]:
        return {
            'enabled': bool(_first_integer(latest, 'counterfactual_enabled') or 0),
            'update_applied': bool(
                _first_integer(latest, 'causal_update_applied') or 0
            ),
            'batch_size': _first_integer(latest, 'causal_batch_size'),
            'rule_rank_loss': _first_number(latest, 'rule_rank_loss'),
            'counterfactual_loss': _first_number(
                latest, 'counterfactual_loss'
            ),
            'replay_size': _first_integer(latest, 'causal_replay_size'),
            'positive_count': _first_integer(latest, 'causal_replay_positive_count'),
            'negative_count': _first_integer(latest, 'causal_replay_negative_count'),
            'counterfactual_count': _first_integer(
                latest, 'causal_replay_counterfactual_count'
            ),
            'rule_batch_size': _first_integer(latest, 'rule_batch_size'),
            'counterfactual_batch_size': _first_integer(
                latest, 'counterfactual_batch_size'
            ),
            'counterfactual_completed': _first_integer(
                latest, 'counterfactual_results_completed'
            ),
            'counterfactual_failed': _first_integer(
                latest, 'counterfactual_results_failed'
            ),
            'counterfactual_samples': _first_integer(
                latest, 'counterfactual_samples_inserted'
            ),
            'actual_token_ratio': _first_number(
                latest, 'counterfactual_actual_token_ratio'
            ),
            'projected_token_ratio': _first_number(
                latest, 'counterfactual_projected_token_ratio'
            ),
            'hard_budget_respected': bool(
                _first_integer(latest, 'counterfactual_hard_budget_respected') or 0
            ),
            'shapley_selected': _first_integer(latest, 'shapley_events_selected'),
            'shapley_completed': _first_integer(latest, 'shapley_tasks_completed'),
            'shapley_samples': _first_integer(latest, 'shapley_samples_inserted'),
        }

    @staticmethod
    def _build_series(
        metrics: list[dict[str, Any]],
        episodes: list[dict[str, Any]],
        system_rows: list[dict[str, Any]],
        gpu_rows: list[dict[str, Any]],
        cgroup_rows: list[dict[str, Any]],
        cpu_capacity: float | None,
    ) -> dict[str, list[dict[str, float]]]:
        return {
            'loss': _series(metrics, ('update_step',), ('loss',)),
            'td_loss': _series(metrics, ('update_step',), ('td_loss',)),
            'structural_loss': _series(
                metrics, ('update_step',), ('structural_loss',)
            ),
            'causal_loss': _series(
                metrics, ('update_step',), ('counterfactual_loss',)
            ),
            'mean_reward': _series(metrics, ('update_step',), ('mean_reward',)),
            'eval_score': _series(
                metrics, ('update_step',), ('eval_score_mean',)
            ),
            'best_eval_score': _series(
                metrics, ('update_step',), ('best_eval_score',)
            ),
            'episode_score': _series(
                episodes, ('episode_index',), ('score', 'episode_score')
            ),
            'updates_per_second': _series(
                metrics, ('update_step',), ('updates_per_second',)
            ),
            'env_steps_per_second': _series(
                metrics, ('update_step',), ('env_steps_per_second',)
            ),
            'actor_batch_size': _series(
                metrics, ('update_step',), ('actor_inference_mean_batch_size',)
            ),
            'cpu_util_percent': [
                {
                    'x': x,
                    'y': min(
                        100.0,
                        max(0.0, raw / (cpu_capacity or count)),
                    ),
                }
                for row in system_rows
                if (x := _first_number(row, 'elapsed_sec')) is not None
                and (raw := _first_number(row, 'target_cpu_percent')) is not None
                and (count := _first_number(row, 'cpu_count')) not in (None, 0)
            ],
            'memory_used_mb': (
                [
                    {'x': x, 'y': used / (1024 * 1024)}
                    for row in cgroup_rows
                    if (x := _first_number(row, 'elapsed_sec')) is not None
                    and (
                        used := _first_number(row, 'memory_working_set_bytes')
                    ) is not None
                ]
                if cgroup_rows
                else _series(system_rows, ('elapsed_sec',), ('mem_used_mb',))
            ),
            'gpu_util_percent': _series(
                gpu_rows, ('elapsed_sec',), ('util_gpu_percent',)
            ),
            'gpu_memory_used_mb': _series(
                gpu_rows, ('elapsed_sec',), ('memory_used_mb',)
            ),
        }


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        state_builder: DashboardStateBuilder,
        static_dir: Path,
    ) -> None:
        self.state_builder = state_builder
        self.static_dir = static_dir.resolve()
        super().__init__(server_address, DashboardRequestHandler)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """固定路由、固定文件白名单的 HTTP handler。"""

    server: DashboardServer
    protocol_version = 'HTTP/1.1'

    def log_request(
        self,
        code: int | str = '-',
        size: int | str = '-',
    ) -> None:
        """记录不含查询字符串的最小访问日志。"""

        try:
            safe_path = unquote(urlsplit(self.path).path)
        except (UnicodeError, ValueError):
            safe_path = '<invalid-path>'
        sys.stderr.write(
            f'[{self.log_date_time_string()}] {self.client_address[0]} '
            f'{self.command} {safe_path} {code} {size}\n'
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        """保留框架诊断；正常访问统一由 log_request 脱敏记录。"""

        sys.stderr.write(
            f'[{self.log_date_time_string()}] {self.client_address[0]} '
            f'{fmt % args}\n'
        )

    def do_HEAD(self) -> None:
        self._dispatch(send_body=False)

    def do_GET(self) -> None:
        self._dispatch(send_body=True)

    def _dispatch(self, *, send_body: bool) -> None:
        try:
            path = unquote(urlsplit(self.path).path)
        except (UnicodeError, ValueError):
            self._send_json({'error': 'bad request'}, HTTPStatus.BAD_REQUEST, send_body)
            return
        if '\\' in path or '\x00' in path:
            self._send_json({'error': 'not found'}, HTTPStatus.NOT_FOUND, send_body)
            return
        if path == '/api/state':
            self._send_json(self.server.state_builder.build(), HTTPStatus.OK, send_body)
            return
        if path == '/api/health':
            state = self.server.state_builder.build()
            data_fresh = bool(state['freshness']['data_fresh'])
            service_ok = state['status'] not in {'failed', 'no_data'}
            self._send_json(
                {
                    'ok': service_ok
                    and (data_fresh or state['status'] == 'completed'),
                    'service_ok': service_ok,
                    'data_fresh': data_fresh,
                    'status': state['status'],
                    'generated_at': state['generated_at'],
                    'schema_version': state['schema_version'],
                },
                HTTPStatus.OK,
                send_body,
            )
            return
        if path.startswith('/api/'):
            self._send_json({'error': 'not found'}, HTTPStatus.NOT_FOUND, send_body)
            return
        if path.startswith('/plots/'):
            name = path.removeprefix('/plots/')
            plot_path = _safe_plot_path(
                self.server.state_builder.run_dir,
                name,
            )
            if '/' in name or plot_path is None:
                self._send_json({'error': 'not found'}, HTTPStatus.NOT_FOUND, send_body)
                return
            self._send_file(
                plot_path,
                'image/png',
                send_body,
            )
            return
        static_name = STATIC_ROUTES.get(path)
        if static_name is not None:
            content_type = mimetypes.guess_type(static_name)[0] or 'application/octet-stream'
            self._send_file(
                self.server.static_dir / static_name,
                content_type,
                send_body,
            )
            return
        self._send_json({'error': 'not found'}, HTTPStatus.NOT_FOUND, send_body)

    def _send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus,
        send_body: bool,
    ) -> None:
        body = json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(',', ':')
        ).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str, send_body: bool) -> None:
        try:
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError
            body = path.read_bytes()
        except OSError:
            self._send_json({'error': 'not found'}, HTTPStatus.NOT_FOUND, send_body)
            return
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        if send_body:
            self.wfile.write(body)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='将训练进度、资源监控和曲线以只读 HTML 面板提供。'
    )
    parser.add_argument('--run-dir', default=None, help='训练 run 目录；省略时自动选最近目录。')
    parser.add_argument(
        '--monitor-dir',
        default=None,
        help='资源监控目录；省略时自动选 runs/resource_monitor 中最近目录。',
    )
    parser.add_argument(
        '--control-dir',
        default=None,
        help='阶段控制目录；省略时自动选 runs/stage_control 中最近目录。',
    )
    parser.add_argument('--host', default=DEFAULT_HOST, help='监听地址，默认仅回环。')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='监听端口。')
    parser.add_argument(
        '--poll-history-limit',
        type=int,
        default=DEFAULT_HISTORY_LIMIT,
        help=f'每条曲线最多返回的点数（10-{MAX_HISTORY_LIMIT}）。',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not (1 <= args.port <= 65535):
        raise SystemExit('--port must be between 1 and 65535')
    if not (10 <= args.poll_history_limit <= MAX_HISTORY_LIMIT):
        raise SystemExit(
            f'--poll-history-limit must be between 10 and {MAX_HISTORY_LIMIT}'
        )
    project_root = Path(__file__).resolve().parents[1]
    builder = DashboardStateBuilder(
        project_root=project_root,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        monitor_dir=Path(args.monitor_dir) if args.monitor_dir else None,
        control_dir=Path(args.control_dir) if args.control_dir else None,
        history_limit=args.poll_history_limit,
    )
    static_dir = project_root / 'src' / 'daxigua_rl' / 'dashboard' / 'static'
    server = DashboardServer((args.host, args.port), builder, static_dir)
    print(
        f'[dashboard] http://{args.host}:{server.server_address[1]} '
        f'run={builder.run_dir.name if builder.run_dir else "auto:none"} '
        f'monitor={builder.monitor_dir.name if builder.monitor_dir else "auto:none"}',
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
