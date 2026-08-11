"""训练队列的轻量只读契约。

队列文件位于 ``runs/training_queue.json``。训练面板只读取和校验它，
不负责启动、停止或重排训练，因此不会把可视化入口扩大为调度授权。
"""

from __future__ import annotations

import json
from pathlib import Path
import time


QUEUE_FORMAT_VERSION = 1
QUEUE_STATUSES = frozenset({
    'queued',
    'waiting',
    'preflight',
    'running',
    'evaluating',
    'completed',
    'failed',
    'cancelled',
})
_VISIBLE_FIELDS = (
    'id',
    'name',
    'status',
    'position',
    'run_dir',
    'config',
    'training_physics_fps',
    'planned_transitions',
    'source_run',
    'depends_on',
    'message',
    'enqueued_at',
    'started_at',
    'completed_at',
)


def queue_path_for_run(run_dir) -> Path:
    """返回同一 ``runs`` 根目录下的统一队列文件。"""

    return Path(run_dir).resolve().parent / 'training_queue.json'


def _safe_number(value, *, integer=False):
    if value is None:
        return None
    try:
        return int(value) if integer else float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _normalise_item(raw, index):
    if not isinstance(raw, dict):
        return None
    identifier = str(raw.get('id') or f'queue-{index + 1}').strip()[:128]
    name = str(raw.get('name') or identifier).strip()[:240]
    status = str(raw.get('status') or 'queued').strip().lower()
    if status not in QUEUE_STATUSES:
        status = 'queued'
    item = {
        key: raw.get(key)
        for key in _VISIBLE_FIELDS
        if key in raw
    }
    item.update({'id': identifier, 'name': name, 'status': status})
    item['position'] = _safe_number(
        raw.get('position', index + 1), integer=True
    )
    item['training_physics_fps'] = _safe_number(
        raw.get('training_physics_fps'), integer=True
    )
    item['planned_transitions'] = _safe_number(
        raw.get('planned_transitions'), integer=True
    )
    for key in ('enqueued_at', 'started_at', 'completed_at'):
        item[key] = _safe_number(raw.get(key))
    return item


def _current_item(run_dir, training, identity):
    training = training if isinstance(training, dict) else {}
    identity = identity if isinstance(identity, dict) else {}
    config = identity.get('training_config')
    config = config if isinstance(config, dict) else {}
    phase = str(training.get('phase') or 'running')
    status = phase if phase in QUEUE_STATUSES else 'running'
    return {
        'id': Path(run_dir).name,
        'name': Path(run_dir).name,
        'status': status,
        'position': 0,
        'run_dir': str(Path(run_dir)),
        'config': identity.get('config_path'),
        'training_physics_fps': _safe_number(
            config.get('training_physics_fps'), integer=True
        ),
        'planned_transitions': _safe_number(
            training.get('total_transitions')
            or config.get('total_transitions'),
            integer=True,
        ),
        'message': training.get('completion_message'),
        'started_at': identity.get('created_at'),
        'completed_at': training.get('completed_at'),
        'transitions': _safe_number(
            training.get('transitions'), integer=True
        ),
        'progress_fraction': _safe_number(
            training.get('progress_fraction')
        ),
    }


def load_training_queue(run_dir, *, training=None):
    """读取队列并合并当前run；损坏文件降级为空队列而不影响面板。"""

    run_dir = Path(run_dir).resolve()
    path = queue_path_for_run(run_dir)
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        payload = {}
    try:
        identity = json.loads(
            (run_dir / 'run_identity.json').read_text(encoding='utf-8')
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        identity = {}

    raw_items = payload.get('items', []) if isinstance(payload, dict) else []
    items = [
        item for index, raw in enumerate(raw_items[:64])
        if (item := _normalise_item(raw, index)) is not None
    ]
    current = _current_item(run_dir, training, identity)
    current_path = str(run_dir)
    matched = None
    for item in items:
        candidate = str(item.get('run_dir') or '')
        if item['id'] == current['id'] or candidate in (
                current_path, run_dir.name):
            matched = item
            break
    if matched is None:
        items.insert(0, current)
    else:
        matched.update({
            key: current[key]
            for key in (
                'status', 'message', 'completed_at', 'transitions',
                'progress_fraction',
            )
            if current.get(key) is not None
        })
        for key in (
                'run_dir', 'config', 'training_physics_fps',
                'planned_transitions', 'started_at'):
            if matched.get(key) is None and current.get(key) is not None:
                matched[key] = current[key]

    order = {
        'running': 0,
        'evaluating': 1,
        'preflight': 2,
        'waiting': 3,
        'queued': 4,
        'failed': 5,
        'completed': 6,
        'cancelled': 7,
    }
    items.sort(key=lambda item: (
        order.get(item['status'], 9),
        item.get('position') if item.get('position') is not None else 10 ** 9,
        item.get('enqueued_at') or 0,
    ))
    return {
        'format_version': QUEUE_FORMAT_VERSION,
        'updated_at': _safe_number(
            payload.get('updated_at') if isinstance(payload, dict) else None
        ) or time.time(),
        'source': str(path),
        'items': items,
        'counts': {
            status: sum(item['status'] == status for item in items)
            for status in QUEUE_STATUSES
        },
    }


def write_training_queue(path, items):
    """原子写入队列计划，供外部接力/调度脚本调用。"""

    path = Path(path)
    normalised = [
        item for index, raw in enumerate(list(items)[:64])
        if (item := _normalise_item(raw, index)) is not None
    ]
    payload = {
        'format_version': QUEUE_FORMAT_VERSION,
        'updated_at': time.time(),
        'items': normalised,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    temporary.replace(path)
    return path


__all__ = [
    'QUEUE_FORMAT_VERSION',
    'QUEUE_STATUSES',
    'load_training_queue',
    'queue_path_for_run',
    'write_training_queue',
]
