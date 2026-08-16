"""Merge Potential 采集清单的只读面板摘要。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


PURPOSE = 'merge_potential_t_merge_collection'
ACTIVE_STATUSES = {'running'}


def _number(value, default=0):
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return default


def _timestamp(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def _relative_path(path, project_root):
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def _row_count(manifest, name):
    final_rows = manifest.get('table_rows')
    flushed_rows = manifest.get('flushed_rows')
    if isinstance(final_rows, dict) and name in final_rows:
        return int(_number(final_rows.get(name)))
    if isinstance(flushed_rows, dict):
        return int(_number(flushed_rows.get(name)))
    return 0


def _normalize_manifest(manifest_path, manifest, project_root, now):
    parameters = manifest.get('parameters')
    parameters = parameters if isinstance(parameters, dict) else {}
    simulator = manifest.get('simulator_config')
    simulator = simulator if isinstance(simulator, dict) else {}
    output_dir = manifest_path.parent
    status = str(manifest.get('status') or 'unknown')
    target = int(_number(parameters.get('episodes')))
    completed = int(_number(manifest.get('completed_episodes')))
    updated_at = (
        manifest.get('updated_at_utc')
        or manifest.get('created_at_utc')
    )
    try:
        modified_at = manifest_path.stat().st_mtime
    except OSError:
        modified_at = 0.0
    last_update = _timestamp(updated_at) or modified_at
    progress_interval = float(_number(
        parameters.get('progress_interval_seconds'), 10.0
    ))
    stale_after = max(120.0, progress_interval * 3.0)
    checkpoint = str(manifest.get('checkpoint') or '')
    progress = completed / target if target > 0 else 0.0
    return {
        'id': _relative_path(output_dir, project_root),
        'name': output_dir.name,
        'run_dir': _relative_path(output_dir, project_root),
        'status': status,
        'stale': bool(
            status in ACTIVE_STATUSES
            and last_update > 0
            and now - last_update > stale_after
        ),
        'created_at_utc': manifest.get('created_at_utc'),
        'updated_at_utc': updated_at,
        'target_episodes': target,
        'completed_episodes': completed,
        'progress_fraction': max(0.0, min(1.0, progress)),
        'transitions': int(_number(manifest.get('transitions'))),
        'decision_steps': int(_number(manifest.get('decision_steps'))),
        'elapsed_seconds': float(_number(manifest.get('elapsed_seconds'))),
        'env_steps_per_second': float(
            _number(manifest.get('env_steps_per_second'))
        ),
        'parallel_envs': int(_number(parameters.get('parallel_envs'))),
        'max_drops': int(_number(parameters.get('max_drops'))),
        'physics_fps': int(_number(simulator.get('physics_fps'))),
        'drop_fast_forward': bool(simulator.get('drop_fast_forward', False)),
        'snapshot_rows': _row_count(manifest, 'snapshots'),
        'merge_source_rows': _row_count(manifest, 'merge_sources'),
        'episode_rows': _row_count(manifest, 'episodes'),
        'peak_cuda_allocated_bytes': int(_number(
            manifest.get('peak_cuda_allocated_bytes')
        )),
        'peak_cuda_reserved_bytes': int(_number(
            manifest.get('peak_cuda_reserved_bytes')
        )),
        'checkpoint_name': Path(checkpoint).name if checkpoint else None,
        'checkpoint_sha256': manifest.get('checkpoint_sha256'),
        'device': manifest.get('device'),
        'cuda_device_name': manifest.get('cuda_device_name'),
        'analysis_ready': (output_dir / 'analysis' / 'analysis_manifest.json').is_file(),
        'failure': manifest.get('failure'),
        '_sort_timestamp': last_update,
    }


def scan_merge_potential_runs(project_root=None, *, limit=8):
    """扫描规范目录下的采集清单，不读取原始分片。"""

    project_root = Path(project_root or Path(__file__).resolve().parents[3])
    analysis_root = project_root / 'runs' / 'analysis'
    now = time.time()
    records = []
    candidates = [analysis_root / 'manifest.json']
    try:
        candidates.extend(analysis_root.glob('*/manifest.json'))
    except OSError:
        pass
    for manifest_path in candidates:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            continue
        try:
            manifest_path.resolve(strict=True).relative_to(
                analysis_root.resolve()
            )
        except (OSError, ValueError):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get('purpose') != PURPOSE:
            continue
        records.append(_normalize_manifest(
            manifest_path, manifest, project_root, now
        ))
    records.sort(key=lambda item: item['_sort_timestamp'], reverse=True)
    fresh_active = next((
        item for item in records
        if item['status'] in ACTIVE_STATUSES and not item['stale']
    ), None)
    current = fresh_active or (records[0] if records else None)
    for item in records:
        item.pop('_sort_timestamp', None)
    return {
        'available': bool(records),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'current': current,
        'runs': records[:max(1, int(limit))],
    }
