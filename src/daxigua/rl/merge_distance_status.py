"""合成步距预测工作流的只读门户摘要。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


COLLECTION_PURPOSE = 'merge_distance_predictor_collection'
DATASET_PURPOSE = 'merge_distance_predictor_dataset'
TRAINING_PURPOSE = 'merge_distance_predictor_training'
ACTIVE_STATUSES = {'running', 'labeling'}


def _number(value, default=0):
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return default


def _optional_number(*values):
    for value in values:
        if (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)):
            return value
    return None


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


def _read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_history(path, limit=100):
    rows = []
    try:
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    continue
                if not isinstance(value, dict):
                    continue
                validation = value.get('validation')
                validation = validation if isinstance(validation, dict) else {}
                rows.append({
                    'epoch': int(_number(value.get('epoch'))),
                    'train_loss': float(_number(value.get('train_loss'))),
                    'validation_nll': float(_number(validation.get('nll'))),
                    'validation_lifecycle_weighted_nll': float(_number(
                        validation.get('lifecycle_weighted_nll')
                    )),
                    'validation_exact_bin_accuracy': float(_number(
                        validation.get('exact_bin_accuracy')
                    )),
                    'validation_adjacent_bin_accuracy': float(_number(
                        validation.get('adjacent_bin_accuracy')
                    )),
                    'epoch_seconds': float(_number(
                        value.get('epoch_seconds')
                    )),
                })
    except OSError:
        pass
    return rows[-max(1, int(limit)):]


def _manifest_time(path, manifest):
    updated_at = manifest.get('updated_at_utc') or manifest.get(
        'created_at_utc'
    )
    try:
        modified_at = path.stat().st_mtime
    except OSError:
        modified_at = 0.0
    return updated_at, _timestamp(updated_at) or modified_at


def _common(path, manifest, project_root, now, kind, stale_after):
    updated_at, last_update = _manifest_time(path, manifest)
    status = str(manifest.get('status') or 'unknown')
    output_dir = path.parent
    return {
        'id': _relative_path(output_dir, project_root),
        'name': output_dir.name,
        'run_dir': _relative_path(output_dir, project_root),
        'kind': kind,
        'status': status,
        'phase': str(manifest.get('phase') or kind),
        'stale': bool(
            status in ACTIVE_STATUSES
            and last_update > 0
            and now - last_update > stale_after
        ),
        'created_at_utc': manifest.get('created_at_utc'),
        'updated_at_utc': updated_at,
        'elapsed_seconds': float(_number(manifest.get('elapsed_seconds'))),
        'failure': manifest.get('failure'),
        '_sort_timestamp': last_update,
    }


def _normalize_collection(path, manifest, project_root, now):
    parameters = manifest.get('parameters')
    parameters = parameters if isinstance(parameters, dict) else {}
    simulator = manifest.get('simulator_config')
    simulator = simulator if isinstance(simulator, dict) else {}
    rows = manifest.get('table_rows')
    rows = rows if isinstance(rows, dict) else manifest.get('flushed_rows')
    rows = rows if isinstance(rows, dict) else {}
    target = int(_number(parameters.get('episodes')))
    completed = int(_number(manifest.get('completed_episodes')))
    record = _common(
        path, manifest, project_root, now, 'collection', 120.0
    )
    record.update({
        'target_episodes': target,
        'completed_episodes': completed,
        'progress_fraction': (
            max(0.0, min(1.0, completed / target)) if target else 0.0
        ),
        'transitions': int(_number(manifest.get('transitions'))),
        'env_steps_per_second': float(_number(
            manifest.get('env_steps_per_second')
        )),
        'parallel_envs': int(_number(parameters.get('parallel_envs'))),
        'scene_rows': int(_number(rows.get('scenes'))),
        'merge_source_rows': int(_number(rows.get('merge_sources'))),
        'physics_fps': int(_number(simulator.get('physics_fps'))),
        'drop_fast_forward': bool(simulator.get('drop_fast_forward', False)),
        'checkpoint_name': Path(str(
            manifest.get('checkpoint') or ''
        )).name or None,
        'checkpoint_sha256': manifest.get('checkpoint_sha256'),
        'cuda_device_name': manifest.get('cuda_device_name'),
    })
    return record


def _normalize_dataset(path, manifest, project_root, now):
    total = int(_number(manifest.get('source_scene_shards')))
    completed = int(_number(manifest.get('completed_scene_shards')))
    record = _common(path, manifest, project_root, now, 'labeling', 600.0)
    record['name'] = f'{path.parent.parent.name} / 标签'
    record.update({
        'target_shards': total,
        'completed_shards': completed,
        'progress_fraction': (
            max(0.0, min(1.0, completed / total)) if total else (
                1.0 if record['status'] == 'complete' else 0.0
            )
        ),
        'scene_rows': int(_number(manifest.get('scene_rows'))),
        'resolved_fruit_samples': int(_number(
            manifest.get('resolved_fruit_samples')
        )),
        'unique_observed_fruits': int(_number(
            manifest.get('unique_observed_fruits')
        )),
    })
    return record


def _normalize_training(path, manifest, project_root, now):
    arguments = manifest.get('arguments')
    arguments = arguments if isinstance(arguments, dict) else {}
    history = _read_history(path.parent / 'metrics.jsonl')
    latest = history[-1] if history else {}
    total = int(_number(
        manifest.get('total_epochs'), _number(arguments.get('epochs'))
    ))
    completed = int(_number(
        manifest.get('completed_epochs'), _number(latest.get('epoch'))
    ))
    current = int(_number(
        manifest.get('current_epoch'), completed
    ))
    validation = manifest.get('latest_validation')
    validation = validation if isinstance(validation, dict) else {}
    record = _common(path, manifest, project_root, now, 'training', 600.0)
    record.update({
        'current_epoch': current,
        'completed_epochs': completed,
        'total_epochs': total,
        'progress_fraction': float(_number(
            manifest.get('progress_fraction'),
            min(1.0, completed / total) if total else 0.0,
        )),
        'epoch_batch': int(_number(manifest.get('epoch_batch'))),
        'train_loss': _optional_number(
            manifest.get('train_loss'), latest.get('train_loss')
        ),
        'validation_nll': _optional_number(
            validation.get('nll'), latest.get('validation_nll')
        ),
        'validation_lifecycle_weighted_nll': _optional_number(
            validation.get('lifecycle_weighted_nll'),
            latest.get('validation_lifecycle_weighted_nll'),
        ),
        'validation_exact_bin_accuracy': _optional_number(
            validation.get('exact_bin_accuracy'),
            latest.get('validation_exact_bin_accuracy'),
        ),
        'validation_adjacent_bin_accuracy': _optional_number(
            validation.get('adjacent_bin_accuracy'),
            latest.get('validation_adjacent_bin_accuracy'),
        ),
        'best_epoch': int(_number(manifest.get('best_epoch'))),
        'best_validation_nll': _optional_number(
            manifest.get('best_validation_nll')
        ),
        'parameter_count': int(_number(manifest.get('parameter_count'))),
        'device': manifest.get('device'),
        'cuda_device_name': manifest.get('cuda_device_name'),
        'checkpoint_name': Path(str(
            manifest.get('checkpoint') or ''
        )).name or None,
        'checkpoint_sha256': manifest.get('checkpoint_sha256'),
        'history': history,
    })
    record['progress_fraction'] = max(
        0.0, min(1.0, record['progress_fraction'])
    )
    return record


def _candidate_paths(project_root):
    runs_root = project_root / 'runs'
    analysis_root = runs_root / 'analysis'
    training_root = runs_root / 'merge_distance'
    candidates = []
    candidates.extend((
        (analysis_root / 'manifest.json', COLLECTION_PURPOSE),
        (training_root / 'manifest.json', TRAINING_PURPOSE),
    ))
    try:
        candidates.extend(
            (path, COLLECTION_PURPOSE)
            for path in analysis_root.glob('*/manifest.json')
        )
        candidates.extend(
            (path, DATASET_PURPOSE)
            for path in analysis_root.glob('*/predictor/dataset_manifest.json')
        )
        candidates.extend(
            (path, TRAINING_PURPOSE)
            for path in training_root.glob('*/manifest.json')
        )
    except OSError:
        pass
    return candidates


def scan_merge_distance_runs(project_root=None, *, limit=12):
    """扫描规范目录中的预测工作流清单和小型epoch日志。"""

    project_root = Path(project_root or Path(__file__).resolve().parents[3])
    now = time.time()
    records = []
    seen = set()
    normalizers = {
        COLLECTION_PURPOSE: _normalize_collection,
        DATASET_PURPOSE: _normalize_dataset,
        TRAINING_PURPOSE: _normalize_training,
    }
    for path, expected_purpose in _candidate_paths(project_root):
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to((project_root / 'runs').resolve())
        except (OSError, ValueError):
            continue
        if resolved in seen or path.is_symlink() or not path.is_file():
            continue
        seen.add(resolved)
        manifest = _read_json(path)
        if manifest is None or manifest.get('purpose') != expected_purpose:
            continue
        records.append(normalizers[expected_purpose](
            path, manifest, project_root, now
        ))
    records.sort(key=lambda item: item['_sort_timestamp'], reverse=True)
    current = next((
        item for item in records
        if item['status'] in ACTIVE_STATUSES and not item['stale']
    ), records[0] if records else None)
    for item in records:
        item.pop('_sort_timestamp', None)
    return {
        'available': bool(records),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'current': current,
        'runs': records[:max(1, int(limit))],
    }


__all__ = ['scan_merge_distance_runs']
