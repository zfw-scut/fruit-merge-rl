"""将本地统计产物整理为门户可读取的通用表格数据集。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Any


_TABLE_LABELS = {
    'lifecycle_by_level': '等级生命周期',
    'lifecycle_t_merge_histogram': 'T_merge 分布',
    'horizon_probabilities_by_level': '时间窗口概率',
    'factor_relationships_by_level': '单因素关系',
    'factor_interactions_by_level': '双因素交互',
}
_MAX_DATASETS = 12
_MAX_CSV_BYTES = 5 * 1024 * 1024
_MAX_TABLE_ROWS = 50_000


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _csv_value(raw: str) -> Any:
    value = raw.strip()
    if not value or value.lower() in {'nan', 'none', 'null'}:
        return None
    if value.lower() in {'true', 'false'}:
        return value.lower() == 'true'
    if re.fullmatch(r'[+-]?\d+', value):
        try:
            return int(value)
        except ValueError:
            return value
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else None


def _read_csv_table(path: Path) -> dict[str, Any] | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > _MAX_CSV_BYTES:
        return None

    rows: list[dict[str, Any]] = []
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            if not columns:
                return None
            for index, raw_row in enumerate(reader):
                if index >= _MAX_TABLE_ROWS:
                    break
                rows.append({column: _csv_value(raw_row.get(column, ''))
                             for column in columns})
    except (OSError, csv.Error):
        return None

    table_id = path.stem
    return {
        'id': table_id,
        'label': _TABLE_LABELS.get(
            table_id, table_id.replace('_', ' ').strip().title()
        ),
        'source': path.name,
        'columns': columns,
        'rows': rows,
        'row_count': len(rows),
        'truncated': len(rows) >= _MAX_TABLE_ROWS,
    }


def _relative(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def scan_analysis_datasets(project_root: Path) -> dict[str, Any]:
    """发现已汇总的统计目录，只读取小型清单与CSV分析表。"""

    analysis_root = project_root / 'runs' / 'analysis'
    if not analysis_root.exists():
        return {'available': False, 'datasets': []}

    manifests = sorted(
        analysis_root.glob('*/analysis/analysis_manifest.json'),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:_MAX_DATASETS]
    datasets: list[dict[str, Any]] = []
    for analysis_manifest_path in manifests:
        analysis_manifest = _read_json(analysis_manifest_path)
        if analysis_manifest is None:
            continue
        run_dir = analysis_manifest_path.parent.parent
        run_manifest = _read_json(run_dir / 'manifest.json') or {}
        analysis_dir = analysis_manifest_path.parent
        tables = [
            table
            for csv_path in sorted(analysis_dir.glob('*.csv'))
            if (table := _read_csv_table(csv_path)) is not None
        ]
        if not tables:
            continue

        purpose = str(run_manifest.get('purpose') or '')
        dataset_kind = (
            'merge_potential'
            if purpose == 'merge_potential_t_merge_collection'
            else purpose or 'generic'
        )
        result = analysis_manifest.get('result')
        result = result if isinstance(result, dict) else {}
        simulator = run_manifest.get('simulator_config')
        simulator = simulator if isinstance(simulator, dict) else {}
        parameters = run_manifest.get('parameters')
        parameters = parameters if isinstance(parameters, dict) else {}
        table_rows = run_manifest.get('table_rows')
        table_rows = table_rows if isinstance(table_rows, dict) else {}
        checkpoint = Path(str(run_manifest.get('checkpoint') or '')).name

        datasets.append({
            'id': _relative(project_root, run_dir),
            'kind': dataset_kind,
            'name': run_dir.name,
            'status': str(run_manifest.get('status') or 'summarized'),
            'created_at': run_manifest.get('created_at_utc'),
            'updated_at': run_manifest.get('updated_at_utc'),
            'run_dir': _relative(project_root, run_dir),
            'metadata': {
                'episodes': int(result.get(
                    'episodes', run_manifest.get('completed_episodes', 0)
                ) or 0),
                'transitions': int(run_manifest.get('transitions', 0) or 0),
                'unique_observed_fruits': int(result.get(
                    'unique_observed_fruits', 0
                ) or 0),
                'snapshot_rows': int(result.get(
                    'snapshot_rows', table_rows.get('snapshots', 0)
                ) or 0),
                'merge_sources': int(result.get(
                    'merge_sources', table_rows.get('merge_sources', 0)
                ) or 0),
                'physics_fps': int(simulator.get('physics_fps', 0) or 0),
                'drop_fast_forward': bool(simulator.get(
                    'drop_fast_forward', False
                )),
                'max_drops': int(parameters.get('max_drops', 0) or 0),
                'checkpoint': checkpoint,
                'checkpoint_sha256': str(
                    run_manifest.get('checkpoint_sha256') or ''
                ),
                'horizons': analysis_manifest.get('horizons', []),
                'factor_bins': int(analysis_manifest.get('factor_bins', 0) or 0),
                'interaction_bins': int(
                    analysis_manifest.get('interaction_bins', 0) or 0
                ),
            },
            'tables': tables,
        })

    return {'available': bool(datasets), 'datasets': datasets}
