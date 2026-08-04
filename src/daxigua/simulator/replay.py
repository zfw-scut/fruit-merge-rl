"""把逐帧物理追踪保存为可移植归档和离线浏览器回放。"""

from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path
from urllib.parse import quote

import torch

from daxigua.core import fruit_name, fruit_radius

from .config import SimulatorConfig
from .replay_web import (
    render_replay_catalog,
    render_replay_document,
    render_replay_fragment,
)
from .types import BatchSimulationTrace


REPLAY_PAYLOAD_VERSION = 2
TRACE_ARCHIVE_VERSION = 4
DEFAULT_FRUIT_TEXTURE_DIR = (
    Path(__file__).resolve().parents[3] / 'assets' / 'fruits'
)


def _normalize_traces(trace):
    if isinstance(trace, BatchSimulationTrace):
        traces = (trace,)
    else:
        try:
            traces = tuple(trace)
        except TypeError as error:
            raise TypeError(
                'trace must be BatchSimulationTrace or a sequence of traces'
            ) from error
        if not traces or not all(
                isinstance(item, BatchSimulationTrace) for item in traces):
            raise TypeError(
                'trace sequence must contain BatchSimulationTrace values'
            )
    return traces


def _cpu_trace_without_redundant_copy(trace):
    for field_name in trace.__dataclass_fields__:
        value = getattr(trace, field_name)
        if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
            return trace.cpu()
    return trace


def trace_to_payload(trace, config=None, *, compact=False):
    """把一次或连续多次投放追踪转换为浏览器数据。

    默认返回带字段名的可读记录，保持早期调试调用兼容。HTML 写入器会启用
    ``compact``，把每帧的重复键名换成固定数组，长局文件会明显缩小。
    """

    traces = _normalize_traces(trace)
    config = config or SimulatorConfig()
    if not isinstance(config, SimulatorConfig):
        raise TypeError('config must be SimulatorConfig')
    traces = tuple(_cpu_trace_without_redundant_copy(item) for item in traces)
    first = traces[0]
    env_indices = first.env_indices.tolist()
    for item in traces[1:]:
        if item.env_indices.tolist() != env_indices:
            raise ValueError('all traces must contain the same environments')
        if (
                item.physics_fps != first.physics_fps
                or item.frame_stride != first.frame_stride):
            raise ValueError('all traces must use the same timing configuration')

    clips = [
        {
            'env': int(env_index),
            'drops': len(traces),
            'total_frames': 0,
            'records': [],
            'drop_summaries': [],
        }
        for env_index in env_indices
    ]
    frame_offsets = [0 for _ in clips]
    previous_finished = [False for _ in clips]
    for drop_index, item in enumerate(traces):
        for row, clip in enumerate(clips):
            record_count = int(item.record_counts[row])
            if record_count <= 0:
                raise ValueError('each trace row must contain at least one record')
            action = int(item.actions[row])
            final_local_frame = int(
                item.frame_numbers[row, record_count - 1]
            )
            clip['drop_summaries'].append({
                'drop': drop_index + 1,
                'action': action,
                'frames': final_local_frame,
                'score_delta': int(item.score_deltas[row]),
                'stable': bool(item.stable[row]),
                'done': bool(item.done[row]),
                'truncated': bool(item.truncated[row]),
                'settle_timeout': (
                    False
                    if item.settle_timeout is None
                    else bool(item.settle_timeout[row])
                ),
                'reset_before': previous_finished[row],
            })
            for record_index in range(record_count):
                active_slots = torch.nonzero(
                    item.active[row, record_index], as_tuple=False
                ).flatten()
                fruits = []
                for slot in active_slots.tolist():
                    fruits.append([
                        int(item.fruit_ids[row, record_index, slot]),
                        int(item.levels[row, record_index, slot]),
                        round(float(item.positions[row, record_index, slot, 0]), 3),
                        round(float(item.positions[row, record_index, slot, 1]), 3),
                        round(float(item.physics_radii[row, record_index, slot]), 3),
                        round(float(item.angles[row, record_index, slot]), 4),
                        round(float(item.velocities[row, record_index, slot, 0]), 3),
                        round(float(item.velocities[row, record_index, slot, 1]), 3),
                        round(
                            float(item.angular_velocities[
                                row, record_index, slot
                            ]),
                            4,
                        ),
                    ])
                local_frame = int(item.frame_numbers[row, record_index])
                frame = frame_offsets[row] + local_frame
                score = int(item.scores[row, record_index])
                merges = int(item.merge_counts[row, record_index])
                if compact:
                    record = [
                        frame,
                        local_frame,
                        drop_index + 1,
                        score,
                        merges,
                        fruits,
                    ]
                else:
                    record = {
                        'frame': frame,
                        'local_frame': local_frame,
                        'drop': drop_index + 1,
                        'action': action,
                        'drop_start': record_index == 0,
                        'reset': (
                            record_index == 0 and previous_finished[row]
                        ),
                        'score': score,
                        'merges': merges,
                        'fruits': fruits,
                    }
                clip['records'].append(record)
            frame_offsets[row] += final_local_frame
            clip['total_frames'] = frame_offsets[row]
            previous_finished[row] = bool(
                item.done[row] or item.truncated[row]
            )
    return {
        'format_version': REPLAY_PAYLOAD_VERSION,
        'compact_records': bool(compact),
        'record_schema': (
            ['frame', 'local_frame', 'drop', 'score', 'merges', 'fruits']
            if compact
            else None
        ),
        'board': {
            'width': config.board_width,
            'height': config.board_height,
            'spawn_y': config.spawn_y,
            'wall_width': config.wall_width,
        },
        'physics_fps': first.physics_fps,
        'frame_stride': first.frame_stride,
        'fruit_names': [fruit_name(level) for level in range(1, 12)],
        'fruit_display_radii': [fruit_radius(level) for level in range(1, 12)],
        'clips': clips,
    }


def load_fruit_texture_data_urls(texture_dir=None):
    """读取 11 张水果 PNG，并返回可嵌入离线 HTML 的 data URL。"""

    directory = (
        DEFAULT_FRUIT_TEXTURE_DIR
        if texture_dir is None
        else Path(texture_dir)
    )
    paths = [directory / f'{level:02d}.png' for level in range(1, 12)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            'fruit replay textures are incomplete: ' + ', '.join(missing)
        )
    urls = [None]
    for path in paths:
        content = path.read_bytes()
        if not content.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError(f'fruit texture is not a PNG file: {path}')
        encoded = base64.b64encode(content).decode('ascii')
        urls.append(f'data:image/png;base64,{encoded}')
    return urls


def _safe_json(value):
    return (
        json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        .replace('</', '<\\/')
        .replace('\u2028', '\\u2028')
        .replace('\u2029', '\\u2029')
    )


def _encode_payload(payload, *, compressed):
    raw = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    if not compressed:
        return 'json', _safe_json(payload)
    content = gzip.compress(
        raw.encode('utf-8'), compresslevel=6, mtime=0
    )
    encoded = base64.b64encode(content).decode('ascii')
    return 'gzip-base64', _safe_json(encoded)


def write_replay_html(
        path,
        trace,
        config=None,
        *,
        title='CUDA 物理回放',
        texture_dir=None,
        use_textures=True,
        compress_payload=True):
    """生成不依赖服务端或外部文件的完整回放 HTML。"""

    payload = trace_to_payload(trace, config, compact=True)
    textures = (
        load_fruit_texture_data_urls(texture_dir)
        if use_textures
        else [None] * 12
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_encoding, payload_data = _encode_payload(
        payload, compressed=compress_payload
    )
    html = render_replay_document(
        title=str(title),
        payload_data=payload_data,
        payload_encoding=payload_encoding,
        textures_json=_safe_json(textures),
    )
    output_path.write_text(html, encoding='utf-8')
    return output_path


def write_replay_fragment(
        path,
        trace,
        config=None,
        *,
        title='CUDA 物理回放',
        texture_dir=None,
        use_textures=True,
        compress_payload=True):
    """生成适合嵌入 Codex 对话、且不会与其他实例冲突的回放片段。"""

    payload = trace_to_payload(trace, config, compact=True)
    textures = (
        load_fruit_texture_data_urls(texture_dir)
        if use_textures
        else [None] * 12
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_encoding, payload_data = _encode_payload(
        payload, compressed=compress_payload
    )
    fragment = render_replay_fragment(
        title=str(title),
        payload_data=payload_data,
        payload_encoding=payload_encoding,
        textures_json=_safe_json(textures),
    )
    output_path.write_text(fragment, encoding='utf-8')
    return output_path


def save_trace_archive(path, trace, *, compression_level=1):
    """保存可重新渲染的追踪归档；``.gz`` 后缀启用快速压缩。"""

    traces = tuple(
        _cpu_trace_without_redundant_copy(item)
        for item in _normalize_traces(trace)
    )
    archive = {
        'format_version': TRACE_ARCHIVE_VERSION,
        'steps': [
            {
                field_name: getattr(step_trace, field_name)
                for field_name in step_trace.__dataclass_fields__
            }
            for step_trace in traces
        ],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == '.gz':
        with gzip.open(
                output_path, 'wb', compresslevel=int(compression_level)) as file:
            torch.save(archive, file)
    else:
        torch.save(archive, output_path)
    return output_path


def load_trace_archive(path):
    """以 ``weights_only`` 安全模式载入本项目的 PT 或 PT.GZ 追踪归档。"""

    input_path = Path(path)
    if input_path.suffix.lower() == '.gz':
        with gzip.open(input_path, 'rb') as file:
            archive = torch.load(
                file, map_location='cpu', weights_only=True
            )
    else:
        archive = torch.load(
            input_path, map_location='cpu', weights_only=True
        )
    if not isinstance(archive, dict) or not isinstance(
            archive.get('steps'), list):
        raise ValueError('trace archive does not contain a steps list')
    if not archive['steps']:
        raise ValueError('trace archive contains no steps')
    traces = []
    expected_fields = set(BatchSimulationTrace.__dataclass_fields__)
    for index, values in enumerate(archive['steps']):
        if not isinstance(values, dict):
            raise ValueError(f'trace step {index} is not a mapping')
        missing = expected_fields.difference(values)
        if missing:
            raise ValueError(
                f'trace step {index} misses fields: {sorted(missing)}'
            )
        traces.append(BatchSimulationTrace(**{
            name: values[name] for name in expected_fields
        }))
    return tuple(traces)


def _relative_replay_url(replay_path, catalog_directory):
    replay_path = Path(replay_path)
    if not replay_path.is_absolute():
        replay_path = replay_path.resolve()
    try:
        relative = os.path.relpath(replay_path, catalog_directory)
    except ValueError:
        return replay_path.as_uri()
    return quote(relative.replace('\\', '/'), safe='/:')


def write_replay_catalog(
        path,
        entries,
        *,
        title='回放目录',
        description='选择一局后在右侧加载；任一时刻只加载一个长局回放。'):
    """生成单标签页多局目录，通过 iframe 按需加载选中的回放。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for entry in entries:
        if 'replay' not in entry:
            raise ValueError('each catalog entry must contain replay')
        normalized.append({
            'env_index': entry.get('env_index'),
            'step_count': int(entry.get('step_count', 0)),
            'score': int(entry.get('score', 0)),
            'physics_frames': int(entry.get(
                'physics_frames_in_replay',
                entry.get('frames_in_replay', 0),
            )),
            'end_kind': str(entry.get('end_kind', 'unknown')),
            'href': _relative_replay_url(
                entry['replay'], output_path.parent.resolve()
            ),
            'trace': (
                str(entry['trace']) if entry.get('trace') else None
            ),
        })
    html = render_replay_catalog(
        title=str(title),
        description=str(description),
        entries_json=_safe_json(normalized),
    )
    output_path.write_text(html, encoding='utf-8')
    return output_path
