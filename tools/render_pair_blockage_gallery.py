"""把当前几何堵塞检测器的测试集识别结果渲染为人工画廊。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import torch  # noqa: E402

from daxigua.rl.merge_potential_stats import table_shards  # noqa: E402
from daxigua.rl.pair_blockage import (  # noqa: E402
    BLOCKAGE_AREA,
    BLOCKAGE_TABLE,
    PairBlockageModel,
    PairBlockageModelConfig,
    SPLIT_TEST,
    SPLIT_VALIDATION,
)


COLORS = (
    '#7d3c98', '#d35400', '#e67e22', '#f4d03f', '#73c66b', '#d94b45',
    '#f1948a', '#d4a017', '#8d6e63', '#239b56', '#117864',
)
CONFUSION_ORDER = ('TP', 'FP', 'FN', 'TN')
CONFUSION_TITLES = {
    'TP': 'Correct blocked-now detection',
    'FP': 'False blocked-now alarm',
    'FN': 'Missed blocked-now state',
    'TN': 'Correct current-safe detection',
}
BLOCKED_KINDS = frozenset(('TP', 'FN'))
CORRECT_KINDS = frozenset(('TP', 'TN'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='按等级和混淆类型渲染当前几何堵塞测试集场景。'
    )
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument(
        '--split', choices=('validation', 'test'), default='test'
    )
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--per-bucket', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument('--pool-multiplier', type=int, default=64)
    parser.add_argument('--transition-reserve-multiplier', type=int, default=16)
    parser.add_argument(
        '--autocast-bfloat16',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(argv)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    temporary.replace(path)


def _resolve_device(value):
    if value == 'auto':
        value = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(value)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA device is unavailable')
    return device


def confusion_kind(predicted, label):
    """返回当前帧二分类混淆类型。"""

    if bool(predicted):
        return 'TP' if bool(label) else 'FP'
    return 'FN' if bool(label) else 'TN'


def candidate_identity(candidate):
    """正例按事件去重，负例按同一局中的水果对去重。"""

    event_id = int(candidate['event_id'])
    if event_id >= 0:
        return ('event', event_id)
    first, second = sorted((
        int(candidate['fruit_id_i']), int(candidate['fruit_id_j'])
    ))
    return ('pair', int(candidate['episode_id']), first, second)


def select_distinct_candidates(candidates, count):
    selected = []
    identities = set()
    for candidate in sorted(
            candidates, key=lambda item: item['priority'], reverse=True):
        identity = candidate_identity(candidate)
        if identity in identities:
            continue
        identities.add(identity)
        selected.append(candidate)
        if len(selected) >= int(count):
            break
    return selected


def _autocast_context(device, enabled):
    if device.type == 'cuda' and enabled:
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return nullcontext()


def _scene_frame(columns, index):
    return {
        'pair_present': True,
        'label': bool(columns['label'][index].item()),
        'episode_id': int(columns['episode_id'][index].item()),
        'step': int(columns['step'][index].item()),
        'pair_slot_i': int(columns['pair_slot_i'][index].item()),
        'pair_slot_j': int(columns['pair_slot_j'][index].item()),
        'fruit_id_i': int(columns['fruit_id_i'][index].item()),
        'fruit_id_j': int(columns['fruit_id_j'][index].item()),
        'level': int(columns['level'][index].item()),
        'positions': columns['positions'][index].clone(),
        'levels': columns['levels'][index].clone(),
        'physics_radii': columns['physics_radii'][index].clone(),
        'active': columns['active'][index].clone(),
        'event_id': int(columns['event_id'][index].item()),
        'offset_from_onset': int(
            columns['offset_from_onset'][index].item()
        ),
        'offset_to_end': int(columns['offset_to_end'][index].item()),
    }


def _candidate(columns, index, probability, kind):
    candidate = _scene_frame(columns, index)
    candidate.update({
        'kind': kind,
        'probability': float(probability),
        'priority': (
            float(probability) if kind in ('TP', 'FP')
            else 1.0 - float(probability)
        ),
    })
    return candidate


def _append_pool(pool, candidates, limit):
    pool.extend(candidates)
    pool.sort(key=lambda item: item['priority'], reverse=True)
    del pool[int(limit):]


def collect_candidates(args):
    dataset_dir = Path(args.dataset_dir).resolve()
    paths = table_shards(
        dataset_dir, BLOCKAGE_TABLE, area=BLOCKAGE_AREA
    )
    if not paths:
        raise FileNotFoundError('current-blockage labeled shards not found')
    device = _resolve_device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    if checkpoint.get('model_type') != 'pair_geometry_current_blockage_v1':
        raise ValueError('unsupported current-blockage checkpoint')
    model = PairBlockageModel(
        PairBlockageModelConfig(**checkpoint['model_config'])
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    split_value = (
        SPLIT_VALIDATION if args.split == 'validation' else SPLIT_TEST
    )
    pool_limit = max(
        int(args.per_bucket),
        int(args.per_bucket) * int(args.pool_multiplier),
    )
    pools = {
        (level, kind): []
        for level in range(7, 12)
        for kind in CONFUSION_ORDER
    }
    counts = {
        str(level): {kind: 0 for kind in CONFUSION_ORDER}
        for level in range(7, 12)
    }
    with torch.inference_mode():
        for path in paths:
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if (
                    payload.get('format_version') != 1
                    or payload.get('table') != BLOCKAGE_TABLE):
                raise ValueError(f'invalid pair-blockage shard: {path}')
            shard = payload['columns']
            rows = int(shard['label'].shape[0])
            for start in range(0, rows, int(args.batch_size)):
                stop = min(rows, start + int(args.batch_size))
                selected_rows = torch.nonzero(
                    shard['split'][start:stop].eq(split_value),
                    as_tuple=False,
                ).flatten() + start
                if selected_rows.numel() == 0:
                    continue
                columns = {
                    name: values.index_select(0, selected_rows)
                    for name, values in shard.items()
                }
                model_columns = {
                    name: columns[name].to(device, non_blocking=True)
                    for name in model.REQUIRED_COLUMNS
                }
                with _autocast_context(
                        device, args.autocast_bfloat16):
                    probabilities = torch.sigmoid(model(model_columns))
                probabilities = probabilities.float().cpu()
                predicted = probabilities.ge(float(args.threshold))
                labels = columns['label'].bool()
                levels = columns['level'].long()
                for level in range(7, 12):
                    level_mask = levels.eq(level)
                    for kind in CONFUSION_ORDER:
                        positive_prediction = kind in ('TP', 'FP')
                        positive_label = kind in BLOCKED_KINDS
                        mask = (
                            level_mask
                            & predicted.eq(positive_prediction)
                            & labels.eq(positive_label)
                        )
                        indices = torch.nonzero(
                            mask, as_tuple=False
                        ).flatten()
                        counts[str(level)][kind] += int(indices.numel())
                        if indices.numel() == 0:
                            continue
                        priorities = (
                            probabilities[indices]
                            if positive_prediction
                            else 1.0 - probabilities[indices]
                        )
                        take = min(pool_limit, int(indices.numel()))
                        best = torch.topk(
                            priorities, take, sorted=False
                        ).indices
                        candidates = []
                        for local_index in indices[best].tolist():
                            candidates.append(_candidate(
                                columns,
                                local_index,
                                probabilities[local_index].item(),
                                kind,
                            ))
                        _append_pool(
                            pools[(level, kind)], candidates, pool_limit
                        )
    selected = {}
    for (level, kind), values in pools.items():
        count = int(args.per_bucket)
        if kind in BLOCKED_KINDS:
            count *= int(args.transition_reserve_multiplier)
        selected[(level, kind)] = select_distinct_candidates(values, count)
    return model, selected, counts, device


def _pair_key(frame):
    first, second = sorted((
        int(frame['fruit_id_i']), int(frame['fruit_id_j'])
    ))
    return int(frame['episode_id']), first, second


def _raw_pre_event_frame(columns, index, candidate):
    fruit_ids = columns['fruit_ids'][index]
    matches_i = torch.nonzero(
        fruit_ids.eq(int(candidate['fruit_id_i'])), as_tuple=False
    ).flatten()
    matches_j = torch.nonzero(
        fruit_ids.eq(int(candidate['fruit_id_j'])), as_tuple=False
    ).flatten()
    slot_i = int(matches_i[0].item()) if matches_i.numel() else -1
    slot_j = int(matches_j[0].item()) if matches_j.numel() else -1
    return {
        'pair_present': slot_i >= 0 and slot_j >= 0,
        'probability': None,
        'kind': None,
        'label': False,
        'episode_id': int(columns['episode_id'][index].item()),
        'step': int(columns['step'][index].item()),
        'pair_slot_i': slot_i,
        'pair_slot_j': slot_j,
        'fruit_id_i': int(candidate['fruit_id_i']),
        'fruit_id_j': int(candidate['fruit_id_j']),
        'level': int(candidate['level']),
        'positions': columns['positions'][index].clone(),
        'levels': columns['levels'][index].clone(),
        'physics_radii': columns['physics_radii'][index].clone(),
        'active': columns['active'][index].clone(),
        'event_id': -1,
        'offset_from_onset': 0,
        'offset_to_end': 0,
    }


def _predict_frames(
        model, frames, device, *, threshold, autocast_bfloat16):
    for frame in frames:
        if not frame.get('pair_present', True):
            frame['probability'] = None
            frame['kind'] = None
    predictable = [
        frame for frame in frames if frame.get('pair_present', True)
    ]
    if not predictable:
        return
    columns = {
        'positions': torch.stack([
            frame['positions'] for frame in predictable
        ]),
        'levels': torch.stack([frame['levels'] for frame in predictable]),
        'physics_radii': torch.stack([
            frame['physics_radii'] for frame in predictable
        ]),
        'active': torch.stack([frame['active'] for frame in predictable]),
        'pair_slot_i': torch.tensor([
            frame['pair_slot_i'] for frame in predictable
        ], dtype=torch.int64),
        'pair_slot_j': torch.tensor([
            frame['pair_slot_j'] for frame in predictable
        ], dtype=torch.int64),
    }
    model_columns = {
        name: value.to(device, non_blocking=True)
        for name, value in columns.items()
    }
    with torch.inference_mode(), _autocast_context(
            device, autocast_bfloat16):
        probabilities = torch.sigmoid(model(model_columns)).float().cpu()
    for frame, probability in zip(predictable, probabilities.tolist()):
        frame['probability'] = float(probability)
        frame['kind'] = confusion_kind(
            probability >= float(threshold), frame['label']
        )


def attach_positive_transitions(
        dataset_dir, selected, model, device, *, threshold,
        autocast_bfloat16=True):
    """为TP/FN样本补齐堵塞前一帧与事件起点帧。"""

    candidates = [
        candidate
        for values in selected.values()
        for candidate in values
        if candidate['label']
    ]
    if not candidates:
        return {'positive_cases': 0, 'complete_cases': 0}
    requests_by_pair = {}
    target_episodes = set()
    for candidate in candidates:
        candidate['onset_step'] = (
            int(candidate['step'])
            - int(candidate['offset_from_onset'])
        )
        candidate['before_onset_frame'] = None
        candidate['onset_frame'] = None
        key = _pair_key(candidate)
        requests_by_pair.setdefault(key, []).append(candidate)
        target_episodes.add(key[0])

    episode_values = torch.tensor(
        sorted(target_episodes), dtype=torch.int64
    )
    for path in table_shards(
            Path(dataset_dir).resolve(), BLOCKAGE_TABLE,
            area=BLOCKAGE_AREA):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        columns = payload['columns']
        relevant = torch.nonzero(
            torch.isin(columns['episode_id'], episode_values),
            as_tuple=False,
        ).flatten()
        for index in relevant.tolist():
            key = (
                int(columns['episode_id'][index].item()),
                *sorted((
                    int(columns['fruit_id_i'][index].item()),
                    int(columns['fruit_id_j'][index].item()),
                )),
            )
            requests = requests_by_pair.get(key)
            if not requests:
                continue
            step = int(columns['step'][index].item())
            event_id = int(columns['event_id'][index].item())
            is_blocked = bool(columns['label'][index].item())
            for candidate in requests:
                if step < candidate['onset_step'] and not is_blocked:
                    current = candidate['before_onset_frame']
                    if current is None or step > int(current['step']):
                        candidate['before_onset_frame'] = _scene_frame(
                            columns, index
                        )
                if event_id == int(candidate['event_id']):
                    current = candidate['onset_frame']
                    offset = int(
                        columns['offset_from_onset'][index].item()
                    )
                    if (
                            current is None
                            or offset < int(current['offset_from_onset'])):
                        candidate['onset_frame'] = _scene_frame(
                            columns, index
                        )

    missing_by_episode = {}
    for candidate in candidates:
        if candidate['before_onset_frame'] is None:
            missing_by_episode.setdefault(
                int(candidate['episode_id']), []
            ).append(candidate)
    if missing_by_episode:
        missing_episodes = torch.tensor(
            sorted(missing_by_episode), dtype=torch.int64
        )
        for path in table_shards(
                Path(dataset_dir).resolve(), 'pair_risk_exposures',
                area='raw'):
            payload = torch.load(
                path, map_location='cpu', weights_only=False
            )
            columns = payload['columns']
            relevant = torch.nonzero(
                torch.isin(columns['episode_id'], missing_episodes),
                as_tuple=False,
            ).flatten()
            for index in relevant.tolist():
                episode = int(columns['episode_id'][index].item())
                step = int(columns['step'][index].item())
                for candidate in missing_by_episode.get(episode, ()):
                    if step >= int(candidate['onset_step']):
                        continue
                    current = candidate['before_onset_frame']
                    if current is None or step > int(current['step']):
                        candidate['before_onset_frame'] = (
                            _raw_pre_event_frame(
                                columns, index, candidate
                            )
                        )

    inference_frames = []
    seen = set()
    for candidate in candidates:
        for name in ('before_onset_frame', 'onset_frame'):
            frame = candidate[name]
            if frame is None or not frame.get('pair_present', True):
                continue
            identity = (
                *_pair_key(frame), int(frame['step'])
            )
            if identity in seen:
                continue
            seen.add(identity)
            inference_frames.append(frame)
    _predict_frames(
        model, inference_frames, device,
        threshold=threshold,
        autocast_bfloat16=autocast_bfloat16,
    )
    probability_by_identity = {
        (*_pair_key(frame), int(frame['step'])): (
            frame['probability'], frame['kind']
        )
        for frame in inference_frames
    }
    complete = 0
    for candidate in candidates:
        for name in ('before_onset_frame', 'onset_frame'):
            frame = candidate[name]
            if frame is None:
                continue
            if frame.get('pair_present', True):
                probability, kind = probability_by_identity[
                    (*_pair_key(frame), int(frame['step']))
                ]
                frame['probability'] = probability
                frame['kind'] = kind
        candidate['transition'] = (
            {'role': 'before_onset',
             'frame': candidate['before_onset_frame']},
            {'role': 'onset', 'frame': candidate['onset_frame']},
            {'role': 'classified', 'frame': candidate},
        )
        if (
                candidate['before_onset_frame'] is not None
                and candidate['onset_frame'] is not None):
            complete += 1
    return {
        'positive_cases': len(candidates),
        'complete_cases': complete,
    }


def trim_gallery_selection(selected, count):
    """正例优先选择具有完整起点前后对照的高置信事件。"""

    result = {}
    for key, candidates in selected.items():
        kind = key[1]
        if kind in BLOCKED_KINDS:
            candidates = sorted(
                candidates,
                key=lambda item: (
                    item.get('before_onset_frame') is not None,
                    item.get('onset_frame') is not None,
                    int(item['step']) != int(item.get(
                        'onset_step', item['step']
                    )),
                    item['priority'],
                ),
                reverse=True,
            )
        result[key] = candidates[:int(count)]
    return result


def _board_config(dataset_dir, model_config):
    collection_manifest = _load_json(Path(dataset_dir) / 'manifest.json')
    simulator = collection_manifest.get('simulator_config') or {}
    return {
        'board_width': float(simulator.get(
            'board_width', model_config.board_width
        )),
        'board_height': float(simulator.get(
            'board_height', model_config.board_height
        )),
        'wall_width': float(simulator.get('wall_width', 20.0)),
        'spawn_y': float(simulator.get('spawn_y', 252.0)),
    }


def draw_scene(axis, candidate, board, *, show_title=True, title=None):
    width = float(board['board_width'])
    height = float(board['board_height'])
    wall = float(board['wall_width'])
    spawn_y = float(board['spawn_y'])
    kind = candidate['kind']
    correct = kind in CORRECT_KINDS
    accent = '#21a179' if correct else '#e63946'
    label_accent = '#d63031' if candidate['label'] else '#2d8f67'

    axis.set_facecolor('#f7f1e5')
    axis.add_patch(Rectangle(
        (wall, 0), width - 2.0 * wall, height - wall,
        fill=False, edgecolor='#4b3b2a', linewidth=2.2,
    ))
    axis.axhline(
        spawn_y, color='#c0392b', linewidth=1.0,
        linestyle='--', alpha=0.75,
    )
    target_slots = {
        int(candidate['pair_slot_i']), int(candidate['pair_slot_j'])
    }
    target_positions = []
    active_slots = torch.nonzero(
        candidate['active'], as_tuple=False
    ).flatten().tolist()
    for slot in active_slots:
        level = int(candidate['levels'][slot].item())
        x, y = (
            float(value) for value in candidate['positions'][slot].tolist()
        )
        radius = float(candidate['physics_radii'][slot].item())
        selected = slot in target_slots
        fruit_id = None
        if slot == int(candidate['pair_slot_i']):
            fruit_id = int(candidate['fruit_id_i'])
        elif slot == int(candidate['pair_slot_j']):
            fruit_id = int(candidate['fruit_id_j'])
        axis.add_patch(Circle(
            (x, y), radius,
            facecolor=COLORS[max(0, min(10, level - 1))],
            edgecolor=label_accent if selected else 'white',
            linewidth=4.0 if selected else 0.9,
            alpha=0.94,
            zorder=3 if selected else 2,
        ))
        label = f'L{level}'
        if fruit_id is not None:
            label += f'\n#{fruit_id}'
        axis.text(
            x, y, label,
            ha='center', va='center', fontsize=7,
            color='white', weight='bold', zorder=4,
        )
        if selected:
            target_positions.append((x, y))
    if len(target_positions) == 2:
        axis.plot(
            [target_positions[0][0], target_positions[1][0]],
            [target_positions[0][1], target_positions[1][1]],
            color=label_accent, linewidth=2.2, linestyle=':', zorder=5,
        )
    if show_title:
        axis.set_title(
            title or (
                f"L{int(candidate['level'])} {kind} · "
                f"score={float(candidate['probability']):.3f}"
            ),
            fontsize=10, color=accent, weight='bold', pad=5,
        )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])


def _semantic_lines(candidate, threshold):
    actual = 'BLOCKED NOW' if candidate['label'] else 'NOT BLOCKED NOW'
    predicted = (
        'BLOCKED NOW'
        if float(candidate['probability']) >= float(threshold)
        else 'NOT BLOCKED NOW'
    )
    lines = [
        f'Trajectory label: {actual}',
        f'Model decision: {predicted}',
        f'Blocked-now score: {float(candidate["probability"]):.4f}',
        f'Decision threshold: {float(threshold):.3f}',
        '',
        f'Episode: {int(candidate["episode_id"])}',
        f'Drop: {int(candidate["step"])}',
        f'Target pair: #{int(candidate["fruit_id_i"])} / '
        f'#{int(candidate["fruit_id_j"])}',
    ]
    if candidate['label']:
        lines.extend((
            '',
            f'From confirmed onset: '
            f'+{int(candidate["offset_from_onset"])} drops',
            f'To event end: {int(candidate["offset_to_end"])} drops',
            f'Event ID: {int(candidate["event_id"])}',
        ))
    else:
        lines.extend((
            '',
            'This frame is outside every blockage interval',
            'confirmed from the subsequent trajectory.',
        ))
    return lines


def _transition_title(role, frame, candidate):
    role_titles = {
        'before_onset': 'Before onset',
        'onset': 'Detected onset',
        'classified': 'Classified frame',
    }
    if frame is None:
        return f'{role_titles[role]}\nscene unavailable'
    if not frame.get('pair_present', True):
        label = 'TARGET PAIR NOT YET COMPLETE'
    else:
        label = 'BLOCKED' if frame['label'] else 'SAFE'
    probability = frame.get('probability')
    probability_text = (
        f'{float(probability):.3f}'
        if probability is not None else 'n/a'
    )
    same = (
        role == 'classified'
        and int(frame['step']) == int(candidate['onset_step'])
    )
    same_text = ' · same as onset' if same else ''
    return (
        f'{role_titles[role]}{same_text}\n'
        f'drop={int(frame["step"])} · '
        f'score={probability_text} · {label}'
    )


def _draw_missing_scene(axis, title):
    axis.set_facecolor('#f7f1e5')
    axis.axis('off')
    axis.text(
        0.5, 0.5, title,
        ha='center', va='center', transform=axis.transAxes,
        fontsize=10, color='#6b6259', weight='bold',
    )


def _draw_info(axis, candidate, threshold, *, compact=False):
    axis.axis('off')
    kind = candidate['kind']
    accent = '#21a179' if kind in CORRECT_KINDS else '#e63946'
    axis.text(
        0.0, 0.98,
        f"L{int(candidate['level'])} {kind}\n"
        f"{CONFUSION_TITLES[kind]}",
        ha='left', va='top', transform=axis.transAxes,
        fontsize=15 if compact else 17,
        color=accent, weight='bold', linespacing=1.35,
    )
    axis.text(
        0.0, 0.79, '\n'.join(_semantic_lines(candidate, threshold)),
        ha='left', va='top', transform=axis.transAxes,
        fontsize=9.2 if compact else 10.5,
        color='#312a25', linespacing=1.48,
    )
    axis.text(
        0.0, 0.025,
        'Every score classifies that displayed frame as blocked now.\n'
        'The preceding frame is context, not a future-risk forecast.',
        ha='left', va='bottom', transform=axis.transAxes,
        fontsize=8.4 if compact else 9.3,
        color='#6b6259', linespacing=1.4,
    )


def render_candidate(candidate, board, threshold, output_path):
    transition = candidate.get('transition')
    if candidate['label'] and transition:
        figure, axes = plt.subplots(
            1, 4, figsize=(14.2, 8.4),
            gridspec_kw={'width_ratios': (2.1, 2.1, 2.1, 1.65)},
            squeeze=False,
        )
        for axis, entry in zip(axes[0][:3], transition):
            frame = entry['frame']
            title = _transition_title(
                entry['role'], frame, candidate
            )
            if frame is None:
                _draw_missing_scene(axis, title)
            else:
                draw_scene(axis, frame, board, title=title)
        _draw_info(axes[0][3], candidate, threshold, compact=True)
        figure.tight_layout()
        figure.savefig(output_path, dpi=155, bbox_inches='tight')
        plt.close(figure)
        return

    figure, axes = plt.subplots(
        1, 2, figsize=(8.5, 9.0),
        gridspec_kw={'width_ratios': (3.0, 1.75)}, squeeze=False,
    )
    draw_scene(axes[0][0], candidate, board, show_title=False)
    _draw_info(axes[0][1], candidate, threshold)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def render_kind_overview(selected, board, kind, output_path):
    if kind in BLOCKED_KINDS:
        figure, axes = plt.subplots(
            5, 3, figsize=(9.8, 23.5), squeeze=False
        )
        for row, level in enumerate(range(7, 12)):
            candidate = (
                selected[(level, kind)][0]
                if selected[(level, kind)] else None
            )
            for column, role in enumerate((
                    'before_onset', 'onset', 'classified')):
                axis = axes[row][column]
                if candidate is None:
                    _draw_missing_scene(axis, f'L{level} {kind}\nno sample')
                    continue
                entry = candidate['transition'][column]
                frame = entry['frame']
                title = _transition_title(role, frame, candidate)
                if frame is None:
                    _draw_missing_scene(axis, title)
                else:
                    draw_scene(axis, frame, board, title=title)
                if column == 0:
                    axis.text(
                        -0.06, 0.5, f'L{level} {kind}',
                        rotation=90, ha='right', va='center',
                        transform=axis.transAxes,
                        fontsize=12, weight='bold',
                    )
        figure.suptitle(
            f'PB-GEO current-blockage transition gallery · {kind} · '
            f'{CONFUSION_TITLES[kind]}',
            fontsize=15, weight='bold', y=0.998,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.985))
        figure.savefig(output_path, dpi=125, bbox_inches='tight')
        plt.close(figure)
        return

    figure, axes = plt.subplots(1, 5, figsize=(15.5, 7.0), squeeze=False)
    for column, level in enumerate(range(7, 12)):
        candidate = selected[(level, kind)][0] if selected[(
            level, kind
        )] else None
        axis = axes[0][column]
        if candidate is None:
            axis.axis('off')
            axis.text(
                0.5, 0.5, f'L{level} {kind}\nno sample',
                ha='center', va='center', transform=axis.transAxes,
            )
        else:
            draw_scene(axis, candidate, board)
    figure.suptitle(
        f'PB-GEO current-blockage gallery · {kind} · '
        f'{CONFUSION_TITLES[kind]}',
        fontsize=15, weight='bold', y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close(figure)


def _report_candidate(candidate, filename):
    result = {
        'file': filename,
        'kind': candidate['kind'],
        'level': int(candidate['level']),
        'probability': float(candidate['probability']),
        'label': bool(candidate['label']),
        'episode_id': int(candidate['episode_id']),
        'step': int(candidate['step']),
        'fruit_id_i': int(candidate['fruit_id_i']),
        'fruit_id_j': int(candidate['fruit_id_j']),
        'event_id': int(candidate['event_id']),
        'offset_from_onset': int(candidate['offset_from_onset']),
        'offset_to_end': int(candidate['offset_to_end']),
    }
    transition = candidate.get('transition')
    if transition:
        result['transition'] = []
        for entry in transition:
            frame = entry['frame']
            result['transition'].append({
                'role': entry['role'],
                'available': frame is not None,
                'step': int(frame['step']) if frame is not None else None,
                'label': bool(frame['label']) if frame is not None else None,
                'probability': (
                    float(frame['probability'])
                    if (
                        frame is not None
                        and frame.get('probability') is not None
                    ) else None
                ),
                'kind': frame['kind'] if frame is not None else None,
            })
    return result


def run(args):
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'gallery output is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    model, selected, counts, device = collect_candidates(args)
    transition_scan = attach_positive_transitions(
        args.dataset_dir, selected, model, device,
        threshold=args.threshold,
        autocast_bfloat16=args.autocast_bfloat16,
    )
    selected = trim_gallery_selection(selected, args.per_bucket)
    rendered_positive = [
        candidate
        for values in selected.values()
        for candidate in values
        if candidate['label']
    ]
    transition_summary = {
        **transition_scan,
        'rendered_positive_cases': len(rendered_positive),
        'rendered_complete_cases': sum(
            candidate.get('before_onset_frame') is not None
            and candidate.get('onset_frame') is not None
            for candidate in rendered_positive
        ),
    }
    board = _board_config(args.dataset_dir, model.config)
    report_rows = []
    for level in range(7, 12):
        for kind in CONFUSION_ORDER:
            for index, candidate in enumerate(selected[(level, kind)], 1):
                filename = f'L{level}_{kind}_{index:02d}.png'
                render_candidate(
                    candidate, board, args.threshold, output_dir / filename
                )
                report_rows.append(_report_candidate(candidate, filename))
    overview_names = []
    for kind in CONFUSION_ORDER:
        filename = f'overview_{kind}.png'
        render_kind_overview(
            selected, board, kind, output_dir / filename
        )
        overview_names.append(filename)
    report = {
        'format_version': 2,
        'purpose': 'pair_blockage_current_geometry_human_gallery',
        'label_semantics': 'confirmed_event_onset_le_t_le_event_end',
        'prediction_semantics': 'blocked_now_score',
        'transition_semantics': (
            'positive_cases_show_previous_exposure_onset_and_classified_frame'
        ),
        'dataset_dir': str(Path(args.dataset_dir).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'device': str(device),
        'split': args.split,
        'threshold': float(args.threshold),
        'per_bucket': int(args.per_bucket),
        'confusion_counts': counts,
        'transition_summary': transition_summary,
        'overviews': overview_names,
        'images': report_rows,
    }
    _atomic_json(output_dir / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
