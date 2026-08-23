"""把轻量堵塞风险模型的测试集识别结果渲染为场景画廊。"""

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

from daxigua.rl.pair_risk import (  # noqa: E402
    EVENT_CONFIRMED,
    PairRiskModel,
    PairRiskModelConfig,
    SPLIT_TEST,
    SPLIT_VALIDATION,
)
from daxigua.rl.merge_potential_stats import (  # noqa: E402
    load_table_columns,
    table_shards,
)


COLORS = (
    '#7d3c98', '#d35400', '#e67e22', '#f4d03f', '#73c66b', '#d94b45',
    '#f1948a', '#d4a017', '#8d6e63', '#239b56', '#117864',
)
CONFUSION_ORDER = ('TP', 'FP', 'FN', 'TN')
CONFUSION_TITLES = {
    'TP': 'Correct risk',
    'FP': 'False alarm',
    'FN': 'Missed risk',
    'TN': 'Correct safe',
}
CORRECT_KINDS = frozenset(('TP', 'TN'))
POSITIVE_TIME_BUCKETS = (
    'at_or_after_onset', 'lead_1_4', 'lead_5_12', 'lead_13_24',
)
NEGATIVE_TIME_BUCKET = 'no_event'
TIME_BUCKET_TITLES = {
    'at_or_after_onset': 'At or after detected onset',
    'lead_1_4': 'Early warning: 1–4 drops',
    'lead_5_12': 'Early warning: 5–12 drops',
    'lead_13_24': 'Early warning: 13–24 drops',
    'no_event': 'No onset in forecast horizon',
}
FRAME_ROLE_TITLES = {
    'prediction': 'Prediction frame',
    'onset': 'Traditional onset',
    'confirmation': 'Traditional confirmation',
    'forecast_midpoint': 'Forecast midpoint',
    'forecast_end': 'Forecast horizon end',
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='按等级和混淆类型渲染堵塞风险测试集场景。'
    )
    parser.add_argument('dataset_dir', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument(
        '--split', choices=('validation', 'test'), default='test'
    )
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--per-bucket', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--pool-multiplier', type=int, default=32)
    parser.add_argument('--timeline-reserve-multiplier', type=int, default=8)
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
    if bool(predicted):
        return 'TP' if bool(label) else 'FP'
    return 'FN' if bool(label) else 'TN'


def warning_time_bucket(label, lead_to_onset):
    """把样本按相对堵塞起点的距离分组。"""

    if not bool(label):
        return NEGATIVE_TIME_BUCKET
    lead = int(lead_to_onset)
    if lead <= 0:
        return 'at_or_after_onset'
    if lead <= 4:
        return 'lead_1_4'
    if lead <= 12:
        return 'lead_5_12'
    return 'lead_13_24'


def timeline_target_specs(
        candidate, *, forecast_horizon, confirmation_drops,
        confirmed_step=None):
    """返回三联图各帧的语义角色和目标投放位置。"""

    step = int(candidate['step'])
    result = [{'role': 'prediction', 'target_step': step}]
    if bool(candidate['label']):
        onset = step + int(candidate['lead_to_onset'])
        confirmation = (
            int(confirmed_step)
            if confirmed_step is not None
            else onset + int(confirmation_drops)
        )
        result.extend((
            {'role': 'onset', 'target_step': onset},
            {'role': 'confirmation', 'target_step': confirmation},
        ))
    else:
        horizon = int(forecast_horizon)
        result.extend((
            {
                'role': 'forecast_midpoint',
                'target_step': step + max(1, horizon // 2),
            },
            {'role': 'forecast_end', 'target_step': step + horizon},
        ))
    return result


def _priority(kind, probability):
    probability = float(probability)
    return probability if kind in ('TP', 'FP') else -probability


def _identity(candidate):
    if int(candidate['event_id']) >= 0:
        return ('event', int(candidate['event_id']))
    return (
        'negative', int(candidate['episode_id']),
        int(candidate['fruit_id_i']), int(candidate['fruit_id_j']),
    )


def select_unique_candidates(candidates, count, *, seen=None):
    selected = []
    seen = set() if seen is None else seen
    for candidate in sorted(
            candidates, key=lambda item: item['priority'], reverse=True):
        identity = _identity(candidate)
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(candidate)
        if len(selected) >= int(count):
            break
    return selected


def _candidate(columns, row, probability, kind):
    scalar_names = (
        'episode_id', 'step', 'pair_slot_i', 'pair_slot_j',
        'fruit_id_i', 'fruit_id_j', 'level', 'label', 'event_id',
        'lead_to_onset', 'danger_progress', 'over_danger_line',
    )
    result = {
        name: columns[name][row].item() for name in scalar_names
    }
    for name in (
            'positions', 'levels', 'fruit_ids', 'physics_radii', 'age_frames',
            'active', 'fruit_queue'):
        result[name] = columns[name][row].detach().cpu().clone()
    result.update({
        'probability': float(probability),
        'kind': str(kind),
        'priority': _priority(kind, probability),
        'time_bucket': warning_time_bucket(
            result['label'], result['lead_to_onset']
        ),
    })
    return result


def _model_columns(columns, indices, device):
    return {
        name: columns[name].index_select(0, indices).to(
            device, non_blocking=True
        )
        for name in PairRiskModel.REQUIRED_COLUMNS
    }


@torch.inference_mode()
def collect_candidates(args):
    if not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError('threshold must be between 0 and 1')
    if args.per_bucket <= 0 or args.batch_size <= 0:
        raise ValueError('per-bucket and batch-size must be positive')
    if args.pool_multiplier <= 0:
        raise ValueError('pool-multiplier must be positive')
    if args.timeline_reserve_multiplier <= 0:
        raise ValueError('timeline-reserve-multiplier must be positive')
    dataset_dir = Path(args.dataset_dir).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    device = _resolve_device(args.device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint.get('model_type') != 'pair_conditioned_deep_sets_risk_v1':
        raise ValueError('unsupported pair-risk checkpoint')
    model = PairRiskModel(
        PairRiskModelConfig(**checkpoint['model_config'])
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    split_value = (
        SPLIT_TEST if args.split == 'test' else SPLIT_VALIDATION
    )
    paths = tuple(sorted(
        (dataset_dir / 'labeled').glob('pair_risk_samples-*.pt')
    ))
    if not paths:
        raise FileNotFoundError('pair-risk dataset has no labeled shards')
    pool_limit = int(args.per_bucket) * int(args.pool_multiplier)
    pools = {
        (level, kind, bucket): []
        for level in range(7, 12)
        for kind in CONFUSION_ORDER
        for bucket in (
            POSITIVE_TIME_BUCKETS
            if kind in ('TP', 'FN') else (NEGATIVE_TIME_BUCKET,)
        )
    }
    counts = {
        str(level): {kind: 0 for kind in CONFUSION_ORDER}
        for level in range(7, 12)
    }
    time_counts = {
        str(level): {
            kind: {
                bucket: 0
                for bucket in (
                    POSITIVE_TIME_BUCKETS
                    if kind in ('TP', 'FN') else (NEGATIVE_TIME_BUCKET,)
                )
            }
            for kind in CONFUSION_ORDER
        }
        for level in range(7, 12)
    }
    for path in paths:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if (
                payload.get('format_version') != 1
                or payload.get('table') != 'pair_risk_samples'):
            raise ValueError(f'invalid pair-risk shard: {path}')
        columns = payload['columns']
        split_rows = torch.nonzero(
            columns['split'].eq(split_value), as_tuple=False
        ).flatten()
        for rows in split_rows.split(int(args.batch_size)):
            autocast = (
                torch.autocast(device_type='cuda', dtype=torch.bfloat16)
                if device.type == 'cuda' and args.autocast_bfloat16
                else nullcontext()
            )
            with autocast:
                logits = model(_model_columns(columns, rows, device))
            probabilities = torch.sigmoid(logits.float()).cpu()
            labels = columns['label'].index_select(0, rows).bool()
            levels = columns['level'].index_select(0, rows).long()
            predicted = probabilities.ge(float(args.threshold))
            for level in range(7, 12):
                level_mask = levels.eq(level)
                for kind in CONFUSION_ORDER:
                    if kind == 'TP':
                        mask = level_mask & predicted & labels
                    elif kind == 'FP':
                        mask = level_mask & predicted & ~labels
                    elif kind == 'FN':
                        mask = level_mask & ~predicted & labels
                    else:
                        mask = level_mask & ~predicted & ~labels
                    matches = torch.nonzero(mask, as_tuple=False).flatten()
                    counts[str(level)][kind] += int(matches.numel())
                    if matches.numel() == 0:
                        continue
                    grouped_matches = {}
                    for local_row in matches.tolist():
                        source_row = int(rows[local_row].item())
                        bucket_name = warning_time_bucket(
                            bool(labels[local_row].item()),
                            int(columns['lead_to_onset'][source_row].item()),
                        )
                        grouped_matches.setdefault(bucket_name, []).append(
                            local_row
                        )
                    for bucket_name, bucket_rows in grouped_matches.items():
                        time_counts[str(level)][kind][bucket_name] += len(
                            bucket_rows
                        )
                        bucket_matches = torch.tensor(
                            bucket_rows, dtype=torch.long
                        )
                        priorities = (
                            probabilities[bucket_matches]
                            if kind in ('TP', 'FP')
                            else -probabilities[bucket_matches]
                        )
                        keep = min(pool_limit, int(bucket_matches.numel()))
                        chosen = bucket_matches[
                            torch.topk(priorities, keep).indices
                        ]
                        bucket = pools[(level, kind, bucket_name)]
                        for local_row in chosen.tolist():
                            source_row = int(rows[local_row].item())
                            bucket.append(_candidate(
                                columns, source_row,
                                float(probabilities[local_row].item()), kind,
                            ))
                        bucket.sort(
                            key=lambda item: item['priority'], reverse=True
                        )
                        del bucket[pool_limit:]
    selected = {}
    reserve_count = (
        int(args.per_bucket) * int(args.timeline_reserve_multiplier)
    )
    for level in range(7, 12):
        for kind in CONFUSION_ORDER:
            buckets = (
                POSITIVE_TIME_BUCKETS
                if kind in ('TP', 'FN') else (NEGATIVE_TIME_BUCKET,)
            )
            for bucket_name in buckets:
                selected[(level, kind, bucket_name)] = (
                    select_unique_candidates(
                        pools[(level, kind, bucket_name)],
                        reserve_count,
                    )
                )
    return model, selected, counts, time_counts, device


def _event_confirmation_index(dataset_dir):
    columns = load_table_columns(
        dataset_dir,
        'pair_risk_events',
        (
            'event_kind', 'episode_id', 'onset_step', 'event_step',
            'fruit_id_i', 'fruit_id_j',
        ),
        area='raw',
    )
    result = {}
    for kind, episode, onset, event_step, first, second in zip(
            columns['event_kind'].tolist(),
            columns['episode_id'].tolist(),
            columns['onset_step'].tolist(),
            columns['event_step'].tolist(),
            columns['fruit_id_i'].tolist(),
            columns['fruit_id_j'].tolist()):
        if int(kind) != EVENT_CONFIRMED:
            continue
        result[(
            int(episode), int(first), int(second), int(onset)
        )] = int(event_step)
    return result


def _copy_scene(columns, row, candidate):
    result = {
        'episode_id': int(columns['episode_id'][row].item()),
        'step': int(columns['step'][row].item()),
        'danger_progress': float(columns['danger_progress'][row].item()),
        'over_danger_line': bool(columns['over_danger_line'][row].item()),
    }
    for name in (
            'positions', 'levels', 'fruit_ids', 'physics_radii', 'age_frames',
            'active', 'fruit_queue'):
        result[name] = columns[name][row].detach().cpu().clone()
    active = result['active'].bool()
    fruit_ids = result['fruit_ids'].long()
    slots = []
    for target_id in (
            int(candidate['fruit_id_i']), int(candidate['fruit_id_j'])):
        matches = torch.nonzero(
            active & fruit_ids.eq(target_id), as_tuple=False
        ).flatten()
        slots.append(int(matches[0].item()) if matches.numel() else -1)
    result['pair_slot_i'], result['pair_slot_j'] = slots
    result['target_pair_present'] = all(slot >= 0 for slot in slots)
    result['probability'] = None
    return result


def _frame_model_columns(frames, device):
    result = {}
    for name in PairRiskModel.REQUIRED_COLUMNS:
        values = []
        for frame in frames:
            value = frame[name]
            if torch.is_tensor(value):
                values.append(value)
            else:
                values.append(torch.as_tensor(value))
        result[name] = torch.stack(values).to(device, non_blocking=True)
    return result


@torch.inference_mode()
def attach_timeline_frames(
        dataset_dir, selected, model, device, *, forecast_horizon,
        confirmation_drops, exposure_stride, batch_size,
        autocast_bfloat16):
    """为每个候选补齐起点/确认或窗口中点/终点场景。"""

    confirmation_index = _event_confirmation_index(dataset_dir)
    candidates = [
        candidate
        for values in selected.values()
        for candidate in values
    ]
    requests_by_episode = {}
    for gallery_id, candidate in enumerate(candidates):
        candidate['gallery_id'] = gallery_id
        candidate['forecast_horizon'] = int(forecast_horizon)
        candidate['confirmation_drops'] = int(confirmation_drops)
        confirmed_step = None
        if bool(candidate['label']):
            onset = int(candidate['step']) + int(candidate['lead_to_onset'])
            confirmed_step = confirmation_index.get((
                int(candidate['episode_id']), int(candidate['fruit_id_i']),
                int(candidate['fruit_id_j']), onset,
            ))
        specs = timeline_target_specs(
            candidate,
            forecast_horizon=forecast_horizon,
            confirmation_drops=confirmation_drops,
            confirmed_step=confirmed_step,
        )
        candidate['onset_step'] = (
            int(candidate['step']) + int(candidate['lead_to_onset'])
            if bool(candidate['label']) else None
        )
        candidate['confirmed_step'] = confirmed_step
        candidate['label_resolution_step'] = (
            int(candidate['step']) + int(forecast_horizon)
            + int(confirmation_drops)
            if not bool(candidate['label']) else None
        )
        timeline = []
        for spec in specs:
            entry = dict(spec)
            entry['frame'] = candidate if spec['role'] == 'prediction' else None
            entry['distance'] = 0 if spec['role'] == 'prediction' else None
            entry['rank'] = (0, 0) if spec['role'] == 'prediction' else None
            timeline.append(entry)
            if spec['role'] != 'prediction':
                requests_by_episode.setdefault(
                    int(candidate['episode_id']), []
                ).append((candidate, entry))
        candidate['timeline'] = timeline

    episode_ids = torch.tensor(
        sorted(requests_by_episode), dtype=torch.int64
    )
    max_distance = max(1, int(exposure_stride))
    if episode_ids.numel() > 0:
        for path in table_shards(
                dataset_dir, 'pair_risk_exposures', area='raw'):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if (
                    payload.get('format_version') != 1
                    or payload.get('table') != 'pair_risk_exposures'):
                raise ValueError(f'invalid pair-risk shard: {path}')
            columns = payload['columns']
            rows = torch.nonzero(
                torch.isin(columns['episode_id'], episode_ids),
                as_tuple=False,
            ).flatten()
            seen_episode_steps = set()
            for row in rows.tolist():
                episode = int(columns['episode_id'][row].item())
                step = int(columns['step'][row].item())
                scene_identity = (episode, step)
                if scene_identity in seen_episode_steps:
                    continue
                seen_episode_steps.add(scene_identity)
                for candidate, entry in requests_by_episode[episode]:
                    delta = step - int(entry['target_step'])
                    distance = abs(delta)
                    if distance > max_distance:
                        continue
                    side_penalty = (
                        0
                        if (
                            entry['role'] in ('onset', 'confirmation')
                            and delta >= 0
                        )
                        else 1
                    )
                    rank = (distance, side_penalty)
                    if entry['rank'] is not None and rank >= entry['rank']:
                        continue
                    entry['frame'] = _copy_scene(columns, row, candidate)
                    entry['distance'] = distance
                    entry['rank'] = rank

    scored_frames = []
    for candidate in candidates:
        candidate['probability'] = float(candidate['probability'])
        for entry in candidate['timeline'][1:]:
            frame = entry['frame']
            if frame is not None and frame['target_pair_present']:
                scored_frames.append(frame)
    for frames in (
            scored_frames[index:index + int(batch_size)]
            for index in range(0, len(scored_frames), int(batch_size))):
        autocast = (
            torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            if device.type == 'cuda' and autocast_bfloat16
            else nullcontext()
        )
        with autocast:
            logits = model(_frame_model_columns(frames, device))
        probabilities = torch.sigmoid(logits.float()).cpu().tolist()
        for frame, probability in zip(frames, probabilities):
            frame['probability'] = float(probability)
    return candidates


def timeline_candidate_quality(candidate):
    frames = [entry['frame'] for entry in candidate['timeline']]
    scene_count = sum(frame is not None for frame in frames)
    pair_count = sum(
        bool(frame.get('target_pair_present', True))
        for frame in frames if frame is not None
    )
    return (
        scene_count == len(frames),
        scene_count,
        pair_count,
        float(candidate['priority']),
    )


def select_complete_timelines(selected, count):
    """优先保留三帧完整、目标水果对可追踪的高置信案例。"""

    return {
        key: sorted(
            candidates, key=timeline_candidate_quality, reverse=True
        )[:int(count)]
        for key, candidates in selected.items()
    }


def _draw_scene(axis, frame, candidate, board, *, role, target_step):
    width = float(board['board_width'])
    height = float(board['board_height'])
    wall = float(board['wall_width'])
    spawn_y = float(board.get('spawn_y', 252.0))
    correct = candidate['kind'] in CORRECT_KINDS
    accent = '#21a179' if correct else '#e63946'
    if frame is None:
        axis.set_facecolor('#f7f1e5')
        axis.axis('off')
        axis.text(
            0.5, 0.5,
            f"{FRAME_ROLE_TITLES[role]}\n"
            f"target drop={int(target_step)}\nscene unavailable",
            ha='center', va='center', transform=axis.transAxes,
            fontsize=10, color='#6b6259', weight='bold',
        )
        return
    axis.set_facecolor('#f7f1e5')
    axis.add_patch(Rectangle(
        (wall, 0), width - 2.0 * wall, height - wall,
        fill=False, edgecolor='#4b3b2a', linewidth=2.2,
    ))
    axis.axhline(
        spawn_y, color='#c0392b', linewidth=1.0,
        linestyle='--', alpha=0.75,
    )
    pair_ids = {
        int(candidate['fruit_id_i']), int(candidate['fruit_id_j'])
    }
    pair_positions = []
    active_slots = torch.nonzero(
        frame['active'], as_tuple=False
    ).flatten().tolist()
    for slot in active_slots:
        level = int(frame['levels'][slot].item())
        x, y = (
            float(value) for value in frame['positions'][slot].tolist()
        )
        radius = float(frame['physics_radii'][slot].item())
        fruit_id = int(frame['fruit_ids'][slot].item())
        selected = fruit_id in pair_ids
        axis.add_patch(Circle(
            (x, y), radius,
            facecolor=COLORS[max(0, min(10, level - 1))],
            edgecolor=accent if selected else 'white',
            linewidth=4.0 if selected else 0.9,
            alpha=0.93,
            zorder=3 if selected else 2,
        ))
        axis.text(
            x, y, f'L{level}\n#{fruit_id}',
            ha='center', va='center', fontsize=7,
            color='white', weight='bold', zorder=4,
        )
        if selected:
            pair_positions.append((x, y))
    if len(pair_positions) == 2:
        axis.plot(
            [pair_positions[0][0], pair_positions[1][0]],
            [pair_positions[0][1], pair_positions[1][1]],
            color=accent, linewidth=2.0, linestyle=':', zorder=5,
        )
    probability = frame.get('probability')
    probability_text = (
        f'p={float(probability):.3f}'
        if probability is not None else 'p=n/a'
    )
    actual_step = int(frame['step'])
    step_text = f'drop={actual_step}'
    if actual_step != int(target_step):
        step_text += f' (target {int(target_step)})'
    if len(pair_positions) != 2:
        step_text += ' · target pair absent'
    axis.set_title(
        f"{FRAME_ROLE_TITLES[role]}\n{step_text} · {probability_text}",
        fontsize=9, color=accent, weight='bold', pad=5,
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect('equal')
    axis.set_xticks([])
    axis.set_yticks([])


def render_candidate(candidate, board, output_path):
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 8.3), squeeze=False)
    for axis, entry in zip(axes[0], candidate['timeline']):
        _draw_scene(
            axis, entry['frame'], candidate, board,
            role=entry['role'], target_step=entry['target_step'],
        )
    lead = int(candidate['lead_to_onset'])
    if bool(candidate['label']):
        relation = (
            f'onset in {lead} drops' if lead > 0
            else ('at onset' if lead == 0 else f'{-lead} drops after onset')
        )
    else:
        relation = (
            f"no onset through +{int(candidate['forecast_horizon'])} drops; "
            f"label resolved at drop {int(candidate['label_resolution_step'])}"
        )
    correct = candidate['kind'] in CORRECT_KINDS
    accent = '#21a179' if correct else '#e63946'
    figure.suptitle(
        f"L{int(candidate['level'])} {candidate['kind']} · "
        f"forecast p={float(candidate['probability']):.3f} · "
        f"episode={int(candidate['episode_id'])} · "
        f"prediction drop={int(candidate['step'])}\n"
        f"{TIME_BUCKET_TITLES[candidate['time_bucket']]} · {relation}",
        fontsize=13, color=accent, weight='bold', y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(figure)


def render_group_overview(selected, board, bucket_name, output_path):
    kinds = (
        ('FP', 'TN') if bucket_name == NEGATIVE_TIME_BUCKET
        else ('TP', 'FN')
    )
    figure, axes = plt.subplots(5, 6, figsize=(21.0, 27.0), squeeze=False)
    for row, level in enumerate(range(7, 12)):
        for kind_index, kind in enumerate(kinds):
            candidates = selected[(level, kind, bucket_name)]
            candidate = candidates[0] if candidates else None
            for frame_index in range(3):
                column = kind_index * 3 + frame_index
                axis = axes[row][column]
                if candidate is None:
                    axis.axis('off')
                    axis.text(
                        0.5, 0.5, f'L{level} {kind}\nno sample',
                        ha='center', va='center', transform=axis.transAxes,
                    )
                else:
                    entry = candidate['timeline'][frame_index]
                    _draw_scene(
                        axis, entry['frame'], candidate, board,
                        role=entry['role'],
                        target_step=entry['target_step'],
                    )
                if row == 0:
                    role = (
                        candidate['timeline'][frame_index]['role']
                        if candidate is not None else (
                            ('prediction', 'forecast_midpoint', 'forecast_end')
                            if bucket_name == NEGATIVE_TIME_BUCKET else
                            ('prediction', 'onset', 'confirmation')
                        )[frame_index]
                    )
                    axis.text(
                        0.5, 1.11,
                        f"{CONFUSION_TITLES[kind]}\n"
                        f"{FRAME_ROLE_TITLES[role]}",
                        ha='center', va='bottom',
                        transform=axis.transAxes, fontsize=11, weight='bold',
                    )
    figure.suptitle(
        f"Pair-risk temporal validation · {TIME_BUCKET_TITLES[bucket_name]}",
        fontsize=17, weight='bold', y=0.998,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.988))
    figure.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(figure)


def _report_candidate(candidate, filename):
    result = {
        'file': filename,
        'kind': candidate['kind'],
        'level': int(candidate['level']),
        'time_bucket': candidate['time_bucket'],
        'probability': float(candidate['probability']),
        'label': bool(candidate['label']),
        'episode_id': int(candidate['episode_id']),
        'step': int(candidate['step']),
        'event_id': int(candidate['event_id']),
        'lead_to_onset': int(candidate['lead_to_onset']),
        'fruit_id_i': int(candidate['fruit_id_i']),
        'fruit_id_j': int(candidate['fruit_id_j']),
        'onset_step': candidate['onset_step'],
        'confirmed_step': candidate['confirmed_step'],
        'label_resolution_step': candidate['label_resolution_step'],
        'frames': [],
    }
    for entry in candidate['timeline']:
        frame = entry['frame']
        result['frames'].append({
            'role': entry['role'],
            'target_step': int(entry['target_step']),
            'shown_step': int(frame['step']) if frame is not None else None,
            'probability': (
                float(frame['probability'])
                if frame is not None and frame.get('probability') is not None
                else None
            ),
            'target_pair_present': (
                bool(frame.get('target_pair_present', True))
                if frame is not None else False
            ),
        })
    return result


def run(args):
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'gallery output is not empty: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_manifest = _load_json(
        Path(args.dataset_dir).resolve() / 'manifest.json'
    )
    labeled_manifest = _load_json(
        Path(args.dataset_dir).resolve() / 'labeled' / 'manifest.json'
    )
    model, selected, counts, time_counts, device = collect_candidates(args)
    forecast_horizon = int(labeled_manifest['forecast_horizon'])
    confirmation_drops = int(labeled_manifest['confirmation_drops'])
    exposure_stride = int(
        (collection_manifest.get('parameters') or {}).get(
            'exposure_stride', 4
        )
    )
    attach_timeline_frames(
        Path(args.dataset_dir).resolve(),
        selected,
        model,
        device,
        forecast_horizon=forecast_horizon,
        confirmation_drops=confirmation_drops,
        exposure_stride=exposure_stride,
        batch_size=args.batch_size,
        autocast_bfloat16=args.autocast_bfloat16,
    )
    selected = select_complete_timelines(selected, args.per_bucket)
    model_config = model.config
    simulator = collection_manifest.get('simulator_config') or {}
    board = {
        'board_width': float(simulator.get(
            'board_width', model_config.board_width
        )),
        'board_height': float(simulator.get(
            'board_height', model_config.board_height
        )),
        'wall_width': float(simulator.get(
            'wall_width', model_config.wall_width
        )),
        'spawn_y': float(simulator.get('spawn_y', 252.0)),
    }
    report_rows = []
    for level in range(7, 12):
        for kind in CONFUSION_ORDER:
            buckets = (
                POSITIVE_TIME_BUCKETS
                if kind in ('TP', 'FN') else (NEGATIVE_TIME_BUCKET,)
            )
            for bucket_name in buckets:
                for index, candidate in enumerate(
                        selected[(level, kind, bucket_name)], 1):
                    filename = (
                        f'L{level}_{kind}_{bucket_name}_{index:02d}.png'
                    )
                    render_candidate(candidate, board, output_dir / filename)
                    report_rows.append(
                        _report_candidate(candidate, filename)
                    )
    overview_names = []
    for bucket_name in (*POSITIVE_TIME_BUCKETS, NEGATIVE_TIME_BUCKET):
        filename = f'overview_{bucket_name}.png'
        render_group_overview(
            selected, board, bucket_name, output_dir / filename
        )
        overview_names.append(filename)
    report = {
        'format_version': 2,
        'purpose': 'pair_risk_prediction_temporal_gallery',
        'dataset_dir': str(Path(args.dataset_dir).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'split': args.split,
        'threshold': float(args.threshold),
        'per_bucket': int(args.per_bucket),
        'timeline_reserve_multiplier': int(
            args.timeline_reserve_multiplier
        ),
        'forecast_horizon': forecast_horizon,
        'confirmation_drops': confirmation_drops,
        'exposure_stride': exposure_stride,
        'confusion_counts': counts,
        'confusion_counts_by_time_bucket': time_counts,
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
