"""基于长期停滞事件监督的轻量单帧水果对风险模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import torch
from torch import nn

from daxigua.core import MAX_FRUIT_LEVEL, fruit_radius

from .merge_potential_stats import ShardedTensorWriter, table_shards


EVENT_CONFIRMED = 1
EVENT_ENDED = 2

END_NATURAL = 1
END_SIMULATOR_TRUNCATED = 2
END_COLLECTOR_STOP = 3
END_DROP_LIMIT = 4

SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_TEST = 2


EXPOSURE_DTYPES = {
    'episode_id': torch.int64,
    'step': torch.int32,
    'pair_slot_i': torch.int16,
    'pair_slot_j': torch.int16,
    'fruit_id_i': torch.int64,
    'fruit_id_j': torch.int64,
    'level': torch.int8,
    'positions': torch.float32,
    'velocities': torch.float32,
    'angular_velocities': torch.float32,
    'levels': torch.int8,
    'fruit_ids': torch.int64,
    'physics_radii': torch.float32,
    'age_frames': torch.int32,
    'active': torch.bool,
    'fruit_queue': torch.int8,
    'danger_progress': torch.float32,
    'over_danger_line': torch.bool,
}

EVENT_DTYPES = {
    'episode_id': torch.int64,
    'event_kind': torch.int8,
    'onset_step': torch.int32,
    'event_step': torch.int32,
    'fruit_id_i': torch.int64,
    'fruit_id_j': torch.int64,
    'level': torch.int8,
}

EPISODE_DTYPES = {
    'episode_id': torch.int64,
    'seed': torch.int64,
    'end_step': torch.int32,
    'end_kind': torch.int8,
}

LABELED_DTYPES = {
    **EXPOSURE_DTYPES,
    'label': torch.bool,
    'split': torch.int8,
    'event_id': torch.int64,
    'lead_to_onset': torch.int16,
}


@dataclass(frozen=True, slots=True)
class PairRiskModelConfig:
    """Pair-conditioned Deep Sets 的轻量结构参数。"""

    max_fruits: int = 64
    queue_length: int = 4
    level_embedding_dim: int = 8
    context_hidden_dim: int = 48
    head_hidden_dim: int = 64
    board_width: float = 560.0
    board_height: float = 1120.0
    wall_width: float = 20.0
    physics_fps: float = 30.0

    def __post_init__(self):
        for name in (
                'max_fruits', 'queue_length', 'level_embedding_dim',
                'context_hidden_dim', 'head_hidden_dim'):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in (
                'board_width', 'board_height', 'wall_width', 'physics_fps'):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f'{name} must be positive')


def _mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class PairRiskModel(nn.Module):
    """不构图，以目标水果对为坐标系汇总其余水果。"""

    REQUIRED_COLUMNS = (
        'positions', 'levels', 'physics_radii', 'age_frames', 'active',
        'fruit_queue', 'danger_progress', 'over_danger_line',
        'pair_slot_i', 'pair_slot_j',
    )

    def __init__(self, config: PairRiskModelConfig | None = None):
        super().__init__()
        self.config = config or PairRiskModelConfig()
        level_dim = self.config.level_embedding_dim
        context_hidden = self.config.context_hidden_dim
        self.level_embedding = nn.Embedding(
            MAX_FRUIT_LEVEL + 1, level_dim, padding_idx=0
        )
        self.context_encoder = _mlp(
            level_dim + 10, context_hidden, context_hidden
        )
        # 配对静态几何14维、配对等级、mean/max上下文、4个队列等级和3个全局量。
        head_input = (
            14 + level_dim + context_hidden * 2
            + self.config.queue_length * level_dim + 3
        )
        self.risk_head = nn.Sequential(
            nn.Linear(head_input, self.config.head_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden_dim, 1),
        )
        result_radii = [0.0]
        for level in range(1, MAX_FRUIT_LEVEL + 1):
            result_radii.append(max(0.0, float(fruit_radius(level)) - 1.0))
        self.register_buffer(
            'result_radii', torch.tensor(result_radii, dtype=torch.float32)
        )

    def forward(self, columns):
        missing = set(self.REQUIRED_COLUMNS) - set(columns)
        if missing:
            raise ValueError(f'pair-risk input misses columns: {sorted(missing)}')
        position = columns['positions'].float()
        levels = columns['levels'].long().clamp(0, MAX_FRUIT_LEVEL)
        radii = columns['physics_radii'].float()
        age_frames = columns['age_frames'].float()
        active = columns['active'].bool()
        queue = columns['fruit_queue'].long().clamp(0, MAX_FRUIT_LEVEL)
        batch_size, fruit_count, _ = position.shape
        if fruit_count != self.config.max_fruits:
            raise ValueError('pair-risk fruit capacity does not match config')
        batch = torch.arange(batch_size, device=position.device)
        slot_i = columns['pair_slot_i'].long()
        slot_j = columns['pair_slot_j'].long()
        if slot_i.shape != (batch_size,) or slot_j.shape != (batch_size,):
            raise ValueError('pair slots must have shape [batch]')

        position_i = position[batch, slot_i]
        position_j = position[batch, slot_j]
        radius_i = radii[batch, slot_i]
        radius_j = radii[batch, slot_j]
        pair_level = levels[batch, slot_i]

        delta = position_j - position_i
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        axis = delta / distance.unsqueeze(-1)
        flip = (axis[:, 0] < 0.0) | (
            axis[:, 0].abs().le(1e-6) & axis[:, 1].lt(0.0)
        )
        axis = torch.where(flip[:, None], -axis, axis)
        perpendicular = torch.stack((-axis[:, 1], axis[:, 0]), dim=-1)
        fraction = radius_i / (radius_i + radius_j).clamp_min(1.0)
        result_center = position_i + fraction[:, None] * delta
        result_exists = pair_level < MAX_FRUIT_LEVEL
        result_level = (pair_level + 1).clamp_max(MAX_FRUIT_LEVEL)
        result_radius = self.result_radii[result_level]
        result_radius = torch.where(
            result_exists, result_radius, torch.zeros_like(result_radius)
        )

        width = float(self.config.board_width)
        height = float(self.config.board_height)
        wall = float(self.config.wall_width)
        radius_min = torch.minimum(radius_i, radius_j).clamp_min(1.0)
        radius_max = torch.maximum(radius_i, radius_j)
        surface_gap = distance - radius_i - radius_j
        left_clearance = result_center[:, 0] - wall - result_radius
        right_clearance = width - wall - result_radius - result_center[:, 0]
        bottom_clearance = height - wall - result_radius - result_center[:, 1]
        pair_features = torch.stack((
            result_center[:, 0] / width * 2.0 - 1.0,
            result_center[:, 1] / height * 2.0 - 1.0,
            axis[:, 0],
            axis[:, 1],
            distance / width,
            surface_gap / radius_min,
            radius_min / width,
            radius_max / width,
            result_radius / width,
            result_exists.to(position.dtype),
            left_clearance / width,
            right_clearance / width,
            bottom_clearance / height,
            (position_i[:, 1] - position_j[:, 1]).abs() / height,
        ), dim=-1)

        relative = position - result_center[:, None, :]
        along = (relative * axis[:, None, :]).sum(dim=-1)
        across = (relative * perpendicular[:, None, :]).sum(dim=-1)
        center_distance = torch.linalg.vector_norm(relative, dim=-1)
        level_delta = levels.float() - pair_level[:, None].float()
        age_seconds = age_frames / float(self.config.physics_fps)
        context_continuous = torch.stack((
            along / width,
            across / width,
            center_distance / width,
            (center_distance - radii - result_radius[:, None]) / width,
            radii / width,
            levels.float() / MAX_FRUIT_LEVEL,
            level_delta / MAX_FRUIT_LEVEL,
            torch.log1p(age_seconds.clamp_min(0.0)) / math.log(61.0),
            position[..., 0] / width * 2.0 - 1.0,
            position[..., 1] / height * 2.0 - 1.0,
        ), dim=-1)
        context = self.context_encoder(torch.cat((
            self.level_embedding(levels), context_continuous
        ), dim=-1))
        fruit_slots = torch.arange(fruit_count, device=position.device)
        context_mask = active & (
            fruit_slots[None, :] != slot_i[:, None]
        ) & (fruit_slots[None, :] != slot_j[:, None])
        mask_float = context_mask.unsqueeze(-1).to(context.dtype)
        count = context_mask.sum(dim=1, keepdim=True).clamp_min(1)
        context_mean = (context * mask_float).sum(dim=1) / count.to(
            context.dtype
        )
        fill = torch.finfo(context.dtype).min
        context_max = context.masked_fill(
            ~context_mask.unsqueeze(-1), fill
        ).amax(dim=1)
        context_max = torch.where(
            context_mask.any(dim=1, keepdim=True),
            context_max,
            torch.zeros_like(context_max),
        )

        queue_hidden = self.level_embedding(queue).flatten(1)
        global_features = torch.stack((
            active.sum(dim=1).to(position.dtype) / self.config.max_fruits,
            columns['danger_progress'].float(),
            columns['over_danger_line'].float(),
        ), dim=-1)
        hidden = torch.cat((
            pair_features,
            self.level_embedding(pair_level),
            context_mean,
            context_max,
            queue_hidden,
            global_features,
        ), dim=-1)
        return self.risk_head(hidden).squeeze(-1)


def canonical_pair_ids(first, second):
    return torch.minimum(first, second), torch.maximum(first, second)


def extract_pair_exposures(
        observation,
        episode_ids,
        pair_i,
        pair_j,
        *,
        exposure_stride=4):
    """提取当前决策状态中全部同级L7～L11水果对。"""

    if int(exposure_stride) <= 0:
        raise ValueError('exposure_stride must be positive')
    levels_i = observation.levels[:, pair_i]
    levels_j = observation.levels[:, pair_j]
    ids_i = observation.fruit_ids[:, pair_i]
    ids_j = observation.fruit_ids[:, pair_j]
    valid = (
        (episode_ids >= 0)[:, None]
        & observation.active[:, pair_i]
        & observation.active[:, pair_j]
        & ids_i.gt(0)
        & ids_j.gt(0)
        & levels_i.eq(levels_j)
        & levels_i.ge(7)
        & levels_i.le(11)
        & observation.step_count.remainder(int(exposure_stride))[:, None].eq(0)
    )
    indices = torch.nonzero(valid, as_tuple=False)
    rows = indices[:, 0]
    pair_indices = indices[:, 1]
    slots_i = pair_i[pair_indices]
    slots_j = pair_j[pair_indices]
    selected_ids_i = ids_i[rows, pair_indices]
    selected_ids_j = ids_j[rows, pair_indices]
    low_id, high_id = canonical_pair_ids(selected_ids_i, selected_ids_j)
    return {
        'episode_id': episode_ids[rows],
        'step': observation.step_count[rows],
        'pair_slot_i': slots_i,
        'pair_slot_j': slots_j,
        'fruit_id_i': low_id,
        'fruit_id_j': high_id,
        'level': levels_i[rows, pair_indices],
        'positions': observation.positions[rows],
        'velocities': observation.velocities[rows],
        'angular_velocities': observation.angular_velocities[rows],
        'levels': observation.levels[rows],
        'fruit_ids': observation.fruit_ids[rows],
        'physics_radii': observation.physics_radii[rows],
        'age_frames': observation.age_frames[rows],
        'active': observation.active[rows],
        'fruit_queue': observation.fruit_queue[rows],
        'danger_progress': observation.danger_progress[rows],
        'over_danger_line': observation.over_danger_line[rows],
    }


def extract_pair_events(update, episode_ids, step_count):
    """把确认和确认后结束事件压成一个定长列集合。"""

    parts = []
    for event_kind, mask in (
            (EVENT_CONFIRMED, update.confirmed),
            (EVENT_ENDED, update.ended & update.ended_after_confirmation)):
        mask = mask & (episode_ids >= 0)[:, None]
        indices = torch.nonzero(mask, as_tuple=False)
        rows = indices[:, 0]
        pairs = indices[:, 1]
        first, second = canonical_pair_ids(
            update.fruit_id_i[rows, pairs],
            update.fruit_id_j[rows, pairs],
        )
        parts.append({
            'episode_id': episode_ids[rows],
            'event_kind': torch.full_like(
                rows, int(event_kind), dtype=torch.int8
            ),
            'onset_step': update.onset_steps[rows, pairs],
            'event_step': step_count[rows],
            'fruit_id_i': first,
            'fruit_id_j': second,
            'level': update.levels[rows, pairs],
        })
    return {
        name: torch.cat(tuple(part[name] for part in parts), dim=0)
        for name in EVENT_DTYPES
    }


def extract_episode_rows(
        observation, rows, episode_ids, seeds, end_kinds):
    return {
        'episode_id': episode_ids[rows],
        'seed': seeds[rows],
        'end_step': observation.step_count[rows],
        'end_kind': end_kinds,
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)


def _load_small_table(dataset_dir, table):
    parts = None
    for path in table_shards(dataset_dir, table, area='raw'):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('format_version') != 1 or payload.get('table') != table:
            raise ValueError(f'invalid {table} shard: {path}')
        columns = payload['columns']
        if parts is None:
            parts = {name: [] for name in columns}
        for name, value in columns.items():
            parts[name].append(value)
    if parts is None:
        return {}
    return {
        name: torch.cat(values, dim=0) for name, values in parts.items()
    }


def _event_index(dataset_dir):
    episodes = _load_small_table(dataset_dir, 'pair_risk_episodes')
    events = _load_small_table(dataset_dir, 'pair_risk_events')
    if not episodes:
        raise ValueError('pair-risk dataset has no episode table')
    episode_end = {
        int(episode): (int(end_step), int(end_kind))
        for episode, end_step, end_kind in zip(
            episodes['episode_id'].tolist(),
            episodes['end_step'].tolist(),
            episodes['end_kind'].tolist(),
        )
    }
    end_by_key = {}
    if events:
        for kind, episode, onset, step, first, second in zip(
                events['event_kind'].tolist(),
                events['episode_id'].tolist(),
                events['onset_step'].tolist(),
                events['event_step'].tolist(),
                events['fruit_id_i'].tolist(),
                events['fruit_id_j'].tolist()):
            if int(kind) == EVENT_ENDED:
                key = (int(episode), int(first), int(second), int(onset))
                end_by_key.setdefault(key, []).append(int(step))
    indexed = {}
    next_event_id = 0
    if events:
        rows = zip(
            events['event_kind'].tolist(),
            events['episode_id'].tolist(),
            events['onset_step'].tolist(),
            events['event_step'].tolist(),
            events['fruit_id_i'].tolist(),
            events['fruit_id_j'].tolist(),
            events['level'].tolist(),
        )
        for kind, episode, onset, step, first, second, level in rows:
            if int(kind) != EVENT_CONFIRMED:
                continue
            key = (int(episode), int(first), int(second), int(onset))
            episode_step = episode_end.get(int(episode), (int(step), 0))[0]
            candidates = [
                value for value in end_by_key.get(key, ())
                if value >= int(step)
            ]
            end_step = min(candidates) if candidates else episode_step
            pair_key = (int(episode), int(first), int(second))
            indexed.setdefault(pair_key, []).append({
                'event_id': next_event_id,
                'level': int(level),
                'onset': int(onset),
                'confirmed': int(step),
                'end': int(end_step),
            })
            next_event_id += 1
    for values in indexed.values():
        values.sort(key=lambda item: item['onset'])
    return episode_end, indexed, next_event_id


def _stable_episode_key(episode_id):
    return (
        int(episode_id) * 6_364_136_223_846_793_005
        + 1_442_695_040_888_963_407
    ) & ((1 << 64) - 1)


def _split_targets(count):
    count = int(count)
    if count <= 1:
        return {
            SPLIT_TRAIN: count,
            SPLIT_VALIDATION: 0,
            SPLIT_TEST: 0,
        }
    if count == 2:
        return {
            SPLIT_TRAIN: 1,
            SPLIT_VALIDATION: 1,
            SPLIT_TEST: 0,
        }
    validation = max(1, round(count * 0.1))
    test = max(1, round(count * 0.1))
    if validation + test >= count:
        validation = 1
        test = 1
    return {
        SPLIT_TRAIN: count - validation - test,
        SPLIT_VALIDATION: validation,
        SPLIT_TEST: test,
    }


def stratified_episode_splits(episode_ids, events_by_pair):
    """按完整对局分组，并优先平衡稀有等级的事件对局。"""

    episode_ids = tuple(sorted(int(value) for value in episode_ids))
    levels_by_episode = {episode: set() for episode in episode_ids}
    for (episode, _first, _second), events in events_by_pair.items():
        levels_by_episode.setdefault(int(episode), set()).update(
            int(event['level']) for event in events
        )
    episodes_by_level = {
        level: [
            episode for episode in episode_ids
            if level in levels_by_episode.get(episode, ())
        ]
        for level in range(7, 12)
    }
    assignments = {}
    # 先处理最稀有等级，防止它被常见等级的划分提前耗尽。
    level_order = sorted(
        range(7, 12), key=lambda level: len(episodes_by_level[level])
    )
    split_order = (SPLIT_VALIDATION, SPLIT_TEST, SPLIT_TRAIN)
    for level in level_order:
        level_episodes = episodes_by_level[level]
        targets = _split_targets(len(level_episodes))
        current = {
            split: sum(
                assignments.get(episode) == split
                for episode in level_episodes
            )
            for split in split_order
        }
        unassigned = sorted(
            (
                episode for episode in level_episodes
                if episode not in assignments
            ),
            key=_stable_episode_key,
        )
        for episode in unassigned:
            split = max(
                split_order,
                key=lambda value: (
                    targets[value] - current[value],
                    targets[value] > 0,
                    value == SPLIT_TRAIN,
                ),
            )
            assignments[episode] = split
            current[split] += 1

    overall_targets = _split_targets(len(episode_ids))
    overall = {
        split: sum(value == split for value in assignments.values())
        for split in split_order
    }
    for episode in sorted(
            (value for value in episode_ids if value not in assignments),
            key=_stable_episode_key):
        split = max(
            split_order,
            key=lambda value: (
                overall_targets[value] - overall[value],
                value == SPLIT_TRAIN,
            ),
        )
        assignments[episode] = split
        overall[split] += 1
    return assignments


def finalize_pair_risk_dataset(
        dataset_dir,
        *,
        forecast_horizon=24,
        confirmation_drops=24,
        shard_rows=65_536):
    """把稀疏场景与确认事件关联为模型可直接训练的样本。"""

    dataset_dir = Path(dataset_dir).resolve()
    forecast_horizon = int(forecast_horizon)
    confirmation_drops = int(confirmation_drops)
    if forecast_horizon <= 0 or confirmation_drops <= 0:
        raise ValueError('risk label horizons must be positive')
    episode_end, events_by_pair, event_count = _event_index(dataset_dir)
    episode_splits = stratified_episode_splits(
        episode_end.keys(), events_by_pair
    )
    events_by_level = {str(level): 0 for level in range(7, 12)}
    events_by_split_level = {
        split: {str(level): 0 for level in range(7, 12)}
        for split in ('train', 'validation', 'test')
    }
    split_names = {
        SPLIT_TRAIN: 'train',
        SPLIT_VALIDATION: 'validation',
        SPLIT_TEST: 'test',
    }
    for (episode, _first, _second), events in events_by_pair.items():
        split_name = split_names[episode_splits[int(episode)]]
        for event in events:
            level = str(int(event['level']))
            events_by_level[level] += 1
            events_by_split_level[split_name][level] += 1
    output_dir = dataset_dir / 'labeled'
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob('pair_risk_samples-*.pt')):
        raise FileExistsError('labeled pair-risk shards already exist')
    writer = ShardedTensorWriter(
        output_dir,
        'pair_risk_samples',
        LABELED_DTYPES,
        shard_rows=shard_rows,
        background=False,
    )
    counts = {
        split: {
            level: {'positive': 0, 'negative': 0}
            for level in range(7, 12)
        }
        for split in ('train', 'validation', 'test')
    }
    censored = 0
    post_confirmation_skipped = 0
    raw_rows = 0
    try:
        for path in table_shards(
                dataset_dir, 'pair_risk_exposures', area='raw'):
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if (
                    payload.get('format_version') != 1
                    or payload.get('table') != 'pair_risk_exposures'):
                raise ValueError(f'invalid exposure shard: {path}')
            columns = payload['columns']
            rows = int(columns['episode_id'].shape[0])
            raw_rows += rows
            labels = torch.full((rows,), -1, dtype=torch.int8)
            split_values = torch.empty(rows, dtype=torch.int8)
            event_ids = torch.full((rows,), -1, dtype=torch.int64)
            lead_values = torch.zeros(rows, dtype=torch.int16)
            metadata = zip(
                columns['episode_id'].tolist(),
                columns['step'].tolist(),
                columns['fruit_id_i'].tolist(),
                columns['fruit_id_j'].tolist(),
            )
            for index, (episode, step, first, second) in enumerate(metadata):
                episode = int(episode)
                step = int(step)
                split_values[index] = episode_splits[episode]
                intervals = events_by_pair.get(
                    (episode, int(first), int(second)), ()
                )
                matched = None
                after_confirmation = False
                for event in intervals:
                    if event['onset'] - forecast_horizon <= step <= event['confirmed']:
                        matched = event
                        break
                    if event['confirmed'] < step <= event['end']:
                        after_confirmation = True
                if matched is not None:
                    labels[index] = 1
                    event_ids[index] = matched['event_id']
                    lead = matched['onset'] - step
                    lead_values[index] = max(-32768, min(32767, lead))
                    continue
                if after_confirmation:
                    post_confirmation_skipped += 1
                    continue
                end = episode_end.get(episode)
                resolution = step + forecast_horizon + confirmation_drops
                if end is not None and end[0] >= resolution:
                    labels[index] = 0
                else:
                    censored += 1
            keep = labels.ge(0)
            selected = torch.nonzero(keep, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            output = {
                name: columns[name].index_select(0, selected)
                for name in EXPOSURE_DTYPES
            }
            output.update({
                'label': labels.index_select(0, selected).bool(),
                'split': split_values.index_select(0, selected),
                'event_id': event_ids.index_select(0, selected),
                'lead_to_onset': lead_values.index_select(0, selected),
            })
            writer.append(output)
            selected_levels = output['level'].long()
            for split_value, split_name in split_names.items():
                for level in range(7, 12):
                    level_mask = (
                        output['split'].eq(split_value)
                        & selected_levels.eq(level)
                    )
                    positive = int((level_mask & output['label']).sum().item())
                    total = int(level_mask.sum().item())
                    counts[split_name][level]['positive'] += positive
                    counts[split_name][level]['negative'] += total - positive
    finally:
        writer.close()
    result = {
        'format_version': 1,
        'purpose': 'pair_failure_single_frame_risk_dataset',
        'forecast_horizon': forecast_horizon,
        'confirmation_drops': confirmation_drops,
        'raw_exposure_rows': raw_rows,
        'confirmed_events': event_count,
        'confirmed_events_by_level': events_by_level,
        'confirmed_events_by_split_level': events_by_split_level,
        'labeled_rows': writer.total_rows,
        'censored_rows': censored,
        'post_confirmation_rows_skipped': post_confirmation_skipped,
        'shards': writer.shard_count,
        'counts': counts,
        'split_strategy': 'episode_grouped_rare_level_stratified_v1',
        'episode_split_counts': {
            split_name: sum(
                value == split_value
                for value in episode_splits.values()
            )
            for split_value, split_name in split_names.items()
        },
    }
    _atomic_json(output_dir / 'manifest.json', result)
    return result


def average_precision(scores, labels):
    scores = torch.as_tensor(scores).float().flatten()
    labels = torch.as_tensor(labels).bool().flatten()
    positives = int(labels.sum().item())
    if scores.numel() == 0 or positives == 0:
        return math.nan
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    precision = sorted_labels.cumsum(0).float() / torch.arange(
        1, sorted_labels.numel() + 1, dtype=torch.float32
    )
    return float(precision[sorted_labels].mean().item())


def risk_metrics(logits, labels, *, threshold=0.5):
    logits = torch.as_tensor(logits).float().flatten()
    labels = torch.as_tensor(labels).bool().flatten()
    if logits.numel() != labels.numel():
        raise ValueError('metric logits and labels differ in length')
    if logits.numel() == 0:
        return {'samples': 0}
    probabilities = torch.sigmoid(logits)
    predicted = probabilities >= float(threshold)
    true_positive = int((predicted & labels).sum().item())
    false_positive = int((predicted & ~labels).sum().item())
    false_negative = int((~predicted & labels).sum().item())
    true_negative = int((~predicted & ~labels).sum().item())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        'samples': int(labels.numel()),
        'positives': int(labels.sum().item()),
        'negatives': int((~labels).sum().item()),
        'average_precision': average_precision(probabilities, labels),
        'accuracy': (true_positive + true_negative) / labels.numel(),
        'precision': precision,
        'recall': recall,
        'f1': 2.0 * precision * recall / max(1e-12, precision + recall),
        'brier': float(((probabilities - labels.float()) ** 2).mean().item()),
        'false_positives_per_1000_negatives': (
            false_positive / max(1, true_negative + false_positive) * 1000.0
        ),
    }


def checkpoint_payload(model, *, training, dataset_manifest, history):
    return {
        'format_version': 1,
        'model_type': 'pair_conditioned_deep_sets_risk_v1',
        'model_config': asdict(model.config),
        'model_state_dict': model.state_dict(),
        'training': dict(training),
        'dataset_manifest': dict(dataset_manifest),
        'history': list(history),
    }


__all__ = [
    'END_COLLECTOR_STOP',
    'END_DROP_LIMIT',
    'END_NATURAL',
    'END_SIMULATOR_TRUNCATED',
    'EPISODE_DTYPES',
    'EVENT_CONFIRMED',
    'EVENT_DTYPES',
    'EVENT_ENDED',
    'EXPOSURE_DTYPES',
    'LABELED_DTYPES',
    'PairRiskModel',
    'PairRiskModelConfig',
    'SPLIT_TEST',
    'SPLIT_TRAIN',
    'SPLIT_VALIDATION',
    'average_precision',
    'checkpoint_payload',
    'extract_episode_rows',
    'extract_pair_events',
    'extract_pair_exposures',
    'finalize_pair_risk_dataset',
    'risk_metrics',
]
