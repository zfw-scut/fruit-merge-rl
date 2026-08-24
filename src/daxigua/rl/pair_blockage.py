"""由长期轨迹确认、只读取当前几何的轻量水果对堵塞检测器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import torch
from torch import nn

from daxigua.core import MAX_FRUIT_LEVEL

from .merge_potential_stats import ShardedTensorWriter, table_shards
from .pair_risk import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    _event_index,
    risk_metrics,
    stratified_episode_splits,
)


BLOCKAGE_AREA = 'blocked_now'
BLOCKAGE_TABLE = 'pair_blockage_samples'

BLOCKAGE_LABELED_DTYPES = {
    'episode_id': torch.int64,
    'step': torch.int32,
    'pair_slot_i': torch.int16,
    'pair_slot_j': torch.int16,
    'fruit_id_i': torch.int64,
    'fruit_id_j': torch.int64,
    'level': torch.int8,
    'positions': torch.float32,
    'levels': torch.int8,
    'physics_radii': torch.float32,
    'active': torch.bool,
    'label': torch.bool,
    'split': torch.int8,
    'event_id': torch.int64,
    'offset_from_onset': torch.int16,
    'offset_to_end': torch.int16,
}


@dataclass(frozen=True, slots=True)
class PairBlockageModelConfig:
    """Pair-conditioned Deep Sets 的纯几何轻量配置。"""

    max_fruits: int = 64
    level_embedding_dim: int = 8
    context_hidden_dim: int = 40
    head_hidden_dim: int = 48
    board_width: float = 560.0
    board_height: float = 1120.0

    def __post_init__(self):
        for name in (
                'max_fruits', 'level_embedding_dim',
                'context_hidden_dim', 'head_hidden_dim'):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive')
        for name in ('board_width', 'board_height'):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f'{name} must be positive')


def _mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class PairBlockageModel(nn.Module):
    """从单帧水果集合判断目标同级水果对当前是否已经堵塞。"""

    REQUIRED_COLUMNS = (
        'positions', 'levels', 'physics_radii', 'active',
        'pair_slot_i', 'pair_slot_j',
    )

    def __init__(self, config: PairBlockageModelConfig | None = None):
        super().__init__()
        self.config = config or PairBlockageModelConfig()
        level_dim = self.config.level_embedding_dim
        context_hidden = self.config.context_hidden_dim
        self.level_embedding = nn.Embedding(
            MAX_FRUIT_LEVEL + 1, level_dim, padding_idx=0
        )
        # 其余水果只使用相对位置、绝对位置、半径和等级。
        self.context_encoder = _mlp(
            level_dim + 10, context_hidden, context_hidden
        )
        # 10维目标对几何、目标等级、mean/max集合摘要和有效水果比例。
        head_input = 10 + level_dim + context_hidden * 2 + 1
        self.blockage_head = nn.Sequential(
            nn.Linear(head_input, self.config.head_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.head_hidden_dim, 1),
        )

    def forward(self, columns):
        missing = set(self.REQUIRED_COLUMNS) - set(columns)
        if missing:
            raise ValueError(
                f'pair-blockage input misses columns: {sorted(missing)}'
            )
        position = columns['positions'].float()
        levels = columns['levels'].long().clamp(0, MAX_FRUIT_LEVEL)
        radii = columns['physics_radii'].float()
        active = columns['active'].bool()
        if position.ndim != 3 or position.shape[-1] != 2:
            raise ValueError('positions must have shape [batch, fruits, 2]')
        batch_size, fruit_count, _ = position.shape
        if fruit_count != self.config.max_fruits:
            raise ValueError(
                'pair-blockage fruit capacity does not match config'
            )
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
        pair_center = (position_i + position_j) * 0.5
        anchor_i = torch.where(flip[:, None], position_j, position_i)
        anchor_j = torch.where(flip[:, None], position_i, position_j)
        anchor_radius_i = torch.where(flip, radius_j, radius_i)
        anchor_radius_j = torch.where(flip, radius_i, radius_j)

        width = float(self.config.board_width)
        height = float(self.config.board_height)
        radius_min = torch.minimum(radius_i, radius_j).clamp_min(1.0)
        radius_max = torch.maximum(radius_i, radius_j)
        radius_sum = (radius_i + radius_j).clamp_min(1.0)
        surface_gap = distance - radius_sum
        pair_features = torch.stack((
            pair_center[:, 0] / width * 2.0 - 1.0,
            pair_center[:, 1] / height * 2.0 - 1.0,
            axis[:, 0],
            axis[:, 1],
            distance / width,
            surface_gap / radius_sum,
            radius_min / width,
            radius_max / width,
            delta[:, 0].abs() / width,
            delta[:, 1].abs() / height,
        ), dim=-1)

        relative = position - pair_center[:, None, :]
        along = (relative * axis[:, None, :]).sum(dim=-1)
        across = (relative * perpendicular[:, None, :]).sum(dim=-1)
        distance_i = torch.linalg.vector_norm(
            position - anchor_i[:, None, :], dim=-1
        )
        distance_j = torch.linalg.vector_norm(
            position - anchor_j[:, None, :], dim=-1
        )
        center_distance = torch.linalg.vector_norm(relative, dim=-1)
        level_delta = levels.float() - pair_level[:, None].float()
        context_continuous = torch.stack((
            along / width,
            across / height,
            center_distance / width,
            (distance_i - radii - anchor_radius_i[:, None]) / width,
            (distance_j - radii - anchor_radius_j[:, None]) / width,
            radii / width,
            levels.float() / MAX_FRUIT_LEVEL,
            level_delta / MAX_FRUIT_LEVEL,
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
        active_fraction = (
            active.sum(dim=1).to(position.dtype) / self.config.max_fruits
        ).unsqueeze(-1)
        hidden = torch.cat((
            pair_features,
            self.level_embedding(pair_level),
            context_mean,
            context_max,
            active_fraction,
        ), dim=-1)
        return self.blockage_head(hidden).squeeze(-1)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    os.replace(temporary, path)


def _bounded_int16(value):
    return max(-32768, min(32767, int(value)))


def finalize_pair_blockage_dataset(
        dataset_dir,
        *,
        confirmation_drops=24,
        shard_rows=65_536,
        output_area=BLOCKAGE_AREA):
    """把已确认事件的完整存续区间标为“当前已堵塞”。"""

    dataset_dir = Path(dataset_dir).resolve()
    confirmation_drops = int(confirmation_drops)
    if confirmation_drops <= 0:
        raise ValueError('confirmation_drops must be positive')
    episode_end, events_by_pair, event_count = _event_index(dataset_dir)
    episode_splits = stratified_episode_splits(
        episode_end.keys(), events_by_pair
    )
    split_names = {
        SPLIT_TRAIN: 'train',
        SPLIT_VALIDATION: 'validation',
        SPLIT_TEST: 'test',
    }
    events_by_level = {str(level): 0 for level in range(7, 12)}
    events_by_split_level = {
        split: {str(level): 0 for level in range(7, 12)}
        for split in split_names.values()
    }
    for (episode, _first, _second), events in events_by_pair.items():
        split_name = split_names[episode_splits[int(episode)]]
        for event in events:
            level = str(int(event['level']))
            events_by_level[level] += 1
            events_by_split_level[split_name][level] += 1

    output_dir = dataset_dir / str(output_area)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob(f'{BLOCKAGE_TABLE}-*.pt')):
        raise FileExistsError('pair-blockage labeled shards already exist')
    writer = ShardedTensorWriter(
        output_dir,
        BLOCKAGE_TABLE,
        BLOCKAGE_LABELED_DTYPES,
        shard_rows=int(shard_rows),
        background=False,
    )
    counts = {
        split: {
            level: {'positive': 0, 'negative': 0}
            for level in range(7, 12)
        }
        for split in split_names.values()
    }
    raw_rows = 0
    censored = 0
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
            from_onset = torch.zeros(rows, dtype=torch.int16)
            to_end = torch.zeros(rows, dtype=torch.int16)
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
                matched = next((
                    event for event in intervals
                    if event['onset'] <= step <= event['end']
                ), None)
                if matched is not None:
                    labels[index] = 1
                    event_ids[index] = matched['event_id']
                    from_onset[index] = _bounded_int16(
                        step - matched['onset']
                    )
                    to_end[index] = _bounded_int16(matched['end'] - step)
                    continue
                end = episode_end.get(episode)
                if end is not None and end[0] >= step + confirmation_drops:
                    labels[index] = 0
                else:
                    censored += 1

            selected = torch.nonzero(labels.ge(0), as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            output = {
                name: columns[name].index_select(0, selected)
                for name in (
                    'episode_id', 'step', 'pair_slot_i', 'pair_slot_j',
                    'fruit_id_i', 'fruit_id_j', 'level', 'positions',
                    'levels', 'physics_radii', 'active',
                )
            }
            output.update({
                'label': labels.index_select(0, selected).bool(),
                'split': split_values.index_select(0, selected),
                'event_id': event_ids.index_select(0, selected),
                'offset_from_onset': from_onset.index_select(0, selected),
                'offset_to_end': to_end.index_select(0, selected),
            })
            writer.append(output)
            for split_value, split_name in split_names.items():
                for level in range(7, 12):
                    mask = output['split'].eq(split_value) & output[
                        'level'
                    ].eq(level)
                    positive = int((mask & output['label']).sum().item())
                    total = int(mask.sum().item())
                    counts[split_name][level]['positive'] += positive
                    counts[split_name][level]['negative'] += total - positive
    finally:
        writer.close()

    result = {
        'format_version': 1,
        'purpose': 'pair_failure_current_blockage_geometry_dataset',
        'label_semantics': 'confirmed_event_onset_le_t_le_event_end',
        'input_semantics': 'current_positions_radii_levels_only',
        'confirmation_drops': confirmation_drops,
        'raw_exposure_rows': raw_rows,
        'confirmed_events': event_count,
        'confirmed_events_by_level': events_by_level,
        'confirmed_events_by_split_level': events_by_split_level,
        'labeled_rows': writer.total_rows,
        'censored_rows': censored,
        'shards': writer.shard_count,
        'counts': counts,
        'split_strategy': 'episode_grouped_rare_level_stratified_v1',
        'episode_split_counts': {
            split_name: sum(
                value == split_value for value in episode_splits.values()
            )
            for split_value, split_name in split_names.items()
        },
    }
    _atomic_json(output_dir / 'manifest.json', result)
    return result


def checkpoint_payload(model, *, training, dataset_manifest, history):
    return {
        'format_version': 1,
        'model_type': 'pair_geometry_current_blockage_v1',
        'model_config': asdict(model.config),
        'model_state_dict': model.state_dict(),
        'training': dict(training),
        'dataset_manifest': dict(dataset_manifest),
        'history': list(history),
    }


__all__ = [
    'BLOCKAGE_AREA',
    'BLOCKAGE_LABELED_DTYPES',
    'BLOCKAGE_TABLE',
    'PairBlockageModel',
    'PairBlockageModelConfig',
    'SPLIT_TEST',
    'SPLIT_TRAIN',
    'SPLIT_VALIDATION',
    'checkpoint_payload',
    'finalize_pair_blockage_dataset',
    'risk_metrics',
]
