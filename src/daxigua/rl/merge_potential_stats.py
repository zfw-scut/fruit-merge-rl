"""Merge Potential 数据采集与离线统计的张量表工具。"""

from __future__ import annotations

from collections import deque
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import torch

from daxigua.core import MAX_FRUIT_LEVEL, MIN_FRUIT_LEVEL


SAMPLE_FIRST_OBSERVATION = 1
SAMPLE_PERIODIC = 2

OUTCOME_MERGED = 0
OUTCOME_TERMINAL_UNMERGED = 1
OUTCOME_CENSORED = 2

END_NATURAL = 0
END_SIMULATOR_TRUNCATED = 1
END_DROP_LIMIT = 2
END_COLLECTOR_STOP = 3

_KEY_STRIDE = 1 << 32


SNAPSHOT_DTYPES = {
    'episode_id': torch.int64,
    'fruit_id': torch.int64,
    'observed_drop': torch.int32,
    'sample_kind': torch.uint8,
    'slot': torch.int16,
    'level': torch.int8,
    'x': torch.float32,
    'y': torch.float32,
    'radius': torch.float32,
    'age_frames': torch.int32,
    'scene_occupancy_ratio': torch.float32,
    'scene_fruit_count': torch.int16,
    'scene_score': torch.int32,
    'scene_max_level': torch.int8,
    'center_height_normalized': torch.float32,
    'top_height_normalized': torch.float32,
    'same_level_peer_count': torch.int16,
    'has_same_level_peer': torch.bool,
    'nearest_same_level_center_distance': torch.float32,
    'nearest_same_level_surface_gap': torch.float32,
    'nearest_same_level_center_distance_normalized': torch.float32,
    'nearest_same_level_surface_gap_normalized': torch.float32,
}

MERGE_SOURCE_DTYPES = {
    'episode_id': torch.int64,
    'fruit_id': torch.int64,
    'merge_drop': torch.int32,
    'source_level': torch.int8,
    'new_level': torch.int8,
    'new_fruit_id': torch.int64,
    'event_index': torch.int16,
    'source_index': torch.int8,
    'x': torch.float32,
    'y': torch.float32,
    'score_delta': torch.int32,
}

EPISODE_DTYPES = {
    'episode_id': torch.int64,
    'seed': torch.int64,
    'end_drop': torch.int32,
    'score': torch.int32,
    'max_level': torch.int8,
    'final_fruit_count': torch.int16,
    'end_kind': torch.int8,
}


def fruit_key(episode_id, fruit_id):
    """把局号和局内水果ID编码为可排序的int64键。"""

    episode_id = torch.as_tensor(episode_id, dtype=torch.int64)
    fruit_id = torch.as_tensor(fruit_id, dtype=torch.int64)
    if bool((episode_id < 0).any().item()):
        raise ValueError('episode_id must be non-negative')
    if bool(((fruit_id <= 0) | (fruit_id >= _KEY_STRIDE)).any().item()):
        raise ValueError('fruit_id must be in [1, 2**32)')
    return episode_id * _KEY_STRIDE + fruit_id


def empty_columns(dtypes, *, device='cpu'):
    return {
        name: torch.empty(0, dtype=dtype, device=device)
        for name, dtype in dtypes.items()
    }


class FruitSnapshotSampler:
    """按局内水果身份选择首次观察与稀疏周期快照。"""

    def __init__(
            self,
            num_envs,
            max_fruits,
            *,
            device,
            snapshot_stride=8,
            max_snapshots_per_fruit=32,
            snapshots_per_scale=4):
        snapshot_stride = int(snapshot_stride)
        max_snapshots_per_fruit = int(max_snapshots_per_fruit)
        snapshots_per_scale = int(snapshots_per_scale)
        if snapshot_stride <= 0:
            raise ValueError('snapshot_stride must be positive')
        if max_snapshots_per_fruit <= 0:
            raise ValueError('max_snapshots_per_fruit must be positive')
        if snapshots_per_scale <= 0:
            raise ValueError('snapshots_per_scale must be positive')
        shape = (int(num_envs), int(max_fruits))
        self.snapshot_stride = snapshot_stride
        self.max_snapshots_per_fruit = max_snapshots_per_fruit
        self.snapshots_per_scale = snapshots_per_scale
        self.tracked_ids = torch.zeros(
            shape, dtype=torch.int64, device=device
        )
        self.sample_counts = torch.zeros(
            shape, dtype=torch.int16, device=device
        )
        self.next_sample_steps = torch.zeros(
            shape, dtype=torch.int64, device=device
        )

    @torch.no_grad()
    def select(self, observation, enabled=None):
        active = observation.active
        if enabled is not None:
            enabled = torch.as_tensor(
                enabled, dtype=torch.bool, device=active.device
            )
            active = active & enabled[:, None]
        current_ids = torch.where(
            active, observation.fruit_ids, torch.zeros_like(
                observation.fruit_ids
            )
        )
        identity_changed = current_ids != self.tracked_ids
        current_steps = observation.step_count[:, None]
        self.sample_counts = torch.where(
            identity_changed,
            torch.zeros_like(self.sample_counts),
            self.sample_counts,
        )
        self.next_sample_steps = torch.where(
            identity_changed,
            current_steps + self.snapshot_stride,
            self.next_sample_steps,
        )
        first = active & identity_changed
        under_cap = self.sample_counts < self.max_snapshots_per_fruit
        periodic = (
            active
            & ~first
            & under_cap
            & (current_steps >= self.next_sample_steps)
        )
        kinds = (
            first.to(torch.uint8) * SAMPLE_FIRST_OBSERVATION
            + periodic.to(torch.uint8) * SAMPLE_PERIODIC
        )
        selected = (kinds > 0) & under_cap
        kinds = torch.where(selected, kinds, torch.zeros_like(kinds))
        scale = torch.div(
            (self.sample_counts.to(torch.int64) - 1).clamp_min(0),
            self.snapshots_per_scale,
            rounding_mode='floor',
        ).clamp_max(20)
        intervals = self.snapshot_stride * torch.bitwise_left_shift(
            torch.ones_like(scale), scale
        )
        self.next_sample_steps = torch.where(
            periodic,
            current_steps + intervals,
            self.next_sample_steps,
        )
        self.sample_counts += selected.to(self.sample_counts.dtype)
        self.tracked_ids.copy_(current_ids)
        return kinds

    @torch.no_grad()
    def reset(self, rows):
        rows = torch.as_tensor(
            rows, dtype=torch.int64, device=self.tracked_ids.device
        )
        if rows.numel() == 0:
            return
        self.tracked_ids[rows] = 0
        self.sample_counts[rows] = 0
        self.next_sample_steps[rows] = 0


@torch.no_grad()
def extract_snapshot_features(
        observation,
        sample_kinds,
        episode_ids,
        simulator_config,
        *,
        env_chunk_size=256):
    """从稳定决策状态提取被选水果的轻量几何与场景因素。"""

    if sample_kinds.shape != observation.active.shape:
        raise ValueError('sample_kinds must match observation.active')
    episode_ids = torch.as_tensor(
        episode_ids, dtype=torch.int64, device=observation.active.device
    )
    if episode_ids.shape != observation.active.shape[:1]:
        raise ValueError('episode_ids must have shape [batch]')
    env_chunk_size = int(env_chunk_size)
    if env_chunk_size <= 0:
        raise ValueError('env_chunk_size must be positive')

    device = observation.active.device
    chunks = {name: [] for name in SNAPSHOT_DTYPES}
    batch_size = int(observation.active.shape[0])
    bottom = float(
        simulator_config.board_height - simulator_config.wall_width
    )
    vertical_span = max(1.0, bottom - float(simulator_config.spawn_y))
    playable_width = max(
        1.0,
        float(
            simulator_config.board_width
            - 2.0 * simulator_config.wall_width
        ),
    )
    nan = float('nan')

    for start in range(0, batch_size, env_chunk_size):
        stop = min(batch_size, start + env_chunk_size)
        selected = sample_kinds[start:stop] > 0
        coordinates = torch.nonzero(selected, as_tuple=False)
        if coordinates.numel() == 0:
            continue
        env_local = coordinates[:, 0]
        slots = coordinates[:, 1]
        positions = observation.positions[start:stop].float()
        levels = observation.levels[start:stop]
        radii = observation.physics_radii[start:stop].float()
        active = observation.active[start:stop]
        distances = torch.cdist(positions, positions)
        same_level = (
            active[:, :, None]
            & active[:, None, :]
            & (levels[:, :, None] == levels[:, None, :])
        )
        diagonal = torch.eye(
            levels.shape[1], dtype=torch.bool, device=device
        )[None, :, :]
        same_level &= ~diagonal
        masked_distances = distances.masked_fill(~same_level, float('inf'))
        nearest_center, nearest_slot = masked_distances.min(dim=2)
        has_peer = torch.isfinite(nearest_center)
        nearest_radius = radii.gather(1, nearest_slot)
        nearest_gap = nearest_center - radii - nearest_radius
        nearest_center = torch.where(
            has_peer, nearest_center, torch.full_like(nearest_center, nan)
        )
        nearest_gap = torch.where(
            has_peer, nearest_gap, torch.full_like(nearest_gap, nan)
        )
        global_env = env_local + start
        selected_y = observation.positions[global_env, slots, 1].float()
        selected_radius = observation.physics_radii[
            global_env, slots
        ].float()

        values = {
            'episode_id': episode_ids[global_env],
            'fruit_id': observation.fruit_ids[global_env, slots],
            'observed_drop': observation.step_count[global_env].to(
                torch.int32
            ),
            'sample_kind': sample_kinds[global_env, slots].to(torch.uint8),
            'slot': slots.to(torch.int16),
            'level': observation.levels[global_env, slots].to(torch.int8),
            'x': observation.positions[global_env, slots, 0].float(),
            'y': selected_y,
            'radius': selected_radius,
            'age_frames': observation.age_frames[
                global_env, slots
            ].to(torch.int32),
            'scene_occupancy_ratio': (
                1.0 - observation.empty_space_ratio[global_env].float()
            ),
            'scene_fruit_count': observation.fruit_count[
                global_env
            ].to(torch.int16),
            'scene_score': observation.score[global_env].to(torch.int32),
            'scene_max_level': observation.max_level[
                global_env
            ].to(torch.int8),
            'center_height_normalized': (bottom - selected_y) / vertical_span,
            'top_height_normalized': (
                bottom - (selected_y - selected_radius)
            ) / vertical_span,
            'same_level_peer_count': same_level.sum(dim=2)[
                env_local, slots
            ].to(torch.int16),
            'has_same_level_peer': has_peer[env_local, slots],
            'nearest_same_level_center_distance': nearest_center[
                env_local, slots
            ],
            'nearest_same_level_surface_gap': nearest_gap[env_local, slots],
            'nearest_same_level_center_distance_normalized': (
                nearest_center[env_local, slots] / playable_width
            ),
            'nearest_same_level_surface_gap_normalized': (
                nearest_gap[env_local, slots] / playable_width
            ),
        }
        for name, dtype in SNAPSHOT_DTYPES.items():
            chunks[name].append(values[name].to(dtype=dtype))

    if not any(chunks.values()):
        return empty_columns(SNAPSHOT_DTYPES, device=device)
    return {
        name: torch.cat(parts, dim=0)
        for name, parts in chunks.items()
    }


@torch.no_grad()
def extract_merge_sources(events, episode_ids, merge_drops):
    """把定长合成事件展开为每个来源水果一行。"""

    episode_ids = torch.as_tensor(
        episode_ids, dtype=torch.int64, device=events.count.device
    )
    merge_drops = torch.as_tensor(
        merge_drops, dtype=torch.int64, device=events.count.device
    )
    event_slots = int(events.source_ids.shape[1])
    valid = (
        torch.arange(event_slots, device=events.count.device)[None, :]
        < events.count[:, None]
    )
    coordinates = torch.nonzero(valid, as_tuple=False)
    if coordinates.numel() == 0:
        return empty_columns(
            MERGE_SOURCE_DTYPES, device=events.count.device
        )
    env_rows = coordinates[:, 0]
    event_indices = coordinates[:, 1]
    repeated_env = env_rows.repeat_interleave(2)
    repeated_event = event_indices.repeat_interleave(2)
    source_index = torch.arange(
        2, dtype=torch.int8, device=events.count.device
    ).repeat(coordinates.shape[0])
    source_ids = events.source_ids[env_rows, event_indices].reshape(-1)
    values = {
        'episode_id': episode_ids[repeated_env],
        'fruit_id': source_ids,
        'merge_drop': merge_drops[repeated_env].to(torch.int32),
        'source_level': events.source_levels[
            repeated_env, repeated_event
        ].to(torch.int8),
        'new_level': events.new_levels[
            repeated_env, repeated_event
        ].to(torch.int8),
        'new_fruit_id': events.new_fruit_ids[
            repeated_env, repeated_event
        ],
        'event_index': repeated_event.to(torch.int16),
        'source_index': source_index,
        'x': events.positions[repeated_env, repeated_event, 0].float(),
        'y': events.positions[repeated_env, repeated_event, 1].float(),
        'score_delta': events.score_deltas[
            repeated_env, repeated_event
        ].to(torch.int32),
    }
    return {
        name: values[name].to(dtype=dtype)
        for name, dtype in MERGE_SOURCE_DTYPES.items()
    }


def extract_episode_rows(
        observation,
        rows,
        episode_ids,
        seeds,
        end_kinds):
    rows = torch.as_tensor(
        rows, dtype=torch.int64, device=observation.score.device
    )
    if rows.numel() == 0:
        return empty_columns(EPISODE_DTYPES, device=observation.score.device)
    episode_ids = torch.as_tensor(
        episode_ids, dtype=torch.int64, device=observation.score.device
    )
    seeds = torch.as_tensor(
        seeds, dtype=torch.int64, device=observation.score.device
    )
    end_kinds = torch.as_tensor(
        end_kinds, dtype=torch.int8, device=observation.score.device
    )
    values = {
        'episode_id': episode_ids[rows],
        'seed': seeds[rows],
        'end_drop': observation.step_count[rows].to(torch.int32),
        'score': observation.score[rows].to(torch.int32),
        'max_level': observation.max_level[rows].to(torch.int8),
        'final_fruit_count': observation.fruit_count[rows].to(torch.int16),
        'end_kind': end_kinds,
    }
    return {
        name: values[name].to(dtype=dtype)
        for name, dtype in EPISODE_DTYPES.items()
    }


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


class ShardedTensorWriter:
    """把CPU张量批次合并为原子写入的分片表。"""

    def __init__(
            self,
            root,
            table,
            dtypes,
            *,
            shard_rows=1_000_000,
            background=True,
            queue_depth=2):
        self.root = Path(root)
        self.table = str(table)
        self.dtypes = dict(dtypes)
        self.shard_rows = int(shard_rows)
        self.queue_depth = int(queue_depth)
        if self.shard_rows <= 0:
            raise ValueError('shard_rows must be positive')
        if self.queue_depth <= 0:
            raise ValueError('queue_depth must be positive')
        self.root.mkdir(parents=True, exist_ok=True)
        self._parts = {name: [] for name in self.dtypes}
        self._rows = 0
        self.total_rows = 0
        self.shard_count = 0
        self._executor = (
            ThreadPoolExecutor(max_workers=1)
            if background else None
        )
        self._pending = deque()

    def append(self, columns):
        if set(columns) != set(self.dtypes):
            raise ValueError(f'{self.table} columns do not match schema')
        lengths = {int(value.shape[0]) for value in columns.values()}
        if len(lengths) != 1:
            raise ValueError(f'{self.table} columns have different lengths')
        rows = lengths.pop()
        if rows == 0:
            return
        for name, dtype in self.dtypes.items():
            value = columns[name].detach().to('cpu', dtype=dtype)
            self._parts[name].append(value)
        self._rows += rows
        self.total_rows += rows
        if self._rows >= self.shard_rows:
            self.flush()

    def _wait_for_capacity(self):
        while len(self._pending) >= self.queue_depth:
            self._pending.popleft().result()

    def flush(self):
        if self._rows == 0:
            return
        columns = {
            name: torch.cat(parts, dim=0)
            for name, parts in self._parts.items()
        }
        payload = {
            'format_version': 1,
            'table': self.table,
            'rows': int(self._rows),
            'columns': columns,
        }
        path = self.root / f'{self.table}-{self.shard_count:06d}.pt'
        self.shard_count += 1
        self._parts = {name: [] for name in self.dtypes}
        self._rows = 0
        if self._executor is None:
            _atomic_torch_save(path, payload)
            return
        self._wait_for_capacity()
        self._pending.append(
            self._executor.submit(_atomic_torch_save, path, payload)
        )

    def close(self):
        self.flush()
        while self._pending:
            self._pending.popleft().result()
        if self._executor is not None:
            self._executor.shutdown(wait=True)


class DeviceTableAccumulator:
    """在设备端累积若干决策步，降低可变长结果回传频率。"""

    def __init__(self, writer, *, transfer_interval=8):
        self.writer = writer
        self.transfer_interval = int(transfer_interval)
        if self.transfer_interval <= 0:
            raise ValueError('transfer_interval must be positive')
        self._parts = {name: [] for name in writer.dtypes}
        self._steps = 0

    def append(self, columns):
        if set(columns) != set(self._parts):
            raise ValueError('device columns do not match writer schema')
        rows = int(next(iter(columns.values())).shape[0])
        if rows > 0:
            for name, value in columns.items():
                self._parts[name].append(value.detach().clone())

    def advance(self):
        self._steps += 1
        if self._steps >= self.transfer_interval:
            self.flush()

    def flush(self):
        if any(self._parts.values()):
            self.writer.append({
                name: torch.cat(parts, dim=0)
                for name, parts in self._parts.items()
            })
        self._parts = {name: [] for name in self.writer.dtypes}
        self._steps = 0

    def close(self):
        self.flush()
        self.writer.close()


def table_shards(dataset_dir, table, *, area='raw'):
    root = Path(dataset_dir) / area
    return tuple(sorted(root.glob(f'{table}-*.pt')))


def load_table_columns(dataset_dir, table, names, *, area='raw'):
    names = tuple(names)
    parts = {name: [] for name in names}
    for path in table_shards(dataset_dir, table, area=area):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('format_version') != 1 or payload.get('table') != table:
            raise ValueError(f'invalid {table} shard: {path}')
        for name in names:
            parts[name].append(payload['columns'][name])
    return {
        name: (
            torch.cat(values, dim=0)
            if values else torch.empty(0)
        )
        for name, values in parts.items()
    }


def _quantile(values, fraction):
    if values.numel() == 0:
        return math.nan
    return float(torch.quantile(values.float(), float(fraction)).item())


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@dataclass
class _FactorAccumulator:
    bins: int
    horizons: tuple[int, ...]

    def __post_init__(self):
        shape = (MAX_FRUIT_LEVEL + 1, self.bins)
        self.raw_rows = torch.zeros(shape, dtype=torch.int64)
        self.weight = torch.zeros(shape, dtype=torch.float64)
        self.resolved = torch.zeros(shape, dtype=torch.float64)
        self.eventual = torch.zeros(shape, dtype=torch.float64)
        self.terminal = torch.zeros(shape, dtype=torch.float64)
        self.censored = torch.zeros(shape, dtype=torch.float64)
        self.eligible = torch.zeros(
            (*shape, len(self.horizons)), dtype=torch.float64
        )
        self.within = torch.zeros_like(self.eligible)

    def add(self, levels, bin_indices, weights, outcomes, t_merge, followup):
        valid = (
            (levels >= MIN_FRUIT_LEVEL)
            & (levels <= MAX_FRUIT_LEVEL)
            & (bin_indices >= 0)
            & (bin_indices < self.bins)
        )
        if not bool(valid.any().item()):
            return
        levels = levels[valid].to(torch.int64)
        bins = bin_indices[valid].to(torch.int64)
        weights = weights[valid].to(torch.float64)
        outcomes = outcomes[valid]
        t_merge = t_merge[valid]
        followup = followup[valid]
        flat = levels * self.bins + bins
        size = (MAX_FRUIT_LEVEL + 1) * self.bins

        def add_to(target, values):
            target.view(-1).add_(torch.bincount(
                flat, weights=values, minlength=size
            ))

        self.raw_rows.view(-1).add_(torch.bincount(flat, minlength=size))
        add_to(self.weight, weights)
        resolved = outcomes != OUTCOME_CENSORED
        merged = outcomes == OUTCOME_MERGED
        terminal = outcomes == OUTCOME_TERMINAL_UNMERGED
        censored = outcomes == OUTCOME_CENSORED
        add_to(self.resolved, weights * resolved)
        add_to(self.eventual, weights * merged)
        add_to(self.terminal, weights * terminal)
        add_to(self.censored, weights * censored)
        for index, horizon in enumerate(self.horizons):
            eligible = resolved | (followup >= horizon)
            within = merged & (t_merge <= horizon)
            self.eligible[..., index].add_(torch.bincount(
                flat,
                weights=weights * eligible,
                minlength=size,
            ).reshape(MAX_FRUIT_LEVEL + 1, self.bins))
            self.within[..., index].add_(torch.bincount(
                flat,
                weights=weights * within,
                minlength=size,
            ).reshape(MAX_FRUIT_LEVEL + 1, self.bins))


def _safe_ratio(numerator, denominator):
    return (
        float(numerator / denominator)
        if float(denominator) > 0.0 else math.nan
    )


def _labeled_snapshot_payload(columns, merge_keys, merge_drops, episodes):
    keys = fruit_key(columns['episode_id'], columns['fruit_id'])
    positions = torch.searchsorted(merge_keys, keys)
    in_range = positions < merge_keys.numel()
    matched = torch.zeros_like(in_range)
    matched[in_range] = merge_keys[positions[in_range]] == keys[in_range]
    event_drop = torch.full_like(columns['observed_drop'], -1)
    event_drop[matched] = merge_drops[positions[matched]].to(event_drop.dtype)
    episode_id = columns['episode_id'].to(torch.int64)
    end_drop = episodes['end_drop'][episode_id].to(torch.int32)
    end_kind = episodes['end_kind'][episode_id].to(torch.int8)
    natural = end_kind == END_NATURAL
    outcome = torch.full(
        matched.shape, OUTCOME_CENSORED, dtype=torch.int8
    )
    outcome[natural & ~matched] = OUTCOME_TERMINAL_UNMERGED
    outcome[matched] = OUTCOME_MERGED
    t_merge = torch.where(
        matched,
        event_drop - columns['observed_drop'].to(torch.int32),
        torch.full_like(event_drop, -1),
    )
    if bool((matched & (t_merge <= 0)).any().item()):
        raise ValueError('observed fruit merged before or at its snapshot')
    result = dict(columns)
    result.update({
        'outcome': outcome,
        'merge_drop': event_drop,
        't_merge': t_merge,
        'episode_end_drop': end_drop,
        'followup_drops': end_drop - columns['observed_drop'].to(torch.int32),
        'episode_end_kind': end_kind,
    })
    return result, keys


def summarize_dataset(
        dataset_dir,
        *,
        output_dir=None,
        horizons=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
        factor_bins=10,
        interaction_bins=5,
        peer_count_cap=8,
        labeled_shard_rows=1_000_000):
    """关联原始事件，输出标签分片与可重新绘图的基础统计表。"""

    dataset_dir = Path(dataset_dir)
    output_dir = (
        Path(output_dir) if output_dir is not None
        else dataset_dir / 'analysis'
    )
    horizons = tuple(sorted({int(value) for value in horizons}))
    if not horizons or horizons[0] <= 0:
        raise ValueError('horizons must contain positive integers')
    factor_bins = int(factor_bins)
    interaction_bins = int(interaction_bins)
    peer_count_cap = int(peer_count_cap)
    if factor_bins <= 1 or interaction_bins <= 1 or peer_count_cap <= 0:
        raise ValueError('factor bin counts must be positive')

    episode_columns = load_table_columns(
        dataset_dir, 'episodes', ('episode_id', 'end_drop', 'end_kind')
    )
    if episode_columns['episode_id'].numel() == 0:
        raise ValueError('dataset has no completed or censored episodes')
    maximum_episode = int(episode_columns['episode_id'].max().item())
    episodes = {
        'end_drop': torch.full((maximum_episode + 1,), -1, dtype=torch.int32),
        'end_kind': torch.full(
            (maximum_episode + 1,), END_COLLECTOR_STOP, dtype=torch.int8
        ),
    }
    ids = episode_columns['episode_id'].to(torch.int64)
    if int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError('episode table contains duplicate episode IDs')
    episodes['end_drop'][ids] = episode_columns['end_drop'].to(torch.int32)
    episodes['end_kind'][ids] = episode_columns['end_kind'].to(torch.int8)

    merges = load_table_columns(
        dataset_dir, 'merge_sources', ('episode_id', 'fruit_id', 'merge_drop')
    )
    if merges['episode_id'].numel() > 0:
        merge_keys = fruit_key(merges['episode_id'], merges['fruit_id'])
        order = torch.argsort(merge_keys)
        merge_keys = merge_keys[order]
        merge_drops = merges['merge_drop'][order].to(torch.int32)
        if bool((merge_keys[1:] == merge_keys[:-1]).any().item()):
            raise ValueError('a fruit appears in more than one merge event')
    else:
        merge_keys = torch.empty(0, dtype=torch.int64)
        merge_drops = torch.empty(0, dtype=torch.int32)

    snapshot_paths = table_shards(dataset_dir, 'snapshots')
    if not snapshot_paths:
        raise ValueError('dataset has no snapshot shards')
    all_keys = []
    for path in snapshot_paths:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        columns = payload['columns']
        all_keys.append(fruit_key(columns['episode_id'], columns['fruit_id']))
    unique_keys, sample_counts = torch.unique(
        torch.cat(all_keys), sorted=True, return_counts=True
    )
    del all_keys

    labeled_root = output_dir / 'labeled'
    if labeled_root.exists() and any(labeled_root.iterdir()):
        raise FileExistsError(
            f'labeled output directory is not empty: {labeled_root}'
        )
    labeled_dtypes = dict(SNAPSHOT_DTYPES)
    labeled_dtypes.update({
        'outcome': torch.int8,
        'merge_drop': torch.int32,
        't_merge': torch.int32,
        'episode_end_drop': torch.int32,
        'followup_drops': torch.int32,
        'episode_end_kind': torch.int8,
        'fruit_snapshot_count': torch.int32,
        'fruit_weight': torch.float32,
    })
    labeled_writer = ShardedTensorWriter(
        labeled_root,
        'snapshots_labeled',
        labeled_dtypes,
        shard_rows=labeled_shard_rows,
        background=False,
    )

    lifecycle_rows = []
    lifecycle_histogram = {}
    lifecycle_level_counts = {
        level: {'total': 0, 'merged': 0, 'terminal': 0, 'censored': 0, 't': []}
        for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1)
    }
    lifecycle_horizon = {
        level: {
            horizon: {'eligible': 0, 'within': 0}
            for horizon in horizons
        }
        for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1)
    }
    snapshot_horizon = {
        level: {
            horizon: {'eligible': 0.0, 'within': 0.0}
            for horizon in horizons
        }
        for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1)
    }
    factor_specs = {
        'scene_occupancy_ratio': _FactorAccumulator(factor_bins, horizons),
        'center_height_normalized': _FactorAccumulator(factor_bins, horizons),
        'same_level_peer_count': _FactorAccumulator(
            peer_count_cap + 1, horizons
        ),
        'nearest_same_level_center_distance_normalized': _FactorAccumulator(
            factor_bins + 1, horizons
        ),
    }
    interaction_peer_cap = min(peer_count_cap, 4)
    interaction_specs = {
        'occupancy_x_nearest_same_level_distance': _FactorAccumulator(
            interaction_bins * (interaction_bins + 1), horizons
        ),
        'height_x_same_level_peer_count': _FactorAccumulator(
            interaction_bins * (interaction_peer_cap + 1), horizons
        ),
    }

    for path in snapshot_paths:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        columns = payload['columns']
        labeled, keys = _labeled_snapshot_payload(
            columns, merge_keys, merge_drops, episodes
        )
        count_positions = torch.searchsorted(unique_keys, keys)
        counts = sample_counts[count_positions].to(torch.int32)
        weights = counts.to(torch.float32).reciprocal()
        labeled['fruit_snapshot_count'] = counts
        labeled['fruit_weight'] = weights
        labeled_writer.append(labeled)

        levels = labeled['level'].to(torch.int64)
        outcomes = labeled['outcome']
        t_merge = labeled['t_merge']
        followup = labeled['followup_drops']
        first = (
            labeled['sample_kind'].to(torch.int64)
            & SAMPLE_FIRST_OBSERVATION
        ) != 0
        for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1):
            chosen = first & (levels == level)
            if not bool(chosen.any().item()):
                continue
            selected_outcomes = outcomes[chosen]
            selected_t = t_merge[chosen]
            stats = lifecycle_level_counts[level]
            stats['total'] += int(chosen.sum().item())
            stats['merged'] += int(
                (selected_outcomes == OUTCOME_MERGED).sum().item()
            )
            stats['terminal'] += int(
                (selected_outcomes == OUTCOME_TERMINAL_UNMERGED).sum().item()
            )
            stats['censored'] += int(
                (selected_outcomes == OUTCOME_CENSORED).sum().item()
            )
            merged_t = selected_t[selected_outcomes == OUTCOME_MERGED]
            if merged_t.numel() > 0:
                stats['t'].append(merged_t)
                values, counts_t = torch.unique(merged_t, return_counts=True)
                for value, count in zip(values.tolist(), counts_t.tolist()):
                    key = (level, int(value))
                    lifecycle_histogram[key] = (
                        lifecycle_histogram.get(key, 0) + int(count)
                    )
            selected_followup = followup[chosen]
            for horizon in horizons:
                horizon_stats = lifecycle_horizon[level][horizon]
                eligible = (
                    (selected_outcomes != OUTCOME_CENSORED)
                    | (selected_followup >= horizon)
                )
                within = (
                    (selected_outcomes == OUTCOME_MERGED)
                    & (selected_t <= horizon)
                )
                horizon_stats['eligible'] += int(eligible.sum().item())
                horizon_stats['within'] += int(within.sum().item())

        for horizon in horizons:
            eligible = (outcomes != OUTCOME_CENSORED) | (followup >= horizon)
            within = (
                (outcomes == OUTCOME_MERGED) & (t_merge <= horizon)
            )
            for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1):
                chosen = levels == level
                if not bool(chosen.any().item()):
                    continue
                stats = snapshot_horizon[level][horizon]
                stats['eligible'] += float(
                    weights[chosen & eligible].sum().item()
                )
                stats['within'] += float(
                    weights[chosen & within].sum().item()
                )

        occupancy_bins = torch.clamp(
            torch.floor(labeled['scene_occupancy_ratio'] * factor_bins),
            0,
            factor_bins - 1,
        ).to(torch.int64)
        height_bins = torch.clamp(
            torch.floor(labeled['center_height_normalized'] * factor_bins),
            0,
            factor_bins - 1,
        ).to(torch.int64)
        peer_bins = labeled['same_level_peer_count'].to(torch.int64).clamp(
            0, peer_count_cap
        )
        nearest = labeled[
            'nearest_same_level_center_distance_normalized'
        ]
        nearest_bins = torch.clamp(
            torch.floor(nearest.nan_to_num(0.0) * factor_bins),
            0,
            factor_bins - 1,
        ).to(torch.int64)
        nearest_bins[~labeled['has_same_level_peer']] = factor_bins
        for name, bins in (
                ('scene_occupancy_ratio', occupancy_bins),
                ('center_height_normalized', height_bins),
                ('same_level_peer_count', peer_bins),
                ('nearest_same_level_center_distance_normalized', nearest_bins)):
            factor_specs[name].add(
                levels, bins, weights, outcomes, t_merge, followup
            )
        occupancy_interaction = torch.clamp(
            torch.floor(
                labeled['scene_occupancy_ratio'] * interaction_bins
            ),
            0,
            interaction_bins - 1,
        ).to(torch.int64)
        height_interaction = torch.clamp(
            torch.floor(
                labeled['center_height_normalized'] * interaction_bins
            ),
            0,
            interaction_bins - 1,
        ).to(torch.int64)
        nearest_interaction = torch.clamp(
            torch.floor(nearest.nan_to_num(0.0) * interaction_bins),
            0,
            interaction_bins - 1,
        ).to(torch.int64)
        nearest_interaction[~labeled['has_same_level_peer']] = (
            interaction_bins
        )
        peer_interaction = labeled['same_level_peer_count'].to(
            torch.int64
        ).clamp(0, interaction_peer_cap)
        interaction_specs[
            'occupancy_x_nearest_same_level_distance'
        ].add(
            levels,
            occupancy_interaction * (interaction_bins + 1)
            + nearest_interaction,
            weights,
            outcomes,
            t_merge,
            followup,
        )
        interaction_specs['height_x_same_level_peer_count'].add(
            levels,
            height_interaction * (interaction_peer_cap + 1)
            + peer_interaction,
            weights,
            outcomes,
            t_merge,
            followup,
        )

    labeled_writer.close()

    for level, stats in lifecycle_level_counts.items():
        merged_t = (
            torch.cat(stats['t']).to(torch.float32)
            if stats['t'] else torch.empty(0)
        )
        resolved = stats['merged'] + stats['terminal']
        lifecycle_rows.append({
            'level': level,
            'fruits': stats['total'],
            'resolved_fruits': resolved,
            'merged_fruits': stats['merged'],
            'terminal_unmerged_fruits': stats['terminal'],
            'censored_fruits': stats['censored'],
            'eventual_merge_probability_resolved': _safe_ratio(
                stats['merged'], resolved
            ),
            'terminal_unmerged_probability_resolved': _safe_ratio(
                stats['terminal'], resolved
            ),
            'censored_probability_all': _safe_ratio(
                stats['censored'], stats['total']
            ),
            'merged_t_p25': _quantile(merged_t, 0.25),
            'merged_t_median': _quantile(merged_t, 0.50),
            'merged_t_p75': _quantile(merged_t, 0.75),
            'merged_t_p90': _quantile(merged_t, 0.90),
            'merged_t_p95': _quantile(merged_t, 0.95),
        })
    _write_csv(
        output_dir / 'lifecycle_by_level.csv',
        tuple(lifecycle_rows[0]),
        lifecycle_rows,
    )
    histogram_rows = [
        {'level': level, 't_merge': t_merge, 'count': count}
        for (level, t_merge), count in sorted(lifecycle_histogram.items())
    ]
    _write_csv(
        output_dir / 'lifecycle_t_merge_histogram.csv',
        ('level', 't_merge', 'count'),
        histogram_rows,
    )
    horizon_rows = []
    for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1):
        stats = lifecycle_level_counts[level]
        for horizon in horizons:
            lifecycle_stats = lifecycle_horizon[level][horizon]
            snapshot_stats = snapshot_horizon[level][horizon]
            horizon_rows.append({
                'level': level,
                'horizon_drops': horizon,
                'lifecycle_eligible_fruits': lifecycle_stats['eligible'],
                'lifecycle_merged_within': lifecycle_stats['within'],
                'lifecycle_probability': _safe_ratio(
                    lifecycle_stats['within'], lifecycle_stats['eligible']
                ),
                'snapshot_eligible_fruit_weight': snapshot_stats['eligible'],
                'snapshot_merged_within_weight': snapshot_stats['within'],
                'snapshot_probability': _safe_ratio(
                    snapshot_stats['within'], snapshot_stats['eligible']
                ),
            })
    _write_csv(
        output_dir / 'horizon_probabilities_by_level.csv',
        tuple(horizon_rows[0]),
        horizon_rows,
    )

    factor_rows = []
    for name, accumulator in factor_specs.items():
        for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1):
            for bin_index in range(accumulator.bins):
                weight = accumulator.weight[level, bin_index].item()
                if weight <= 0.0:
                    continue
                if name == 'same_level_peer_count':
                    lower = bin_index
                    upper = (
                        math.inf if bin_index == peer_count_cap
                        else bin_index
                    )
                    missing = False
                elif (
                    name == 'nearest_same_level_center_distance_normalized'
                    and bin_index == factor_bins
                ):
                    lower = math.nan
                    upper = math.nan
                    missing = True
                else:
                    lower = bin_index / factor_bins
                    upper = (bin_index + 1) / factor_bins
                    missing = False
                base = {
                    'level': level,
                    'factor': name,
                    'bin_index': bin_index,
                    'bin_lower': lower,
                    'bin_upper': upper,
                    'missing_same_level_peer': missing,
                    'raw_snapshot_rows': int(
                        accumulator.raw_rows[level, bin_index].item()
                    ),
                    'fruit_normalized_weight': weight,
                    'resolved_weight': accumulator.resolved[
                        level, bin_index
                    ].item(),
                    'eventual_merge_probability_resolved': _safe_ratio(
                        accumulator.eventual[level, bin_index].item(),
                        accumulator.resolved[level, bin_index].item(),
                    ),
                    'terminal_unmerged_probability_resolved': _safe_ratio(
                        accumulator.terminal[level, bin_index].item(),
                        accumulator.resolved[level, bin_index].item(),
                    ),
                    'censored_weight': accumulator.censored[
                        level, bin_index
                    ].item(),
                }
                for horizon_index, horizon in enumerate(horizons):
                    eligible = accumulator.eligible[
                        level, bin_index, horizon_index
                    ].item()
                    within = accumulator.within[
                        level, bin_index, horizon_index
                    ].item()
                    base[f'eligible_weight_h{horizon}'] = eligible
                    base[f'merge_probability_h{horizon}'] = _safe_ratio(
                        within, eligible
                    )
                factor_rows.append(base)
    _write_csv(
        output_dir / 'factor_relationships_by_level.csv',
        tuple(factor_rows[0]),
        factor_rows,
    )
    interaction_rows = []
    for name, accumulator in interaction_specs.items():
        if name == 'occupancy_x_nearest_same_level_distance':
            second_bins = interaction_bins + 1
        else:
            second_bins = interaction_peer_cap + 1
        for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1):
            for combined_bin in range(accumulator.bins):
                weight = accumulator.weight[level, combined_bin].item()
                if weight <= 0.0:
                    continue
                first_bin = combined_bin // second_bins
                second_bin = combined_bin % second_bins
                if name == 'occupancy_x_nearest_same_level_distance':
                    second_missing_or_capped = second_bin == interaction_bins
                    second_lower = (
                        math.nan if second_missing_or_capped
                        else second_bin / interaction_bins
                    )
                    second_upper = (
                        math.nan if second_missing_or_capped
                        else (second_bin + 1) / interaction_bins
                    )
                else:
                    second_missing_or_capped = (
                        second_bin == interaction_peer_cap
                    )
                    second_lower = second_bin
                    second_upper = (
                        math.inf if second_missing_or_capped else second_bin
                    )
                base = {
                    'level': level,
                    'interaction': name,
                    'first_bin': first_bin,
                    'second_bin': second_bin,
                    'first_bin_lower': first_bin / interaction_bins,
                    'first_bin_upper': (first_bin + 1) / interaction_bins,
                    'second_bin_lower': second_lower,
                    'second_bin_upper': second_upper,
                    'second_is_missing_or_capped': second_missing_or_capped,
                    'raw_snapshot_rows': int(
                        accumulator.raw_rows[level, combined_bin].item()
                    ),
                    'fruit_normalized_weight': weight,
                    'resolved_weight': accumulator.resolved[
                        level, combined_bin
                    ].item(),
                    'eventual_merge_probability_resolved': _safe_ratio(
                        accumulator.eventual[level, combined_bin].item(),
                        accumulator.resolved[level, combined_bin].item(),
                    ),
                    'terminal_unmerged_probability_resolved': _safe_ratio(
                        accumulator.terminal[level, combined_bin].item(),
                        accumulator.resolved[level, combined_bin].item(),
                    ),
                    'censored_weight': accumulator.censored[
                        level, combined_bin
                    ].item(),
                }
                for horizon_index, horizon in enumerate(horizons):
                    eligible = accumulator.eligible[
                        level, combined_bin, horizon_index
                    ].item()
                    within = accumulator.within[
                        level, combined_bin, horizon_index
                    ].item()
                    base[f'eligible_weight_h{horizon}'] = eligible
                    base[f'merge_probability_h{horizon}'] = _safe_ratio(
                        within, eligible
                    )
                interaction_rows.append(base)
    _write_csv(
        output_dir / 'factor_interactions_by_level.csv',
        tuple(interaction_rows[0]),
        interaction_rows,
    )
    _atomic_torch_save(output_dir / 'fruit_sample_counts.pt', {
        'format_version': 1,
        'fruit_keys': unique_keys,
        'snapshot_counts': sample_counts.to(torch.int32),
    })
    result = {
        'episodes': int(ids.numel()),
        'merge_sources': int(merge_keys.numel()),
        'unique_observed_fruits': int(unique_keys.numel()),
        'snapshot_rows': int(sample_counts.sum().item()),
        'labeled_shards': int(labeled_writer.shard_count),
        'output_dir': str(output_dir.resolve()),
    }
    analysis_manifest = {
        'format_version': 1,
        'horizons': list(horizons),
        'factor_bins': factor_bins,
        'interaction_bins': interaction_bins,
        'peer_count_cap': peer_count_cap,
        'result': result,
    }
    manifest_path = output_dir / 'analysis_manifest.json'
    temporary = manifest_path.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(analysis_manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    os.replace(temporary, manifest_path)
    return result


__all__ = [
    'DeviceTableAccumulator',
    'END_COLLECTOR_STOP',
    'END_DROP_LIMIT',
    'END_NATURAL',
    'END_SIMULATOR_TRUNCATED',
    'EPISODE_DTYPES',
    'FruitSnapshotSampler',
    'MERGE_SOURCE_DTYPES',
    'OUTCOME_CENSORED',
    'OUTCOME_MERGED',
    'OUTCOME_TERMINAL_UNMERGED',
    'SAMPLE_FIRST_OBSERVATION',
    'SAMPLE_PERIODIC',
    'SNAPSHOT_DTYPES',
    'ShardedTensorWriter',
    'extract_episode_rows',
    'extract_merge_sources',
    'extract_snapshot_features',
    'fruit_key',
    'load_table_columns',
    'summarize_dataset',
    'table_shards',
]
