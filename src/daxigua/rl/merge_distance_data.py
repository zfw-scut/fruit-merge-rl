"""合成步距预测器的按场景采集、标签关联和分片读取契约。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import torch

from .merge_distance import DEFAULT_MERGE_HORIZONS, merge_distance_targets
from .merge_potential_stats import (
    END_COLLECTOR_STOP,
    END_NATURAL,
    OUTCOME_CENSORED,
    OUTCOME_MERGED,
    OUTCOME_TERMINAL_UNMERGED,
    ShardedTensorWriter,
    fruit_key,
    load_table_columns,
    table_shards,
)
from .observations import TensorState


SCENE_DTYPES = {
    'episode_id': torch.int64,
    'observed_drop': torch.int32,
    'positions': torch.float32,
    'velocities': torch.float32,
    'angular_velocities': torch.float32,
    'levels': torch.int8,
    'physics_radii': torch.float32,
    'age_frames': torch.int32,
    'active': torch.bool,
    'fruit_ids': torch.int64,
    'fruit_queue': torch.int8,
    'danger_progress': torch.float32,
    'over_danger_line': torch.bool,
}

LABELED_SCENE_DTYPES = dict(SCENE_DTYPES)
LABELED_SCENE_DTYPES.update({
    'outcome': torch.int8,
    't_merge': torch.int32,
    'fruit_snapshot_count': torch.int32,
    'fruit_weight': torch.float32,
    'episode_end_drop': torch.int32,
    'episode_end_kind': torch.int8,
})


class SceneSnapshotSampler:
    """首次非空场景后按固定投放间隔抽取完整决策状态。"""

    def __init__(
            self,
            num_envs,
            *,
            device,
            scene_stride=4,
            max_scenes_per_episode=1024):
        self.scene_stride = int(scene_stride)
        self.max_scenes_per_episode = int(max_scenes_per_episode)
        if self.scene_stride <= 0:
            raise ValueError('scene_stride must be positive')
        if self.max_scenes_per_episode <= 0:
            raise ValueError('max_scenes_per_episode must be positive')
        self.last_drop = torch.full(
            (int(num_envs),), -1, dtype=torch.int64, device=device
        )
        self.sample_counts = torch.zeros(
            int(num_envs), dtype=torch.int32, device=device
        )

    @torch.no_grad()
    def select(self, observation, enabled=None):
        enabled = (
            torch.ones_like(observation.step_count, dtype=torch.bool)
            if enabled is None else
            torch.as_tensor(
                enabled,
                dtype=torch.bool,
                device=observation.step_count.device,
            )
        )
        nonempty = observation.active.any(dim=1)
        current = observation.step_count.to(torch.int64)
        first = self.last_drop < 0
        due = first | ((current - self.last_drop) >= self.scene_stride)
        selected = (
            enabled
            & nonempty
            & due
            & (self.sample_counts < self.max_scenes_per_episode)
        )
        self.last_drop = torch.where(selected, current, self.last_drop)
        self.sample_counts += selected.to(self.sample_counts.dtype)
        return selected

    @torch.no_grad()
    def reset(self, rows):
        rows = torch.as_tensor(
            rows, dtype=torch.int64, device=self.last_drop.device
        )
        if rows.numel() == 0:
            return
        self.last_drop[rows] = -1
        self.sample_counts[rows] = 0


@torch.no_grad()
def extract_scene_rows(observation, selected_rows, episode_ids):
    """把选中的完整场景保持固定槽位写成张量表。"""

    selected_rows = torch.as_tensor(
        selected_rows,
        dtype=torch.bool,
        device=observation.positions.device,
    )
    if selected_rows.shape != observation.step_count.shape:
        raise ValueError('selected_rows must have shape [batch]')
    rows = torch.nonzero(selected_rows, as_tuple=False).flatten()
    episode_ids = torch.as_tensor(
        episode_ids, dtype=torch.int64, device=observation.positions.device
    )
    values = {
        'episode_id': episode_ids[rows],
        'observed_drop': observation.step_count[rows],
        'positions': observation.positions[rows],
        'velocities': observation.velocities[rows],
        'angular_velocities': observation.angular_velocities[rows],
        'levels': observation.levels[rows],
        'physics_radii': observation.physics_radii[rows],
        'age_frames': observation.age_frames[rows],
        'active': observation.active[rows],
        'fruit_ids': observation.fruit_ids[rows],
        'fruit_queue': observation.fruit_queue[rows],
        'danger_progress': observation.danger_progress[rows],
        'over_danger_line': observation.over_danger_line[rows],
    }
    return {
        name: values[name].to(dtype=dtype)
        for name, dtype in SCENE_DTYPES.items()
    }


def episode_split(episode_ids):
    """按完整episode固定划分90%训练、5%验证和5%测试。"""

    buckets = torch.remainder(
        torch.as_tensor(episode_ids, dtype=torch.int64), 100
    )
    result = torch.zeros_like(buckets, dtype=torch.int8)
    result[(buckets >= 5) & (buckets < 10)] = 1
    result[buckets < 5] = 2
    return result


def split_mask(episode_ids, split):
    split_id = {'train': 0, 'validation': 1, 'test': 2}.get(str(split))
    if split_id is None:
        raise ValueError(f'unknown dataset split: {split}')
    return episode_split(episode_ids) == split_id


def _load_episode_index(dataset_dir):
    columns = load_table_columns(
        dataset_dir, 'episodes', ('episode_id', 'end_drop', 'end_kind')
    )
    if columns['episode_id'].numel() == 0:
        raise ValueError('dataset has no completed or censored episodes')
    ids = columns['episode_id'].to(torch.int64)
    if int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError('episode table contains duplicate episode IDs')
    maximum = int(ids.max().item())
    end_drop = torch.full((maximum + 1,), -1, dtype=torch.int32)
    end_kind = torch.full(
        (maximum + 1,), END_COLLECTOR_STOP, dtype=torch.int8
    )
    end_drop[ids] = columns['end_drop'].to(torch.int32)
    end_kind[ids] = columns['end_kind'].to(torch.int8)
    return end_drop, end_kind


def _load_merge_index(dataset_dir):
    columns = load_table_columns(
        dataset_dir, 'merge_sources', ('episode_id', 'fruit_id', 'merge_drop')
    )
    if columns['episode_id'].numel() == 0:
        return (
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.int32),
        )
    keys = fruit_key(columns['episode_id'], columns['fruit_id'])
    order = torch.argsort(keys)
    keys = keys[order]
    drops = columns['merge_drop'][order].to(torch.int32)
    if bool((keys[1:] == keys[:-1]).any().item()):
        raise ValueError('a fruit appears in more than one merge event')
    return keys, drops


def _scene_keys(columns):
    active = columns['active'].to(torch.bool)
    episodes = columns['episode_id'][:, None].expand_as(
        columns['fruit_ids']
    )
    keys = torch.zeros_like(columns['fruit_ids'], dtype=torch.int64)
    keys[active] = fruit_key(
        episodes[active], columns['fruit_ids'][active]
    )
    return keys


def _label_scene_columns(
        columns,
        merge_keys,
        merge_drops,
        episode_end_drop,
        episode_end_kind,
        unique_scene_keys,
        scene_key_counts):
    active = columns['active'].to(torch.bool)
    keys = _scene_keys(columns)
    active_keys = keys[active]
    matched = torch.zeros_like(active)
    event_drops = torch.full_like(
        columns['fruit_ids'], -1, dtype=torch.int32
    )
    if active_keys.numel() > 0 and merge_keys.numel() > 0:
        positions = torch.searchsorted(merge_keys, active_keys)
        in_range = positions < merge_keys.numel()
        active_matched = torch.zeros_like(in_range)
        active_matched[in_range] = (
            merge_keys[positions[in_range]] == active_keys[in_range]
        )
        matched[active] = active_matched
        active_event_drops = torch.full_like(active_keys, -1, dtype=torch.int32)
        active_event_drops[active_matched] = merge_drops[
            positions[active_matched]
        ]
        event_drops[active] = active_event_drops

    episode_ids = columns['episode_id'].to(torch.int64)
    end_drop = episode_end_drop[episode_ids]
    end_kind = episode_end_kind[episode_ids]
    natural = (end_kind == END_NATURAL)[:, None]
    outcome = torch.full_like(
        columns['fruit_ids'], OUTCOME_CENSORED, dtype=torch.int8
    )
    outcome[active & natural & ~matched] = OUTCOME_TERMINAL_UNMERGED
    outcome[active & matched] = OUTCOME_MERGED
    observed = columns['observed_drop'][:, None].expand_as(event_drops)
    t_merge = torch.where(
        matched,
        event_drops - observed.to(torch.int32),
        torch.full_like(event_drops, -1),
    )
    if bool((matched & (t_merge <= 0)).any().item()):
        raise ValueError('observed fruit merged before or at its scene')

    counts = torch.zeros_like(keys, dtype=torch.int32)
    if active_keys.numel() > 0:
        count_positions = torch.searchsorted(unique_scene_keys, active_keys)
        counts[active] = scene_key_counts[count_positions].to(torch.int32)
    weights = torch.zeros_like(keys, dtype=torch.float32)
    weights[active] = counts[active].to(torch.float32).reciprocal()

    result = dict(columns)
    result.update({
        'outcome': outcome,
        't_merge': t_merge,
        'fruit_snapshot_count': counts,
        'fruit_weight': weights,
        'episode_end_drop': end_drop,
        'episode_end_kind': end_kind,
    })
    return result


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    temporary.replace(path)


def label_scene_dataset(
        dataset_dir,
        *,
        output_dir=None,
        horizons=DEFAULT_MERGE_HORIZONS,
        shard_rows=32_768):
    """把完整场景与未来事件关联为逐槽位监督分片。"""

    dataset_dir = Path(dataset_dir)
    output_dir = (
        Path(output_dir) if output_dir is not None
        else dataset_dir / 'predictor'
    )
    horizons = tuple(int(value) for value in horizons)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f'predictor output directory is not empty: {output_dir}'
        )
    labeled_root = output_dir / 'labeled'
    labeled_root.mkdir(parents=True, exist_ok=True)

    source_manifest_path = dataset_dir / 'manifest.json'
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding='utf-8'))
        if source_manifest_path.exists() else {}
    )

    scene_paths = table_shards(dataset_dir, 'scenes')
    if not scene_paths:
        raise ValueError('dataset has no scene shards')
    episode_end_drop, episode_end_kind = _load_episode_index(dataset_dir)
    merge_keys, merge_drops = _load_merge_index(dataset_dir)

    all_scene_keys = []
    for path in scene_paths:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('format_version') != 1 or payload.get('table') != 'scenes':
            raise ValueError(f'invalid scene shard: {path}')
        columns = payload['columns']
        keys = _scene_keys(columns)
        all_scene_keys.append(keys[columns['active'].to(torch.bool)])
    unique_scene_keys, scene_key_counts = torch.unique(
        torch.cat(all_scene_keys), sorted=True, return_counts=True
    )
    del all_scene_keys

    writer = ShardedTensorWriter(
        labeled_root,
        'scenes_labeled',
        LABELED_SCENE_DTYPES,
        shard_rows=int(shard_rows),
        background=False,
    )
    split_names = ('train', 'validation', 'test')
    split_scene_counts = Counter()
    split_fruit_counts = Counter()
    class_counts = {
        name: torch.zeros(12, len(horizons) + 2, dtype=torch.int64)
        for name in split_names
    }
    try:
        for path in scene_paths:
            payload = torch.load(path, map_location='cpu', weights_only=False)
            labeled = _label_scene_columns(
                payload['columns'],
                merge_keys,
                merge_drops,
                episode_end_drop,
                episode_end_kind,
                unique_scene_keys,
                scene_key_counts,
            )
            writer.append(labeled)
            targets, resolved = merge_distance_targets(
                labeled['outcome'], labeled['t_merge'], horizons
            )
            valid = labeled['active'] & resolved
            splits = episode_split(labeled['episode_id'])
            for split_id, name in enumerate(split_names):
                scene_mask = splits == split_id
                split_scene_counts[name] += int(scene_mask.sum().item())
                fruit_mask = valid & scene_mask[:, None]
                split_fruit_counts[name] += int(fruit_mask.sum().item())
                if not bool(fruit_mask.any().item()):
                    continue
                levels = labeled['levels'][fruit_mask].to(torch.int64)
                classes = targets[fruit_mask]
                flat = levels * (len(horizons) + 2) + classes
                class_counts[name].view(-1).add_(torch.bincount(
                    flat,
                    minlength=12 * (len(horizons) + 2),
                ))
    finally:
        writer.close()

    manifest = {
        'format_version': 1,
        'purpose': 'merge_distance_predictor_dataset',
        'source_dataset': str(dataset_dir.resolve()),
        'source_identity': {
            'checkpoint': source_manifest.get('checkpoint'),
            'checkpoint_sha256': source_manifest.get('checkpoint_sha256'),
            'physics_identity': source_manifest.get('physics_identity'),
            'simulator_config': source_manifest.get('simulator_config'),
            'physics_fps': source_manifest.get('simulator_config', {}).get(
                'physics_fps', 30
            ),
            'policy': source_manifest.get('policy'),
            'git_revision': source_manifest.get('git_revision'),
        },
        'horizons': list(horizons),
        'class_semantics': {
            'horizon_classes': [
                (
                    str(value)
                    if index == 0 else
                    f'{horizons[index - 1] + 1}-{value}'
                )
                for index, value in enumerate(horizons)
            ],
            'tail_class': f'> {horizons[-1]} and eventually merged',
            'terminal_unmerged_class': 'natural terminal before merge',
        },
        'split_rule': {
            'train': 'episode_id % 100 >= 10',
            'validation': '5 <= episode_id % 100 < 10',
            'test': 'episode_id % 100 < 5',
        },
        'scene_rows': int(writer.total_rows),
        'labeled_shards': int(writer.shard_count),
        'unique_observed_fruits': int(unique_scene_keys.numel()),
        'split_scene_counts': dict(split_scene_counts),
        'split_resolved_fruit_samples': dict(split_fruit_counts),
        'class_counts_by_level': {
            name: class_counts[name].tolist() for name in split_names
        },
    }
    _atomic_json(output_dir / 'dataset_manifest.json', manifest)
    return manifest


def labeled_scene_paths(predictor_dataset_dir):
    return tuple(sorted(
        Path(predictor_dataset_dir).joinpath('labeled').glob(
            'scenes_labeled-*.pt'
        )
    ))


def load_labeled_scene_shard(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (
            payload.get('format_version') != 1
            or payload.get('table') != 'scenes_labeled'):
        raise ValueError(f'invalid labeled scene shard: {path}')
    return payload['columns']


def scene_columns_to_state(
        columns,
        rows,
        *,
        device,
        physics_fps,
        non_blocking=False):
    rows = torch.as_tensor(rows, dtype=torch.int64)

    def take(name):
        return columns[name].index_select(0, rows).to(
            device, non_blocking=non_blocking
        )

    return TensorState(
        positions=take('positions'),
        velocities=take('velocities'),
        angular_velocities=take('angular_velocities'),
        levels=take('levels'),
        physics_radii=take('physics_radii'),
        age_frames=take('age_frames'),
        active=take('active'),
        fruit_queue=take('fruit_queue'),
        danger_progress=take('danger_progress'),
        over_danger_line=take('over_danger_line'),
        physics_fps=float(physics_fps),
    )


__all__ = [
    'LABELED_SCENE_DTYPES',
    'SCENE_DTYPES',
    'SceneSnapshotSampler',
    'episode_split',
    'extract_scene_rows',
    'label_scene_dataset',
    'labeled_scene_paths',
    'load_labeled_scene_shard',
    'scene_columns_to_state',
    'split_mask',
]
