"""可复用的稳定批量场景快照数据集。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .types import BatchObservation


STATIC_SCENE_DATASET_FORMAT = 'daxigua_static_scene_dataset'
STATIC_SCENE_DATASET_VERSION = 1
OBSERVATION_FIELDS = tuple(BatchObservation.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class StaticSceneDataset:
    """一批与模拟器生命周期解耦的决策边界场景。"""

    observation: BatchObservation
    metadata: dict[str, torch.Tensor]
    manifest: dict

    @property
    def batch_size(self):
        return int(self.observation.positions.shape[0])


def _batch_size(observation):
    if not isinstance(observation, BatchObservation):
        raise TypeError('observation must be BatchObservation')
    batch_size = int(observation.positions.shape[0])
    for field_name in OBSERVATION_FIELDS:
        value = getattr(observation, field_name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'observation.{field_name} must be a tensor')
        if value.ndim == 0 or int(value.shape[0]) != batch_size:
            raise ValueError(
                f'observation.{field_name} has an invalid batch dimension'
            )
    return batch_size


def save_static_scene_dataset(
        path,
        observation,
        *,
        metadata=None,
        manifest=None):
    """把批量状态保存成仅包含张量和基础类型的可移植 ``.pt`` 文件。"""

    batch_size = _batch_size(observation)
    metadata = dict(metadata or {})
    for name, value in metadata.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'metadata {name!r} must be a tensor')
        if value.ndim == 0 or int(value.shape[0]) != batch_size:
            raise ValueError(f'metadata {name!r} has an invalid batch dimension')
    payload = {
        'format': STATIC_SCENE_DATASET_FORMAT,
        'format_version': STATIC_SCENE_DATASET_VERSION,
        'observation': {
            name: getattr(observation, name).detach().cpu()
            for name in OBSERVATION_FIELDS
        },
        'metadata': {
            name: value.detach().cpu() for name, value in metadata.items()
        },
        'manifest': dict(manifest or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _load_payload(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility.
        return torch.load(path, map_location='cpu')


def load_static_scene_dataset(path, *, device='cpu', rows=None):
    """读取场景数据集，可选按行抽取并直接迁移到目标设备。"""

    payload = _load_payload(Path(path))
    if not isinstance(payload, dict):
        raise ValueError('static scene dataset payload must be a dict')
    if payload.get('format') != STATIC_SCENE_DATASET_FORMAT:
        raise ValueError('unsupported static scene dataset format')
    if payload.get('format_version') != STATIC_SCENE_DATASET_VERSION:
        raise ValueError('unsupported static scene dataset version')
    observation_values = payload.get('observation')
    metadata_values = payload.get('metadata')
    if not isinstance(observation_values, dict):
        raise ValueError('dataset observation payload is missing')
    if not isinstance(metadata_values, dict):
        raise ValueError('dataset metadata payload is missing')

    first = observation_values.get('positions')
    if not isinstance(first, torch.Tensor) or first.ndim == 0:
        raise ValueError('dataset positions tensor is missing')
    batch_size = int(first.shape[0])
    if rows is not None:
        rows = torch.as_tensor(rows, dtype=torch.int64, device='cpu').flatten()
        if bool(((rows < 0) | (rows >= batch_size)).any().item()):
            raise IndexError('static scene dataset row is outside the batch')

    resolved = torch.device(device)

    def select(value, name):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'dataset field {name!r} must be a tensor')
        if value.ndim == 0 or int(value.shape[0]) != batch_size:
            raise ValueError(f'dataset field {name!r} has invalid batch size')
        if rows is not None:
            value = value.index_select(0, rows)
        return value.to(resolved)

    missing = [name for name in OBSERVATION_FIELDS
               if name not in observation_values]
    if missing:
        raise ValueError(f'dataset observation misses fields: {missing}')
    observation = BatchObservation(**{
        name: select(observation_values[name], name)
        for name in OBSERVATION_FIELDS
    })
    metadata = {
        name: select(value, f'metadata.{name}')
        for name, value in metadata_values.items()
    }
    dataset = StaticSceneDataset(
        observation=observation,
        metadata=metadata,
        manifest=dict(payload.get('manifest') or {}),
    )
    _batch_size(dataset.observation)
    return dataset


__all__ = [
    'OBSERVATION_FIELDS',
    'STATIC_SCENE_DATASET_FORMAT',
    'STATIC_SCENE_DATASET_VERSION',
    'StaticSceneDataset',
    'load_static_scene_dataset',
    'save_static_scene_dataset',
]
