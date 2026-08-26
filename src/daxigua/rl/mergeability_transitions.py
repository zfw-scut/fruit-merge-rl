"""可合成性负变化分层、轻量场景快照与优先级蓄水池。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math

import torch

from daxigua.simulator.types import BatchObservation


@dataclass(frozen=True, slots=True)
class NegativeSeverityBand:
    """一个按 ``|negative delta|`` 定义的人工观察档位。"""

    code: int
    key: str
    label: str
    minimum_magnitude: float
    maximum_magnitude: float | None
    distribution_semantics: str


DEFAULT_NEGATIVE_SEVERITY_BANDS = (
    NegativeSeverityBand(
        0, 'slight', '轻微', 0.0, 7_147.75, 'negative magnitude below P50'
    ),
    NegativeSeverityBand(
        1, 'moderate', '一般', 7_147.75, 20_346.16,
        'negative magnitude P50-P75',
    ),
    NegativeSeverityBand(
        2, 'severe', '严重', 20_346.16, 61_597.65,
        'negative magnitude P75-P95',
    ),
    NegativeSeverityBand(
        3, 'extreme', '极端', 61_597.65, None,
        'negative magnitude at or above P95',
    ),
)


# 来自2026-08-26本地SAB-128自然分布中，空间负变化幅度的P50/P75/P95。
# 与旧面积指标分开登记，避免两套完全不同量纲的阈值被误用。
SPATIAL_NEGATIVE_SEVERITY_BANDS = (
    NegativeSeverityBand(
        0, 'slight', '轻微', 0.0, 0.0191891,
        'spatial negative magnitude below P50',
    ),
    NegativeSeverityBand(
        1, 'moderate', '一般', 0.0191891, 0.0990873,
        'spatial negative magnitude P50-P75',
    ),
    NegativeSeverityBand(
        2, 'severe', '严重', 0.0990873, 0.2547998,
        'spatial negative magnitude P75-P95',
    ),
    NegativeSeverityBand(
        3, 'extreme', '极端', 0.2547998, None,
        'spatial negative magnitude at or above P95',
    ),
)


COMPACT_SCENE_FIELDS = (
    'positions',
    'levels',
    'physics_radii',
    'fruit_ids',
    'active',
    'fruit_queue',
    'score',
    'step_count',
    'fruit_count',
    'max_level',
    'done',
)


def severity_band_manifest(bands=DEFAULT_NEGATIVE_SEVERITY_BANDS):
    return [asdict(band) for band in bands]


def negative_severity_codes(
        delta,
        valid,
        bands=DEFAULT_NEGATIVE_SEVERITY_BANDS):
    """将有效负变化映射到档位；非负或无效变化返回 ``-1``。"""

    if not isinstance(delta, torch.Tensor) or not isinstance(valid, torch.Tensor):
        raise TypeError('delta and valid must be tensors')
    if delta.shape != valid.shape:
        raise ValueError('delta and valid must share shape')
    if valid.dtype != torch.bool:
        raise TypeError('valid must be bool')
    codes = torch.full(delta.shape, -1, dtype=torch.int8, device=delta.device)
    magnitude = -delta
    base = valid & torch.isfinite(delta) & (delta < 0.0)
    for band in bands:
        selected = base & (magnitude >= float(band.minimum_magnitude))
        if band.minimum_magnitude == 0.0:
            selected &= magnitude > 0.0
        if band.maximum_magnitude is not None:
            selected &= magnitude < float(band.maximum_magnitude)
        codes[selected] = int(band.code)
    return codes


def _validate_observation(observation, mergeability_score):
    if not isinstance(observation, BatchObservation):
        raise TypeError('observation must be BatchObservation')
    if not isinstance(mergeability_score, torch.Tensor):
        raise TypeError('mergeability_score must be a tensor')
    if mergeability_score.shape != observation.active.shape:
        raise ValueError('mergeability_score does not match observation')


def clone_compact_scene_batch(observation, mergeability_score):
    """在当前设备保存下一投放所需的紧凑决策边界快照。"""

    _validate_observation(observation, mergeability_score)
    values = {
        name: getattr(observation, name).detach().clone()
        for name in COMPACT_SCENE_FIELDS
    }
    values['mergeability_score'] = mergeability_score.detach().clone()
    return values


def update_compact_scene_batch(target, observation, mergeability_score):
    """原位更新预分配快照，避免长期rollout逐步重新分配。"""

    _validate_observation(observation, mergeability_score)
    expected = set(COMPACT_SCENE_FIELDS) | {'mergeability_score'}
    if set(target) != expected:
        raise ValueError('compact scene target has unexpected fields')
    for name in COMPACT_SCENE_FIELDS:
        target[name].copy_(getattr(observation, name))
    target['mergeability_score'].copy_(mergeability_score)
    return target


def select_compact_scene_rows(batch, rows):
    """只把被选中的少量场景迁移到CPU。"""

    if not batch:
        raise ValueError('compact scene batch is empty')
    first = next(iter(batch.values()))
    rows = torch.as_tensor(rows, dtype=torch.int64, device=first.device).flatten()
    return {
        name: value.index_select(0, rows).detach().cpu().clone()
        for name, value in batch.items()
    }


def capture_compact_scene_rows(observation, mergeability_score, rows):
    """从当前观察直接抽取少量CPU快照，不克隆完整批次。"""

    _validate_observation(observation, mergeability_score)
    rows = torch.as_tensor(
        rows, dtype=torch.int64, device=observation.positions.device
    ).flatten()
    values = {
        name: getattr(observation, name).index_select(0, rows)
        .detach().cpu().clone()
        for name in COMPACT_SCENE_FIELDS
    }
    values['mergeability_score'] = (
        mergeability_score.index_select(0, rows).detach().cpu().clone()
    )
    return values


def compact_scene_row(batch, row):
    """把一个已迁移批次拆成可独立序列化的单场景字典。"""

    row = int(row)
    return {name: value[row].clone() for name, value in batch.items()}


class PriorityReservoir:
    """按随机优先级保留每个档位的全局最高K项。"""

    def __init__(self, band_count, capacity):
        if int(band_count) <= 0 or int(capacity) <= 0:
            raise ValueError('band_count and capacity must be positive')
        self.band_count = int(band_count)
        self.capacity = int(capacity)
        self._heaps = [[] for _ in range(self.band_count)]
        self._serial = 0

    def minimum_priority(self, band_code):
        heap = self._heaps[int(band_code)]
        return float(heap[0][0]) if len(heap) >= self.capacity else -1.0

    def add(self, band_code, priority, sample):
        band_code = int(band_code)
        priority = float(priority)
        if not 0 <= band_code < self.band_count:
            raise IndexError('band code is outside reservoir')
        if not math.isfinite(priority) or not 0.0 <= priority < 1.0:
            raise ValueError('priority must be finite and in [0, 1)')
        item = (priority, self._serial, sample)
        self._serial += 1
        heap = self._heaps[band_code]
        if len(heap) < self.capacity:
            heapq.heappush(heap, item)
            return True
        if priority <= heap[0][0]:
            return False
        heapq.heapreplace(heap, item)
        return True

    def samples(self, band_code):
        return [
            item[2]
            for item in sorted(
                self._heaps[int(band_code)], key=lambda item: item[0],
                reverse=True,
            )
        ]

    def selected_counts(self):
        return tuple(len(heap) for heap in self._heaps)


__all__ = [
    'COMPACT_SCENE_FIELDS',
    'DEFAULT_NEGATIVE_SEVERITY_BANDS',
    'NegativeSeverityBand',
    'PriorityReservoir',
    'SPATIAL_NEGATIVE_SEVERITY_BANDS',
    'capture_compact_scene_rows',
    'clone_compact_scene_batch',
    'compact_scene_row',
    'negative_severity_codes',
    'select_compact_scene_rows',
    'severity_band_manifest',
    'update_compact_scene_batch',
]
