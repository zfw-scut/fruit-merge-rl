"""可合成性场景总量与逐投放变化量统计契约。"""

from __future__ import annotations

import math

import torch

from .mergeability import MergeabilityResult
from .observations import TensorState


SCENE_VALUE_DTYPES = {
    'environment_id': torch.int32,
    'episode_id': torch.int64,
    'episode_seed': torch.int64,
    'decision_step': torch.int32,
    'episode_drop': torch.int32,
    'score': torch.int32,
    'fruit_count': torch.int16,
    'max_level': torch.int8,
    'scene_mergeability': torch.float32,
    'occupied_area': torch.float32,
    'area_weighted_mean': torch.float32,
    'delta': torch.float32,
    'delta_valid': torch.bool,
    'done': torch.bool,
}


def scene_mergeability_values(state, result):
    """返回 ``Σ(M·πr²)``、总圆面积与面积加权平均分。"""

    if not isinstance(state, TensorState):
        raise TypeError('state must be TensorState')
    if not isinstance(result, MergeabilityResult):
        raise TypeError('result must be MergeabilityResult')
    if result.score.shape != state.active.shape:
        raise ValueError('mergeability result does not match state shape')
    area = (
        math.pi
        * state.physics_radii.square()
        * state.active.to(state.positions.dtype)
    )
    occupied_area = area.sum(dim=1)
    weighted_sum = (result.score * area).sum(dim=1)
    weighted_mean = torch.where(
        occupied_area > 0.0,
        weighted_sum / occupied_area.clamp_min(1e-12),
        torch.zeros_like(weighted_sum),
    )
    return weighted_sum, occupied_area, weighted_mean


def scene_mergeability_delta(current, previous, previous_valid):
    """同一局存在上一投放时计算差值，否则写入NaN并标为无效。"""

    if current.shape != previous.shape or current.shape != previous_valid.shape:
        raise ValueError('delta tensors must share shape')
    if previous_valid.dtype != torch.bool:
        raise TypeError('previous_valid must be bool')
    delta = current - previous
    delta = torch.where(
        previous_valid,
        delta,
        torch.full_like(delta, float('nan')),
    )
    return delta, previous_valid.clone()


__all__ = [
    'SCENE_VALUE_DTYPES',
    'scene_mergeability_delta',
    'scene_mergeability_values',
]
