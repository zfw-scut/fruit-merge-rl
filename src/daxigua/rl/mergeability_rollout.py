"""可合成性场景总量与逐投放变化量统计契约。"""

from __future__ import annotations

import math
from typing import NamedTuple

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
    'spatial_scene_score': torch.float32,
    'material_mass': torch.float32,
    'spatial_delta': torch.float32,
    'spatial_delta_valid': torch.bool,
    'lineage_coverage': torch.float32,
    'merge_count': torch.int16,
    'merge_score': torch.int32,
    'done': torch.bool,
}


class LineageSpatialChange(NamedTuple):
    before_score: torch.Tensor
    after_score: torch.Tensor
    delta: torch.Tensor
    comparable_mass: torch.Tensor
    total_before_mass: torch.Tensor
    coverage: torch.Tensor
    valid: torch.Tensor


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


def fruit_material_mass(levels, active, dtype=torch.float32):
    """把Ln换算成 ``2**(n-1)`` 个L1等价材料。"""

    if levels.shape != active.shape or active.dtype != torch.bool:
        raise ValueError('levels and bool active mask must share shape')
    if levels.is_floating_point():
        raise TypeError('levels must use an integer dtype')
    exponent = (levels.to(torch.long) - 1).clamp_min(0)
    mass = torch.pow(
        torch.full(levels.shape, 2.0, dtype=dtype, device=levels.device),
        exponent.to(dtype),
    )
    return mass * active.to(dtype)


def scene_spatial_values(state, result):
    """返回材料守恒加权的当前空间分数与总材料重量。"""

    if not isinstance(state, TensorState):
        raise TypeError('state must be TensorState')
    if not isinstance(result, MergeabilityResult):
        raise TypeError('result must be MergeabilityResult')
    if result.spatial_score.shape != state.active.shape:
        raise ValueError('mergeability result does not match state shape')
    mass = fruit_material_mass(
        state.levels, state.active, dtype=state.positions.dtype
    )
    total_mass = mass.sum(dim=1)
    weighted = (result.spatial_score * mass).sum(dim=1)
    score = torch.where(
        total_mass > 0.0,
        weighted / total_mass.clamp_min(1e-12),
        torch.zeros_like(weighted),
    )
    return score, total_mass


def lineage_aligned_spatial_change(
        before_fruit_ids,
        before_levels,
        before_active,
        before_spatial_score,
        after_fruit_ids,
        after_active,
        after_spatial_score,
        merge_count,
        merge_source_ids,
        merge_new_fruit_ids):
    """沿一次投放的合成谱系比较投放前已有材料的空间分数。"""

    shape = before_fruit_ids.shape
    for name, value in (
            ('before_levels', before_levels),
            ('before_active', before_active),
            ('before_spatial_score', before_spatial_score),
            ('after_fruit_ids', after_fruit_ids),
            ('after_active', after_active),
            ('after_spatial_score', after_spatial_score)):
        if value.shape != shape:
            raise ValueError(f'{name} must match before fruit shape')
    if before_active.dtype != torch.bool or after_active.dtype != torch.bool:
        raise TypeError('active masks must be bool')
    if merge_count.shape != shape[:1]:
        raise ValueError('merge_count must have shape [B]')
    if (
            merge_source_ids.ndim != 3
            or merge_source_ids.shape[0] != shape[0]
            or merge_source_ids.shape[2] != 2):
        raise ValueError('merge_source_ids must have shape [B, E, 2]')
    if merge_new_fruit_ids.shape != merge_source_ids.shape[:2]:
        raise ValueError('merge_new_fruit_ids must have shape [B, E]')

    current_ids = before_fruit_ids.to(torch.long).clone()
    traceable = before_active.clone()
    events = merge_source_ids.shape[1]
    for event_index in range(events):
        event_valid = merge_count > event_index
        sources = merge_source_ids[:, event_index].to(torch.long)
        replacement = merge_new_fruit_ids[:, event_index].to(torch.long)
        matched = (
            (current_ids.unsqueeze(2) == sources.unsqueeze(1)).any(dim=2)
            & (current_ids > 0)
            & event_valid.unsqueeze(1)
            & traceable
        )
        replacement_valid = replacement > 0
        current_ids = torch.where(
            matched,
            replacement.unsqueeze(1),
            current_ids,
        )
        traceable &= ~(matched & ~replacement_valid.unsqueeze(1))

    after_matches = (
        current_ids.unsqueeze(2) == after_fruit_ids.to(torch.long).unsqueeze(1)
    ) & after_active.unsqueeze(1) & (current_ids > 0).unsqueeze(2)
    found = after_matches.any(dim=2) & traceable & before_active
    after_slots = after_matches.to(torch.int64).argmax(dim=2)
    mapped_after_score = after_spatial_score.gather(1, after_slots)

    mass = fruit_material_mass(
        before_levels, before_active, dtype=before_spatial_score.dtype
    )
    comparable_mass_by_fruit = mass * found.to(mass.dtype)
    comparable_mass = comparable_mass_by_fruit.sum(dim=1)
    total_before_mass = mass.sum(dim=1)
    before_weighted = (
        before_spatial_score * comparable_mass_by_fruit
    ).sum(dim=1)
    after_weighted = (
        mapped_after_score * comparable_mass_by_fruit
    ).sum(dim=1)
    before_score = torch.where(
        comparable_mass > 0.0,
        before_weighted / comparable_mass.clamp_min(1e-12),
        torch.zeros_like(before_weighted),
    )
    after_score = torch.where(
        comparable_mass > 0.0,
        after_weighted / comparable_mass.clamp_min(1e-12),
        torch.zeros_like(after_weighted),
    )
    coverage = torch.where(
        total_before_mass > 0.0,
        comparable_mass / total_before_mass.clamp_min(1e-12),
        torch.zeros_like(comparable_mass),
    )
    valid = (total_before_mass > 0.0) & (coverage >= 1.0 - 1e-6)
    delta = torch.where(
        valid,
        after_score - before_score,
        torch.full_like(after_score, float('nan')),
    )
    return LineageSpatialChange(
        before_score=before_score,
        after_score=after_score,
        delta=delta,
        comparable_mass=comparable_mass,
        total_before_mass=total_before_mass,
        coverage=coverage,
        valid=valid,
    )


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
    'LineageSpatialChange',
    'SCENE_VALUE_DTYPES',
    'fruit_material_mass',
    'lineage_aligned_spatial_change',
    'scene_mergeability_delta',
    'scene_mergeability_values',
    'scene_spatial_values',
]
