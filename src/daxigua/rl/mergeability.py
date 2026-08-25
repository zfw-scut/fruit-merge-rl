"""固定张量形状的单水果可合成性第一代计算器。

该模块只读取稳定决策状态的水果几何，不修改模拟器、Replay 或 Policy 输入。
内部升级与外部供给通过独立组件组合，后续可以单独替换外部空间算法。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import torch
from torch import nn

from daxigua.core import (
    MAX_FRUIT_LEVEL,
    MIN_FRUIT_LEVEL,
    SPAWN_FRUIT_MAX_LEVEL,
    dropped_fruit_physics_radius,
    merged_fruit_physics_radius,
)

from .observations import TensorState


MERGEABILITY_SOURCE_NONE = 0
MERGEABILITY_SOURCE_INTERNAL = 1
MERGEABILITY_SOURCE_EXTERNAL = 2


@dataclass(frozen=True, slots=True)
class MergeabilityConfig:
    """第一代算法的可替换轻量参数。"""

    board_width: float = 560.0
    spawn_y: float = 252.0
    wall_width: float = 20.0
    neighborhood_scale: float = 1.05
    top_k: int = 3
    score_cost_scale: float = 4.0
    probe_offsets: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)

    def __post_init__(self):
        if self.board_width <= 0.0:
            raise ValueError('board_width must be positive')
        if self.wall_width <= 0.0 or self.wall_width * 2 >= self.board_width:
            raise ValueError('wall_width leaves no playable board width')
        if self.spawn_y < 0.0:
            raise ValueError('spawn_y must be non-negative')
        if self.neighborhood_scale < 1.0:
            raise ValueError('neighborhood_scale must be at least 1')
        if (
                isinstance(self.top_k, bool)
                or not isinstance(self.top_k, int)
                or self.top_k <= 0):
            raise ValueError('top_k must be positive')
        if self.score_cost_scale <= 0.0:
            raise ValueError('score_cost_scale must be positive')
        if not self.probe_offsets:
            raise ValueError('probe_offsets cannot be empty')
        if any(abs(float(offset)) > 1.0 for offset in self.probe_offsets):
            raise ValueError('probe offsets must stay within target radius')

    @classmethod
    def from_simulator_config(cls, simulator_config, **overrides):
        values = {
            'board_width': float(simulator_config.board_width),
            'spawn_y': float(simulator_config.spawn_y),
            'wall_width': float(simulator_config.wall_width),
        }
        values.update(overrides)
        return cls(**values)


class ExternalSupplyEstimate(NamedTuple):
    difficulty: torch.Tensor
    capacity_radius: torch.Tensor
    capacity_level: torch.Tensor
    entry_level: torch.Tensor


class MergeabilityResult(NamedTuple):
    score: torch.Tensor
    difficulty: torch.Tensor
    internal_score: torch.Tensor
    internal_difficulty: torch.Tensor
    external_score: torch.Tensor
    external_difficulty: torch.Tensor
    source: torch.Tensor
    primary_dependency_slot: torch.Tensor
    dependency_slots: torch.Tensor
    dependency_valid: torch.Tensor
    external_capacity_radius: torch.Tensor
    external_capacity_level: torch.Tensor
    external_entry_level: torch.Tensor


class ExternalSupplyStrategy(Protocol):
    def estimate(
            self,
            positions: torch.Tensor,
            radii: torch.Tensor,
            levels: torch.Tensor,
            active: torch.Tensor) -> ExternalSupplyEstimate:
        ...


def _standard_radius_table():
    values = [0.0]
    for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1):
        values.append(float(max(
            dropped_fruit_physics_radius(level),
            merged_fruit_physics_radius(level),
        )))
    return torch.tensor(values, dtype=torch.float32)


class VerticalCorridorExternalSupply(nn.Module):
    """用少量固定垂直探针估计顶部通路可进入水果尺度。"""

    def __init__(self, config: MergeabilityConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            'standard_radii', _standard_radius_table(), persistent=False
        )

    def estimate(self, positions, radii, levels, active):
        dtype = positions.dtype
        fruits = radii.shape[1]
        x = positions[..., 0]
        y = positions[..., 1]
        candidate_x = x.unsqueeze(1)
        candidate_y = y.unsqueeze(1)
        candidate_radius = radii.unsqueeze(1)
        target_y = y.unsqueeze(2)

        indices = torch.arange(fruits, device=positions.device)
        not_self = indices[:, None] != indices[None, :]
        between_top_and_target = (
            active.unsqueeze(1)
            & active.unsqueeze(2)
            & not_self.unsqueeze(0)
            & (candidate_y < target_y)
            & (candidate_y + candidate_radius >= float(self.config.spawn_y))
        )
        capacity = torch.zeros_like(radii)
        vertical_headroom = (y - float(self.config.spawn_y)).clamp_min(0.0)

        # 按探针循环可避免同时物化 [B, N, N, P]，保持大批量显存可控。
        for offset in self.config.probe_offsets:
            offset = float(offset)
            horizontal_offset = radii * offset
            probe_x = x + horizontal_offset
            wall_clearance = torch.minimum(
                probe_x - float(self.config.wall_width),
                float(self.config.board_width - self.config.wall_width)
                - probe_x,
            )
            approach_clearance = (
                vertical_headroom.square() + horizontal_offset.square()
            ).sqrt() - radii
            obstacle_clearance = (
                probe_x.unsqueeze(2) - candidate_x
            ).abs() - candidate_radius
            obstacle_clearance.masked_fill_(
                ~between_top_and_target, float('inf')
            )
            obstacle_clearance = obstacle_clearance.amin(dim=2)
            probe_capacity = torch.minimum(
                torch.minimum(wall_clearance, approach_clearance),
                obstacle_clearance,
            ).clamp_min(0.0)
            capacity = torch.maximum(capacity, probe_capacity)

        capacity = torch.where(active, capacity, torch.zeros_like(capacity))
        standard_radii = self.standard_radii.to(
            device=positions.device, dtype=dtype
        )
        fits = capacity.unsqueeze(-1) >= standard_radii[1:].view(1, 1, -1)
        level_values = torch.arange(
            MIN_FRUIT_LEVEL,
            MAX_FRUIT_LEVEL + 1,
            device=positions.device,
            dtype=torch.long,
        ).view(1, 1, -1)
        capacity_level = torch.where(
            fits, level_values, torch.zeros_like(level_values)
        ).amax(dim=-1)
        capacity_level = torch.where(
            active, capacity_level, torch.zeros_like(capacity_level)
        )

        target_level = levels.to(torch.long).clamp(
            MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL
        )
        entry_level = torch.minimum(
            capacity_level,
            torch.full_like(capacity_level, SPAWN_FRUIT_MAX_LEVEL),
        )
        entry_level = torch.minimum(entry_level, target_level)
        available = active & (entry_level >= MIN_FRUIT_LEVEL)
        level_gap = (target_level - entry_level).clamp_min(0)
        difficulty = torch.pow(
            torch.full_like(capacity, 2.0), level_gap.to(dtype)
        )
        difficulty = torch.where(
            available,
            difficulty,
            torch.full_like(difficulty, float('inf')),
        )
        return ExternalSupplyEstimate(
            difficulty=difficulty,
            capacity_radius=capacity,
            capacity_level=capacity_level,
            entry_level=entry_level,
        )

    forward = estimate


class MergeabilityCalculator(nn.Module):
    """计算每颗水果的内部、外部和最终可合成性。"""

    def __init__(
            self,
            config: MergeabilityConfig | None = None,
            *,
            external_supply: ExternalSupplyStrategy | None = None):
        super().__init__()
        self.config = config or MergeabilityConfig()
        self.external_supply = (
            external_supply
            if external_supply is not None
            else VerticalCorridorExternalSupply(self.config)
        )

    def _validate(self, positions, radii, levels, active):
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError('positions must have shape [B, N, 2]')
        expected = positions.shape[:2]
        if radii.shape != expected:
            raise ValueError('radii must have shape [B, N]')
        if levels.shape != expected:
            raise ValueError('levels must have shape [B, N]')
        if levels.is_floating_point():
            raise ValueError('levels must use an integer dtype')
        if active.shape != expected or active.dtype != torch.bool:
            raise ValueError('active must be bool with shape [B, N]')
        if not positions.is_floating_point() or radii.dtype != positions.dtype:
            raise ValueError('positions and radii must use the same float dtype')
        if any(value.device != positions.device for value in (
                radii, levels, active)):
            raise ValueError('all inputs must be on the same device')

    def _dependency_candidates(self, positions, radii, levels, active):
        _, fruits, _ = positions.shape
        squared_norm = positions.square().sum(dim=-1)
        distance_squared = (
            squared_norm.unsqueeze(2)
            + squared_norm.unsqueeze(1)
            - 2.0 * torch.bmm(positions, positions.transpose(1, 2))
        ).clamp_min(0.0)
        target_radius = radii.unsqueeze(2)
        candidate_radius = radii.unsqueeze(1)
        neighborhood_radius = (
            target_radius * float(self.config.neighborhood_scale)
            + candidate_radius
        ).clamp_min(1.0)
        indices = torch.arange(fruits, device=positions.device)
        pair_valid = (
            active.unsqueeze(2)
            & active.unsqueeze(1)
            & (indices[:, None] != indices[None, :]).unsqueeze(0)
        )
        neighbor = pair_valid & (
            distance_squared <= neighborhood_radius.square()
        )
        target_level = levels.to(torch.long).unsqueeze(2)
        candidate_level = levels.to(torch.long).unsqueeze(1)
        same = neighbor & (candidate_level == target_level)
        lower = neighbor & (candidate_level < target_level)
        eligible = torch.where(same.any(dim=2, keepdim=True), same, lower)

        level_gap = (target_level - candidate_level).clamp_min(0)
        normalized_distance = distance_squared / neighborhood_radius.square()
        priority = -2.0 * level_gap.to(positions.dtype) - normalized_distance
        priority = priority.masked_fill(~eligible, float('-inf'))
        selected_count = min(self.config.top_k, fruits)
        selected_priority, selected_slots = priority.topk(
            selected_count, dim=2, largest=True
        )
        selected_valid = torch.isfinite(selected_priority)
        if selected_count < self.config.top_k:
            padding = self.config.top_k - selected_count
            selected_slots = torch.cat((
                selected_slots,
                torch.zeros(
                    (*selected_slots.shape[:2], padding),
                    dtype=selected_slots.dtype,
                    device=selected_slots.device,
                ),
            ), dim=2)
            selected_valid = torch.cat((
                selected_valid,
                torch.zeros(
                    (*selected_valid.shape[:2], padding),
                    dtype=torch.bool,
                    device=selected_valid.device,
                ),
            ), dim=2)
        output_slots = torch.where(
            selected_valid, selected_slots, torch.full_like(selected_slots, -1)
        )
        return output_slots, selected_valid

    def _score(self, difficulty):
        score = 1.0 / (
            1.0 + difficulty / float(self.config.score_cost_scale)
        )
        return torch.where(
            torch.isfinite(difficulty), score, torch.zeros_like(score)
        )

    def compute(self, positions, radii, levels, active):
        self._validate(positions, radii, levels, active)
        batch, fruits = levels.shape
        dependency_slots, dependency_valid = self._dependency_candidates(
            positions, radii, levels, active
        )
        safe_slots = dependency_slots.clamp_min(0)
        expanded_levels = levels.to(torch.long).unsqueeze(1).expand(
            batch, fruits, fruits
        )
        dependency_levels = expanded_levels.gather(2, safe_slots)
        target_levels = levels.to(torch.long).unsqueeze(2)
        level_gap = (target_levels - dependency_levels).clamp_min(0)
        same_dependency = dependency_valid & (level_gap == 0)
        lower_dependency = dependency_valid & (level_gap > 0)

        external = self.external_supply.estimate(
            positions, radii, levels, active
        )
        overall_difficulty = external.difficulty.clone()
        internal_difficulty = torch.full_like(
            overall_difficulty, float('inf')
        )
        primary_dependency = torch.full(
            (batch, fruits),
            -1,
            dtype=torch.long,
            device=positions.device,
        )
        infinity = torch.full(
            (batch, fruits, self.config.top_k),
            float('inf'),
            dtype=positions.dtype,
            device=positions.device,
        )

        # 等级是固定的 1..11，按级循环只产生少量大张量 Kernel，避免逐环境对象。
        for target_level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1):
            level_mask = active & (levels == target_level)
            dependency_difficulty = overall_difficulty.unsqueeze(1).expand(
                batch, fruits, fruits
            ).gather(2, safe_slots)
            transmission_cost = torch.where(
                level_gap > 1,
                torch.pow(
                    torch.full_like(dependency_difficulty, 2.0),
                    (level_gap - 1).to(positions.dtype),
                ) - 1.0,
                torch.zeros_like(dependency_difficulty),
            )
            lower_cost = torch.where(
                lower_dependency,
                dependency_difficulty + transmission_cost,
                infinity,
            )
            best_lower_cost, best_lower_position = lower_cost.min(dim=2)
            has_same = same_dependency.any(dim=2)
            first_same_position = same_dependency.to(torch.int64).argmax(dim=2)
            current_internal = torch.where(
                has_same, torch.zeros_like(best_lower_cost), best_lower_cost
            )
            selected_position = torch.where(
                has_same, first_same_position, best_lower_position
            )
            selected_slot = safe_slots.gather(
                2, selected_position.unsqueeze(2)
            ).squeeze(2)
            selected_slot = torch.where(
                torch.isfinite(current_internal),
                selected_slot,
                torch.full_like(selected_slot, -1),
            )
            internal_difficulty = torch.where(
                level_mask, current_internal, internal_difficulty
            )
            primary_dependency = torch.where(
                level_mask, selected_slot, primary_dependency
            )
            overall_difficulty = torch.where(
                level_mask,
                torch.minimum(current_internal, external.difficulty),
                overall_difficulty,
            )

        internal_available = torch.isfinite(internal_difficulty)
        external_available = torch.isfinite(external.difficulty)
        use_internal = internal_available & (
            ~external_available | (internal_difficulty <= external.difficulty)
        )
        source = torch.full(
            (batch, fruits),
            MERGEABILITY_SOURCE_NONE,
            dtype=torch.int8,
            device=positions.device,
        )
        source = torch.where(
            active & external_available,
            torch.full_like(source, MERGEABILITY_SOURCE_EXTERNAL),
            source,
        )
        source = torch.where(
            active & use_internal,
            torch.full_like(source, MERGEABILITY_SOURCE_INTERNAL),
            source,
        )
        primary_dependency = torch.where(
            source == MERGEABILITY_SOURCE_INTERNAL,
            primary_dependency,
            torch.full_like(primary_dependency, -1),
        )
        score = self._score(overall_difficulty) * active.to(positions.dtype)
        return MergeabilityResult(
            score=score,
            difficulty=overall_difficulty,
            internal_score=self._score(internal_difficulty),
            internal_difficulty=internal_difficulty,
            external_score=self._score(external.difficulty),
            external_difficulty=external.difficulty,
            source=source,
            primary_dependency_slot=primary_dependency,
            dependency_slots=dependency_slots,
            dependency_valid=dependency_valid,
            external_capacity_radius=external.capacity_radius,
            external_capacity_level=external.capacity_level,
            external_entry_level=external.entry_level,
        )

    def forward(self, state: TensorState):
        if not isinstance(state, TensorState):
            raise TypeError('state must be TensorState')
        return self.compute(
            state.positions,
            state.physics_radii,
            state.levels,
            state.active,
        )


__all__ = [
    'ExternalSupplyEstimate',
    'ExternalSupplyStrategy',
    'MERGEABILITY_SOURCE_EXTERNAL',
    'MERGEABILITY_SOURCE_INTERNAL',
    'MERGEABILITY_SOURCE_NONE',
    'MergeabilityCalculator',
    'MergeabilityConfig',
    'MergeabilityResult',
    'VerticalCorridorExternalSupply',
]
