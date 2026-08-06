"""Reward V2 的定长可投放空间几何与缓存奖励计算。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from daxigua.core.rules import (
    MAX_FRUIT_LEVEL,
    SPAWN_FRUIT_MAX_LEVEL,
    SPAWN_FRUIT_MIN_LEVEL,
    dropped_fruit_physics_radius,
    fruit_radius,
)

from .config import SimulatorConfig
from .types import BatchObservation, BatchStepResult


@dataclass(frozen=True, slots=True)
class SpatialRewardConfig:
    """Reward V2 的稳定数值参数。"""

    queue_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    reward_scale: float = 1.0

    def __post_init__(self):
        if len(self.queue_weights) != 3:
            raise ValueError('queue_weights must contain three values')
        weights = tuple(float(value) for value in self.queue_weights)
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError('queue_weights must be finite and non-negative')
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError('queue_weights must sum to one')
        reward_scale = float(self.reward_scale)
        if not math.isfinite(reward_scale) or reward_scale <= 0.0:
            raise ValueError('reward_scale must be finite and positive')
        object.__setattr__(self, 'queue_weights', weights)
        object.__setattr__(self, 'reward_scale', reward_scale)


@dataclass(frozen=True, slots=True)
class AccessibleSpaceBatch:
    """一批候选水果的21列可投放空间诊断。"""

    candidate_levels: torch.Tensor
    drop_x: torch.Tensor
    depths: torch.Tensor
    normalized_areas: torch.Tensor


@dataclass(frozen=True, slots=True)
class SpatialRewardStep:
    """一次热路径 Reward V2 计算得到的轻量 Tensor 明细。"""

    reward: torch.Tensor
    previous_potential: torch.Tensor
    next_potential: torch.Tensor
    raw_space_delta: torch.Tensor
    compensation: torch.Tensor
    unscaled_reward: torch.Tensor

    def scalar_metrics(self):
        """返回可直接送入GPU指标累加器的标量 Tensor。"""

        return {
            'spatial_previous_potential': self.previous_potential.mean(),
            'spatial_next_potential': self.next_potential.mean(),
            'spatial_raw_delta': self.raw_space_delta.mean(),
            'spatial_compensation': self.compensation.mean(),
            'spatial_unscaled_reward': self.unscaled_reward.mean(),
            'spatial_reward': self.reward.mean(),
            'spatial_positive_rate': (self.reward > 0).float().mean(),
            'spatial_negative_rate': (self.reward < 0).float().mean(),
        }


@dataclass(frozen=True, slots=True)
class SpatialRewardDiagnostics:
    """评估与前端使用的完整前后状态奖励解释。"""

    before: AccessibleSpaceBatch
    after: AccessibleSpaceBatch
    per_slot_compensation: torch.Tensor
    per_slot_raw_delta: torch.Tensor
    per_slot_unscaled_reward: torch.Tensor
    previous_potential: torch.Tensor
    next_potential: torch.Tensor
    compensation: torch.Tensor
    raw_space_delta: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor


def _rule_table(function):
    return [0.0] + [
        float(function(level))
        for level in range(1, MAX_FRUIT_LEVEL + 1)
    ]


def _action_positions(simulator_config, display_radius):
    left = simulator_config.wall_width + display_radius + 2.0
    right = (
        simulator_config.board_width
        - simulator_config.wall_width
        - display_radius
        - 2.0
    )
    denominator = simulator_config.action_count - 1
    return tuple(
        left + (right - left) * index / denominator
        for index in range(simulator_config.action_count)
    )


def build_standard_compensation_table(simulator_config):
    """按同一21列几何生成1～5级水果的标准空间占用补偿表。"""

    if not isinstance(simulator_config, SimulatorConfig):
        raise TypeError('simulator_config must be SimulatorConfig')
    table = [
        [0.0 for _ in range(MAX_FRUIT_LEVEL + 1)]
        for _ in range(MAX_FRUIT_LEVEL + 1)
    ]
    spawn_y = float(simulator_config.spawn_y)
    floor = float(
        simulator_config.board_height - simulator_config.wall_width
    )
    action_count = simulator_config.action_count

    for drop_level in range(
            SPAWN_FRUIT_MIN_LEVEL, SPAWN_FRUIT_MAX_LEVEL + 1):
        drop_display_radius = float(fruit_radius(drop_level))
        drop_radius = float(dropped_fruit_physics_radius(drop_level))
        obstacle_y = floor - drop_radius
        obstacle_positions = _action_positions(
            simulator_config, drop_display_radius
        )

        for future_level in range(
                SPAWN_FRUIT_MIN_LEVEL, SPAWN_FRUIT_MAX_LEVEL + 1):
            future_display_radius = float(fruit_radius(future_level))
            future_radius = float(
                dropped_fruit_physics_radius(future_level)
            )
            future_positions = _action_positions(
                simulator_config, future_display_radius
            )
            floor_contact_y = floor - future_radius
            empty_depth = max(0.0, floor_contact_y - spawn_y)
            horizontal_span = future_positions[-1] - future_positions[0]
            empty_area = max(1e-9, horizontal_span * empty_depth)
            radius_sum = drop_radius + future_radius
            losses = []

            for obstacle_x in obstacle_positions:
                depths = []
                for probe_x in future_positions:
                    dx = abs(probe_x - obstacle_x)
                    contact_y = floor_contact_y
                    if dx < radius_sum:
                        half_height = math.sqrt(max(
                            0.0, radius_sum * radius_sum - dx * dx
                        ))
                        upper = obstacle_y - half_height
                        lower = obstacle_y + half_height
                        if lower >= spawn_y:
                            contact_y = min(
                                contact_y, max(spawn_y, upper)
                            )
                    depths.append(max(
                        0.0, min(empty_depth, contact_y - spawn_y)
                    ))

                column_step = horizontal_span / (action_count - 1)
                area = column_step * (
                    0.5 * depths[0]
                    + sum(depths[1:-1])
                    + 0.5 * depths[-1]
                )
                losses.append(max(0.0, min(1.0, 1.0 - area / empty_area)))
            table[drop_level][future_level] = sum(losses) / len(losses)
    return tuple(tuple(row) for row in table)


class AccessibleSpaceCalculator:
    """用GPU定长张量计算候选水果的21列垂直可投放空间。"""

    def __init__(
            self,
            simulator_config,
            *,
            device,
            reward_config=None):
        if not isinstance(simulator_config, SimulatorConfig):
            raise TypeError('simulator_config must be SimulatorConfig')
        if simulator_config.action_count != 21:
            raise ValueError('Reward V2 requires exactly 21 actions')
        self.simulator_config = simulator_config
        self.reward_config = reward_config or SpatialRewardConfig()
        if not isinstance(self.reward_config, SpatialRewardConfig):
            raise TypeError('reward_config must be SpatialRewardConfig')
        self.device = torch.device(device)
        self.display_radii = torch.tensor(
            _rule_table(fruit_radius),
            dtype=torch.float32,
            device=self.device,
        )
        self.drop_radii = torch.tensor(
            _rule_table(dropped_fruit_physics_radius),
            dtype=torch.float32,
            device=self.device,
        )
        self.action_alpha = torch.linspace(
            0.0,
            1.0,
            simulator_config.action_count,
            dtype=torch.float32,
            device=self.device,
        )
        trapezoid_weights = torch.ones(
            simulator_config.action_count,
            dtype=torch.float32,
            device=self.device,
        )
        trapezoid_weights[0] = 0.5
        trapezoid_weights[-1] = 0.5
        self.trapezoid_weights = trapezoid_weights
        self.queue_weights = torch.tensor(
            self.reward_config.queue_weights,
            dtype=torch.float32,
            device=self.device,
        )
        self.compensation_table = torch.tensor(
            build_standard_compensation_table(simulator_config),
            dtype=torch.float32,
            device=self.device,
        )

    def _candidate_geometry(self, candidate_levels):
        display_radius = self.display_radii[candidate_levels]
        physics_radius = self.drop_radii[candidate_levels]
        left = (
            float(self.simulator_config.wall_width)
            + display_radius
            + 2.0
        )
        right = (
            float(
                self.simulator_config.board_width
                - self.simulator_config.wall_width
            )
            - display_radius
            - 2.0
        )
        drop_x = (
            left[..., None]
            + (right - left)[..., None] * self.action_alpha
        )
        floor_y = (
            float(
                self.simulator_config.board_height
                - self.simulator_config.wall_width
            )
            - physics_radius
        )
        return physics_radius, left, right, floor_y, drop_x

    @torch.no_grad()
    def analyze(self, observation, candidate_levels):
        """返回形状 ``[B,K,21]`` 的深度与 ``[B,K]`` 的归一化面积。"""

        if not isinstance(observation, BatchObservation):
            raise TypeError('observation must be BatchObservation')
        candidate_levels = torch.as_tensor(
            candidate_levels,
            dtype=torch.int64,
            device=self.device,
        )
        if candidate_levels.ndim != 2:
            raise ValueError('candidate_levels must have shape [B, K]')
        batch_size = candidate_levels.shape[0]
        if batch_size > observation.positions.shape[0]:
            raise ValueError('candidate batch exceeds observation batch')

        query_radius, left, right, floor_y, drop_x = (
            self._candidate_geometry(candidate_levels)
        )
        obstacle_x = observation.positions[:batch_size, None, None, :, 0]
        obstacle_y = observation.positions[:batch_size, None, None, :, 1]
        obstacle_radius = observation.physics_radii[
            :batch_size, None, None, :
        ]
        active = observation.active[:batch_size, None, None, :]

        dx = (drop_x[..., None] - obstacle_x).abs()
        radius_sum = query_radius[..., None, None] + obstacle_radius
        intersects = active & (dx < radius_sum)
        half_height = (
            radius_sum.square() - dx.square()
        ).clamp_min(0.0).sqrt()
        upper = obstacle_y - half_height
        lower = obstacle_y + half_height
        spawn_y = float(self.simulator_config.spawn_y)
        blocks = intersects & (lower >= spawn_y)
        contact_y = torch.maximum(
            upper, torch.full_like(upper, spawn_y)
        )
        contact_y = torch.where(
            blocks,
            contact_y,
            torch.full_like(contact_y, float('inf')),
        )
        first_obstacle_y = contact_y.amin(dim=-1)
        first_contact_y = torch.minimum(
            first_obstacle_y, floor_y[..., None]
        )
        empty_depth = (floor_y - spawn_y).clamp_min(0.0)
        depths = (first_contact_y - spawn_y).clamp_min(0.0)
        depths = torch.minimum(depths, empty_depth[..., None])

        horizontal_span = (right - left).clamp_min(0.0)
        column_step = horizontal_span / (
            self.simulator_config.action_count - 1
        )
        area = (
            depths * self.trapezoid_weights
        ).sum(dim=-1) * column_step
        empty_area = (horizontal_span * empty_depth).clamp_min(1e-9)
        normalized = (area / empty_area).clamp(0.0, 1.0)
        return AccessibleSpaceBatch(
            candidate_levels=candidate_levels,
            drop_x=drop_x,
            depths=depths,
            normalized_areas=normalized,
        )

    @torch.no_grad()
    def normalized_areas(self, observation, candidate_levels):
        return self.analyze(observation, candidate_levels).normalized_areas

    def weighted_potential(self, normalized_areas):
        if normalized_areas.shape[-1] != 3:
            raise ValueError('weighted potential requires three queue slots')
        return (normalized_areas * self.queue_weights).sum(dim=-1)

    def compensation_by_slot(self, drop_levels, future_levels):
        drop_levels = torch.as_tensor(
            drop_levels, dtype=torch.int64, device=self.device
        )
        future_levels = torch.as_tensor(
            future_levels, dtype=torch.int64, device=self.device
        )
        return self.compensation_table[
            drop_levels[..., None], future_levels
        ]


class SpatialRewardComputer:
    """复用相邻决策状态势能的正式训练热路径Reward V2。"""

    requires_previous_state = True

    def __init__(
            self,
            simulator_config,
            *,
            device,
            reward_config=None):
        self.calculator = AccessibleSpaceCalculator(
            simulator_config,
            device=device,
            reward_config=reward_config,
        )
        self.reward_config = self.calculator.reward_config
        self._previous_potential = None

    @property
    def initialized(self):
        return self._previous_potential is not None

    @torch.no_grad()
    def initialize(self, observation):
        queue_levels = observation.fruit_queue[:, 1:4]
        areas = self.calculator.normalized_areas(
            observation, queue_levels
        )
        self._previous_potential = self.calculator.weighted_potential(areas)
        return self._previous_potential

    @torch.no_grad()
    def reset_rows(self, reset_mask):
        """空场reset的三个队列水果势能都为1，无需重新运行几何。"""

        if self._previous_potential is None:
            return
        reset_mask = torch.as_tensor(
            reset_mask,
            dtype=torch.bool,
            device=self._previous_potential.device,
        )
        self._previous_potential.masked_fill_(reset_mask, 1.0)

    @torch.no_grad()
    def step(self, result, *, batch_size=None):
        if not isinstance(result, BatchStepResult):
            raise TypeError('result must be BatchStepResult')
        if self._previous_potential is None:
            raise RuntimeError('SpatialRewardComputer must be initialized')
        if batch_size is None:
            batch_size = result.drop.queue_after.shape[0]
        batch_size = int(batch_size)
        queue_before = result.drop.queue_before[:batch_size]
        queue_after = result.drop.queue_after[:batch_size]

        # 新状态一次计算q0～q3：前三项完成当前奖励，后三项缓存给下一步。
        areas = self.calculator.normalized_areas(
            result.observation, queue_after
        )
        measured_next = self.calculator.weighted_potential(areas[:, :3])
        terminals = result.physics.done[:batch_size]
        next_potential = torch.where(
            terminals, torch.zeros_like(measured_next), measured_next
        )
        previous_potential = self._previous_potential[:batch_size].clone()
        per_slot_compensation = self.calculator.compensation_by_slot(
            queue_before[:, 0], queue_before[:, 1:4]
        )
        compensation = self.calculator.weighted_potential(
            per_slot_compensation
        )
        raw_space_delta = next_potential - previous_potential
        unscaled_reward = raw_space_delta + compensation
        reward = self.reward_config.reward_scale * unscaled_reward

        following_potential = self.calculator.weighted_potential(areas[:, 1:4])
        self._previous_potential[:batch_size].copy_(following_potential)
        return SpatialRewardStep(
            reward=reward,
            previous_potential=previous_potential,
            next_potential=next_potential,
            raw_space_delta=raw_space_delta,
            compensation=compensation,
            unscaled_reward=unscaled_reward,
        )


@torch.no_grad()
def diagnose_spatial_reward(
        calculator,
        previous_observation,
        result,
        *,
        batch_size=None):
    """重算前后21列明细，供抽样评估和前端解释，不进入训练热路径。"""

    if not isinstance(calculator, AccessibleSpaceCalculator):
        raise TypeError('calculator must be AccessibleSpaceCalculator')
    if not isinstance(previous_observation, BatchObservation):
        raise TypeError('previous_observation must be BatchObservation')
    if not isinstance(result, BatchStepResult):
        raise TypeError('result must be BatchStepResult')
    if batch_size is None:
        batch_size = result.drop.queue_before.shape[0]
    batch_size = int(batch_size)
    queue_before = result.drop.queue_before[:batch_size]
    aligned_levels = queue_before[:, 1:4]
    before = calculator.analyze(previous_observation, aligned_levels)
    after = calculator.analyze(result.observation, aligned_levels)
    terminal = result.physics.done[:batch_size, None]
    after_areas = torch.where(
        terminal,
        torch.zeros_like(after.normalized_areas),
        after.normalized_areas,
    )
    per_slot_raw_delta = after_areas - before.normalized_areas
    per_slot_compensation = calculator.compensation_by_slot(
        queue_before[:, 0], aligned_levels
    )
    per_slot_unscaled = per_slot_raw_delta + per_slot_compensation
    previous_potential = calculator.weighted_potential(
        before.normalized_areas
    )
    next_potential = calculator.weighted_potential(after_areas)
    compensation = calculator.weighted_potential(per_slot_compensation)
    raw_space_delta = next_potential - previous_potential
    reward = (
        calculator.reward_config.reward_scale
        * calculator.weighted_potential(per_slot_unscaled)
    )
    return SpatialRewardDiagnostics(
        before=before,
        after=after,
        per_slot_compensation=per_slot_compensation,
        per_slot_raw_delta=per_slot_raw_delta,
        per_slot_unscaled_reward=per_slot_unscaled,
        previous_potential=previous_potential,
        next_potential=next_potential,
        compensation=compensation,
        raw_space_delta=raw_space_delta,
        reward=reward,
        terminal=terminal.squeeze(-1),
    )


__all__ = [
    'AccessibleSpaceBatch',
    'AccessibleSpaceCalculator',
    'SpatialRewardComputer',
    'SpatialRewardConfig',
    'SpatialRewardDiagnostics',
    'SpatialRewardStep',
    'build_standard_compensation_table',
    'diagnose_spatial_reward',
]
