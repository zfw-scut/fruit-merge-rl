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
    reference_mode: str = 'empty_average'

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
        if self.reference_mode not in ('empty_average', 'best_no_merge'):
            raise ValueError(
                'reference_mode must be empty_average or best_no_merge'
            )
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
    reference_loss: torch.Tensor
    unscaled_reward: torch.Tensor

    @property
    def compensation(self):
        """兼容旧监控读取；V2.1中该值是当前状态参考损失。"""

        return self.reference_loss

    def scalar_metrics(self):
        """返回可直接送入GPU指标累加器的标量 Tensor。"""

        return {
            'spatial_previous_potential': self.previous_potential.mean(),
            'spatial_next_potential': self.next_potential.mean(),
            'spatial_raw_delta': self.raw_space_delta.mean(),
            'spatial_reference_loss': self.reference_loss.mean(),
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
    per_slot_reference_loss: torch.Tensor
    per_slot_raw_delta: torch.Tensor
    per_slot_unscaled_reward: torch.Tensor
    previous_potential: torch.Tensor
    next_potential: torch.Tensor
    reference_loss: torch.Tensor
    reference_action: torch.Tensor
    raw_space_delta: torch.Tensor
    reward: torch.Tensor
    terminal: torch.Tensor

    @property
    def per_slot_compensation(self):
        return self.per_slot_reference_loss

    @property
    def compensation(self):
        return self.reference_loss


@dataclass(frozen=True, slots=True)
class NoMergeReferenceBatch:
    """当前状态中最佳普通无合成投放的空间损失。"""

    loss: torch.Tensor
    action: torch.Tensor
    per_slot_loss: torch.Tensor


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


def _build_empty_action_loss_table(simulator_config):
    """生成空场中每种投放/未来水果组合的21动作空间损失。"""

    if not isinstance(simulator_config, SimulatorConfig):
        raise TypeError('simulator_config must be SimulatorConfig')
    table = [
        [
            [0.0 for _ in range(simulator_config.action_count)]
            for _ in range(MAX_FRUIT_LEVEL + 1)
        ]
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
            table[drop_level][future_level] = losses
    return tuple(
        tuple(tuple(actions) for actions in future_rows)
        for future_rows in table
    )


def build_standard_compensation_table(simulator_config):
    """生成旧版空场21动作平均空间占用补偿表。"""

    action_losses = _build_empty_action_loss_table(simulator_config)
    table = []
    for future_rows in action_losses:
        table.append(tuple(
            sum(actions) / len(actions) for actions in future_rows
        ))
    return tuple(table)


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
        self.empty_action_loss_table = torch.tensor(
            _build_empty_action_loss_table(simulator_config),
            dtype=torch.float32,
            device=self.device,
        )
        self.compensation_table = self.empty_action_loss_table.mean(dim=-1)

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

    def _validate_reference_analysis(self, analysis):
        if not isinstance(analysis, AccessibleSpaceBatch):
            raise TypeError('analysis must be AccessibleSpaceBatch')
        if analysis.candidate_levels.shape[-1] != 4:
            raise ValueError('no-merge reference requires q0 through q3')

    @torch.no_grad()
    def no_merge_reference(self, analysis):
        """计算当前状态下21个无合成幽灵投放中的最小空间损失。"""

        self._validate_reference_analysis(analysis)
        levels = analysis.candidate_levels
        drop_radius = self.drop_radii[levels[:, 0]]
        future_radius = self.drop_radii[levels[:, 1:4]]
        ghost_x = analysis.drop_x[:, 0]
        ghost_y = float(self.simulator_config.spawn_y) + analysis.depths[:, 0]
        future_x = analysis.drop_x[:, 1:4]

        dx = (
            future_x[:, :, None, :]
            - ghost_x[:, None, :, None]
        ).abs()
        radius_sum = (
            future_radius[:, :, None, None]
            + drop_radius[:, None, None, None]
        )
        intersects = dx < radius_sum
        half_height = (
            radius_sum.square() - dx.square()
        ).clamp_min(0.0).sqrt()
        upper = ghost_y[:, None, :, None] - half_height
        lower = ghost_y[:, None, :, None] + half_height
        spawn_y = float(self.simulator_config.spawn_y)
        blocks = intersects & (lower >= spawn_y)
        ghost_depth = (upper.clamp_min(spawn_y) - spawn_y).clamp_min(0.0)
        ghost_depth = torch.where(
            blocks,
            ghost_depth,
            torch.full_like(ghost_depth, float('inf')),
        )
        depths = torch.minimum(
            analysis.depths[:, 1:4, None, :], ghost_depth
        )

        _, left, right, floor_y, _ = self._candidate_geometry(levels[:, 1:4])
        horizontal_span = (right - left).clamp_min(0.0)
        column_step = horizontal_span / (
            self.simulator_config.action_count - 1
        )
        area = (
            depths * self.trapezoid_weights
        ).sum(dim=-1) * column_step[:, :, None]
        empty_depth = (
            floor_y - float(self.simulator_config.spawn_y)
        ).clamp_min(0.0)
        empty_area = (horizontal_span * empty_depth).clamp_min(1e-9)
        ghost_areas = (area / empty_area[:, :, None]).clamp(0.0, 1.0)
        per_slot_action_loss = (
            analysis.normalized_areas[:, 1:4, None] - ghost_areas
        ).clamp_min(0.0)
        action_loss = (
            per_slot_action_loss * self.queue_weights[None, :, None]
        ).sum(dim=1)
        best_loss, best_action = action_loss.min(dim=-1)
        best_slot_loss = per_slot_action_loss.gather(
            2,
            best_action[:, None, None].expand(-1, 3, 1),
        ).squeeze(-1)
        return NoMergeReferenceBatch(
            loss=best_loss,
            action=best_action,
            per_slot_loss=best_slot_loss,
        )

    def empty_no_merge_reference(self, fruit_queue):
        """不运行场景几何，直接查表得到reset空场的最佳参考损失。"""

        fruit_queue = torch.as_tensor(
            fruit_queue, dtype=torch.int64, device=self.device
        )
        if fruit_queue.ndim != 2 or fruit_queue.shape[-1] < 4:
            raise ValueError('fruit_queue must have shape [B, >=4]')
        batch_size = fruit_queue.shape[0]
        actions = torch.arange(
            self.simulator_config.action_count,
            dtype=torch.int64,
            device=self.device,
        )
        per_slot_action_loss = self.empty_action_loss_table[
            fruit_queue[:, 0, None, None],
            fruit_queue[:, 1:4, None],
            actions[None, None, :],
        ]
        action_loss = (
            per_slot_action_loss * self.queue_weights[None, :, None]
        ).sum(dim=1)
        best_loss, best_action = action_loss.min(dim=-1)
        best_slot_loss = per_slot_action_loss.gather(
            2,
            best_action[:, None, None].expand(batch_size, 3, 1),
        ).squeeze(-1)
        return NoMergeReferenceBatch(
            loss=best_loss,
            action=best_action,
            per_slot_loss=best_slot_loss,
        )


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
        self._reference_loss = None

    @property
    def initialized(self):
        return self._previous_potential is not None

    @torch.no_grad()
    def initialize(self, observation):
        queue_levels = observation.fruit_queue[:, :4]
        analysis = self.calculator.analyze(observation, queue_levels)
        self._previous_potential = self.calculator.weighted_potential(
            analysis.normalized_areas[:, 1:4]
        )
        if self.reward_config.reference_mode == 'best_no_merge':
            self._reference_loss = self.calculator.no_merge_reference(
                analysis
            ).loss
        return self._previous_potential

    @torch.no_grad()
    def reset_rows(self, reset_mask, observation=None):
        """重置势能；V2.1同时按新队列查表恢复状态相关参考损失。"""

        if self._previous_potential is None:
            return
        reset_mask = torch.as_tensor(
            reset_mask,
            dtype=torch.bool,
            device=self._previous_potential.device,
        )
        self._previous_potential.masked_fill_(reset_mask, 1.0)
        if self.reward_config.reference_mode == 'best_no_merge':
            if observation is None:
                raise ValueError(
                    'best_no_merge reset requires the reset observation'
                )
            reference = self.calculator.empty_no_merge_reference(
                observation.fruit_queue[reset_mask]
            )
            self._reference_loss[reset_mask] = reference.loss

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
        analysis = self.calculator.analyze(result.observation, queue_after)
        areas = analysis.normalized_areas
        measured_next = self.calculator.weighted_potential(areas[:, :3])
        terminals = result.physics.done[:batch_size]
        next_potential = torch.where(
            terminals, torch.zeros_like(measured_next), measured_next
        )
        previous_potential = self._previous_potential[:batch_size].clone()
        if self.reward_config.reference_mode == 'best_no_merge':
            reference_loss = self._reference_loss[:batch_size].clone()
        else:
            per_slot_reference = self.calculator.compensation_by_slot(
                queue_before[:, 0], queue_before[:, 1:4]
            )
            reference_loss = self.calculator.weighted_potential(
                per_slot_reference
            )
        raw_space_delta = next_potential - previous_potential
        unscaled_reward = raw_space_delta + reference_loss
        reward = self.reward_config.reward_scale * unscaled_reward

        following_potential = self.calculator.weighted_potential(areas[:, 1:4])
        self._previous_potential[:batch_size].copy_(following_potential)
        if self.reward_config.reference_mode == 'best_no_merge':
            following_reference = self.calculator.no_merge_reference(analysis)
            self._reference_loss[:batch_size].copy_(following_reference.loss)
        return SpatialRewardStep(
            reward=reward,
            previous_potential=previous_potential,
            next_potential=next_potential,
            raw_space_delta=raw_space_delta,
            reference_loss=reference_loss,
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
    before_all = calculator.analyze(previous_observation, queue_before[:, :4])
    before = AccessibleSpaceBatch(
        candidate_levels=before_all.candidate_levels[:, 1:4],
        drop_x=before_all.drop_x[:, 1:4],
        depths=before_all.depths[:, 1:4],
        normalized_areas=before_all.normalized_areas[:, 1:4],
    )
    after = calculator.analyze(result.observation, aligned_levels)
    terminal = result.physics.done[:batch_size, None]
    after_areas = torch.where(
        terminal,
        torch.zeros_like(after.normalized_areas),
        after.normalized_areas,
    )
    per_slot_raw_delta = after_areas - before.normalized_areas
    if calculator.reward_config.reference_mode == 'best_no_merge':
        reference = calculator.no_merge_reference(before_all)
        per_slot_reference = reference.per_slot_loss
        reference_loss = reference.loss
        reference_action = reference.action
    else:
        per_slot_reference = calculator.compensation_by_slot(
            queue_before[:, 0], aligned_levels
        )
        reference_loss = calculator.weighted_potential(per_slot_reference)
        reference_action = torch.full(
            (batch_size,), -1, dtype=torch.int64, device=calculator.device
        )
    per_slot_unscaled = per_slot_raw_delta + per_slot_reference
    previous_potential = calculator.weighted_potential(
        before.normalized_areas
    )
    next_potential = calculator.weighted_potential(after_areas)
    raw_space_delta = next_potential - previous_potential
    # 与训练热路径保持完全相同的浮点运算顺序，避免诊断显示与实际奖励
    # 因加权求和结合顺序不同而产生约1e-7的偏差。
    reward = calculator.reward_config.reward_scale * (
        raw_space_delta + reference_loss
    )
    return SpatialRewardDiagnostics(
        before=before,
        after=after,
        per_slot_reference_loss=per_slot_reference,
        per_slot_raw_delta=per_slot_raw_delta,
        per_slot_unscaled_reward=per_slot_unscaled,
        previous_potential=previous_potential,
        next_potential=next_potential,
        reference_loss=reference_loss,
        reference_action=reference_action,
        raw_space_delta=raw_space_delta,
        reward=reward,
        terminal=terminal.squeeze(-1),
    )


__all__ = [
    'AccessibleSpaceBatch',
    'AccessibleSpaceCalculator',
    'NoMergeReferenceBatch',
    'SpatialRewardComputer',
    'SpatialRewardConfig',
    'SpatialRewardDiagnostics',
    'SpatialRewardStep',
    'build_standard_compensation_table',
    'diagnose_spatial_reward',
]
