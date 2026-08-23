"""同级高等级水果对的决策级长期停滞检测。

检测器只读取每次投放完成后的稳定 ``BatchObservation``，不读取瞬时速度、
角速度或逐物理帧轨迹。它使用固定水果槽位对和 ``fruit_id`` 在 CPU/CUDA 上
进行同形状张量更新，为离线失败标签与反事实长期验证提供事件起点和确认点。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class PairFailureConfig:
    """长期停滞检测的少量尺度无关参数。"""

    motion_window_drops: int = 4
    confirmation_drops: int = 24
    max_net_displacement_ratio: float = 0.12
    adjacent_surface_gap_ratio: float = 1.25

    def __post_init__(self):
        if self.motion_window_drops <= 0:
            raise ValueError('motion_window_drops must be positive')
        if self.confirmation_drops <= self.motion_window_drops:
            raise ValueError(
                'confirmation_drops must exceed motion_window_drops'
            )
        if self.max_net_displacement_ratio < 0.0:
            raise ValueError(
                'max_net_displacement_ratio must be non-negative'
            )
        if self.adjacent_surface_gap_ratio < 0.0:
            raise ValueError(
                'adjacent_surface_gap_ratio must be non-negative'
            )


@dataclass(frozen=True, slots=True)
class PairFailureUpdate:
    """一次决策边界更新产生的固定形状事件张量。"""

    started: torch.Tensor
    confirmed: torch.Tensor
    ended: torch.Tensor
    ended_after_confirmation: torch.Tensor
    active_candidates: torch.Tensor
    onset_steps: torch.Tensor
    duration_drops: torch.Tensor
    levels: torch.Tensor
    fruit_id_i: torch.Tensor
    fruit_id_j: torch.Tensor
    net_displacement_ratio_i: torch.Tensor
    net_displacement_ratio_j: torch.Tensor
    surface_gap_ratio: torch.Tensor


class PairFailureTracker:
    """以固定槽位对批量跟踪 L7～L11 同级水果的长期停滞。"""

    def __init__(
            self,
            num_envs: int,
            max_fruits: int,
            *,
            device,
            config: PairFailureConfig | None = None):
        if num_envs <= 0:
            raise ValueError('num_envs must be positive')
        if max_fruits < 2:
            raise ValueError('max_fruits must be at least 2')
        self.num_envs = int(num_envs)
        self.max_fruits = int(max_fruits)
        self.device = torch.device(device)
        self.config = config or PairFailureConfig()

        pair_indices = torch.triu_indices(
            self.max_fruits,
            self.max_fruits,
            offset=1,
            device=self.device,
        )
        self.pair_i = pair_indices[0]
        self.pair_j = pair_indices[1]
        self.pair_count = int(self.pair_i.numel())

        history_size = self.config.motion_window_drops + 1
        self._position_history = torch.zeros(
            history_size,
            self.num_envs,
            self.max_fruits,
            2,
            dtype=torch.float32,
            device=self.device,
        )
        self._id_history = torch.zeros(
            history_size,
            self.num_envs,
            self.max_fruits,
            dtype=torch.int64,
            device=self.device,
        )
        self._history_age = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._history_cursor = -1

        slot_shape = (self.num_envs, self.max_fruits)
        pair_shape = (self.num_envs, self.pair_count)
        self._previous_ids = torch.zeros(
            slot_shape, dtype=torch.int64, device=self.device
        )
        self._previous_levels = torch.zeros_like(self._previous_ids)
        self._adjacency_streak = torch.zeros(
            pair_shape, dtype=torch.int32, device=self.device
        )
        self._candidate_streak = torch.zeros_like(self._adjacency_streak)
        self._onset_steps = torch.full(
            pair_shape, -1, dtype=torch.int64, device=self.device
        )
        self._confirmed = torch.zeros(
            pair_shape, dtype=torch.bool, device=self.device
        )

    def _normalize_reset_mask(self, mask):
        if mask is None:
            return torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if mask.shape != (self.num_envs,):
            raise ValueError('reset mask has an invalid shape')
        return mask

    @torch.no_grad()
    def reset(self, mask=None):
        """清除指定环境的跨决策历史，不修改模拟器状态。"""

        mask = self._normalize_reset_mask(mask)
        self._position_history[:, mask] = 0.0
        self._id_history[:, mask] = 0
        self._history_age[mask] = 0
        self._previous_ids[mask] = 0
        self._previous_levels[mask] = 0
        self._adjacency_streak[mask] = 0
        self._candidate_streak[mask] = 0
        self._onset_steps[mask] = -1
        self._confirmed[mask] = False

    def _validate_inputs(
            self, positions, levels, physics_radii, fruit_ids, active,
            step_count):
        expected_slots = (self.num_envs, self.max_fruits)
        if positions.shape != (*expected_slots, 2):
            raise ValueError('positions has an invalid shape')
        for name, value in (
                ('levels', levels),
                ('physics_radii', physics_radii),
                ('fruit_ids', fruit_ids),
                ('active', active)):
            if value.shape != expected_slots:
                raise ValueError(f'{name} has an invalid shape')
        if step_count.shape != (self.num_envs,):
            raise ValueError('step_count has an invalid shape')
        values = (
            positions, levels, physics_radii, fruit_ids, active, step_count
        )
        if any(value.device != self.device for value in values):
            raise ValueError('all detector inputs must use the tracker device')

    @torch.no_grad()
    def update_observation(self, observation):
        """用一个稳定决策状态更新检测器。"""

        return self.update(
            observation.positions,
            observation.levels,
            observation.physics_radii,
            observation.fruit_ids,
            observation.active,
            observation.step_count,
        )

    @torch.no_grad()
    def update(
            self,
            positions,
            levels,
            physics_radii,
            fruit_ids,
            active,
            step_count):
        """更新固定水果对状态，并仅在开始、确认和结束时置事件位。"""

        positions = torch.as_tensor(
            positions, dtype=torch.float32, device=self.device
        )
        levels = torch.as_tensor(levels, dtype=torch.int64, device=self.device)
        physics_radii = torch.as_tensor(
            physics_radii, dtype=torch.float32, device=self.device
        )
        fruit_ids = torch.as_tensor(
            fruit_ids, dtype=torch.int64, device=self.device
        )
        active = torch.as_tensor(active, dtype=torch.bool, device=self.device)
        step_count = torch.as_tensor(
            step_count, dtype=torch.int64, device=self.device
        )
        self._validate_inputs(
            positions, levels, physics_radii, fruit_ids, active, step_count
        )

        history_size = self.config.motion_window_drops + 1
        self._history_cursor = (self._history_cursor + 1) % history_size
        cursor = self._history_cursor
        self._position_history[cursor].copy_(positions)
        self._id_history[cursor].copy_(torch.where(
            active, fruit_ids, torch.zeros_like(fruit_ids)
        ))
        self._history_age.add_(1).clamp_(max=history_size)

        reference_index = (
            cursor - self.config.motion_window_drops
        ) % history_size
        reference_positions = self._position_history[reference_index]
        reference_ids = self._id_history[reference_index]
        history_valid = (
            self._history_age > self.config.motion_window_drops
        )[:, None]
        same_reference_fruit = (
            history_valid
            & active
            & fruit_ids.gt(0)
            & fruit_ids.eq(reference_ids)
        )
        net_displacement_ratio = torch.linalg.vector_norm(
            positions - reference_positions, dim=-1
        ) / physics_radii.clamp_min(1.0)
        stationary = (
            same_reference_fruit
            & net_displacement_ratio.le(
                self.config.max_net_displacement_ratio
            )
        )

        ids_i = fruit_ids[:, self.pair_i]
        ids_j = fruit_ids[:, self.pair_j]
        levels_i = levels[:, self.pair_i]
        levels_j = levels[:, self.pair_j]
        pair_active = (
            active[:, self.pair_i]
            & active[:, self.pair_j]
            & ids_i.gt(0)
            & ids_j.gt(0)
        )
        same_level = pair_active & levels_i.eq(levels_j)
        stationary_pair = (
            stationary[:, self.pair_i] & stationary[:, self.pair_j]
        )

        position_delta = (
            positions[:, self.pair_i] - positions[:, self.pair_j]
        )
        center_distance = torch.linalg.vector_norm(position_delta, dim=-1)
        radii_i = physics_radii[:, self.pair_i]
        radii_j = physics_radii[:, self.pair_j]
        surface_gap_ratio = (
            center_distance - radii_i - radii_j
        ).clamp_min(0.0) / torch.minimum(radii_i, radii_j).clamp_min(1.0)

        previous_ids_i = self._previous_ids[:, self.pair_i]
        previous_ids_j = self._previous_ids[:, self.pair_j]
        same_pair_as_previous = (
            ids_i.eq(previous_ids_i)
            & ids_j.eq(previous_ids_j)
            & ids_i.gt(0)
            & ids_j.gt(0)
        )
        medium_pair = same_level & levels_i.ge(7) & levels_i.le(8)
        adjacent_now = (
            medium_pair
            & surface_gap_ratio.le(
                self.config.adjacent_surface_gap_ratio
            )
        )
        self._adjacency_streak = torch.where(
            adjacent_now,
            torch.where(
                same_pair_as_previous,
                self._adjacency_streak + 1,
                torch.ones_like(self._adjacency_streak),
            ),
            torch.zeros_like(self._adjacency_streak),
        )
        adjacent_for_motion_window = (
            self._adjacency_streak
            >= self.config.motion_window_drops + 1
        )

        high_pair = same_level & levels_i.ge(9) & levels_i.le(11)
        eligible = stationary_pair & (
            high_pair | (medium_pair & adjacent_for_motion_window)
        )

        previous_streak = self._candidate_streak
        previous_confirmed = self._confirmed
        previous_onset = self._onset_steps
        started = eligible & previous_streak.eq(0)
        ended = ~eligible & previous_streak.gt(0)
        next_streak = torch.where(
            eligible,
            previous_streak + 1,
            torch.zeros_like(previous_streak),
        )
        new_onset = (
            step_count[:, None] - self.config.motion_window_drops
        )
        reported_onset = torch.where(started, new_onset, previous_onset)
        duration_drops = torch.where(
            eligible,
            next_streak.to(torch.int64)
            + self.config.motion_window_drops - 1,
            torch.where(
                ended,
                previous_streak.to(torch.int64)
                + self.config.motion_window_drops - 1,
                torch.zeros_like(previous_streak, dtype=torch.int64),
            ),
        )
        confirmed = (
            eligible
            & ~previous_confirmed
            & duration_drops.ge(self.config.confirmation_drops)
        )
        next_confirmed = torch.where(
            eligible,
            previous_confirmed | confirmed,
            torch.zeros_like(previous_confirmed),
        )
        ended_after_confirmation = ended & previous_confirmed

        event_ids_i = torch.where(ended, previous_ids_i, ids_i)
        event_ids_j = torch.where(ended, previous_ids_j, ids_j)
        previous_levels_i = self._previous_levels[:, self.pair_i]
        event_levels = torch.where(ended, previous_levels_i, levels_i)

        self._candidate_streak = next_streak
        self._onset_steps = torch.where(
            eligible, reported_onset, torch.full_like(reported_onset, -1)
        )
        self._confirmed = next_confirmed
        self._previous_ids.copy_(torch.where(
            active, fruit_ids, torch.zeros_like(fruit_ids)
        ))
        self._previous_levels.copy_(torch.where(
            active, levels, torch.zeros_like(levels)
        ))
        return PairFailureUpdate(
            started=started,
            confirmed=confirmed,
            ended=ended,
            ended_after_confirmation=ended_after_confirmation,
            active_candidates=eligible,
            onset_steps=reported_onset,
            duration_drops=duration_drops,
            levels=event_levels,
            fruit_id_i=event_ids_i,
            fruit_id_j=event_ids_j,
            net_displacement_ratio_i=net_displacement_ratio[:, self.pair_i],
            net_displacement_ratio_j=net_displacement_ratio[:, self.pair_j],
            surface_gap_ratio=surface_gap_ratio,
        )


__all__ = [
    'PairFailureConfig',
    'PairFailureTracker',
    'PairFailureUpdate',
]
