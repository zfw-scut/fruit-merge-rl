"""把一次实际投放的物理事实压缩成定长辅助监督标签。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from daxigua.simulator import BatchStepResult

from .contact_geometry import (
    CONTACT_SPECIAL_CANDIDATE_COUNT,
    build_contact_candidate_geometry,
)
from .observations import TensorState


CONTACT_TYPE_COUNT = 4
GENERATED_FRUIT_COUNT = 3
LEVEL_CLASS_COUNT = 12
CONTACT_LEVEL_DELTA_CLASS_COUNT = 21


@dataclass(frozen=True, slots=True)
class ActionEffectTargets:
    """GPU Replay 中每个实际动作对应的一组定长效果标签。"""

    merge_happened: torch.Tensor
    merge_count: torch.Tensor
    q0_participated: torch.Tensor
    q0_lineage_depth: torch.Tensor
    q0_final_level: torch.Tensor
    contact_type_bits: torch.Tensor
    contact_primary_type: torch.Tensor
    contact_target: torch.Tensor
    contact_position: torch.Tensor
    contact_position_residual: torch.Tensor
    contact_level_delta: torch.Tensor
    contact_normal: torch.Tensor
    contact_age: torch.Tensor
    contact_normal_speed: torch.Tensor
    generation_exists: torch.Tensor
    generation_position: torch.Tensor
    generation_level: torch.Tensor
    score_delta: torch.Tensor
    fruit_count_delta: torch.Tensor
    final_exists: torch.Tensor
    final_state: torch.Tensor
    stable: torch.Tensor
    settle_timeout: torch.Tensor
    terminal: torch.Tensor
    settle_duration: torch.Tensor
    danger_delta: torch.Tensor
    over_danger_line: torch.Tensor

    def index_select(self, rows):
        return type(self)(**{
            name: getattr(self, name).index_select(0, rows)
            for name in self.__dataclass_fields__
        })

    @property
    def batch_size(self):
        return int(self.merge_happened.shape[0])


def _gather_slots(values, slots):
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, slots]


@torch.no_grad()
def build_action_effect_targets(
        current,
        next_state,
        step_result,
        *,
        board_width=560.0,
        board_height=1120.0,
        spawn_y=252.0,
        wall_width=20.0,
        velocity_scale=None,
        gravity_y=1800.0,
        max_physics_frames=600,
        current_fruit_count=None,
        current_danger_progress=None):
    """为本批实际执行动作生成标签；前三个新水果按全局事件顺序取。"""

    if not isinstance(current, TensorState) or not isinstance(
            next_state, TensorState):
        raise TypeError('current and next_state must be TensorState')
    if not isinstance(step_result, BatchStepResult):
        raise TypeError('step_result must be BatchStepResult')
    effects = step_result.physics.action_effects
    if effects is None:
        raise RuntimeError('simulator action-effect tracking is disabled')
    if current.batch_size != next_state.batch_size:
        raise ValueError('state batches must have equal size')

    batch = current.batch_size
    device = current.device
    dtype = current.positions.dtype
    rows = torch.arange(batch, dtype=torch.int64, device=device)
    effects = effects.index_select(rows)
    velocity_scale = float(velocity_scale or math.sqrt(
        2.0 * float(gravity_y) * float(board_height)
    ))
    events = step_result.physics.merge_events
    event_count = events.count.index_select(0, rows)
    event_source_levels = events.source_levels.index_select(0, rows)
    event_new_levels = events.new_levels.index_select(0, rows)
    event_positions = events.positions.index_select(0, rows)
    event_slots = torch.arange(
        event_source_levels.shape[1], device=device
    ).unsqueeze(0)
    event_valid = event_slots < event_count.unsqueeze(1)
    generated = event_valid & (event_new_levels > 0)
    generation_rank = generated.to(torch.int64).cumsum(dim=1) - 1
    generation_exists = torch.zeros(
        (batch, GENERATED_FRUIT_COUNT), dtype=torch.bool, device=device
    )
    generation_position = torch.zeros(
        (batch, GENERATED_FRUIT_COUNT, 2), dtype=dtype, device=device
    )
    generation_level = torch.zeros(
        (batch, GENERATED_FRUIT_COUNT), dtype=torch.int64, device=device
    )
    for rank in range(GENERATED_FRUIT_COUNT):
        selected = generated & generation_rank.eq(rank)
        generation_exists[:, rank] = selected.any(dim=1)
        selected_float = selected.to(dtype)
        generation_position[:, rank] = (
            event_positions * selected_float.unsqueeze(-1)
        ).sum(dim=1)
        generation_level[:, rank] = (
            event_new_levels * selected.to(torch.int64)
        ).sum(dim=1)

    generation_position[..., 0] = (
        generation_position[..., 0] / float(board_width) * 2.0 - 1.0
    )
    generation_position[..., 1] = (
        generation_position[..., 1] / float(board_height) * 2.0 - 1.0
    )
    generation_position *= generation_exists.unsqueeze(-1).to(dtype)

    contact_valid = effects.first_contact_age_frames >= 0
    contact_type_bits = torch.stack(tuple(
        effects.first_contact_type_mask.bitwise_and(1 << bit).ne(0)
        for bit in range(CONTACT_TYPE_COUNT)
    ), dim=1)
    contact_position = effects.first_contact_position.clone()
    contact_position[:, 0] = (
        contact_position[:, 0] / float(board_width) * 2.0 - 1.0
    )
    contact_position[:, 1] = (
        contact_position[:, 1] / float(board_height) * 2.0 - 1.0
    )
    contact_position *= contact_valid.unsqueeze(1).to(dtype)
    target_slot = effects.first_contact_target_slot.clamp(
        0, current.positions.shape[1] - 1
    )
    fruit_target_valid = (
        effects.first_contact_target_slot >= 0
    ) & _gather_slots(current.active, target_slot)
    contact_target = torch.where(
        effects.first_contact_primary_type == 4,
        torch.where(
            fruit_target_valid,
            effects.first_contact_target_slot
            + CONTACT_SPECIAL_CANDIDATE_COUNT,
            torch.full_like(
                effects.first_contact_primary_type,
                CONTACT_SPECIAL_CANDIDATE_COUNT - 1,
            ),
        ),
        effects.first_contact_primary_type,
    ).clamp(
        0,
        current.positions.shape[1] + CONTACT_SPECIAL_CANDIDATE_COUNT - 1,
    )
    contact_geometry = build_contact_candidate_geometry(
        current,
        step_result.drop.drop_x.index_select(0, rows).unsqueeze(1),
        step_result.drop.physics_radius.index_select(0, rows).unsqueeze(1),
        board_width=board_width,
        board_height=board_height,
        spawn_y=spawn_y,
        wall_width=wall_width,
        velocity_scale=velocity_scale,
    )
    candidate_positions = contact_geometry.positions[:, 0]
    contact_prior = candidate_positions.gather(
        1,
        contact_target[:, None, None].expand(-1, 1, 2),
    ).squeeze(1)
    contact_position_residual = (
        contact_position - contact_prior
    ) * contact_valid.unsqueeze(1).to(dtype)
    contact_normal = effects.first_contact_normal * (
        contact_valid.unsqueeze(1).to(dtype)
    )
    contact_age = torch.where(
        contact_valid,
        effects.first_contact_age_frames.to(dtype)
        / max(float(current.physics_fps), 1.0),
        torch.zeros(batch, dtype=dtype, device=device),
    )
    contact_age = torch.log1p(contact_age) / math.log(21.0)
    contact_normal_speed = torch.tanh(
        effects.first_contact_normal_speed / velocity_scale
    ) * contact_valid.to(dtype)
    contact_level_delta = (
        effects.first_contact_level_delta.clamp(-10, 10) + 10
    )

    final_slots = effects.q0_final_slot.clamp(
        0, next_state.positions.shape[1] - 1
    )
    final_exists = effects.q0_final_slot.ge(0) & _gather_slots(
        next_state.active, final_slots
    )
    final_position = _gather_slots(next_state.positions, final_slots)
    final_velocity = _gather_slots(next_state.velocities, final_slots)
    final_angular = _gather_slots(
        next_state.angular_velocities, final_slots
    )
    final_state = torch.stack((
        final_position[:, 0] / float(board_width) * 2.0 - 1.0,
        final_position[:, 1] / float(board_height) * 2.0 - 1.0,
        torch.tanh(final_velocity[:, 0] / velocity_scale),
        torch.tanh(final_velocity[:, 1] / velocity_scale),
        torch.tanh(final_angular / 10.0),
    ), dim=1)
    final_state *= final_exists.unsqueeze(1).to(dtype)

    current_count = (
        current.active.sum(dim=1)
        if current_fruit_count is None
        else current_fruit_count
    )
    next_count = next_state.active.sum(dim=1)
    score_delta = step_result.physics.score_delta.index_select(0, rows).to(dtype)
    score_delta = torch.log1p(score_delta.clamp_min(0.0)) / math.log(67.0)
    settle_timeout = step_result.physics.settle_timeout
    if settle_timeout is None:
        settle_timeout = torch.zeros(
            batch, dtype=torch.bool, device=device
        )
    else:
        settle_timeout = settle_timeout.index_select(0, rows)
    frames = step_result.physics.frames_simulated.index_select(0, rows).to(dtype)

    return ActionEffectTargets(
        merge_happened=event_count.gt(0),
        merge_count=event_count.clamp(0, 8),
        q0_participated=effects.q0_participated,
        q0_lineage_depth=effects.q0_lineage_depth.clamp(0, 7),
        q0_final_level=effects.q0_final_level.clamp(0, 11),
        contact_type_bits=contact_type_bits,
        contact_primary_type=effects.first_contact_primary_type.clamp(0, 4),
        contact_target=contact_target,
        contact_position=contact_position,
        contact_position_residual=contact_position_residual,
        contact_level_delta=contact_level_delta,
        contact_normal=contact_normal,
        contact_age=contact_age,
        contact_normal_speed=contact_normal_speed,
        generation_exists=generation_exists,
        generation_position=generation_position,
        generation_level=generation_level.clamp(0, 11),
        score_delta=score_delta,
        fruit_count_delta=(next_count - current_count).to(dtype).clamp(-8, 8) / 8.0,
        final_exists=final_exists,
        final_state=final_state,
        stable=step_result.physics.stable.index_select(0, rows),
        settle_timeout=settle_timeout,
        terminal=(
            step_result.physics.done.index_select(0, rows)
            | step_result.physics.truncated.index_select(0, rows)
        ),
        settle_duration=(frames / max(float(max_physics_frames), 1.0)).clamp(0, 1),
        danger_delta=(
            next_state.danger_progress - (
                current.danger_progress
                if current_danger_progress is None
                else current_danger_progress
            )
        ).to(dtype).clamp(-1, 1),
        over_danger_line=next_state.over_danger_line,
    )
