"""首次接触候选的共享几何先验。

候选编号保持定长：0=未接触，1=地面，2=左墙，3=右墙，4=动态水果，
5+slot=投放前场内水果。
模型和监督标签共用本模块，避免候选坐标与残差基准在两条路径中漂移。
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from .observations import TensorState


CONTACT_SPECIAL_CANDIDATE_COUNT = 5
CONTACT_GEOMETRY_FEATURE_DIM = 10


class ContactCandidateGeometry(NamedTuple):
    positions: torch.Tensor
    fruit_features: torch.Tensor
    valid: torch.Tensor


def build_contact_candidate_geometry(
        state,
        drop_x,
        drop_radius,
        *,
        board_width=560.0,
        board_height=1120.0,
        spawn_y=252.0,
        wall_width=20.0,
        velocity_scale=2007.9840636817814):
    """构造全部动作或实际动作的候选位置、几何特征和有效掩码。

    ``drop_x`` 与 ``drop_radius`` 使用 ``[B, A]``；实际动作标签可令
    ``A=1``。位置统一归一化到模型使用的 ``[-1, 1]`` 坐标系。
    """

    if not isinstance(state, TensorState):
        raise TypeError('state must be TensorState')
    if drop_x.ndim != 2 or drop_x.shape[0] != state.batch_size:
        raise ValueError('drop_x must have shape [B, A]')
    if drop_radius.ndim == 1:
        drop_radius = drop_radius.unsqueeze(1).expand_as(drop_x)
    if drop_radius.shape != drop_x.shape:
        raise ValueError('drop_radius must have shape [B] or [B, A]')

    dtype = state.positions.dtype
    device = state.device
    batch, actions = drop_x.shape
    fruits = state.positions.shape[1]
    fruit_x = state.positions[..., 0].unsqueeze(1)
    fruit_y = state.positions[..., 1].unsqueeze(1)
    fruit_radius = state.physics_radii.unsqueeze(1)
    dx = drop_x.unsqueeze(-1) - fruit_x
    radius_sum = drop_radius.unsqueeze(-1) + fruit_radius
    safe_radius_sum = radius_sum.clamp_min(1.0)
    clipped_dx = dx.clamp(-safe_radius_sum, safe_radius_sum)
    root = (
        radius_sum.square() - clipped_dx.square()
    ).clamp_min(0.0).sqrt()
    intersects = dx.abs() <= radius_sum

    q0_center_x = drop_x.unsqueeze(-1).expand(-1, -1, fruits)
    q0_center_y = fruit_y - root
    normal_x = clipped_dx / safe_radius_sum
    normal_y = -root / safe_radius_sum
    contact_x = q0_center_x - normal_x * drop_radius.unsqueeze(-1)
    contact_y = q0_center_y - normal_y * drop_radius.unsqueeze(-1)
    fruit_positions = torch.stack((
        contact_x / float(board_width) * 2.0 - 1.0,
        contact_y / float(board_height) * 2.0 - 1.0,
    ), dim=-1)

    normalized_drop_x = drop_x / float(board_width) * 2.0 - 1.0
    normalized_spawn_y = torch.full_like(
        drop_x, float(spawn_y) / float(board_height) * 2.0 - 1.0
    )
    normalized_floor_y = torch.full_like(
        drop_x,
        float(board_height - wall_width) / float(board_height) * 2.0 - 1.0,
    )
    normalized_left_x = torch.full_like(
        drop_x, float(wall_width) / float(board_width) * 2.0 - 1.0
    )
    normalized_right_x = torch.full_like(
        drop_x,
        float(board_width - wall_width) / float(board_width) * 2.0 - 1.0,
    )
    special_positions = torch.stack((
        torch.stack((normalized_drop_x, normalized_spawn_y), dim=-1),
        torch.stack((normalized_drop_x, normalized_floor_y), dim=-1),
        torch.stack((normalized_left_x, normalized_spawn_y), dim=-1),
        torch.stack((normalized_right_x, normalized_spawn_y), dim=-1),
        torch.stack((normalized_drop_x, normalized_spawn_y), dim=-1),
    ), dim=2)
    positions = torch.cat((special_positions, fruit_positions), dim=2)

    q0_level = state.fruit_queue[:, 0].to(torch.long)
    level_delta = (
        state.levels.unsqueeze(1) - q0_level[:, None, None]
    ).to(dtype)
    age_seconds = state.age_frames.to(dtype).unsqueeze(1) / max(
        float(state.physics_fps), 1.0
    )
    fruit_features = torch.stack((
        dx / float(board_width),
        dx.abs() / safe_radius_sum,
        fruit_y.expand(-1, actions, -1) / float(board_height) * 2.0 - 1.0,
        contact_y / float(board_height) * 2.0 - 1.0,
        intersects.to(dtype),
        level_delta.expand(-1, actions, -1) / 10.0,
        torch.tanh(
            state.velocities[..., 0].unsqueeze(1).expand(-1, actions, -1)
            / float(velocity_scale)
        ),
        torch.tanh(
            state.velocities[..., 1].unsqueeze(1).expand(-1, actions, -1)
            / float(velocity_scale)
        ),
        fruit_radius.expand(-1, actions, -1) / float(board_width),
        torch.log1p(age_seconds.expand(-1, actions, -1)) / 4.0,
    ), dim=-1)

    special_valid = torch.ones(
        (batch, actions, CONTACT_SPECIAL_CANDIDATE_COUNT),
        dtype=torch.bool,
        device=device,
    )
    fruit_valid = state.active.unsqueeze(1).expand(-1, actions, -1)
    valid = torch.cat((special_valid, fruit_valid), dim=2)
    fruit_features = fruit_features * fruit_valid.unsqueeze(-1).to(dtype)
    return ContactCandidateGeometry(positions, fruit_features, valid)


__all__ = [
    'CONTACT_GEOMETRY_FEATURE_DIM',
    'CONTACT_SPECIAL_CANDIDATE_COUNT',
    'ContactCandidateGeometry',
    'build_contact_candidate_geometry',
]
