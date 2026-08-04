"""基于 PyTorch Tensor 的多环境并行物理模拟器。"""

from __future__ import annotations

import math
from numbers import Integral

import torch

from daxigua.core import (
    ActionCandidate,
    BoardGeometry,
    DropResult,
    FruitState,
    GameState,
    PhysicsResult,
    dropped_fruit_physics_radius,
    fruit_mass,
    fruit_radius,
    merge_score,
    merged_fruit_physics_radius,
)

from .config import SimulatorConfig
from .types import (
    BatchDropResult,
    BatchMergeEvents,
    BatchObservation,
    BatchPhysicsResult,
    BatchSimulationTrace,
    BatchStepResult,
)


def _cross_2d(left, right):
    """计算两个二维向量的标量叉积。"""

    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _angular_cross(angular_velocity, radius_vector):
    """计算二维角速度在接触半径上产生的线速度。"""

    return torch.stack(
        (
            -angular_velocity * radius_vector[..., 1],
            angular_velocity * radius_vector[..., 0],
        ),
        dim=-1,
    )


class TensorVectorSimulator:
    """在同一设备上并行运行多个合成大西瓜环境。

    ``device='cuda'`` 是正式高吞吐路径；``device='cpu'`` 运行完全相同的
    Tensor 内核，便于无 GPU 的契约测试和调试。
    """

    _RNG_MULTIPLIER = 1103515245
    _RNG_INCREMENT = 12345
    _RNG_MASK = 0x7FFFFFFF
    _SEED_STRIDE = 747796405

    def __init__(
            self,
            num_envs,
            *,
            config=None,
            device='cuda'):
        if isinstance(num_envs, bool) or not isinstance(num_envs, Integral):
            raise TypeError('num_envs must be an integer')
        if num_envs <= 0:
            raise ValueError('num_envs must be positive')

        self.num_envs = int(num_envs)
        self.config = config or SimulatorConfig()
        if not isinstance(self.config, SimulatorConfig):
            raise TypeError('config must be SimulatorConfig')

        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available in the current PyTorch runtime')
        self.float_dtype = torch.float32

        batch = self.num_envs
        capacity = self.config.max_fruits
        queue_length = self.config.queue_length

        self.positions = torch.zeros(
            (batch, capacity, 2),
            dtype=self.float_dtype,
            device=self.device,
        )
        self.velocities = torch.zeros_like(self.positions)
        self._frame_start_positions = torch.empty_like(self.positions)
        self.angles = torch.zeros(
            (batch, capacity), dtype=self.float_dtype, device=self.device
        )
        self.angular_velocities = torch.zeros_like(self.angles)
        self.levels = torch.zeros(
            (batch, capacity), dtype=torch.int64, device=self.device
        )
        self.physics_radii = torch.zeros_like(
            self.angles, dtype=self.float_dtype
        )
        self.masses = torch.zeros_like(self.physics_radii)
        self.inverse_masses = torch.zeros_like(self.physics_radii)
        self.inverse_inertias = torch.zeros_like(self.physics_radii)
        self.fruit_ids = torch.zeros_like(self.levels)
        self.age_frames = torch.zeros_like(self.levels)
        self.active = torch.zeros(
            (batch, capacity), dtype=torch.bool, device=self.device
        )

        self.fruit_queue = torch.zeros(
            (batch, queue_length), dtype=torch.int64, device=self.device
        )
        self.score = torch.zeros(batch, dtype=torch.int64, device=self.device)
        self.last_score = torch.zeros_like(self.score)
        self.step_count = torch.zeros_like(self.score)
        self.physics_frame = torch.zeros_like(self.score)
        self.fail_frames = torch.zeros_like(self.score)
        self.next_fruit_id = torch.ones_like(self.score)
        self.rng_state = torch.zeros_like(self.score)
        self.episode_count = torch.full_like(self.score, -1)
        self.terminated = torch.zeros(
            batch, dtype=torch.bool, device=self.device
        )
        self.needs_reset = torch.ones_like(self.terminated)

        self._last_drop_level = torch.zeros_like(self.score)
        self._last_drop_x = torch.zeros(
            batch, dtype=self.float_dtype, device=self.device
        )
        self._last_drop_id = torch.zeros_like(self.score)
        self._last_queue_before = torch.zeros_like(self.fruit_queue)
        self._last_queue_after = torch.zeros_like(self.fruit_queue)

        self._event_count = torch.zeros_like(self.score)
        self._event_source_levels = torch.zeros(
            (batch, capacity), dtype=torch.int64, device=self.device
        )
        self._event_new_levels = torch.zeros_like(
            self._event_source_levels
        )
        self._event_positions = torch.zeros(
            (batch, capacity, 2),
            dtype=self.float_dtype,
            device=self.device,
        )
        self._event_score_deltas = torch.zeros_like(
            self._event_source_levels
        )
        self._event_source_ids = torch.zeros(
            (batch, capacity, 2), dtype=torch.int64, device=self.device
        )
        self._event_new_fruit_ids = torch.zeros_like(
            self._event_source_levels
        )

        pair_indices = torch.triu_indices(
            capacity,
            capacity,
            offset=1,
            device=self.device,
        )
        self._pair_i = pair_indices[0]
        self._pair_j = pair_indices[1]
        self._pair_count = int(self._pair_i.shape[0])
        self._pair_ids = torch.arange(
            self._pair_count, dtype=torch.int64, device=self.device
        ).view(1, -1)
        self._env_indices = torch.arange(
            batch, dtype=torch.int64, device=self.device
        )
        self._all_enabled = torch.ones(
            batch, dtype=torch.bool, device=self.device
        )

        self._display_radii = self._rule_table(fruit_radius)
        self._dropped_radii = self._rule_table(
            dropped_fruit_physics_radius
        )
        self._merged_radii = self._rule_table(
            merged_fruit_physics_radius
        )
        self._mass_table = self._rule_table(fruit_mass)
        self._merge_scores = self._rule_table(
            merge_score, dtype=torch.int64
        )

        # 正常训练热路径不分配追踪缓冲；禁用追踪时把这三个标量作为
        # CUDA 扩展的不可访问占位指针传入。
        self._trace_dummy_float = torch.zeros(
            1, dtype=self.float_dtype, device=self.device
        )
        self._trace_dummy_int = torch.zeros(
            1, dtype=torch.int64, device=self.device
        )
        self._trace_dummy_bool = torch.zeros(
            1, dtype=torch.bool, device=self.device
        )

        self._last_batch_result = None
        self.reset(seeds=0)

    def _rule_table(self, function, dtype=None):
        dtype = dtype or self.float_dtype
        values = [0]
        for level in range(1, 12):
            values.append(function(level))
        return torch.tensor(values, dtype=dtype, device=self.device)

    def _normalize_mask(self, mask):
        if mask is None:
            return torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        result = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if result.shape != (self.num_envs,):
            raise ValueError(
                f'mask must have shape ({self.num_envs},)'
            )
        return result

    def _reset_rows(self, tensor, mask, value=0):
        tensor[mask] = value

    def _seed_values(self, mask, seeds):
        env_ids = self._env_indices[mask]
        count = int(env_ids.numel())
        if seeds is None:
            values = (
                env_ids
                + 1
                + self.episode_count[mask] * self._SEED_STRIDE
            )
        elif isinstance(seeds, Integral) and not isinstance(seeds, bool):
            values = int(seeds) + env_ids * self._SEED_STRIDE
        else:
            values = torch.as_tensor(
                seeds, dtype=torch.int64, device=self.device
            ).flatten()
            if values.numel() == self.num_envs:
                values = values[mask]
            elif values.numel() != count:
                raise ValueError(
                    'seeds must be scalar, one per environment, or one per reset row'
                )
        return values & self._RNG_MASK

    def _draw_spawn_levels(self, mask):
        next_state = (
            self.rng_state * self._RNG_MULTIPLIER + self._RNG_INCREMENT
        ) & self._RNG_MASK
        self.rng_state = torch.where(mask, next_state, self.rng_state)
        return next_state.remainder(5) + 1

    @torch.no_grad()
    def reset(self, mask=None, *, seeds=None, fruit_queue=None):
        """全量或按 mask 重置环境，并返回批量状态视图。"""

        mask = self._normalize_mask(mask)
        if not bool(mask.any().item()):
            return self.observe()

        self.episode_count[mask] += 1
        for tensor in (
                self.positions,
                self.velocities,
                self.angles,
                self.angular_velocities,
                self.levels,
                self.physics_radii,
                self.masses,
                self.inverse_masses,
                self.inverse_inertias,
                self.fruit_ids,
                self.age_frames,
                self.active,
                self.score,
                self.last_score,
                self.step_count,
                self.physics_frame,
                self.fail_frames,
                self._last_drop_level,
                self._last_drop_x,
                self._last_drop_id,
                self._last_queue_before,
                self._last_queue_after):
            self._reset_rows(tensor, mask)
        self.next_fruit_id[mask] = 1
        self.terminated[mask] = False
        self.needs_reset[mask] = False
        self.rng_state[mask] = self._seed_values(mask, seeds)

        if fruit_queue is None:
            for offset in range(self.config.queue_length):
                levels = self._draw_spawn_levels(mask)
                self.fruit_queue[mask, offset] = levels[mask]
        else:
            queue = torch.as_tensor(
                fruit_queue, dtype=torch.int64, device=self.device
            )
            reset_count = int(mask.sum().item())
            if queue.ndim == 1:
                if queue.shape != (self.config.queue_length,):
                    raise ValueError('fruit_queue has an invalid shape')
                queue = queue.view(1, -1).expand(reset_count, -1)
            elif queue.shape == (self.num_envs, self.config.queue_length):
                queue = queue[mask]
            elif queue.shape != (reset_count, self.config.queue_length):
                raise ValueError('fruit_queue has an invalid shape')
            if bool(((queue < 1) | (queue > 5)).any().item()):
                raise ValueError('fruit_queue levels must be in [1, 5]')
            self.fruit_queue[mask] = queue

        self._clear_event_buffers()
        self._last_batch_result = None
        return self.observe()

    def _clear_event_buffers(self):
        self._event_count.zero_()
        self._event_source_levels.zero_()
        self._event_new_levels.zero_()
        self._event_positions.zero_()
        self._event_score_deltas.zero_()
        self._event_source_ids.zero_()
        self._event_new_fruit_ids.zero_()

    def action_positions(self):
        """返回每个环境当前水果的全部投放位置。"""

        current_level = self.fruit_queue[:, 0]
        radius = self._display_radii[current_level]
        left = self.config.wall_width + radius + 2.0
        right = self.config.board_width - self.config.wall_width - radius - 2.0
        normalized = torch.linspace(
            0.0,
            1.0,
            self.config.action_count,
            dtype=self.float_dtype,
            device=self.device,
        )
        return left[:, None] + (right - left)[:, None] * normalized[None, :]

    def _validate_actions(self, actions, enabled_mask=None):
        actions = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        )
        if actions.shape != (self.num_envs,):
            raise ValueError(
                f'actions must have shape ({self.num_envs},)'
            )
        if bool(
                ((actions < 0) | (actions >= self.config.action_count))
                .any()
                .item()):
            raise IndexError('action index out of range')
        blocked = (
            self.needs_reset
            if enabled_mask is None
            else self.needs_reset & enabled_mask
        )
        if bool(blocked.any().item()):
            raise RuntimeError('all finished environments must be reset before step')
        return actions

    def _set_mass_properties(self, env_ids, slots, levels, radii):
        mass = self._mass_table[levels]
        self.masses[env_ids, slots] = mass
        self.inverse_masses[env_ids, slots] = mass.reciprocal()
        inertia = 0.5 * mass * radii.square()
        self.inverse_inertias[env_ids, slots] = inertia.reciprocal()

    def _drop(self, actions):
        inactive = ~self.active
        if bool((~inactive.any(dim=1)).any().item()):
            raise RuntimeError(
                'max_fruits capacity exhausted; increase SimulatorConfig.max_fruits'
            )

        slots = inactive.to(torch.int8).argmax(dim=1)
        env_ids = self._env_indices
        level = self.fruit_queue[:, 0]
        positions = self.action_positions()
        drop_x = positions.gather(1, actions[:, None]).squeeze(1)
        radius = self._dropped_radii[level]
        fruit_id = self.next_fruit_id.clone()

        self._last_queue_before.copy_(self.fruit_queue)
        self.positions[env_ids, slots, 0] = drop_x
        self.positions[env_ids, slots, 1] = float(self.config.spawn_y)
        self.velocities[env_ids, slots] = 0.0
        self.velocities[env_ids, slots, 1] = 80.0
        self.angles[env_ids, slots] = 0.0
        self.angular_velocities[env_ids, slots] = 0.0
        self.levels[env_ids, slots] = level
        self.physics_radii[env_ids, slots] = radius
        self.fruit_ids[env_ids, slots] = fruit_id
        self.age_frames[env_ids, slots] = 0
        self.active[env_ids, slots] = True
        self._set_mass_properties(env_ids, slots, level, radius)
        self.next_fruit_id += 1

        self.fruit_queue[:, :-1] = self.fruit_queue[:, 1:].clone()
        live_mask = torch.ones_like(self.needs_reset)
        self.fruit_queue[:, -1] = self._draw_spawn_levels(live_mask)
        self.step_count += 1

        self._last_drop_level.copy_(level)
        self._last_drop_x.copy_(drop_x)
        self._last_drop_id.copy_(fruit_id)
        self._last_queue_after.copy_(self.fruit_queue)

    def _drop_fast_forward_eligibility(self):
        """判断投放前各环境能否安全跳过新水果的无碰撞下落。"""

        if not self.config.drop_fast_forward:
            return torch.zeros_like(self.needs_reset)
        stable_linear = self.velocities.square().sum(dim=-1) <= (
            self.config.stable_velocity_epsilon ** 2
        )
        stable_angular = self.angular_velocities.abs() <= (
            self.config.stable_angular_velocity_epsilon
        )
        all_stable = ((stable_linear & stable_angular) | ~self.active).all(dim=1)
        # 投放后，原来的最新水果也会参与顶线失败计时。该情况下跳帧会
        # 漏掉 fail_frames 的逐帧演化，因此保守地退回完整模拟。
        clear_of_danger_line = ~(
            self.active
            & (self.positions[..., 1] < float(self.config.spawn_y))
        ).any(dim=1)
        return all_stable & clear_of_danger_line

    def _fast_forward_new_drop(self, eligible):
        """把新水果推进到首次接触前一帧，返回跳过的语义帧数。"""

        skipped = torch.zeros_like(self.score)
        if not self.config.drop_fast_forward or not bool(eligible.any().item()):
            return skipped

        new_mask = self.active & (
            self.fruit_ids == self._last_drop_id[:, None]
        )
        new_slots = new_mask.to(torch.int8).argmax(dim=1)
        env_ids = self._env_indices
        drop_x = self.positions[env_ids, new_slots, 0]
        drop_y = self.positions[env_ids, new_slots, 1]
        drop_radius = self.physics_radii[env_ids, new_slots]
        drop_velocity_y = self.velocities[env_ids, new_slots, 1]

        floor_contact_y = (
            float(self.config.board_height - self.config.wall_width)
            - drop_radius
        )
        dx = drop_x[:, None] - self.positions[..., 0]
        radius_sum = drop_radius[:, None] + self.physics_radii
        collision_mask = self.active & ~new_mask & (dx.abs() < radius_sum)
        vertical_offset = (
            radius_sum.square() - dx.square()
        ).clamp_min(0.0).sqrt()
        candidate_y = self.positions[..., 1] - vertical_offset
        collision_mask &= candidate_y > drop_y[:, None]
        candidate_y = torch.where(
            collision_mask,
            candidate_y,
            torch.full_like(candidate_y, float('inf')),
        )
        contact_y = torch.minimum(floor_contact_y, candidate_y.min(dim=1).values)

        dt = self.config.dt
        frame_damping = float(self.config.damping) ** dt
        advancing = eligible.clone()
        for _ in range(self.config.max_physics_frames):
            next_velocity_y = (
                drop_velocity_y + self.config.gravity_y * dt
            ) * frame_damping
            next_y = drop_y + next_velocity_y * dt
            can_skip = advancing & (next_y < contact_y)
            drop_velocity_y = torch.where(
                can_skip, next_velocity_y, drop_velocity_y
            )
            drop_y = torch.where(can_skip, next_y, drop_y)
            skipped += can_skip.to(torch.int64)
            advancing = can_skip
            if not bool(advancing.any().item()):
                break

        self.positions[env_ids, new_slots, 1] = drop_y
        self.velocities[env_ids, new_slots, 1] = drop_velocity_y
        self.age_frames += self.active.to(torch.int64) * skipped[:, None]
        self.physics_frame += skipped
        self.fail_frames = torch.where(
            skipped > 0, torch.zeros_like(self.fail_frames), self.fail_frames
        )
        return skipped

    def _integrate(self, running):
        active = self.active & running[:, None]
        active_float = active.to(self.float_dtype)
        active_vector = active_float.unsqueeze(-1)
        dt = self.config.dt
        damping = float(self.config.damping) ** dt

        self.velocities[..., 1] += (
            self.config.gravity_y * dt * active_float
        )
        self.velocities *= 1.0 + (damping - 1.0) * active_vector
        self.positions += self.velocities * (dt * active_vector)
        self.angles += self.angular_velocities * (dt * active_float)
        self.age_frames += active.to(torch.int64)

    def _apply_wall_contact(self, penetration, normal, running):
        active = self.active & running[:, None]
        contact = active & (penetration > 0.0)
        contact_float = contact.to(self.float_dtype)
        normal = torch.tensor(
            normal, dtype=self.float_dtype, device=self.device
        )

        self.positions += (
            normal.view(1, 1, 2)
            * penetration.clamp_min(0.0).unsqueeze(-1)
            * contact_float.unsqueeze(-1)
        )

        normal_velocity = (
            self.velocities * normal.view(1, 1, 2)
        ).sum(dim=-1)
        approaching = contact & (normal_velocity < 0.0)
        approaching_float = approaching.to(self.float_dtype)
        normal_impulse = (
            -(1.0 + self.config.fruit_elasticity)
            * normal_velocity
            / self.inverse_masses.clamp_min(1e-12)
        ) * approaching_float

        tangent = torch.stack((-normal[1], normal[0])).view(1, 1, 2)
        radius_vector = (
            -normal.view(1, 1, 2) * self.physics_radii.unsqueeze(-1)
        )
        contact_velocity = self.velocities + _angular_cross(
            self.angular_velocities, radius_vector
        )
        tangent_velocity = (contact_velocity * tangent).sum(dim=-1)
        radius_cross_tangent = _cross_2d(radius_vector, tangent)
        tangent_denominator = (
            self.inverse_masses
            + radius_cross_tangent.square() * self.inverse_inertias
        ).clamp_min(1e-12)
        tangent_impulse = -tangent_velocity / tangent_denominator
        maximum_friction = self.config.wall_friction * normal_impulse
        tangent_impulse = torch.maximum(
            -maximum_friction,
            torch.minimum(maximum_friction, tangent_impulse),
        ) * approaching_float

        impulse = (
            normal_impulse.unsqueeze(-1) * normal.view(1, 1, 2)
            + tangent_impulse.unsqueeze(-1) * tangent
        )
        self.velocities += impulse * self.inverse_masses.unsqueeze(-1)
        self.angular_velocities += (
            _cross_2d(radius_vector, impulse) * self.inverse_inertias
        )

    def _resolve_walls(self, running):
        radius = self.physics_radii
        left_boundary = self.config.wall_width + radius
        right_boundary = self.config.board_width - self.config.wall_width - radius
        floor_boundary = self.config.board_height - self.config.wall_width - radius

        self._apply_wall_contact(
            left_boundary - self.positions[..., 0],
            (1.0, 0.0),
            running,
        )
        self._apply_wall_contact(
            self.positions[..., 0] - right_boundary,
            (-1.0, 0.0),
            running,
        )
        self._apply_wall_contact(
            self.positions[..., 1] - floor_boundary,
            (0.0, -1.0),
            running,
        )

    def _pair_geometry(self, running, extra_distance=0.0):
        pair_i = self._pair_i
        pair_j = self._pair_j
        position_i = self.positions[:, pair_i]
        position_j = self.positions[:, pair_j]
        delta = position_j - position_i
        distance_squared = delta.square().sum(dim=-1)
        distance = distance_squared.clamp_min(1e-12).sqrt()
        normal = delta / distance.unsqueeze(-1)

        coincident = distance_squared < 1e-12
        fallback = torch.zeros_like(normal)
        fallback[..., 0] = 1.0
        normal = torch.where(coincident.unsqueeze(-1), fallback, normal)

        radius_sum = (
            self.physics_radii[:, pair_i]
            + self.physics_radii[:, pair_j]
        )
        pair_active = (
            self.active[:, pair_i]
            & self.active[:, pair_j]
            & running[:, None]
        )
        contact = pair_active & (distance <= radius_sum + extra_distance)
        return contact, normal, distance, radius_sum

    def _scatter_pair_vectors(self, values_i, values_j):
        batch = self.num_envs
        pair_i = self._pair_i.view(1, -1, 1).expand(batch, -1, 2)
        pair_j = self._pair_j.view(1, -1, 1).expand(batch, -1, 2)
        result = torch.zeros_like(self.positions)
        result.scatter_add_(1, pair_i, values_i)
        result.scatter_add_(1, pair_j, values_j)
        return result

    def _scatter_pair_scalars(self, values_i, values_j, template):
        batch = self.num_envs
        pair_i = self._pair_i.view(1, -1).expand(batch, -1)
        pair_j = self._pair_j.view(1, -1).expand(batch, -1)
        result = torch.zeros_like(template)
        result.scatter_add_(1, pair_i, values_i)
        result.scatter_add_(1, pair_j, values_j)
        return result

    def _resolve_fruit_contacts(self, running):
        contact, normal, distance, radius_sum = self._pair_geometry(running)
        contact_float = contact.to(self.float_dtype)
        pair_i = self._pair_i
        pair_j = self._pair_j

        inverse_mass_i = self.inverse_masses[:, pair_i]
        inverse_mass_j = self.inverse_masses[:, pair_j]
        inverse_mass_sum = (inverse_mass_i + inverse_mass_j).clamp_min(1e-12)
        penetration = (radius_sum - distance - self.config.contact_slop).clamp_min(0.0)
        correction_size = (
            self.config.position_correction
            * penetration
            / inverse_mass_sum
            * contact_float
        )
        correction = correction_size.unsqueeze(-1) * normal
        position_delta = self._scatter_pair_vectors(
            -correction * inverse_mass_i.unsqueeze(-1),
            correction * inverse_mass_j.unsqueeze(-1),
        )
        self.positions += position_delta

        radius_i = self.physics_radii[:, pair_i]
        radius_j = self.physics_radii[:, pair_j]
        radius_vector_i = normal * radius_i.unsqueeze(-1)
        radius_vector_j = -normal * radius_j.unsqueeze(-1)
        velocity_i = self.velocities[:, pair_i] + _angular_cross(
            self.angular_velocities[:, pair_i], radius_vector_i
        )
        velocity_j = self.velocities[:, pair_j] + _angular_cross(
            self.angular_velocities[:, pair_j], radius_vector_j
        )
        relative_velocity = velocity_j - velocity_i
        normal_velocity = (relative_velocity * normal).sum(dim=-1)
        approaching = contact & (normal_velocity < 0.0)
        approaching_float = approaching.to(self.float_dtype)
        normal_impulse = (
            -(1.0 + self.config.fruit_elasticity)
            * normal_velocity
            / inverse_mass_sum
            * approaching_float
        )

        tangent = torch.stack((-normal[..., 1], normal[..., 0]), dim=-1)
        tangent_velocity = (relative_velocity * tangent).sum(dim=-1)
        radius_cross_tangent_i = _cross_2d(radius_vector_i, tangent)
        radius_cross_tangent_j = _cross_2d(radius_vector_j, tangent)
        tangent_denominator = (
            inverse_mass_sum
            + radius_cross_tangent_i.square()
            * self.inverse_inertias[:, pair_i]
            + radius_cross_tangent_j.square()
            * self.inverse_inertias[:, pair_j]
        ).clamp_min(1e-12)
        tangent_impulse = -tangent_velocity / tangent_denominator
        maximum_friction = self.config.fruit_friction * normal_impulse
        tangent_impulse = torch.maximum(
            -maximum_friction,
            torch.minimum(maximum_friction, tangent_impulse),
        ) * approaching_float

        impulse = (
            normal_impulse.unsqueeze(-1) * normal
            + tangent_impulse.unsqueeze(-1) * tangent
        )
        velocity_delta = self._scatter_pair_vectors(
            -impulse * inverse_mass_i.unsqueeze(-1),
            impulse * inverse_mass_j.unsqueeze(-1),
        )
        self.velocities += velocity_delta

        angular_delta = self._scatter_pair_scalars(
            -_cross_2d(radius_vector_i, impulse)
            * self.inverse_inertias[:, pair_i],
            _cross_2d(radius_vector_j, impulse)
            * self.inverse_inertias[:, pair_j],
            self.angular_velocities,
        )
        self.angular_velocities += angular_delta

    def _deterministic_merge_pairs(self, running):
        contact, _, _, _ = self._pair_geometry(
            running, extra_distance=self.config.merge_tolerance
        )
        same_level = self.levels[:, self._pair_i] == self.levels[:, self._pair_j]
        candidates = contact & same_level & (self.levels[:, self._pair_i] > 0)

        index_i = self._pair_i.view(1, -1).expand(self.num_envs, -1)
        index_j = self._pair_j.view(1, -1).expand(self.num_envs, -1)
        sentinel = self._pair_count
        selected = torch.zeros_like(candidates)
        claimed = torch.zeros_like(self.active)
        remaining = candidates
        # 互为最小 pair 只能得到无冲突集，在完全接触图中未必极大。
        # 固定轮次继续处理未被占用的顶点，得到与 CUDA 顺序扫描
        # 一致的确定性贪心极大匹配，且不需要设备同步。
        for _ in range(self.config.max_fruits // 2):
            pair_values = torch.where(
                remaining,
                self._pair_ids.expand(self.num_envs, -1),
                sentinel,
            )
            best = torch.full(
                (self.num_envs, self.config.max_fruits),
                sentinel,
                dtype=torch.int64,
                device=self.device,
            )
            best.scatter_reduce_(
                1, index_i, pair_values, reduce='amin', include_self=True
            )
            best.scatter_reduce_(
                1, index_j, pair_values, reduce='amin', include_self=True
            )
            chosen = (
                remaining
                & (best.gather(1, index_i) == self._pair_ids)
                & (best.gather(1, index_j) == self._pair_ids)
            )
            selected |= chosen
            claimed |= self._pair_mask_to_slots(chosen, self._pair_i)
            claimed |= self._pair_mask_to_slots(chosen, self._pair_j)
            remaining = (
                candidates
                & ~claimed.gather(1, index_i)
                & ~claimed.gather(1, index_j)
            )
        return selected

    def _append_events(
            self,
            selected,
            source_level,
            target_level,
            midpoint,
            event_score,
            source_ids,
            new_fruit_id):
        selected_int = selected.to(torch.int64)
        rank = selected_int.cumsum(dim=1) - 1
        event_slot = (self._event_count[:, None] + rank).clamp(
            0, self.config.max_fruits - 1
        )
        selected_float = selected.to(self.float_dtype)

        def scatter_scalar(buffer, values):
            buffer.scatter_add_(
                1, event_slot, values * selected_int
            )

        scatter_scalar(self._event_source_levels, source_level)
        scatter_scalar(self._event_new_levels, target_level)
        scatter_scalar(self._event_score_deltas, event_score)
        scatter_scalar(self._event_new_fruit_ids, new_fruit_id)
        self._event_positions.scatter_add_(
            1,
            event_slot.unsqueeze(-1).expand(-1, -1, 2),
            midpoint * selected_float.unsqueeze(-1),
        )
        self._event_source_ids.scatter_add_(
            1,
            event_slot.unsqueeze(-1).expand(-1, -1, 2),
            source_ids * selected_int.unsqueeze(-1),
        )
        self._event_count += selected_int.sum(dim=1)

    def _pair_mask_to_slots(self, pair_mask, pair_slots):
        # CUDA scatter_reduce 不支持 Bool，用 int8 完成同样的 amax。
        result = torch.zeros_like(self.active, dtype=torch.int8)
        indices = pair_slots.view(1, -1).expand(self.num_envs, -1)
        result.scatter_reduce_(
            1,
            indices,
            pair_mask.to(torch.int8),
            reduce='amax',
            include_self=True,
        )
        return result.bool()

    def _scatter_selected_scalars(self, selected, pair_slots, values, dtype):
        result = torch.zeros(
            (self.num_envs, self.config.max_fruits),
            dtype=dtype,
            device=self.device,
        )
        indices = pair_slots.view(1, -1).expand(self.num_envs, -1)
        result.scatter_add_(1, indices, values * selected.to(values.dtype))
        return result

    def _scatter_selected_vectors(self, selected, pair_slots, values):
        result = torch.zeros_like(self.positions)
        indices = pair_slots.view(1, -1, 1).expand(
            self.num_envs, -1, 2
        )
        result.scatter_add_(
            1, indices, values * selected.to(values.dtype).unsqueeze(-1)
        )
        return result

    def _resolve_merges(self, running):
        selected = self._deterministic_merge_pairs(running)
        selected_int = selected.to(torch.int64)
        pair_i = self._pair_i
        pair_j = self._pair_j
        source_level = self.levels[:, pair_i]
        normal_merge = selected & (source_level < 11)
        watermelon_merge = selected & (source_level == 11)
        target_level = torch.where(
            normal_merge, source_level + 1, torch.zeros_like(source_level)
        )
        midpoint = 0.5 * (
            self.positions[:, pair_i] + self.positions[:, pair_j]
        )
        source_mass_i = self.masses[:, pair_i]
        source_mass_j = self.masses[:, pair_j]
        source_velocity_i = self.velocities[:, pair_i]
        source_velocity_j = self.velocities[:, pair_j]
        pair_target_radius = torch.where(
            normal_merge,
            self._merged_radii[target_level],
            torch.ones_like(source_mass_i),
        )
        pair_target_mass = torch.where(
            normal_merge,
            self._mass_table[target_level],
            torch.ones_like(source_mass_i),
        ).clamp_min(1e-12)
        pair_target_inertia = (
            0.5 * pair_target_mass * pair_target_radius.square()
        ).clamp_min(1e-12)
        pair_linear_momentum = (
            source_mass_i.unsqueeze(-1) * source_velocity_i
            + source_mass_j.unsqueeze(-1) * source_velocity_j
        )
        pair_target_velocity = (
            pair_linear_momentum / pair_target_mass.unsqueeze(-1)
        )
        radius_i = self.positions[:, pair_i] - midpoint
        radius_j = self.positions[:, pair_j] - midpoint
        momentum_i = source_mass_i.unsqueeze(-1) * source_velocity_i
        momentum_j = source_mass_j.unsqueeze(-1) * source_velocity_j
        orbital_angular_momentum = (
            radius_i[..., 0] * momentum_i[..., 1]
            - radius_i[..., 1] * momentum_i[..., 0]
            + radius_j[..., 0] * momentum_j[..., 1]
            - radius_j[..., 1] * momentum_j[..., 0]
        )
        source_inertia_i = self.inverse_inertias[
            :, pair_i
        ].clamp_min(1e-12).reciprocal()
        source_inertia_j = self.inverse_inertias[
            :, pair_j
        ].clamp_min(1e-12).reciprocal()
        pair_angular_momentum = (
            source_inertia_i * self.angular_velocities[:, pair_i]
            + source_inertia_j * self.angular_velocities[:, pair_j]
            + orbital_angular_momentum
        )
        pair_target_angular_velocity = (
            pair_angular_momentum / pair_target_inertia
        )
        source_ids = torch.stack(
            (self.fruit_ids[:, pair_i], self.fruit_ids[:, pair_j]),
            dim=-1,
        )

        normal_rank = normal_merge.to(torch.int64).cumsum(dim=1) - 1
        pair_new_id = torch.where(
            normal_merge,
            self.next_fruit_id[:, None] + normal_rank,
            torch.zeros_like(normal_rank),
        )
        normal_count = normal_merge.to(torch.int64).sum(dim=1)
        self.next_fruit_id += normal_count

        event_score = self._merge_scores[source_level]
        self._append_events(
            selected,
            source_level,
            target_level,
            midpoint,
            event_score,
            source_ids,
            pair_new_id,
        )

        frame_score = event_score * selected_int
        frame_score_total = frame_score.sum(dim=1)
        has_merge = selected.any(dim=1)
        last_pair = torch.where(
            selected,
            self._pair_ids.expand(self.num_envs, -1),
            torch.zeros_like(self._pair_ids).expand(self.num_envs, -1),
        ).amax(dim=1)
        last_delta = frame_score.gather(1, last_pair[:, None]).squeeze(1)
        self.last_score = torch.where(
            has_merge,
            self.score + frame_score_total - last_delta,
            self.last_score,
        )
        self.score += frame_score_total

        normal_target_slots = self._pair_mask_to_slots(normal_merge, pair_i)
        watermelon_i_slots = self._pair_mask_to_slots(
            watermelon_merge, pair_i
        )
        source_j_slots = self._pair_mask_to_slots(selected, pair_j)
        deactivate = watermelon_i_slots | source_j_slots

        slot_level = self._scatter_selected_scalars(
            normal_merge,
            pair_i,
            target_level,
            torch.int64,
        )
        slot_new_id = self._scatter_selected_scalars(
            normal_merge,
            pair_i,
            pair_new_id,
            torch.int64,
        )
        slot_position = self._scatter_selected_vectors(
            normal_merge, pair_i, midpoint
        )
        slot_velocity = self._scatter_selected_vectors(
            normal_merge, pair_i, pair_target_velocity
        )
        slot_angular_velocity = self._scatter_selected_scalars(
            normal_merge,
            pair_i,
            pair_target_angular_velocity,
            self.float_dtype,
        )

        self.active &= ~deactivate
        inactive_vector = deactivate.unsqueeze(-1)
        self.velocities = torch.where(
            inactive_vector, torch.zeros_like(self.velocities), self.velocities
        )
        self.angular_velocities = torch.where(
            deactivate,
            torch.zeros_like(self.angular_velocities),
            self.angular_velocities,
        )

        if normal_target_slots.dtype != torch.bool:
            normal_target_slots = normal_target_slots.bool()
        target_vector = normal_target_slots.unsqueeze(-1)
        self.positions = torch.where(
            target_vector, slot_position, self.positions
        )
        self.velocities = torch.where(
            target_vector, slot_velocity, self.velocities
        )
        self.angles = torch.where(
            normal_target_slots, torch.zeros_like(self.angles), self.angles
        )
        self.angular_velocities = torch.where(
            normal_target_slots,
            slot_angular_velocity,
            self.angular_velocities,
        )
        self.levels = torch.where(
            normal_target_slots, slot_level, self.levels
        )
        self.fruit_ids = torch.where(
            normal_target_slots, slot_new_id, self.fruit_ids
        )
        self.age_frames = torch.where(
            normal_target_slots, torch.zeros_like(self.age_frames), self.age_frames
        )
        target_radius = self._merged_radii[slot_level]
        target_mass = self._mass_table[slot_level]
        target_inverse_mass = torch.where(
            normal_target_slots,
            target_mass.clamp_min(1e-12).reciprocal(),
            self.inverse_masses,
        )
        target_inverse_inertia = (
            0.5 * target_mass * target_radius.square()
        ).clamp_min(1e-12).reciprocal()
        self.physics_radii = torch.where(
            normal_target_slots, target_radius, self.physics_radii
        )
        self.masses = torch.where(
            normal_target_slots, target_mass, self.masses
        )
        self.inverse_masses = torch.where(
            normal_target_slots, target_inverse_mass, self.inverse_masses
        )
        self.inverse_inertias = torch.where(
            normal_target_slots,
            target_inverse_inertia,
            self.inverse_inertias,
        )

        # 清理已消失槽位的离散数据，避免调试读数时混淆。
        clear_slots = deactivate & ~normal_target_slots
        self.levels = torch.where(clear_slots, torch.zeros_like(self.levels), self.levels)
        self.fruit_ids = torch.where(
            clear_slots, torch.zeros_like(self.fruit_ids), self.fruit_ids
        )
        self.physics_radii = torch.where(
            clear_slots,
            torch.zeros_like(self.physics_radii),
            self.physics_radii,
        )

    def _stable_environments(self):
        speed = self.velocities.square().sum(dim=-1).sqrt()
        fruit_stable = (
            (speed <= self.config.stable_velocity_epsilon)
            & (
                self.angular_velocities.abs()
                <= self.config.stable_angular_velocity_epsilon
            )
        ) | ~self.active
        return fruit_stable.all(dim=1)

    def _correct_kinematic_rest_velocity(
            self, running, frame_start_positions, quiet_frame_count):
        """让连续静止环境的线速度与真实逐帧位移保持一致。"""

        required_frames = self.config.kinematic_rest_frames
        if required_frames == 0:
            return
        displacement = self.positions - frame_start_positions
        already_resting = quiet_frame_count >= required_frames
        displacement_epsilon = torch.where(
            already_resting,
            torch.full_like(
                quiet_frame_count,
                self.config.stable_velocity_epsilon * self.config.dt,
                dtype=self.float_dtype,
            ),
            torch.full_like(
                quiet_frame_count,
                self.config.kinematic_rest_displacement_epsilon,
                dtype=self.float_dtype,
            ),
        )
        quiet_slots = (
            displacement.square().sum(dim=-1)
            <= displacement_epsilon.square()
        ) & self.active & running[:, None]
        quiet_frame_count.copy_(torch.where(
            quiet_slots,
            quiet_frame_count + 1,
            torch.zeros_like(quiet_frame_count),
        ))
        correct = (
            self.active
            & running[:, None]
            & (quiet_frame_count >= required_frames)
        )
        if bool(correct.any().item()):
            self.velocities[correct] = 0.0

    def _update_termination(self, running):
        newest_id = torch.where(
            self.active, self.fruit_ids, torch.zeros_like(self.fruit_ids)
        ).amax(dim=1)
        checked = self.active & (self.fruit_ids != newest_id[:, None])
        over_line = (
            checked
            & (
                torch.trunc(self.positions[..., 1]).to(torch.int64)
                < self.config.spawn_y
            )
        ).any(dim=1)
        self.fail_frames = torch.where(
            running & over_line,
            self.fail_frames + 1,
            torch.where(running, torch.zeros_like(self.fail_frames), self.fail_frames),
        )
        return running & (self.fail_frames > self.config.danger_frame_limit)

    def _merge_event_view(self):
        return BatchMergeEvents(
            count=self._event_count,
            source_levels=self._event_source_levels,
            new_levels=self._event_new_levels,
            positions=self._event_positions,
            score_deltas=self._event_score_deltas,
            source_ids=self._event_source_ids,
            new_fruit_ids=self._event_new_fruit_ids,
        )

    def _drop_result_view(self):
        return BatchDropResult(
            dropped_levels=self._last_drop_level,
            drop_x=self._last_drop_x,
            fruit_ids=self._last_drop_id,
            queue_before=self._last_queue_before,
            queue_after=self._last_queue_after,
        )

    def _prepare_cuda_trace(self, env_indices, actions, frame_stride):
        if isinstance(frame_stride, bool) or not isinstance(
                frame_stride, Integral):
            raise TypeError('frame_stride must be an integer')
        frame_stride = int(frame_stride)
        if frame_stride <= 0:
            raise ValueError('frame_stride must be positive')

        env_indices = torch.as_tensor(
            env_indices, dtype=torch.int64, device=self.device
        )
        if env_indices.ndim != 1 or env_indices.numel() == 0:
            raise ValueError('env_indices must be a non-empty 1-D sequence')
        if bool(
                ((env_indices < 0) | (env_indices >= self.num_envs))
                .any()
                .item()):
            raise IndexError('trace environment index out of range')
        if int(torch.unique(env_indices).numel()) != int(env_indices.numel()):
            raise ValueError('env_indices must not contain duplicates')

        trace_count = int(env_indices.numel())
        trace_capacity = (
            (self.config.max_physics_frames + frame_stride - 1)
            // frame_stride
            + 1
        )
        fruit_shape = (
            trace_count,
            trace_capacity,
            self.config.max_fruits,
        )
        frame_shape = (trace_count, trace_capacity)
        trace_rows = torch.full(
            (self.num_envs,), -1, dtype=torch.int64, device=self.device
        )
        trace_rows[env_indices] = torch.arange(
            trace_count, dtype=torch.int64, device=self.device
        )
        buffers = {
            'env_indices': env_indices.clone(),
            'actions': actions[env_indices].clone(),
            'record_counts': torch.zeros(
                trace_count, dtype=torch.int64, device=self.device
            ),
            'frame_numbers': torch.empty(
                frame_shape, dtype=torch.int64, device=self.device
            ),
            'positions': torch.empty(
                fruit_shape + (2,), dtype=self.float_dtype, device=self.device
            ),
            'velocities': torch.empty(
                fruit_shape + (2,), dtype=self.float_dtype, device=self.device
            ),
            'angles': torch.empty(
                fruit_shape, dtype=self.float_dtype, device=self.device
            ),
            'angular_velocities': torch.empty(
                fruit_shape, dtype=self.float_dtype, device=self.device
            ),
            'levels': torch.empty(
                fruit_shape, dtype=torch.int64, device=self.device
            ),
            'physics_radii': torch.empty(
                fruit_shape, dtype=self.float_dtype, device=self.device
            ),
            'fruit_ids': torch.empty(
                fruit_shape, dtype=torch.int64, device=self.device
            ),
            'active': torch.empty(
                fruit_shape, dtype=torch.bool, device=self.device
            ),
            'scores': torch.empty(
                frame_shape, dtype=torch.int64, device=self.device
            ),
            'merge_counts': torch.empty(
                frame_shape, dtype=torch.int64, device=self.device
            ),
            'trace_rows': trace_rows,
            'trace_capacity': trace_capacity,
            'frame_stride': frame_stride,
        }
        return buffers

    def _step_cuda_extension(
            self,
            actions,
            *,
            enabled_mask=None,
            trace_env_indices=None,
            trace_frame_stride=1):
        """使用单 Kernel 完成整个批量投放和物理稳定过程。"""

        from .cuda_backend import MAX_CUDA_FRUITS, load_cuda_extension

        if self.config.max_fruits > MAX_CUDA_FRUITS:
            raise ValueError(
                f'compiled CUDA backend supports max_fruits <= {MAX_CUDA_FRUITS}'
            )
        enabled_mask = (
            self._all_enabled if enabled_mask is None else enabled_mask
        )
        inactive = ~self.active
        if bool(((~inactive.any(dim=1)) & enabled_mask).any().item()):
            raise RuntimeError(
                'max_fruits capacity exhausted; increase SimulatorConfig.max_fruits'
            )

        extension = load_cuda_extension()
        score_before = self.score.clone()
        self._clear_event_buffers()
        frames_simulated = torch.zeros_like(self.score)
        fast_forwarded_frames = torch.zeros_like(self.score)
        stable_result = torch.zeros_like(self.terminated)
        done_result = torch.zeros_like(self.terminated)
        truncated_result = torch.zeros_like(self.terminated)
        trace = (
            None
            if trace_env_indices is None
            else self._prepare_cuda_trace(
                trace_env_indices, actions, trace_frame_stride
            )
        )
        if trace is None:
            trace_count = 0
            trace_capacity = 1
            trace_stride = 1
            trace_rows = self._trace_dummy_int
            trace_positions = self._trace_dummy_float
            trace_velocities = self._trace_dummy_float
            trace_angles = self._trace_dummy_float
            trace_angular_velocities = self._trace_dummy_float
            trace_levels = self._trace_dummy_int
            trace_physics_radii = self._trace_dummy_float
            trace_fruit_ids = self._trace_dummy_int
            trace_active = self._trace_dummy_bool
            trace_scores = self._trace_dummy_int
            trace_merge_counts = self._trace_dummy_int
            trace_frame_numbers = self._trace_dummy_int
            trace_record_counts = self._trace_dummy_int
        else:
            trace_count = int(trace['env_indices'].numel())
            trace_capacity = trace['trace_capacity']
            trace_stride = trace['frame_stride']
            trace_rows = trace['trace_rows']
            trace_positions = trace['positions']
            trace_velocities = trace['velocities']
            trace_angles = trace['angles']
            trace_angular_velocities = trace['angular_velocities']
            trace_levels = trace['levels']
            trace_physics_radii = trace['physics_radii']
            trace_fruit_ids = trace['fruit_ids']
            trace_active = trace['active']
            trace_scores = trace['scores']
            trace_merge_counts = trace['merge_counts']
            trace_frame_numbers = trace['frame_numbers']
            trace_record_counts = trace['record_counts']
        extension.vector_step(
            actions.contiguous(),
            enabled_mask.contiguous(),
            self.positions,
            self.velocities,
            self._frame_start_positions,
            self.angles,
            self.angular_velocities,
            self.levels,
            self.physics_radii,
            self.masses,
            self.inverse_masses,
            self.inverse_inertias,
            self.fruit_ids,
            self.age_frames,
            self.active,
            self.fruit_queue,
            self.score,
            self.last_score,
            self.step_count,
            self.physics_frame,
            self.fail_frames,
            self.next_fruit_id,
            self.rng_state,
            self.terminated,
            self.needs_reset,
            self._last_drop_level,
            self._last_drop_x,
            self._last_drop_id,
            self._last_queue_before,
            self._last_queue_after,
            self._event_count,
            self._event_source_levels,
            self._event_new_levels,
            self._event_positions,
            self._event_score_deltas,
            self._event_source_ids,
            self._event_new_fruit_ids,
            self._display_radii,
            self._dropped_radii,
            self._merged_radii,
            self._mass_table,
            self._merge_scores,
            frames_simulated,
            fast_forwarded_frames,
            stable_result,
            done_result,
            truncated_result,
            trace_rows,
            trace_positions,
            trace_velocities,
            trace_angles,
            trace_angular_velocities,
            trace_levels,
            trace_physics_radii,
            trace_fruit_ids,
            trace_active,
            trace_scores,
            trace_merge_counts,
            trace_frame_numbers,
            trace_record_counts,
            trace_count,
            trace_capacity,
            trace_stride,
            self.config.board_width,
            self.config.board_height,
            self.config.spawn_y,
            self.config.wall_width,
            self.config.action_count,
            self.config.max_fruits,
            self.config.queue_length,
            self.config.physics_fps,
            self.config.max_physics_frames,
            self.config.stable_frames,
            self.config.solver_iterations,
            self.config.drop_fast_forward,
            self.config.kinematic_rest_frames,
            self.config.kinematic_rest_displacement_epsilon,
            self.config.gravity_y,
            self.config.damping,
            self.config.fruit_elasticity,
            self.config.fruit_friction,
            self.config.wall_friction,
            self.config.stable_velocity_epsilon,
            self.config.stable_angular_velocity_epsilon,
            self.config.danger_frame_limit,
            self.config.contact_slop,
            self.config.position_correction,
            self.config.merge_tolerance,
            self.config.cuda_threads_per_block,
        )
        settle_timeout_result = (
            enabled_mask
            & (frames_simulated == self.config.max_physics_frames)
            & ~stable_result
            & ~done_result
            & ~truncated_result
        )
        physics = BatchPhysicsResult(
            frames_simulated=frames_simulated,
            stable=stable_result,
            done=done_result,
            truncated=truncated_result,
            settle_timeout=settle_timeout_result,
            fast_forwarded_frames=fast_forwarded_frames,
            score_delta=self.score - score_before,
            merge_events=self._merge_event_view(),
        )
        result = BatchStepResult(
            observation=self.observe(),
            drop=self._drop_result_view(),
            physics=physics,
        )
        self._last_batch_result = result
        if trace is None:
            return result
        simulation_trace = BatchSimulationTrace(
            env_indices=trace['env_indices'],
            actions=trace['actions'],
            record_counts=trace['record_counts'],
            frame_numbers=trace['frame_numbers'],
            positions=trace['positions'],
            velocities=trace['velocities'],
            angles=trace['angles'],
            angular_velocities=trace['angular_velocities'],
            levels=trace['levels'],
            physics_radii=trace['physics_radii'],
            fruit_ids=trace['fruit_ids'],
            active=trace['active'],
            scores=trace['scores'],
            merge_counts=trace['merge_counts'],
            stable=stable_result[trace['env_indices']].clone(),
            done=done_result[trace['env_indices']].clone(),
            truncated=truncated_result[trace['env_indices']].clone(),
            settle_timeout=(
                settle_timeout_result[trace['env_indices']].clone()
            ),
            score_deltas=(
                self.score - score_before
            )[trace['env_indices']].clone(),
            physics_fps=self.config.physics_fps,
            frame_stride=trace_stride,
        )
        return result, simulation_trace

    @torch.no_grad()
    def step(self, actions):
        """为所有环境各执行一次完整投放。"""

        actions = self._validate_actions(actions)
        if self.device.type == 'cuda' and self.config.use_cuda_extension:
            return self._step_cuda_extension(actions)
        score_before = self.score.clone()
        self._clear_event_buffers()
        fast_forward_eligible = self._drop_fast_forward_eligibility()
        self._drop(actions)

        fast_forwarded_frames = self._fast_forward_new_drop(
            fast_forward_eligible
        )
        frames_simulated = fast_forwarded_frames.clone()
        running = frames_simulated < self.config.max_physics_frames
        budget_exhausted = ~running
        stable_count = torch.zeros_like(self.score)
        kinematic_quiet_count = torch.zeros_like(self.levels)
        stable_result = torch.zeros_like(running)
        done_result = torch.zeros_like(running)

        for frame_index in range(self.config.max_physics_frames):
            frame_start_positions = self.positions.clone()
            self._integrate(running)
            for _ in range(self.config.solver_iterations):
                self._resolve_walls(running)
                self._resolve_fruit_contacts(running)
            self._resolve_walls(running)
            self._resolve_merges(running)
            self._correct_kinematic_rest_velocity(
                running,
                frame_start_positions,
                kinematic_quiet_count,
            )

            frames_simulated += running.to(torch.int64)
            self.physics_frame += running.to(torch.int64)
            newly_done = self._update_termination(running)
            done_result |= newly_done
            running &= ~newly_done

            stable_now = self._stable_environments() & running
            stable_count = torch.where(
                stable_now,
                stable_count + 1,
                torch.zeros_like(stable_count),
            )
            newly_stable = running & (
                stable_count >= self.config.stable_frames
            )
            stable_result |= newly_stable
            running &= ~newly_stable

            newly_exhausted = running & (
                frames_simulated >= self.config.max_physics_frames
            )
            budget_exhausted |= newly_exhausted
            running &= ~newly_exhausted

            should_sync = (
                (frame_index + 1) % self.config.sync_interval_frames == 0
                or frame_index + 1 == self.config.max_physics_frames
            )
            if should_sync and not bool(running.any().item()):
                break

        settle_timeout = budget_exhausted
        truncated = torch.zeros_like(running)
        self.terminated = done_result
        self.needs_reset = done_result | truncated
        physics = BatchPhysicsResult(
            frames_simulated=frames_simulated,
            stable=stable_result,
            done=done_result,
            truncated=truncated,
            settle_timeout=settle_timeout,
            fast_forwarded_frames=fast_forwarded_frames,
            score_delta=self.score - score_before,
            merge_events=self._merge_event_view(),
        )
        result = BatchStepResult(
            observation=self.observe(),
            drop=self._drop_result_view(),
            physics=physics,
        )
        self._last_batch_result = result
        return result

    @torch.no_grad()
    def step_with_trace(self, actions, env_indices, *, frame_stride=1):
        """执行一次 CUDA 投放，并记录指定环境的逐帧物理状态。

        该接口用于抽样审计和回放，不应放入正式训练热路径。未被抽样的环境
        仍正常并行步进，且不产生逐帧记录。
        """

        if self.device.type != 'cuda' or not self.config.use_cuda_extension:
            raise RuntimeError(
                'step_with_trace requires the compiled CUDA backend'
            )
        actions = self._validate_actions(actions)
        return self._step_cuda_extension(
            actions,
            trace_env_indices=env_indices,
            trace_frame_stride=frame_stride,
        )

    @torch.no_grad()
    def step_masked(self, actions, enabled_mask):
        """只让掩码为真的 CUDA 环境执行一次完整投放。

        该接口用于逐环境停止的评估和压力测试。禁用环境的状态、计数、
        结果和 RNG 均保持不变。
        """

        if self.device.type != 'cuda' or not self.config.use_cuda_extension:
            raise RuntimeError('step_masked requires the compiled CUDA backend')
        enabled_mask = self._normalize_mask(enabled_mask)
        actions = self._validate_actions(actions, enabled_mask)
        return self._step_cuda_extension(
            actions,
            enabled_mask=enabled_mask,
        )

    @torch.no_grad()
    def step_masked_with_trace(
            self,
            actions,
            enabled_mask,
            trace_env_indices,
            *,
            frame_stride=1):
        """掩码步进并逐帧记录其中指定的活动环境。"""

        if self.device.type != 'cuda' or not self.config.use_cuda_extension:
            raise RuntimeError(
                'step_masked_with_trace requires the compiled CUDA backend'
            )
        enabled_mask = self._normalize_mask(enabled_mask)
        trace_env_indices = torch.as_tensor(
            trace_env_indices, dtype=torch.int64, device=self.device
        )
        if trace_env_indices.ndim != 1 or trace_env_indices.numel() == 0:
            raise ValueError(
                'trace_env_indices must be a non-empty 1-D sequence'
            )
        if bool(
                ((trace_env_indices < 0)
                 | (trace_env_indices >= self.num_envs)).any().item()):
            raise IndexError('trace environment index out of range')
        if not bool(enabled_mask[trace_env_indices].all().item()):
            raise ValueError('every traced environment must be enabled')
        actions = self._validate_actions(actions, enabled_mask)
        return self._step_cuda_extension(
            actions,
            enabled_mask=enabled_mask,
            trace_env_indices=trace_env_indices,
            trace_frame_stride=frame_stride,
        )

    def observe(self):
        """返回当前批量 Tensor 状态的零拷贝视图。"""

        fruit_count = self.active.to(torch.int64).sum(dim=1)
        max_level = torch.where(
            self.active, self.levels, torch.zeros_like(self.levels)
        ).amax(dim=1)
        fruit_top = self.positions[..., 1] - self.physics_radii
        highest_top = torch.where(
            self.active,
            fruit_top,
            torch.full_like(fruit_top, float('inf')),
        ).amin(dim=1)
        max_height = torch.where(
            fruit_count > 0,
            self.config.board_height - highest_top,
            torch.zeros_like(highest_top),
        )
        playable_area = max(
            1.0,
            self.config.board_width
            * (self.config.board_height - self.config.spawn_y),
        )
        fruit_area = (
            math.pi
            * self.physics_radii.square()
            * self.active.to(self.float_dtype)
        ).sum(dim=1)
        empty_space_ratio = (1.0 - fruit_area / playable_area).clamp(0.0, 1.0)
        return BatchObservation(
            positions=self.positions,
            velocities=self.velocities,
            angles=self.angles,
            angular_velocities=self.angular_velocities,
            levels=self.levels,
            physics_radii=self.physics_radii,
            fruit_ids=self.fruit_ids,
            age_frames=self.age_frames,
            active=self.active,
            fruit_queue=self.fruit_queue,
            score=self.score,
            last_score=self.last_score,
            step_count=self.step_count,
            physics_frame=self.physics_frame,
            done=self.terminated,
            fruit_count=fruit_count,
            max_level=max_level,
            max_height=max_height,
            empty_space_ratio=empty_space_ratio,
        )

    def action_candidates(self, env_index):
        """返回单个环境的 Python 动作契约。"""

        positions = self.action_positions()[env_index].tolist()
        level = int(self.fruit_queue[env_index, 0].item())
        radius = float(fruit_radius(level))
        physics_radius = float(dropped_fruit_physics_radius(level))
        return tuple(
            ActionCandidate(
                action_index=index,
                drop_x=float(position),
                normalized_drop_x=index / (self.config.action_count - 1),
                current_level=level,
                current_radius=radius,
                current_physics_radius=physics_radius,
            )
            for index, position in enumerate(positions)
        )

    def state_at(self, env_index):
        """把单个环境转为现有 ``GameState`` 契约。"""

        if env_index < 0 or env_index >= self.num_envs:
            raise IndexError('env_index out of range')
        active_slots = torch.nonzero(
            self.active[env_index], as_tuple=False
        ).flatten()
        if active_slots.numel():
            ids = self.fruit_ids[env_index, active_slots]
            active_slots = active_slots[torch.argsort(ids)]
        fruits = []
        for slot in active_slots.tolist():
            level = int(self.levels[env_index, slot].item())
            radius = float(fruit_radius(level))
            physics_radius = float(
                self.physics_radii[env_index, slot].item()
            )
            x, y = (
                float(value)
                for value in self.positions[env_index, slot].tolist()
            )
            vx, vy = (
                float(value)
                for value in self.velocities[env_index, slot].tolist()
            )
            angular_velocity = float(
                self.angular_velocities[env_index, slot].item()
            )
            stable = (
                math.hypot(vx, vy)
                <= self.config.stable_velocity_epsilon
                and abs(angular_velocity)
                <= self.config.stable_angular_velocity_epsilon
            )
            fruits.append(
                FruitState(
                    fruit_id=int(self.fruit_ids[env_index, slot].item()),
                    level=level,
                    radius=radius,
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    angle=float(self.angles[env_index, slot].item()),
                    angular_velocity=angular_velocity,
                    age_frames=int(
                        self.age_frames[env_index, slot].item()
                    ),
                    stable=stable,
                    distance_to_left_wall=float(
                        x - (self.config.wall_width + physics_radius)
                    ),
                    distance_to_right_wall=float(
                        self.config.board_width
                        - self.config.wall_width
                        - physics_radius
                        - x
                    ),
                    distance_to_floor=float(
                        self.config.board_height
                        - self.config.wall_width
                        - physics_radius
                        - y
                    ),
                    distance_to_danger_line=float(
                        y - physics_radius - self.config.spawn_y
                    ),
                    physics_radius=physics_radius,
                )
            )
        observation = self.observe()
        return GameState(
            board_fruits=tuple(fruits),
            fruit_queue=tuple(
                int(value) for value in self.fruit_queue[env_index].tolist()
            ),
            score=int(self.score[env_index].item()),
            last_score=int(self.last_score[env_index].item()),
            step_count=int(self.step_count[env_index].item()),
            physics_frame=int(self.physics_frame[env_index].item()),
            done=bool(self.terminated[env_index].item()),
            geometry=BoardGeometry(
                width=self.config.board_width,
                height=self.config.board_height,
                spawn_y=self.config.spawn_y,
                wall_width=self.config.wall_width,
                floor_y=self.config.board_height - self.config.wall_width,
            ),
            max_height=float(observation.max_height[env_index].item()),
            fruit_count=len(fruits),
            max_level=int(observation.max_level[env_index].item()),
            empty_space_ratio=float(
                observation.empty_space_ratio[env_index].item()
            ),
        )

    def drop_result_at(self, env_index):
        """返回最近一次批量投放中的单环境结果。"""

        return DropResult(
            dropped_level=int(self._last_drop_level[env_index].item()),
            drop_x=float(self._last_drop_x[env_index].item()),
            fruit_id=int(self._last_drop_id[env_index].item()),
            queue_before=tuple(
                int(value)
                for value in self._last_queue_before[env_index].tolist()
            ),
            queue_after=tuple(
                int(value)
                for value in self._last_queue_after[env_index].tolist()
            ),
        )

    def physics_result_at(self, env_index):
        """返回最近一次批量步进中的单环境结果。"""

        if self._last_batch_result is None:
            raise RuntimeError('step must be called before reading PhysicsResult')
        physics = self._last_batch_result.physics
        return PhysicsResult(
            frames_simulated=int(physics.frames_simulated[env_index].item()),
            stable=bool(physics.stable[env_index].item()),
            done=bool(physics.done[env_index].item()),
            truncated=bool(physics.truncated[env_index].item()),
            score_delta=int(physics.score_delta[env_index].item()),
            merge_events=physics.merge_events.to_python(env_index),
            settle_timeout=bool(physics.settle_timeout[env_index].item()),
        )


class VectorEnv:
    """把批量物理内核与显式奖励计算器组合。"""

    def __init__(self, simulator, reward_computer):
        if not isinstance(simulator, TensorVectorSimulator):
            raise TypeError('simulator must be TensorVectorSimulator')
        if reward_computer is None or not callable(reward_computer):
            raise TypeError('reward_computer must be callable')
        self.simulator = simulator
        self.reward_computer = reward_computer

    def reset(self, mask=None, *, seeds=None, fruit_queue=None):
        observation = self.simulator.reset(
            mask, seeds=seeds, fruit_queue=fruit_queue
        )
        return observation, {'reward_defined': True}

    def step(self, actions):
        requires_previous = bool(
            getattr(self.reward_computer, 'requires_previous_state', True)
        )
        previous = (
            self.simulator.observe().clone() if requires_previous else None
        )
        result = self.simulator.step(actions)
        rewards = self.reward_computer(
            previous, result.observation, result.physics
        )
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.simulator.device
        )
        if rewards.shape != (self.simulator.num_envs,):
            raise ValueError('reward_computer must return shape [num_envs]')
        info = {
            'drop_result': result.drop,
            'physics': result.physics,
            'score_delta': result.physics.score_delta,
            'merge_events': result.physics.merge_events,
            'settle_timeout': result.physics.settle_timeout,
        }
        return (
            result.observation,
            rewards,
            result.physics.done,
            result.physics.truncated,
            info,
        )


class SingleEnvAdapter:
    """类 Gymnasium 的单环境 Python 兼容层。"""

    def __init__(self, vector_env):
        if not isinstance(vector_env, VectorEnv):
            raise TypeError('vector_env must be VectorEnv')
        if vector_env.simulator.num_envs != 1:
            raise ValueError('SingleEnvAdapter requires exactly one environment')
        self.vector_env = vector_env

    @property
    def simulator(self):
        return self.vector_env.simulator

    def reset(self, seed=None, fruit_queue=None):
        self.vector_env.reset(seeds=seed, fruit_queue=fruit_queue)
        return self.simulator.state_at(0), {
            'action_candidates': self.simulator.action_candidates(0)
        }

    def action_candidates(self):
        return self.simulator.action_candidates(0)

    def step(self, action_index):
        result = self.vector_env.step(
            torch.tensor(
                [action_index], dtype=torch.int64, device=self.simulator.device
            )
        )
        _observation, rewards, terminated, truncated, _batch_info = result
        info = {
            'drop_result': self.simulator.drop_result_at(0),
            'physics_result': self.simulator.physics_result_at(0),
            'score_delta': int(
                self.simulator._last_batch_result.physics.score_delta[0].item()
            ),
            'merge_events': self.simulator._merge_event_view().to_python(0),
            'frames_simulated': int(
                self.simulator._last_batch_result.physics.frames_simulated[0].item()
            ),
            'stable': bool(
                self.simulator._last_batch_result.physics.stable[0].item()
            ),
            'settle_timeout': bool(
                self.simulator._last_batch_result.physics
                .settle_timeout[0].item()
            ),
            'action_candidates': (
                ()
                if bool(terminated[0].item())
                else self.simulator.action_candidates(0)
            ),
        }
        return (
            self.simulator.state_at(0),
            float(rewards[0].item()),
            bool(terminated[0].item()),
            bool(truncated[0].item()),
            info,
        )
