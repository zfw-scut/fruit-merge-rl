"""定长张量实现的第一版物理 GNN-Dueling DQN。"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from daxigua.core import (
    MAX_FRUIT_LEVEL,
    dropped_fruit_physics_radius,
    fruit_mass,
    fruit_radius,
)

from .config import ModelConfig
from .contact_geometry import (
    CONTACT_GEOMETRY_FEATURE_DIM,
    CONTACT_SPECIAL_CANDIDATE_COUNT,
    build_contact_candidate_geometry,
)
from .observations import TensorState


class ActionEffectPredictions(NamedTuple):
    merge_logit: torch.Tensor
    merge_count_logits: torch.Tensor
    q0_participated_logit: torch.Tensor
    q0_lineage_depth_logits: torch.Tensor
    q0_final_level_logits: torch.Tensor
    contact_type_logits: torch.Tensor
    contact_primary_type_logits: torch.Tensor
    contact_position: torch.Tensor
    contact_level_delta_logits: torch.Tensor
    contact_normal: torch.Tensor
    contact_age: torch.Tensor
    contact_normal_speed: torch.Tensor
    generation_exists_logits: torch.Tensor
    generation_position: torch.Tensor
    generation_level_logits: torch.Tensor
    score_delta: torch.Tensor
    fruit_count_delta: torch.Tensor
    final_exists_logit: torch.Tensor
    final_state: torch.Tensor
    stable_logit: torch.Tensor
    settle_timeout_logit: torch.Tensor
    terminal_logit: torch.Tensor
    settle_duration: torch.Tensor
    danger_delta: torch.Tensor
    over_danger_line_logit: torch.Tensor
    contact_target_logits: torch.Tensor | None
    contact_position_residual: torch.Tensor | None


class ModelOutput(NamedTuple):
    q_values: torch.Tensor
    head_q_values: torch.Tensor
    action_effects: ActionEffectPredictions | None


def _mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _masked_max(values, mask, dim):
    fill = torch.finfo(values.dtype).min
    masked = values.masked_fill(~mask.unsqueeze(-1), fill)
    result = masked.amax(dim=dim)
    valid = mask.any(dim=dim, keepdim=False).unsqueeze(-1)
    return torch.where(valid, result, torch.zeros_like(result))


def _masked_softmax(logits, mask, dim):
    masked = logits.masked_fill(~mask, -1e4)
    weights = torch.softmax(masked, dim=dim) * mask.to(logits.dtype)
    denominator = weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    return weights / denominator


def _gather_pair(values, indices):
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch[:, None, None], torch.arange(
        values.shape[1], device=values.device
    )[None, :, None], indices]


def _gather_nodes(values, indices):
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch[:, None, None], indices]


class PhysicalMessageLayer(nn.Module):
    """边条件门控消息层，保持定长邻接和残差 LayerNorm。"""

    def __init__(
            self, hidden_dim, edge_dim, boundary_edge_dim, edge_hidden_dim):
        super().__init__()
        self.edge_encoder = _mlp(edge_dim, edge_hidden_dim, hidden_dim)
        self.source_projection = nn.Linear(hidden_dim, hidden_dim)
        self.edge_gate = nn.Linear(hidden_dim * 2 + edge_dim, 1)
        self.boundary_message = _mlp(
            hidden_dim * 2 + boundary_edge_dim,
            hidden_dim,
            hidden_dim,
        )
        self.update = _mlp(hidden_dim * 3, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            hidden,
            neighbor_hidden,
            edge_features,
            edge_mask,
            boundary_hidden,
            boundary_features,
            active):
        target = hidden.unsqueeze(2).expand_as(neighbor_hidden)
        edge_encoded = self.edge_encoder(edge_features)
        gate = torch.sigmoid(self.edge_gate(torch.cat(
            (target, neighbor_hidden, edge_features), dim=-1
        )))
        messages = F.silu(
            self.source_projection(neighbor_hidden) + edge_encoded
        ) * gate
        messages = messages * edge_mask.unsqueeze(-1).to(messages.dtype)
        counts = edge_mask.sum(dim=2, keepdim=True).clamp_min(1)
        neighbor_summary = messages.sum(dim=2) / counts.sqrt().to(messages.dtype)

        batch, fruits, boundaries, _ = boundary_features.shape
        boundary_state = boundary_hidden.view(
            1, 1, boundaries, -1
        ).expand(batch, fruits, -1, -1)
        fruit_state = hidden.unsqueeze(2).expand(-1, -1, boundaries, -1)
        boundary_messages = self.boundary_message(torch.cat(
            (fruit_state, boundary_state, boundary_features), dim=-1
        )).mean(dim=2)

        delta = self.update(torch.cat(
            (hidden, neighbor_summary, boundary_messages), dim=-1
        ))
        result = self.norm(hidden + delta)
        return result * active.unsqueeze(-1).to(result.dtype)


class QueueMessageLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.relative_position = nn.Embedding(7, hidden_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.update = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden):
        queue_length = hidden.shape[1]
        source = hidden.unsqueeze(1).expand(-1, queue_length, -1, -1)
        positions = torch.arange(queue_length, device=hidden.device)
        relative = positions[None, :] - positions[:, None] + 3
        relation = self.relative_position(relative).unsqueeze(0)
        mask = ~torch.eye(
            queue_length, dtype=torch.bool, device=hidden.device
        ).unsqueeze(0)
        messages = F.silu(self.message(source) + relation)
        messages = messages * mask.unsqueeze(-1).to(messages.dtype)
        summary = messages.sum(dim=2) / math.sqrt(max(1, queue_length - 1))
        return self.norm(hidden + self.update(torch.cat((hidden, summary), -1)))


class BaselineGnnDqn(nn.Module):
    """共享物理水果 GNN、队列编码和 21 个单向动作探针。"""

    EDGE_DIM = 14
    BOUNDARY_EDGE_DIM = 5
    ACTION_LIGHT_DIM = 8
    ACTION_DETAIL_DIM = 10

    def __init__(
            self,
            config=None,
            *,
            board_width=560,
            board_height=1120,
            spawn_y=252,
            wall_width=20,
            gravity_y=1800.0):
        super().__init__()
        self.config = config or ModelConfig()
        if not isinstance(self.config, ModelConfig):
            raise TypeError('config must be ModelConfig')
        self.board_width = float(board_width)
        self.board_height = float(board_height)
        self.spawn_y = float(spawn_y)
        self.wall_width = float(wall_width)
        self.velocity_scale = math.sqrt(
            max(1.0, 2.0 * float(gravity_y) * self.board_height)
        )

        level_dim = self.config.level_embedding_dim
        hidden_dim = self.config.hidden_dim
        queue_hidden = self.config.queue_hidden_dim
        self.level_embedding = nn.Embedding(
            MAX_FRUIT_LEVEL + 1, level_dim, padding_idx=0
        )
        self.queue_position_embedding = nn.Embedding(
            self.config.queue_length, level_dim
        )
        self.fruit_encoder = _mlp(level_dim + 9, hidden_dim, hidden_dim)
        self.boundary_embeddings = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.normal_(self.boundary_embeddings, std=0.02)
        self.physical_layers = nn.ModuleList(
            PhysicalMessageLayer(
                hidden_dim,
                self.EDGE_DIM,
                self.BOUNDARY_EDGE_DIM,
                self.config.edge_hidden_dim,
            )
            for _ in range(self.config.message_layers)
        )

        self.queue_encoder = _mlp(
            level_dim * 2 + 1, queue_hidden, queue_hidden
        )
        self.queue_layers = nn.ModuleList(
            QueueMessageLayer(queue_hidden)
            for _ in range(self.config.queue_layers)
        )

        global_input = (
            hidden_dim * 2
            + 1
            + queue_hidden * self.config.queue_length
            + 2
        )
        self.global_encoder = _mlp(global_input, hidden_dim, hidden_dim)

        self.action_light_encoder = _mlp(
            self.ACTION_LIGHT_DIM, hidden_dim // 2, hidden_dim
        )
        self.action_detail_encoder = _mlp(
            hidden_dim + self.ACTION_DETAIL_DIM,
            hidden_dim,
            hidden_dim,
        )
        action_base_input = (
            level_dim + 3 + hidden_dim
            + queue_hidden * self.config.queue_length
        )
        self.action_base_encoder = _mlp(
            action_base_input, hidden_dim, hidden_dim
        )
        self.action_norm = nn.LayerNorm(hidden_dim)
        self.value_head = _mlp(hidden_dim, hidden_dim, 1)
        self.advantage_head = _mlp(hidden_dim, hidden_dim, 1)
        self.extra_value_heads = nn.ModuleList(
            _mlp(hidden_dim, hidden_dim, 1)
            for _ in range(self.config.policy_head_count - 1)
        )
        self.extra_advantage_heads = nn.ModuleList(
            _mlp(hidden_dim, hidden_dim, 1)
            for _ in range(self.config.policy_head_count - 1)
        )
        if self.config.action_effect_enabled:
            self.action_effect_encoder = _mlp(
                hidden_dim, hidden_dim, hidden_dim
            )
            self.action_effect_binary_head = nn.Linear(hidden_dim, 14)
            self.action_effect_categorical_head = nn.Linear(hidden_dim, 91)
            self.action_effect_continuous_head = nn.Linear(hidden_dim, 21)
        else:
            self.action_effect_encoder = None
            self.action_effect_binary_head = None
            self.action_effect_categorical_head = None
            self.action_effect_continuous_head = None
        if self.config.structured_contact_enabled:
            contact_hidden = max(16, hidden_dim // 4)
            self.contact_query = nn.Linear(
                hidden_dim, contact_hidden, bias=False
            )
            self.contact_fruit_projection = nn.Linear(
                hidden_dim, contact_hidden, bias=False
            )
            self.contact_geometry_encoder = _mlp(
                CONTACT_GEOMETRY_FEATURE_DIM,
                contact_hidden,
                contact_hidden,
            )
            self.contact_special_embeddings = nn.Parameter(torch.empty(
                CONTACT_SPECIAL_CANDIDATE_COUNT, contact_hidden
            ))
            nn.init.normal_(self.contact_special_embeddings, std=0.02)
            self.contact_special_score = nn.Linear(
                hidden_dim, CONTACT_SPECIAL_CANDIDATE_COUNT
            )
            self.contact_residual_head = nn.Linear(contact_hidden, 2)
            self.contact_hidden_dim = contact_hidden
        else:
            self.contact_query = None
            self.contact_fruit_projection = None
            self.contact_geometry_encoder = None
            self.register_parameter('contact_special_embeddings', None)
            self.contact_special_score = None
            self.contact_residual_head = None
            self.contact_hidden_dim = 0

        display_radii = [0.0] + [
            float(fruit_radius(level))
            for level in range(1, MAX_FRUIT_LEVEL + 1)
        ]
        drop_radii = [0.0] + [
            float(dropped_fruit_physics_radius(level))
            for level in range(1, MAX_FRUIT_LEVEL + 1)
        ]
        masses = [0.0] + [
            float(fruit_mass(level))
            for level in range(1, MAX_FRUIT_LEVEL + 1)
        ]
        self.register_buffer(
            'display_radii', torch.tensor(display_radii), persistent=True
        )
        self.register_buffer(
            'drop_radii', torch.tensor(drop_radii), persistent=True
        )
        self.register_buffer(
            'mass_table', torch.tensor(masses), persistent=True
        )
        self.register_buffer(
            'action_fraction',
            torch.linspace(0.0, 1.0, self.config.action_count),
            persistent=True,
        )

    def _validate_state(self, state):
        if not isinstance(state, TensorState):
            raise TypeError('state must be TensorState')
        if state.positions.ndim != 3 or state.positions.shape[2] != 2:
            raise ValueError('positions must have shape [B, N, 2]')
        if state.positions.shape[1] != self.config.max_fruits:
            raise ValueError('state fruit capacity does not match model config')
        if state.fruit_queue.shape[1] != self.config.queue_length:
            raise ValueError('state queue length does not match model config')

    def _fruit_features(self, state):
        levels = state.levels.to(torch.long).clamp(0, MAX_FRUIT_LEVEL)
        active = state.active
        position = state.positions
        velocity = state.velocities
        radii = state.physics_radii
        mass = self.mass_table[levels]
        age_seconds = state.age_frames.to(position.dtype) / state.physics_fps
        continuous = torch.stack(
            (
                position[..., 0] / self.board_width * 2.0 - 1.0,
                position[..., 1] / self.board_height * 2.0 - 1.0,
                torch.tanh(velocity[..., 0] / self.velocity_scale),
                torch.tanh(velocity[..., 1] / self.velocity_scale),
                torch.tanh(state.angular_velocities / 10.0),
                radii / self.board_width,
                torch.log1p(mass) / math.log(17.0),
                torch.log1p(age_seconds) / math.log(61.0),
                (position[..., 1] - radii - self.spawn_y)
                / self.board_height,
            ),
            dim=-1,
        )
        hidden = self.fruit_encoder(torch.cat(
            (self.level_embedding(levels), continuous), dim=-1
        ))
        return hidden * active.unsqueeze(-1).to(hidden.dtype)

    def _physical_graph(self, state):
        position = state.positions
        velocity = state.velocities
        radii = state.physics_radii
        active = state.active
        batch, fruits, _ = position.shape

        relative = position.unsqueeze(1) - position.unsqueeze(2)
        relative_velocity = velocity.unsqueeze(1) - velocity.unsqueeze(2)
        distance = relative.square().sum(dim=-1).clamp_min(1e-8).sqrt()
        radius_sum = radii.unsqueeze(1) + radii.unsqueeze(2)
        gap = distance - radius_sum
        pair_valid = active.unsqueeze(1) & active.unsqueeze(2)
        pair_valid &= ~torch.eye(
            fruits, dtype=torch.bool, device=position.device
        ).unsqueeze(0)

        contact_limit = torch.maximum(
            torch.ones_like(radius_sum), radius_sum * 0.02
        )
        contact = pair_valid & (gap <= contact_limit)

        nearest_count = min(self.config.nearest_neighbors, fruits)
        nearest_score = gap / radius_sum.clamp_min(1.0)
        nearest_indices = nearest_score.masked_fill(
            ~pair_valid, float('inf')
        ).topk(nearest_count, dim=2, largest=False).indices
        nearest = torch.zeros_like(pair_valid)
        nearest.scatter_(2, nearest_indices, True)
        nearest &= pair_valid

        speed_squared = relative_velocity.square().sum(dim=-1).clamp_min(1e-6)
        ttc = -(
            relative * relative_velocity
        ).sum(dim=-1) / speed_squared
        closest = relative + relative_velocity * ttc.unsqueeze(-1)
        closest_distance = closest.square().sum(dim=-1).sqrt()
        motion_candidate = (
            pair_valid
            & (ttc > 0.0)
            & (ttc <= 0.5)
            & (closest_distance <= radius_sum * 1.25)
        )
        motion_count = min(self.config.motion_neighbors, fruits)
        motion_indices = ttc.masked_fill(
            ~motion_candidate, float('inf')
        ).topk(motion_count, dim=2, largest=False).indices
        motion = torch.zeros_like(pair_valid)
        motion.scatter_(2, motion_indices, True)
        motion &= motion_candidate

        horizontal_overlap = relative[..., 0].abs() <= radius_sum * 1.5
        vertical_base = pair_valid & horizontal_overlap
        vertical = torch.zeros_like(pair_valid)
        per_direction = min(
            self.config.vertical_neighbors_per_direction, fruits
        )
        for direction_mask in (
                relative[..., 1] < 0.0,
                relative[..., 1] > 0.0):
            candidates = vertical_base & direction_mask
            indices = relative[..., 1].abs().masked_fill(
                ~candidates, float('inf')
            ).topk(per_direction, dim=2, largest=False).indices
            selected = torch.zeros_like(pair_valid)
            selected.scatter_(2, indices, True)
            vertical |= selected & candidates

        normalized_gap = gap / radius_sum.clamp_min(1.0)
        priority = torch.full_like(gap, float('inf'))
        priority = torch.where(contact, normalized_gap.clamp_min(-1.0), priority)
        priority = torch.minimum(
            priority,
            torch.where(motion, 1.0 + ttc, torch.full_like(ttc, float('inf'))),
        )
        priority = torch.minimum(
            priority,
            torch.where(
                vertical,
                2.0 + relative[..., 1].abs() / self.board_height,
                torch.full_like(gap, float('inf')),
            ),
        )
        priority = torch.minimum(
            priority,
            torch.where(
                nearest,
                3.0 + normalized_gap.clamp_min(0.0),
                torch.full_like(gap, float('inf')),
            ),
        )

        neighbor_count = min(self.config.max_neighbors, fruits)
        neighbor_indices = priority.topk(
            neighbor_count, dim=2, largest=False
        ).indices
        edge_mask = torch.isfinite(_gather_pair(priority, neighbor_indices))

        unit = relative / distance.unsqueeze(-1).clamp_min(1e-6)
        normal_speed = (relative_velocity * unit).sum(dim=-1)
        tangent_speed = (
            relative_velocity[..., 0] * -unit[..., 1]
            + relative_velocity[..., 1] * unit[..., 0]
        )
        level_i = state.levels.unsqueeze(2)
        level_j = state.levels.unsqueeze(1)
        edge_full = torch.stack(
            (
                relative[..., 0] / radius_sum.clamp_min(1.0),
                relative[..., 1] / radius_sum.clamp_min(1.0),
                distance / radius_sum.clamp_min(1.0),
                normalized_gap,
                relative_velocity[..., 0] / self.velocity_scale,
                relative_velocity[..., 1] / self.velocity_scale,
                normal_speed / self.velocity_scale,
                tangent_speed / self.velocity_scale,
                (level_i == level_j).to(position.dtype),
                contact.to(position.dtype),
                nearest.to(position.dtype),
                motion.to(position.dtype),
                vertical.to(position.dtype),
                torch.where(
                    torch.isfinite(ttc), ttc.clamp(0.0, 1.0), torch.ones_like(ttc)
                ),
            ),
            dim=-1,
        )
        edge_features = _gather_pair(edge_full, neighbor_indices)
        return neighbor_indices, edge_features, edge_mask

    def _boundary_features(self, state):
        x = state.positions[..., 0]
        y = state.positions[..., 1]
        vx = state.velocities[..., 0]
        vy = state.velocities[..., 1]
        radius = state.physics_radii
        clearances = torch.stack(
            (
                x - self.wall_width - radius,
                self.board_width - self.wall_width - radius - x,
                self.board_height - self.wall_width - radius - y,
            ),
            dim=-1,
        )
        normal_velocity = torch.stack((-vx, vx, vy), dim=-1)
        tangent_velocity = torch.stack((vy, vy, vx), dim=-1)
        boundary_type = torch.arange(
            3, device=x.device, dtype=x.dtype
        ).view(1, 1, 3).expand_as(clearances) / 2.0
        return torch.stack(
            (
                clearances / self.board_width,
                normal_velocity / self.velocity_scale,
                tangent_velocity / self.velocity_scale,
                (clearances <= 1.0).to(x.dtype),
                boundary_type,
            ),
            dim=-1,
        )

    def _encode_queue(self, state):
        levels = state.fruit_queue.to(torch.long).clamp(0, MAX_FRUIT_LEVEL)
        batch = levels.shape[0]
        positions = torch.arange(
            self.config.queue_length, device=levels.device
        )
        position_embed = self.queue_position_embedding(positions).unsqueeze(0)
        position_embed = position_embed.expand(batch, -1, -1)
        order = positions.to(state.positions.dtype).view(1, -1, 1)
        order = order.expand(batch, -1, -1) / max(
            1, self.config.queue_length - 1
        )
        hidden = self.queue_encoder(torch.cat(
            (self.level_embedding(levels), position_embed, order), dim=-1
        ))
        for layer in self.queue_layers:
            hidden = layer(hidden)
        return hidden

    def _encode_fruits(self, state):
        hidden = self._fruit_features(state)
        neighbor_indices, edge_features, edge_mask = self._physical_graph(state)
        boundary_features = self._boundary_features(state)
        for layer in self.physical_layers:
            neighbor_hidden = _gather_nodes(hidden, neighbor_indices)
            hidden = layer(
                hidden,
                neighbor_hidden,
                edge_features,
                edge_mask,
                self.boundary_embeddings,
                boundary_features,
                state.active,
            )
        return hidden

    def _global_context(self, state, fruit_hidden, queue_hidden):
        active_float = state.active.unsqueeze(-1).to(fruit_hidden.dtype)
        count = state.active.sum(dim=1, keepdim=True)
        fruit_sum = (fruit_hidden * active_float).sum(dim=1)
        fruit_sum = fruit_sum / count.clamp_min(1).sqrt().to(fruit_sum.dtype)
        fruit_max = _masked_max(fruit_hidden, state.active, dim=1)
        queue_flat = queue_hidden.flatten(1)
        inputs = torch.cat(
            (
                fruit_sum,
                fruit_max,
                count.to(fruit_hidden.dtype) / self.config.max_fruits,
                queue_flat,
                state.danger_progress.to(fruit_hidden.dtype).unsqueeze(-1),
                state.over_danger_line.to(fruit_hidden.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        return self.global_encoder(inputs), queue_flat

    def _action_context(
            self, state, fruit_hidden, queue_hidden, global_hidden, queue_flat):
        batch, fruits, hidden_dim = fruit_hidden.shape
        actions = self.config.action_count
        q0_level = state.fruit_queue[:, 0].to(torch.long).clamp(
            0, MAX_FRUIT_LEVEL
        )
        display_radius = self.display_radii[q0_level]
        drop_radius = self.drop_radii[q0_level]
        left = self.wall_width + display_radius + 2.0
        right = self.board_width - self.wall_width - display_radius - 2.0
        drop_x = left[:, None] + (
            right - left
        )[:, None] * self.action_fraction[None, :]

        fruit_x = state.positions[..., 0].unsqueeze(1)
        fruit_y = state.positions[..., 1].unsqueeze(1)
        fruit_radius_value = state.physics_radii.unsqueeze(1)
        dx_pixels = fruit_x - drop_x.unsqueeze(-1)
        dy_pixels = fruit_y - self.spawn_y
        radius_sum = fruit_radius_value + drop_radius[:, None, None]
        clearance = dx_pixels.abs() - radius_sum
        same_level = (
            state.levels.unsqueeze(1) == q0_level[:, None, None]
        ) & state.active.unsqueeze(1)
        same_level = same_level.expand(-1, actions, -1)
        moving_toward = (
            -dx_pixels.sign() * state.velocities[..., 0].unsqueeze(1)
            / self.velocity_scale
        )
        danger_distance = (
            fruit_y - fruit_radius_value - self.spawn_y
        ) / self.board_height
        light_features = torch.stack(
            (
                dx_pixels / self.board_width,
                dy_pixels.expand(-1, actions, -1) / self.board_height,
                torch.tanh(clearance / radius_sum.clamp_min(1.0)),
                same_level.to(fruit_hidden.dtype),
                moving_toward,
                danger_distance.expand(-1, actions, -1),
                state.angular_velocities.unsqueeze(1).expand(-1, actions, -1)
                .tanh(),
                state.active.unsqueeze(1).expand(-1, actions, -1)
                .to(fruit_hidden.dtype),
            ),
            dim=-1,
        )
        light_encoded = self.action_light_encoder(light_features)
        light_mask = state.active.unsqueeze(1).expand(-1, actions, -1)
        light_logits = -clearance.abs() / radius_sum.clamp_min(1.0)
        light_weights = _masked_softmax(light_logits, light_mask, dim=2)
        light_summary = (
            light_encoded * light_weights.unsqueeze(-1)
        ).sum(dim=2)

        intersects = light_mask & (clearance <= radius_sum * 0.25)
        root = (radius_sum.square() - dx_pixels.square()).clamp_min(0.0).sqrt()
        contact_y = fruit_y - root
        key_score = torch.where(
            intersects,
            contact_y / self.board_height,
            2.0
            + dx_pixels.abs() / self.board_width
            + dy_pixels.expand(-1, actions, -1).abs() / self.board_height,
        )
        key_score = key_score - moving_toward.clamp_min(0.0) * 0.1
        key_count = min(self.config.action_key_fruits, fruits)
        key_indices = key_score.masked_fill(
            ~light_mask, float('inf')
        ).topk(key_count, dim=2, largest=False).indices
        key_valid = torch.isfinite(torch.gather(key_score, 2, key_indices))
        key_fruit_hidden = _gather_nodes(fruit_hidden, key_indices)
        key_light = torch.gather(
            light_features,
            2,
            key_indices.unsqueeze(-1).expand(-1, -1, -1, self.ACTION_LIGHT_DIM),
        )
        key_contact_y = torch.gather(contact_y, 2, key_indices).unsqueeze(-1)
        key_intersects = torch.gather(
            intersects, 2, key_indices
        ).to(fruit_hidden.dtype).unsqueeze(-1)
        detail_features = torch.cat(
            (
                key_light,
                key_contact_y / self.board_height,
                key_intersects,
            ),
            dim=-1,
        )
        detail_encoded = self.action_detail_encoder(torch.cat(
            (key_fruit_hidden, detail_features), dim=-1
        ))
        detail_score = torch.gather(key_score, 2, key_indices)
        detail_weights = _masked_softmax(-detail_score, key_valid, dim=2)
        detail_summary = (
            detail_encoded * detail_weights.unsqueeze(-1)
        ).sum(dim=2)

        q0_embedding = self.level_embedding(q0_level).unsqueeze(1).expand(
            -1, actions, -1
        )
        action_scalar = torch.stack(
            (
                drop_x / self.board_width * 2.0 - 1.0,
                display_radius[:, None].expand(-1, actions) / self.board_width,
                drop_radius[:, None].expand(-1, actions) / self.board_width,
            ),
            dim=-1,
        )
        base = self.action_base_encoder(torch.cat(
            (
                q0_embedding,
                action_scalar,
                global_hidden.unsqueeze(1).expand(-1, actions, -1),
                queue_flat.unsqueeze(1).expand(-1, actions, -1),
            ),
            dim=-1,
        ))
        return self.action_norm(base + light_summary + detail_summary)

    def _structured_contact_predictions(
            self, state, fruit_hidden, action_hidden):
        q0_level = state.fruit_queue[:, 0].to(torch.long).clamp(
            0, MAX_FRUIT_LEVEL
        )
        display_radius = self.display_radii[q0_level]
        drop_radius = self.drop_radii[q0_level]
        left = self.wall_width + display_radius + 2.0
        right = self.board_width - self.wall_width - display_radius - 2.0
        drop_x = left[:, None] + (
            right - left
        )[:, None] * self.action_fraction[None, :]
        geometry = build_contact_candidate_geometry(
            state,
            drop_x,
            drop_radius,
            board_width=self.board_width,
            board_height=self.board_height,
            spawn_y=self.spawn_y,
            wall_width=self.wall_width,
            velocity_scale=self.velocity_scale,
        )
        query = self.contact_query(action_hidden)
        fruit_candidates = (
            self.contact_fruit_projection(fruit_hidden).unsqueeze(1)
            + self.contact_geometry_encoder(geometry.fruit_features)
        )
        special_candidates = self.contact_special_embeddings[
            None, None
        ].expand(
            action_hidden.shape[0], action_hidden.shape[1], -1, -1
        )
        scale = math.sqrt(float(self.contact_hidden_dim))
        special_logits = self.contact_special_score(action_hidden) + (
            query.unsqueeze(2) * special_candidates
        ).sum(dim=-1) / scale
        fruit_logits = (
            query.unsqueeze(2) * fruit_candidates
        ).sum(dim=-1) / scale
        candidate_logits = torch.cat(
            (special_logits, fruit_logits), dim=2
        ).masked_fill(~geometry.valid, -1e4)
        candidates = torch.cat(
            (special_candidates, fruit_candidates), dim=2
        )
        candidate_residual = self.contact_residual_head(
            F.silu(candidates + query.unsqueeze(2))
        )
        selected = candidate_logits.argmax(dim=2)
        gather_index = selected[..., None, None].expand(-1, -1, 1, 2)
        selected_prior = geometry.positions.gather(
            2, gather_index
        ).squeeze(2)
        selected_residual = candidate_residual.gather(
            2, gather_index
        ).squeeze(2)
        primary_logits = torch.stack((
            candidate_logits[..., 0],
            candidate_logits[..., 1],
            candidate_logits[..., 2],
            candidate_logits[..., 3],
            torch.logsumexp(candidate_logits[..., 4:], dim=-1),
        ), dim=-1)
        return (
            candidate_logits,
            candidate_residual,
            selected_prior + selected_residual,
            primary_logits,
        )

    def _predict_action_effects(self, state, fruit_hidden, action_hidden):
        if not self.config.action_effect_enabled:
            return None
        hidden = self.action_effect_encoder(action_hidden)
        binary = self.action_effect_binary_head(hidden)
        categorical = self.action_effect_categorical_head(hidden)
        continuous = self.action_effect_continuous_head(hidden)
        batch, actions, _ = hidden.shape
        contact_target_logits = None
        contact_position_residual = None
        contact_position = continuous[..., 0:2]
        contact_primary_type_logits = categorical[..., 29:34]
        if self.config.structured_contact_enabled:
            (
                contact_target_logits,
                contact_position_residual,
                contact_position,
                contact_primary_type_logits,
            ) = self._structured_contact_predictions(
                state, fruit_hidden, hidden
            )
        return ActionEffectPredictions(
            merge_logit=binary[..., 0],
            merge_count_logits=categorical[..., 0:9],
            q0_participated_logit=binary[..., 1],
            q0_lineage_depth_logits=categorical[..., 9:17],
            q0_final_level_logits=categorical[..., 17:29],
            contact_type_logits=binary[..., 2:6],
            contact_primary_type_logits=contact_primary_type_logits,
            contact_position=contact_position,
            contact_level_delta_logits=categorical[..., 34:55],
            contact_normal=continuous[..., 2:4],
            contact_age=continuous[..., 4],
            contact_normal_speed=continuous[..., 5],
            generation_exists_logits=binary[..., 6:9],
            generation_position=continuous[..., 6:12].reshape(
                batch, actions, 3, 2
            ),
            generation_level_logits=categorical[..., 55:91].reshape(
                batch, actions, 3, 12
            ),
            score_delta=continuous[..., 12],
            fruit_count_delta=continuous[..., 13],
            final_exists_logit=binary[..., 9],
            final_state=continuous[..., 14:19],
            stable_logit=binary[..., 10],
            settle_timeout_logit=binary[..., 11],
            terminal_logit=binary[..., 12],
            settle_duration=continuous[..., 19],
            danger_delta=continuous[..., 20],
            over_danger_line_logit=binary[..., 13],
            contact_target_logits=contact_target_logits,
            contact_position_residual=contact_position_residual,
        )

    def forward_with_details(
            self,
            state,
            predict_action_effects=True,
            action_effect_batch_size=None):
        self._validate_state(state)
        fruit_hidden = self._encode_fruits(state)
        queue_hidden = self._encode_queue(state)
        global_hidden, queue_flat = self._global_context(
            state, fruit_hidden, queue_hidden
        )
        action_hidden = self._action_context(
            state,
            fruit_hidden,
            queue_hidden,
            global_hidden,
            queue_flat,
        )
        head_values = []
        value_heads = (self.value_head, *self.extra_value_heads)
        advantage_heads = (self.advantage_head, *self.extra_advantage_heads)
        for value_head, advantage_head in zip(value_heads, advantage_heads):
            value = value_head(global_hidden)
            advantage = advantage_head(action_hidden).squeeze(-1)
            head_values.append(
                value + advantage - advantage.mean(dim=1, keepdim=True)
            )
        head_q_values = torch.stack(head_values, dim=1)
        return ModelOutput(
            q_values=head_q_values.mean(dim=1),
            head_q_values=head_q_values,
            action_effects=(
                self._predict_action_effects(
                    state if action_effect_batch_size is None else
                    state.batch_slice(action_effect_batch_size),
                    fruit_hidden if action_effect_batch_size is None else
                    fruit_hidden[:action_effect_batch_size],
                    action_hidden if action_effect_batch_size is None else
                    action_hidden[:action_effect_batch_size],
                )
                if predict_action_effects
                else None
            ),
        )

    def forward(
            self,
            state,
            return_details=False,
            predict_action_effects=False,
            action_effect_batch_size=None):
        output = self.forward_with_details(
            state, predict_action_effects, action_effect_batch_size
        )
        return output if return_details else output.q_values
