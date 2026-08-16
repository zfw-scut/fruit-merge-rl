"""逐水果预测距离下一次合成投放步距的轻量图网络。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from daxigua.core import MAX_FRUIT_LEVEL, fruit_mass

from .merge_potential_stats import (
    OUTCOME_CENSORED,
    OUTCOME_MERGED,
    OUTCOME_TERMINAL_UNMERGED,
)
from .model import (
    BaselineGnnDqn,
    PhysicalMessageLayer,
    QueueMessageLayer,
    _masked_max,
    _mlp,
)


DEFAULT_MERGE_HORIZONS = (
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
)


@dataclass(frozen=True, slots=True)
class MergeDistanceConfig:
    """独立预测器的冻结结构配置。"""

    max_fruits: int = 64
    queue_length: int = 4
    hidden_dim: int = 64
    edge_hidden_dim: int = 64
    message_layers: int = 2
    queue_hidden_dim: int = 32
    queue_layers: int = 1
    level_embedding_dim: int = 12
    max_neighbors: int = 12
    nearest_neighbors: int = 4
    motion_neighbors: int = 2
    vertical_neighbors_per_direction: int = 2
    horizons: tuple[int, ...] = DEFAULT_MERGE_HORIZONS

    def __post_init__(self):
        integer_names = (
            'max_fruits', 'queue_length', 'hidden_dim', 'edge_hidden_dim',
            'message_layers', 'queue_hidden_dim', 'queue_layers',
            'level_embedding_dim', 'max_neighbors', 'nearest_neighbors',
            'motion_neighbors', 'vertical_neighbors_per_direction',
        )
        for name in integer_names:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.queue_length != 4:
            raise ValueError('queue_length must be 4')
        if self.max_neighbors > self.max_fruits:
            raise ValueError('max_neighbors cannot exceed max_fruits')
        horizons = tuple(int(value) for value in self.horizons)
        if not horizons or horizons[0] <= 0:
            raise ValueError('horizons must contain positive integers')
        if any(left >= right for left, right in zip(horizons, horizons[1:])):
            raise ValueError('horizons must be strictly increasing')
        object.__setattr__(self, 'horizons', horizons)

    @property
    def merge_class_count(self):
        """时间区间、长尾合成和自然终局未合成的总类别数。"""

        return len(self.horizons) + 2

    @property
    def tail_class(self):
        return len(self.horizons)

    @property
    def terminal_unmerged_class(self):
        return len(self.horizons) + 1

    def to_dict(self):
        payload = asdict(self)
        payload['horizons'] = list(self.horizons)
        return payload

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        if 'horizons' in values:
            values['horizons'] = tuple(values['horizons'])
        return cls(**values)


class MergeDistanceOutput(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor


class MergeDistancePredictor(nn.Module):
    """复用现有物理图语义，对每个活动水果输出离散生存分布。"""

    EDGE_DIM = BaselineGnnDqn.EDGE_DIM
    BOUNDARY_EDGE_DIM = BaselineGnnDqn.BOUNDARY_EDGE_DIM

    # 这些方法只依赖共同的几何属性与编码器。直接复用可避免预测器和
    # SAB 主模型产生两套略有差异的水果图定义。
    _validate_state = BaselineGnnDqn._validate_state
    _fruit_features = BaselineGnnDqn._fruit_features
    _physical_graph = BaselineGnnDqn._physical_graph
    _boundary_features = BaselineGnnDqn._boundary_features
    _encode_queue = BaselineGnnDqn._encode_queue
    _encode_fruits = BaselineGnnDqn._encode_fruits

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
        self.config = config or MergeDistanceConfig()
        if not isinstance(self.config, MergeDistanceConfig):
            raise TypeError('config must be MergeDistanceConfig')
        self.board_width = float(board_width)
        self.board_height = float(board_height)
        self.spawn_y = float(spawn_y)
        self.wall_width = float(wall_width)
        self.gravity_y = float(gravity_y)
        self.velocity_scale = math.sqrt(
            max(1.0, 2.0 * self.gravity_y * self.board_height)
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
        self.node_context = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.output_head = nn.Linear(
            hidden_dim, self.config.merge_class_count
        )

        masses = [0.0] + [
            float(fruit_mass(level))
            for level in range(1, MAX_FRUIT_LEVEL + 1)
        ]
        self.register_buffer(
            'mass_table', torch.tensor(masses), persistent=True
        )

    def _global_context(self, state, fruit_hidden, queue_hidden):
        active_float = state.active.unsqueeze(-1).to(fruit_hidden.dtype)
        count = state.active.sum(dim=1, keepdim=True)
        fruit_sum = (fruit_hidden * active_float).sum(dim=1)
        fruit_sum = fruit_sum / count.clamp_min(1).sqrt().to(fruit_sum.dtype)
        fruit_max = _masked_max(fruit_hidden, state.active, dim=1)
        queue_flat = queue_hidden.flatten(1)
        inputs = torch.cat((
            fruit_sum,
            fruit_max,
            count.to(fruit_hidden.dtype) / self.config.max_fruits,
            queue_flat,
            state.danger_progress.to(fruit_hidden.dtype).unsqueeze(-1),
            state.over_danger_line.to(fruit_hidden.dtype).unsqueeze(-1),
        ), dim=-1)
        return self.global_encoder(inputs)

    def forward(self, state):
        self._validate_state(state)
        fruit_hidden = self._encode_fruits(state)
        queue_hidden = self._encode_queue(state)
        global_hidden = self._global_context(
            state, fruit_hidden, queue_hidden
        )
        global_nodes = global_hidden.unsqueeze(1).expand(
            -1, self.config.max_fruits, -1
        )
        node_hidden = self.node_context(torch.cat(
            (fruit_hidden, global_nodes), dim=-1
        ))
        logits = self.output_head(node_hidden)
        logits = logits * state.active.unsqueeze(-1).to(logits.dtype)
        return MergeDistanceOutput(
            logits=logits,
            probabilities=torch.softmax(logits, dim=-1),
        )

    @property
    def geometry_config(self):
        return {
            'board_width': self.board_width,
            'board_height': self.board_height,
            'spawn_y': self.spawn_y,
            'wall_width': self.wall_width,
            'gravity_y': self.gravity_y,
        }


def merge_distance_targets(outcomes, t_merge, horizons):
    """把未来事实映射为时间区间、长尾或自然终局未合成类别。"""

    outcomes = torch.as_tensor(outcomes)
    t_merge = torch.as_tensor(t_merge, device=outcomes.device)
    boundaries = torch.as_tensor(
        tuple(horizons), dtype=t_merge.dtype, device=t_merge.device
    )
    targets = torch.bucketize(t_merge.clamp_min(1), boundaries, right=False)
    terminal_class = len(tuple(horizons)) + 1
    targets = torch.where(
        outcomes == OUTCOME_TERMINAL_UNMERGED,
        torch.full_like(targets, terminal_class),
        targets,
    )
    valid = outcomes != OUTCOME_CENSORED
    if bool(((outcomes == OUTCOME_MERGED) & (t_merge <= 0)).any().item()):
        raise ValueError('merged targets require positive t_merge')
    return targets.to(torch.long), valid


def merge_distance_loss(
        logits,
        active,
        outcomes,
        t_merge,
        horizons,
        *,
        sample_weights=None):
    """对已解析水果计算加权离散生存负对数似然。"""

    targets, resolved = merge_distance_targets(outcomes, t_merge, horizons)
    valid = active.to(torch.bool) & resolved
    if not bool(valid.any().item()):
        return logits.sum() * 0.0
    losses = F.cross_entropy(
        logits[valid].float(), targets[valid], reduction='none'
    )
    if sample_weights is None:
        return losses.mean()
    weights = torch.as_tensor(
        sample_weights, dtype=losses.dtype, device=losses.device
    )[valid]
    return (losses * weights).sum() / weights.sum().clamp_min(1e-8)


def cumulative_merge_probabilities(probabilities, horizons):
    """返回每个时间上界前已完成合成的单调累计概率。"""

    horizon_count = len(tuple(horizons))
    if probabilities.shape[-1] != horizon_count + 2:
        raise ValueError('probability class count does not match horizons')
    return probabilities[..., :horizon_count].cumsum(dim=-1)


__all__ = [
    'DEFAULT_MERGE_HORIZONS',
    'MergeDistanceConfig',
    'MergeDistanceOutput',
    'MergeDistancePredictor',
    'cumulative_merge_probabilities',
    'merge_distance_loss',
    'merge_distance_targets',
]
