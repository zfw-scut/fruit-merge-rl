"""与物理内核解耦的批量奖励接口。"""

from typing import Protocol

import torch

from .types import BatchObservation, BatchPhysicsResult


class RewardComputer(Protocol):
    """强化学习奖励计算器的最小契约。"""

    def __call__(
            self,
            previous: BatchObservation,
            current: BatchObservation,
            physics: BatchPhysicsResult) -> torch.Tensor:
        """返回形状为 ``[num_envs]`` 的浮点 Tensor。"""


class ZeroReward:
    """显式的零奖励；只用于物理和接口测试。"""

    requires_previous_state = False

    def __call__(self, previous, current, physics):
        del previous, current
        return torch.zeros_like(physics.score_delta, dtype=torch.float32)


class GameScoreReward:
    """把游戏分数增量显式映射为奖励。

    这是可选适配器，不是项目默认的正式 RL 奖励。
    """

    requires_previous_state = False

    def __call__(self, previous, current, physics):
        del previous, current
        return physics.score_delta.to(dtype=torch.float32)
