"""游戏核心逻辑包。

这里放不依赖具体训练算法的核心组件，例如水果对象、物理世界、碰撞合成和失败判断。
表现层可以调用核心逻辑；核心逻辑不应该 import 表现层或 RL 层。
"""

from .engine import HeadlessGame
from .state import ActionCandidate, DropResult, FruitState, GameState, PhysicsResult


__all__ = [
    'ActionCandidate',
    'DropResult',
    'FruitState',
    'GameState',
    'HeadlessGame',
    'PhysicsResult',
]
