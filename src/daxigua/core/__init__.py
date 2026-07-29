"""游戏核心逻辑包。

这里放不依赖具体训练算法的核心组件，例如水果对象、物理世界、碰撞合成和失败判断。
表现层可以调用核心逻辑；核心逻辑不应该 import 表现层或 RL 层。

``state`` 中的只读数据结构会被 Android 侧的纯 Python 分析桥复用。物理引擎
``HeadlessGame`` 依赖 Pymunk，因此这里只在调用方真正读取它时再导入；这样
``from daxigua.core.state import GameState`` 不会无意中把桌面物理依赖带到手机端。
"""

from .state import (
    ActionCandidate,
    BoundaryPhysicsSnapshot,
    DropResult,
    ENGINE_SNAPSHOT_SCHEMA_VERSION,
    EngineActionOutcome,
    EngineConfigSnapshot,
    EngineEpisodeSnapshot,
    EngineSnapshot,
    FruitPhysicsSnapshot,
    FruitState,
    GameState,
    OriginalActionReplayReport,
    PhysicsResult,
)


__all__ = [
    'ActionCandidate',
    'BoundaryPhysicsSnapshot',
    'DropResult',
    'ENGINE_SNAPSHOT_SCHEMA_VERSION',
    'EngineActionOutcome',
    'EngineConfigSnapshot',
    'EngineEpisodeSnapshot',
    'EngineSnapshot',
    'FruitPhysicsSnapshot',
    'FruitState',
    'GameState',
    'HeadlessGame',
    'OriginalActionReplayReport',
    'PhysicsResult',
]


def __getattr__(name):
    """按需导入依赖 Pymunk 的物理引擎，同时保留原公开导入方式。"""

    if name == 'HeadlessGame':
        from .engine import HeadlessGame

        # 缓存解析结果，避免后续属性访问重复进入懒加载分支。
        globals()[name] = HeadlessGame
        return HeadlessGame
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
