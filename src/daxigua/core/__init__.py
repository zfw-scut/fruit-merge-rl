"""不依赖游戏模拟器和训练框架的领域规则与状态契约。"""

from .rules import (
    FRUIT_QUEUE_LENGTH,
    FRUIT_RADII,
    MAX_FRUIT_LEVEL,
    MIN_FRUIT_LEVEL,
    SPAWN_FRUIT_MAX_LEVEL,
    SPAWN_FRUIT_MIN_LEVEL,
    dropped_fruit_physics_radius,
    fruit_mass,
    fruit_radius,
    merge_score,
    merge_target_level,
    merged_fruit_physics_radius,
    random_spawn_level,
)
from .state import (
    ActionCandidate,
    BoardGeometry,
    DropResult,
    FruitState,
    GameState,
    MergeEvent,
    PhysicsResult,
)

__all__ = [
    'ActionCandidate',
    'BoardGeometry',
    'DropResult',
    'FRUIT_QUEUE_LENGTH',
    'FRUIT_RADII',
    'FruitState',
    'GameState',
    'MAX_FRUIT_LEVEL',
    'MIN_FRUIT_LEVEL',
    'MergeEvent',
    'PhysicsResult',
    'SPAWN_FRUIT_MAX_LEVEL',
    'SPAWN_FRUIT_MIN_LEVEL',
    'dropped_fruit_physics_radius',
    'fruit_mass',
    'fruit_radius',
    'merge_score',
    'merge_target_level',
    'merged_fruit_physics_radius',
    'random_spawn_level',
]
