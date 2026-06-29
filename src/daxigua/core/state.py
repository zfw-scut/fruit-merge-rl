"""训练友好的游戏状态数据结构。

这里的类型全部是普通 Python 数据，不包含 pygame Surface、pymunk Shape 或模型张量。
这样可以让游戏本体向外暴露稳定接口，同时避免 RL 代码直接读取内部物理对象。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FruitState:
    """场地中一个真实水果的状态快照。"""

    fruit_id: int
    level: int
    radius: float
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    age_frames: int
    stable: bool
    distance_to_left_wall: float
    distance_to_right_wall: float
    distance_to_floor: float
    distance_to_danger_line: float


@dataclass(frozen=True)
class BoardGeometry:
    """游戏场地几何信息。"""

    width: int
    height: int
    spawn_y: int
    wall_width: int
    floor_y: int


@dataclass(frozen=True)
class ActionCandidate:
    """一个离散投放动作候选。"""

    action_index: int
    drop_x: float
    normalized_drop_x: float
    current_level: int
    current_radius: float


@dataclass(frozen=True)
class MergeEvent:
    """一次同级水果合成事件。"""

    new_level: int
    x: float
    y: float
    score_delta: int
    source_ids: tuple
    new_fruit_id: int


@dataclass(frozen=True)
class GameState:
    """一个完整游戏状态快照。"""

    board_fruits: tuple
    fruit_queue: tuple
    score: int
    last_score: int
    step_count: int
    physics_frame: int
    done: bool
    geometry: BoardGeometry
    max_height: float
    fruit_count: int
    max_level: int
    empty_space_ratio: float


@dataclass(frozen=True)
class DropResult:
    """投放动作执行后的即时结果，不包含后续物理稳定过程。"""

    dropped_level: int
    drop_x: float
    fruit_id: int
    queue_before: tuple
    queue_after: tuple


@dataclass(frozen=True)
class PhysicsResult:
    """一次投放后推进物理世界得到的结果。"""

    frames_simulated: int
    stable: bool
    done: bool
    truncated: bool
    score_delta: int
    merge_events: tuple = field(default_factory=tuple)
