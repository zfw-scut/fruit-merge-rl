"""训练友好的游戏状态数据结构。

这里的类型全部是普通 Python 数据，不包含 pygame Surface、pymunk Shape 或模型张量。
这样可以让游戏本体向外暴露稳定接口，同时避免 RL 代码直接读取内部物理对象。
"""

from dataclasses import dataclass, field, replace
import hashlib
import hmac
import json
import math
import pickle


# 完整物理快照的格式版本。它和训练侧归因版本分离：前者只描述能否恢复同一个
# HeadlessGame 物理边界，后者描述奖励与因果标签语义。
ENGINE_SNAPSHOT_SCHEMA_VERSION = 1


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
    physics_radius: float | None = None

    def __post_init__(self):
        """补齐真实碰撞半径，同时兼容旧调用方只传显示半径。"""

        physics_radius = self.radius if self.physics_radius is None else self.physics_radius
        object.__setattr__(self, 'radius', float(self.radius))
        object.__setattr__(self, 'physics_radius', float(physics_radius))

        if self.radius <= 0:
            raise ValueError('radius must be positive')
        if self.physics_radius <= 0:
            raise ValueError('physics_radius must be positive')


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
    current_physics_radius: float | None = None

    def __post_init__(self):
        """补齐待投放水果碰撞半径，同时保留旧构造接口。"""

        physics_radius = (
            self.current_radius
            if self.current_physics_radius is None
            else self.current_physics_radius
        )
        object.__setattr__(self, 'current_radius', float(self.current_radius))
        object.__setattr__(self, 'current_physics_radius', float(physics_radius))

        if self.current_radius <= 0:
            raise ValueError('current_radius must be positive')
        if self.current_physics_radius <= 0:
            raise ValueError('current_physics_radius must be positive')


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


@dataclass(frozen=True, slots=True)
class EngineConfigSnapshot:
    """会影响 HeadlessGame 后续物理演化的完整配置镜像。

    Pymunk ``Space`` 的二进制快照仍是恢复时的真实数据源；这里保留显式字段，是为了
    在反序列化不可信 blob 前先拒绝跨场地、跨物理模式或跨求解器参数恢复。
    """

    width: int
    height: int
    spawn_y: int
    wall_width: int
    fps: int
    space_iterations: int
    gravity: tuple
    queue_length: int
    create_time: float
    stable_velocity_epsilon: float
    stable_angular_velocity_epsilon: float
    space_damping: float
    collision_slop: float
    collision_bias: float
    collision_persistence: int
    idle_speed_threshold: float
    sleep_time_threshold: float
    threaded: bool
    threads: int

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError('snapshot board dimensions must be positive')
        if self.spawn_y < 0 or self.spawn_y >= self.height:
            raise ValueError('snapshot spawn_y must be inside the board')
        if self.wall_width <= 0:
            raise ValueError('snapshot wall_width must be positive')
        if self.fps <= 0 or self.space_iterations <= 0:
            raise ValueError('snapshot physics cadence must be positive')
        if len(self.gravity) != 2:
            raise ValueError('snapshot gravity must contain exactly two values')
        if not all(math.isfinite(float(value)) for value in self.gravity):
            raise ValueError('snapshot gravity values must be finite')
        if self.queue_length <= 0:
            raise ValueError('snapshot queue_length must be positive')
        if self.create_time < 0:
            raise ValueError('snapshot create_time must be non-negative')
        if self.stable_velocity_epsilon < 0:
            raise ValueError(
                'snapshot stable_velocity_epsilon must be non-negative'
            )
        if self.stable_angular_velocity_epsilon < 0:
            raise ValueError(
                'snapshot stable_angular_velocity_epsilon must be non-negative'
            )
        if self.space_damping < 0:
            raise ValueError('snapshot space_damping must be non-negative')
        if self.collision_slop < 0:
            raise ValueError('snapshot collision_slop must be non-negative')
        if self.collision_persistence < 0:
            raise ValueError(
                'snapshot collision_persistence must be non-negative'
            )
        if self.idle_speed_threshold < 0:
            raise ValueError(
                'snapshot idle_speed_threshold must be non-negative'
            )
        if self.sleep_time_threshold < 0:
            raise ValueError(
                'snapshot sleep_time_threshold must be non-negative'
            )
        if self.threads <= 0:
            raise ValueError('snapshot threads must be positive')

        object.__setattr__(
            self,
            'gravity',
            tuple(float(value) for value in self.gravity),
        )

    @property
    def fingerprint(self):
        """返回跨进程稳定的配置指纹。

        使用字段名 JSON 而不是 Python ``hash()``，避免 Windows spawn 后随机哈希种子
        不同导致同一配置被误判为不兼容。
        """

        payload = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }
        encoded = json.dumps(
            payload,
            allow_nan=True,
            ensure_ascii=True,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('ascii')
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class FruitPhysicsSnapshot:
    """一个动态水果的语义和底层刚体/形状镜像。"""

    shape_hashid: int
    fruit_id: int
    level: int
    age_frames: int
    body_mass: float
    body_moment: float
    body_type: int
    position: tuple
    velocity: tuple
    force: tuple
    angle: float
    angular_velocity: float
    torque: float
    center_of_gravity: tuple
    is_sleeping: bool
    radius: float
    offset: tuple
    elasticity: float
    friction: float
    sensor: bool
    collision_type: int
    filter_group: int
    filter_categories: int
    filter_mask: int
    surface_velocity: tuple

    def __post_init__(self):
        if self.shape_hashid < 0:
            raise ValueError('fruit shape_hashid must be non-negative')
        if self.fruit_id <= 0:
            raise ValueError('fruit_id must be positive')
        if self.level <= 0:
            raise ValueError('fruit level must be positive')
        if self.age_frames < 0:
            raise ValueError('fruit age_frames must be non-negative')
        if self.body_mass <= 0 or self.body_moment <= 0:
            raise ValueError('fruit body mass and moment must be positive')
        if self.radius <= 0:
            raise ValueError('fruit physics radius must be positive')

        pair_names = (
            'position',
            'velocity',
            'force',
            'center_of_gravity',
            'offset',
            'surface_velocity',
        )
        for pair_name in pair_names:
            values = getattr(self, pair_name)
            if len(values) != 2:
                raise ValueError(
                    f'fruit {pair_name} must contain exactly two values'
                )
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f'fruit {pair_name} values must be finite')
            object.__setattr__(
                self,
                pair_name,
                tuple(float(value) for value in values),
            )


@dataclass(frozen=True, slots=True)
class BoundaryPhysicsSnapshot:
    """一个静态边界 Segment 的显式物理镜像。"""

    shape_hashid: int
    endpoint_a: tuple
    endpoint_b: tuple
    radius: float
    elasticity: float
    friction: float
    sensor: bool
    collision_type: int
    filter_group: int
    filter_categories: int
    filter_mask: int
    surface_velocity: tuple

    def __post_init__(self):
        if self.shape_hashid < 0:
            raise ValueError('boundary shape_hashid must be non-negative')
        if self.radius <= 0:
            raise ValueError('boundary radius must be positive')
        for pair_name in (
                'endpoint_a',
                'endpoint_b',
                'surface_velocity'):
            values = getattr(self, pair_name)
            if len(values) != 2:
                raise ValueError(
                    f'boundary {pair_name} must contain exactly two values'
                )
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f'boundary {pair_name} values must be finite')
            object.__setattr__(
                self,
                pair_name,
                tuple(float(value) for value in values),
            )


@dataclass(frozen=True, slots=True)
class EngineEpisodeSnapshot:
    """不属于 Pymunk Space、但会影响后续游戏规则和随机队列的状态。"""

    score: int
    last_score: int
    fail_count: int
    alive: bool
    step_count: int
    physics_frame: int
    next_fruit_id: int
    fruit_queue: tuple
    rng_state: tuple
    last_merge_events: tuple = field(default_factory=tuple)

    def __post_init__(self):
        if self.score < 0 or self.last_score < 0:
            raise ValueError('snapshot scores must be non-negative')
        if self.fail_count < 0:
            raise ValueError('snapshot fail_count must be non-negative')
        if self.step_count < 0 or self.physics_frame < 0:
            raise ValueError('snapshot step counters must be non-negative')
        if self.next_fruit_id <= 0:
            raise ValueError('snapshot next_fruit_id must be positive')
        if not self.fruit_queue:
            raise ValueError('snapshot fruit_queue must not be empty')
        if not isinstance(self.rng_state, tuple):
            raise TypeError('snapshot rng_state must be a tuple')


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """可跨 Windows spawn 传输的完整 HeadlessGame 物理快照。

    ``space_blob`` 使用 Pymunk 自带序列化 state 保存接触 arbiter、Shape ID 计数器
    和求解器时间戳。其余字段既保存游戏规则层状态，也作为 blob 恢复后的完整性镜像。
    """

    schema_version: int
    pymunk_version: str
    chipmunk_version: str
    config: EngineConfigSnapshot
    config_fingerprint: str
    episode: EngineEpisodeSnapshot
    fruits: tuple
    boundaries: tuple
    ball_shape_hashids: tuple
    segment_shape_hashids: tuple
    space_blob: bytes
    checksum: str

    def __post_init__(self):
        if self.schema_version <= 0:
            raise ValueError('snapshot schema_version must be positive')
        if not self.pymunk_version or not self.chipmunk_version:
            raise ValueError('snapshot physics versions must not be empty')
        if not isinstance(self.config, EngineConfigSnapshot):
            raise TypeError('snapshot config must be EngineConfigSnapshot')
        if not isinstance(self.episode, EngineEpisodeSnapshot):
            raise TypeError('snapshot episode must be EngineEpisodeSnapshot')
        if self.config_fingerprint != self.config.fingerprint:
            raise ValueError('snapshot config fingerprint is inconsistent')
        if not isinstance(self.space_blob, bytes) or not self.space_blob:
            raise ValueError('snapshot space_blob must be non-empty bytes')
        if not isinstance(self.checksum, str):
            raise TypeError('snapshot checksum must be a string')

        fruit_hashids = tuple(
            int(fruit.shape_hashid)
            for fruit in self.fruits
        )
        if len(set(fruit_hashids)) != len(fruit_hashids):
            raise ValueError('snapshot fruit shape_hashids must be unique')
        if tuple(self.ball_shape_hashids) != fruit_hashids:
            raise ValueError(
                'snapshot fruit mirrors must follow ball_shape_hashids order'
            )

        boundary_hashids = tuple(
            int(boundary.shape_hashid)
            for boundary in self.boundaries
        )
        if len(set(boundary_hashids)) != len(boundary_hashids):
            raise ValueError('snapshot boundary shape_hashids must be unique')
        if tuple(self.segment_shape_hashids) != boundary_hashids:
            raise ValueError(
                'snapshot boundary mirrors must follow '
                'segment_shape_hashids order'
            )
        if set(fruit_hashids).intersection(boundary_hashids):
            raise ValueError(
                'snapshot dynamic and static shape_hashids must be disjoint'
            )

        fruit_ids = tuple(int(fruit.fruit_id) for fruit in self.fruits)
        if len(set(fruit_ids)) != len(fruit_ids):
            raise ValueError('snapshot fruit_ids must be unique')
        if fruit_ids and self.episode.next_fruit_id <= max(fruit_ids):
            raise ValueError(
                'snapshot next_fruit_id must exceed every live fruit_id'
            )

        object.__setattr__(
            self,
            'ball_shape_hashids',
            tuple(int(value) for value in self.ball_shape_hashids),
        )
        object.__setattr__(
            self,
            'segment_shape_hashids',
            tuple(int(value) for value in self.segment_shape_hashids),
        )

    def expected_checksum(self):
        """计算除 ``checksum`` 自身之外所有快照内容的 SHA-256。"""

        payload = pickle.dumps(
            (
                self.schema_version,
                self.pymunk_version,
                self.chipmunk_version,
                self.config,
                self.config_fingerprint,
                self.episode,
                self.fruits,
                self.boundaries,
                self.ball_shape_hashids,
                self.segment_shape_hashids,
                self.space_blob,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return hashlib.sha256(payload).hexdigest()

    @property
    def checksum_valid(self):
        """返回快照内容是否仍与创建时一致。"""

        return hmac.compare_digest(self.checksum, self.expected_checksum())

    def sealed(self):
        """返回写入当前内容校验和的新快照。"""

        return replace(self, checksum=self.expected_checksum())


@dataclass(frozen=True, slots=True)
class EngineActionOutcome:
    """一次完整投放的纯数据结果，用于原动作复现比较。"""

    drop_result: DropResult
    physics_result: PhysicsResult
    final_state: GameState
    fail_count: int
    next_fruit_id: int
    rng_state: tuple


@dataclass(frozen=True, slots=True)
class OriginalActionReplayReport:
    """恢复快照后重演原动作的比较结果。"""

    matches: bool
    mismatch_codes: tuple
    max_position_error: float
    max_velocity_error: float
    max_angle_error: float
    actual_outcome: EngineActionOutcome
