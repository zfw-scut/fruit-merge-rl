"""无渲染游戏引擎。

`HeadlessGame` 是后续训练环境优先使用的游戏本体接口。它只负责规则和物理：
- 不创建 pygame 窗口。
- 不处理键盘、鼠标或音效。
- 不返回 pymunk 内部对象给外部调用者。
- 通过 `GameState`、`ActionCandidate` 等纯数据结构暴露状态。

RL 代码应该通过这个模块访问游戏，而不是直接读取 `daxigua.app.Board`。
"""

import math
import pickle
import random
from dataclasses import dataclass

import pymunk

from ..config import DEFAULT_WINDOW_SIZE, FPS, SPAWN_LINE_Y
from .rules import (
    FRUIT_QUEUE_LENGTH,
    MAX_FRUIT_LEVEL,
    dropped_fruit_physics_radius,
    fruit_mass,
    fruit_radius,
    merge_score,
    merged_fruit_physics_radius,
    random_spawn_level,
)
from .state import (
    ActionCandidate,
    BoardGeometry,
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
    MergeEvent,
    OriginalActionReplayReport,
    PhysicsResult,
)


_SNAPSHOT_PHASE_READY = 'ready'
_SNAPSHOT_PHASE_ACTION_PENDING = 'action_pending'
_SNAPSHOT_PHASE_ADVANCING = 'advancing'
_SNAPSHOT_PHASE_UNSTABLE = 'unstable'
_SNAPSHOT_PHASE_TERMINAL = 'terminal'
_SNAPSHOT_PHASE_INVALID = 'invalid'

# Pymunk 7.3 在反序列化 Space 时会按加入顺序重新分配 cpShape hash ID，尤其在原
# Space 已删除过水果、ID 存在空洞时，恢复后的私有 ``_hashid`` 不再等于原值。
# Shape 的普通 Python 属性会随 pickle 保留，因此用这个 token 保存原 hash 身份，
# 既能重建 balls/meta 顺序，也不去修改已加入 broadphase 的底层 hash ID。
# 不使用前导下划线：Pymunk PickleMixin 会忽略未声明的私有属性，但会保留普通
# Python 扩展属性。
_SNAPSHOT_SHAPE_TOKEN_ATTR = 'daxigua_snapshot_shape_hashid'


def _post_solve_merge(arbiter, space, data):
    """Pymunk 可 pickle 的模块顶层碰撞入口。

    Pymunk 会序列化 collision handler 的回调函数，但不会序列化 handler ``data``。
    因此回调必须位于模块顶层，恢复 ``Space`` 后再由 ``_setup_collision_handler()``
    把新的 HeadlessGame 实例写回 data，避免闭包继续引用原游戏对象。
    """

    game = data.get('game') if isinstance(data, dict) else None
    if not isinstance(game, HeadlessGame):
        raise RuntimeError(
            'merge collision handler is not bound to a HeadlessGame'
        )
    if game.space is not space:
        raise RuntimeError(
            'merge collision handler is bound to a different Pymunk Space'
        )
    game._handle_post_solve_merge(arbiter, space)


@dataclass
class _FruitRuntime:
    """引擎内部运行时元数据。

    这是私有类型，只用于把 pymunk shape 和稳定的水果状态关联起来。
    外部接口只应该看到 `FruitState`。
    """

    fruit_id: int
    level: int
    age_frames: int = 0


class HeadlessGame:
    """无渲染合成大西瓜核心环境。"""

    def __init__(
            self,
            width=None,
            height=None,
            spawn_y=SPAWN_LINE_Y,
            fps=FPS,
            space_iterations=32,
            gravity=(0, 1800),
            queue_length=FRUIT_QUEUE_LENGTH,
            create_time=2.0,
            seed=None):
        # 固定场地几何。默认和手动游戏窗口保持一致。
        default_width, default_height = DEFAULT_WINDOW_SIZE
        self.width = int(width or default_width)
        self.height = int(height or default_height)
        self.spawn_y = int(spawn_y)
        self.wall_width = 20
        self.fps = int(fps or FPS)
        if self.fps <= 0:
            raise ValueError('fps must be positive')

        # `space.iterations` 是 Chipmunk/Pymunk 每个物理步求解约束的迭代次数。
        # 数值越高碰撞和堆叠越精细，但每帧耗时也更高；训练 fast 模式会显式降低它。
        self.space_iterations = int(space_iterations)
        if self.space_iterations <= 0:
            raise ValueError('space_iterations must be positive')

        self.gravity = gravity
        self.queue_length = queue_length
        self.create_time = create_time

        # 稳定判定阈值。速度低于这些阈值并持续若干帧，就认为一次投放后的物理过程结束。
        self.stable_velocity_epsilon = 35.0
        self.stable_angular_velocity_epsilon = 4.0

        # 使用独立随机数生成器，避免训练环境和 UI 随机数互相影响。
        self.rng = random.Random(seed)

        # reset 会创建 pymunk Space 和所有运行时状态。
        self.reset(seed=seed)

    def reset(self, seed=None, fruit_queue=None):
        """重置游戏并返回初始状态。"""

        if seed is not None:
            self.rng.seed(seed)

        # 每次 reset 重新创建 Space，比逐个清理旧对象更可靠，也更适合训练反复重置。
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.space.iterations = self.space_iterations
        self.space.damping = 0.995

        self.balls = []
        self.segments = []
        self._fruit_meta = {}
        self._next_fruit_id = 1
        self._last_merge_events = []
        self.lock = False

        self.score = 0
        self.last_score = 0
        self.fail_count = 0
        self.alive = True
        self.step_count = 0
        self.physics_frame = 0

        self.fruit_queue = list(fruit_queue or [])
        self._fill_fruit_queue()

        self._init_segments()
        self._setup_collision_handler()
        # reset 产生空场地稳定边界，可以立刻保存下一动作之前的快照。
        self._snapshot_phase = _SNAPSHOT_PHASE_READY
        return self.get_state()

    def _fill_fruit_queue(self):
        """把待投放水果队列补足到固定长度。"""

        while len(self.fruit_queue) < self.queue_length:
            self.fruit_queue.append(random_spawn_level(self.rng))

    def _advance_fruit_queue(self):
        """投放后推进 q0 到 q3 队列。"""

        if self.fruit_queue:
            self.fruit_queue.pop(0)
        self._fill_fruit_queue()

    def _init_segments(self):
        """创建左墙、底板和右墙。"""

        borders = (
            ((0, 0), (0, self.height)),
            ((0, self.height), (self.width, self.height)),
            ((self.width, self.height), (self.width, 0)),
        )
        for from_, to_ in borders:
            self.segments.append(self._create_segment(from_, to_, 20))

    def _create_segment(self, from_, to_, thickness):
        """创建静态边界线段。"""

        segment_shape = pymunk.Segment(self.space.static_body, from_, to_, thickness)
        segment_shape.friction = 0.6
        self.space.add(segment_shape)
        setattr(
            segment_shape,
            _SNAPSHOT_SHAPE_TOKEN_ATTR,
            int(segment_shape._hashid),
        )
        return segment_shape

    def _setup_collision_handler(self):
        """注册同级水果碰撞合成回调。"""

        for level in range(1, MAX_FRUIT_LEVEL):
            if hasattr(self.space, 'add_collision_handler'):
                # 兼容 Pymunk 6.x：旧接口把 handler.data 字典传给回调。
                handler = self.space.add_collision_handler(level, level)
                handler.post_solve = _post_solve_merge
                handler.data['game'] = self
            else:
                # Pymunk 7.x 的 data 本身不会进入 Space pickle。restore 后再次调用
                # 本方法即可把新实例安全绑定到已恢复的 handler。
                self.space.on_collision(
                    level,
                    level,
                    post_solve=_post_solve_merge,
                    data={'game': self},
                )

    def _handle_post_solve_merge(self, arbiter, space):
        """执行一次同级碰撞合成；由模块顶层可序列化回调转入。"""

        if self.lock:
            return

        self.lock = True
        try:
            shape_a, shape_b = arbiter.shapes
            new_level = shape_a.collision_type + 1

            if new_level > MAX_FRUIT_LEVEL:
                return

            meta_a = self._meta_for(shape_a)
            meta_b = self._meta_for(shape_b)
            if not meta_a or not meta_b:
                return

            x1, y1 = shape_a.body.position
            x2, y2 = shape_b.body.position
            if y1 > y2:
                x, y = x1, y1
            else:
                x, y = x2, y2

            source_ids = (meta_a.fruit_id, meta_b.fruit_id)
            self._remove_ball(shape_a)
            self._remove_ball(shape_b)

            new_ball = self._create_ball(
                x,
                y,
                new_level,
                physics_radius=merged_fruit_physics_radius(new_level))

            score_delta = merge_score(new_level)
            if score_delta:
                self.last_score = self.score
                self.score += score_delta

            self._last_merge_events.append(
                MergeEvent(
                    new_level=new_level,
                    x=float(x),
                    y=float(y),
                    score_delta=score_delta,
                    source_ids=source_ids,
                    new_fruit_id=self._meta_for(new_ball).fruit_id,
                )
            )
        finally:
            self.lock = False

    def _create_ball(self, x, y, level, physics_radius=None):
        """创建一个物理水果，并记录内部元数据。"""

        radius = physics_radius if physics_radius is not None else dropped_fruit_physics_radius(level)
        mass = fruit_mass(level)
        moment = pymunk.moment_for_circle(mass, 0, radius)

        body = pymunk.Body(mass, moment)
        body.position = x, y

        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.18
        shape.friction = 0.88
        shape.collision_type = level

        self.space.add(body, shape)
        # 在 collision callback 内新增 shape 时，Pymunk 因 Space 仍被锁定会延迟
        # 真正 add，此刻 ``shape._hashid`` 暂时还是 0。水果 ID 与创建顺序同样严格
        # 单调，因此用“静态边界数 + fruit_id - 1”得到原本将分配的稳定 token。
        setattr(
            shape,
            _SNAPSHOT_SHAPE_TOKEN_ATTR,
            int(len(self.segments) + self._next_fruit_id - 1),
        )
        self.balls.append(shape)
        self._fruit_meta[id(shape)] = _FruitRuntime(self._next_fruit_id, level)
        self._next_fruit_id += 1
        return shape

    def _remove_ball(self, shape):
        """从物理世界和运行时索引中移除一个水果。"""

        if shape in self.balls:
            self.balls.remove(shape)
        self._fruit_meta.pop(id(shape), None)

        try:
            self.space.remove(shape, shape.body)
        except Exception:
            # 碰撞回调中可能遇到已经被同一轮合成移除的 shape，容错即可。
            pass

    def _meta_for(self, shape):
        """读取 shape 对应的内部水果元数据。"""

        return self._fruit_meta.get(id(shape))

    def current_level(self):
        """返回当前 q0 水果等级。"""

        self._fill_fruit_queue()
        return self.fruit_queue[0]

    def clamp_drop_x(self, x, fruit_level=None):
        """把投放横坐标限制在当前水果可合法投放的范围内。"""

        level = fruit_level or self.current_level()
        radius = fruit_radius(level)
        left = self.wall_width + radius + 2
        right = self.width - self.wall_width - radius - 2
        return max(left, min(right, float(x)))

    def get_action_candidates(self, k=15):
        """生成离散投放动作候选。"""

        if k <= 0:
            raise ValueError('action candidate count must be positive')

        current_level = self.current_level()
        current_radius = fruit_radius(current_level)
        current_physics_radius = dropped_fruit_physics_radius(current_level)
        left = self.wall_width + current_radius + 2
        right = self.width - self.wall_width - current_radius - 2

        if k == 1:
            positions = [(left + right) / 2]
        else:
            step = (right - left) / (k - 1)
            positions = [left + step * index for index in range(k)]

        return [
            ActionCandidate(
                action_index=index,
                drop_x=position,
                normalized_drop_x=0.0 if right == left else (position - left) / (right - left),
                current_level=current_level,
                current_radius=current_radius,
                current_physics_radius=current_physics_radius,
            )
            for index, position in enumerate(positions)
        ]

    def drop_at(self, x):
        """在指定横坐标投放当前 q0 水果。"""

        if self.is_done():
            raise RuntimeError('cannot drop fruit after game is done')

        queue_before = tuple(self.fruit_queue)
        level = self.current_level()
        drop_x = self.clamp_drop_x(x, level)

        ball = self._create_ball(drop_x, self.spawn_y, level)
        ball.body.velocity = (0, 80)
        fruit_id = self._meta_for(ball).fruit_id

        self._advance_fruit_queue()
        self.step_count += 1
        # 从创建水果到 advance_physics 返回稳定结果之间，Space 内仍有未结算动作，
        # 此时即使瞬时速度很小也不能创建可用于反事实的稳定边界快照。
        self._snapshot_phase = _SNAPSHOT_PHASE_ACTION_PENDING

        return DropResult(
            dropped_level=level,
            drop_x=drop_x,
            fruit_id=fruit_id,
            queue_before=queue_before,
            queue_after=tuple(self.fruit_queue),
        )

    def advance_physics(self, max_frames=None, until_stable=True, stable_frames=15):
        """推进物理世界，直到稳定、失败或达到最大帧数。"""

        frame_limit = self.fps * 6 if max_frames is None else int(max_frames)
        stable_frames = int(stable_frames)
        if frame_limit <= 0:
            raise ValueError('max_frames must be positive')
        if until_stable and stable_frames <= 0:
            raise ValueError('stable_frames must be positive')

        frames_simulated = 0
        stable_count = 0
        score_before = self.score
        self._last_merge_events = []
        self._snapshot_phase = _SNAPSHOT_PHASE_ADVANCING

        try:
            # 每次 step 都会调用 advance_physics，直到稳定或失败才返回。每次 step
            # 只允许一次 advance_physics。
            while frames_simulated < frame_limit and not self.is_done():
                self.space.step(1 / self.fps)
                self.physics_frame += 1
                frames_simulated += 1

                for ball in self.balls:
                    meta = self._meta_for(ball)
                    if meta:
                        meta.age_frames += 1

                if self.check_fail():
                    break

                if until_stable:
                    if self._is_stable():
                        stable_count += 1
                        if stable_count >= stable_frames:
                            break
                    else:
                        stable_count = 0

            # `until_stable=True` 要求连续满足 stable_frames 帧；不能用最后一帧的
            # 瞬时速度代替，否则恰好在帧上限首次静止时会漏报 truncated。
            stable = (
                stable_count >= stable_frames
                if until_stable
                else self._is_stable()
            )
            truncated = (
                until_stable
                and frames_simulated >= frame_limit
                and not stable
                and not self.is_done()
            )

            result = PhysicsResult(
                frames_simulated=frames_simulated,
                stable=stable,
                done=self.is_done(),
                truncated=truncated,
                score_delta=self.score - score_before,
                merge_events=tuple(self._last_merge_events),
            )
        except Exception:
            # 异常中断可能留下延迟增删或部分求解状态，之后必须先 reset。
            self._snapshot_phase = _SNAPSHOT_PHASE_INVALID
            raise

        if result.done:
            self._snapshot_phase = _SNAPSHOT_PHASE_TERMINAL
        elif (
                until_stable
                and result.stable
                and not result.truncated):
            self._snapshot_phase = _SNAPSHOT_PHASE_READY
        else:
            self._snapshot_phase = _SNAPSHOT_PHASE_UNSTABLE
        return result

    def _is_stable(self):
        """判断当前所有水果是否基本静止。"""

        for ball in self.balls:
            vx, vy = ball.body.velocity
            speed = math.hypot(vx, vy)
            if speed > self.stable_velocity_epsilon:
                return False
            if abs(ball.body.angular_velocity) > self.stable_angular_velocity_epsilon:
                return False
        return True

    def check_fail(self):
        """检测是否有水果持续越过死亡线。"""

        exists_over_line = False

        if self.balls:
            for ball in self.balls[:-1]:
                if int(ball.body.position[1]) < self.spawn_y:
                    self.fail_count += 1
                    exists_over_line = True
                    break

        if exists_over_line:
            if self.fail_count > self.fps * self.create_time:
                self.alive = False
                return True
            return False

        self.fail_count = 0
        return False

    def is_done(self):
        """返回当前局是否结束。"""

        return not self.alive

    def get_state(self):
        """返回训练友好的纯数据状态快照。"""

        fruits = tuple(self._fruit_state(ball) for ball in self.balls if self._meta_for(ball))
        max_level = max((fruit.level for fruit in fruits), default=0)

        if fruits:
            highest_top = min(fruit.y - fruit.physics_radius for fruit in fruits)
            max_height = self.height - highest_top
        else:
            max_height = 0.0

        playable_area = max(1.0, self.width * (self.height - self.spawn_y))
        fruit_area = sum(
            math.pi * fruit.physics_radius * fruit.physics_radius
            for fruit in fruits
        )
        empty_space_ratio = max(0.0, min(1.0, 1 - fruit_area / playable_area))

        return GameState(
            board_fruits=fruits,
            fruit_queue=tuple(self.fruit_queue),
            score=self.score,
            last_score=self.last_score,
            step_count=self.step_count,
            physics_frame=self.physics_frame,
            done=self.is_done(),
            geometry=BoardGeometry(
                width=self.width,
                height=self.height,
                spawn_y=self.spawn_y,
                wall_width=self.wall_width,
                floor_y=self.height - self.wall_width,
            ),
            max_height=max_height,
            fruit_count=len(fruits),
            max_level=max_level,
            empty_space_ratio=empty_space_ratio,
        )

    def _fruit_state(self, ball):
        """把内部 pymunk shape 转换为公开 FruitState。"""

        meta = self._meta_for(ball)
        level = meta.level
        radius = fruit_radius(level)
        physics_radius = float(ball.radius)
        x, y = ball.body.position
        vx, vy = ball.body.velocity
        stable = (
            math.hypot(vx, vy) <= self.stable_velocity_epsilon
            and abs(ball.body.angular_velocity) <= self.stable_angular_velocity_epsilon
        )

        return FruitState(
            fruit_id=meta.fruit_id,
            level=level,
            radius=float(radius),
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
            angle=float(ball.body.angle),
            angular_velocity=float(ball.body.angular_velocity),
            age_frames=meta.age_frames,
            stable=stable,
            distance_to_left_wall=float(x - (self.wall_width + physics_radius)),
            distance_to_right_wall=float(
                (self.width - self.wall_width - physics_radius) - x
            ),
            distance_to_floor=float(
                (self.height - self.wall_width - physics_radius) - y
            ),
            distance_to_danger_line=float((y - physics_radius) - self.spawn_y),
            physics_radius=physics_radius,
        )

    def _current_config_snapshot(self, space=None):
        """读取当前实例和指定 Space 的完整物理配置。"""

        space = self.space if space is None else space
        return EngineConfigSnapshot(
            width=int(self.width),
            height=int(self.height),
            spawn_y=int(self.spawn_y),
            wall_width=int(self.wall_width),
            fps=int(self.fps),
            space_iterations=int(space.iterations),
            gravity=tuple(float(value) for value in space.gravity),
            queue_length=int(self.queue_length),
            create_time=float(self.create_time),
            stable_velocity_epsilon=float(
                self.stable_velocity_epsilon
            ),
            stable_angular_velocity_epsilon=float(
                self.stable_angular_velocity_epsilon
            ),
            space_damping=float(space.damping),
            collision_slop=float(space.collision_slop),
            collision_bias=float(space.collision_bias),
            collision_persistence=int(space.collision_persistence),
            idle_speed_threshold=float(space.idle_speed_threshold),
            sleep_time_threshold=float(space.sleep_time_threshold),
            threaded=bool(space.threaded),
            threads=int(space.threads),
        )

    @staticmethod
    def _snapshot_shape_hashid(shape):
        """返回跨 Space pickle 保留的 Shape 身份 token。"""

        value = getattr(
            shape,
            _SNAPSHOT_SHAPE_TOKEN_ATTR,
            shape._hashid,
        )
        return int(value)

    @staticmethod
    def _fruit_physics_snapshot(shape, meta):
        """把一个动态 Circle 和游戏元数据转换为纯数据镜像。"""

        body = shape.body
        shape_filter = shape.filter
        return FruitPhysicsSnapshot(
            shape_hashid=HeadlessGame._snapshot_shape_hashid(shape),
            fruit_id=int(meta.fruit_id),
            level=int(meta.level),
            age_frames=int(meta.age_frames),
            body_mass=float(body.mass),
            body_moment=float(body.moment),
            body_type=int(body.body_type),
            position=tuple(float(value) for value in body.position),
            velocity=tuple(float(value) for value in body.velocity),
            force=tuple(float(value) for value in body.force),
            angle=float(body.angle),
            angular_velocity=float(body.angular_velocity),
            torque=float(body.torque),
            center_of_gravity=tuple(
                float(value)
                for value in body.center_of_gravity
            ),
            is_sleeping=bool(body.is_sleeping),
            radius=float(shape.radius),
            offset=tuple(float(value) for value in shape.offset),
            elasticity=float(shape.elasticity),
            friction=float(shape.friction),
            sensor=bool(shape.sensor),
            collision_type=int(shape.collision_type),
            filter_group=int(shape_filter.group),
            filter_categories=int(shape_filter.categories),
            filter_mask=int(shape_filter.mask),
            surface_velocity=tuple(
                float(value)
                for value in shape.surface_velocity
            ),
        )

    @staticmethod
    def _boundary_physics_snapshot(shape):
        """把一个静态 Segment 转换为纯数据镜像。"""

        shape_filter = shape.filter
        return BoundaryPhysicsSnapshot(
            shape_hashid=HeadlessGame._snapshot_shape_hashid(shape),
            endpoint_a=tuple(float(value) for value in shape.a),
            endpoint_b=tuple(float(value) for value in shape.b),
            radius=float(shape.radius),
            elasticity=float(shape.elasticity),
            friction=float(shape.friction),
            sensor=bool(shape.sensor),
            collision_type=int(shape.collision_type),
            filter_group=int(shape_filter.group),
            filter_categories=int(shape_filter.categories),
            filter_mask=int(shape_filter.mask),
            surface_velocity=tuple(
                float(value)
                for value in shape.surface_velocity
            ),
        )

    def _validate_snapshot_boundary(self):
        """确保当前恰好处于可恢复的动作前稳定边界。"""

        if self._snapshot_phase != _SNAPSHOT_PHASE_READY:
            raise RuntimeError(
                'EngineSnapshot requires a reset or fully stable '
                f'action boundary, got phase={self._snapshot_phase!r}'
            )
        if self.is_done():
            raise RuntimeError(
                'cannot capture EngineSnapshot after terminal state'
            )
        if self.lock:
            raise RuntimeError(
                'cannot capture EngineSnapshot inside collision callback'
            )
        if bool(getattr(self.space, '_locked', False)):
            raise RuntimeError(
                'cannot capture EngineSnapshot while Pymunk Space is locked'
            )

        pending_internal_names = (
            '_add_later',
            '_remove_later',
            '_post_step_callbacks',
        )
        for field_name in pending_internal_names:
            if bool(getattr(self.space, field_name, ())):
                raise RuntimeError(
                    'cannot capture EngineSnapshot with pending Pymunk '
                    f'work in {field_name}'
                )

        # `_bodies_to_check` 只保存下次 step 前要检查质量合法性的 Body；Space
        # restore 会正常重新生成它，不是尚未提交的物理操作。`_removed_shapes`
        # 也只是让已从底层 Space 删除的 Python wrapper 多存活一帧。因此二者都
        # 不属于快照门禁，真正待提交的 add/remove/post-step 队列仍在上面拒绝。

        if self.space.constraints:
            raise RuntimeError(
                'EngineSnapshot does not support unexpected constraints'
            )
        if self.space.threaded:
            raise RuntimeError(
                'EngineSnapshot currently requires non-threaded Pymunk'
            )
        if not math.isinf(float(self.space.sleep_time_threshold)):
            # Pymunk 的 Body pickle 不保存 idle/sleep group 状态。当前游戏本来就
            # 禁用 sleeping，因此遇到未来自定义模式时宁可拒绝，也不制造伪标签。
            raise RuntimeError(
                'EngineSnapshot requires sleeping to remain disabled'
            )

        if len(self.fruit_queue) != int(self.queue_length):
            raise RuntimeError(
                'fruit_queue length must equal queue_length before snapshot'
            )
        if len(self.balls) != len(self._fruit_meta):
            raise RuntimeError(
                'fruit runtime metadata is out of sync with balls'
            )

        for shape in self.balls:
            if not isinstance(shape, pymunk.Circle):
                raise RuntimeError(
                    'EngineSnapshot only supports Circle fruit shapes'
                )
            if self._meta_for(shape) is None:
                raise RuntimeError(
                    'every fruit shape must have runtime metadata'
                )
            if shape.body.is_sleeping:
                raise RuntimeError(
                    'EngineSnapshot cannot capture sleeping fruit bodies'
                )
            if (
                    shape.body._velocity_func is not None
                    or shape.body._position_func is not None):
                raise RuntimeError(
                    'EngineSnapshot does not support custom body '
                    'integration callbacks'
                )

        if not all(
                isinstance(shape, pymunk.Segment)
                for shape in self.segments):
            raise RuntimeError(
                'EngineSnapshot only supports Segment board boundaries'
            )

        expected_shapes = set(self.balls).union(self.segments)
        if set(self.space.shapes) != expected_shapes:
            raise RuntimeError(
                'Pymunk Space contains shapes not owned by HeadlessGame'
            )

    def capture_snapshot(self, *, canonicalize=True):
        """捕获一个可用于原动作重演和稀疏反事实的完整物理快照。

        Pymunk 自带的 Space 序列化 state 会保留 cached arbiters、接触点、Shape ID
        counter、timestamp 和 current timestep。手工只保存位置/速度会丢失这些
        同帧碰撞排序信息，因此 ``space_blob`` 是恢复的真实来源，显式镜像负责审计。

        ``canonicalize=True`` 会用第一份快照原地重建一次当前 Space，再返回重建后的
        第二份快照。Pymunk 没有公开空间索引树的序列化接口；若真实分支继续使用原
        broadphase、反事实分支使用重建 broadphase，密集水果堆会迅速出现数值分叉。
        稳定边界上的这次规范化不改变公开 GameState，却保证随后真实动作和所有恢复
        分支从同一内部表示出发。只做离线检查时可显式传 ``canonicalize=False``。
        """

        self._validate_snapshot_boundary()
        visible_state_before = self.get_state()
        config = self._current_config_snapshot()
        fruits = tuple(
            self._fruit_physics_snapshot(shape, self._meta_for(shape))
            for shape in self.balls
        )
        boundaries = tuple(
            self._boundary_physics_snapshot(shape)
            for shape in self.segments
        )
        episode = EngineEpisodeSnapshot(
            score=int(self.score),
            last_score=int(self.last_score),
            fail_count=int(self.fail_count),
            alive=bool(self.alive),
            step_count=int(self.step_count),
            physics_frame=int(self.physics_frame),
            next_fruit_id=int(self._next_fruit_id),
            fruit_queue=tuple(int(level) for level in self.fruit_queue),
            rng_state=self.rng.getstate(),
            last_merge_events=tuple(self._last_merge_events),
        )
        try:
            space_blob = pickle.dumps(
                # 不能直接 pickle Space：Pymunk 7.3 的 __setstate__ 会先 add
                # 所有 Shape，最后才恢复 shapeIDCounter，导致有删除空洞的 hash ID
                # 被压紧。保存原生 state 后，restore 可在每个 Shape add 前设置计数器。
                self.space.__getstate__(),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except Exception as exc:
            raise RuntimeError(
                'failed to serialize Pymunk Space for EngineSnapshot'
            ) from exc

        snapshot = EngineSnapshot(
            schema_version=ENGINE_SNAPSHOT_SCHEMA_VERSION,
            pymunk_version=str(pymunk.version),
            chipmunk_version=str(pymunk.chipmunk_version),
            config=config,
            config_fingerprint=config.fingerprint,
            episode=episode,
            fruits=fruits,
            boundaries=boundaries,
            ball_shape_hashids=tuple(
                fruit.shape_hashid
                for fruit in fruits
            ),
            segment_shape_hashids=tuple(
                boundary.shape_hashid
                for boundary in boundaries
            ),
            space_blob=space_blob,
            checksum='',
        ).sealed()
        if not canonicalize:
            return snapshot

        self.restore_snapshot(snapshot)
        if self.get_state() != visible_state_before:
            raise RuntimeError(
                'EngineSnapshot canonicalization changed public GameState'
            )
        return self.capture_snapshot(canonicalize=False)

    @staticmethod
    def _validate_snapshot_integrity(snapshot):
        """在读取 Pymunk blob 前验证纯数据头、版本和内容校验和。"""

        if not isinstance(snapshot, EngineSnapshot):
            raise TypeError('snapshot must be EngineSnapshot')
        if snapshot.schema_version != ENGINE_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                'unsupported EngineSnapshot schema_version: '
                f'{snapshot.schema_version}'
            )
        if snapshot.pymunk_version != str(pymunk.version):
            raise ValueError(
                'EngineSnapshot Pymunk version mismatch: '
                f'{snapshot.pymunk_version!r} != {pymunk.version!r}'
            )
        if snapshot.chipmunk_version != str(pymunk.chipmunk_version):
            raise ValueError(
                'EngineSnapshot Chipmunk version mismatch: '
                f'{snapshot.chipmunk_version!r} != '
                f'{pymunk.chipmunk_version!r}'
            )
        if snapshot.config_fingerprint != snapshot.config.fingerprint:
            raise ValueError(
                'EngineSnapshot config fingerprint is invalid'
            )
        if not snapshot.checksum_valid:
            raise ValueError('EngineSnapshot checksum mismatch')
        if snapshot.config.threaded:
            raise ValueError(
                'threaded EngineSnapshot payloads are unsupported'
            )
        if not math.isinf(snapshot.config.sleep_time_threshold):
            raise ValueError(
                'EngineSnapshot payload requires unsupported sleeping state'
            )
        if (
                len(snapshot.episode.fruit_queue)
                != snapshot.config.queue_length):
            raise ValueError(
                'EngineSnapshot fruit_queue length does not match config'
            )

    @staticmethod
    def _shape_map(space):
        """按 Pymunk 持久化 Shape hash ID 建立恢复索引。"""

        shapes_by_hashid = {}
        for shape in space.shapes:
            shape_hashid = HeadlessGame._snapshot_shape_hashid(shape)
            if shape_hashid in shapes_by_hashid:
                raise ValueError(
                    'restored Pymunk Space has duplicate shape hash IDs'
                )
            shapes_by_hashid[shape_hashid] = shape
        return shapes_by_hashid

    @staticmethod
    def _restore_space_blob(space_blob):
        """恢复 Pymunk Space，并保留带删除空洞的原始 Shape hash ID。

        Pymunk 7.3 原生 ``Space.__setstate__`` 的顺序是：

        ``add(all shapes) -> restore shapeIDCounter``。

        ``cpSpaceAddShape`` 会在 add 时重新分配 hash ID，所以经历过合成删除的 Space
        会把例如 7 压成 3，进而改变 broadphase 和同帧碰撞顺序。这里仍使用 Pymunk
        自己生成的 state，只把 shapes 拆成单个 add，并在每次 add 前设置对应 token。
        cached arbiter 在所有 Shape 身份恢复后才由原生逻辑加入。
        """

        try:
            space_state = pickle.loads(space_blob)
        except Exception as exc:
            raise ValueError(
                'EngineSnapshot contains an unreadable Pymunk Space state'
            ) from exc
        if (
                not isinstance(space_state, dict)
                or not {'init', 'general', 'custom', 'special'}.issubset(
                    space_state)):
            raise ValueError(
                'EngineSnapshot space_blob is not a Pymunk Space state'
            )

        rewritten_special = []
        saw_shapes = False
        for field_name, value in space_state['special']:
            if field_name != 'shapes':
                rewritten_special.append((field_name, value))
                continue

            saw_shapes = True
            for shape in value:
                shape_hashid = HeadlessGame._snapshot_shape_hashid(shape)
                rewritten_special.append(
                    ('shapeIDCounter', shape_hashid)
                )
                rewritten_special.append(('shapes', [shape]))
        if not saw_shapes:
            raise ValueError(
                'EngineSnapshot Pymunk state does not contain shapes'
            )
        space_state['special'] = rewritten_special

        try:
            restored_space = pymunk.Space.__new__(pymunk.Space)
            restored_space.__setstate__(space_state)
        except Exception as exc:
            raise ValueError(
                'EngineSnapshot Pymunk Space state could not be restored'
            ) from exc
        return restored_space

    def restore_snapshot(self, snapshot):
        """把当前同配置 HeadlessGame 原地恢复到 ``snapshot``。

        该方法先在局部变量中完成反序列化和镜像核对，只有所有检查通过后才替换当前
        Space，尽量避免损坏快照让现有游戏停在半恢复状态。
        """

        self._validate_snapshot_integrity(snapshot)
        current_config = self._current_config_snapshot()
        if current_config.fingerprint != snapshot.config_fingerprint:
            raise ValueError(
                'cannot restore EngineSnapshot into a different engine config'
            )

        # 快照只在本项目内部创建和消费；pickle 不应接受外部不可信输入。
        restored_space = self._restore_space_blob(snapshot.space_blob)
        if self._current_config_snapshot(
                space=restored_space) != snapshot.config:
            raise ValueError(
                'restored Pymunk Space does not match snapshot config mirror'
            )

        shapes_by_hashid = self._shape_map(restored_space)
        expected_hashids = set(snapshot.ball_shape_hashids).union(
            snapshot.segment_shape_hashids
        )
        if set(shapes_by_hashid) != expected_hashids:
            raise ValueError(
                'restored Pymunk shapes do not match snapshot shape order'
            )

        try:
            restored_balls = [
                shapes_by_hashid[shape_hashid]
                for shape_hashid in snapshot.ball_shape_hashids
            ]
            restored_segments = [
                shapes_by_hashid[shape_hashid]
                for shape_hashid in snapshot.segment_shape_hashids
            ]
        except KeyError as exc:
            raise ValueError(
                'EngineSnapshot references a missing Pymunk shape'
            ) from exc
        if not all(
                isinstance(shape, pymunk.Circle)
                for shape in restored_balls):
            raise ValueError(
                'EngineSnapshot fruit order contains a non-Circle shape'
            )
        if not all(
                isinstance(shape, pymunk.Segment)
                for shape in restored_segments):
            raise ValueError(
                'EngineSnapshot boundary order contains a non-Segment shape'
            )
        if len(restored_space.bodies) != len(restored_balls):
            raise ValueError(
                'restored Pymunk body count does not match live fruits'
            )

        restored_runtime_by_hashid = {
            fruit.shape_hashid: _FruitRuntime(
                fruit_id=fruit.fruit_id,
                level=fruit.level,
                age_frames=fruit.age_frames,
            )
            for fruit in snapshot.fruits
        }
        for shape, expected in zip(restored_balls, snapshot.fruits):
            actual = self._fruit_physics_snapshot(
                shape,
                restored_runtime_by_hashid[expected.shape_hashid],
            )
            if actual != expected:
                raise ValueError(
                    'restored fruit body does not match snapshot mirror: '
                    f'fruit_id={expected.fruit_id}'
                )
        for shape, expected in zip(
                restored_segments,
                snapshot.boundaries):
            if self._boundary_physics_snapshot(shape) != expected:
                raise ValueError(
                    'restored boundary does not match snapshot mirror: '
                    f'shape_hashid={expected.shape_hashid}'
                )

        restored_rng = random.Random()
        try:
            restored_rng.setstate(snapshot.episode.rng_state)
        except Exception as exc:
            raise ValueError(
                'EngineSnapshot contains an invalid RNG state'
            ) from exc

        # 所有校验已经完成，从这里开始一次性提交恢复结果。
        config = snapshot.config
        self.width = int(config.width)
        self.height = int(config.height)
        self.spawn_y = int(config.spawn_y)
        self.wall_width = int(config.wall_width)
        self.fps = int(config.fps)
        self.space_iterations = int(config.space_iterations)
        self.gravity = tuple(config.gravity)
        self.queue_length = int(config.queue_length)
        self.create_time = float(config.create_time)
        self.stable_velocity_epsilon = float(
            config.stable_velocity_epsilon
        )
        self.stable_angular_velocity_epsilon = float(
            config.stable_angular_velocity_epsilon
        )
        self.space = restored_space
        self.balls = list(restored_balls)
        self.segments = list(restored_segments)
        self._fruit_meta = {
            id(shape): restored_runtime_by_hashid[
                self._snapshot_shape_hashid(shape)
            ]
            for shape in self.balls
        }
        self._next_fruit_id = int(snapshot.episode.next_fruit_id)
        self._last_merge_events = list(
            snapshot.episode.last_merge_events
        )
        self.lock = False
        self.score = int(snapshot.episode.score)
        self.last_score = int(snapshot.episode.last_score)
        self.fail_count = int(snapshot.episode.fail_count)
        self.alive = bool(snapshot.episode.alive)
        self.step_count = int(snapshot.episode.step_count)
        self.physics_frame = int(snapshot.episode.physics_frame)
        self.fruit_queue = list(snapshot.episode.fruit_queue)
        self.rng = restored_rng
        self._snapshot_phase = _SNAPSHOT_PHASE_READY

        # Space pickle 保留回调函数，但刻意不保留 handler data。必须在新实例上重新
        # 写入 game 引用；否则第一次同级碰撞会得到未绑定回调。
        self._setup_collision_handler()
        return self

    @classmethod
    def from_snapshot(cls, snapshot):
        """创建配置相同的新实例并恢复快照。"""

        cls._validate_snapshot_integrity(snapshot)
        config = snapshot.config
        game = cls(
            width=config.width,
            height=config.height,
            spawn_y=config.spawn_y,
            fps=config.fps,
            space_iterations=config.space_iterations,
            gravity=config.gravity,
            queue_length=config.queue_length,
            create_time=config.create_time,
            seed=0,
        )
        game.wall_width = int(config.wall_width)
        game.stable_velocity_epsilon = float(
            config.stable_velocity_epsilon
        )
        game.stable_angular_velocity_epsilon = float(
            config.stable_angular_velocity_epsilon
        )
        game.space.damping = float(config.space_damping)
        game.space.collision_slop = float(config.collision_slop)
        game.space.collision_bias = float(config.collision_bias)
        game.space.collision_persistence = int(
            config.collision_persistence
        )
        game.space.idle_speed_threshold = float(
            config.idle_speed_threshold
        )
        game.space.sleep_time_threshold = float(
            config.sleep_time_threshold
        )
        if int(game.space.threads) != int(config.threads):
            raise ValueError(
                'current Pymunk runtime cannot reproduce snapshot threads'
            )
        return game.restore_snapshot(snapshot)

    def execute_action(
            self,
            drop_x,
            *,
            max_frames=None,
            stable_frames=15):
        """执行一次完整投放并返回用于确定性比较的纯数据结果。"""

        drop_result = self.drop_at(drop_x)
        physics_result = self.advance_physics(
            max_frames=max_frames,
            until_stable=True,
            stable_frames=stable_frames,
        )
        return EngineActionOutcome(
            drop_result=drop_result,
            physics_result=physics_result,
            final_state=self.get_state(),
            fail_count=int(self.fail_count),
            next_fruit_id=int(self._next_fruit_id),
            rng_state=self.rng.getstate(),
        )

    @classmethod
    def replay_action(
            cls,
            snapshot,
            drop_x,
            *,
            max_frames=None,
            stable_frames=15):
        """从快照分支重演一个动作，不修改创建快照的原游戏。"""

        game = cls.from_snapshot(snapshot)
        return game.execute_action(
            drop_x,
            max_frames=max_frames,
            stable_frames=stable_frames,
        )

    @staticmethod
    def compare_action_outcomes(
            expected,
            actual,
            *,
            position_tolerance=5e-2,
            velocity_tolerance=1e-3,
            angle_tolerance=1e-6):
        """比较真实动作与恢复分支，返回保守的原动作复现报告。"""

        # Pymunk 7.3 恢复 cached arbiter 后，同一连锁的瞬时 MergeEvent 坐标实测会有
        # 约 0.03 像素差异；默认 0.05 只容忍这种亚像素偏差，ID、等级、顺序、得分、
        # 帧数和终止语义仍要求完全一致。

        if not isinstance(expected, EngineActionOutcome):
            raise TypeError('expected must be EngineActionOutcome')
        if not isinstance(actual, EngineActionOutcome):
            raise TypeError('actual must be EngineActionOutcome')
        position_tolerance = float(position_tolerance)
        velocity_tolerance = float(velocity_tolerance)
        angle_tolerance = float(angle_tolerance)
        if min(
                position_tolerance,
                velocity_tolerance,
                angle_tolerance) < 0:
            raise ValueError('replay tolerances must be non-negative')

        mismatch_codes = []

        def add_mismatch(code):
            if code not in mismatch_codes:
                mismatch_codes.append(code)

        max_position_error = 0.0
        max_velocity_error = 0.0
        max_angle_error = 0.0

        expected_drop = expected.drop_result
        actual_drop = actual.drop_result
        if (
                expected_drop.dropped_level,
                expected_drop.fruit_id,
                expected_drop.queue_before,
                expected_drop.queue_after,
        ) != (
                actual_drop.dropped_level,
                actual_drop.fruit_id,
                actual_drop.queue_before,
                actual_drop.queue_after,
        ):
            add_mismatch('drop_semantics')
        drop_x_error = abs(
            float(expected_drop.drop_x) - float(actual_drop.drop_x)
        )
        max_position_error = max(max_position_error, drop_x_error)
        if drop_x_error > position_tolerance:
            add_mismatch('drop_x')

        expected_physics = expected.physics_result
        actual_physics = actual.physics_result
        if (
                expected_physics.frames_simulated,
                expected_physics.stable,
                expected_physics.done,
                expected_physics.truncated,
                expected_physics.score_delta,
        ) != (
                actual_physics.frames_simulated,
                actual_physics.stable,
                actual_physics.done,
                actual_physics.truncated,
                actual_physics.score_delta,
        ):
            add_mismatch('physics_result')

        if (
                len(expected_physics.merge_events)
                != len(actual_physics.merge_events)):
            add_mismatch('merge_event_count')
        for expected_event, actual_event in zip(
                expected_physics.merge_events,
                actual_physics.merge_events):
            if (
                    expected_event.new_level,
                    expected_event.score_delta,
                    expected_event.source_ids,
                    expected_event.new_fruit_id,
            ) != (
                    actual_event.new_level,
                    actual_event.score_delta,
                    actual_event.source_ids,
                    actual_event.new_fruit_id,
            ):
                add_mismatch('merge_event_semantics')
            event_position_error = max(
                abs(float(expected_event.x) - float(actual_event.x)),
                abs(float(expected_event.y) - float(actual_event.y)),
            )
            max_position_error = max(
                max_position_error,
                event_position_error,
            )
            if event_position_error > position_tolerance:
                add_mismatch('merge_event_position')

        expected_state = expected.final_state
        actual_state = actual.final_state
        exact_state_fields = (
            'fruit_queue',
            'score',
            'last_score',
            'step_count',
            'physics_frame',
            'done',
            'geometry',
            'fruit_count',
            'max_level',
        )
        if any(
                getattr(expected_state, field_name)
                != getattr(actual_state, field_name)
                for field_name in exact_state_fields):
            add_mismatch('final_state_semantics')
        if (
                abs(
                    float(expected_state.max_height)
                    - float(actual_state.max_height)
                ) > position_tolerance
                or abs(
                    float(expected_state.empty_space_ratio)
                    - float(actual_state.empty_space_ratio)
                ) > 1e-12):
            add_mismatch('final_state_derived')

        expected_order = tuple(
            fruit.fruit_id
            for fruit in expected_state.board_fruits
        )
        actual_order = tuple(
            fruit.fruit_id
            for fruit in actual_state.board_fruits
        )
        if expected_order != actual_order:
            add_mismatch('fruit_order')
        expected_fruits = {
            fruit.fruit_id: fruit
            for fruit in expected_state.board_fruits
        }
        actual_fruits = {
            fruit.fruit_id: fruit
            for fruit in actual_state.board_fruits
        }
        if set(expected_fruits) != set(actual_fruits):
            add_mismatch('fruit_ids')
        for fruit_id in sorted(set(expected_fruits).intersection(
                actual_fruits)):
            expected_fruit = expected_fruits[fruit_id]
            actual_fruit = actual_fruits[fruit_id]
            if (
                    expected_fruit.level,
                    expected_fruit.radius,
                    expected_fruit.physics_radius,
                    expected_fruit.age_frames,
                    expected_fruit.stable,
            ) != (
                    actual_fruit.level,
                    actual_fruit.radius,
                    actual_fruit.physics_radius,
                    actual_fruit.age_frames,
                    actual_fruit.stable,
            ):
                add_mismatch('fruit_semantics')

            position_error = max(
                abs(float(expected_fruit.x) - float(actual_fruit.x)),
                abs(float(expected_fruit.y) - float(actual_fruit.y)),
            )
            velocity_error = max(
                abs(float(expected_fruit.vx) - float(actual_fruit.vx)),
                abs(float(expected_fruit.vy) - float(actual_fruit.vy)),
            )
            angle_error = max(
                abs(
                    float(expected_fruit.angle)
                    - float(actual_fruit.angle)
                ),
                abs(
                    float(expected_fruit.angular_velocity)
                    - float(actual_fruit.angular_velocity)
                ),
            )
            max_position_error = max(
                max_position_error,
                position_error,
            )
            max_velocity_error = max(
                max_velocity_error,
                velocity_error,
            )
            max_angle_error = max(max_angle_error, angle_error)
            if position_error > position_tolerance:
                add_mismatch('fruit_position')
            if velocity_error > velocity_tolerance:
                add_mismatch('fruit_velocity')
            if angle_error > angle_tolerance:
                add_mismatch('fruit_angle')

        if expected.fail_count != actual.fail_count:
            add_mismatch('fail_count')
        if expected.next_fruit_id != actual.next_fruit_id:
            add_mismatch('next_fruit_id')
        if expected.rng_state != actual.rng_state:
            add_mismatch('rng_state')

        return OriginalActionReplayReport(
            matches=not mismatch_codes,
            mismatch_codes=tuple(mismatch_codes),
            max_position_error=float(max_position_error),
            max_velocity_error=float(max_velocity_error),
            max_angle_error=float(max_angle_error),
            actual_outcome=actual,
        )

    @classmethod
    def replay_and_compare_original_action(
            cls,
            snapshot,
            expected_outcome,
            drop_x,
            *,
            max_frames=None,
            stable_frames=15,
            position_tolerance=5e-2,
            velocity_tolerance=1e-3,
            angle_tolerance=1e-6):
        """恢复、重演并一次性完成原动作确定性门禁。"""

        actual_outcome = cls.replay_action(
            snapshot,
            drop_x,
            max_frames=max_frames,
            stable_frames=stable_frames,
        )
        return cls.compare_action_outcomes(
            expected_outcome,
            actual_outcome,
            position_tolerance=position_tolerance,
            velocity_tolerance=velocity_tolerance,
            angle_tolerance=angle_tolerance,
        )
