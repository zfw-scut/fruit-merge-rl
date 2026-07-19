"""无渲染游戏引擎。

`HeadlessGame` 是后续训练环境优先使用的游戏本体接口。它只负责规则和物理：
- 不创建 pygame 窗口。
- 不处理键盘、鼠标或音效。
- 不返回 pymunk 内部对象给外部调用者。
- 通过 `GameState`、`ActionCandidate` 等纯数据结构暴露状态。

RL 代码应该通过这个模块访问游戏，而不是直接读取 `daxigua.app.Board`。
"""

import math
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
    DropResult,
    FruitState,
    GameState,
    MergeEvent,
    PhysicsResult,
)


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
        return segment_shape

    def _setup_collision_handler(self):
        """注册同级水果碰撞合成回调。"""

        def post_solve_merge(arbiter, space, data):
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

        for level in range(1, MAX_FRUIT_LEVEL):
            if hasattr(self.space, 'add_collision_handler'):
                self.space.add_collision_handler(level, level).post_solve = post_solve_merge
            else:
                self.space.on_collision(level, level, post_solve=post_solve_merge)

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

        return DropResult(
            dropped_level=level,
            drop_x=drop_x,
            fruit_id=fruit_id,
            queue_before=queue_before,
            queue_after=tuple(self.fruit_queue),
        )

    def advance_physics(self, max_frames=None, until_stable=True, stable_frames=15):
        """推进物理世界，直到稳定、失败或达到最大帧数。"""

        frame_limit = max_frames or self.fps * 6
        frames_simulated = 0
        stable_count = 0
        score_before = self.score
        self._last_merge_events = []

        # 每次 step 都会调用 advance_physics，直到稳定或失败才返回。每次 step 只允许一次 advance_physics。
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

        stable = self._is_stable()
        truncated = frames_simulated >= frame_limit and not stable and not self.is_done()

        return PhysicsResult(
            frames_simulated=frames_simulated,
            stable=stable,
            done=self.is_done(),
            truncated=truncated,
            score_delta=self.score - score_before,
            merge_events=tuple(self._last_merge_events),
        )

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
            highest_top = min(fruit.y - fruit.radius for fruit in fruits)
            max_height = self.height - highest_top
        else:
            max_height = 0.0

        playable_area = max(1.0, self.width * (self.height - self.spawn_y))
        fruit_area = sum(math.pi * fruit.radius * fruit.radius for fruit in fruits)
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
            distance_to_left_wall=float(x - (self.wall_width + radius)),
            distance_to_right_wall=float((self.width - self.wall_width - radius) - x),
            distance_to_floor=float((self.height - self.wall_width - radius) - y),
            distance_to_danger_line=float((y - radius) - self.spawn_y),
        )
