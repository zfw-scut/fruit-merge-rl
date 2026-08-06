"""使用 Pymunk 的单环境行为参考实现。

本模块只用于差分测试、调试和回退，不进入批量训练热路径。
它按当前 ``rules.py`` 重新实现已确认的规则，不依赖旧分支文件。
"""

from dataclasses import dataclass
import math
import random

import pymunk

from daxigua.core import (
    ActionCandidate,
    BoardGeometry,
    DropResult,
    FruitState,
    GameState,
    MAX_FRUIT_LEVEL,
    MergeEvent,
    PhysicsResult,
    dropped_fruit_physics_radius,
    fruit_mass,
    fruit_radius,
    is_mergeable_level,
    merge_position,
    merge_score,
    merge_target_level,
    merged_fruit_physics_radius,
    random_spawn_level,
)

from .config import SimulatorConfig


@dataclass(slots=True)
class _FruitRuntime:
    fruit_id: int
    level: int
    age_frames: int = 0


def _post_solve_merge(arbiter, space, data):
    del space
    data['game']._handle_merge(arbiter)


class PymunkReferenceGame:
    """与 CUDA 后端使用同一规则和公开状态契约的参考环境。"""

    def __init__(self, config=None, *, seed=None):
        self.config = config or SimulatorConfig(use_cuda_extension=False)
        if not isinstance(self.config, SimulatorConfig):
            raise TypeError('config must be SimulatorConfig')
        self.rng = random.Random(seed)
        self.reset(seed=seed)

    def reset(self, seed=None, fruit_queue=None):
        if seed is not None:
            self.rng.seed(seed)
        self.space = pymunk.Space()
        self.space.gravity = (0.0, float(self.config.gravity_y))
        self.space.iterations = 32
        self.space.damping = float(self.config.damping)
        self.balls = []
        self.segments = []
        self._fruit_meta = {}
        self._next_fruit_id = 1
        self._last_merge_events = []
        self.lock = False
        self.score = 0
        self.last_score = 0
        self.fail_frames = 0
        self.alive = True
        self.step_count = 0
        self.physics_frame = 0
        self.fruit_queue = list(fruit_queue or ())
        if any(level < 1 or level > 5 for level in self.fruit_queue):
            raise ValueError('fruit_queue levels must be in [1, 5]')
        if len(self.fruit_queue) > self.config.queue_length:
            raise ValueError('fruit_queue is longer than queue_length')
        self._fill_queue()
        self._init_walls()
        self._setup_collision_handlers()
        return self.get_state()

    def _fill_queue(self):
        while len(self.fruit_queue) < self.config.queue_length:
            self.fruit_queue.append(random_spawn_level(self.rng))

    def _advance_queue(self):
        self.fruit_queue.pop(0)
        self._fill_queue()

    def _init_walls(self):
        borders = (
            ((0, 0), (0, self.config.board_height)),
            ((0, self.config.board_height),
             (self.config.board_width, self.config.board_height)),
            ((self.config.board_width, self.config.board_height),
             (self.config.board_width, 0)),
        )
        for start, end in borders:
            shape = pymunk.Segment(
                self.space.static_body,
                start,
                end,
                self.config.wall_width,
            )
            shape.friction = self.config.wall_friction
            self.space.add(shape)
            self.segments.append(shape)

    def _setup_collision_handlers(self):
        for level in range(1, MAX_FRUIT_LEVEL + 1):
            if not is_mergeable_level(level):
                continue
            if hasattr(self.space, 'on_collision'):
                self.space.on_collision(
                    level,
                    level,
                    post_solve=_post_solve_merge,
                    data={'game': self},
                )
            else:
                handler = self.space.add_collision_handler(level, level)
                handler.post_solve = _post_solve_merge
                handler.data['game'] = self

    def _create_ball(
            self,
            x,
            y,
            level,
            *,
            physics_radius=None,
            fruit_id=None,
            age_frames=0):
        resolved_id = self._next_fruit_id if fruit_id is None else int(fruit_id)
        if resolved_id <= 0:
            raise ValueError('fruit_id must be positive')
        if any(
                meta.fruit_id == resolved_id
                for meta in self._fruit_meta.values()):
            raise ValueError('fruit_id must be unique')
        radius = float(
            dropped_fruit_physics_radius(level)
            if physics_radius is None
            else physics_radius
        )
        mass = float(fruit_mass(level))
        body = pymunk.Body(
            mass,
            pymunk.moment_for_circle(mass, 0.0, radius),
        )
        body.position = float(x), float(y)
        shape = pymunk.Circle(body, radius)
        shape.elasticity = self.config.fruit_elasticity
        shape.friction = self.config.fruit_friction
        shape.collision_type = level
        self.space.add(body, shape)
        self.balls.append(shape)
        self._fruit_meta[id(shape)] = _FruitRuntime(
            fruit_id=resolved_id,
            level=level,
            age_frames=int(age_frames),
        )
        self._next_fruit_id = max(self._next_fruit_id, resolved_id + 1)
        return shape

    def _remove_ball(self, shape):
        if shape in self.balls:
            self.balls.remove(shape)
        self._fruit_meta.pop(id(shape), None)
        try:
            self.space.remove(shape, shape.body)
        except Exception:
            pass

    def remove_fruit(self, fruit_id):
        """按稳定水果 ID 从当前物理世界移除水果。"""
        if isinstance(fruit_id, bool) or not isinstance(fruit_id, int):
            raise TypeError('fruit_id must be an integer')
        if fruit_id <= 0:
            raise ValueError('fruit_id must be positive')
        for shape in tuple(self.balls):
            meta = self._fruit_meta.get(id(shape))
            if meta is not None and meta.fruit_id == fruit_id:
                self._remove_ball(shape)
                return True
        return False

    def _handle_merge(self, arbiter):
        if self.lock:
            return
        self.lock = True
        try:
            shape_a, shape_b = arbiter.shapes
            meta_a = self._fruit_meta.get(id(shape_a))
            meta_b = self._fruit_meta.get(id(shape_b))
            if meta_a is None or meta_b is None:
                return
            source_level = meta_a.level
            if source_level != meta_b.level:
                return
            position = merge_position(shape_a.body.position, shape_b.body.position)
            source_ids = (meta_a.fruit_id, meta_b.fruit_id)
            body_a = shape_a.body
            body_b = shape_b.body
            momentum_x = (
                body_a.mass * body_a.velocity.x
                + body_b.mass * body_b.velocity.x
            )
            momentum_y = (
                body_a.mass * body_a.velocity.y
                + body_b.mass * body_b.velocity.y
            )
            radius_a = (
                body_a.position.x - position[0],
                body_a.position.y - position[1],
            )
            radius_b = (
                body_b.position.x - position[0],
                body_b.position.y - position[1],
            )
            linear_momentum_a = (
                body_a.mass * body_a.velocity.x,
                body_a.mass * body_a.velocity.y,
            )
            linear_momentum_b = (
                body_b.mass * body_b.velocity.x,
                body_b.mass * body_b.velocity.y,
            )
            angular_momentum = (
                body_a.moment * body_a.angular_velocity
                + body_b.moment * body_b.angular_velocity
                + radius_a[0] * linear_momentum_a[1]
                - radius_a[1] * linear_momentum_a[0]
                + radius_b[0] * linear_momentum_b[1]
                - radius_b[1] * linear_momentum_b[0]
            )
            self._remove_ball(shape_a)
            self._remove_ball(shape_b)
            target_level = merge_target_level(source_level)
            new_fruit_id = None
            if target_level is not None:
                new_ball = self._create_ball(
                    *position,
                    target_level,
                    physics_radius=merged_fruit_physics_radius(target_level),
                )
                new_ball.body.velocity = (
                    momentum_x / new_ball.body.mass,
                    momentum_y / new_ball.body.mass,
                )
                new_ball.body.angular_velocity = (
                    angular_momentum / new_ball.body.moment
                )
                new_fruit_id = self._fruit_meta[id(new_ball)].fruit_id
            score_delta = merge_score(source_level)
            self.last_score = self.score
            self.score += score_delta
            self._last_merge_events.append(
                MergeEvent(
                    new_level=target_level,
                    x=float(position[0]),
                    y=float(position[1]),
                    score_delta=score_delta,
                    source_ids=source_ids,
                    new_fruit_id=new_fruit_id,
                )
            )
        finally:
            self.lock = False

    def current_level(self):
        self._fill_queue()
        return self.fruit_queue[0]

    def action_candidates(self):
        level = self.current_level()
        radius = fruit_radius(level)
        physics_radius = dropped_fruit_physics_radius(level)
        left = self.config.wall_width + radius + 2
        right = self.config.board_width - self.config.wall_width - radius - 2
        return tuple(
            ActionCandidate(
                action_index=index,
                drop_x=left + (right - left) * index
                / (self.config.action_count - 1),
                normalized_drop_x=index / (self.config.action_count - 1),
                current_level=level,
                current_radius=radius,
                current_physics_radius=physics_radius,
            )
            for index in range(self.config.action_count)
        )

    def spawn_fruit(
            self,
            level,
            x,
            *,
            y=None,
            vx=0.0,
            vy=80.0,
            angle=0.0,
            angular_velocity=0.0,
            physics_radius=None,
            fruit_id=None,
            age_frames=0,
            count_step=True):
        """立即向当前物理世界加入水果，不等待已有水果稳定。"""

        if isinstance(level, bool) or not isinstance(level, int):
            raise TypeError('level must be an integer')
        if level < 1 or level > MAX_FRUIT_LEVEL:
            raise ValueError('level is outside the supported range')
        if len(self.balls) >= self.config.max_fruits:
            raise RuntimeError('fruit capacity is exhausted')
        if not self.alive:
            raise RuntimeError('cannot spawn fruit after game is done')
        radius = float(fruit_radius(level))
        left = self.config.wall_width + radius + 2.0
        right = self.config.board_width - self.config.wall_width - radius - 2.0
        drop_x = max(left, min(right, float(x)))
        drop_y = self.config.spawn_y if y is None else float(y)
        shape = self._create_ball(
            drop_x,
            drop_y,
            level,
            physics_radius=physics_radius,
            fruit_id=fruit_id,
            age_frames=age_frames,
        )
        shape.body.velocity = float(vx), float(vy)
        shape.body.angle = float(angle)
        shape.body.angular_velocity = float(angular_velocity)
        if count_step:
            self.step_count += 1
        return self._fruit_meta[id(shape)].fruit_id

    def replace_state(
            self,
            fruits,
            *,
            fruit_queue,
            score=0,
            last_score=None,
            step_count=0,
            physics_frame=0):
        """用公开水果快照替换当前世界，供调试会话和存档恢复使用。"""

        fruits = tuple(fruits)
        if len(fruits) > self.config.max_fruits:
            raise ValueError('fruit count exceeds simulator capacity')
        self.reset(fruit_queue=fruit_queue)
        for fruit in fruits:
            if not isinstance(fruit, FruitState):
                raise TypeError('fruits must contain FruitState values')
            self.spawn_fruit(
                fruit.level,
                fruit.x,
                y=fruit.y,
                vx=fruit.vx,
                vy=fruit.vy,
                angle=fruit.angle,
                angular_velocity=fruit.angular_velocity,
                physics_radius=fruit.physics_radius,
                fruit_id=fruit.fruit_id,
                age_frames=fruit.age_frames,
                count_step=False,
            )
        self.score = int(score)
        self.last_score = self.score if last_score is None else int(last_score)
        self.step_count = int(step_count)
        self.physics_frame = int(physics_frame)
        return self.get_state()

    def drop(self, action_index):
        if not self.alive:
            raise RuntimeError('cannot drop fruit after game is done')
        candidates = self.action_candidates()
        if action_index < 0 or action_index >= len(candidates):
            raise IndexError('action_index out of range')
        queue_before = tuple(self.fruit_queue)
        level = self.current_level()
        drop_x = candidates[action_index].drop_x
        fruit_id = self.spawn_fruit(level, drop_x, count_step=False)
        self._advance_queue()
        self.step_count += 1
        return DropResult(
            dropped_level=level,
            drop_x=float(drop_x),
            fruit_id=fruit_id,
            queue_before=queue_before,
            queue_after=tuple(self.fruit_queue),
        )

    def _is_stable(self):
        return all(
            math.hypot(*shape.body.velocity)
            <= self.config.stable_velocity_epsilon
            and abs(shape.body.angular_velocity)
            <= self.config.stable_angular_velocity_epsilon
            for shape in self.balls
        )

    def is_stable(self):
        """返回当前世界是否已经静止，不推进物理时间。"""

        return self._is_stable()

    def _check_fail(self):
        over_line = any(
            int(shape.body.position.y) < self.config.spawn_y
            for shape in self.balls[:-1]
        )
        self.fail_frames = self.fail_frames + 1 if over_line else 0
        if self.fail_frames > self.config.danger_frame_limit:
            self.alive = False
        return not self.alive

    def _advance_one_frame(self):
        self.space.step(self.config.dt)
        self.physics_frame += 1
        for shape in self.balls:
            meta = self._fruit_meta.get(id(shape))
            if meta is not None:
                meta.age_frames += 1
        self._check_fail()

    def advance_frame(self):
        """只推进一个固定物理帧，供实时交互循环持续调用。"""

        score_before = self.score
        self._last_merge_events = []
        frames = 0
        if self.alive:
            self._advance_one_frame()
            frames = 1
        return PhysicsResult(
            frames_simulated=frames,
            stable=self._is_stable(),
            done=not self.alive,
            truncated=False,
            score_delta=self.score - score_before,
            merge_events=tuple(self._last_merge_events),
            settle_timeout=False,
        )

    def advance_physics(self):
        frames = 0
        stable_count = 0
        score_before = self.score
        self._last_merge_events = []
        while frames < self.config.max_physics_frames and self.alive:
            self._advance_one_frame()
            frames += 1
            if not self.alive:
                break
            if self._is_stable():
                stable_count += 1
                if stable_count >= self.config.stable_frames:
                    break
            else:
                stable_count = 0
        stable = stable_count >= self.config.stable_frames
        settle_timeout = (
            frames >= self.config.max_physics_frames
            and not stable
            and self.alive
        )
        return PhysicsResult(
            frames_simulated=frames,
            stable=stable,
            done=not self.alive,
            truncated=False,
            score_delta=self.score - score_before,
            merge_events=tuple(self._last_merge_events),
            settle_timeout=settle_timeout,
        )

    def step(self, action_index):
        drop_result = self.drop(action_index)
        physics_result = self.advance_physics()
        return self.get_state(), drop_result, physics_result

    def get_state(self):
        fruits = tuple(self._fruit_state(shape) for shape in self.balls)
        highest_top = min(
            (fruit.y - fruit.physics_radius for fruit in fruits),
            default=float(self.config.board_height),
        )
        playable_area = max(
            1.0,
            self.config.board_width
            * (self.config.board_height - self.config.spawn_y),
        )
        fruit_area = sum(
            math.pi * fruit.physics_radius ** 2 for fruit in fruits
        )
        return GameState(
            board_fruits=fruits,
            fruit_queue=tuple(self.fruit_queue),
            score=self.score,
            last_score=self.last_score,
            step_count=self.step_count,
            physics_frame=self.physics_frame,
            done=not self.alive,
            geometry=BoardGeometry(
                width=self.config.board_width,
                height=self.config.board_height,
                spawn_y=self.config.spawn_y,
                wall_width=self.config.wall_width,
                floor_y=self.config.board_height - self.config.wall_width,
            ),
            max_height=(
                0.0
                if not fruits
                else self.config.board_height - highest_top
            ),
            fruit_count=len(fruits),
            max_level=max((fruit.level for fruit in fruits), default=0),
            empty_space_ratio=max(
                0.0, min(1.0, 1.0 - fruit_area / playable_area)
            ),
        )

    def _fruit_state(self, shape):
        meta = self._fruit_meta[id(shape)]
        x, y = shape.body.position
        vx, vy = shape.body.velocity
        physics_radius = float(shape.radius)
        return FruitState(
            fruit_id=meta.fruit_id,
            level=meta.level,
            radius=float(fruit_radius(meta.level)),
            x=float(x),
            y=float(y),
            vx=float(vx),
            vy=float(vy),
            angle=float(shape.body.angle),
            angular_velocity=float(shape.body.angular_velocity),
            age_frames=meta.age_frames,
            stable=(
                math.hypot(vx, vy) <= self.config.stable_velocity_epsilon
                and abs(shape.body.angular_velocity)
                <= self.config.stable_angular_velocity_epsilon
            ),
            distance_to_left_wall=float(
                x - (self.config.wall_width + physics_radius)
            ),
            distance_to_right_wall=float(
                self.config.board_width
                - self.config.wall_width
                - physics_radius
                - x
            ),
            distance_to_floor=float(
                self.config.board_height
                - self.config.wall_width
                - physics_radius
                - y
            ),
            distance_to_danger_line=float(
                y - physics_radius - self.config.spawn_y
            ),
            physics_radius=physics_radius,
        )
