"""把 pygame 游戏窗口状态适配成 RL 图构建输入。

本模块属于 `daxigua_rl`，用于“观看模型实际游玩”这类场景。
它读取 `daxigua.app.Board` 暴露出来的运行时字段，但不要求游戏本体 import RL。
"""

from __future__ import annotations

import math

from daxigua.core.rules import fruit_radius
from daxigua.core.state import ActionCandidate, BoardGeometry, FruitState, GameState


STABLE_VELOCITY_EPSILON = 35.0
STABLE_ANGULAR_VELOCITY_EPSILON = 4.0


def board_action_candidates(board, action_count=15):
    """根据当前 pygame `Board` 生成离散投放动作候选。

    返回值和 `HeadlessGame.get_action_candidates()` 保持同一数据结构，
    这样 `GraphBuilder` 不需要知道当前状态来自 headless 环境还是可视化窗口。
    """

    action_count = int(action_count)
    if action_count <= 0:
        raise ValueError('action_count must be positive')

    if not getattr(board, 'waiting', False) or getattr(board, 'i', None) is None:
        return ()

    current_level = int(board.i)
    current_radius = float(fruit_radius(current_level))
    wall_width = float(getattr(board, 'wall_width', 20))
    left = wall_width + current_radius + 2
    right = float(board.WIDTH) - wall_width - current_radius - 2

    if action_count == 1:
        positions = [(left + right) / 2]
    else:
        step = (right - left) / (action_count - 1)
        positions = [left + step * index for index in range(action_count)]

    return tuple(
        ActionCandidate(
            action_index=index,
            drop_x=float(position),
            normalized_drop_x=0.0 if right == left else float((position - left) / (right - left)),
            current_level=current_level,
            current_radius=current_radius,
        )
        for index, position in enumerate(positions)
    )


def board_game_state(board):
    """把当前 pygame `Board` 转换成训练侧 `GameState` 快照。"""

    fruits = tuple(_fruit_state(board, ball, index) for index, ball in enumerate(board.balls))
    max_level = max((fruit.level for fruit in fruits), default=0)

    if fruits:
        highest_top = min(fruit.y - fruit.radius for fruit in fruits)
        max_height = float(board.HEIGHT - highest_top)
    else:
        max_height = 0.0

    playable_area = max(1.0, float(board.WIDTH) * float(board.HEIGHT - board.init_y))
    fruit_area = sum(math.pi * fruit.radius * fruit.radius for fruit in fruits)
    empty_space_ratio = max(0.0, min(1.0, 1 - fruit_area / playable_area))

    return GameState(
        board_fruits=fruits,
        fruit_queue=tuple(getattr(board, 'fruit_queue', ())),
        score=int(getattr(board, 'score', 0)),
        last_score=int(getattr(board, 'last_score', 0)),
        step_count=0,
        physics_frame=0,
        done=not bool(getattr(board, 'alive', True)),
        geometry=BoardGeometry(
            width=int(board.WIDTH),
            height=int(board.HEIGHT),
            spawn_y=int(board.init_y),
            wall_width=int(getattr(board, 'wall_width', 20)),
            floor_y=int(board.HEIGHT - getattr(board, 'wall_width', 20)),
        ),
        max_height=max_height,
        fruit_count=len(fruits),
        max_level=max_level,
        empty_space_ratio=empty_space_ratio,
    )


def _fruit_state(board, ball, index):
    """把可视化 Board 中的 pymunk shape 转成 `FruitState`。"""

    level = int(ball.collision_type)
    radius = float(fruit_radius(level))
    x, y = ball.body.position
    vx, vy = ball.body.velocity
    angular_velocity = float(ball.body.angular_velocity)
    speed = math.hypot(float(vx), float(vy))
    stable = (
        speed <= STABLE_VELOCITY_EPSILON
        and abs(angular_velocity) <= STABLE_ANGULAR_VELOCITY_EPSILON
    )

    wall_width = float(getattr(board, 'wall_width', 20))
    return FruitState(
        fruit_id=index + 1,
        level=level,
        radius=radius,
        x=float(x),
        y=float(y),
        vx=float(vx),
        vy=float(vy),
        angle=float(ball.body.angle),
        angular_velocity=angular_velocity,
        age_frames=0,
        stable=stable,
        distance_to_left_wall=float(x - (wall_width + radius)),
        distance_to_right_wall=float((board.WIDTH - wall_width - radius) - x),
        distance_to_floor=float((board.HEIGHT - wall_width - radius) - y),
        distance_to_danger_line=float((y - radius) - board.init_y),
    )
