"""用原 pygame 游戏画面观看 DQN 模型游玩。

本脚本采用“方案 B”：复用 `daxigua.app.Board` 的真实游戏画面，只在 RL 侧
注入一个自动控制器。游戏本体不 import RL，也不需要知道模型存在。
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from daxigua.config import DEFAULT_WINDOW_SIZE, FPS, SPAWN_LINE_Y
from daxigua_rl.attribution import ANALYSIS_ACTION_COUNT, StateAnalyzer
from daxigua_rl.graph import GraphBuilder
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.playable_adapter import (
    STABLE_ANGULAR_VELOCITY_EPSILON,
    STABLE_VELOCITY_EPSILON,
    board_action_candidates,
    board_game_state,
    board_is_stable,
)
from daxigua_rl.training.checkpointing import (
    extract_inference_checkpoint,
)
from daxigua_rl.training.identity import TransitionKey


def parse_args(argv=None):
    """解析观看脚本参数。"""

    parser = argparse.ArgumentParser(description='加载 DQN checkpoint，并用原游戏画面观看模型游玩。')
    parser.add_argument('--checkpoint', required=True, help='训练脚本保存的 checkpoint 路径。')
    parser.add_argument('--device', default='cpu', help='模型运行设备，例如 cpu、cuda 或 cuda:0。')
    parser.add_argument('--action-count', type=int, default=None, help='候选动作数量；默认读取 checkpoint args。')
    parser.add_argument('--seed', type=int, default=None, help='观看时的随机种子；默认读取 checkpoint args。')
    parser.add_argument(
        '--board-width',
        type=int,
        default=None,
        help='观看场地宽度；默认读取 checkpoint，缺字段时使用当前默认 560。',
    )
    parser.add_argument(
        '--board-height',
        type=int,
        default=None,
        help='观看场地高度；默认读取 checkpoint，缺字段时使用当前默认 1120。',
    )
    parser.add_argument(
        '--spawn-y',
        type=int,
        default=None,
        help='生成线 y 坐标；默认读取 checkpoint，缺字段时使用当前默认 252。',
    )
    parser.add_argument(
        '--decision-delay-ms',
        type=int,
        default=240,
        help='局面稳定且模型选定落点后等待多久再投放，方便肉眼观察。',
    )
    parser.add_argument('--print-actions', action='store_true', help='在终端打印每次模型选择的动作和 Q 值摘要。')
    return parser.parse_args(argv)


def resolve_device(device_name):
    """解析 torch 设备。"""

    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA device, but torch.cuda.is_available() is False')
    return device


def load_checkpoint(path, device):
    """加载训练 checkpoint。"""

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def build_model_from_checkpoint(checkpoint, device):
    """根据 checkpoint 里的训练参数重建 GNN-Q 模型。"""

    args, online_model_state = extract_inference_checkpoint(checkpoint)
    model = GNNQNetwork(
        hidden_dim=int(args.get('hidden_dim', 128)),
        message_layers=int(args.get('message_layers', 3)),
        activation=args.get('activation', 'silu'),
        dropout=float(args.get('dropout', 0.0)),
    ).to(device)
    model.load_state_dict(online_model_state)
    model.eval()
    return model


def checkpoint_stable_window_seconds(checkpoint_args):
    """把训练连续稳定帧换算成等效物理时间。"""

    physics_fps = int(checkpoint_args.get('physics_fps', FPS))
    stable_frames = int(checkpoint_args.get('stable_frames', 15))
    if physics_fps <= 0:
        raise ValueError('checkpoint physics_fps must be positive')
    if stable_frames <= 0:
        raise ValueError('checkpoint stable_frames must be positive')
    return float(stable_frames) / float(physics_fps)


def resolve_board_geometry(
        checkpoint_args,
        *,
        board_width=None,
        board_height=None,
        spawn_y=None):
    """按“命令行 > checkpoint > 当前默认值”解析观看场地。

    旧 checkpoint 没有保存显式几何字段时直接使用当前新默认场地；如需审计
    历史画面，调用者仍可显式传入尺寸，但项目不再维护旧尺寸为默认路径。
    """

    default_width, default_height = DEFAULT_WINDOW_SIZE
    width = (
        checkpoint_args.get('board_width', default_width)
        if board_width is None
        else board_width
    )
    height = (
        checkpoint_args.get('board_height', default_height)
        if board_height is None
        else board_height
    )
    resolved_spawn_y = (
        checkpoint_args.get('spawn_y', SPAWN_LINE_Y)
        if spawn_y is None
        else spawn_y
    )
    width = int(width)
    height = int(height)
    resolved_spawn_y = int(resolved_spawn_y)
    if width <= 0 or height <= 0:
        raise ValueError('board width and height must be positive')
    if resolved_spawn_y < 0 or resolved_spawn_y >= height:
        raise ValueError('spawn_y must be inside the board')
    return width, height, resolved_spawn_y


class BoardStabilityGate:
    """要求真实 Board 连续稳定满训练等效物理窗口后才放行。"""

    def __init__(
            self,
            stable_window_seconds,
            velocity_epsilon=STABLE_VELOCITY_EPSILON,
            angular_velocity_epsilon=(
                STABLE_ANGULAR_VELOCITY_EPSILON
            )):
        stable_window_seconds = float(stable_window_seconds)
        if (
                not math.isfinite(stable_window_seconds)
                or stable_window_seconds <= 0):
            raise ValueError(
                'stable_window_seconds must be finite and positive'
            )
        self.stable_window_seconds = stable_window_seconds
        self.velocity_epsilon = float(velocity_epsilon)
        self.angular_velocity_epsilon = float(
            angular_velocity_epsilon
        )
        self.stable_frame_count = 0
        self._topology_signature = None

    def reset(self):
        """清除当前连续稳定窗口。"""

        self.stable_frame_count = 0
        self._topology_signature = None

    def update(self, board):
        """消费一个已经完成物理步进的渲染帧并返回是否已放行。"""

        topology_signature = tuple(
            id(ball)
            for ball in getattr(board, 'balls', ())
            if ball is not None
        )
        if topology_signature != self._topology_signature:
            # 投放、合成和清场都会改变物理拓扑；即使新刚体初速度恰好很小，也必须
            # 从当前拓扑重新累计完整稳定窗口，不能沿用旧局面的稳定帧。
            self.stable_frame_count = 0
            self._topology_signature = topology_signature

        if not board_is_stable(
                board,
                velocity_epsilon=self.velocity_epsilon,
                angular_velocity_epsilon=(
                    self.angular_velocity_epsilon
                )):
            self.stable_frame_count = 0
            return False

        board_fps = float(getattr(board, 'FPS', FPS))
        if not math.isfinite(board_fps) or board_fps <= 0:
            raise ValueError('board FPS must be finite and positive')
        required_frames = max(
            1,
            int(math.ceil(
                self.stable_window_seconds * board_fps - 1e-12
            )),
        )
        self.stable_frame_count += 1
        return self.stable_frame_count >= required_frames


class DQNVisualController:
    """在 pygame `Board` 上执行 DQN 模型决策。"""

    def __init__(
            self,
            model,
            graph_builder,
            action_count,
            device,
            stable_window_seconds,
            decision_delay_ms=240,
            print_actions=False):
        self.model = model
        self.graph_builder = graph_builder
        self.action_count = int(action_count)
        self.device = device
        self.decision_delay_ms = int(decision_delay_ms)
        self.print_actions = print_actions
        self.state_analyzer = StateAnalyzer()
        self.stability_gate = BoardStabilityGate(
            stable_window_seconds=stable_window_seconds,
        )

        # pending 动作让模型先把预览水果移动到目标位置，再短暂停顿后投放。
        self.pending_action = None
        self.pending_drop_at = 0
        self.decision_count = 0

    def reset_episode(self):
        """清除跨局或手动重开时不能继续沿用的决策状态。"""

        self.pending_action = None
        self.pending_drop_at = 0
        self.stability_gate.reset()

    def update(self, board):
        """每帧由 `DQNBoard` 调用，必要时选择并投放动作。"""

        # ``Board.next_frame()`` 已在本次调用前推进一帧物理。只有全场水果按训练阈值
        # 连续稳定满等效物理窗口，才允许把当前状态标记为 stable boundary 并构图。
        if not self.stability_gate.update(board):
            self.pending_action = None
            return

        if not board._can_drop():
            self.pending_action = None
            return

        now = board.pg_time_get_ticks()

        if self.pending_action is None:
            self.pending_action = self._choose_action(board)
            self.pending_drop_at = now + self.decision_delay_ms

        # 持续把投放目标交给原游戏的输入系统，这样画面上能看到虚线和当前水果位置。
        board.input_mode = 'keyboard'
        board.aim_x = board._clamp_drop_x(self.pending_action.drop_x, self.pending_action.current_level)

        if now < self.pending_drop_at:
            return

        # 投放前把平滑位置也对齐到模型选择的落点，避免缓动尚未追到目标时提前投放。
        board.mouse_x = board.aim_x
        board._drop_current()
        self.pending_action = None
        self.stability_gate.reset()

    def _choose_action(self, board):
        """根据当前原游戏局面选择一个 action candidate。"""

        state = board_game_state(board)
        actions = board_action_candidates(board, self.action_count)
        if not actions:
            raise RuntimeError('no action candidates while board is ready to drop')

        state_analysis = None
        if len(actions) == ANALYSIS_ACTION_COUNT:
            # 观看入口没有 worker/episode 归因状态机；仅需一个当前边界内部一致的
            # key 来运行同一静态分析器。step_count 已足以保证当前水果引用对齐。
            state_analysis = self.state_analyzer.analyze(
                state,
                actions,
                TransitionKey(
                    worker_id=0,
                    episode_id=0,
                    step_index=int(state.step_count),
                ),
                stable_boundary=True,
            )
        graph = self.graph_builder.build(
            state,
            actions,
            state_analysis=state_analysis,
        )
        with torch.no_grad():
            q_values = self.model(graph).detach().cpu()

        action_offset = int(torch.argmax(q_values).item())
        action = actions[action_offset]

        self.decision_count += 1
        if self.print_actions:
            print(
                'decision={} action_offset={} drop_x={:.2f} level={} q={:+.4f} '
                'q_min={:+.4f} q_mean={:+.4f} q_max={:+.4f}'.format(
                    self.decision_count,
                    action_offset,
                    action.drop_x,
                    action.current_level,
                    float(q_values[action_offset].item()),
                    float(q_values.min().item()),
                    float(q_values.mean().item()),
                    float(q_values.max().item()),
                ),
                flush=True,
            )

        return action


def create_dqn_board(
        controller,
        *,
        board_width=None,
        board_height=None,
        spawn_y=None):
    """懒加载原游戏 Board，并创建带 DQN 控制器的子类实例。"""

    from daxigua.app import Board

    class DQNBoard(Board):
        """带 DQN 自动控制器的原游戏 Board。"""

        def __init__(self, controller):
            self.ai_controller = controller
            super().__init__(
                board_width=board_width,
                board_height=board_height,
                spawn_y=spawn_y,
            )

        def reset(self):
            """重开或终局清场时同步清除观看控制器的稳定窗口。"""

            super().reset()
            self.ai_controller.reset_episode()

        def pg_time_get_ticks(self):
            """包装 pygame ticks，避免 controller 直接 import pygame。"""

            import pygame as pg

            return pg.time.get_ticks()

        def next_frame(self):
            """先运行原游戏一帧，再让 AI 在同一个 Board 上做控制。"""

            super().next_frame()
            self.ai_controller.update(self)

    return DQNBoard(controller)


def main():
    """命令行入口。"""

    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_args, _online_model_state = (
        extract_inference_checkpoint(checkpoint)
    )

    seed = args.seed if args.seed is not None else checkpoint_args.get('seed')
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))

    action_count = (
        args.action_count
        or int(checkpoint_args.get(
            'action_count',
            ANALYSIS_ACTION_COUNT,
        ))
    )
    board_width, board_height, spawn_y = resolve_board_geometry(
        checkpoint_args,
        board_width=args.board_width,
        board_height=args.board_height,
        spawn_y=args.spawn_y,
    )
    model = build_model_from_checkpoint(checkpoint, device)
    controller = DQNVisualController(
        model=model,
        graph_builder=GraphBuilder(),
        action_count=action_count,
        device=device,
        stable_window_seconds=checkpoint_stable_window_seconds(
            checkpoint_args
        ),
        decision_delay_ms=args.decision_delay_ms,
        print_actions=args.print_actions,
    )

    board = create_dqn_board(
        controller,
        board_width=board_width,
        board_height=board_height,
        spawn_y=spawn_y,
    )
    board.run()


if __name__ == '__main__':
    main()
