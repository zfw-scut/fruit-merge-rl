"""用原 pygame 游戏画面观看 DQN 模型游玩。

本脚本采用“方案 B”：复用 `daxigua.app.Board` 的真实游戏画面，只在 RL 侧
注入一个自动控制器。游戏本体不 import RL，也不需要知道模型存在。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from daxigua_rl.graph import GraphBuilder
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.playable_adapter import board_action_candidates, board_game_state


def parse_args():
    """解析观看脚本参数。"""

    parser = argparse.ArgumentParser(description='加载 DQN checkpoint，并用原游戏画面观看模型游玩。')
    parser.add_argument('--checkpoint', required=True, help='训练脚本保存的 checkpoint 路径。')
    parser.add_argument('--device', default='cpu', help='模型运行设备，例如 cpu、cuda 或 cuda:0。')
    parser.add_argument('--action-count', type=int, default=None, help='候选动作数量；默认读取 checkpoint args。')
    parser.add_argument('--seed', type=int, default=None, help='观看时的随机种子；默认读取 checkpoint args。')
    parser.add_argument('--decision-delay-ms', type=int, default=240, help='模型选定落点后等待多久再投放，方便肉眼观察。')
    parser.add_argument('--print-actions', action='store_true', help='在终端打印每次模型选择的动作和 Q 值摘要。')
    return parser.parse_args()


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

    args = checkpoint.get('args', {})
    model = GNNQNetwork(
        hidden_dim=int(args.get('hidden_dim', 128)),
        message_layers=int(args.get('message_layers', 3)),
        activation=args.get('activation', 'silu'),
        dropout=float(args.get('dropout', 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint['online_model'])
    model.eval()
    return model


class DQNVisualController:
    """在 pygame `Board` 上执行 DQN 模型决策。"""

    def __init__(
            self,
            model,
            graph_builder,
            action_count,
            device,
            decision_delay_ms=240,
            print_actions=False):
        self.model = model
        self.graph_builder = graph_builder
        self.action_count = int(action_count)
        self.device = device
        self.decision_delay_ms = int(decision_delay_ms)
        self.print_actions = print_actions

        # pending 动作让模型先把预览水果移动到目标位置，再短暂停顿后投放。
        self.pending_action = None
        self.pending_drop_at = 0
        self.decision_count = 0

    def update(self, board):
        """每帧由 `DQNBoard` 调用，必要时选择并投放动作。"""

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

    def _choose_action(self, board):
        """根据当前原游戏局面选择一个 action candidate。"""

        state = board_game_state(board)
        actions = board_action_candidates(board, self.action_count)
        if not actions:
            raise RuntimeError('no action candidates while board is ready to drop')

        graph = self.graph_builder.build(state, actions)
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


def create_dqn_board(controller):
    """懒加载原游戏 Board，并创建带 DQN 控制器的子类实例。"""

    from daxigua.app import Board

    class DQNBoard(Board):
        """带 DQN 自动控制器的原游戏 Board。"""

        def __init__(self, controller):
            self.ai_controller = controller
            super().__init__()

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
    checkpoint_args = checkpoint.get('args', {})

    seed = args.seed if args.seed is not None else checkpoint_args.get('seed')
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))

    action_count = args.action_count or int(checkpoint_args.get('action_count', 15))
    model = build_model_from_checkpoint(checkpoint, device)
    controller = DQNVisualController(
        model=model,
        graph_builder=GraphBuilder(),
        action_count=action_count,
        device=device,
        decision_delay_ms=args.decision_delay_ms,
        print_actions=args.print_actions,
    )

    board = create_dqn_board(controller)
    board.run()


if __name__ == '__main__':
    main()
