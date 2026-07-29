"""当前 560x1120 游戏场地与观看适配器的轻量回归测试。"""

from __future__ import annotations

import os
import unittest


# 必须在导入 pygame 应用层前设置；CI 和无桌面环境无需真实窗口/音频设备。
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame as pg

from daxigua.app import Board, parse_args
from daxigua_rl.attribution import ANALYSIS_ACTION_COUNT
from daxigua_rl.playable_adapter import (
    board_action_candidates,
    board_game_state,
)


class GameGeometryTest(unittest.TestCase):
    def tearDown(self):
        pg.quit()

    def test_new_project_defaults_drive_render_physics_and_rl_adapter(self):
        args = parse_args(())
        self.assertIsNone(args.board_width)
        self.assertIsNone(args.board_height)
        self.assertIsNone(args.spawn_y)

        board = Board()
        self.assertEqual(board.RES, (560, 1120))
        self.assertEqual(board.init_y, 252)
        self.assertEqual(board.surface.get_size(), (560, 1120))
        self.assertEqual(len(board.segments), 3)

        state = board_game_state(board)
        actions = board_action_candidates(
            board,
            ANALYSIS_ACTION_COUNT,
        )
        self.assertEqual(state.geometry.width, 560)
        self.assertEqual(state.geometry.height, 1120)
        self.assertEqual(state.geometry.spawn_y, 252)
        self.assertEqual(len(actions), 21)
        self.assertLess(actions[0].drop_x, actions[-1].drop_x)

        board.drop_ready_at = 0
        board.mouse_x = board.WIDTH / 2
        board._drop_current()
        board.next_frame()
        self.assertEqual(len(board.balls), 1)
        self.assertEqual(len(board.fruits), 1)


if __name__ == '__main__':
    unittest.main()
