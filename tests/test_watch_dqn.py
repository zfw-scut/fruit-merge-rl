import math
import unittest
from types import SimpleNamespace

from daxigua_rl.playable_adapter import board_is_stable
from daxigua_rl.scripts.watch_dqn import (
    BoardStabilityGate,
    DQNVisualController,
    checkpoint_stable_window_seconds,
)


class _FakeBody:
    def __init__(self, velocity=(0.0, 0.0), angular_velocity=0.0):
        self.velocity = velocity
        self.angular_velocity = angular_velocity


class _FakeBall:
    def __init__(self, velocity=(0.0, 0.0), angular_velocity=0.0):
        self.body = _FakeBody(
            velocity=velocity,
            angular_velocity=angular_velocity,
        )


class _FakeBoard:
    FPS = 120

    def __init__(self):
        self.balls = [_FakeBall()]
        self.lock = False
        self.can_drop = True
        self.ticks = 0
        self.input_mode = 'mouse'
        self.aim_x = 0.0
        self.mouse_x = 0.0
        self.drop_calls = 0

    def _can_drop(self):
        return self.can_drop

    def pg_time_get_ticks(self):
        return self.ticks

    @staticmethod
    def _clamp_drop_x(drop_x, _level):
        return float(drop_x)

    def _drop_current(self):
        self.drop_calls += 1
        self.can_drop = False


class _RecordingController(DQNVisualController):
    def __init__(self, *, stable_window_seconds, decision_delay_ms=0):
        super().__init__(
            model=None,
            graph_builder=None,
            action_count=15,
            device=None,
            stable_window_seconds=stable_window_seconds,
            decision_delay_ms=decision_delay_ms,
        )
        self.choose_calls = 0

    def _choose_action(self, _board):
        self.choose_calls += 1
        return SimpleNamespace(drop_x=123.0, current_level=1)


class PlayableStabilityTests(unittest.TestCase):
    def test_board_stability_matches_training_motion_thresholds(self):
        board = _FakeBoard()

        self.assertTrue(board_is_stable(board))
        board.balls[0].body.velocity = (35.0, 0.0)
        board.balls[0].body.angular_velocity = -4.0
        self.assertTrue(board_is_stable(board))

        board.balls.append(_FakeBall(velocity=(35.01, 0.0)))
        self.assertFalse(board_is_stable(board))
        board.balls[-1].body.velocity = (0.0, 0.0)
        board.balls[-1].body.angular_velocity = 4.01
        self.assertFalse(board_is_stable(board))

        board.balls[-1].body.angular_velocity = math.nan
        self.assertFalse(board_is_stable(board))
        board.balls.clear()
        self.assertTrue(board_is_stable(board))
        board.lock = True
        self.assertFalse(board_is_stable(board))

    def test_fast30_stable_window_becomes_24_visual_frames(self):
        self.assertEqual(
            checkpoint_stable_window_seconds({
                'physics_fps': 30,
                'stable_frames': 6,
            }),
            0.2,
        )
        board = _FakeBoard()
        gate = BoardStabilityGate(stable_window_seconds=0.2)

        for _ in range(23):
            self.assertFalse(gate.update(board))
        self.assertTrue(gate.update(board))

    def test_unstable_frame_and_topology_change_restart_window(self):
        board = _FakeBoard()
        gate = BoardStabilityGate(stable_window_seconds=3 / board.FPS)

        self.assertFalse(gate.update(board))
        self.assertFalse(gate.update(board))
        board.balls[0].body.velocity = (36.0, 0.0)
        self.assertFalse(gate.update(board))
        board.balls[0].body.velocity = (0.0, 0.0)
        self.assertFalse(gate.update(board))
        self.assertFalse(gate.update(board))
        self.assertTrue(gate.update(board))

        board.balls[0] = _FakeBall()
        self.assertFalse(gate.update(board))
        self.assertFalse(gate.update(board))
        self.assertTrue(gate.update(board))


class VisualControllerTimingTests(unittest.TestCase):
    def test_cooldown_does_not_prevent_stability_accumulation(self):
        board = _FakeBoard()
        board.can_drop = False
        controller = _RecordingController(
            stable_window_seconds=2 / board.FPS,
        )

        controller.update(board)
        controller.update(board)
        self.assertEqual(controller.choose_calls, 0)
        self.assertEqual(board.drop_calls, 0)

        board.can_drop = True
        controller.update(board)
        self.assertEqual(controller.choose_calls, 1)
        self.assertEqual(board.drop_calls, 1)

    def test_motion_during_preview_cancels_and_recomputes_action(self):
        board = _FakeBoard()
        controller = _RecordingController(
            stable_window_seconds=1 / board.FPS,
            decision_delay_ms=100,
        )

        controller.update(board)
        self.assertEqual(controller.choose_calls, 1)
        self.assertEqual(board.drop_calls, 0)

        board.ticks = 50
        board.balls[0].body.velocity = (100.0, 0.0)
        controller.update(board)
        self.assertIsNone(controller.pending_action)

        board.ticks = 100
        board.balls[0].body.velocity = (0.0, 0.0)
        controller.update(board)
        self.assertEqual(controller.choose_calls, 2)

        board.ticks = 199
        controller.update(board)
        self.assertEqual(board.drop_calls, 0)
        board.ticks = 200
        controller.update(board)
        self.assertEqual(board.drop_calls, 1)

    def test_merge_topology_change_cancels_pending_action(self):
        board = _FakeBoard()
        controller = _RecordingController(
            stable_window_seconds=2 / board.FPS,
            decision_delay_ms=100,
        )

        controller.update(board)
        board.ticks = 1
        controller.update(board)
        self.assertEqual(controller.choose_calls, 1)

        board.ticks = 50
        board.balls[0] = _FakeBall()
        controller.update(board)
        self.assertIsNone(controller.pending_action)
        self.assertEqual(controller.choose_calls, 1)

        board.ticks = 51
        controller.update(board)
        self.assertEqual(controller.choose_calls, 2)
        board.ticks = 151
        controller.update(board)
        self.assertEqual(board.drop_calls, 1)

    def test_successful_drop_and_episode_reset_clear_stability(self):
        board = _FakeBoard()
        controller = _RecordingController(
            stable_window_seconds=2 / board.FPS,
        )

        controller.update(board)
        controller.update(board)
        self.assertEqual(board.drop_calls, 1)
        self.assertEqual(controller.stability_gate.stable_frame_count, 0)

        board.can_drop = True
        controller.update(board)
        self.assertEqual(controller.choose_calls, 1)
        controller.reset_episode()
        self.assertEqual(controller.stability_gate.stable_frame_count, 0)
        self.assertIsNone(controller.pending_action)


if __name__ == '__main__':
    unittest.main()
