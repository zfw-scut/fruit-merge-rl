"""HeadlessGame 完整物理快照与原动作确定性复现测试。"""

from __future__ import annotations

import multiprocessing
import pickle
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import FrozenInstanceError, replace

from daxigua.core import (
    ENGINE_SNAPSHOT_SCHEMA_VERSION,
    EngineSnapshot,
    HeadlessGame,
)


FAST_MAX_FRAMES = 240
FAST_STABLE_FRAMES = 6


def _spawn_replay(snapshot, drop_x):
    """Windows spawn 子进程中的顶层恢复入口。"""

    return HeadlessGame.replay_action(
        snapshot,
        drop_x,
        max_frames=FAST_MAX_FRAMES,
        stable_frames=FAST_STABLE_FRAMES,
    )


class EngineSnapshotTest(unittest.TestCase):
    """验证快照完整性、恢复身份和真实 Pymunk 重演。"""

    @staticmethod
    def _game(*, seed=0, fruit_queue=None):
        game = HeadlessGame(
            seed=seed,
            fps=30,
            space_iterations=8,
        )
        if fruit_queue is not None:
            game.reset(fruit_queue=fruit_queue)
        return game

    @staticmethod
    def _execute(game, drop_x=200):
        return game.execute_action(
            drop_x,
            max_frames=FAST_MAX_FRAMES,
            stable_frames=FAST_STABLE_FRAMES,
        )

    def test_reset_snapshot_is_frozen_pickle_safe_and_restores_rng(self):
        game = self._game(seed=7)
        snapshot = game.capture_snapshot()

        self.assertIsInstance(snapshot, EngineSnapshot)
        self.assertEqual(
            snapshot.schema_version,
            ENGINE_SNAPSHOT_SCHEMA_VERSION,
        )
        self.assertTrue(snapshot.checksum_valid)
        self.assertEqual(len(snapshot.boundaries), 3)
        self.assertEqual(snapshot.fruits, ())
        with self.assertRaises(FrozenInstanceError):
            snapshot.checksum = 'changed'

        roundtripped = pickle.loads(pickle.dumps(snapshot))
        restored = HeadlessGame.from_snapshot(roundtripped)
        self.assertTrue(roundtripped.checksum_valid)
        self.assertEqual(restored.get_state(), game.get_state())
        self.assertEqual(restored.rng.getstate(), game.rng.getstate())
        self.assertEqual(restored.fruit_queue, game.fruit_queue)

        expected = self._execute(game, drop_x=137)
        actual = self._execute(restored, drop_x=137)
        report = HeadlessGame.compare_action_outcomes(expected, actual)
        self.assertTrue(report.matches, report.mismatch_codes)

    def test_snapshot_restores_cached_arbiters_meta_and_shape_order(self):
        game = self._game(seed=0, fruit_queue=(3, 2, 1, 4))
        self._execute(game, drop_x=120)
        self._execute(game, drop_x=280)
        self.assertGreater(len(game.space._get_arbiters()), 0)

        snapshot = game.capture_snapshot()
        raw_space = HeadlessGame._restore_space_blob(snapshot.space_blob)
        self.assertEqual(
            len(raw_space._get_arbiters()),
            len(game.space._get_arbiters()),
        )

        restored = HeadlessGame.from_snapshot(snapshot)
        self.assertEqual(
            tuple(
                restored._snapshot_shape_hashid(shape)
                for shape in restored.balls
            ),
            snapshot.ball_shape_hashids,
        )
        self.assertEqual(
            tuple(
                restored._snapshot_shape_hashid(shape)
                for shape in restored.segments
            ),
            snapshot.segment_shape_hashids,
        )
        self.assertEqual(
            tuple(
                restored._meta_for(shape).fruit_id
                for shape in restored.balls
            ),
            tuple(fruit.fruit_id for fruit in snapshot.fruits),
        )
        self.assertTrue(all(
            id(shape) in restored._fruit_meta
            for shape in restored.balls
        ))

    def test_direct_merge_replays_ids_and_rebinds_callback_to_clone(self):
        game = self._game(seed=0, fruit_queue=(1, 1, 1, 1))
        self._execute(game)
        snapshot = game.capture_snapshot()

        expected = self._execute(game)
        self.assertEqual(
            tuple(event.new_level for event in expected.physics_result.merge_events),
            (2,),
        )
        original_after_expected = (
            game.score,
            game._next_fruit_id,
            tuple(shape._hashid for shape in game.balls),
        )

        restored = HeadlessGame.from_snapshot(snapshot)
        actual = self._execute(restored)
        # 若 handler data 仍引用原 game，这次合成会悄悄修改原实例。
        self.assertEqual(
            (
                game.score,
                game._next_fruit_id,
                tuple(shape._hashid for shape in game.balls),
            ),
            original_after_expected,
        )
        report = HeadlessGame.compare_action_outcomes(expected, actual)
        self.assertTrue(report.matches, report.mismatch_codes)
        self.assertEqual(
            actual.physics_result.merge_events[0].source_ids,
            expected.physics_result.merge_events[0].source_ids,
        )
        self.assertEqual(
            actual.physics_result.merge_events[0].new_fruit_id,
            expected.physics_result.merge_events[0].new_fruit_id,
        )

    def test_two_level_chain_replays_ordered_merge_semantics(self):
        # 先放 2 级，再在其上堆一个 1 级。第三颗 1 级会先合成 2 级，随后与
        # 底部 2 级继续合成 3 级，是检验 arbiter/Shape ID 顺序的关键场景。
        game = self._game(seed=0, fruit_queue=(2, 1, 1, 1))
        self._execute(game)
        self._execute(game)
        snapshot = game.capture_snapshot()

        expected = self._execute(game)
        self.assertEqual(
            tuple(event.new_level for event in expected.physics_result.merge_events),
            (2, 3),
        )
        report = HeadlessGame.replay_and_compare_original_action(
            snapshot,
            expected,
            200,
            max_frames=FAST_MAX_FRAMES,
            stable_frames=FAST_STABLE_FRAMES,
        )

        self.assertTrue(report.matches, report.mismatch_codes)
        self.assertLessEqual(report.max_position_error, 0.05)
        self.assertEqual(
            report.actual_outcome.physics_result.score_delta,
            expected.physics_result.score_delta,
        )
        self.assertEqual(
            tuple(
                (
                    event.source_ids,
                    event.new_fruit_id,
                    event.new_level,
                    event.score_delta,
                )
                for event in report.actual_outcome.physics_result.merge_events
            ),
            tuple(
                (
                    event.source_ids,
                    event.new_fruit_id,
                    event.new_level,
                    event.score_delta,
                )
                for event in expected.physics_result.merge_events
            ),
        )

    def test_canonical_capture_keeps_longer_factual_replay_deterministic(self):
        game = self._game(seed=11)
        action_offsets = (0, 14, 7, 3, 11, 5, 9, 1, 13, 6, 8, 2)

        for step_index in range(36):
            public_state_before = game.get_state()
            snapshot = game.capture_snapshot()
            # 默认 canonicalize 只替换隐藏的 broadphase/arbiters 表示，不得改变
            # 任何对环境、图构建或 reward 可见的字段。
            self.assertEqual(game.get_state(), public_state_before)
            candidates = game.get_action_candidates(15)
            action = candidates[action_offsets[step_index % len(action_offsets)]]
            expected = self._execute(game, action.drop_x)
            report = HeadlessGame.replay_and_compare_original_action(
                snapshot,
                expected,
                action.drop_x,
                max_frames=FAST_MAX_FRAMES,
                stable_frames=FAST_STABLE_FRAMES,
            )
            self.assertTrue(
                report.matches,
                (step_index, report.mismatch_codes),
            )
            if expected.physics_result.done or expected.physics_result.truncated:
                game.reset()

    def test_chain_snapshot_preserves_score_age_queue_and_next_id(self):
        game = self._game(seed=0, fruit_queue=(2, 1, 1, 1))
        self._execute(game)
        self._execute(game)
        self._execute(game)
        snapshot = game.capture_snapshot()
        restored = HeadlessGame.from_snapshot(snapshot)

        self.assertEqual(restored.score, 5)
        self.assertEqual(restored.last_score, 2)
        self.assertEqual(restored._next_fruit_id, 6)
        self.assertEqual(restored.fruit_queue, list(snapshot.episode.fruit_queue))
        self.assertEqual(restored.rng.getstate(), snapshot.episode.rng_state)
        self.assertEqual(
            tuple(
                (
                    restored._meta_for(shape).fruit_id,
                    restored._meta_for(shape).level,
                    restored._meta_for(shape).age_frames,
                )
                for shape in restored.balls
            ),
            tuple(
                (fruit.fruit_id, fruit.level, fruit.age_frames)
                for fruit in snapshot.fruits
            ),
        )

    def test_terminal_original_action_replays_fail_counter_and_ball_order(self):
        game = self._game(seed=0, fruit_queue=(1, 2, 3, 4))
        # 构造一个已持续越线但尚未超过失败阈值的旧水果。下一次 drop 会成为
        # balls[-1]，check_fail 必须继续检查旧水果并在第一物理帧终止。
        old_ball = game._create_ball(100, 100, 1)
        game.space.step(1 / game.fps)
        old_ball.body.position = (100, 100)
        old_ball.body.velocity = (0, 0)
        game.space.reindex_shapes_for_body(old_ball.body)
        game.fail_count = int(game.fps * game.create_time)
        snapshot = game.capture_snapshot()

        expected = self._execute(game, drop_x=300)
        self.assertTrue(expected.physics_result.done)
        self.assertFalse(expected.physics_result.truncated)
        self.assertEqual(expected.physics_result.frames_simulated, 1)
        report = HeadlessGame.replay_and_compare_original_action(
            snapshot,
            expected,
            300,
            max_frames=FAST_MAX_FRAMES,
            stable_frames=FAST_STABLE_FRAMES,
        )

        self.assertTrue(report.matches, report.mismatch_codes)
        self.assertTrue(report.actual_outcome.final_state.done)
        self.assertEqual(
            report.actual_outcome.fail_count,
            expected.fail_count,
        )

    def test_capture_rejects_non_boundary_terminal_and_sleeping_modes(self):
        game = self._game(seed=0)
        game.drop_at(200)
        with self.assertRaisesRegex(RuntimeError, 'phase'):
            game.capture_snapshot()

        game.reset(seed=0)
        result = game.advance_physics(max_frames=1, stable_frames=2)
        self.assertTrue(result.truncated)
        with self.assertRaisesRegex(RuntimeError, 'phase'):
            game.capture_snapshot()

        game.reset(seed=0)
        game.alive = False
        with self.assertRaisesRegex(RuntimeError, 'terminal'):
            game.capture_snapshot()

        game.reset(seed=0)
        game.space.sleep_time_threshold = 1.0
        with self.assertRaisesRegex(RuntimeError, 'sleeping'):
            game.capture_snapshot()

    def test_checksum_version_and_cross_config_mismatches_are_rejected(self):
        game = self._game(seed=0)
        snapshot = game.capture_snapshot()

        corrupted_tail = bytes((snapshot.space_blob[-1] ^ 1,))
        corrupted = replace(
            snapshot,
            space_blob=snapshot.space_blob[:-1] + corrupted_tail,
        )
        with self.assertRaisesRegex(ValueError, 'checksum'):
            HeadlessGame.from_snapshot(corrupted)

        wrong_version = replace(snapshot, pymunk_version='0.0.invalid')
        with self.assertRaisesRegex(ValueError, 'Pymunk version'):
            HeadlessGame.from_snapshot(wrong_version)

        different_config = HeadlessGame(
            seed=0,
            fps=30,
            space_iterations=8,
            gravity=(0, 0),
        )
        with self.assertRaisesRegex(ValueError, 'different engine config'):
            different_config.restore_snapshot(snapshot)

    def test_resealed_semantic_mirror_tampering_is_rejected(self):
        game = self._game(seed=0, fruit_queue=(1, 2, 3, 4))
        self._execute(game)
        snapshot = game.capture_snapshot()
        first = snapshot.fruits[0]
        tampered_fruit = replace(
            first,
            position=(first.position[0] + 1.0, first.position[1]),
        )
        tampered = replace(
            snapshot,
            fruits=(tampered_fruit,) + snapshot.fruits[1:],
            checksum='',
        ).sealed()
        self.assertTrue(tampered.checksum_valid)

        with self.assertRaisesRegex(ValueError, 'fruit body'):
            HeadlessGame.from_snapshot(tampered)

    def test_snapshot_and_replay_survive_real_windows_spawn(self):
        game = self._game(seed=5, fruit_queue=(1, 1, 2, 3))
        self._execute(game)
        snapshot = game.capture_snapshot()
        expected = self._execute(game)
        context = multiprocessing.get_context('spawn')

        with ProcessPoolExecutor(
                max_workers=1,
                mp_context=context) as executor:
            actual = executor.submit(
                _spawn_replay,
                snapshot,
                200,
            ).result(timeout=30)

        report = HeadlessGame.compare_action_outcomes(expected, actual)
        self.assertTrue(report.matches, report.mismatch_codes)


if __name__ == '__main__':
    unittest.main()
