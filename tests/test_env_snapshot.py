"""DaxiguaEnv 与 EngineSnapshot 的动作执行桥接测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from daxigua.core import HeadlessGame
from daxigua.core.state import EngineActionOutcome
from daxigua_rl.attribution import StateAnalyzerConfig
from daxigua_rl.env import DaxiguaEnv, DaxiguaEnvConfig
from daxigua_rl.reward import RewardConfig
from daxigua_rl.training import TransitionKey


FAST_FPS = 30
FAST_ITERATIONS = 8
FAST_MAX_FRAMES = 240
FAST_STABLE_FRAMES = 6


def _env_config(**overrides):
    values = {
        'action_count': 5,
        'physics_fps': FAST_FPS,
        'max_physics_frames': FAST_MAX_FRAMES,
        'stable_frames': FAST_STABLE_FRAMES,
        'space_iterations': FAST_ITERATIONS,
        'state_analyzer_config': StateAnalyzerConfig(
            grid_cell_size=16.0,
        ),
    }
    values.update(overrides)
    return DaxiguaEnvConfig(**values)


def _reset_snapshot(*, seed=0, fruit_queue=(1, 1, 2, 3)):
    game = HeadlessGame(
        seed=seed,
        fps=FAST_FPS,
        space_iterations=FAST_ITERATIONS,
    )
    game.reset(fruit_queue=fruit_queue)
    return game, game.capture_snapshot()


class DaxiguaEnvSnapshotTest(unittest.TestCase):
    """验证环境 step 与完整物理快照共享同一动作契约。"""

    def test_step_uses_execute_action_and_matches_legacy_sequence(self):
        _source, snapshot = _reset_snapshot(seed=7)
        config = _env_config()

        legacy_game = HeadlessGame.from_snapshot(snapshot)
        legacy_action = legacy_game.get_action_candidates(
            config.action_count
        )[2]
        legacy_drop = legacy_game.drop_at(legacy_action.drop_x)
        legacy_physics = legacy_game.advance_physics(
            max_frames=config.max_physics_frames,
            until_stable=True,
            stable_frames=config.stable_frames,
        )
        legacy_obs = legacy_game.get_state()

        env = DaxiguaEnv.from_snapshot(snapshot, config=config)
        with patch.object(
                env.game,
                'execute_action',
                wraps=env.game.execute_action) as execute_action:
            obs, _reward, terminated, truncated, info = env.step(
                2,
                transition_key=TransitionKey(4, 8, 0),
            )

        execute_action.assert_called_once_with(
            legacy_action.drop_x,
            max_frames=config.max_physics_frames,
            stable_frames=config.stable_frames,
        )
        self.assertEqual(obs, legacy_obs)
        self.assertEqual(info['drop_result'], legacy_drop)
        self.assertEqual(info['physics_result'], legacy_physics)
        self.assertEqual(terminated, legacy_physics.done)
        self.assertEqual(truncated, legacy_physics.truncated)

    def test_engine_outcome_info_is_consistent_with_observation(self):
        _source, snapshot = _reset_snapshot(seed=11)
        env = DaxiguaEnv.from_snapshot(snapshot, config=_env_config())

        obs, _reward, terminated, truncated, info = env.step(
            1,
            transition_key=TransitionKey(2, 3, 0),
        )
        outcome = info['engine_action_outcome']

        self.assertIsInstance(outcome, EngineActionOutcome)
        self.assertIs(info['drop_result'], outcome.drop_result)
        self.assertIs(info['physics_result'], outcome.physics_result)
        self.assertIs(obs, outcome.final_state)
        self.assertEqual(obs, env.game.get_state())
        self.assertEqual(terminated, outcome.physics_result.done)
        self.assertEqual(truncated, outcome.physics_result.truncated)
        self.assertEqual(
            info['score_delta'],
            outcome.physics_result.score_delta,
        )
        self.assertEqual(
            info['merge_events'],
            outcome.physics_result.merge_events,
        )
        self.assertEqual(
            info['frames_simulated'],
            outcome.physics_result.frames_simulated,
        )
        self.assertEqual(info['stable'], outcome.physics_result.stable)

    def test_restored_environment_steps_without_reset_using_same_key(self):
        source, _initial_snapshot = _reset_snapshot(seed=19)
        first_action = source.get_action_candidates(5)[3]
        first_outcome = source.execute_action(
            first_action.drop_x,
            max_frames=FAST_MAX_FRAMES,
            stable_frames=FAST_STABLE_FRAMES,
        )
        self.assertTrue(first_outcome.physics_result.stable)
        snapshot = source.capture_snapshot()
        config = _env_config()
        env = DaxiguaEnv.from_snapshot(snapshot, config=config)
        action_index = 4
        action = env.action_candidates()[action_index]
        expected = HeadlessGame.replay_action(
            snapshot,
            action.drop_x,
            max_frames=config.max_physics_frames,
            stable_frames=config.stable_frames,
        )
        transition_key = TransitionKey(
            worker_id=9,
            episode_id=12,
            step_index=snapshot.episode.step_count,
        )

        obs, _reward, _terminated, _truncated, info = env.step(
            action_index,
            transition_key=transition_key,
        )

        self.assertEqual(
            info['previous_state_analysis'].transition_key,
            transition_key,
        )
        self.assertEqual(obs, expected.final_state)
        report = HeadlessGame.compare_action_outcomes(
            expected,
            info['engine_action_outcome'],
        )
        self.assertTrue(report.matches, report.mismatch_codes)

    def test_factory_preserves_rl_config_and_rejects_physics_mismatch(self):
        _source, snapshot = _reset_snapshot(seed=23)
        reward_config = RewardConfig(
            gamma=0.87,
            lambda_phi=0.4,
            capacity_weight=0.5,
            recoverability_weight=0.3,
            chain_readiness_weight=0.2,
        )
        analyzer_config = StateAnalyzerConfig(
            grid_cell_size=12.0,
            max_motifs=16,
        )
        config = _env_config(
            action_count=7,
            reward_config=reward_config,
            state_analyzer_config=analyzer_config,
        )

        env = DaxiguaEnv.from_snapshot(snapshot, config=config)
        self.assertIs(env.config, config)
        self.assertIs(env.config.reward_config, reward_config)
        self.assertIs(env.state_analyzer.config, analyzer_config)

        inferred = DaxiguaEnv.from_snapshot(snapshot)
        self.assertEqual(inferred.config.physics_fps, FAST_FPS)
        self.assertEqual(
            inferred.config.space_iterations,
            FAST_ITERATIONS,
        )

        with self.assertRaisesRegex(ValueError, 'physics_fps'):
            DaxiguaEnv.from_snapshot(
                snapshot,
                config=_env_config(physics_fps=FAST_FPS + 1),
            )
        with self.assertRaisesRegex(ValueError, 'space_iterations'):
            DaxiguaEnv.from_snapshot(
                snapshot,
                config=_env_config(
                    space_iterations=FAST_ITERATIONS + 1,
                ),
            )
        with self.assertRaisesRegex(TypeError, 'DaxiguaEnvConfig'):
            DaxiguaEnv.from_snapshot(snapshot, config=object())


if __name__ == '__main__':
    unittest.main()
