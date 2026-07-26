"""Reward V2 在环境、collector 和 Windows spawn 边界上的集成测试。"""

from __future__ import annotations

import math
import multiprocessing
import pickle
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

from daxigua.core.rules import dropped_fruit_physics_radius, fruit_radius
from daxigua.core.state import (
    ActionCandidate,
    BoardGeometry,
    DropResult,
    GameState,
    PhysicsResult,
)
from daxigua_rl.attribution import (
    ANALYSIS_ACTION_COUNT,
    StateAnalyzer,
    StateAnalyzerConfig,
    drop_x_positions_for_level,
)
from daxigua_rl.env import DaxiguaEnv, DaxiguaEnvConfig
from daxigua_rl.graph import GraphBuilder
from daxigua_rl.reward import REWARD_BREAKDOWN_FIELDS, RewardConfig
from daxigua_rl.training import ReplayBuffer, RolloutCollector, TransitionKey
from daxigua_rl.training.collector import RolloutStats
from daxigua_rl.training.parallel_collector import _merge_rollout_stats


GEOMETRY = BoardGeometry(
    width=400,
    height=800,
    spawn_y=180,
    wall_width=20,
    floor_y=780,
)


def _state(*, step_count=0, physics_frame=0, done=False):
    """构造空盘状态，让 integration 测试只关注分析边界而非物理随机性。"""

    return GameState(
        board_fruits=(),
        fruit_queue=(1, 2, 3, 4),
        score=0,
        last_score=0,
        step_count=int(step_count),
        physics_frame=int(physics_frame),
        done=bool(done),
        geometry=GEOMETRY,
        max_height=0.0,
        fruit_count=0,
        max_level=0,
        empty_space_ratio=1.0,
    )


def _physics(*, stable=True, done=False, truncated=False):
    """构造无合成的确定性物理结果。"""

    return PhysicsResult(
        frames_simulated=1,
        stable=bool(stable),
        done=bool(done),
        truncated=bool(truncated),
        score_delta=0,
        merge_events=(),
    )


def _spawn_roundtrip(value):
    """供 spawn 子进程调用的顶层 pickle roundtrip helper。"""

    return pickle.loads(pickle.dumps(value))


class _ScriptedGame:
    """只实现 ``DaxiguaEnv`` 所需接口的确定性空盘游戏。"""

    def __init__(self, physics_results):
        self._physics_results = tuple(physics_results)
        self._physics_offset = 0
        self._state = _state()

    def reset(self, seed=None, fruit_queue=None):
        del seed
        self._state = _state()
        if fruit_queue is not None:
            self._state = replace(
                self._state,
                fruit_queue=tuple(fruit_queue),
            )
        return self._state

    def get_state(self):
        return self._state

    def get_action_candidates(self, action_count):
        level = int(self._state.fruit_queue[0])
        positions = drop_x_positions_for_level(
            GEOMETRY,
            level,
            action_count=int(action_count),
        )
        left = positions[0]
        right = positions[-1]
        return tuple(
            ActionCandidate(
                action_index=offset,
                drop_x=drop_x,
                normalized_drop_x=(
                    0.0
                    if right == left
                    else (drop_x - left) / (right - left)
                ),
                current_level=level,
                current_radius=fruit_radius(level),
                current_physics_radius=dropped_fruit_physics_radius(level),
            )
            for offset, drop_x in enumerate(positions)
        )

    def drop_at(self, drop_x):
        queue_before = self._state.fruit_queue
        self._state = replace(
            self._state,
            step_count=self._state.step_count + 1,
        )
        return DropResult(
            dropped_level=queue_before[0],
            drop_x=float(drop_x),
            fruit_id=self._state.step_count,
            queue_before=queue_before,
            queue_after=self._state.fruit_queue,
        )

    def advance_physics(self, max_frames, until_stable, stable_frames):
        del max_frames, until_stable, stable_frames
        if self._physics_offset >= len(self._physics_results):
            raise RuntimeError('scripted physics result sequence exhausted')
        result = self._physics_results[self._physics_offset]
        self._physics_offset += 1
        self._state = replace(
            self._state,
            physics_frame=self._state.physics_frame + result.frames_simulated,
            done=result.done,
        )
        return result


class _CountingAnalyzer(StateAnalyzer):
    """保留真实分析结果，同时记录环境是否重复计算相同边界。"""

    def __init__(self):
        super().__init__(StateAnalyzerConfig(grid_cell_size=16.0))
        self.calls = []

    def analyze(
            self,
            state,
            action_candidates,
            transition_key,
            **kwargs):
        candidates = tuple(action_candidates)
        self.calls.append((
            state.step_count,
            len(candidates),
            transition_key,
            kwargs.get('stable_boundary'),
            kwargs.get('incoming_transition_key'),
        ))
        return super().analyze(
            state,
            candidates,
            transition_key,
            **kwargs,
        )


class RewardV2EnvironmentIntegrationTest(unittest.TestCase):
    """验证 StateAnalyzer 缓存、相邻 key、terminal 与 truncated。"""

    def test_terminal_keeps_reward_none_but_builds_attribution_analysis(self):
        analyzer = _CountingAnalyzer()
        env = DaxiguaEnv(
            config=DaxiguaEnvConfig(
                # 环境动作数可以较小，但状态分析必须始终使用规范 15 列。
                action_count=3,
                reward_config=RewardConfig(gamma=0.99),
            ),
            game=_ScriptedGame((
                _physics(stable=True),
                _physics(stable=True, done=True),
            )),
            state_analyzer=analyzer,
        )
        env.reset()
        first_key = TransitionKey(7, 3, 0)

        _, first_reward, first_terminated, first_truncated, first_info = env.step(
            1,
            transition_key=first_key,
        )
        second_key = TransitionKey(7, 3, 1)
        _, second_reward, second_terminated, second_truncated, second_info = env.step(
            1,
            transition_key=second_key,
        )

        self.assertFalse(first_terminated)
        self.assertFalse(first_truncated)
        self.assertTrue(second_terminated)
        self.assertFalse(second_truncated)
        self.assertEqual(first_info['state_analysis_calls'], 2)
        self.assertFalse(first_info['state_analysis_cache_hit'])
        self.assertEqual(second_info['state_analysis_calls'], 1)
        self.assertTrue(second_info['state_analysis_cache_hit'])
        self.assertEqual(len(analyzer.calls), 3)
        self.assertEqual(
            analyzer.calls,
            [
                (0, ANALYSIS_ACTION_COUNT, first_key, True, None),
                (
                    1,
                    ANALYSIS_ACTION_COUNT,
                    second_key,
                    True,
                    first_key,
                ),
                (
                    2,
                    ANALYSIS_ACTION_COUNT,
                    TransitionKey(7, 3, 2),
                    True,
                    second_key,
                ),
            ],
        )
        self.assertIs(
            second_info['previous_state_analysis'],
            first_info['next_state_analysis'],
        )
        self.assertIsNone(second_info['next_state_analysis'])
        self.assertEqual(
            second_info[
                'post_action_state_analysis'
            ].transition_key,
            TransitionKey(7, 3, 2),
        )
        self.assertAlmostEqual(first_reward, -0.0045, places=12)
        self.assertAlmostEqual(second_reward, -0.45, places=12)

    def test_truncated_keeps_degraded_next_analysis_and_reset_drops_cache(self):
        analyzer = _CountingAnalyzer()
        env = DaxiguaEnv(
            config=DaxiguaEnvConfig(action_count=3),
            game=_ScriptedGame((
                _physics(stable=False, truncated=True),
                _physics(stable=True),
            )),
            state_analyzer=analyzer,
        )
        env.reset()
        first_key = TransitionKey(2, 5, 0)

        _, reward, terminated, truncated, info = env.step(
            0,
            transition_key=first_key,
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIsNotNone(info['next_state_analysis'])
        self.assertEqual(
            info['next_state_analysis'].transition_key,
            TransitionKey(2, 5, 1),
        )
        self.assertEqual(
            info['next_state_analysis'].incoming_transition_key,
            first_key,
        )
        self.assertTrue(info['next_state_analysis'].diagnostics.degraded)
        self.assertFalse(
            info['next_state_analysis'].diagnostics.valid_for_attribution
        )
        self.assertEqual(info['state_analysis_calls'], 2)
        self.assertEqual(info['state_analysis_degraded_count'], 1)
        self.assertAlmostEqual(
            reward,
            info['reward_breakdown'].potential_shaping_reward,
            places=12,
        )
        with self.assertRaises(RuntimeError):
            env.step(0, transition_key=TransitionKey(2, 5, 1))

        env.reset()
        _, _, _, _, after_reset = env.step(
            0,
            transition_key=TransitionKey(2, 6, 0),
        )
        self.assertEqual(after_reset['state_analysis_calls'], 2)
        self.assertFalse(after_reset['state_analysis_cache_hit'])
        self.assertEqual(len(analyzer.calls), 4)

    def test_direct_environment_keys_are_local_and_reset_starts_new_episode(self):
        analyzer = _CountingAnalyzer()
        env = DaxiguaEnv(
            config=DaxiguaEnvConfig(action_count=3),
            game=_ScriptedGame((
                _physics(stable=True, done=True),
                _physics(stable=True, done=True),
            )),
            state_analyzer=analyzer,
        )

        env.reset()
        _, _, _, _, first = env.step(0)
        env.reset()
        _, _, _, _, second = env.step(0)

        self.assertEqual(
            first['previous_state_analysis'].transition_key,
            TransitionKey(0, 0, 0),
        )
        self.assertEqual(
            second['previous_state_analysis'].transition_key,
            TransitionKey(0, 1, 0),
        )


class RewardV2CollectorIntegrationTest(unittest.TestCase):
    """验证 collector 将分析身份、耗时和 shaping 分布完整汇总。"""

    def test_collector_preserves_keys_cache_counts_and_potential_distribution(self):
        analyzer = _CountingAnalyzer()
        env = DaxiguaEnv(
            config=DaxiguaEnvConfig(
                action_count=3,
                reward_config=RewardConfig(gamma=0.99),
            ),
            game=_ScriptedGame((
                _physics(stable=True),
                _physics(stable=True, done=True),
            )),
            state_analyzer=analyzer,
        )
        replay_buffer = ReplayBuffer(capacity=4, seed=0)
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay_buffer,
            seed=0,
            worker_id=9,
            # 该空盘脚本刻意不创建 drop_result 对应的水果，只测试 Reward V2
            # 分析缓存；真实谱系测试使用满足物理身份不变量的专用 fixture。
            attribution_tracker=False,
        )

        stats = collector.collect_steps(2, epsilon=1.0)

        self.assertEqual(
            tuple(key.as_tuple() for key in stats.transition_keys),
            ((9, 0, 0), (9, 0, 1)),
        )
        self.assertEqual(stats.state_analysis_calls, 3)
        self.assertEqual(stats.state_analysis_cache_hits, 1)
        self.assertEqual(stats.state_analysis_degraded_count, 0)
        self.assertGreaterEqual(stats.state_analysis_seconds, 0.0)
        self.assertGreaterEqual(stats.mean_state_analysis_ms, 0.0)
        self.assertEqual(stats.state_analysis_cache_hit_rate, 0.5)
        self.assertEqual(len(stats.potential_shaping_abs_values), 2)
        self.assertAlmostEqual(
            stats.potential_shaping_abs_values[0],
            0.0045,
            places=12,
        )
        self.assertAlmostEqual(
            stats.potential_shaping_abs_values[1],
            0.45,
            places=12,
        )
        self.assertAlmostEqual(
            stats.p95_abs_potential_shaping_reward,
            0.0045 * 0.05 + 0.45 * 0.95,
            places=12,
        )
        self.assertEqual(len(replay_buffer), 2)

    def test_rollout_stats_p95_uses_linear_interpolation(self):
        stats = RolloutStats(
            steps=20,
            episodes=0,
            total_reward=0.0,
            potential_shaping_abs_values=tuple(range(1, 21)),
        )

        self.assertAlmostEqual(
            stats.p95_abs_potential_shaping_reward,
            19.05,
            places=12,
        )
        self.assertEqual(
            RolloutStats(
                steps=0,
                episodes=0,
                total_reward=0.0,
            ).p95_abs_potential_shaping_reward,
            0.0,
        )

    def test_parallel_stats_merge_preserves_analysis_and_shaping_fields(self):
        first = RolloutStats(
            steps=2,
            episodes=0,
            total_reward=1.0,
            reward_breakdown_totals=tuple(
                (field_name, 1.0)
                for field_name in REWARD_BREAKDOWN_FIELDS
            ),
            potential_shaping_abs_values=(0.1, 0.2),
            transition_keys=(TransitionKey(0, 0, 0), TransitionKey(0, 0, 1)),
            state_analysis_calls=3,
            state_analysis_seconds=0.03,
            state_analysis_cache_hits=1,
            state_analysis_degraded_count=0,
        )
        second = RolloutStats(
            steps=1,
            episodes=0,
            total_reward=2.0,
            reward_breakdown_totals=tuple(
                (field_name, 2.0)
                for field_name in REWARD_BREAKDOWN_FIELDS
            ),
            potential_shaping_abs_values=(0.4,),
            transition_keys=(TransitionKey(1, 0, 0),),
            state_analysis_calls=2,
            state_analysis_seconds=0.02,
            state_analysis_cache_hits=0,
            state_analysis_degraded_count=1,
        )

        merged = _merge_rollout_stats(
            worker_stats=(first, second),
            buffer_size=3,
            collect_seconds=0.5,
        )

        self.assertEqual(merged.steps, 3)
        self.assertEqual(merged.total_reward, 3.0)
        self.assertEqual(merged.potential_shaping_abs_values, (0.1, 0.2, 0.4))
        self.assertEqual(merged.state_analysis_calls, 5)
        self.assertAlmostEqual(merged.state_analysis_seconds, 0.05)
        self.assertEqual(merged.state_analysis_cache_hits, 1)
        self.assertEqual(merged.state_analysis_degraded_count, 1)
        self.assertEqual(
            merged.reward_breakdown_totals_dict['task_reward'],
            3.0,
        )

    def test_reward_config_env_config_and_stats_survive_spawn_pickle(self):
        config = DaxiguaEnvConfig(
            action_count=15,
            physics_fps=30,
            max_physics_frames=240,
            stable_frames=6,
            space_iterations=8,
            reward_config=RewardConfig(
                gamma=0.87,
                lambda_phi=0.4,
                capacity_weight=0.5,
                recoverability_weight=0.3,
                chain_readiness_weight=0.2,
            ),
            state_analyzer_config=StateAnalyzerConfig(
                grid_cell_size=12.0,
                max_motifs=16,
            ),
        )
        stats = RolloutStats(
            steps=1,
            episodes=0,
            total_reward=0.25,
            potential_shaping_abs_values=(0.25,),
            transition_keys=(TransitionKey(3, 4, 5),),
            state_analysis_calls=2,
            state_analysis_seconds=0.01,
        )
        context = multiprocessing.get_context('spawn')

        with ProcessPoolExecutor(
                max_workers=1,
                mp_context=context) as executor:
            restored_config = executor.submit(
                _spawn_roundtrip,
                config,
            ).result(timeout=30)
            restored_stats = executor.submit(
                _spawn_roundtrip,
                stats,
            ).result(timeout=30)

        self.assertEqual(restored_config, config)
        self.assertEqual(
            restored_config.state_analyzer_config.fingerprint,
            config.state_analyzer_config.fingerprint,
        )
        self.assertEqual(restored_stats, stats)
        self.assertEqual(restored_stats.transition_keys[0], TransitionKey(3, 4, 5))


if __name__ == '__main__':
    unittest.main()
