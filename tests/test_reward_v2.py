"""Reward V2 的纯公式与数据契约测试。

这些测试使用带 ``StateAnalysis`` spec 的只读 mock，只验证奖励层本身，不运行几何
分析或 Pymunk。环境如何缓存相邻分析由 ``test_reward_v2_integration.py`` 单独覆盖。
"""

from __future__ import annotations

import inspect
import math
import pickle
import unittest
from unittest.mock import NonCallableMock

from daxigua.core.rules import merge_score
from daxigua.core.state import MergeEvent, PhysicsResult
from daxigua_rl.attribution import StateAnalysis
from daxigua_rl.reward import (
    REWARD_BREAKDOWN_FIELDS,
    RewardConfig,
    compute_reward,
    compute_state_potential,
    merge_utility,
)
from daxigua_rl.training.identity import TransitionKey


def _analysis(
        capacity,
        recoverability,
        chain_readiness,
        *,
        step_index=0,
        worker_id=0,
        episode_id=0,
        incoming_transition_key=None,
        fingerprint='sha256:test-analyzer',
        degraded=False):
    """构造只包含 reward 所消费字段的 ``StateAnalysis`` 测试替身。"""

    analysis = NonCallableMock(spec=StateAnalysis)
    analysis.top_connected_capacity = float(capacity)
    analysis.recoverability = float(recoverability)
    analysis.chain_readiness = float(chain_readiness)
    analysis.transition_key = TransitionKey(
        worker_id,
        episode_id,
        step_index,
    )
    analysis.incoming_transition_key = incoming_transition_key
    analysis.analyzer_config_fingerprint = fingerprint
    analysis.diagnostics = NonCallableMock()
    analysis.diagnostics.degraded = bool(degraded)
    analysis.diagnostics.valid_for_attribution = not degraded
    return analysis


def _merge_event(new_level, *, new_fruit_id=None, source_ids=None):
    """按游戏规则构造一条内部一致的合成事件。"""

    if new_fruit_id is None:
        new_fruit_id = 100 + int(new_level)
    if source_ids is None:
        source_ids = (new_fruit_id - 2, new_fruit_id - 1)
    return MergeEvent(
        new_level=int(new_level),
        x=200.0,
        y=500.0,
        score_delta=merge_score(int(new_level)),
        source_ids=tuple(source_ids),
        new_fruit_id=int(new_fruit_id),
    )


def _physics(*events, done=False, truncated=False, stable=True):
    """由合成事件构造奖励层需要的 ``PhysicsResult``。"""

    return PhysicsResult(
        frames_simulated=1,
        stable=bool(stable),
        done=bool(done),
        truncated=bool(truncated),
        score_delta=sum(event.score_delta for event in events),
        merge_events=tuple(events),
    )


class MergeUtilityTest(unittest.TestCase):
    """指数合成任务效用不能退回游戏原始分数或固定连锁奖励。"""

    def test_all_merge_levels_follow_exponential_utility(self):
        expected = {
            level: 2.0 ** ((level - 2) / 2.0)
            for level in range(2, 12)
        }

        for level, value in expected.items():
            with self.subTest(level=level):
                self.assertAlmostEqual(merge_utility(level), value, places=12)

        for level in range(2, 10):
            self.assertAlmostEqual(
                merge_utility(level + 2),
                2.0 * merge_utility(level),
                places=12,
            )
        for level in range(2, 11):
            self.assertAlmostEqual(
                merge_utility(level + 1) / merge_utility(level),
                math.sqrt(2.0),
                places=12,
            )

    def test_multi_merge_reward_is_exact_event_utility_sum(self):
        events = tuple(
            _merge_event(level, new_fruit_id=100 + offset)
            for offset, level in enumerate((2, 3, 4))
        )
        previous = _analysis(0.0, 0.0, 0.0, step_index=0)
        next_analysis = _analysis(
            0.0,
            0.0,
            0.0,
            step_index=1,
            incoming_transition_key=previous.transition_key,
        )

        reward, breakdown = compute_reward(
            previous,
            next_analysis,
            _physics(*events),
            RewardConfig(gamma=1.0),
        )

        expected = 1.0 + math.sqrt(2.0) + 2.0
        self.assertAlmostEqual(reward, expected, places=12)
        self.assertAlmostEqual(breakdown.task_reward, expected, places=12)
        self.assertEqual(breakdown.merge_event_count, 3)

    def test_level_eleven_uses_strategic_utility_without_global_clipping(self):
        event = _merge_event(11)
        previous = _analysis(0.0, 0.0, 0.0, step_index=0)
        next_analysis = _analysis(
            0.0,
            0.0,
            0.0,
            step_index=1,
            incoming_transition_key=previous.transition_key,
        )

        reward, breakdown = compute_reward(
            previous,
            next_analysis,
            _physics(event),
        )

        self.assertEqual(event.score_delta, 100)
        self.assertAlmostEqual(reward, 2.0 ** 4.5, places=12)
        self.assertAlmostEqual(breakdown.task_reward, 2.0 ** 4.5, places=12)
        self.assertNotEqual(breakdown.task_reward, event.score_delta)

    def test_invalid_level_and_inconsistent_event_stream_are_rejected(self):
        with self.assertRaises(ValueError):
            merge_utility(1)
        with self.assertRaises(ValueError):
            merge_utility(12)
        with self.assertRaises(TypeError):
            merge_utility(True)

        previous = _analysis(0.0, 0.0, 0.0, step_index=0)
        next_analysis = _analysis(
            0.0,
            0.0,
            0.0,
            step_index=1,
            incoming_transition_key=previous.transition_key,
        )
        event = _merge_event(5)
        inconsistent = PhysicsResult(
            frames_simulated=1,
            stable=True,
            done=False,
            truncated=False,
            score_delta=event.score_delta + 1,
            merge_events=(event,),
        )

        with self.assertRaises(ValueError):
            compute_reward(previous, next_analysis, inconsistent)


class StatePotentialTest(unittest.TestCase):
    """验证 Phi 权重、差分、terminal 和 truncated 数学。"""

    def test_default_weighted_potential_and_empty_board_baseline(self):
        mixed = _analysis(0.2, 0.8, 0.4)
        empty = _analysis(1.0, 1.0, 0.0)

        self.assertAlmostEqual(
            compute_state_potential(mixed),
            0.6 * 0.2 + 0.3 * 0.8 + 0.1 * 0.4,
            places=12,
        )
        self.assertAlmostEqual(compute_state_potential(empty), 0.9, places=12)

    def test_potential_shaping_satisfies_discounted_telescope_identity(self):
        config = RewardConfig(gamma=0.9, lambda_phi=0.5)
        s0 = _analysis(0.0, 2.0 / 3.0, 0.0, step_index=0)
        s1 = _analysis(
            1.0,
            2.0 / 3.0,
            0.0,
            step_index=1,
            incoming_transition_key=s0.transition_key,
        )
        s2 = _analysis(
            0.5,
            1.0 / 3.0,
            0.0,
            step_index=2,
            incoming_transition_key=s1.transition_key,
        )
        # 上述三个合法归一化替身分别给出 Phi=0.2、0.8、0.4。
        _, first = compute_reward(s0, s1, _physics(), config)
        _, second = compute_reward(s1, s2, _physics(), config)

        discounted_sum = (
            first.potential_shaping_reward
            + config.gamma * second.potential_shaping_reward
        )
        expected = config.lambda_phi * (
            -first.previous_potential
            + config.gamma ** 2 * second.next_potential
        )
        self.assertAlmostEqual(first.potential_shaping_reward, 0.26, places=12)
        self.assertAlmostEqual(second.potential_shaping_reward, -0.22, places=12)
        self.assertAlmostEqual(discounted_sum, 0.062, places=12)
        self.assertAlmostEqual(discounted_sum, expected, places=12)

    def test_terminal_forces_zero_next_potential_without_default_fixed_penalty(self):
        previous = _analysis(1.0, 1.0, 0.0, step_index=4)
        supplied_next = _analysis(
            1.0,
            1.0,
            1.0,
            step_index=5,
            incoming_transition_key=previous.transition_key,
        )
        config = RewardConfig(gamma=0.99, lambda_phi=0.5)

        reward, breakdown = compute_reward(
            previous,
            supplied_next,
            _physics(done=True),
            config,
        )

        self.assertAlmostEqual(breakdown.previous_potential, 0.9, places=12)
        self.assertEqual(breakdown.next_potential, 0.0)
        self.assertEqual(breakdown.next_top_connected_capacity, 0.0)
        self.assertEqual(breakdown.next_recoverability, 0.0)
        self.assertEqual(breakdown.next_chain_readiness, 0.0)
        self.assertEqual(breakdown.terminal_penalty, 0.0)
        self.assertAlmostEqual(reward, -0.45, places=12)

        reward_without_next, _ = compute_reward(
            previous,
            None,
            _physics(done=True),
            config,
        )
        self.assertAlmostEqual(reward_without_next, reward, places=12)

    def test_explicit_terminal_penalty_is_opt_in_only(self):
        previous = _analysis(0.0, 0.0, 0.0, step_index=0)
        reward, breakdown = compute_reward(
            previous,
            None,
            _physics(done=True),
            RewardConfig(terminal_penalty=-2.0),
        )

        self.assertEqual(reward, -2.0)
        self.assertEqual(breakdown.terminal_penalty, -2.0)

    def test_truncated_keeps_unstable_next_potential(self):
        previous = _analysis(0.0, 0.0, 0.0, step_index=0)
        next_analysis = _analysis(
            1.0,
            1.0,
            1.0,
            step_index=1,
            incoming_transition_key=previous.transition_key,
            degraded=True,
        )
        config = RewardConfig(gamma=0.9, lambda_phi=0.5)

        reward, breakdown = compute_reward(
            previous,
            next_analysis,
            _physics(truncated=True, stable=False),
            config,
        )

        self.assertTrue(next_analysis.diagnostics.degraded)
        self.assertAlmostEqual(breakdown.next_potential, 1.0, places=12)
        self.assertAlmostEqual(breakdown.potential_delta, 0.9, places=12)
        self.assertAlmostEqual(reward, 0.45, places=12)

    def test_same_structural_inputs_have_no_survival_or_height_reward(self):
        config = RewardConfig(gamma=0.99)
        first_previous = _analysis(0.4, 0.7, 0.2, step_index=0)
        first_next = _analysis(
            0.5,
            0.6,
            0.3,
            step_index=1,
            incoming_transition_key=first_previous.transition_key,
        )
        later_previous = _analysis(0.4, 0.7, 0.2, step_index=20)
        later_next = _analysis(
            0.5,
            0.6,
            0.3,
            step_index=21,
            incoming_transition_key=later_previous.transition_key,
        )

        first_reward, _ = compute_reward(
            first_previous,
            first_next,
            _physics(),
            config,
        )
        later_reward, _ = compute_reward(
            later_previous,
            later_next,
            _physics(),
            config,
        )

        self.assertAlmostEqual(first_reward, later_reward, places=12)
        self.assertNotIn('previous_state', inspect.signature(compute_reward).parameters)
        self.assertNotIn('next_state', inspect.signature(compute_reward).parameters)
        for removed_field in (
                'score_reward',
                'survival_bonus',
                'height_delta_reward',
                'danger_penalty',
                'previous_height_ratio',
                'next_height_ratio',
                'height_delta_ratio'):
            self.assertNotIn(removed_field, REWARD_BREAKDOWN_FIELDS)


class RewardContractTest(unittest.TestCase):
    """配置、相邻状态和 breakdown 必须可以严格审计。"""

    def test_breakdown_fields_reconstruct_total_in_stable_order(self):
        event = _merge_event(4)
        previous = _analysis(0.2, 0.4, 0.6, step_index=0)
        next_analysis = _analysis(
            0.8,
            0.6,
            0.4,
            step_index=1,
            incoming_transition_key=previous.transition_key,
        )

        reward, breakdown = compute_reward(
            previous,
            next_analysis,
            _physics(event),
        )
        values = breakdown.to_dict()

        self.assertEqual(tuple(values), REWARD_BREAKDOWN_FIELDS)
        self.assertAlmostEqual(
            reward,
            (
                breakdown.task_reward
                + breakdown.potential_shaping_reward
                + breakdown.terminal_penalty
            ),
            places=12,
        )
        self.assertEqual(values['merge_event_count'], 1)
        self.assertTrue(
            all(
                math.isfinite(float(value))
                for value in values.values()
            )
        )

    def test_nonterminal_requires_immediately_adjacent_matching_analysis(self):
        previous = _analysis(0.0, 0.0, 0.0, step_index=3)

        with self.assertRaises(ValueError):
            compute_reward(previous, None, _physics())
        with self.assertRaises(ValueError):
            compute_reward(
                previous,
                _analysis(0.0, 0.0, 0.0, step_index=5),
                _physics(),
            )
        with self.assertRaises(ValueError):
            compute_reward(
                previous,
                _analysis(
                    0.0,
                    0.0,
                    0.0,
                    step_index=4,
                    worker_id=1,
                ),
                _physics(),
            )
        with self.assertRaises(ValueError):
            compute_reward(
                previous,
                _analysis(
                    0.0,
                    0.0,
                    0.0,
                    step_index=4,
                    incoming_transition_key=previous.transition_key,
                    fingerprint='sha256:different',
                ),
                _physics(),
            )

    def test_config_validation_and_pickle_roundtrip(self):
        config = RewardConfig(
            gamma=0.87,
            lambda_phi=0.4,
            capacity_weight=0.5,
            recoverability_weight=0.3,
            chain_readiness_weight=0.2,
            terminal_penalty=-1.0,
        )
        self.assertEqual(pickle.loads(pickle.dumps(config)), config)

        with self.assertRaises(ValueError):
            RewardConfig(gamma=1.01)
        with self.assertRaises(ValueError):
            RewardConfig(lambda_phi=float('nan'))
        with self.assertRaises(ValueError):
            RewardConfig(
                capacity_weight=0.5,
                recoverability_weight=0.3,
                chain_readiness_weight=0.1,
            )
        with self.assertRaises(ValueError):
            RewardConfig(terminal_penalty=1.0)


if __name__ == '__main__':
    unittest.main()
