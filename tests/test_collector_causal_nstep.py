"""collector 的 n-step 与规则因果回放接入测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from daxigua_rl import (
    DaxiguaEnv,
    DaxiguaEnvConfig,
    GraphBuilder,
)
from daxigua_rl.attribution.causal_replay import CausalReplayBuffer
from daxigua_rl.attribution import ANALYSIS_ACTION_COUNT
from daxigua_rl.attribution.counterfactual_proposal import (
    CounterfactualProposal,
)
from daxigua_rl.training import (
    ParallelRolloutCollector,
    ReplayBuffer,
    RolloutCollector,
)
from daxigua_rl.training.collector import EpsilonGreedyPolicy
from daxigua_rl.training.parallel_collector import (
    _load_from_bytes,
    _save_to_bytes,
    _worker_torch_thread_counts,
)


class _AlwaysLeftPolicy(EpsilonGreedyPolicy):

    def random_action_offset(self, action_count):
        if int(action_count) <= 0:
            raise ValueError('action_count must be positive')
        return 0


class CollectorCausalNStepTest(unittest.TestCase):

    def test_three_step_tail_survives_collect_boundaries_and_truncation_flushes(self):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(
            action_count=7,
            max_physics_frames=240,
            stable_frames=6,
        ))
        replay = ReplayBuffer(capacity=16, seed=3)
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay,
            policy=_AlwaysLeftPolicy(seed=4),
            attribution_tracker=False,
            n_step=3,
            gamma=0.5,
        )
        collector.reset(seed=4, fruit_queue=(1, 1, 1, 1))

        first = collector.collect_steps(1, epsilon=1.0)
        second = collector.collect_steps(1, epsilon=1.0)

        self.assertEqual(first.steps, 1)
        self.assertEqual(first.replay_transitions_emitted, 0)
        self.assertEqual(first.n_step_pending_count, 1)
        self.assertEqual(second.steps, 1)
        self.assertEqual(second.replay_transitions_emitted, 0)
        self.assertEqual(second.n_step_pending_count, 2)
        self.assertEqual(len(replay), 0)

        # 第三步强制形成 truncated episode 边界；accumulator 必须一次发射
        # 3/2/1 三条尾部，且 truncated 仍允许 bootstrap。
        env.config.max_physics_frames = 1
        final = collector.collect_steps(1, epsilon=1.0)
        transitions = replay.to_tuple()

        self.assertEqual(final.steps, 1)
        self.assertEqual(final.truncated_episodes, 1)
        self.assertEqual(final.replay_transitions_emitted, 3)
        self.assertEqual(final.n_step_pending_count, 0)
        self.assertEqual(
            tuple(item.bootstrap_steps for item in transitions),
            (3, 2, 1),
        )
        self.assertTrue(all(item.truncated for item in transitions))
        self.assertTrue(all(item.next_graph is not None for item in transitions))
        self.assertTrue(all(item.can_bootstrap for item in transitions))

    def test_rule_samples_keep_action_pre_state_and_policy_provenance(self):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(
            action_count=ANALYSIS_ACTION_COUNT,
            max_physics_frames=300,
            stable_frames=6,
        ))
        replay = ReplayBuffer(capacity=16, seed=5)
        causal_replay = CausalReplayBuffer(capacity=16, seed=6)
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay,
            policy=_AlwaysLeftPolicy(seed=7),
            causal_replay_buffer=causal_replay,
            n_step=3,
            policy_version='policy-test-1',
        )
        collector.reset(fruit_queue=(1, 1, 1, 1))

        first = collector.collect_steps(1, epsilon=1.0)
        stats = collector.collect_steps(1, epsilon=1.0)
        samples = causal_replay.to_tuple()

        self.assertEqual(first.steps, 1)
        self.assertEqual(first.causal_rule_samples_generated, 0)
        self.assertEqual(first.causal_context_count, 1)
        self.assertEqual(stats.steps, 1)
        self.assertEqual(stats.replay_transitions_emitted, 0)
        self.assertEqual(stats.n_step_pending_count, 2)
        self.assertEqual(stats.causal_rule_build_calls, 1)
        self.assertGreaterEqual(stats.causal_rule_input_event_count, 1)
        self.assertGreaterEqual(stats.causal_rule_budget_count, 1)
        self.assertEqual(
            stats.causal_rule_samples_generated,
            len(samples),
        )
        self.assertEqual(stats.causal_samples_pushed, len(samples))
        self.assertEqual(stats.causal_samples_emitted, len(samples))
        self.assertEqual(stats.causal_buffer_size, len(samples))
        self.assertGreater(len(samples), 0)
        for sample in samples:
            self.assertEqual(sample.policy_version, 'policy-test-1')
            self.assertEqual(sample.actual_action_offset, 0)
            self.assertEqual(
                sample.comparison_action_offset,
                ANALYSIS_ACTION_COUNT - 1,
            )
            self.assertEqual(
                sample.graph.action_indices.tolist(),
                list(range(ANALYSIS_ACTION_COUNT)),
            )

        # episode 边界必须先让 tracker 解决事件并构建标签，再丢弃整局 context。
        env.config.max_physics_frames = 1
        boundary = collector.collect_steps(1, epsilon=1.0)
        self.assertEqual(boundary.truncated_episodes, 1)
        self.assertEqual(boundary.causal_context_count, 0)

        # parallel worker 把两类样本放进同一次 torch.save；pickle memo 必须让
        # 同一动作图在解码后仍共享引用，主进程分别 extend 两池也不得复制。
        first_transition = replay.to_tuple()[0]
        first_sample = next(
            sample
            for sample in causal_replay.to_tuple()
            if sample.transition_key.step_index == 0
        )
        self.assertIs(first_transition.graph, first_sample.graph)
        payload = _save_to_bytes((
            (first_transition,),
            (first_sample,),
        ))
        transitions, causal_samples = _load_from_bytes(payload)
        self.assertIs(transitions[0].graph, causal_samples[0].graph)

    def test_manual_reset_and_close_flush_short_tail_without_silent_loss(self):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(
            action_count=7,
            max_physics_frames=240,
            stable_frames=6,
        ))
        replay = ReplayBuffer(capacity=16, seed=11)
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay,
            policy=_AlwaysLeftPolicy(seed=12),
            attribution_tracker=False,
            n_step=3,
            gamma=0.99,
        )
        collector.reset(seed=12, fruit_queue=(1, 1, 1, 1))

        collected = collector.collect_steps(2, epsilon=1.0)
        self.assertEqual(collected.replay_transitions_emitted, 0)
        self.assertEqual(collected.n_step_pending_count, 2)

        collector.reset(seed=13)
        self.assertEqual(
            tuple(item.bootstrap_steps for item in replay.to_tuple()),
            (2, 1),
        )
        after_reset = collector.collect_steps(1, epsilon=1.0)
        self.assertEqual(after_reset.replay_transitions_emitted, 2)
        self.assertEqual(after_reset.n_step_forced_flush_emitted, 2)
        self.assertEqual(after_reset.n_step_pending_count, 1)

        collector.close()
        self.assertEqual(collector.close_n_step_flush_emitted, 1)
        self.assertEqual(
            tuple(
                item.bootstrap_steps
                for item in collector.close_n_step_transitions
            ),
            (1,),
        )
        self.assertEqual(len(replay), 3)

    def test_causal_replay_requires_tracker_and_default_n_step_is_legacy_one(self):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(action_count=7))
        replay = ReplayBuffer(capacity=4)
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay,
            attribution_tracker=False,
        )
        self.assertEqual(collector.n_step, 1)

        with self.assertRaises(ValueError):
            RolloutCollector(
                env=env,
                graph_builder=GraphBuilder(),
                replay_buffer=replay,
                attribution_tracker=False,
                causal_replay_buffer=CausalReplayBuffer(capacity=4),
            )

    def test_counterfactual_default_off_never_captures_snapshot(self):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(
            action_count=ANALYSIS_ACTION_COUNT,
            max_physics_frames=240,
            stable_frames=6,
        ))
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=ReplayBuffer(capacity=8, seed=21),
            policy=_AlwaysLeftPolicy(seed=22),
        )
        collector.reset(seed=22, fruit_queue=(1, 1, 1, 1))

        with patch.object(
                env.game,
                'capture_snapshot',
                wraps=env.game.capture_snapshot) as capture:
            stats = collector.collect_steps(2, epsilon=1.0)

        capture.assert_not_called()
        self.assertEqual(stats.counterfactual_snapshot_calls, 0)
        self.assertEqual(stats.counterfactual_snapshot_failures, 0)
        self.assertEqual(stats.counterfactual_history_size, 0)
        self.assertEqual(stats.counterfactual_proposal_build_calls, 0)
        self.assertEqual(stats.counterfactual_proposals_generated, 0)
        self.assertEqual(collector.drain_counterfactual_proposals(), ())

    def test_single_collector_generates_drains_and_clears_proposals(self):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(
            action_count=ANALYSIS_ACTION_COUNT,
            max_physics_frames=300,
            stable_frames=6,
        ))
        replay = ReplayBuffer(capacity=16, seed=23)
        causal_replay = CausalReplayBuffer(capacity=16, seed=24)
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay,
            policy=_AlwaysLeftPolicy(seed=25),
            causal_replay_buffer=causal_replay,
            counterfactual_enabled=True,
            counterfactual_ring_size=32,
        )
        collector.reset(seed=25, fruit_queue=(1, 1, 1, 1))

        first = collector.collect_steps(1, epsilon=1.0)
        second = collector.collect_steps(1, epsilon=1.0)
        proposals = collector.drain_counterfactual_proposals()

        self.assertEqual(first.counterfactual_snapshot_calls, 1)
        self.assertEqual(first.counterfactual_history_size, 1)
        self.assertEqual(second.counterfactual_snapshot_calls, 1)
        self.assertEqual(second.counterfactual_snapshot_failures, 0)
        self.assertEqual(second.counterfactual_history_size, 2)
        self.assertGreater(second.counterfactual_proposals_generated, 0)
        self.assertEqual(
            len(proposals),
            second.counterfactual_proposals_generated,
        )
        self.assertTrue(all(
            isinstance(proposal, CounterfactualProposal)
            for proposal in proposals
        ))
        payload = _save_to_bytes((
            replay.to_tuple(),
            causal_replay.to_tuple(),
            proposals,
        ))
        loaded_transitions, loaded_samples, loaded_proposals = (
            _load_from_bytes(payload)
        )
        loaded_samples_by_key = {
            sample.transition_key: sample
            for sample in loaded_samples
        }
        matching = tuple(
            proposal
            for proposal in loaded_proposals
            if proposal.transition_key in loaded_samples_by_key
        )
        self.assertTrue(matching)
        self.assertTrue(any(
            proposal.context.graph
            is loaded_samples_by_key[proposal.transition_key].graph
            for proposal in matching
        ))
        self.assertEqual(len(loaded_transitions), len(replay))
        self.assertEqual(collector.drain_counterfactual_proposals(), ())

        # episode 结束后先完成本步 proposal 生成，再清除该局历史。
        env.config.max_physics_frames = 1
        boundary = collector.collect_steps(1, epsilon=1.0)
        self.assertEqual(boundary.truncated_episodes, 1)
        self.assertEqual(boundary.counterfactual_snapshot_calls, 1)
        self.assertEqual(boundary.counterfactual_history_size, 0)

    def test_single_collector_throttles_only_proposal_transfer(self):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(
            action_count=ANALYSIS_ACTION_COUNT,
            max_physics_frames=300,
            stable_frames=6,
        ))
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=ReplayBuffer(capacity=16, seed=125),
            policy=_AlwaysLeftPolicy(seed=126),
            causal_replay_buffer=CausalReplayBuffer(
                capacity=16,
                seed=127,
            ),
            counterfactual_enabled=True,
            counterfactual_ring_size=32,
            counterfactual_proposal_sample_rate=0.0,
        )
        collector.reset(seed=126, fruit_queue=(1, 1, 1, 1))

        collector.collect_steps(1, epsilon=1.0)
        stats = collector.collect_steps(1, epsilon=1.0)

        self.assertGreater(stats.counterfactual_proposals_generated, 0)
        self.assertEqual(
            stats.counterfactual_proposals_transfer_selected,
            0,
        )
        self.assertEqual(
            stats.counterfactual_proposals_transfer_throttled,
            stats.counterfactual_proposals_generated,
        )
        self.assertEqual(
            dict(stats.counterfactual_proposal_skip_reason_counts).get(
                'transfer_throttle',
                0,
            ),
            stats.counterfactual_proposals_generated,
        )
        self.assertEqual(collector.drain_counterfactual_proposals(), ())
        # 规则因果样本仍完整进入独立 replay，抽样只影响物理 proposal。
        self.assertGreater(len(collector.causal_replay_buffer), 0)

    def test_counterfactual_ring_expiry_reports_skip_without_fake_proposal(self):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(
            action_count=ANALYSIS_ACTION_COUNT,
            max_physics_frames=300,
            stable_frames=6,
        ))
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=ReplayBuffer(capacity=8, seed=25),
            policy=_AlwaysLeftPolicy(seed=26),
            counterfactual_enabled=True,
            counterfactual_ring_size=1,
        )
        collector.reset(seed=26, fruit_queue=(1, 1, 1, 1))

        collector.collect_steps(1, epsilon=1.0)
        stats = collector.collect_steps(1, epsilon=1.0)

        self.assertEqual(stats.counterfactual_history_evictions, 1)
        self.assertEqual(stats.counterfactual_history_size, 1)
        self.assertEqual(stats.counterfactual_proposals_generated, 0)
        self.assertGreaterEqual(
            dict(stats.counterfactual_proposal_skip_reason_counts).get(
                'missing_history',
                0,
            ),
            1,
        )
        self.assertEqual(collector.drain_counterfactual_proposals(), ())

    def test_parallel_workers_preserve_independent_three_step_tails(self):
        replay = ReplayBuffer(capacity=32, seed=8)
        causal_replay = CausalReplayBuffer(capacity=32, seed=9)
        collector = ParallelRolloutCollector(
            worker_count=2,
            env_config=DaxiguaEnvConfig(
                action_count=ANALYSIS_ACTION_COUNT,
                physics_fps=30,
                max_physics_frames=80,
                stable_frames=3,
                space_iterations=8,
            ),
            replay_buffer=replay,
            causal_replay_buffer=causal_replay,
            n_step=3,
            gamma=0.99,
            seed=10,
            counterfactual_enabled=True,
        )
        try:
            thread_counts = tuple(
                executor.submit(
                    _worker_torch_thread_counts
                ).result()
                for executor in collector._executors
            )
            first = collector.collect_steps(4, epsilon=1.0)
            second = collector.collect_steps(2, epsilon=1.0)
            proposals = collector.drain_counterfactual_proposals()
            self.assertEqual(
                collector.drain_counterfactual_proposals(),
                (),
            )
        finally:
            finalizations = collector.close()

        self.assertTrue(all(
            intra_op_threads == 1
            for intra_op_threads, _interop_threads in thread_counts
        ))
        self.assertEqual(first.steps, 4)
        self.assertEqual(len(first.transition_keys), 4)
        self.assertEqual(first.replay_transitions_emitted, 0)
        self.assertEqual(first.n_step_pending_count, 4)
        self.assertEqual(first.counterfactual_snapshot_calls, 4)
        self.assertEqual(first.counterfactual_snapshot_failures, 0)
        self.assertEqual(
            first.counterfactual_proposals_serialized,
            first.counterfactual_proposals_generated,
        )
        self.assertEqual(second.steps, 2)
        self.assertEqual(len(second.transition_keys), 2)
        self.assertEqual(second.counterfactual_snapshot_calls, 2)
        self.assertEqual(second.counterfactual_snapshot_failures, 0)
        self.assertEqual(
            second.counterfactual_proposals_serialized,
            second.counterfactual_proposals_generated,
        )
        self.assertEqual(
            len(proposals),
            (
                first.counterfactual_proposals_serialized
                + second.counterfactual_proposals_serialized
            ),
        )
        self.assertTrue(all(
            isinstance(proposal, CounterfactualProposal)
            for proposal in proposals
        ))
        proposal_bytes = (
            first.counterfactual_proposal_serialized_bytes
            + second.counterfactual_proposal_serialized_bytes
        )
        # 真实随机轨迹可能在这 6 步内没有命中稀疏触发白名单；命中时必须
        # 完整序列化，未命中时则不能伪造 payload 字节。
        self.assertEqual(proposal_bytes > 0, bool(proposals))
        # 某个 worker 的第三次投放可能刚好形成真实终局，从而额外发射
        # 2/1-step 尾巴；无论是否终局，累计 raw step 都必须精确分解为
        # 已发射 replay 与每个 worker 尚待补齐的尾巴。
        self.assertGreaterEqual(second.replay_transitions_emitted, 2)
        self.assertEqual(
            second.replay_transitions_emitted
            + second.n_step_pending_count,
            first.steps + second.steps,
        )
        self.assertEqual(
            second.replay_transitions_emitted,
            6 - second.n_step_pending_count,
        )
        self.assertEqual(
            sum(
                summary.n_step_flush_emitted
                for summary in finalizations
            ),
            second.n_step_pending_count,
        )
        self.assertEqual(
            len(replay),
            first.steps + second.steps,
        )
        self.assertEqual(
            len(replay) - sum(
                summary.n_step_flush_emitted
                for summary in finalizations
            ),
            second.replay_transitions_emitted,
        )
        self.assertTrue(all(
            1 <= item.bootstrap_steps <= 3
            for item in replay.to_tuple()
        ))
        self.assertEqual(
            second.causal_buffer_size,
            len(causal_replay),
        )


if __name__ == '__main__':
    unittest.main()
