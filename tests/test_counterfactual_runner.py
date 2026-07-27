"""真实物理反事实 runner、冻结 target policy 与标签转换测试。"""

from __future__ import annotations

import io
import multiprocessing
import pickle
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from daxigua.core.engine import HeadlessGame
from daxigua_rl.attribution.causal_replay import CausalTransitionContext
from daxigua_rl.attribution.counterfactual import (
    CounterfactualConfig,
    CounterfactualTask,
    FrozenGNNModelConfig,
    FrozenTargetPolicyPayload,
    engine_action_outcome_fingerprint,
    stable_counterfactual_task_id,
)
from daxigua_rl.attribution.counterfactual_runner import (
    counterfactual_result_to_causal_samples,
    freeze_target_policy_payload,
    run_counterfactual_task,
)
from daxigua_rl.attribution.schema import AttributionEventKey
from daxigua_rl.attribution.state_analyzer import (
    StateAnalyzer,
    StateAnalyzerConfig,
)
from daxigua_rl.graph import GraphBuilder
from daxigua_rl.graph.tensor import graph_to_tensor
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.reward import RewardConfig
from daxigua_rl.training.identity import TransitionKey


def _build_task(
        *,
        snapshot,
        factual_outcome,
        target_policy,
        event_index=1,
        actual_action_offset=0,
        alternatives=(3, 9),
        horizon=2):
    transition_key = TransitionKey(
        0,
        0,
        snapshot.episode.step_count,
    )
    config_fingerprint = CounterfactualConfig().fingerprint
    factual_fingerprint = engine_action_outcome_fingerprint(
        factual_outcome
    )
    task_id = stable_counterfactual_task_id(
        budget_key=AttributionEventKey(0, 0, event_index),
        transition_key=transition_key,
        snapshot_checksum=snapshot.checksum,
        factual_outcome_fingerprint=factual_fingerprint,
        target_policy_fingerprint=target_policy.fingerprint,
        actual_action_offset=actual_action_offset,
        alternative_action_offsets=alternatives,
        trigger_reasons=('random_rule_audit',),
        attribution_version='runner-test-v1',
        config_fingerprint=config_fingerprint,
    )
    return CounterfactualTask(
        task_id=task_id,
        budget_key=AttributionEventKey(0, 0, event_index),
        transition_key=transition_key,
        snapshot=snapshot,
        factual_outcome=factual_outcome,
        factual_outcome_fingerprint=factual_fingerprint,
        target_policy=target_policy,
        actual_action_offset=actual_action_offset,
        alternative_action_offsets=tuple(alternatives),
        trigger_reasons=('random_rule_audit',),
        priority=1.0,
        estimated_tokens=horizon * (1 + len(alternatives)),
        horizon=horizon,
        created_real_step=0,
        attribution_version='runner-test-v1',
        scheduler_config_fingerprint=config_fingerprint,
        label_confidence=0.73,
        attribution_delay=4,
    )


def _zero_target_payload(*, max_physics_frames=360):
    model_config = FrozenGNNModelConfig(
        hidden_dim=8,
        message_layers=1,
        activation='silu',
        dropout=0.0,
    )
    model = GNNQNetwork(**model_config.model_kwargs)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    return freeze_target_policy_payload(
        model=model,
        model_config=model_config,
        policy_version='target-zero-v1',
        gamma=0.99,
        max_physics_frames=max_physics_frames,
        stable_frames=15,
        reward_config=RewardConfig(gamma=0.99),
        state_analyzer_config=StateAnalyzerConfig(),
    )


def _build_fixture():
    payload = _zero_target_payload()
    game = HeadlessGame(seed=17)
    game.reset(seed=17, fruit_queue=(1, 2, 3, 1))
    # capture 会规范化原 game；真实 outcome 必须从规范化后的同一个实例继续执行。
    snapshot = game.capture_snapshot()
    state = game.get_state()
    candidates = tuple(game.get_action_candidates(15))
    transition_key = TransitionKey(0, 0, state.step_count)
    analyzer = StateAnalyzer(
        config=payload.state_analyzer_config
    )
    analysis = analyzer.analyze(
        state,
        candidates,
        transition_key,
        stable_boundary=True,
    )
    graph = graph_to_tensor(
        GraphBuilder().build(state, candidates),
    )
    context = CausalTransitionContext(
        graph=graph,
        state_analysis=analysis,
        actual_action_offset=0,
        actual_action_index=0,
        policy_version='online-policy-v9',
    )
    factual_outcome = game.execute_action(
        candidates[0].drop_x,
        max_frames=payload.max_physics_frames,
        stable_frames=payload.stable_frames,
    )
    task = _build_task(
        snapshot=snapshot,
        factual_outcome=factual_outcome,
        target_policy=payload,
    )
    return payload, task, context


def _replay_report(
        *,
        reproduction_status,
        mismatch_codes,
        maxima=(0.0, 0.0, 0.0, 0.0, 0.0)):
    return SimpleNamespace(
        matches=False,
        reproduction_status=reproduction_status,
        mismatch_codes=tuple(mismatch_codes),
        max_merge_event_position_error=maxima[0],
        max_fruit_position_error=maxima[1],
        max_linear_velocity_error=maxima[2],
        max_orientation_error=maxima[3],
        max_angular_velocity_error=maxima[4],
    )


class CounterfactualPhysicalRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.task, cls.context = _build_fixture()

    def test_payload_is_frozen_fingerprinted_and_pickle_safe(self):
        restored = pickle.loads(pickle.dumps(self.payload))

        self.assertEqual(restored, self.payload)
        self.assertEqual(
            restored.expected_fingerprint(),
            self.payload.fingerprint,
        )
        self.assertEqual(len(self.payload.state_dict_sha256), 64)
        with self.assertRaises(ValueError):
            replace(
                self.payload,
                state_dict_bytes=b'tampered-state-dict',
            )
        with self.assertRaises(TypeError):
            FrozenTargetPolicyPayload.create(
                policy_version='tensor-is-not-bytes',
                model_config=self.payload.model_config,
                graph_builder_config=(
                    self.payload.graph_builder_config
                ),
                state_dict_bytes=torch.zeros(1),
                gamma=self.payload.gamma,
                max_physics_frames=(
                    self.payload.max_physics_frames
                ),
                stable_frames=self.payload.stable_frames,
                reward_config=self.payload.reward_config,
                state_analyzer_config=(
                    self.payload.state_analyzer_config
                ),
            )

    def test_actual_reproduction_and_two_alternative_return_deltas(self):
        result = run_counterfactual_task(self.task)

        self.assertTrue(result.original_reproduced)
        self.assertTrue(result.label_ready)
        self.assertIn(result.status, {'completed', 'partial'})
        self.assertEqual(
            tuple(
                branch.action_offset
                for branch in result.branches
            ),
            (0, 3, 9),
        )
        self.assertEqual(
            tuple(
                action_offset
                for action_offset, _delta in result.return_deltas
            ),
            (3, 9),
        )
        self.assertEqual(result.simulated_steps, 6)

        samples = counterfactual_result_to_causal_samples(
            self.task,
            result,
            self.context,
        )
        self.assertEqual(len(samples), 2)
        for sample, (comparison_offset, delta) in zip(
                samples,
                result.return_deltas):
            self.assertIs(sample.graph, self.context.graph)
            self.assertEqual(sample.actual_action_offset, 0)
            self.assertEqual(
                sample.comparison_action_offset,
                comparison_offset,
            )
            self.assertEqual(
                sample.direction,
                1 if delta > 0.0 else -1,
            )
            self.assertEqual(sample.target_delta, delta)
            self.assertEqual(sample.confidence, 0.73)
            self.assertEqual(sample.delay, 4)
            self.assertEqual(
                sample.transition_key,
                self.task.transition_key,
            )
            self.assertEqual(
                sample.policy_version,
                self.context.policy_version,
            )
            self.assertEqual(
                sample.supervision_kind,
                'counterfactual',
            )
            self.assertEqual(sample.stratum, 'counterfactual')

    def test_tampered_factual_outcome_fails_gate_and_never_labels(self):
        tampered_outcome = replace(
            self.task.factual_outcome,
            fail_count=self.task.factual_outcome.fail_count + 1,
        )
        tampered_task = _build_task(
            snapshot=self.task.snapshot,
            factual_outcome=tampered_outcome,
            target_policy=self.payload,
            event_index=1,
        )
        self.assertNotEqual(
            tampered_task.task_id,
            self.task.task_id,
        )

        result = run_counterfactual_task(tampered_task)

        self.assertEqual(result.status, 'failed')
        self.assertFalse(result.original_reproduced)
        self.assertFalse(result.label_ready)
        self.assertEqual(result.return_deltas, ())
        self.assertEqual(
            result.failure_reason,
            'original_reproduction_mismatch',
        )
        self.assertEqual(
            result.reproduction_outcome,
            'semantic_divergence_drop',
        )
        self.assertEqual(
            counterfactual_result_to_causal_samples(
                tampered_task,
                result,
                self.context,
            ),
            (),
        )

    def test_numeric_jitter_has_distinct_outcome_and_never_labels(self):
        report = _replay_report(
            reproduction_status='numeric_jitter_drop',
            mismatch_codes=(
                'merge_event_position',
                'fruit_position',
                'fruit_velocity',
                'fruit_angle',
            ),
            maxima=(0.1, 0.2, 0.3, 0.4, 0.5),
        )
        with patch.object(
                HeadlessGame,
                'compare_action_outcomes',
                return_value=report):
            result = run_counterfactual_task(self.task)

        self.assertEqual(result.status, 'failed')
        self.assertFalse(result.original_reproduced)
        self.assertFalse(result.label_ready)
        self.assertEqual(result.return_deltas, ())
        self.assertEqual(
            result.failure_reason,
            'original_reproduction_numeric_jitter',
        )
        self.assertEqual(
            result.reproduction_outcome,
            'numeric_jitter_drop',
        )
        self.assertEqual(
            result.diagnostic_codes,
            (
                'original_mismatch_merge_event_position',
                'original_mismatch_fruit_position',
                'original_mismatch_fruit_velocity',
                'original_mismatch_fruit_angle',
            ),
        )
        self.assertEqual(
            (
                result.replay_max_merge_event_position_error,
                result.replay_max_fruit_position_error,
                result.replay_max_linear_velocity_error,
                result.replay_max_orientation_error,
                result.replay_max_angular_velocity_error,
            ),
            (0.1, 0.2, 0.3, 0.4, 0.5),
        )
        self.assertEqual(
            result.branches[0].failure_reason,
            'original_reproduction_numeric_jitter',
        )
        self.assertEqual(
            counterfactual_result_to_causal_samples(
                self.task,
                result,
                self.context,
            ),
            (),
        )

    def test_frozen_policy_is_deterministic_across_repeated_runs(self):
        first = run_counterfactual_task(self.task)
        second = run_counterfactual_task(self.task)

        self.assertEqual(first, second)
        self.assertEqual(first.return_deltas, second.return_deltas)

    def test_spawn_pickle_and_physical_execution(self):
        context = multiprocessing.get_context('spawn')
        started = time.perf_counter()
        with ProcessPoolExecutor(
                max_workers=1,
                mp_context=context) as executor:
            result = executor.submit(
                run_counterfactual_task,
                self.task,
            ).result(timeout=30.0)
        elapsed = time.perf_counter() - started

        self.assertTrue(result.label_ready)
        self.assertEqual(len(result.return_deltas), 2)
        self.assertLess(elapsed, 30.0)

    def test_non_cpu_target_degrades_after_successful_factual_gate(self):
        model = GNNQNetwork(
            **self.payload.model_config.model_kwargs
        )
        if torch.cuda.is_available():
            model.to(device='cuda')
        stream = io.BytesIO()
        torch.save(
            {
                'format': 'counterfactual_cpu_state_dict_v1',
                'device': 'cuda',
                'state_dict': model.state_dict(),
            },
            stream,
        )
        bad_payload = FrozenTargetPolicyPayload.create(
            policy_version='broken-target-v1',
            model_config=self.payload.model_config,
            graph_builder_config=self.payload.graph_builder_config,
            state_dict_bytes=stream.getvalue(),
            gamma=self.payload.gamma,
            max_physics_frames=self.payload.max_physics_frames,
            stable_frames=self.payload.stable_frames,
            reward_config=self.payload.reward_config,
            state_analyzer_config=(
                self.payload.state_analyzer_config
            ),
        )
        task = _build_task(
            snapshot=self.task.snapshot,
            factual_outcome=self.task.factual_outcome,
            target_policy=bad_payload,
            event_index=1,
        )
        self.assertNotEqual(
            task.target_policy.fingerprint,
            self.task.target_policy.fingerprint,
        )
        self.assertNotEqual(task.task_id, self.task.task_id)

        result = run_counterfactual_task(task)

        self.assertEqual(result.status, 'failed')
        self.assertTrue(result.original_reproduced)
        self.assertFalse(result.label_ready)
        self.assertEqual(
            result.failure_reason,
            'target_policy_initialization_failure',
        )
        self.assertEqual(result.return_deltas, ())


if __name__ == '__main__':
    unittest.main()
