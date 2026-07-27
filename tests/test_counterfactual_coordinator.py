"""主进程反事实协调器、共享预算与降级路径专项测试。"""

from __future__ import annotations

import json
import unittest
from concurrent.futures import Future

import torch

from test_counterfactual_runner import _build_fixture

from daxigua_rl.attribution.causal_replay import CausalReplayBuffer
from daxigua_rl.attribution.counterfactual import (
    CounterfactualBranchResult,
    CounterfactualConfig,
    CounterfactualResult,
    FrozenGNNModelConfig,
)
from daxigua_rl.attribution.counterfactual_proposal import (
    CounterfactualHistoryEntry,
    CounterfactualProposal,
    stable_counterfactual_proposal_id,
)
from daxigua_rl.attribution.schema import (
    AttributionEvent,
    AttributionEventKey,
    AttributionEvidence,
    Contributor,
)
from daxigua_rl.attribution.state_analyzer import StateAnalyzerConfig
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.reward import RewardConfig
from daxigua_rl.training.counterfactual_coordinator import (
    CounterfactualCoordinator,
    recommended_counterfactual_worker_count,
)


class _ImmediateExecutor:
    """在 submit 内完成 future，覆盖协调器的零等待收割路径。"""

    def __init__(self):
        self.futures = []

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        self.futures.append(future)
        return future


class _DeferredExecutor:
    """永不主动运行 future，供 close 的取消和元数据清理测试使用。"""

    def __init__(self):
        self.futures = []

    def submit(self, _function, *_args):
        future = Future()
        self.futures.append(future)
        return future


def _completed_coordinator_runner(task):
    branches = tuple(
        CounterfactualBranchResult(
            action_offset=action_offset,
            status='completed',
            objective_return=(
                10.0
                if action_offset == task.actual_action_offset
                else 4.0 - action_offset * 0.01
            ),
            simulated_steps=1,
        )
        for action_offset in task.requested_action_offsets
    )
    return CounterfactualResult(
        task_id=task.task_id,
        status='completed',
        actual_action_offset=task.actual_action_offset,
        original_reproduced=True,
        branches=branches,
        simulated_steps=len(branches),
    )


def _unreproduced_coordinator_runner(task):
    return CounterfactualResult(
        task_id=task.task_id,
        status='failed',
        actual_action_offset=task.actual_action_offset,
        original_reproduced=False,
        branches=(),
        simulated_steps=0,
        failure_reason='original_reproduction_mismatch',
    )


def _raising_coordinator_runner(_task):
    raise RuntimeError('intentional coordinator runner failure')


def _zero_model():
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
    return model, model_config


def _refresh(coordinator, *, version='coordinator-target-v1'):
    model, model_config = _zero_model()
    payload = coordinator.refresh_target_policy(
        model=model,
        model_config=model_config,
        policy_version=version,
        gamma=0.99,
        max_physics_frames=360,
        stable_frames=15,
        reward_config=RewardConfig(gamma=0.99),
        state_analyzer_config=StateAnalyzerConfig(),
    )
    return model, model_config, payload


def _proposal(*, event_index=1):
    _payload, task, context = _build_fixture()
    event_id = AttributionEventKey(0, 0, event_index)
    contributor = Contributor(
        transition_key=context.transition_key,
        action_offset=context.actual_action_offset,
        action_index=context.actual_action_index,
        fruit_id=task.factual_outcome.drop_result.fruit_id,
        evidence_type='coordinator_test',
        raw_evidence_weight=1.0,
        contribution_weight=1.0,
        role='material',
    )
    event = AttributionEvent(
        event_id=event_id,
        episode_key=(0, 0),
        attribution_version='coordinator-attribution-v1',
        tracker_config_fingerprint='coordinator-tracker-v1',
        detected_step=context.transition_key.step_index,
        resolved_step=context.transition_key.step_index + 4,
        event_type='DIRECT_TRIGGER',
        status='confirmed',
        sign=1,
        target_fruit_ids=(
            task.factual_outcome.drop_result.fruit_id,
        ),
        contributors=(contributor,),
        utility=4.0,
        link_confidence=0.80,
        placement_confidence=0.73,
        evidence=AttributionEvidence(
            reason_codes=('coordinator_test',),
        ),
        budget_key=event_id,
        resolution_reason='coordinator_test',
    )
    history_entry = CounterfactualHistoryEntry(
        transition_key=context.transition_key,
        context=context,
        snapshot=task.snapshot,
        factual_outcome=task.factual_outcome,
        alternative_action_offsets=(
            task.alternative_action_offsets
        ),
    )
    trace = (history_entry,)
    candidate_keys = (context.transition_key,)
    proposal_id = stable_counterfactual_proposal_id(
        budget_key=event_id,
        representative_event=event,
        contributor=contributor,
        context=context,
        snapshot=task.snapshot,
        factual_outcome=task.factual_outcome,
        alternative_action_offsets=(
            task.alternative_action_offsets
        ),
        trigger_reasons=('random_rule_audit',),
        coalition_trace_entries=trace,
        coalition_candidate_keys=candidate_keys,
    )
    return CounterfactualProposal(
        proposal_id=proposal_id,
        representative_event=event,
        budget_key=event_id,
        contributor=contributor,
        context=context,
        snapshot=task.snapshot,
        factual_outcome=task.factual_outcome,
        actual_action_offset=context.actual_action_offset,
        alternative_action_offsets=(
            task.alternative_action_offsets
        ),
        trigger_reasons=('random_rule_audit',),
        coalition_trace_entries=trace,
        coalition_candidate_keys=candidate_keys,
        utility=event.utility,
        confidence=min(
            event.link_confidence,
            event.placement_confidence,
        ),
        delay=event.delay,
    )


def _coordinator(
        runner,
        *,
        executor=None,
        config=None,
        replay=None):
    return CounterfactualCoordinator(
        causal_replay_buffer=(
            replay
            if replay is not None
            else CausalReplayBuffer(capacity=64, seed=7)
        ),
        rollout_worker_count=1,
        cpu_count=8,
        worker_count=1,
        scheduler_config=config,
        executor=executor or _ImmediateExecutor(),
        runner=runner,
    )


class CounterfactualCoordinatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal = _proposal()

    def test_recommended_workers_and_zero_worker_degradation(self):
        self.assertEqual(
            recommended_counterfactual_worker_count(
                cpu_count=16,
                rollout_worker_count=8,
            ),
            4,
        )
        self.assertEqual(
            recommended_counterfactual_worker_count(
                cpu_count=4,
                rollout_worker_count=3,
            ),
            0,
        )
        replay = CausalReplayBuffer(capacity=8)
        coordinator = CounterfactualCoordinator(
            causal_replay_buffer=replay,
            rollout_worker_count=1,
            cpu_count=3,
            worker_count=None,
        )
        try:
            self.assertFalse(coordinator.enabled)
            self.assertIsNone(_refresh(coordinator)[2])
            self.assertEqual(coordinator.record_real_steps(12), 12)
            submission = coordinator.submit(self.proposal)
            self.assertFalse(submission.accepted)
            self.assertEqual(
                submission.drop_reason,
                'disabled_no_workers',
            )
            reservation = coordinator.reserve_external_tokens(
                'disabled-shapley',
                1,
            )
            self.assertFalse(reservation.accepted)
            self.assertEqual(
                reservation.drop_reason,
                'external_disabled_no_workers',
            )
            json.dumps(coordinator.checkpoint_state())
        finally:
            coordinator.close()

    def test_completed_result_inserts_only_real_counterfactual_labels(self):
        replay = CausalReplayBuffer(capacity=16, seed=9)
        coordinator = _coordinator(
            _completed_coordinator_runner,
            replay=replay,
        )
        try:
            _refresh(coordinator)
            coordinator.record_real_steps(400)
            submission = coordinator.submit(self.proposal)

            self.assertTrue(submission.accepted)
            self.assertEqual(len(replay), 2)
            samples = replay.sample(2)
            self.assertTrue(all(
                sample.supervision_kind == 'counterfactual'
                for sample in samples
            ))
            self.assertEqual(
                {sample.comparison_action_offset for sample in samples},
                set(self.proposal.alternative_action_offsets),
            )

            stats = coordinator.stats
            self.assertEqual(stats.pending_task_count, 0)
            self.assertEqual(stats.active_task_ids, ())
            self.assertEqual(stats.cumulative.results_completed, 1)
            self.assertEqual(stats.cumulative.reproduction_passed, 1)
            self.assertEqual(stats.cumulative.label_ready_results, 1)
            self.assertEqual(stats.cumulative.samples_generated, 2)
            self.assertEqual(stats.cumulative.samples_inserted, 2)
            self.assertEqual(stats.scheduler.tokens_consumed, 3)
            self.assertTrue(stats.hard_budget_respected)
        finally:
            coordinator.close()

    def test_unreproduced_and_runner_exception_never_create_labels(self):
        replay = CausalReplayBuffer(capacity=16)
        unreproduced = _coordinator(
            _unreproduced_coordinator_runner,
            replay=replay,
        )
        try:
            _refresh(unreproduced)
            unreproduced.record_real_steps(400)
            self.assertTrue(
                unreproduced.submit(self.proposal).accepted
            )
            stats = unreproduced.stats
            self.assertEqual(len(replay), 0)
            self.assertEqual(stats.cumulative.results_failed, 1)
            self.assertEqual(stats.cumulative.reproduction_failed, 1)
            self.assertEqual(stats.cumulative.label_ready_results, 0)
            self.assertEqual(stats.cumulative.samples_inserted, 0)
        finally:
            unreproduced.close()

        replay = CausalReplayBuffer(capacity=16)
        raised = _coordinator(
            _raising_coordinator_runner,
            replay=replay,
        )
        try:
            _refresh(raised)
            raised.record_real_steps(400)
            self.assertTrue(raised.submit(self.proposal).accepted)
            stats = raised.stats
            reasons = dict(stats.cumulative.drop_reason_counts)
            self.assertEqual(len(replay), 0)
            self.assertEqual(stats.pending_task_count, 0)
            self.assertEqual(
                stats.cumulative.pending_cleanups_without_result,
                1,
            )
            self.assertEqual(reasons['runner_failure'], 1)
            self.assertEqual(reasons['pending_without_result'], 1)
        finally:
            raised.close()

    def test_target_refresh_replaces_frozen_version_and_fingerprint(self):
        coordinator = _coordinator(
            _completed_coordinator_runner
        )
        try:
            _model, _config, first = _refresh(
                coordinator,
                version='target-sync-1',
            )
            model, model_config = _zero_model()
            with torch.no_grad():
                next(model.parameters()).add_(1.0)
            second = coordinator.refresh_target_policy(
                model=model,
                model_config=model_config,
                policy_version='target-sync-2',
                gamma=0.99,
                max_physics_frames=360,
                stable_frames=15,
                reward_config=RewardConfig(gamma=0.99),
                state_analyzer_config=StateAnalyzerConfig(),
            )

            self.assertEqual(
                coordinator.stats.target_policy_version,
                'target-sync-2',
            )
            self.assertNotEqual(
                first.fingerprint,
                second.fingerprint,
            )
            self.assertEqual(
                coordinator.stats.cumulative.target_refreshes,
                2,
            )
        finally:
            coordinator.close()

    def test_budget_rejection_window_reset_and_checkpoint_are_lightweight(self):
        coordinator = _coordinator(
            _completed_coordinator_runner
        )
        try:
            unavailable = coordinator.submit(self.proposal)
            self.assertEqual(
                unavailable.drop_reason,
                'target_policy_unavailable',
            )
            _refresh(coordinator)
            coordinator.record_real_steps(256)
            budget_rejected = coordinator.submit(self.proposal)
            self.assertFalse(budget_rejected.accepted)
            self.assertEqual(
                budget_rejected.drop_reason,
                'soft_token_budget',
            )

            first_window = coordinator.snapshot_stats(
                reset_window=True
            )
            self.assertEqual(
                first_window.window.proposals_received,
                2,
            )
            self.assertEqual(
                coordinator.stats.window.proposals_received,
                0,
            )

            encoded = json.dumps(
                coordinator.checkpoint_state(),
                sort_keys=True,
            )
            self.assertNotIn('state_dict_bytes', encoded)
            self.assertNotIn('snapshot', encoded)
            self.assertNotIn('context', encoded)
            self.assertNotIn('future', encoded)
            json.dumps(coordinator.summary())
        finally:
            coordinator.close()

    def test_external_shapley_wrapper_uses_same_hard_budget(self):
        coordinator = _coordinator(
            _completed_coordinator_runner
        )
        try:
            coordinator.record_real_steps(1_000)
            reserved = coordinator.reserve_external_tokens(
                'coordinator-shapley',
                50,
            )
            self.assertTrue(reserved.accepted)
            settled = coordinator.settle_external_tokens(
                'coordinator-shapley',
                40,
            )
            self.assertTrue(settled.accepted)
            repeated = coordinator.refund_external_tokens(
                'coordinator-shapley'
            )
            self.assertFalse(repeated.accepted)

            stats = coordinator.stats
            self.assertEqual(stats.scheduler.tokens_reserved, 0)
            self.assertEqual(stats.scheduler.tokens_consumed, 40)
            self.assertEqual(stats.scheduler.tokens_refunded, 10)
            self.assertEqual(
                stats.scheduler.external_reservations_settled,
                1,
            )
            self.assertTrue(stats.hard_budget_respected)
        finally:
            coordinator.close()

    def test_close_cancels_pending_and_is_idempotent(self):
        replay = CausalReplayBuffer(capacity=16)
        executor = _DeferredExecutor()
        config = CounterfactualConfig(
            min_real_steps=1,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
        )
        coordinator = _coordinator(
            _completed_coordinator_runner,
            executor=executor,
            config=config,
            replay=replay,
        )
        _refresh(coordinator)
        coordinator.record_real_steps(100)
        submission = coordinator.submit(self.proposal)
        self.assertTrue(submission.accepted)
        self.assertEqual(coordinator.stats.pending_task_count, 1)

        first = coordinator.close(wait=True)
        second = coordinator.close(wait=True)

        self.assertEqual(first, second)
        self.assertTrue(first.closed)
        self.assertEqual(first.pending_task_count, 0)
        self.assertEqual(first.active_task_ids, ())
        self.assertEqual(first.scheduler.tokens_reserved, 0)
        self.assertEqual(first.scheduler.tokens_refunded, 30)
        self.assertEqual(len(replay), 0)


if __name__ == '__main__':
    unittest.main()
