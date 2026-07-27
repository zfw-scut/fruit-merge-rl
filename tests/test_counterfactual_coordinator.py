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
    effective_cpu_count,
    recommended_counterfactual_worker_count,
)


class _ImmediateExecutor:
    """在 submit 内完成 future，覆盖协调器的零等待收割路径。"""

    def __init__(self):
        self.futures = []
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))
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
    branch = CounterfactualBranchResult(
        action_offset=task.actual_action_offset,
        status='failed',
        objective_return=None,
        simulated_steps=1,
        failure_reason='original_reproduction_mismatch',
        diagnostic_codes=('original_mismatch_state_checksum',),
    )
    return CounterfactualResult(
        task_id=task.task_id,
        status='failed',
        actual_action_offset=task.actual_action_offset,
        original_reproduced=False,
        branches=(branch,),
        simulated_steps=1,
        failure_reason='original_reproduction_mismatch',
        diagnostic_codes=('original_mismatch_state_checksum',),
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


def _proposal(
        *,
        event_index=1,
        utility=4.0,
        trigger_reasons=('random_rule_audit',),
        placement_confidence=0.73):
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
        utility=utility,
        link_confidence=0.80,
        placement_confidence=placement_confidence,
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
        trigger_reasons=trigger_reasons,
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
        trigger_reasons=trigger_reasons,
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
        replay=None,
        failure_record_capacity=32):
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
        failure_record_capacity=failure_record_capacity,
    )


class CounterfactualCoordinatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proposal = _proposal()

    def test_recommended_workers_and_zero_worker_degradation(self):
        self.assertEqual(
            effective_cpu_count(
                cpu_count=32,
                affinity_count=12,
                quota_count=6,
            ),
            6,
        )
        self.assertEqual(
            recommended_counterfactual_worker_count(
                cpu_count=16,
                rollout_worker_count=8,
            ),
            4,
        )
        self.assertEqual(
            recommended_counterfactual_worker_count(
                cpu_count=16,
                rollout_worker_count=8,
                cpu_core_ratio=0.125,
            ),
            2,
        )
        self.assertEqual(
            recommended_counterfactual_worker_count(
                cpu_count=4,
                rollout_worker_count=3,
            ),
            0,
        )
        with self.assertRaises(ValueError):
            recommended_counterfactual_worker_count(
                cpu_count=16,
                rollout_worker_count=8,
                cpu_core_ratio=0.0,
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
            submission = unreproduced.submit(self.proposal)
            self.assertTrue(submission.accepted)
            stats = unreproduced.stats
            self.assertEqual(len(replay), 0)
            self.assertEqual(stats.cumulative.results_failed, 1)
            self.assertEqual(stats.cumulative.reproduction_failed, 1)
            self.assertEqual(stats.cumulative.label_ready_results, 0)
            self.assertEqual(stats.cumulative.samples_inserted, 0)
            self.assertEqual(
                stats.cumulative.failure_records_created,
                1,
            )
            self.assertEqual(stats.failure_record_count, 1)
            record = stats.recent_failure_records[0]
            self.assertEqual(record.task_id, submission.task_id)
            self.assertEqual(
                record.proposal_id,
                self.proposal.proposal_id,
            )
            self.assertEqual(
                (
                    record.worker_id,
                    record.episode_id,
                    record.step_index,
                ),
                (
                    self.proposal.transition_key.worker_id,
                    self.proposal.transition_key.episode_id,
                    self.proposal.transition_key.step_index,
                ),
            )
            self.assertEqual(record.created_real_step, 400)
            self.assertEqual(record.observed_real_step, 400)
            self.assertEqual(record.result_status, 'failed')
            self.assertFalse(record.original_reproduced)
            self.assertEqual(
                record.failure_reason,
                'original_reproduction_mismatch',
            )
            self.assertEqual(
                record.diagnostic_codes,
                ('original_mismatch_state_checksum',),
            )
            self.assertEqual(
                record.trigger_reasons,
                ('random_rule_audit',),
            )
            self.assertEqual(len(record.branches), 1)
            self.assertEqual(
                record.branches[0].failure_reason,
                'original_reproduction_mismatch',
            )
            self.assertEqual(
                dict(stats.cumulative.failure_reason_counts),
                {'original_reproduction_mismatch': 1},
            )
            self.assertEqual(
                dict(
                    stats.cumulative
                    .failure_diagnostic_code_counts
                ),
                {'original_mismatch_state_checksum': 1},
            )
            self.assertEqual(
                dict(
                    stats.cumulative
                    .failure_trigger_reason_counts
                ),
                {'random_rule_audit': 1},
            )

            shutdown_stats = unreproduced.close()
            self.assertEqual(shutdown_stats.failure_record_count, 1)
            checkpoint = unreproduced.checkpoint_state()
            self.assertEqual(
                checkpoint['recent_failure_records'][0][
                    'failure_reason'
                ],
                'original_reproduction_mismatch',
            )
            encoded = json.dumps(checkpoint, sort_keys=True)
            self.assertLess(len(encoded), 25_000)
            self.assertNotIn('state_dict_bytes', encoded)
            self.assertNotIn('"snapshot"', encoded)
            self.assertNotIn('"context"', encoded)
            self.assertNotIn('"model"', encoded)
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
            self.assertEqual(
                stats.cumulative.failure_records_created,
                0,
            )
            self.assertEqual(stats.recent_failure_records, ())
        finally:
            raised.close()

    def test_failure_records_are_bounded_and_window_counts_reset(self):
        coordinator = _coordinator(
            _unreproduced_coordinator_runner,
            failure_record_capacity=1,
        )
        first = _proposal(event_index=2)
        second = _proposal(
            event_index=3,
            trigger_reasons=('ambiguous_blocking',),
        )
        try:
            _refresh(coordinator)
            coordinator.record_real_steps(1_000)
            self.assertTrue(coordinator.submit(first).accepted)
            self.assertTrue(coordinator.submit(second).accepted)

            stats = coordinator.snapshot_stats(reset_window=True)
            self.assertEqual(stats.failure_record_capacity, 1)
            self.assertEqual(stats.failure_record_count, 1)
            self.assertEqual(
                stats.recent_failure_records[0].proposal_id,
                second.proposal_id,
            )
            self.assertEqual(
                stats.cumulative.failure_records_created,
                2,
            )
            self.assertEqual(
                stats.cumulative.failure_record_evictions,
                1,
            )
            self.assertEqual(
                dict(stats.cumulative.failure_reason_counts),
                {'original_reproduction_mismatch': 2},
            )
            self.assertEqual(
                dict(
                    stats.cumulative
                    .failure_trigger_reason_counts
                ),
                {
                    'ambiguous_blocking': 1,
                    'random_rule_audit': 1,
                },
            )

            after_reset = coordinator.stats
            self.assertEqual(
                after_reset.window.failure_records_created,
                0,
            )
            self.assertEqual(
                after_reset.window.failure_record_evictions,
                0,
            )
            self.assertEqual(
                dict(after_reset.window.failure_reason_counts),
                {},
            )
            self.assertEqual(after_reset.failure_record_count, 1)
            self.assertEqual(
                after_reset.cumulative.failure_records_created,
                2,
            )
        finally:
            coordinator.close()

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

    def test_offer_pool_selects_highest_priority_across_windows(self):
        executor = _ImmediateExecutor()
        config = CounterfactualConfig(
            min_real_steps=100,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
            queue_capacity=4,
        )
        coordinator = _coordinator(
            _completed_coordinator_runner,
            executor=executor,
            config=config,
        )
        low = _proposal(event_index=10, utility=1.0)
        high = _proposal(event_index=11, utility=20.0)
        try:
            _refresh(coordinator)
            self.assertEqual(
                coordinator.offer(low).drop_reason,
                'candidate_buffered',
            )
            self.assertEqual(
                coordinator.offer(high).drop_reason,
                'candidate_buffered',
            )
            self.assertEqual(
                coordinator.stats.candidate_pool_count,
                2,
            )

            coordinator.record_real_steps(100)

            self.assertEqual(len(executor.calls), 1)
            dispatched_task = executor.calls[0][1][-1]
            self.assertEqual(dispatched_task.budget_key, high.budget_key)
            stats = coordinator.stats
            self.assertEqual(stats.candidate_pool_count, 1)
            self.assertEqual(
                stats.cumulative.candidate_dispatch_attempts,
                1,
            )
            self.assertEqual(
                stats.cumulative.candidate_dispatch_admitted,
                1,
            )
        finally:
            coordinator.close()

    def test_offer_many_arbitrates_before_using_available_slot(self):
        executor = _ImmediateExecutor()
        config = CounterfactualConfig(
            min_real_steps=100,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
            queue_capacity=4,
        )
        coordinator = _coordinator(
            _completed_coordinator_runner,
            executor=executor,
            config=config,
        )
        low = _proposal(event_index=20, utility=1.0)
        high = _proposal(event_index=21, utility=20.0)
        try:
            _refresh(coordinator)
            coordinator.record_real_steps(
                100,
                dispatch_candidates=False,
            )
            outcomes = coordinator.offer_many((low, high))

            self.assertEqual(len(executor.calls), 1)
            self.assertEqual(
                executor.calls[0][1][-1].budget_key,
                high.budget_key,
            )
            self.assertEqual(outcomes[0].drop_reason, 'candidate_buffered')
            self.assertTrue(outcomes[1].accepted)
            self.assertEqual(
                coordinator.stats.candidate_pool_count,
                1,
            )
        finally:
            coordinator.close()

    def test_candidate_pool_capacity_evicts_lowest_priority(self):
        executor = _ImmediateExecutor()
        config = CounterfactualConfig(
            min_real_steps=100,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
            queue_capacity=2,
        )
        coordinator = _coordinator(
            _completed_coordinator_runner,
            executor=executor,
            config=config,
        )
        low = _proposal(event_index=30, utility=1.0)
        middle = _proposal(event_index=31, utility=10.0)
        high = _proposal(event_index=32, utility=20.0)
        try:
            _refresh(coordinator)
            outcomes = coordinator.offer_many((low, middle, high))

            self.assertEqual(len(outcomes), 3)
            self.assertEqual(
                outcomes[0].drop_reason,
                'candidate_pool_priority_evicted',
            )
            self.assertEqual(outcomes[1].drop_reason, 'candidate_buffered')
            self.assertEqual(outcomes[2].drop_reason, 'candidate_buffered')
            stats = coordinator.stats
            self.assertEqual(stats.candidate_pool_capacity, 2)
            self.assertEqual(stats.candidate_pool_count, 2)
            self.assertEqual(
                stats.cumulative.candidate_pool_evictions,
                1,
            )
            self.assertEqual(stats.cumulative.proposals_rejected, 1)
            self.assertEqual(stats.cumulative.candidate_offers, 3)

            coordinator.record_real_steps(100)
            self.assertEqual(
                executor.calls[0][1][-1].budget_key,
                high.budget_key,
            )
            final_stats = coordinator.stats
            self.assertEqual(
                final_stats.cumulative.candidate_pool_evictions,
                1,
            )
            self.assertEqual(
                final_stats.cumulative.proposals_rejected,
                1,
            )
        finally:
            coordinator.close()

    def test_budget_retry_keeps_candidate_and_admission_slot(self):
        executor = _ImmediateExecutor()
        config = CounterfactualConfig(
            min_real_steps=100,
            cost_ratio=0.10,
            cost_hard_limit=0.10,
            queue_capacity=2,
        )
        coordinator = _coordinator(
            _completed_coordinator_runner,
            executor=executor,
            config=config,
        )
        proposal = _proposal(event_index=35, utility=4.0)
        try:
            _refresh(coordinator)
            coordinator.record_real_steps(
                100,
                dispatch_candidates=False,
            )
            first = coordinator.offer(proposal)

            self.assertEqual(first.drop_reason, 'candidate_buffered')
            stats = coordinator.stats
            self.assertEqual(stats.candidate_pool_count, 1)
            self.assertEqual(stats.scheduler.admission_slots_used, 0)
            self.assertEqual(
                stats.scheduler.admission_slots_available,
                1,
            )
            self.assertEqual(
                stats.cumulative.candidate_dispatch_attempts,
                1,
            )
            self.assertEqual(
                stats.cumulative.candidate_dispatch_admitted,
                0,
            )
            self.assertEqual(stats.cumulative.proposals_rejected, 0)
            self.assertEqual(len(executor.calls), 0)

            coordinator.record_real_steps(200)

            stats = coordinator.stats
            self.assertEqual(stats.candidate_pool_count, 0)
            self.assertEqual(stats.scheduler.admission_slots_used, 1)
            self.assertEqual(
                stats.scheduler.admission_slots_available,
                2,
            )
            self.assertEqual(
                stats.cumulative.candidate_dispatch_attempts,
                2,
            )
            self.assertEqual(
                stats.cumulative.candidate_dispatch_admitted,
                1,
            )
            self.assertEqual(stats.cumulative.proposals_rejected, 0)
            self.assertEqual(len(executor.calls), 1)
        finally:
            coordinator.close()

    def test_candidate_pool_tie_break_is_arrival_order_independent(self):
        first = _proposal(event_index=40, utility=5.0)
        second = _proposal(event_index=41, utility=5.0)
        expected = max(
            (first, second),
            key=lambda proposal: proposal.proposal_id,
        )

        winners = []
        for order in ((first, second), (second, first)):
            executor = _ImmediateExecutor()
            config = CounterfactualConfig(
                min_real_steps=100,
                cost_ratio=1.0,
                cost_hard_limit=1.0,
                queue_capacity=1,
            )
            coordinator = _coordinator(
                _completed_coordinator_runner,
                executor=executor,
                config=config,
            )
            try:
                _refresh(coordinator)
                coordinator.offer_many(order)
                coordinator.record_real_steps(100)
                winners.append(
                    executor.calls[0][1][-1].budget_key
                )
            finally:
                coordinator.close()

        self.assertEqual(
            winners,
            [expected.budget_key, expected.budget_key],
        )

    def test_close_clears_candidate_pool_without_checkpointing_proposals(self):
        config = CounterfactualConfig(
            min_real_steps=100,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
            queue_capacity=2,
        )
        coordinator = _coordinator(
            _completed_coordinator_runner,
            config=config,
        )
        _refresh(coordinator)
        coordinator.offer_many((
            _proposal(event_index=50, utility=1.0),
            _proposal(event_index=51, utility=2.0),
        ))
        self.assertEqual(coordinator.stats.candidate_pool_count, 2)

        first = coordinator.close()
        second = coordinator.close()
        encoded = json.dumps(
            coordinator.checkpoint_state(),
            sort_keys=True,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.candidate_pool_count, 0)
        self.assertEqual(
            first.cumulative.candidate_close_dropped,
            2,
        )
        self.assertEqual(
            dict(first.cumulative.drop_reason_counts)[
                'coordinator_closed_candidate'
            ],
            2,
        )
        self.assertNotIn('snapshot', encoded)
        self.assertNotIn('context', encoded)
        self.assertNotIn(self.proposal.proposal_id, encoded)

    def test_legacy_submit_still_bypasses_candidate_pool(self):
        coordinator = _coordinator(
            _completed_coordinator_runner
        )
        try:
            _refresh(coordinator)
            submission = coordinator.submit(self.proposal)

            self.assertFalse(submission.accepted)
            self.assertEqual(submission.drop_reason, 'real_step_gate')
            stats = coordinator.stats
            self.assertEqual(stats.candidate_pool_count, 0)
            self.assertEqual(stats.cumulative.candidate_offers, 0)
            self.assertEqual(stats.cumulative.proposals_received, 1)
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
