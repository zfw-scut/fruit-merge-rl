"""局部 Shapley 累计门控、共享预算、spawn 与失败降级测试。"""

from __future__ import annotations

import json
import time
import unittest
from concurrent.futures import Future
from dataclasses import replace

import torch

from daxigua_rl.attribution.causal_replay import CausalReplayBuffer
from daxigua_rl.attribution.counterfactual import (
    CounterfactualTokenDecision,
    LocalShapleyConfig,
)
from daxigua_rl.attribution.counterfactual_proposal import (
    stable_counterfactual_proposal_id,
)
from daxigua_rl.attribution.local_shapley_runner import (
    LocalShapleyResult,
    LocalShapleySubsetResult,
    run_local_shapley_task,
)
from daxigua_rl.training.local_shapley_coordinator import (
    LocalShapleyCoordinator,
)

from test_local_shapley_runner import _build_two_step_proposal


class _ImmediateExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except BaseException as exc:
            future.set_exception(exc)
        return future


class _RaisingSubmitExecutor:
    def submit(self, _function, *_args):
        raise RuntimeError('intentional executor submit failure')


class _SharedBudget:
    """测试用严格 reservation 账本，接口与 CF coordinator 一致。"""

    def __init__(self, *, reject_count=0, reject_reason=None):
        self.reject_count = int(reject_count)
        self.reject_reason = (
            reject_reason or 'external_soft_token_budget'
        )
        self.active = {}
        self.consumed = 0
        self.refunded = 0
        self.reserve_calls = 0
        self.settle_calls = 0
        self.refund_calls = 0

    def reserve_external_tokens(
            self,
            reservation_id,
            tokens,
            *,
            priority=0.0):
        del priority
        self.reserve_calls += 1
        if self.reject_count > 0:
            self.reject_count -= 1
            return CounterfactualTokenDecision(
                accepted=False,
                reservation_id=reservation_id,
                tokens=0,
                drop_reason=self.reject_reason,
            )
        self.active[reservation_id] = int(tokens)
        return CounterfactualTokenDecision(
            accepted=True,
            reservation_id=reservation_id,
            tokens=int(tokens),
        )

    def settle_external_tokens(self, reservation_id, consumed):
        self.settle_calls += 1
        reserved = self.active.pop(reservation_id)
        consumed = int(consumed)
        self.consumed += consumed
        self.refunded += reserved - consumed
        return CounterfactualTokenDecision(
            accepted=True,
            reservation_id=reservation_id,
            tokens=consumed,
        )

    def refund_external_tokens(self, reservation_id):
        self.refund_calls += 1
        reserved = self.active.pop(reservation_id)
        self.refunded += reserved
        return CounterfactualTokenDecision(
            accepted=True,
            reservation_id=reservation_id,
            tokens=reserved,
        )


def _clone_proposal(proposal, event_index):
    event_id = replace(
        proposal.representative_event.event_id,
        event_index=event_index,
    )
    event = replace(
        proposal.representative_event,
        event_id=event_id,
        budget_key=event_id,
    )
    proposal_id = stable_counterfactual_proposal_id(
        budget_key=event_id,
        representative_event=event,
        contributor=proposal.contributor,
        context=proposal.context,
        snapshot=proposal.snapshot,
        factual_outcome=proposal.factual_outcome,
        alternative_action_offsets=(
            proposal.alternative_action_offsets
        ),
        trigger_reasons=proposal.trigger_reasons,
        coalition_trace_entries=(
            proposal.coalition_trace_entries
        ),
        coalition_candidate_keys=(
            proposal.coalition_candidate_keys
        ),
    )
    return replace(
        proposal,
        proposal_id=proposal_id,
        representative_event=event,
        budget_key=event_id,
    )


def _completed_runner(task):
    contributions = tuple(
        (key, float(offset + 1))
        for offset, key in enumerate(task.candidate_keys)
    )
    full_value = sum(value for _key, value in contributions)
    subset = LocalShapleySubsetResult(
        member_keys=task.candidate_keys,
        objective_return=full_value,
        simulated_steps=2,
        terminated=False,
        truncated=False,
    )
    return LocalShapleyResult(
        task_id=task.task_id,
        status='completed',
        grand_reproduced=True,
        contributions=contributions,
        empty_value=0.0,
        full_value=full_value,
        efficiency_residual=0.0,
        efficiency_tolerance=0.01,
        subset_results=(subset,),
        evaluated_subset_count=1,
        cache_hit_count=0,
        permutation_count=8,
        simulated_steps=2,
    )


def _thread_reporting_physical_runner(task):
    result = run_local_shapley_task(task)
    return replace(
        result,
        cache_hit_count=(
            torch.get_num_threads() * 1_000
            + torch.get_num_interop_threads()
        ),
    )


def _failed_runner(task):
    subset = LocalShapleySubsetResult(
        member_keys=task.candidate_keys,
        objective_return=0.0,
        simulated_steps=1,
        terminated=False,
        truncated=False,
    )
    return LocalShapleyResult(
        task_id=task.task_id,
        status='failed',
        grand_reproduced=False,
        contributions=(),
        empty_value=None,
        full_value=None,
        efficiency_residual=None,
        efficiency_tolerance=None,
        subset_results=(subset,),
        evaluated_subset_count=1,
        cache_hit_count=0,
        permutation_count=0,
        simulated_steps=1,
        failure_reason='grand_reproduction_mismatch',
    )


def _raising_runner(_task):
    raise RuntimeError('intentional runner failure')


class LocalShapleyCoordinatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, base = _build_two_step_proposal()
        cls.proposals = tuple(
            _clone_proposal(base, event_index)
            for event_index in range(1, 8)
        )

    def _coordinator(
            self,
            *,
            config,
            budget=None,
            replay=None,
            runner=_completed_runner,
            executor=None,
            seen_capacity=16_384):
        return LocalShapleyCoordinator(
            causal_replay_buffer=(
                replay
                if replay is not None
                else CausalReplayBuffer(capacity=32, seed=11)
            ),
            shared_budget=budget or _SharedBudget(),
            config=config,
            runner=runner,
            executor=executor or _ImmediateExecutor(),
            seen_capacity=seen_capacity,
        )

    def test_cumulative_selector_dedup_settle_and_sample_insertion(self):
        config = LocalShapleyConfig(event_ratio_max=0.5)
        budget = _SharedBudget()
        replay = CausalReplayBuffer(capacity=16, seed=3)
        coordinator = self._coordinator(
            config=config,
            budget=budget,
            replay=replay,
        )
        try:
            first = coordinator.consider(
                self.proposals[0],
                self.payload,
            )
            second = coordinator.consider(
                self.proposals[1],
                self.payload,
            )
            polled = coordinator.poll()

            self.assertFalse(first.selected)
            self.assertFalse(first.skip_counterfactual)
            self.assertTrue(second.selected)
            self.assertTrue(second.skip_counterfactual)
            self.assertTrue(second.accepted)
            self.assertEqual(polled.result_count, 1)
            self.assertEqual(polled.inserted_sample_count, 2)
            self.assertEqual(len(replay), 2)
            self.assertEqual(budget.consumed, 2)
            self.assertEqual(budget.refunded, 6)
            self.assertEqual(budget.settle_calls, 1)

            duplicate = coordinator.consider(
                self.proposals[1],
                self.payload,
            )
            self.assertFalse(duplicate.observed)
            self.assertTrue(duplicate.selected)
            self.assertTrue(duplicate.skip_counterfactual)

            stats = coordinator.stats
            self.assertEqual(stats.observed_event_count, 2)
            self.assertEqual(stats.selected_event_count, 1)
            self.assertEqual(stats.selected_ratio, 0.5)
            self.assertEqual(
                stats.cumulative.samples_inserted,
                2,
            )
            encoded = json.dumps(
                coordinator.checkpoint_state(),
                sort_keys=True,
            )
            self.assertNotIn('snapshot', encoded)
            self.assertNotIn('state_dict', encoded)
            self.assertNotIn('future', encoded)
        finally:
            coordinator.close()

    def test_temporary_budget_rejection_stays_pending_then_retries(self):
        config = LocalShapleyConfig(event_ratio_max=1.0)
        budget = _SharedBudget(reject_count=1)
        coordinator = self._coordinator(
            config=config,
            budget=budget,
        )
        try:
            submission = coordinator.consider(
                self.proposals[0],
                self.payload,
            )

            self.assertTrue(submission.selected)
            self.assertTrue(submission.skip_counterfactual)
            self.assertFalse(submission.accepted)
            self.assertTrue(submission.pending)
            self.assertEqual(
                submission.drop_reason,
                'external_soft_token_budget',
            )
            self.assertEqual(
                coordinator.stats.pending_task_count,
                1,
            )

            coordinator.retry_pending()
            result = coordinator.poll()

            self.assertEqual(result.result_count, 1)
            self.assertEqual(budget.reserve_calls, 2)
            self.assertEqual(budget.settle_calls, 1)
            self.assertEqual(
                coordinator.stats.pending_task_count,
                0,
            )
        finally:
            coordinator.close()

    def test_recent_seen_lru_is_bounded_and_refreshes_duplicates(self):
        coordinator = self._coordinator(
            config=LocalShapleyConfig(event_ratio_max=0.0),
            seen_capacity=2,
        )
        try:
            first = coordinator.consider(
                self.proposals[0],
                self.payload,
            )
            second = coordinator.consider(
                self.proposals[1],
                self.payload,
            )
            duplicate = coordinator.consider(
                self.proposals[0],
                self.payload,
            )
            third = coordinator.consider(
                self.proposals[2],
                self.payload,
            )

            self.assertTrue(first.observed)
            self.assertTrue(second.observed)
            self.assertFalse(duplicate.observed)
            self.assertTrue(third.observed)
            stats = coordinator.stats
            self.assertEqual(stats.recent_seen_capacity, 2)
            self.assertEqual(stats.recent_seen_count, 2)
            self.assertEqual(stats.recent_seen_eviction_count, 1)
            self.assertEqual(
                stats.cumulative.seen_cache_evictions,
                1,
            )

            # duplicate 刷新了 proposal[0]；随后写 proposal[2] 淘汰更旧的
            # proposal[1]，因此它可以作为远期新观察重新进入分母。
            replayed_old = coordinator.consider(
                self.proposals[1],
                self.payload,
            )
            self.assertTrue(replayed_old.observed)
            self.assertEqual(
                coordinator.stats.observed_event_count,
                4,
            )
            self.assertEqual(
                coordinator.stats.recent_seen_count,
                2,
            )
            self.assertEqual(
                coordinator.stats.recent_seen_eviction_count,
                2,
            )
        finally:
            coordinator.close()

    def test_selected_identity_remains_permanent_after_seen_lru_churn(self):
        coordinator = self._coordinator(
            config=LocalShapleyConfig(event_ratio_max=0.5),
            seen_capacity=1,
        )
        try:
            not_selected = coordinator.consider(
                self.proposals[0],
                self.payload,
            )
            selected = coordinator.consider(
                self.proposals[1],
                self.payload,
            )
            churn = coordinator.consider(
                self.proposals[2],
                self.payload,
            )
            duplicate = coordinator.consider(
                self.proposals[1],
                self.payload,
            )

            self.assertFalse(not_selected.selected)
            self.assertTrue(selected.selected)
            self.assertFalse(churn.selected)
            self.assertFalse(duplicate.observed)
            self.assertTrue(duplicate.selected)
            self.assertTrue(duplicate.skip_counterfactual)
            stats = coordinator.stats
            self.assertEqual(stats.observed_event_count, 3)
            self.assertEqual(stats.selected_event_count, 1)
            self.assertEqual(
                stats.permanent_selected_identity_count,
                1,
            )
            self.assertEqual(stats.recent_seen_count, 1)
            self.assertEqual(stats.recent_seen_eviction_count, 1)
        finally:
            coordinator.close()

    def test_budget_starvation_keeps_at_most_four_pending(self):
        config = LocalShapleyConfig(event_ratio_max=1.0)
        budget = _SharedBudget(reject_count=100)
        coordinator = self._coordinator(
            config=config,
            budget=budget,
        )
        try:
            submissions = tuple(
                coordinator.consider(proposal, self.payload)
                for proposal in self.proposals[:5]
            )

            self.assertTrue(all(
                submission.skip_counterfactual
                for submission in submissions
            ))
            self.assertEqual(
                coordinator.stats.pending_task_count,
                4,
            )
            self.assertGreaterEqual(
                coordinator.stats.cumulative.pending_evicted
                + dict(
                    coordinator.stats.cumulative.drop_reason_counts
                ).get('pending_capacity', 0),
                1,
            )
        finally:
            coordinator.close()

    def test_failed_result_settles_actual_and_runner_crash_is_conservative(self):
        config = LocalShapleyConfig(event_ratio_max=1.0)
        replay = CausalReplayBuffer(capacity=8)
        budget = _SharedBudget()
        failed = self._coordinator(
            config=config,
            budget=budget,
            replay=replay,
            runner=_failed_runner,
        )
        try:
            failed.consider(self.proposals[0], self.payload)
            result = failed.poll()

            self.assertEqual(result.result_count, 1)
            self.assertEqual(result.inserted_sample_count, 0)
            self.assertEqual(len(replay), 0)
            self.assertEqual(budget.consumed, 1)
            self.assertEqual(budget.refunded, 7)
            self.assertEqual(
                failed.stats.cumulative.reproduction_failed,
                1,
            )
        finally:
            failed.close()

        crash_budget = _SharedBudget()
        crashed = self._coordinator(
            config=config,
            budget=crash_budget,
            runner=_raising_runner,
        )
        try:
            crashed.consider(self.proposals[1], self.payload)
            result = crashed.poll()

            self.assertEqual(result.result_count, 0)
            self.assertEqual(crash_budget.consumed, 8)
            self.assertEqual(
                crashed.stats.cumulative.results_failed,
                1,
            )
            self.assertEqual(
                crashed.stats.cumulative.samples_inserted,
                0,
            )
        finally:
            crashed.close()

    def test_executor_submit_failure_refunds_full_reservation(self):
        config = LocalShapleyConfig(event_ratio_max=1.0)
        budget = _SharedBudget()
        coordinator = self._coordinator(
            config=config,
            budget=budget,
            executor=_RaisingSubmitExecutor(),
        )
        try:
            submission = coordinator.consider(
                self.proposals[0],
                self.payload,
            )

            self.assertTrue(submission.selected)
            self.assertFalse(submission.accepted)
            self.assertEqual(budget.consumed, 0)
            self.assertEqual(budget.refunded, 8)
            self.assertEqual(budget.refund_calls, 1)
            self.assertEqual(
                coordinator.stats.cumulative
                .executor_submit_failures,
                1,
            )
        finally:
            coordinator.close()

    def test_default_executor_uses_spawn_and_stays_nonblocking(self):
        config = LocalShapleyConfig(event_ratio_max=1.0)
        budget = _SharedBudget()
        replay = CausalReplayBuffer(capacity=16)
        main_intra_threads = torch.get_num_threads()
        main_interop_threads = torch.get_num_interop_threads()
        coordinator = LocalShapleyCoordinator(
            causal_replay_buffer=replay,
            shared_budget=budget,
            config=config,
            runner=_thread_reporting_physical_runner,
        )
        try:
            submission = coordinator.consider(
                self.proposals[0],
                self.payload,
            )
            self.assertTrue(submission.accepted)
            self.assertEqual(len(replay), 0)

            deadline = time.monotonic() + 40.0
            result = coordinator.poll()
            while (
                    result.result_count == 0
                    and time.monotonic() < deadline):
                time.sleep(0.02)
                result = coordinator.poll()

            self.assertEqual(result.result_count, 1)
            self.assertTrue(result.results[0].label_ready)
            self.assertEqual(
                result.results[0].cache_hit_count,
                1_001,
            )
            self.assertEqual(len(replay), 2)
            self.assertEqual(budget.settle_calls, 1)
            self.assertEqual(
                torch.get_num_threads(),
                main_intra_threads,
            )
            self.assertEqual(
                torch.get_num_interop_threads(),
                main_interop_threads,
            )
        finally:
            coordinator.close()


if __name__ == '__main__':
    unittest.main()
