"""预算反事实调度、候选选择与局部 Shapley 专项测试。"""

from __future__ import annotations

import pickle
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from daxigua.core.engine import HeadlessGame
from daxigua.core.state import DropResult, EngineActionOutcome, PhysicsResult
from daxigua_rl.attribution.counterfactual import (
    BudgetedCounterfactualScheduler,
    CounterfactualBranchResult,
    CounterfactualConfig,
    CounterfactualResult,
    CounterfactualTask,
    CumulativeShapleySelector,
    FrozenGNNModelConfig,
    FrozenGraphBuilderConfig,
    FrozenTargetPolicyPayload,
    LocalShapleyConfig,
    SnapshotRing,
    counterfactual_priority,
    counterfactual_trigger_reasons,
    create_counterfactual_task,
    estimate_local_shapley,
    initialize_physics_runner_process,
    local_shapley_candidates,
    local_shapley_eligible,
    paired_shapley_permutations,
    select_counterfactual_alternatives,
)
from daxigua_rl.attribution.state_analyzer import StateAnalyzerConfig
from daxigua_rl.attribution.schema import (
    AttributionEvent,
    AttributionEventKey,
    AttributionEvidence,
    Contributor,
)
from daxigua_rl.reward import RewardConfig, merge_utility
from daxigua_rl.training.identity import TransitionKey


class _ManualFuture:
    """让调度预算测试无需启动进程即可精确控制完成时机。"""

    def __init__(self, function, args):
        self._function = function
        self._args = args
        self._done = False
        self._cancelled = False
        self._result = None
        self._error = None

    def done(self):
        return self._done

    def cancel(self):
        if self._done:
            return False
        self._cancelled = True
        self._done = True
        return True

    def run(self):
        if self._done:
            return
        try:
            self._result = self._function(*self._args)
        except BaseException as exc:
            self._error = exc
        self._done = True

    def result(self):
        if self._cancelled:
            raise RuntimeError('future was cancelled')
        if not self._done:
            raise RuntimeError('future is not done')
        if self._error is not None:
            raise self._error
        return self._result


class _ManualExecutor:
    def __init__(self):
        self.futures = []

    def submit(self, function, *args):
        future = _ManualFuture(function, args)
        self.futures.append(future)
        return future

    def run_next(self):
        future = next(
            item
            for item in self.futures
            if not item.done()
        )
        future.run()
        return future

    def run_all(self):
        for future in self.futures:
            future.run()


def _completed_runner(task):
    branches = tuple(
        CounterfactualBranchResult(
            action_offset=action_offset,
            status='completed',
            objective_return=10.0 - action_offset * 0.1,
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


def _partial_runner(task):
    branches = (
        CounterfactualBranchResult(
            action_offset=task.actual_action_offset,
            status='completed',
            objective_return=5.0,
            simulated_steps=2,
        ),
        CounterfactualBranchResult(
            action_offset=task.alternative_action_offsets[0],
            status='partial',
            objective_return=3.0,
            simulated_steps=1,
            early_stopped=True,
        ),
    )
    return CounterfactualResult(
        task_id=task.task_id,
        status='partial',
        actual_action_offset=task.actual_action_offset,
        original_reproduced=True,
        branches=branches,
        simulated_steps=3,
        failure_reason='unused_branches_cancelled',
    )


def _failing_runner(_task):
    raise RuntimeError('intentional runner failure')


def _slow_completed_runner(task):
    time.sleep(0.05)
    result = _completed_runner(task)
    thread_marker = (
        torch.get_num_threads() * 1_000
        + torch.get_num_interop_threads()
    )
    return replace(
        result,
        branches=(
            replace(
                result.branches[0],
                objective_return=float(thread_marker),
            ),
            *result.branches[1:],
        ),
    )


def _snapshot():
    return HeadlessGame(seed=0).capture_snapshot()


_SCHEDULER_POLICY = FrozenTargetPolicyPayload.create(
    policy_version='scheduler-test-target',
    model_config=FrozenGNNModelConfig(
        hidden_dim=8,
        message_layers=1,
    ),
    graph_builder_config=FrozenGraphBuilderConfig(),
    # 调度契约测试不会执行物理 runner；真实权重 roundtrip 在独立 runner 测试覆盖。
    state_dict_bytes=b'scheduler-contract-placeholder',
    gamma=0.99,
    max_physics_frames=720,
    stable_frames=15,
    reward_config=RewardConfig(),
    state_analyzer_config=StateAnalyzerConfig(),
)


def _scheduler_factual_outcome(snapshot):
    """构造仅供调度契约使用的相邻 outcome，不运行昂贵物理。"""

    game = HeadlessGame.from_snapshot(snapshot)
    state = game.get_state()
    action = game.get_action_candidates(15)[0]
    final_state = replace(
        state,
        step_count=state.step_count + 1,
    )
    return EngineActionOutcome(
        drop_result=DropResult(
            dropped_level=action.current_level,
            drop_x=action.drop_x,
            fruit_id=snapshot.episode.next_fruit_id,
            queue_before=tuple(snapshot.episode.fruit_queue),
            queue_after=tuple(snapshot.episode.fruit_queue),
        ),
        physics_result=PhysicsResult(
            frames_simulated=0,
            stable=True,
            done=False,
            truncated=False,
            score_delta=0,
        ),
        final_state=final_state,
        fail_count=snapshot.episode.fail_count,
        next_fruit_id=snapshot.episode.next_fruit_id + 1,
        rng_state=snapshot.episode.rng_state,
    )


def _task(
        *,
        event_index,
        config,
        priority=1.0,
        alternatives=(14,),
        created_real_step=0,
        snapshot=None):
    snapshot = snapshot or _snapshot()
    transition_key = TransitionKey(0, 0, snapshot.episode.step_count)
    factual_outcome = _scheduler_factual_outcome(snapshot)
    return create_counterfactual_task(
        budget_key=AttributionEventKey(0, 0, event_index),
        transition_key=transition_key,
        snapshot=snapshot,
        factual_outcome=factual_outcome,
        target_policy=_SCHEDULER_POLICY,
        actual_action_offset=0,
        alternative_action_offsets=alternatives,
        trigger_reasons=('random_rule_audit',),
        event_utility=1.0,
        placement_confidence=0.6,
        created_real_step=created_real_step,
        attribution_version='test_v1',
        config=config,
        priority=priority,
    )


def _contributors():
    return (
        Contributor(
            transition_key=TransitionKey(0, 0, 1),
            action_offset=1,
            action_index=1,
            fruit_id=1,
            evidence_type='material',
            raw_evidence_weight=0.6,
            contribution_weight=0.6,
            role='material',
        ),
        Contributor(
            transition_key=TransitionKey(0, 0, 2),
            action_offset=2,
            action_index=2,
            fruit_id=2,
            evidence_type='support',
            raw_evidence_weight=0.4,
            contribution_weight=0.4,
            role='support',
        ),
    )


def _high_value_event():
    event_id = AttributionEventKey(0, 0, 4)
    return AttributionEvent(
        event_id=event_id,
        episode_key=(0, 0),
        attribution_version='test_v1',
        tracker_config_fingerprint='tracker-test',
        detected_step=3,
        resolved_step=3,
        event_type='MERGE_LINEAGE',
        status='confirmed',
        sign=1,
        target_fruit_ids=(1, 2),
        contributors=_contributors(),
        utility=merge_utility(7),
        link_confidence=1.0,
        placement_confidence=0.7,
        evidence=AttributionEvidence(
            reason_codes=('test',),
        ),
        budget_key=event_id,
        resolution_reason='test',
    )


class CounterfactualContractTest(unittest.TestCase):
    def test_config_is_strict_stable_and_pickle_safe(self):
        config = CounterfactualConfig()
        restored = pickle.loads(pickle.dumps(config))
        shapley_config = LocalShapleyConfig()

        self.assertEqual(restored, config)
        self.assertEqual(restored.fingerprint, config.fingerprint)
        self.assertEqual(
            pickle.loads(pickle.dumps(shapley_config)).fingerprint,
            shapley_config.fingerprint,
        )
        self.assertEqual(config.snapshot_ring_size, 32)
        self.assertEqual(config.queue_capacity, 256)

        with self.assertRaises(ValueError):
            CounterfactualConfig(
                cost_ratio=0.11,
                cost_hard_limit=0.10,
            )
        with self.assertRaises(ValueError):
            CounterfactualConfig(
                cost_ratio=0.06,
                cost_hard_limit=0.10,
                external_token_reserve_ratio=0.05,
            )
        with self.assertRaises(ValueError):
            CounterfactualConfig(max_alternatives=4)
        with self.assertRaises(ValueError):
            CounterfactualConfig(max_inflight_per_worker=3)

    def test_snapshot_ring_evicts_oldest_at_capacity(self):
        snapshot = _snapshot()
        ring = SnapshotRing(capacity=2)
        first = TransitionKey(0, 0, 0)
        second = TransitionKey(0, 1, 0)
        third = TransitionKey(0, 2, 0)

        self.assertIsNone(ring.push(first, snapshot))
        self.assertIsNone(ring.push(second, snapshot))
        evicted = ring.push(third, snapshot)

        self.assertEqual(evicted[0], first)
        self.assertIs(evicted[1], snapshot)
        self.assertEqual(ring.keys, (second, third))
        self.assertIsNone(ring.get(first))

    def test_trigger_and_alternative_selection_are_deterministic(self):
        reasons = counterfactual_trigger_reasons(
            new_level=7,
            chain_depth=2,
            possible_blocker_causes=2,
            conflicting_signals=True,
            placement_confidence=0.7,
        )
        self.assertEqual(
            reasons,
            (
                'high_value_merge',
                'multi_stage_chain',
                'ambiguous_blocking',
                'conflicting_signals',
                'middle_placement_confidence',
            ),
        )

        scores = [0.0] * 15
        scores[5] = 10.0
        alternatives = select_counterfactual_alternatives(
            actual_action_offset=2,
            safest_action_scores=scores,
            runner_up_action_offset=7,
        )
        self.assertEqual(alternatives, (12, 5, 7))
        self.assertEqual(len(set(alternatives)), 3)

    def test_task_id_and_priority_are_stable_and_task_is_pickle_safe(self):
        config = CounterfactualConfig()
        snapshot = _snapshot()
        first = _task(
            event_index=1,
            config=config,
            snapshot=snapshot,
        )
        second = _task(
            event_index=1,
            config=config,
            snapshot=snapshot,
        )

        self.assertEqual(first.task_id, second.task_id)
        self.assertEqual(
            pickle.loads(pickle.dumps(first)),
            first,
        )
        with self.assertRaises((AttributeError, TypeError)):
            first.priority = 999.0
        self.assertTrue(hasattr(CounterfactualTask, '__slots__'))
        priority = counterfactual_priority(
            event_utility=2.0,
            trigger_reasons=('high_value_merge',),
            placement_confidence=0.7,
        )
        self.assertAlmostEqual(priority, 6.7)

        with self.assertRaises(ValueError):
            CounterfactualTask(
                task_id='wrong',
                budget_key=first.budget_key,
                transition_key=first.transition_key,
                snapshot=first.snapshot,
                factual_outcome=first.factual_outcome,
                factual_outcome_fingerprint=(
                    first.factual_outcome_fingerprint
                ),
                target_policy=first.target_policy,
                actual_action_offset=first.actual_action_offset,
                alternative_action_offsets=(
                    first.alternative_action_offsets
                ),
                trigger_reasons=first.trigger_reasons,
                priority=first.priority,
                estimated_tokens=first.estimated_tokens,
                horizon=first.horizon,
                created_real_step=first.created_real_step,
                attribution_version=first.attribution_version,
                scheduler_config_fingerprint=(
                    first.scheduler_config_fingerprint
                ),
                label_confidence=first.label_confidence,
                attribution_delay=first.attribution_delay,
            )

    def test_failed_or_unreproduced_result_never_exposes_delta(self):
        result = CounterfactualResult(
            task_id='cf-test',
            status='partial',
            actual_action_offset=0,
            original_reproduced=False,
            branches=(
                CounterfactualBranchResult(
                    action_offset=0,
                    status='completed',
                    objective_return=3.0,
                    simulated_steps=1,
                ),
                CounterfactualBranchResult(
                    action_offset=1,
                    status='completed',
                    objective_return=1.0,
                    simulated_steps=1,
                ),
            ),
            simulated_steps=2,
            failure_reason='original_reproduction_mismatch',
        )
        self.assertFalse(result.label_ready)
        self.assertEqual(result.return_deltas, ())


class BudgetedSchedulerTest(unittest.TestCase):
    def test_physics_initializer_tolerates_interop_runtime_error(self):
        with (
                patch.object(torch, 'set_num_threads') as set_intra,
                patch.object(
                    torch,
                    'set_num_interop_threads',
                    side_effect=RuntimeError('already initialized'),
                ) as set_interop):
            initialize_physics_runner_process()

        set_intra.assert_called_once_with(1)
        set_interop.assert_called_once_with(1)

    def test_external_tokens_share_hard_budget_and_finalize_once(self):
        config = CounterfactualConfig(
            min_real_steps=1,
            soft_budget_borrow_priority=10.0,
        )
        scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_completed_runner,
            config=config,
            executor=_ManualExecutor(),
        )
        scheduler.record_real_steps(1_000)
        try:
            first = scheduler.reserve_external_tokens(
                'shapley-event-a',
                80,
            )
            self.assertTrue(first.accepted)
            self.assertEqual(first.tokens, 80)

            duplicate_active = scheduler.reserve_external_tokens(
                'shapley-event-a',
                1,
            )
            self.assertFalse(duplicate_active.accepted)
            self.assertEqual(
                duplicate_active.drop_reason,
                'duplicate_external_reservation',
            )

            borrowed = scheduler.reserve_external_tokens(
                'shapley-event-b',
                20,
                priority=10.0,
            )
            self.assertTrue(borrowed.accepted)
            hard_rejected = scheduler.reserve_external_tokens(
                'shapley-event-c',
                1,
                priority=10.0,
            )
            self.assertFalse(hard_rejected.accepted)
            self.assertEqual(
                hard_rejected.drop_reason,
                'external_hard_token_budget',
            )
            self.assertEqual(
                scheduler.stats.tokens_reserved,
                100,
            )

            over_consumed = scheduler.settle_external_tokens(
                'shapley-event-a',
                81,
            )
            self.assertFalse(over_consumed.accepted)
            self.assertEqual(
                over_consumed.drop_reason,
                'external_consumed_exceeds_reservation',
            )
            self.assertEqual(
                scheduler.stats.tokens_reserved,
                100,
            )

            settled = scheduler.settle_external_tokens(
                'shapley-event-a',
                50,
            )
            self.assertTrue(settled.accepted)
            self.assertEqual(settled.tokens, 50)
            settled_twice = scheduler.settle_external_tokens(
                'shapley-event-a',
                50,
            )
            self.assertFalse(settled_twice.accepted)
            self.assertEqual(
                settled_twice.drop_reason,
                'external_reservation_already_finalized',
            )
            duplicate_finished = scheduler.reserve_external_tokens(
                'shapley-event-a',
                1,
            )
            self.assertFalse(duplicate_finished.accepted)
            self.assertEqual(
                duplicate_finished.drop_reason,
                'duplicate_external_reservation',
            )

            refunded = scheduler.refund_external_tokens(
                'shapley-event-b'
            )
            self.assertTrue(refunded.accepted)
            self.assertEqual(refunded.tokens, 20)
            refunded_twice = scheduler.refund_external_tokens(
                'shapley-event-b'
            )
            self.assertFalse(refunded_twice.accepted)
            self.assertEqual(
                refunded_twice.drop_reason,
                'external_reservation_already_finalized',
            )

            stats = scheduler.stats
            self.assertEqual(stats.tokens_reserved, 0)
            self.assertEqual(stats.tokens_consumed, 50)
            self.assertEqual(stats.tokens_refunded, 50)
            self.assertEqual(stats.external_active_reservations, 0)
            self.assertEqual(
                stats.external_reservations_accepted,
                2,
            )
            self.assertEqual(
                stats.external_reservations_settled,
                1,
            )
            self.assertEqual(
                stats.external_reservations_refunded,
                1,
            )
            self.assertLessEqual(
                stats.tokens_consumed + stats.tokens_reserved,
                stats.hard_token_limit,
            )
        finally:
            scheduler.close()

    def test_close_refunds_active_external_reservations(self):
        config = CounterfactualConfig(
            min_real_steps=1,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
        )
        scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_completed_runner,
            config=config,
            executor=_ManualExecutor(),
        )
        scheduler.record_real_steps(100)
        self.assertTrue(
            scheduler.reserve_external_tokens(
                'shapley-close',
                30,
            ).accepted
        )

        scheduler.close()
        stats = scheduler.stats

        self.assertEqual(stats.tokens_reserved, 0)
        self.assertEqual(stats.tokens_refunded, 30)
        self.assertEqual(stats.external_active_reservations, 0)
        self.assertEqual(
            stats.external_reservations_refunded,
            1,
        )
        self.assertEqual(
            dict(stats.drop_reason_counts)[
                'external_scheduler_closed'
            ],
            1,
        )

    def test_real_step_gate_and_soft_hard_token_budgets(self):
        config = CounterfactualConfig()
        executor = _ManualExecutor()
        scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_completed_runner,
            config=config,
            executor=executor,
        )
        task = _task(event_index=1, config=config)
        try:
            scheduler.record_real_steps(255)
            rejected = scheduler.submit(task)
            self.assertFalse(rejected.accepted)
            self.assertEqual(rejected.drop_reason, 'real_step_gate')

            scheduler.record_real_steps(1)
            accepted = scheduler.submit(task)
            self.assertTrue(accepted.accepted)

            second = _task(
                event_index=2,
                config=config,
                created_real_step=256,
            )
            rejected_second = scheduler.submit(second)
            self.assertEqual(
                rejected_second.drop_reason,
                'real_step_gate',
            )
            self.assertEqual(scheduler.stats.tokens_reserved, 20)
            self.assertEqual(scheduler.drain_label_results(), ())
        finally:
            scheduler.close(wait=False)

        hard_config = CounterfactualConfig(
            soft_budget_borrow_priority=10.0,
        )
        hard_scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_completed_runner,
            config=hard_config,
            executor=_ManualExecutor(),
        )
        try:
            hard_scheduler.record_real_steps(256)
            too_expensive = _task(
                event_index=3,
                config=hard_config,
                priority=20.0,
                alternatives=(13, 14),
            )
            decision = hard_scheduler.submit(too_expensive)
            self.assertFalse(decision.accepted)
            self.assertEqual(
                decision.drop_reason,
                'hard_token_budget',
            )
        finally:
            hard_scheduler.close(wait=False)

        borrow_config = CounterfactualConfig(
            soft_budget_borrow_priority=10.0,
        )
        borrow_scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_completed_runner,
            config=borrow_config,
            executor=_ManualExecutor(),
        )
        try:
            # 300 步提供 soft=24、hard=30。三分支任务需要 30 token，
            # 只有高优先级借用路径可以接受，而且仍未越过硬上限。
            borrow_scheduler.record_real_steps(300)
            borrowed = borrow_scheduler.submit(_task(
                event_index=4,
                config=borrow_config,
                priority=20.0,
                alternatives=(13, 14),
            ))
            self.assertTrue(borrowed.accepted)
            self.assertEqual(
                borrow_scheduler.stats.soft_budget_borrows,
                1,
            )
        finally:
            borrow_scheduler.close(wait=False)

    def test_external_reserve_cannot_be_consumed_by_ordinary_tasks(self):
        config = CounterfactualConfig(
            cost_ratio=0.06,
            cost_hard_limit=0.10,
            external_token_reserve_ratio=0.01,
            min_real_steps=1,
            soft_budget_borrow_priority=10.0,
        )
        scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_completed_runner,
            config=config,
            executor=_ManualExecutor(),
        )
        try:
            scheduler.record_real_steps(1_000)
            # 每个任务包含 actual + 两个 alternative，共预留 30 token。
            for event_index in range(1, 4):
                decision = scheduler.submit(_task(
                    event_index=event_index,
                    config=config,
                    priority=20.0,
                    alternatives=(13, 14),
                ))
                self.assertTrue(decision.accepted)

            ordinary_overflow = scheduler.submit(_task(
                event_index=4,
                config=config,
                priority=20.0,
                alternatives=(13, 14),
            ))
            self.assertFalse(ordinary_overflow.accepted)
            self.assertEqual(
                ordinary_overflow.drop_reason,
                'hard_token_budget',
            )

            shapley = scheduler.reserve_external_tokens(
                'shapley-reserved-share',
                10,
                priority=20.0,
            )
            self.assertTrue(shapley.accepted)
            external_overflow = scheduler.reserve_external_tokens(
                'shapley-overflow',
                1,
                priority=20.0,
            )
            self.assertFalse(external_overflow.accepted)
            self.assertEqual(
                external_overflow.drop_reason,
                'external_hard_token_budget',
            )

            stats = scheduler.stats
            self.assertEqual(stats.soft_token_limit, 60.0)
            self.assertEqual(
                stats.ordinary_hard_token_limit,
                90.0,
            )
            self.assertEqual(stats.external_token_reserve, 10.0)
            self.assertEqual(stats.hard_token_limit, 100.0)
            self.assertEqual(
                stats.tokens_consumed + stats.tokens_reserved,
                100,
            )
        finally:
            scheduler.close(wait=False)

    def test_active_and_finished_budget_are_deduplicated(self):
        config = CounterfactualConfig(
            min_real_steps=1,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
        )
        executor = _ManualExecutor()
        scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_completed_runner,
            config=config,
            executor=executor,
        )
        scheduler.record_real_steps(100)
        task = _task(event_index=1, config=config)
        try:
            self.assertTrue(scheduler.submit(task).accepted)
            active_duplicate = scheduler.submit(task)
            self.assertEqual(
                active_duplicate.drop_reason,
                'duplicate_budget',
            )

            executor.run_next()
            scheduler.poll()
            finished_duplicate = scheduler.submit(task)
            self.assertEqual(
                finished_duplicate.drop_reason,
                'duplicate_budget',
            )
        finally:
            scheduler.close()

    def test_high_priority_queue_item_evicts_lowest_waiting_item(self):
        config = CounterfactualConfig(
            min_real_steps=1,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
            queue_capacity=2,
            soft_budget_borrow_priority=1000.0,
        )
        executor = _ManualExecutor()
        scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_completed_runner,
            config=config,
            executor=executor,
        )
        scheduler.record_real_steps(1_000)
        try:
            # 前两项占满 inflight=2，随后两项留在有界等待队列。
            for event_index, priority in (
                    (1, 5.0),
                    (2, 5.0),
                    (3, 1.0),
                    (4, 2.0)):
                self.assertTrue(scheduler.submit(_task(
                    event_index=event_index,
                    config=config,
                    priority=priority,
                )).accepted)
            high = scheduler.submit(_task(
                event_index=5,
                config=config,
                priority=3.0,
            ))

            self.assertTrue(high.accepted)
            stats = scheduler.stats
            self.assertEqual(stats.inflight, 2)
            self.assertEqual(stats.queued, 2)
            self.assertEqual(stats.queue_evicted, 1)
            self.assertEqual(
                dict(stats.drop_reason_counts)[
                    'queue_priority_evicted'
                ],
                1,
            )
        finally:
            scheduler.close(wait=False)

    def test_partial_completion_refunds_tokens_and_produces_valid_label(self):
        config = CounterfactualConfig(
            min_real_steps=1,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
        )
        executor = _ManualExecutor()
        scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_partial_runner,
            config=config,
            executor=executor,
        )
        scheduler.record_real_steps(100)
        task = _task(event_index=1, config=config)
        try:
            self.assertTrue(scheduler.submit(task).accepted)
            executor.run_next()
            completed = scheduler.poll()

            self.assertEqual(len(completed), 1)
            self.assertTrue(completed[0].label_ready)
            self.assertEqual(completed[0].return_deltas, ((14, 2.0),))
            stats = scheduler.stats
            self.assertEqual(stats.partial, 1)
            self.assertEqual(stats.tokens_consumed, 3)
            self.assertEqual(stats.tokens_refunded, 17)
            self.assertEqual(stats.labels_ready, 1)
        finally:
            scheduler.close()

    def test_repeated_runner_failure_opens_circuit(self):
        config = CounterfactualConfig(
            min_real_steps=1,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
            circuit_breaker_failures=2,
        )
        executor = _ManualExecutor()
        scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_failing_runner,
            config=config,
            executor=executor,
        )
        scheduler.record_real_steps(100)
        try:
            self.assertTrue(scheduler.submit(_task(
                event_index=1,
                config=config,
            )).accepted)
            self.assertTrue(scheduler.submit(_task(
                event_index=2,
                config=config,
            )).accepted)
            executor.run_all()
            self.assertEqual(scheduler.poll(), ())

            stats = scheduler.stats
            self.assertEqual(stats.failed, 2)
            self.assertTrue(stats.circuit_open)
            rejected = scheduler.submit(_task(
                event_index=3,
                config=config,
            ))
            self.assertEqual(rejected.drop_reason, 'circuit_open')
            self.assertEqual(scheduler.drain_label_results(), ())
        finally:
            first_close = scheduler.close()
            second_close = scheduler.close()
            self.assertEqual(first_close, second_close)

    def test_spawn_execution_is_pickle_safe_and_submit_is_nonblocking(self):
        config = CounterfactualConfig(
            min_real_steps=1,
            cost_ratio=1.0,
            cost_hard_limit=1.0,
            horizon=8,
        )
        main_intra_threads = torch.get_num_threads()
        main_interop_threads = torch.get_num_interop_threads()
        scheduler = BudgetedCounterfactualScheduler(
            worker_count=1,
            runner=_slow_completed_runner,
            config=config,
        )
        scheduler.record_real_steps(100)
        task = _task(event_index=1, config=config)
        try:
            started = time.perf_counter()
            decision = scheduler.submit(task)
            submit_seconds = time.perf_counter() - started

            self.assertTrue(decision.accepted)
            self.assertLess(submit_seconds, 1.0)
            deadline = time.monotonic() + 15.0
            results = ()
            while time.monotonic() < deadline and not results:
                results = scheduler.poll()
                if not results:
                    time.sleep(0.02)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].label_ready)
            self.assertEqual(
                results[0].branches[0].objective_return,
                1_001.0,
            )
            self.assertEqual(
                torch.get_num_threads(),
                main_intra_threads,
            )
            self.assertEqual(
                torch.get_num_interop_threads(),
                main_interop_threads,
            )
        finally:
            scheduler.close()


class LocalShapleyTest(unittest.TestCase):
    def test_eligibility_requires_high_value_synergy_and_two_candidates(self):
        event = _high_value_event()
        config = LocalShapleyConfig()

        self.assertEqual(
            local_shapley_candidates(event.contributors, config),
            (
                TransitionKey(0, 0, 1),
                TransitionKey(0, 0, 2),
            ),
        )
        self.assertTrue(local_shapley_eligible(
            event,
            has_synergy_ambiguity=True,
            config=config,
        ))
        self.assertFalse(local_shapley_eligible(
            event,
            has_synergy_ambiguity=False,
            config=config,
        ))

    def test_paired_permutations_use_four_forward_reverse_pairs(self):
        candidates = (
            TransitionKey(0, 0, 1),
            TransitionKey(0, 0, 2),
            TransitionKey(0, 0, 3),
        )
        pairs = paired_shapley_permutations(
            candidates,
            pair_count=4,
            seed_material='event-1',
        )

        self.assertEqual(len(pairs), 4)
        for forward, reverse in pairs:
            self.assertEqual(reverse, tuple(reversed(forward)))
            self.assertEqual(set(forward), set(candidates))

    def test_subset_cache_and_efficiency_residual(self):
        candidates = (
            TransitionKey(0, 0, 1),
            TransitionKey(0, 0, 2),
            TransitionKey(0, 0, 3),
        )
        weights = {
            candidates[0]: 1.0,
            candidates[1]: 2.0,
            candidates[2]: 4.0,
        }

        def evaluator(subset):
            value = sum(weights[item] for item in subset)
            if candidates[0] in subset and candidates[1] in subset:
                value += 3.0
            return value

        cache = {}
        first = estimate_local_shapley(
            candidates,
            evaluator,
            subset_cache=cache,
        )
        second = estimate_local_shapley(
            candidates,
            evaluator,
            subset_cache=cache,
        )

        self.assertEqual(first.permutation_count, 8)
        self.assertAlmostEqual(first.empty_value, 0.0)
        self.assertAlmostEqual(first.full_value, 10.0)
        self.assertAlmostEqual(first.efficiency_residual, 0.0)
        self.assertAlmostEqual(
            sum(value for _key, value in first.contributions),
            10.0,
        )
        self.assertGreater(first.evaluated_subset_count, 0)
        self.assertEqual(second.evaluated_subset_count, 0)
        self.assertGreater(second.cache_hit_count, 0)

    def test_cumulative_selector_never_exceeds_point_zero_zero_zero_five(self):
        selector = CumulativeShapleySelector(
            LocalShapleyConfig(event_ratio_max=0.0005)
        )
        for _ in range(1_999):
            self.assertFalse(selector.consider(eligible=True))
        self.assertTrue(selector.consider(eligible=True))

        self.assertEqual(selector.observed_event_count, 2_000)
        self.assertEqual(selector.selected_event_count, 1)
        self.assertLessEqual(selector.selected_ratio, 0.0005)


if __name__ == '__main__':
    unittest.main()
