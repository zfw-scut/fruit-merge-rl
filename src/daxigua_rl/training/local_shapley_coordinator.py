"""主训练进程中的极稀疏局部 Shapley 非阻塞协调器。

协调器只选择最高价值且存在协同歧义的 proposal。被选中的 proposal 不再进入普通
反事实分支；物理 token 通过注入的 ``shared_budget`` 与普通反事实共用同一
8%/10% 账本。预算暂时不足时最多保留四个高优先级任务，rollout 主线程从不等待
物理 worker。
"""

from __future__ import annotations

import multiprocessing
import pickle
import threading
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

from daxigua_rl.attribution.causal_replay import (
    CausalReplayBuffer,
    CausalSample,
    stable_budget_key,
)
from daxigua_rl.attribution.counterfactual import (
    CumulativeShapleySelector,
    FrozenTargetPolicyPayload,
    LocalShapleyConfig,
)
from daxigua_rl.attribution.counterfactual_proposal import (
    CounterfactualProposal,
)
from daxigua_rl.attribution.local_shapley_runner import (
    LocalShapleyResult,
    LocalShapleyTask,
    create_local_shapley_task,
    local_shapley_result_to_causal_samples,
    run_local_shapley_task,
)


_TEMPORARY_BUDGET_REJECTIONS = {
    'external_soft_token_budget',
    'external_hard_token_budget',
}


def _validate_runner(runner):
    if not callable(runner):
        raise TypeError('runner must be callable')
    qualname = getattr(runner, '__qualname__', '')
    if getattr(runner, '__name__', None) == '<lambda>' or (
            isinstance(qualname, str)
            and '<locals>' in qualname):
        raise ValueError(
            'runner must be a module-top-level pickleable callable'
        )
    try:
        pickle.dumps(runner, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise TypeError('runner must be pickleable') from exc
    return runner


def _validate_shared_budget(shared_budget):
    required = (
        'reserve_external_tokens',
        'settle_external_tokens',
        'refund_external_tokens',
    )
    missing = tuple(
        name
        for name in required
        if not callable(getattr(shared_budget, name, None))
    )
    if missing:
        raise TypeError(
            f'shared_budget is missing methods: {missing!r}'
        )
    return shared_budget


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalShapleySubmission:
    """一次 proposal 门控结果；selected 必须跳过普通反事实。"""

    proposal_id: str
    observed: bool
    eligible: bool
    selected: bool
    accepted: bool
    pending: bool
    task_id: str | None = None
    drop_reason: str | None = None

    @property
    def skip_counterfactual(self):
        return self.selected


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalShapleyPoll:
    results: tuple[LocalShapleyResult, ...]
    inserted_samples: tuple[CausalSample, ...]

    @property
    def result_count(self):
        return len(self.results)

    @property
    def inserted_sample_count(self):
        return len(self.inserted_samples)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalShapleyActivityStats:
    proposals_received: int
    proposals_duplicate: int
    proposals_eligible: int
    proposals_selected: int
    selector_quota_rejected: int
    tasks_reserved: int
    reserved_tokens_total: int
    consumed_tokens_total: int
    refunded_tokens_total: int
    tasks_pending_budget: int
    pending_evicted: int
    executor_submit_failures: int
    results_completed: int
    results_failed: int
    reproduction_passed: int
    reproduction_failed: int
    label_ready_results: int
    samples_generated: int
    samples_inserted: int
    samples_rejected: int
    simulated_steps: int
    external_settlements: int
    external_refunds: int
    drop_reason_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalShapleyCoordinatorStats:
    enabled: bool
    closed: bool
    active_task_id: str | None
    pending_task_ids: tuple[str, ...]
    observed_event_count: int
    selected_event_count: int
    selected_ratio: float
    cumulative: LocalShapleyActivityStats
    window: LocalShapleyActivityStats

    @property
    def pending_task_count(self):
        return len(self.pending_task_ids)

    @property
    def skip_counterfactual_count(self):
        return self.selected_event_count


class _Accumulator:
    _FIELDS = (
        'proposals_received',
        'proposals_duplicate',
        'proposals_eligible',
        'proposals_selected',
        'selector_quota_rejected',
        'tasks_reserved',
        'reserved_tokens_total',
        'consumed_tokens_total',
        'refunded_tokens_total',
        'tasks_pending_budget',
        'pending_evicted',
        'executor_submit_failures',
        'results_completed',
        'results_failed',
        'reproduction_passed',
        'reproduction_failed',
        'label_ready_results',
        'samples_generated',
        'samples_inserted',
        'samples_rejected',
        'simulated_steps',
        'external_settlements',
        'external_refunds',
    )

    def __init__(self):
        for name in self._FIELDS:
            setattr(self, name, 0)
        self.drop_reason_counts = Counter()

    def increment(self, name, amount=1):
        setattr(self, name, getattr(self, name) + int(amount))

    def drop(self, reason, count=1):
        self.drop_reason_counts[str(reason)] += int(count)

    def snapshot(self):
        return LocalShapleyActivityStats(
            **{
                name: getattr(self, name)
                for name in self._FIELDS
            },
            drop_reason_counts=tuple(sorted(
                self.drop_reason_counts.items()
            )),
        )


@dataclass(slots=True)
class _PendingTask:
    task: LocalShapleyTask
    queued_at: float
    last_budget_reason: str | None = None


@dataclass(slots=True)
class _ActiveTask:
    task: LocalShapleyTask
    future: object
    started_at: float


class LocalShapleyCoordinator:
    """累计比例门控、单 spawn worker 与共享 token 账本的协调器。"""

    def __init__(
            self,
            *,
            causal_replay_buffer,
            shared_budget,
            config=None,
            executor=None,
            runner=run_local_shapley_task,
            pending_capacity=4,
            enabled=True):
        if not isinstance(
                causal_replay_buffer,
                CausalReplayBuffer):
            raise TypeError(
                'causal_replay_buffer must be CausalReplayBuffer'
            )
        self.causal_replay_buffer = causal_replay_buffer
        self.shared_budget = _validate_shared_budget(shared_budget)
        self.config = config or LocalShapleyConfig()
        if not isinstance(self.config, LocalShapleyConfig):
            raise TypeError('config must be LocalShapleyConfig')
        self.pending_capacity = int(pending_capacity)
        if self.pending_capacity < 1 or self.pending_capacity > 4:
            raise ValueError('pending_capacity must be between 1 and 4')
        self.enabled = bool(enabled)
        self.runner = _validate_runner(runner)
        self.selector = CumulativeShapleySelector(self.config)

        self._owns_executor = executor is None
        if executor is None and self.enabled:
            context = multiprocessing.get_context('spawn')
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
            )
        if self.enabled and not callable(
                getattr(executor, 'submit', None)):
            raise TypeError('executor must provide submit()')
        self._executor = executor

        self._lock = threading.RLock()
        self._closed = False
        self._active = None
        self._pending = {}
        self._seen_proposal_ids = set()
        self._seen_event_keys = set()
        self._seen_budget_keys = set()
        self._selected_by_proposal = {}
        self._selected_by_event = {}
        self._selected_by_budget = {}
        self._finished_task_ids = set()
        self._cumulative = _Accumulator()
        self._window = _Accumulator()
        self._close_stats = None

    @staticmethod
    def _event_identity(proposal):
        event_id = proposal.representative_event.event_id
        return (
            event_id.worker_id,
            event_id.episode_id,
            event_id.event_index,
        )

    @staticmethod
    def _eligible(proposal, config):
        reasons = set(proposal.trigger_reasons)
        synergy = bool(
            reasons.intersection({
                'multi_stage_chain',
                'conflicting_signals',
            })
        )
        return (
            proposal.shapley_ready
            and proposal.representative_event.status == 'confirmed'
            and proposal.utility >= config.minimum_utility
            and synergy
            and len(proposal.coalition_candidate_keys)
            >= config.minimum_candidates
        )

    def _increment(self, name, amount=1):
        self._cumulative.increment(name, amount)
        self._window.increment(name, amount)

    def _drop(self, reason, count=1):
        self._cumulative.drop(reason, count)
        self._window.drop(reason, count)

    def consider(
            self,
            proposal,
            target_policy,
            *,
            created_real_step=0):
        """非阻塞观察 proposal，并返回是否应跳过普通反事实。"""

        if not isinstance(proposal, CounterfactualProposal):
            raise TypeError('proposal must be CounterfactualProposal')
        if not isinstance(target_policy, FrozenTargetPolicyPayload):
            raise TypeError(
                'target_policy must be FrozenTargetPolicyPayload'
            )
        with self._lock:
            self._ensure_open()
            self._poll_locked(dispatch=False)
            self._increment('proposals_received')
            event_identity = self._event_identity(proposal)
            budget_identity = stable_budget_key(proposal.budget_key)
            duplicate = (
                proposal.proposal_id in self._seen_proposal_ids
                or event_identity in self._seen_event_keys
                or budget_identity in self._seen_budget_keys
            )
            if duplicate:
                self._increment('proposals_duplicate')
                self._drop('duplicate_event_or_budget')
                selected = bool(
                    self._selected_by_proposal.get(
                        proposal.proposal_id,
                        False,
                    )
                    or self._selected_by_event.get(
                        event_identity,
                        False,
                    )
                    or self._selected_by_budget.get(
                        budget_identity,
                        False,
                    )
                )
                return LocalShapleySubmission(
                    proposal_id=proposal.proposal_id,
                    observed=False,
                    eligible=self._eligible(
                        proposal,
                        self.config,
                    ),
                    selected=selected,
                    accepted=False,
                    pending=False,
                    drop_reason='duplicate_event_or_budget',
                )

            self._seen_proposal_ids.add(proposal.proposal_id)
            self._seen_event_keys.add(event_identity)
            self._seen_budget_keys.add(budget_identity)
            eligible = self._eligible(proposal, self.config)
            if eligible:
                self._increment('proposals_eligible')
            selected = (
                self.enabled
                and self.selector.consider(eligible=eligible)
            )
            if not self.enabled:
                # disabled 模式仍把唯一事件计入累计比例分母。
                self.selector.consider(eligible=False)
                reason = 'disabled'
            elif not eligible:
                reason = 'ineligible'
            elif not selected:
                self._increment('selector_quota_rejected')
                reason = 'selector_cumulative_quota'
            else:
                reason = None

            self._selected_by_proposal[proposal.proposal_id] = selected
            self._selected_by_event[event_identity] = selected
            self._selected_by_budget[budget_identity] = selected
            if not selected:
                self._drop(reason)
                return LocalShapleySubmission(
                    proposal_id=proposal.proposal_id,
                    observed=True,
                    eligible=eligible,
                    selected=False,
                    accepted=False,
                    pending=False,
                    drop_reason=reason,
                )

            self._increment('proposals_selected')
            try:
                task = create_local_shapley_task(
                    proposal,
                    target_policy,
                    config=self.config,
                    created_real_step=created_real_step,
                )
            except Exception:
                self._drop('task_creation_failure')
                return LocalShapleySubmission(
                    proposal_id=proposal.proposal_id,
                    observed=True,
                    eligible=True,
                    selected=True,
                    accepted=False,
                    pending=False,
                    drop_reason='task_creation_failure',
                )

            retained = self._enqueue_locked(task)
            if not retained:
                return LocalShapleySubmission(
                    proposal_id=proposal.proposal_id,
                    observed=True,
                    eligible=True,
                    selected=True,
                    accepted=False,
                    pending=False,
                    task_id=task.task_id,
                    drop_reason='pending_capacity',
                )
            self._dispatch_pending_locked()
            active = (
                self._active is not None
                and self._active.task.task_id == task.task_id
            )
            pending = task.task_id in self._pending
            pending_reason = (
                self._pending[task.task_id].last_budget_reason
                if pending
                else None
            )
            return LocalShapleySubmission(
                proposal_id=proposal.proposal_id,
                observed=True,
                eligible=True,
                selected=True,
                accepted=active or (
                    pending and pending_reason is None
                ),
                pending=pending,
                task_id=task.task_id,
                drop_reason=pending_reason,
            )

    def poll(self):
        """只收割已经完成的 future，绝不等待正在运行的物理任务。"""

        with self._lock:
            self._ensure_open()
            return self._poll_locked(dispatch=True)

    def retry_pending(self):
        """真实步数增长后显式重试预算暂不足的至多四个任务。"""

        with self._lock:
            self._ensure_open()
            result = self._poll_locked(dispatch=False)
            self._dispatch_pending_locked()
            return result

    def _enqueue_locked(self, task):
        if self._active is None and not self._pending:
            self._pending[task.task_id] = _PendingTask(
                task=task,
                queued_at=time.perf_counter(),
            )
            return True
        if len(self._pending) < self.pending_capacity:
            self._pending[task.task_id] = _PendingTask(
                task=task,
                queued_at=time.perf_counter(),
            )
            return True
        lowest = min(
            self._pending.values(),
            key=lambda item: (
                item.task.priority,
                -item.task.created_real_step,
                item.task.task_id,
            ),
        )
        if (
                task.priority,
                -task.created_real_step,
                task.task_id,
        ) <= (
                lowest.task.priority,
                -lowest.task.created_real_step,
                lowest.task.task_id,
        ):
            self._drop('pending_capacity')
            return False
        del self._pending[lowest.task.task_id]
        self._increment('pending_evicted')
        self._drop('pending_priority_evicted')
        self._pending[task.task_id] = _PendingTask(
            task=task,
            queued_at=time.perf_counter(),
        )
        return True

    def _dispatch_pending_locked(self):
        if (
                self._closed
                or not self.enabled
                or self._active is not None):
            return
        while self._pending and self._active is None:
            pending = max(
                self._pending.values(),
                key=lambda item: (
                    item.task.priority,
                    -item.task.created_real_step,
                    item.task.task_id,
                ),
            )
            task = pending.task
            try:
                decision = self.shared_budget.reserve_external_tokens(
                    task.task_id,
                    task.estimated_tokens,
                    priority=task.priority,
                )
            except Exception:
                del self._pending[task.task_id]
                self._drop('external_reservation_exception')
                continue
            if not decision.accepted:
                reason = (
                    decision.drop_reason
                    or 'external_reservation_rejected'
                )
                if reason in _TEMPORARY_BUDGET_REJECTIONS:
                    pending.last_budget_reason = reason
                    self._increment('tasks_pending_budget')
                    self._drop(reason)
                    return
                del self._pending[task.task_id]
                self._drop(reason)
                continue

            del self._pending[task.task_id]
            self._increment('tasks_reserved')
            self._increment(
                'reserved_tokens_total',
                task.estimated_tokens,
            )
            try:
                future = self._executor.submit(
                    self.runner,
                    task,
                )
            except BaseException:
                self._increment('executor_submit_failures')
                self._drop('executor_submit_failure')
                try:
                    refunded = (
                        self.shared_budget.refund_external_tokens(
                            task.task_id
                        )
                    )
                    if refunded.accepted:
                        self._increment('external_refunds')
                        self._increment(
                            'refunded_tokens_total',
                            refunded.tokens,
                        )
                except Exception:
                    self._drop('external_refund_exception')
                continue
            self._active = _ActiveTask(
                task=task,
                future=future,
                started_at=time.perf_counter(),
            )

    def _poll_locked(self, *, dispatch):
        results = []
        inserted_samples = []
        active = self._active
        if active is not None and active.future.done():
            self._active = None
            result, samples = self._finalize_active_locked(active)
            if result is not None:
                results.append(result)
            inserted_samples.extend(samples)
        if dispatch:
            self._dispatch_pending_locked()
        return LocalShapleyPoll(
            results=tuple(results),
            inserted_samples=tuple(inserted_samples),
        )

    def _finalize_active_locked(self, active):
        task = active.task
        try:
            result = active.future.result()
            if not isinstance(result, LocalShapleyResult):
                raise TypeError(
                    'runner must return LocalShapleyResult'
                )
            if result.task_id != task.task_id:
                raise ValueError('runner result task_id mismatch')
            if result.simulated_steps > task.estimated_tokens:
                raise ValueError(
                    'runner exceeded reserved physical tokens'
                )
        except BaseException:
            # runner 进程异常时无法知道已经执行了多少步；按完整预留保守结算。
            self._settle_locked(task, task.estimated_tokens)
            self._increment('results_failed')
            self._increment('reproduction_failed')
            self._drop('runner_failure')
            self._finished_task_ids.add(task.task_id)
            return None, ()

        settled = self._settle_locked(
            task,
            result.simulated_steps,
        )
        self._increment('simulated_steps', result.simulated_steps)
        if result.status == 'completed':
            self._increment('results_completed')
        else:
            self._increment('results_failed')
        if result.grand_reproduced:
            self._increment('reproduction_passed')
        else:
            self._increment('reproduction_failed')
        samples = ()
        if result.label_ready and settled:
            self._increment('label_ready_results')
            try:
                samples = local_shapley_result_to_causal_samples(
                    task,
                    result,
                )
            except Exception:
                self._drop('sample_conversion_failure')
                samples = ()
            self._increment('samples_generated', len(samples))
            inserted = self.causal_replay_buffer.extend(samples)
            self._increment('samples_inserted', inserted)
            self._increment(
                'samples_rejected',
                len(samples) - inserted,
            )
        elif result.label_ready:
            self._drop('budget_settlement_failed_no_label')
        self._finished_task_ids.add(task.task_id)
        return result, samples

    def _settle_locked(self, task, consumed):
        try:
            decision = self.shared_budget.settle_external_tokens(
                task.task_id,
                consumed,
            )
        except Exception:
            self._drop('external_settlement_exception')
            return False
        if not decision.accepted:
            self._drop(
                decision.drop_reason
                or 'external_settlement_rejected'
            )
            return False
        self._increment('external_settlements')
        self._increment('consumed_tokens_total', consumed)
        self._increment(
            'refunded_tokens_total',
            task.estimated_tokens - consumed,
        )
        return True

    @property
    def stats(self):
        with self._lock:
            return self._stats_locked()

    def snapshot_stats(self, *, reset_window=False):
        with self._lock:
            stats = self._stats_locked()
            if reset_window:
                self._window = _Accumulator()
            return stats

    def _stats_locked(self):
        return LocalShapleyCoordinatorStats(
            enabled=self.enabled,
            closed=self._closed,
            active_task_id=(
                None
                if self._active is None
                else self._active.task.task_id
            ),
            pending_task_ids=tuple(sorted(self._pending)),
            observed_event_count=(
                self.selector.observed_event_count
            ),
            selected_event_count=(
                self.selector.selected_event_count
            ),
            selected_ratio=float(self.selector.selected_ratio),
            cumulative=self._cumulative.snapshot(),
            window=self._window.snapshot(),
        )

    def checkpoint_state(self):
        """只返回 JSON-safe 聚合信息，不序列化 future、快照、图或模型。"""

        stats = self.stats
        payload = asdict(stats)
        payload['cumulative']['drop_reason_counts'] = dict(
            stats.cumulative.drop_reason_counts
        )
        payload['window']['drop_reason_counts'] = dict(
            stats.window.drop_reason_counts
        )
        return payload

    def summary(self):
        stats = self.stats
        return {
            'enabled': stats.enabled,
            'closed': stats.closed,
            'observed_events': stats.observed_event_count,
            'selected_events': stats.selected_event_count,
            'selected_ratio': stats.selected_ratio,
            'active': int(stats.active_task_id is not None),
            'pending': stats.pending_task_count,
            'completed': stats.cumulative.results_completed,
            'failed': stats.cumulative.results_failed,
            'label_ready': stats.cumulative.label_ready_results,
            'samples_inserted': (
                stats.cumulative.samples_inserted
            ),
            'simulated_steps': stats.cumulative.simulated_steps,
            'reserved_tokens_total': (
                stats.cumulative.reserved_tokens_total
            ),
            'consumed_tokens_total': (
                stats.cumulative.consumed_tokens_total
            ),
            'refunded_tokens_total': (
                stats.cumulative.refunded_tokens_total
            ),
        }

    def close(self, *, wait=True):
        """幂等关闭；默认等待唯一 active 任务并结算共享账本。"""

        wait = bool(wait)
        with self._lock:
            if self._closed:
                return self._close_stats
            self._closed = True
            for pending in tuple(self._pending.values()):
                self._drop('coordinator_closed_pending')
            self._pending.clear()
            active = self._active

        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(
                wait=wait,
                cancel_futures=not wait,
            )

        with self._lock:
            if active is not None:
                self._active = None
                if wait:
                    self._finalize_active_locked(active)
                else:
                    cancelled = bool(active.future.cancel())
                    if cancelled:
                        try:
                            decision = (
                                self.shared_budget
                                .refund_external_tokens(
                                    active.task.task_id
                                )
                            )
                            if decision.accepted:
                                self._increment(
                                    'external_refunds'
                                )
                                self._increment(
                                    'refunded_tokens_total',
                                    decision.tokens,
                                )
                        except Exception:
                            self._drop(
                                'external_refund_exception'
                            )
                    else:
                        # 无法取消的后台进程按最坏预留结算，避免低报硬预算。
                        self._settle_locked(
                            active.task,
                            active.task.estimated_tokens,
                        )
                    self._drop('coordinator_closed_active')
            self._close_stats = self._stats_locked()
            return self._close_stats

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError('LocalShapleyCoordinator is closed')


__all__ = [
    'LocalShapleyActivityStats',
    'LocalShapleyCoordinator',
    'LocalShapleyCoordinatorStats',
    'LocalShapleyPoll',
    'LocalShapleySubmission',
]
