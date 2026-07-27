"""主进程稀疏反事实协调器。

``CounterfactualCoordinator`` 只在训练主进程存在，负责把 worker 返回的不可变
``CounterfactualProposal`` 绑定到最近一次 target sync 的冻结策略，并交给独立
spawn 调度器。rollout 主路径只调用 ``record_real_steps``、``submit`` 和 ``poll``；
三者都只处理已经完成的 future，不等待物理分支。

协调器不会进入 checkpoint 的内容包括 executor、future、完整快照、上下文图和模型
权重 bytes。checkpoint 只保存版本、稳定指纹、计数和 token 账本摘要。
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from operator import index

from daxigua_rl.attribution.causal_replay import (
    CausalReplayBuffer,
    CausalSample,
)
from daxigua_rl.attribution.counterfactual import (
    BudgetedCounterfactualScheduler,
    CounterfactualConfig,
    CounterfactualResult,
    CounterfactualSchedulerStats,
    CounterfactualTokenDecision,
    create_counterfactual_task,
)
from daxigua_rl.attribution.counterfactual_proposal import (
    CounterfactualProposal,
)
from daxigua_rl.attribution.counterfactual_runner import (
    counterfactual_result_to_causal_samples,
    freeze_target_policy_payload,
    run_counterfactual_task,
)


COUNTERFACTUAL_COORDINATOR_CHECKPOINT_VERSION = 1


def _strict_int(name, value, *, minimum=None):
    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer')
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return result


def _non_empty_text(name, value):
    if not isinstance(value, str):
        raise TypeError(f'{name} must be str')
    result = value.strip()
    if not result:
        raise ValueError(f'{name} must not be empty')
    return result


def recommended_counterfactual_worker_count(
        *,
        cpu_count=None,
        rollout_worker_count):
    """按 ``min(floor(cpu*25%), cpu-rollout-1)`` 推荐独立物理进程数。"""

    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    cpu_count = _strict_int(
        'cpu_count',
        cpu_count,
        minimum=1,
    )
    rollout_worker_count = _strict_int(
        'rollout_worker_count',
        rollout_worker_count,
        minimum=0,
    )
    quarter = math.floor(cpu_count * 0.25)
    spare = cpu_count - rollout_worker_count - 1
    return max(0, min(quarter, spare))


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualCoordinatorSubmission:
    """一个 proposal 的主进程 admission 结果。"""

    proposal_id: str
    accepted: bool
    task_id: str | None = None
    drop_reason: str | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            'proposal_id',
            _non_empty_text('proposal_id', self.proposal_id),
        )
        if not isinstance(self.accepted, bool):
            raise TypeError('accepted must be bool')
        if self.accepted:
            object.__setattr__(
                self,
                'task_id',
                _non_empty_text('task_id', self.task_id),
            )
            if self.drop_reason is not None:
                raise ValueError(
                    'accepted submission cannot have drop_reason'
                )
        else:
            if self.task_id is not None:
                object.__setattr__(
                    self,
                    'task_id',
                    _non_empty_text('task_id', self.task_id),
                )
            object.__setattr__(
                self,
                'drop_reason',
                _non_empty_text(
                    'drop_reason',
                    self.drop_reason,
                ),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualCoordinatorActivityStats:
    """累计或窗口内的协调器事件计数。"""

    proposals_received: int
    proposals_admitted: int
    proposals_rejected: int
    results_completed: int
    results_partial: int
    results_failed: int
    reproduction_passed: int
    reproduction_failed: int
    branches_completed: int
    branches_partial: int
    branches_failed: int
    label_ready_results: int
    samples_generated: int
    samples_inserted: int
    samples_rejected: int
    pending_cleanups_without_result: int
    orphan_or_duplicate_results: int
    target_refreshes: int
    target_refresh_seconds: float
    result_wall_seconds_total: float
    result_wall_seconds_max: float
    drop_reason_counts: tuple[tuple[str, int], ...]

    @property
    def mean_result_wall_seconds(self):
        result_count = (
            self.results_completed
            + self.results_partial
            + self.results_failed
        )
        if result_count <= 0:
            return 0.0
        return self.result_wall_seconds_total / result_count

    def to_dict(self):
        result = asdict(self)
        result['drop_reason_counts'] = dict(
            self.drop_reason_counts
        )
        result['mean_result_wall_seconds'] = (
            self.mean_result_wall_seconds
        )
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualCoordinatorStats:
    """完整、可 pickle/JSON 化的协调器状态快照。"""

    enabled: bool
    closed: bool
    cpu_count: int
    rollout_worker_count: int
    worker_count: int
    target_policy_version: str | None
    target_policy_fingerprint: str | None
    active_task_ids: tuple[str, ...]
    pending_task_count: int
    scheduler: CounterfactualSchedulerStats | None
    cumulative: CounterfactualCoordinatorActivityStats
    window: CounterfactualCoordinatorActivityStats
    actual_token_ratio: float
    projected_token_ratio: float
    soft_budget_utilization: float
    hard_budget_utilization: float
    hard_budget_respected: bool
    circuit_open: bool

    @property
    def proposals_received(self):
        return self.cumulative.proposals_received

    @property
    def proposals_admitted(self):
        return self.cumulative.proposals_admitted

    @property
    def proposals_rejected(self):
        return self.cumulative.proposals_rejected

    @property
    def samples_inserted(self):
        return self.cumulative.samples_inserted

    def to_dict(self):
        result = {
            'enabled': self.enabled,
            'closed': self.closed,
            'cpu_count': self.cpu_count,
            'rollout_worker_count': self.rollout_worker_count,
            'worker_count': self.worker_count,
            'target_policy_version': self.target_policy_version,
            'target_policy_fingerprint': (
                self.target_policy_fingerprint
            ),
            'active_task_ids': list(self.active_task_ids),
            'pending_task_count': self.pending_task_count,
            'scheduler': (
                asdict(self.scheduler)
                if self.scheduler is not None
                else None
            ),
            'cumulative': self.cumulative.to_dict(),
            'window': self.window.to_dict(),
            'actual_token_ratio': self.actual_token_ratio,
            'projected_token_ratio': self.projected_token_ratio,
            'soft_budget_utilization': (
                self.soft_budget_utilization
            ),
            'hard_budget_utilization': (
                self.hard_budget_utilization
            ),
            'hard_budget_respected': self.hard_budget_respected,
            'circuit_open': self.circuit_open,
        }
        if result['scheduler'] is not None:
            result['scheduler']['drop_reason_counts'] = dict(
                self.scheduler.drop_reason_counts
            )
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualCoordinatorPoll:
    """一次非阻塞 poll 已消费的结果和成功写入 replay 的样本。"""

    results: tuple[CounterfactualResult, ...]
    inserted_samples: tuple[CausalSample, ...]

    @property
    def result_count(self):
        return len(self.results)

    @property
    def inserted_sample_count(self):
        return len(self.inserted_samples)


@dataclass(slots=True)
class _PendingTask:
    task: object
    context: object
    proposal_id: str
    submitted_at: float


class _ActivityAccumulator:
    """内部可变计数器；公开面始终转换成 frozen snapshot。"""

    _INTEGER_FIELDS = (
        'proposals_received',
        'proposals_admitted',
        'proposals_rejected',
        'results_completed',
        'results_partial',
        'results_failed',
        'reproduction_passed',
        'reproduction_failed',
        'branches_completed',
        'branches_partial',
        'branches_failed',
        'label_ready_results',
        'samples_generated',
        'samples_inserted',
        'samples_rejected',
        'pending_cleanups_without_result',
        'orphan_or_duplicate_results',
        'target_refreshes',
    )

    def __init__(self):
        for field_name in self._INTEGER_FIELDS:
            setattr(self, field_name, 0)
        self.target_refresh_seconds = 0.0
        self.result_wall_seconds_total = 0.0
        self.result_wall_seconds_max = 0.0
        self.drop_reason_counts = Counter()

    def increment(self, field_name, amount=1):
        setattr(
            self,
            field_name,
            getattr(self, field_name) + int(amount),
        )

    def add_drop_reason(self, reason, count=1):
        self.drop_reason_counts[str(reason)] += int(count)

    def add_result_wall_seconds(self, value):
        value = max(0.0, float(value))
        self.result_wall_seconds_total += value
        self.result_wall_seconds_max = max(
            self.result_wall_seconds_max,
            value,
        )

    def snapshot(self):
        return CounterfactualCoordinatorActivityStats(
            **{
                field_name: getattr(self, field_name)
                for field_name in self._INTEGER_FIELDS
            },
            target_refresh_seconds=float(
                self.target_refresh_seconds
            ),
            result_wall_seconds_total=float(
                self.result_wall_seconds_total
            ),
            result_wall_seconds_max=float(
                self.result_wall_seconds_max
            ),
            drop_reason_counts=tuple(sorted(
                self.drop_reason_counts.items()
            )),
        )


class CounterfactualCoordinator:
    """主进程 proposal→task→result→CausalReplay 的全局闭环。"""

    def __init__(
            self,
            *,
            causal_replay_buffer,
            rollout_worker_count,
            cpu_count=None,
            worker_count=None,
            scheduler_config=None,
            executor=None,
            runner=run_counterfactual_task):
        if not isinstance(
                causal_replay_buffer,
                CausalReplayBuffer):
            raise TypeError(
                'causal_replay_buffer must be CausalReplayBuffer'
            )
        self.causal_replay_buffer = causal_replay_buffer
        self.cpu_count = _strict_int(
            'cpu_count',
            (os.cpu_count() or 1)
            if cpu_count is None
            else cpu_count,
            minimum=1,
        )
        self.rollout_worker_count = _strict_int(
            'rollout_worker_count',
            rollout_worker_count,
            minimum=0,
        )
        self.recommended_worker_count = (
            recommended_counterfactual_worker_count(
                cpu_count=self.cpu_count,
                rollout_worker_count=self.rollout_worker_count,
            )
        )
        if worker_count is None:
            worker_count = self.recommended_worker_count
        self.worker_count = _strict_int(
            'worker_count',
            worker_count,
            minimum=0,
        )
        self.config = scheduler_config or CounterfactualConfig()
        if not isinstance(self.config, CounterfactualConfig):
            raise TypeError(
                'scheduler_config must be CounterfactualConfig'
            )

        self._lock = threading.RLock()
        self._closed = False
        self._target_policy = None
        self._pending = {}
        self._processed_task_ids = set()
        self._cumulative = _ActivityAccumulator()
        self._window = _ActivityAccumulator()
        self._last_scheduler_drop_reasons = Counter()
        self._disabled_real_steps = 0

        self._scheduler = None
        if self.worker_count > 0:
            self._scheduler = BudgetedCounterfactualScheduler(
                worker_count=self.worker_count,
                runner=runner,
                config=self.config,
                executor=executor,
            )

    @property
    def enabled(self):
        return self._scheduler is not None

    @property
    def closed(self):
        return self._closed

    @property
    def target_policy(self):
        return self._target_policy

    @property
    def pending_task_ids(self):
        with self._lock:
            return tuple(sorted(self._pending))

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError(
                'counterfactual coordinator is closed'
            )

    def _increment(self, field_name, amount=1):
        self._cumulative.increment(field_name, amount)
        self._window.increment(field_name, amount)

    def _add_drop_reason(self, reason, count=1):
        self._cumulative.add_drop_reason(reason, count)
        self._window.add_drop_reason(reason, count)

    def _add_result_wall_seconds(self, value):
        self._cumulative.add_result_wall_seconds(value)
        self._window.add_result_wall_seconds(value)

    def refresh_target_policy(
            self,
            *,
            model,
            model_config,
            policy_version,
            gamma,
            max_physics_frames,
            stable_frames,
            reward_config=None,
            state_analyzer_config=None,
            graph_builder_config=None):
        """仅供 target sync 调用；生成一次 CPU payload 并缓存给后续 proposal。"""

        with self._lock:
            self._ensure_open()
            if not self.enabled:
                return None
            started = time.perf_counter()
            payload = freeze_target_policy_payload(
                model=model,
                model_config=model_config,
                policy_version=policy_version,
                gamma=gamma,
                max_physics_frames=max_physics_frames,
                stable_frames=stable_frames,
                reward_config=reward_config,
                state_analyzer_config=state_analyzer_config,
                graph_builder_config=graph_builder_config,
            )
            elapsed = time.perf_counter() - started
            self._target_policy = payload
            self._increment('target_refreshes')
            self._cumulative.target_refresh_seconds += elapsed
            self._window.target_refresh_seconds += elapsed
            return payload

    def record_real_steps(self, step_count):
        """记录真实投放数并顺手消费已完成结果，全程不等待 future。"""

        step_count = _strict_int(
            'step_count',
            step_count,
            minimum=0,
        )
        with self._lock:
            self._ensure_open()
            if not self.enabled:
                self._disabled_real_steps += step_count
                return self._disabled_real_steps
            real_steps = self._scheduler.record_real_steps(step_count)
            self._drain_scheduler_locked()
            return real_steps

    def reserve_external_tokens(
            self,
            reservation_id,
            tokens,
            *,
            priority=0.0):
        """薄封装：让局部 Shapley 与 CF 共用同一个 8%/10% 账本。"""

        reservation_id = _non_empty_text(
            'reservation_id',
            reservation_id,
        )
        with self._lock:
            self._ensure_open()
            if not self.enabled:
                self._add_drop_reason(
                    'external_disabled_no_workers'
                )
                return CounterfactualTokenDecision(
                    accepted=False,
                    reservation_id=reservation_id,
                    tokens=0,
                    drop_reason='external_disabled_no_workers',
                )
            decision = self._scheduler.reserve_external_tokens(
                reservation_id,
                tokens,
                priority=priority,
            )
            self._drain_scheduler_locked()
            return decision

    def settle_external_tokens(self, reservation_id, consumed):
        """薄封装：按 Shapley 实际物理步结算并退回未消费 token。"""

        reservation_id = _non_empty_text(
            'reservation_id',
            reservation_id,
        )
        with self._lock:
            self._ensure_open()
            if not self.enabled:
                self._add_drop_reason(
                    'external_disabled_no_workers'
                )
                return CounterfactualTokenDecision(
                    accepted=False,
                    reservation_id=reservation_id,
                    tokens=0,
                    drop_reason='external_disabled_no_workers',
                )
            decision = self._scheduler.settle_external_tokens(
                reservation_id,
                consumed,
            )
            self._drain_scheduler_locked()
            return decision

    def refund_external_tokens(self, reservation_id):
        """薄封装：Shapley 未运行时完整退款。"""

        reservation_id = _non_empty_text(
            'reservation_id',
            reservation_id,
        )
        with self._lock:
            self._ensure_open()
            if not self.enabled:
                self._add_drop_reason(
                    'external_disabled_no_workers'
                )
                return CounterfactualTokenDecision(
                    accepted=False,
                    reservation_id=reservation_id,
                    tokens=0,
                    drop_reason='external_disabled_no_workers',
                )
            decision = self._scheduler.refund_external_tokens(
                reservation_id
            )
            self._drain_scheduler_locked()
            return decision

    def _coordinator_rejection(self, proposal, reason):
        self._increment('proposals_rejected')
        self._add_drop_reason(reason)
        return CounterfactualCoordinatorSubmission(
            proposal_id=proposal.proposal_id,
            accepted=False,
            task_id=None,
            drop_reason=reason,
        )

    def submit(self, proposal):
        """非阻塞提交一个严格 proposal，并立即消费已经完成的旧任务。"""

        if not isinstance(proposal, CounterfactualProposal):
            raise TypeError(
                'proposal must be CounterfactualProposal'
            )
        with self._lock:
            self._increment('proposals_received')
            if self._closed:
                return self._coordinator_rejection(
                    proposal,
                    'coordinator_closed',
                )
            if not self.enabled:
                return self._coordinator_rejection(
                    proposal,
                    'disabled_no_workers',
                )
            if self._target_policy is None:
                return self._coordinator_rejection(
                    proposal,
                    'target_policy_unavailable',
                )

            task = create_counterfactual_task(
                budget_key=proposal.budget_key,
                transition_key=proposal.transition_key,
                snapshot=proposal.snapshot,
                factual_outcome=proposal.factual_outcome,
                target_policy=self._target_policy,
                actual_action_offset=(
                    proposal.actual_action_offset
                ),
                alternative_action_offsets=(
                    proposal.alternative_action_offsets
                ),
                trigger_reasons=proposal.trigger_reasons,
                event_utility=proposal.utility,
                placement_confidence=proposal.confidence,
                created_real_step=(
                    self._scheduler.stats.real_steps
                ),
                attribution_version=(
                    proposal.attribution_version
                ),
                config=self.config,
                label_confidence=proposal.confidence,
                attribution_delay=proposal.delay,
            )
            if task is None:
                return self._coordinator_rejection(
                    proposal,
                    'ineligible_task',
                )
            decision = self._scheduler.submit(task)
            if decision.accepted:
                self._increment('proposals_admitted')
                self._pending[task.task_id] = _PendingTask(
                    task=task,
                    context=proposal.context,
                    proposal_id=proposal.proposal_id,
                    submitted_at=time.perf_counter(),
                )
            else:
                self._increment('proposals_rejected')

            self._drain_scheduler_locked()
            return CounterfactualCoordinatorSubmission(
                proposal_id=proposal.proposal_id,
                accepted=decision.accepted,
                task_id=decision.task_id,
                drop_reason=decision.drop_reason,
            )

    def submit_many(self, proposals, *, real_steps=0):
        """记录一批真实步并逐个非阻塞 admission。"""

        proposals = tuple(proposals)
        if real_steps:
            self.record_real_steps(real_steps)
        return tuple(self.submit(proposal) for proposal in proposals)

    def poll(self):
        """消费当前已经完成的 future；没有完成项时立即返回空快照。"""

        with self._lock:
            if not self.enabled:
                return CounterfactualCoordinatorPoll(
                    results=(),
                    inserted_samples=(),
                )
            return self._drain_scheduler_locked()

    def _sync_scheduler_drop_reasons_locked(self):
        if not self.enabled:
            return
        current = Counter(dict(
            self._scheduler.stats.drop_reason_counts
        ))
        for reason, count in current.items():
            delta = count - self._last_scheduler_drop_reasons.get(
                reason,
                0,
            )
            if delta > 0:
                self._add_drop_reason(reason, delta)
        self._last_scheduler_drop_reasons = current

    def _reconcile_pending_locked(self):
        if not self.enabled:
            return
        active = set(self._scheduler.active_task_ids)
        stale_ids = tuple(
            task_id
            for task_id in self._pending
            if task_id not in active
        )
        now = time.perf_counter()
        for task_id in stale_ids:
            pending = self._pending.pop(task_id)
            self._increment('pending_cleanups_without_result')
            self._add_drop_reason('pending_without_result')
            self._add_result_wall_seconds(
                now - pending.submitted_at
            )

    def _consume_result_locked(self, result):
        if result.task_id in self._processed_task_ids:
            self._increment('orphan_or_duplicate_results')
            self._add_drop_reason('duplicate_result')
            return ()
        pending = self._pending.pop(result.task_id, None)
        if pending is None:
            self._processed_task_ids.add(result.task_id)
            self._increment('orphan_or_duplicate_results')
            self._add_drop_reason('orphan_result')
            return ()
        self._processed_task_ids.add(result.task_id)
        self._add_result_wall_seconds(
            time.perf_counter() - pending.submitted_at
        )

        self._increment(f'results_{result.status}')
        if result.original_reproduced:
            self._increment('reproduction_passed')
        else:
            self._increment('reproduction_failed')
        for branch in result.branches:
            self._increment(f'branches_{branch.status}')

        if not result.label_ready:
            return ()
        self._increment('label_ready_results')
        try:
            samples = counterfactual_result_to_causal_samples(
                pending.task,
                result,
                pending.context,
            )
        except Exception:
            self._add_drop_reason('label_conversion_failure')
            return ()
        self._increment('samples_generated', len(samples))

        inserted = []
        for sample in samples:
            try:
                accepted = self.causal_replay_buffer.push(sample)
            except Exception:
                accepted = False
                self._add_drop_reason('causal_replay_push_failure')
            if accepted:
                inserted.append(sample)
                self._increment('samples_inserted')
            else:
                self._increment('samples_rejected')
        return tuple(inserted)

    def _drain_scheduler_locked(self):
        results = self._scheduler.drain_results()
        inserted = []
        for result in results:
            inserted.extend(self._consume_result_locked(result))
        self._sync_scheduler_drop_reasons_locked()
        self._reconcile_pending_locked()
        return CounterfactualCoordinatorPoll(
            results=tuple(results),
            inserted_samples=tuple(inserted),
        )

    def close(self, *, wait=True):
        """幂等关闭；默认等待已运行分支并消费最后一批可信标签。"""

        wait = bool(wait)
        with self._lock:
            if self._closed:
                return self.stats
            if self.enabled:
                self._drain_scheduler_locked()
                final_results = self._scheduler.close(wait=wait)
                for result in final_results:
                    self._consume_result_locked(result)
                self._sync_scheduler_drop_reasons_locked()
                self._reconcile_pending_locked()
                # wait=False 时不可取消的运行 future 可能仍在 scheduler 内，但协调器
                # 已关闭且不得再保留其大快照/context。
                if self._pending:
                    now = time.perf_counter()
                    for pending in self._pending.values():
                        self._increment(
                            'pending_cleanups_without_result'
                        )
                        self._add_drop_reason(
                            'close_pending_without_result'
                        )
                        self._add_result_wall_seconds(
                            now - pending.submitted_at
                        )
                    self._pending.clear()
            self._closed = True
            return self.stats

    def _activity_pair(self, *, reset_window):
        cumulative = self._cumulative.snapshot()
        window = self._window.snapshot()
        if reset_window:
            self._window = _ActivityAccumulator()
        return cumulative, window

    def snapshot_stats(self, *, reset_window=False):
        """返回累计+窗口统计；可原子清空下一窗口计数。"""

        reset_window = bool(reset_window)
        with self._lock:
            scheduler_stats = (
                self._scheduler.stats
                if self.enabled
                else None
            )
            active_task_ids = (
                self._scheduler.active_task_ids
                if self.enabled
                else ()
            )
            real_steps = (
                scheduler_stats.real_steps
                if scheduler_stats is not None
                else self._disabled_real_steps
            )
            consumed = (
                scheduler_stats.tokens_consumed
                if scheduler_stats is not None
                else 0
            )
            reserved = (
                scheduler_stats.tokens_reserved
                if scheduler_stats is not None
                else 0
            )
            projected = consumed + reserved
            actual_ratio = (
                consumed / real_steps
                if real_steps > 0
                else 0.0
            )
            projected_ratio = (
                projected / real_steps
                if real_steps > 0
                else 0.0
            )
            soft_limit = (
                scheduler_stats.soft_token_limit
                if scheduler_stats is not None
                else 0.0
            )
            hard_limit = (
                scheduler_stats.hard_token_limit
                if scheduler_stats is not None
                else 0.0
            )
            soft_utilization = (
                projected / soft_limit
                if soft_limit > 0.0
                else 0.0
            )
            hard_utilization = (
                projected / hard_limit
                if hard_limit > 0.0
                else 0.0
            )
            hard_respected = (
                projected <= hard_limit + 1e-12
                and (
                    scheduler_stats is None
                    or scheduler_stats.token_overrun == 0
                )
            ) if real_steps > 0 else projected == 0
            cumulative, window = self._activity_pair(
                reset_window=reset_window
            )
            return CounterfactualCoordinatorStats(
                enabled=self.enabled,
                closed=self._closed,
                cpu_count=self.cpu_count,
                rollout_worker_count=self.rollout_worker_count,
                worker_count=self.worker_count,
                target_policy_version=(
                    self._target_policy.policy_version
                    if self._target_policy is not None
                    else None
                ),
                target_policy_fingerprint=(
                    self._target_policy.fingerprint
                    if self._target_policy is not None
                    else None
                ),
                active_task_ids=active_task_ids,
                pending_task_count=len(self._pending),
                scheduler=scheduler_stats,
                cumulative=cumulative,
                window=window,
                actual_token_ratio=float(actual_ratio),
                projected_token_ratio=float(projected_ratio),
                soft_budget_utilization=float(
                    soft_utilization
                ),
                hard_budget_utilization=float(
                    hard_utilization
                ),
                hard_budget_respected=bool(hard_respected),
                circuit_open=bool(
                    scheduler_stats.circuit_open
                    if scheduler_stats is not None
                    else False
                ),
            )

    @property
    def stats(self):
        return self.snapshot_stats(reset_window=False)

    def checkpoint_state(self):
        """返回不含 executor/future/快照/图/权重的轻量 checkpoint 状态。"""

        stats = self.stats
        return {
            'schema_version': (
                COUNTERFACTUAL_COORDINATOR_CHECKPOINT_VERSION
            ),
            'scheduler_config_fingerprint': self.config.fingerprint,
            'recommended_worker_count': (
                self.recommended_worker_count
            ),
            **stats.to_dict(),
        }

    def summary(self):
        """返回适合日志单行化的累计摘要。"""

        stats = self.stats
        scheduler = stats.scheduler
        return {
            'enabled': stats.enabled,
            'closed': stats.closed,
            'worker_count': stats.worker_count,
            'target_policy_version': (
                stats.target_policy_version
            ),
            'proposals_received': stats.proposals_received,
            'proposals_admitted': stats.proposals_admitted,
            'proposals_rejected': stats.proposals_rejected,
            'pending_task_count': stats.pending_task_count,
            'reproduction_passed': (
                stats.cumulative.reproduction_passed
            ),
            'reproduction_failed': (
                stats.cumulative.reproduction_failed
            ),
            'label_ready_results': (
                stats.cumulative.label_ready_results
            ),
            'samples_inserted': stats.samples_inserted,
            'tokens_consumed': (
                scheduler.tokens_consumed
                if scheduler is not None
                else 0
            ),
            'actual_token_ratio': stats.actual_token_ratio,
            'projected_token_ratio': (
                stats.projected_token_ratio
            ),
            'hard_budget_respected': (
                stats.hard_budget_respected
            ),
            'circuit_open': stats.circuit_open,
            'drop_reason_counts': dict(
                stats.cumulative.drop_reason_counts
            ),
        }


__all__ = [
    'COUNTERFACTUAL_COORDINATOR_CHECKPOINT_VERSION',
    'CounterfactualCoordinator',
    'CounterfactualCoordinatorActivityStats',
    'CounterfactualCoordinatorPoll',
    'CounterfactualCoordinatorStats',
    'CounterfactualCoordinatorSubmission',
    'recommended_counterfactual_worker_count',
]
