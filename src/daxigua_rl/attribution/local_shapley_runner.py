"""极稀疏局部 Shapley 的真实物理重演闭环。

任务以 ``CounterfactualProposal`` 保存的连续真实轨迹为唯一事实来源。runner
首先从最早快照重放 grand coalition，并逐步核对每个真实
``EngineActionOutcome``；任一步不一致都会拒绝整个任务。通过门禁后，subset 中的
候选动作按“在 coalition 中用真实动作、不在时用固定 comparison 动作”重演，
非候选轨迹步始终使用真实动作。

模块中的公开任务、结果和 runner 都位于顶层且可 pickle，可直接交给 Windows
``spawn`` 的 ``ProcessPoolExecutor``。
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from operator import index

from daxigua.core.engine import HeadlessGame
from daxigua.core.state import EngineActionOutcome
from daxigua_rl.env import DaxiguaEnv, DaxiguaEnvConfig
from daxigua_rl.training.identity import TransitionKey

from .causal_replay import (
    CausalSample,
    CausalTransitionContext,
    graph_schema_fingerprint,
    stable_budget_key,
)
from .counterfactual import (
    FrozenTargetPolicyPayload,
    LocalShapleyConfig,
    estimate_local_shapley,
    paired_shapley_permutations,
)
from .counterfactual_proposal import (
    CounterfactualHistoryEntry,
    CounterfactualProposal,
)
from .counterfactual_runner import (
    _bootstrap_value,
    _graph_builder,
    _load_target_model,
)
from .schema import (
    AttributionEventKey,
    MergeValueKey,
)


LOCAL_SHAPLEY_TASK_SCHEMA_VERSION = 1
LOCAL_SHAPLEY_RESULT_STATUSES = ('completed', 'failed')


def _strict_int(name, value, *, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer')
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None and result > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return result


def _finite_float(name, value, *, minimum=None, maximum=None):
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{name} must be a real number') from exc
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None and result > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return result


def _non_empty_text(name, value):
    if not isinstance(value, str):
        raise TypeError(f'{name} must be str')
    result = value.strip()
    if not result:
        raise ValueError(f'{name} must not be empty')
    return result


def _budget_payload(budget_key):
    if isinstance(budget_key, AttributionEventKey):
        return (
            'event',
            budget_key.worker_id,
            budget_key.episode_id,
            budget_key.event_index,
        )
    if isinstance(budget_key, MergeValueKey):
        key = budget_key.transition_key
        return (
            'merge',
            key.worker_id,
            key.episode_id,
            key.step_index,
            budget_key.event_offset,
        )
    raise TypeError(
        'budget_key must be AttributionEventKey or MergeValueKey'
    )


def stable_local_shapley_task_id(
        *,
        proposal_id,
        budget_key,
        event_key,
        trace_entries,
        candidate_keys,
        comparison_action_offsets,
        target_policy_fingerprint,
        attribution_version,
        utility,
        confidence,
        delay,
        tracker_config_fingerprint,
        config_fingerprint,
        schema_version=LOCAL_SHAPLEY_TASK_SCHEMA_VERSION):
    """生成只依赖规范字段、跨 hash seed 稳定的局部 Shapley 任务 ID。"""

    trace_entries = tuple(trace_entries)
    candidate_keys = tuple(candidate_keys)
    comparisons = tuple(comparison_action_offsets)
    if any(
            not isinstance(entry, CounterfactualHistoryEntry)
            for entry in trace_entries):
        raise TypeError('trace_entries must contain history entries')
    if any(
            not isinstance(key, TransitionKey)
            for key in candidate_keys):
        raise TypeError('candidate_keys must contain TransitionKey values')
    if not isinstance(event_key, AttributionEventKey):
        raise TypeError('event_key must be AttributionEventKey')
    comparison_map = {}
    for key, action_offset in comparisons:
        if not isinstance(key, TransitionKey):
            raise TypeError(
                'comparison_action_offsets keys must be TransitionKey'
            )
        comparison_map[key] = _strict_int(
            'comparison action offset',
            action_offset,
            minimum=0,
            maximum=14,
        )
    payload = {
        'schema_version': _strict_int(
            'schema_version',
            schema_version,
            minimum=1,
        ),
        'proposal_id': _non_empty_text('proposal_id', proposal_id),
        'budget_key': _budget_payload(budget_key),
        'event_key': (
            event_key.worker_id,
            event_key.episode_id,
            event_key.event_index,
        ),
        'trace': tuple(
            {
                'transition_key': entry.transition_key.as_tuple(),
                'snapshot_checksum': entry.snapshot.checksum,
                'factual_outcome': hashlib.sha256(
                    pickle.dumps(
                        entry.factual_outcome,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                ).hexdigest(),
                'actual_action_offset': entry.actual_action_offset,
            }
            for entry in trace_entries
        ),
        'candidate_keys': tuple(
            key.as_tuple()
            for key in candidate_keys
        ),
        'comparison_actions': tuple(
            (
                key.as_tuple(),
                comparison_map[key],
            )
            for key in candidate_keys
        ),
        'target_policy_fingerprint': _non_empty_text(
            'target_policy_fingerprint',
            target_policy_fingerprint,
        ),
        'attribution_version': _non_empty_text(
            'attribution_version',
            attribution_version,
        ),
        'utility': _finite_float(
            'utility',
            utility,
            minimum=0.0,
        ),
        'confidence': _finite_float(
            'confidence',
            confidence,
            minimum=0.0,
            maximum=1.0,
        ),
        'delay': _strict_int('delay', delay, minimum=0),
        'tracker_config_fingerprint': _non_empty_text(
            'tracker_config_fingerprint',
            tracker_config_fingerprint,
        ),
        'config_fingerprint': _non_empty_text(
            'config_fingerprint',
            config_fingerprint,
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('ascii')
    return 'local-shapley-' + hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalShapleyTask:
    """一个连续真实轨迹上的 2～4 候选局部 Shapley 物理任务。"""

    task_id: str
    proposal_id: str
    budget_key: AttributionEventKey | MergeValueKey
    event_key: AttributionEventKey
    trace_entries: tuple[CounterfactualHistoryEntry, ...]
    candidate_keys: tuple[TransitionKey, ...]
    comparison_action_offsets: tuple[tuple[TransitionKey, int], ...]
    target_policy: FrozenTargetPolicyPayload
    config: LocalShapleyConfig
    utility: float
    confidence: float
    delay: int
    priority: float
    estimated_tokens: int
    created_real_step: int
    attribution_version: str
    tracker_config_fingerprint: str
    schema_version: int = LOCAL_SHAPLEY_TASK_SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(
            self,
            'task_id',
            _non_empty_text('task_id', self.task_id),
        )
        object.__setattr__(
            self,
            'proposal_id',
            _non_empty_text('proposal_id', self.proposal_id),
        )
        _budget_payload(self.budget_key)
        if not isinstance(self.event_key, AttributionEventKey):
            raise TypeError('event_key must be AttributionEventKey')
        if self.event_key.episode_key != self.budget_key.episode_key:
            raise ValueError(
                'event_key and budget_key must belong to one episode'
            )

        trace = tuple(self.trace_entries)
        if not 2 <= len(trace) <= 12:
            raise ValueError(
                'trace_entries must contain 2 to 12 transitions'
            )
        if any(
                not isinstance(entry, CounterfactualHistoryEntry)
                for entry in trace):
            raise TypeError(
                'trace_entries must contain CounterfactualHistoryEntry'
            )
        trace_keys = tuple(entry.transition_key for entry in trace)
        if trace_keys != tuple(sorted(trace_keys)):
            raise ValueError('trace_entries must be sorted')
        if len(set(trace_keys)) != len(trace_keys):
            raise ValueError('trace_entries must not repeat a transition')
        if any(
                right.step_index != left.step_index + 1
                for left, right in zip(trace_keys, trace_keys[1:])):
            raise ValueError(
                'trace_entries must contain every intermediate step'
            )
        if len({
                (key.worker_id, key.episode_id)
                for key in trace_keys}) != 1:
            raise ValueError('trace_entries must stay in one episode')
        if (
                trace_keys[0].worker_id,
                trace_keys[0].episode_id,
        ) != self.event_key.episode_key:
            raise ValueError(
                'trace and attribution event must share an episode'
            )
        object.__setattr__(self, 'trace_entries', trace)

        candidates = tuple(self.candidate_keys)
        if (
                len(candidates) < 2
                or len(candidates) > 4
                or len(candidates) > self.config.candidate_limit):
            raise ValueError('candidate_keys must contain 2 to 4 values')
        if len(set(candidates)) != len(candidates):
            raise ValueError('candidate_keys must not contain duplicates')
        if tuple(sorted(candidates)) != candidates:
            raise ValueError('candidate_keys must be sorted')
        trace_key_set = set(trace_keys)
        if any(
                not isinstance(key, TransitionKey)
                or key not in trace_key_set
                for key in candidates):
            raise ValueError(
                'every candidate key must belong to the trace'
            )
        object.__setattr__(self, 'candidate_keys', candidates)

        comparisons = tuple(self.comparison_action_offsets)
        if len(comparisons) != len(candidates):
            raise ValueError(
                'comparison_action_offsets must cover every candidate'
            )
        comparison_keys = tuple(key for key, _offset in comparisons)
        if comparison_keys != candidates:
            raise ValueError(
                'comparison_action_offsets must follow candidate order'
            )
        entry_by_key = {
            entry.transition_key: entry
            for entry in trace
        }
        normalized_comparisons = []
        for key, action_offset in comparisons:
            entry = entry_by_key[key]
            action_offset = _strict_int(
                'comparison action offset',
                action_offset,
                minimum=0,
                maximum=entry.context.graph.action_count - 1,
            )
            if action_offset == entry.actual_action_offset:
                raise ValueError(
                    'comparison action must differ from actual action'
                )
            if action_offset not in entry.alternative_action_offsets:
                raise ValueError(
                    'comparison action must come from history alternatives'
                )
            normalized_comparisons.append((key, action_offset))
        object.__setattr__(
            self,
            'comparison_action_offsets',
            tuple(normalized_comparisons),
        )

        if not isinstance(self.target_policy, FrozenTargetPolicyPayload):
            raise TypeError(
                'target_policy must be FrozenTargetPolicyPayload'
            )
        if not isinstance(self.config, LocalShapleyConfig):
            raise TypeError('config must be LocalShapleyConfig')
        utility = _finite_float(
            'utility',
            self.utility,
            minimum=0.0,
        )
        object.__setattr__(self, 'utility', utility)
        confidence = _finite_float(
            'confidence',
            self.confidence,
            minimum=0.0,
            maximum=1.0,
        )
        if confidence <= 0.0:
            raise ValueError('confidence must be positive')
        object.__setattr__(self, 'confidence', confidence)
        object.__setattr__(
            self,
            'delay',
            _strict_int('delay', self.delay, minimum=0),
        )
        object.__setattr__(
            self,
            'priority',
            _finite_float('priority', self.priority, minimum=0.0),
        )
        expected_tokens = (2 ** len(candidates)) * len(trace)
        estimated_tokens = _strict_int(
            'estimated_tokens',
            self.estimated_tokens,
            minimum=1,
        )
        if estimated_tokens != expected_tokens:
            raise ValueError(
                'estimated_tokens must reserve every unique subset trace'
            )
        object.__setattr__(
            self,
            'estimated_tokens',
            estimated_tokens,
        )
        object.__setattr__(
            self,
            'created_real_step',
            _strict_int(
                'created_real_step',
                self.created_real_step,
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            'attribution_version',
            _non_empty_text(
                'attribution_version',
                self.attribution_version,
            ),
        )
        object.__setattr__(
            self,
            'tracker_config_fingerprint',
            _non_empty_text(
                'tracker_config_fingerprint',
                self.tracker_config_fingerprint,
            ),
        )
        schema_version = _strict_int(
            'schema_version',
            self.schema_version,
            minimum=1,
        )
        if schema_version != LOCAL_SHAPLEY_TASK_SCHEMA_VERSION:
            raise ValueError('unsupported LocalShapleyTask schema')
        object.__setattr__(self, 'schema_version', schema_version)

        expected_id = stable_local_shapley_task_id(
            proposal_id=self.proposal_id,
            budget_key=self.budget_key,
            event_key=self.event_key,
            trace_entries=self.trace_entries,
            candidate_keys=self.candidate_keys,
            comparison_action_offsets=(
                self.comparison_action_offsets
            ),
            target_policy_fingerprint=self.target_policy.fingerprint,
            attribution_version=self.attribution_version,
            utility=self.utility,
            confidence=self.confidence,
            delay=self.delay,
            tracker_config_fingerprint=(
                self.tracker_config_fingerprint
            ),
            config_fingerprint=self.config.fingerprint,
            schema_version=self.schema_version,
        )
        if self.task_id != expected_id:
            raise ValueError('task_id does not match task contents')

    @property
    def horizon(self):
        return len(self.trace_entries)

    @property
    def transition_key(self):
        return self.trace_entries[0].transition_key

    @property
    def snapshot(self):
        return self.trace_entries[0].snapshot

    @property
    def comparison_by_key(self):
        return dict(self.comparison_action_offsets)

    @property
    def entry_by_key(self):
        return {
            entry.transition_key: entry
            for entry in self.trace_entries
        }


def create_local_shapley_task(
        proposal,
        target_policy,
        *,
        config=None,
        created_real_step=0):
    """把已选 proposal 绑定到一份冻结 target policy。"""

    if not isinstance(proposal, CounterfactualProposal):
        raise TypeError('proposal must be CounterfactualProposal')
    if not isinstance(target_policy, FrozenTargetPolicyPayload):
        raise TypeError(
            'target_policy must be FrozenTargetPolicyPayload'
        )
    config = config or LocalShapleyConfig()
    if not isinstance(config, LocalShapleyConfig):
        raise TypeError('config must be LocalShapleyConfig')
    if not proposal.shapley_ready:
        raise ValueError('proposal does not contain a Shapley coalition')
    candidate_keys = tuple(
        proposal.coalition_candidate_keys[
            :config.candidate_limit
        ]
    )
    if len(candidate_keys) < config.minimum_candidates:
        raise ValueError('proposal has too few Shapley candidates')
    entries = {
        entry.transition_key: entry
        for entry in proposal.coalition_trace_entries
    }
    comparisons = tuple(
        (
            key,
            entries[key].alternative_action_offsets[0],
        )
        for key in candidate_keys
    )
    event = proposal.representative_event
    task_id = stable_local_shapley_task_id(
        proposal_id=proposal.proposal_id,
        budget_key=proposal.budget_key,
        event_key=event.event_id,
        trace_entries=proposal.coalition_trace_entries,
        candidate_keys=candidate_keys,
        comparison_action_offsets=comparisons,
        target_policy_fingerprint=target_policy.fingerprint,
        attribution_version=proposal.attribution_version,
        utility=proposal.utility,
        confidence=proposal.confidence,
        delay=proposal.delay,
        tracker_config_fingerprint=(
            event.tracker_config_fingerprint
        ),
        config_fingerprint=config.fingerprint,
    )
    return LocalShapleyTask(
        task_id=task_id,
        proposal_id=proposal.proposal_id,
        budget_key=proposal.budget_key,
        event_key=event.event_id,
        trace_entries=proposal.coalition_trace_entries,
        candidate_keys=candidate_keys,
        comparison_action_offsets=comparisons,
        target_policy=target_policy,
        config=config,
        utility=proposal.utility,
        confidence=proposal.confidence,
        delay=proposal.delay,
        priority=10.0 + proposal.utility * proposal.confidence,
        estimated_tokens=(
            (2 ** len(candidate_keys))
            * len(proposal.coalition_trace_entries)
        ),
        created_real_step=created_real_step,
        attribution_version=proposal.attribution_version,
        tracker_config_fingerprint=(
            event.tracker_config_fingerprint
        ),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalShapleySubsetResult:
    """一个唯一 coalition subset 的物理回报。"""

    member_keys: tuple[TransitionKey, ...]
    objective_return: float
    simulated_steps: int
    terminated: bool
    truncated: bool

    def __post_init__(self):
        members = tuple(self.member_keys)
        if any(not isinstance(key, TransitionKey) for key in members):
            raise TypeError('member_keys must contain TransitionKey')
        if tuple(sorted(members)) != members:
            raise ValueError('member_keys must be sorted')
        if len(set(members)) != len(members):
            raise ValueError('member_keys must not contain duplicates')
        object.__setattr__(self, 'member_keys', members)
        object.__setattr__(
            self,
            'objective_return',
            _finite_float('objective_return', self.objective_return),
        )
        object.__setattr__(
            self,
            'simulated_steps',
            _strict_int(
                'simulated_steps',
                self.simulated_steps,
                minimum=0,
            ),
        )
        object.__setattr__(self, 'terminated', bool(self.terminated))
        object.__setattr__(self, 'truncated', bool(self.truncated))


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalShapleyResult:
    """物理门禁、subset 估计和 token 消耗的完整不可变结果。"""

    task_id: str
    status: str
    grand_reproduced: bool
    contributions: tuple[tuple[TransitionKey, float], ...]
    empty_value: float | None
    full_value: float | None
    efficiency_residual: float | None
    efficiency_tolerance: float | None
    subset_results: tuple[LocalShapleySubsetResult, ...]
    evaluated_subset_count: int
    cache_hit_count: int
    permutation_count: int
    simulated_steps: int
    failure_reason: str | None = None
    diagnostic_codes: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(
            self,
            'task_id',
            _non_empty_text('task_id', self.task_id),
        )
        if self.status not in LOCAL_SHAPLEY_RESULT_STATUSES:
            raise ValueError(
                f'status must be one of {LOCAL_SHAPLEY_RESULT_STATUSES!r}'
            )
        object.__setattr__(
            self,
            'grand_reproduced',
            bool(self.grand_reproduced),
        )
        contributions = tuple(self.contributions)
        normalized = []
        for key, contribution in contributions:
            if not isinstance(key, TransitionKey):
                raise TypeError(
                    'contribution keys must be TransitionKey'
                )
            normalized.append((
                key,
                _finite_float('contribution', contribution),
            ))
        if len({key for key, _value in normalized}) != len(normalized):
            raise ValueError('contributions must not repeat candidates')
        object.__setattr__(
            self,
            'contributions',
            tuple(normalized),
        )
        for field_name in (
                'empty_value',
                'full_value',
                'efficiency_residual',
                'efficiency_tolerance'):
            value = getattr(self, field_name)
            if value is not None:
                value = _finite_float(field_name, value)
                if (
                        field_name == 'efficiency_tolerance'
                        and value < 0.0):
                    raise ValueError(
                        'efficiency_tolerance must be non-negative'
                    )
                object.__setattr__(self, field_name, value)
        subsets = tuple(self.subset_results)
        if any(
                not isinstance(item, LocalShapleySubsetResult)
                for item in subsets):
            raise TypeError(
                'subset_results must contain LocalShapleySubsetResult'
            )
        if len({
                item.member_keys
                for item in subsets}) != len(subsets):
            raise ValueError('subset_results must not repeat a subset')
        object.__setattr__(
            self,
            'subset_results',
            tuple(sorted(
                subsets,
                key=lambda item: (
                    len(item.member_keys),
                    item.member_keys,
                ),
            )),
        )
        for field_name in (
                'evaluated_subset_count',
                'cache_hit_count',
                'permutation_count',
                'simulated_steps'):
            object.__setattr__(
                self,
                field_name,
                _strict_int(
                    field_name,
                    getattr(self, field_name),
                    minimum=0,
                ),
            )
        if self.simulated_steps != sum(
                subset.simulated_steps
                for subset in self.subset_results):
            raise ValueError(
                'simulated_steps must equal subset physical work'
            )
        diagnostics = tuple(
            _non_empty_text('diagnostic code', code)
            for code in tuple(self.diagnostic_codes)
        )
        object.__setattr__(self, 'diagnostic_codes', diagnostics)
        if self.status == 'completed':
            if not self.grand_reproduced:
                raise ValueError(
                    'completed result requires grand reproduction'
                )
            if len(self.contributions) < 2:
                raise ValueError(
                    'completed result requires candidate contributions'
                )
            if self.failure_reason is not None:
                raise ValueError(
                    'completed result cannot have failure_reason'
                )
        else:
            object.__setattr__(
                self,
                'failure_reason',
                _non_empty_text(
                    'failure_reason',
                    self.failure_reason,
                ),
            )

    @property
    def label_ready(self):
        return (
            self.status == 'completed'
            and self.grand_reproduced
            and self.efficiency_residual is not None
            and self.efficiency_tolerance is not None
            and abs(self.efficiency_residual)
            <= self.efficiency_tolerance
        )


class _SubsetFailure(RuntimeError):
    def __init__(
            self,
            reason,
            *,
            simulated_steps,
            subset_result=None,
            diagnostic_codes=()):
        super().__init__(reason)
        self.reason = reason
        self.simulated_steps = simulated_steps
        self.subset_result = subset_result
        self.diagnostic_codes = tuple(diagnostic_codes)


def _env_config(task):
    payload = task.target_policy
    snapshot_config = task.snapshot.config
    return DaxiguaEnvConfig(
        action_count=15,
        physics_fps=snapshot_config.fps,
        max_physics_frames=payload.max_physics_frames,
        stable_frames=payload.stable_frames,
        space_iterations=snapshot_config.space_iterations,
        reward_config=payload.reward_config,
        state_analyzer_config=payload.state_analyzer_config,
    )


def _exception_code(exc):
    return f'exception_{type(exc).__name__}'


def _run_subset(
        task,
        members,
        *,
        model,
        graph_builder,
        require_factual_gate):
    """从最早快照运行一个 subset；返回回报与实际物理 token。"""

    members = frozenset(members)
    candidate_set = set(task.candidate_keys)
    comparisons = task.comparison_by_key
    try:
        env = DaxiguaEnv.from_snapshot(
            task.snapshot,
            config=_env_config(task),
        )
    except Exception as exc:
        raise _SubsetFailure(
            'snapshot_restore_failure',
            simulated_steps=0,
            diagnostic_codes=(_exception_code(exc),),
        ) from exc

    objective_return = 0.0
    discount = 1.0
    simulated_steps = 0
    terminated = False
    truncated = False
    try:
        for offset, entry in enumerate(task.trace_entries):
            if terminated or truncated:
                if require_factual_gate:
                    raise _SubsetFailure(
                        'grand_trace_ended_early',
                        simulated_steps=simulated_steps,
                    )
                break
            if offset:
                # proposal 中每个真实动作也从 canonical stable boundary 开始。
                env.game.capture_snapshot()
            key = entry.transition_key
            action_offset = entry.actual_action_offset
            if key in candidate_set and key not in members:
                action_offset = comparisons[key]
            attempted_steps = simulated_steps + 1
            try:
                _obs, reward, terminated, truncated, info = env.step(
                    action_offset,
                    transition_key=key,
                )
            except Exception as exc:
                raise _SubsetFailure(
                    'subset_physics_failure',
                    simulated_steps=attempted_steps,
                    diagnostic_codes=(_exception_code(exc),),
                ) from exc
            simulated_steps = attempted_steps
            objective_return += discount * float(reward)
            discount *= task.target_policy.gamma

            if require_factual_gate:
                outcome = info.get('engine_action_outcome')
                if not isinstance(outcome, EngineActionOutcome):
                    raise _SubsetFailure(
                        'grand_outcome_missing',
                        simulated_steps=simulated_steps,
                    )
                report = HeadlessGame.compare_action_outcomes(
                    entry.factual_outcome,
                    outcome,
                )
                if not report.matches:
                    raise _SubsetFailure(
                        'grand_reproduction_mismatch',
                        simulated_steps=simulated_steps,
                        diagnostic_codes=tuple(
                            f'grand_mismatch_{code}'
                            for code in report.mismatch_codes
                        ),
                    )

        if not terminated:
            bootstrap = _bootstrap_value(
                model,
                graph_builder,
                env,
            )
            objective_return += discount * bootstrap
        if not math.isfinite(objective_return):
            raise ValueError('subset objective return is non-finite')
    except _SubsetFailure:
        raise
    except Exception as exc:
        raise _SubsetFailure(
            'subset_policy_or_bootstrap_failure',
            simulated_steps=simulated_steps,
            diagnostic_codes=(_exception_code(exc),),
        ) from exc

    return LocalShapleySubsetResult(
        member_keys=tuple(sorted(members)),
        objective_return=objective_return,
        simulated_steps=simulated_steps,
        terminated=terminated,
        truncated=truncated,
    )


def _failed_result(
        task,
        *,
        reason,
        grand_reproduced,
        subset_results=(),
        diagnostic_codes=()):
    subsets = tuple(subset_results)
    return LocalShapleyResult(
        task_id=task.task_id,
        status='failed',
        grand_reproduced=grand_reproduced,
        contributions=(),
        empty_value=None,
        full_value=None,
        efficiency_residual=None,
        efficiency_tolerance=None,
        subset_results=subsets,
        evaluated_subset_count=len(subsets),
        cache_hit_count=0,
        permutation_count=0,
        simulated_steps=sum(item.simulated_steps for item in subsets),
        failure_reason=reason,
        diagnostic_codes=tuple(diagnostic_codes),
    )


def run_local_shapley_task(task):
    """执行 grand 门禁、唯一 subset 物理重演和配对排列估计。"""

    if not isinstance(task, LocalShapleyTask):
        raise TypeError('task must be LocalShapleyTask')
    try:
        model = _load_target_model(task.target_policy)
        graph_builder = _graph_builder(task.target_policy)
    except Exception as exc:
        return _failed_result(
            task,
            reason='target_policy_initialization_failure',
            grand_reproduced=False,
            diagnostic_codes=(_exception_code(exc),),
        )

    full = frozenset(task.candidate_keys)
    try:
        grand_result = _run_subset(
            task,
            full,
            model=model,
            graph_builder=graph_builder,
            require_factual_gate=True,
        )
    except _SubsetFailure as exc:
        # 即使失败发生在一步之后，也保守记录已经尝试的实际物理 token。
        synthetic = LocalShapleySubsetResult(
            member_keys=tuple(sorted(full)),
            objective_return=0.0,
            simulated_steps=exc.simulated_steps,
            terminated=False,
            truncated=False,
        )
        return _failed_result(
            task,
            reason=exc.reason,
            grand_reproduced=False,
            subset_results=(synthetic,),
            diagnostic_codes=exc.diagnostic_codes,
        )

    subset_cache = {
        full: grand_result.objective_return,
    }
    subset_results = {
        full: grand_result,
    }

    def evaluator(members):
        members = frozenset(members)
        try:
            subset = _run_subset(
                task,
                members,
                model=model,
                graph_builder=graph_builder,
                require_factual_gate=False,
            )
        except _SubsetFailure as exc:
            # 失败分支同样消耗真实物理预算；保留一个不可用于标签的账本项。
            subset_results[members] = LocalShapleySubsetResult(
                member_keys=tuple(sorted(members)),
                objective_return=0.0,
                simulated_steps=exc.simulated_steps,
                terminated=False,
                truncated=False,
            )
            raise RuntimeError(
                f'{exc.reason}:{"|".join(exc.diagnostic_codes)}'
            ) from exc
        subset_results[members] = subset
        return subset.objective_return

    try:
        permutation_pairs = paired_shapley_permutations(
            task.candidate_keys,
            pair_count=task.config.paired_permutations,
            seed_material=task.task_id,
        )
        estimate = estimate_local_shapley(
            task.candidate_keys,
            evaluator,
            permutation_pairs=permutation_pairs,
            subset_cache=subset_cache,
        )
    except Exception as exc:
        return _failed_result(
            task,
            reason='subset_evaluation_failure',
            grand_reproduced=True,
            subset_results=tuple(subset_results.values()),
            diagnostic_codes=(_exception_code(exc),),
        )

    residual = float(estimate.efficiency_residual)
    tolerance = max(
        1e-5,
        0.01 * abs(estimate.full_value - estimate.empty_value),
    )
    if (
            not math.isfinite(residual)
            or abs(residual) > tolerance):
        return LocalShapleyResult(
            task_id=task.task_id,
            status='failed',
            grand_reproduced=True,
            contributions=estimate.contributions,
            empty_value=estimate.empty_value,
            full_value=estimate.full_value,
            efficiency_residual=residual,
            efficiency_tolerance=tolerance,
            subset_results=tuple(subset_results.values()),
            evaluated_subset_count=len(subset_results),
            cache_hit_count=estimate.cache_hit_count,
            permutation_count=estimate.permutation_count,
            simulated_steps=sum(
                item.simulated_steps
                for item in subset_results.values()
            ),
            failure_reason='efficiency_residual_exceeded',
            diagnostic_codes=('efficiency_gate_failed',),
        )

    return LocalShapleyResult(
        task_id=task.task_id,
        status='completed',
        grand_reproduced=True,
        contributions=estimate.contributions,
        empty_value=estimate.empty_value,
        full_value=estimate.full_value,
        efficiency_residual=residual,
        efficiency_tolerance=tolerance,
        subset_results=tuple(subset_results.values()),
        evaluated_subset_count=len(subset_results),
        cache_hit_count=estimate.cache_hit_count,
        permutation_count=estimate.permutation_count,
        simulated_steps=sum(
            item.simulated_steps
            for item in subset_results.values()
        ),
        diagnostic_codes=('grand_trace_reproduced',),
    )


def local_shapley_result_to_causal_samples(
        task,
        result,
        *,
        zero_delta_epsilon=0.0):
    """把每个可信候选贡献转换为一条 ``CausalSample(shapley)``。"""

    if not isinstance(task, LocalShapleyTask):
        raise TypeError('task must be LocalShapleyTask')
    if not isinstance(result, LocalShapleyResult):
        raise TypeError('result must be LocalShapleyResult')
    if result.task_id != task.task_id:
        raise ValueError('result task_id does not match task')
    epsilon = _finite_float(
        'zero_delta_epsilon',
        zero_delta_epsilon,
        minimum=0.0,
    )
    if not result.label_ready:
        return ()
    contribution_map = dict(result.contributions)
    if set(contribution_map) != set(task.candidate_keys):
        raise ValueError(
            'result contributions do not match task candidates'
        )

    entries = task.entry_by_key
    comparisons = task.comparison_by_key
    budget_key = stable_budget_key(task.budget_key)
    samples = []
    expected_analyzer = (
        task.target_policy.state_analyzer_config.fingerprint
    )
    for key in task.candidate_keys:
        contribution = float(contribution_map[key])
        if not math.isfinite(contribution):
            raise ValueError('Shapley contribution must be finite')
        if abs(contribution) <= epsilon:
            continue
        entry = entries[key]
        context = entry.context
        if not isinstance(context, CausalTransitionContext):
            raise TypeError(
                'trace entry context must be CausalTransitionContext'
            )
        analyzer_fingerprint = (
            context.state_analysis.analyzer_config_fingerprint
        )
        if analyzer_fingerprint != expected_analyzer:
            raise ValueError(
                'context analyzer config differs from target policy'
            )
        graph_fingerprint = graph_schema_fingerprint(context.graph)
        if graph_fingerprint != context.graph_schema_fingerprint:
            raise ValueError('context graph schema changed')
        samples.append(CausalSample(
            graph=context.graph,
            actual_action_offset=entry.actual_action_offset,
            comparison_action_offset=comparisons[key],
            direction=1 if contribution > 0.0 else -1,
            target_margin=0.0,
            confidence=task.confidence,
            cause_type='LOCAL_SHAPLEY',
            delay=task.delay,
            transition_key=key,
            attribution_version=task.attribution_version,
            supervision_kind='shapley',
            stratum='counterfactual',
            event_key=(
                f'local-shapley-task-v1:{task.task_id}:'
                f'{key.worker_id}:{key.episode_id}:{key.step_index}'
            ),
            budget_key=budget_key,
            target_delta=contribution,
            policy_version=context.policy_version,
            tracker_config_fingerprint=(
                task.tracker_config_fingerprint
            ),
            analyzer_config_fingerprint=analyzer_fingerprint,
            graph_schema_fingerprint=graph_fingerprint,
        ))
    return tuple(samples)


__all__ = [
    'LOCAL_SHAPLEY_RESULT_STATUSES',
    'LOCAL_SHAPLEY_TASK_SCHEMA_VERSION',
    'LocalShapleyResult',
    'LocalShapleySubsetResult',
    'LocalShapleyTask',
    'create_local_shapley_task',
    'local_shapley_result_to_causal_samples',
    'run_local_shapley_task',
    'stable_local_shapley_task_id',
]
