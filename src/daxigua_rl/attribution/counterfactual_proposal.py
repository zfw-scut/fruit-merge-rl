"""worker-local 稀疏反事实 proposal 生成。

本模块只保存真实轨迹最近若干稳定动作边界，并把已确认归因事件压缩成不可变
``CounterfactualProposal``。它不冻结 target model、不执行物理分支，也不消费全局
反事实预算；这些职责属于主进程调度器和独立 runner。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from operator import index

from daxigua.core.state import EngineActionOutcome, EngineSnapshot
from daxigua_rl.reward import merge_utility
from daxigua_rl.training.identity import TransitionKey

from .causal_replay import (
    CausalTransitionContext,
    stable_budget_key,
    stable_event_key,
)
from .counterfactual import (
    COUNTERFACTUAL_TRIGGER_REASONS,
    counterfactual_trigger_reasons,
    engine_action_outcome_fingerprint,
    select_counterfactual_alternatives,
)
from .schema import (
    ANALYSIS_ACTION_COUNT,
    AttributionEvent,
    AttributionEventKey,
    Contributor,
    MergeLineageRecord,
    MergeValueKey,
)


COUNTERFACTUAL_PROPOSAL_SCHEMA_VERSION = 1
DEFAULT_RANDOM_AUDIT_MODULUS = 2048
COUNTERFACTUAL_TRANSFER_ALWAYS_REASONS = frozenset({
    'high_value_merge',
})
_MECHANICALLY_UNIQUE_EVENT_TYPES = {
    'DIRECT_TRIGGER',
    'MERGE_LINEAGE',
    'REACHABILITY_SEALED',
}


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


def _canonical_trigger_reasons(reasons):
    values = tuple(reasons)
    if any(not isinstance(reason, str) for reason in values):
        raise TypeError('trigger_reasons must contain strings')
    unknown = tuple(
        reason
        for reason in values
        if reason not in COUNTERFACTUAL_TRIGGER_REASONS
    )
    if unknown:
        raise ValueError(
            f'unsupported counterfactual trigger reasons: {unknown!r}'
        )
    if not values:
        raise ValueError('trigger_reasons must not be empty')
    if len(set(values)) != len(values):
        raise ValueError('trigger_reasons must not contain duplicates')
    selected = set(values)
    return tuple(
        reason
        for reason in COUNTERFACTUAL_TRIGGER_REASONS
        if reason in selected
    )


def stable_counterfactual_proposal_id(
        *,
        budget_key,
        representative_event,
        contributor,
        context,
        snapshot,
        factual_outcome,
        alternative_action_offsets,
        trigger_reasons,
        coalition_trace_entries=(),
        coalition_candidate_keys=(),
        schema_version=COUNTERFACTUAL_PROPOSAL_SCHEMA_VERSION):
    """为 proposal 的完整执行来源生成跨 hash seed 稳定的身份。"""

    if not isinstance(representative_event, AttributionEvent):
        raise TypeError('representative_event must be AttributionEvent')
    if not isinstance(contributor, Contributor):
        raise TypeError('contributor must be Contributor')
    if not isinstance(context, CausalTransitionContext):
        raise TypeError('context must be CausalTransitionContext')
    if not isinstance(snapshot, EngineSnapshot):
        raise TypeError('snapshot must be EngineSnapshot')
    if not isinstance(factual_outcome, EngineActionOutcome):
        raise TypeError('factual_outcome must be EngineActionOutcome')
    schema_version = _strict_int(
        'schema_version',
        schema_version,
        minimum=1,
    )
    alternatives = tuple(
        _strict_int(
            f'alternative_action_offsets[{offset}]',
            action_offset,
            minimum=0,
            maximum=ANALYSIS_ACTION_COUNT - 1,
        )
        for offset, action_offset
        in enumerate(tuple(alternative_action_offsets))
    )
    reasons = _canonical_trigger_reasons(trigger_reasons)
    trace_entries = tuple(coalition_trace_entries)
    if trace_entries:
        if any(
                not isinstance(entry, CounterfactualHistoryEntry)
                for entry in trace_entries):
            raise TypeError(
                'coalition_trace_entries must contain history entries'
            )
        trace_payload = tuple(
            {
                'transition_key': entry.transition_key.as_tuple(),
                'snapshot_checksum': entry.snapshot.checksum,
                'factual_outcome_fingerprint': (
                    engine_action_outcome_fingerprint(
                        entry.factual_outcome
                    )
                ),
                'actual_action_offset': entry.actual_action_offset,
                'alternative_action_offsets': (
                    entry.alternative_action_offsets
                ),
            }
            for entry in trace_entries
        )
    else:
        trace_payload = ({
            'transition_key': context.transition_key.as_tuple(),
            'snapshot_checksum': snapshot.checksum,
            'factual_outcome_fingerprint': (
                engine_action_outcome_fingerprint(factual_outcome)
            ),
            'actual_action_offset': context.actual_action_offset,
            'alternative_action_offsets': alternatives,
        },)
    candidate_keys = tuple(coalition_candidate_keys)
    if candidate_keys:
        if any(
                not isinstance(key, TransitionKey)
                for key in candidate_keys):
            raise TypeError(
                'coalition_candidate_keys must contain TransitionKey values'
            )
    else:
        candidate_keys = (context.transition_key,)
    payload = {
        'schema_version': schema_version,
        'budget_key': stable_budget_key(budget_key),
        'event_key': stable_event_key(
            representative_event.event_id
        ),
        'attribution_version': (
            representative_event.attribution_version
        ),
        'contributor': {
            'transition_key': contributor.transition_key.as_tuple(),
            'action_offset': contributor.action_offset,
            'action_index': contributor.action_index,
            'fruit_id': contributor.fruit_id,
            'role': contributor.role,
        },
        'context_graph_schema': context.graph_schema_fingerprint,
        'context_policy_version': context.policy_version,
        'snapshot_checksum': snapshot.checksum,
        'factual_outcome_fingerprint': (
            engine_action_outcome_fingerprint(factual_outcome)
        ),
        'alternative_action_offsets': alternatives,
        'trigger_reasons': reasons,
        'coalition_trace': trace_payload,
        'coalition_candidate_keys': tuple(
            key.as_tuple()
            for key in candidate_keys
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('ascii')
    return 'cf-proposal-' + hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualHistoryEntry:
    """一个动作前稳定边界及其真实相邻动作结果。"""

    transition_key: TransitionKey
    context: CausalTransitionContext
    snapshot: EngineSnapshot
    factual_outcome: EngineActionOutcome
    alternative_action_offsets: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        if not isinstance(self.context, CausalTransitionContext):
            raise TypeError('context must be CausalTransitionContext')
        if self.context.transition_key != self.transition_key:
            raise ValueError(
                'context transition key must match history entry'
            )
        if not isinstance(self.snapshot, EngineSnapshot):
            raise TypeError('snapshot must be EngineSnapshot')
        if not self.snapshot.checksum_valid:
            raise ValueError('snapshot checksum is invalid')
        if (
                self.snapshot.episode.step_count
                != self.transition_key.step_index):
            raise ValueError(
                'snapshot step_count must match transition key'
            )
        if not isinstance(self.factual_outcome, EngineActionOutcome):
            raise TypeError(
                'factual_outcome must be EngineActionOutcome'
            )
        if (
                tuple(self.factual_outcome.drop_result.queue_before)
                != tuple(self.snapshot.episode.fruit_queue)):
            raise ValueError(
                'factual queue_before must match snapshot fruit queue'
            )
        drop_result = self.factual_outcome.drop_result
        if drop_result.dropped_level != drop_result.queue_before[0]:
            raise ValueError(
                'factual dropped level must match queue_before q0'
            )
        if drop_result.fruit_id != self.snapshot.episode.next_fruit_id:
            raise ValueError(
                'factual dropped fruit id must match snapshot next id'
            )
        if (
                tuple(drop_result.queue_after[:-1])
                != tuple(drop_result.queue_before[1:])):
            raise ValueError(
                'factual queue_after must shift queue_before'
            )
        if (
                tuple(drop_result.queue_after)
                != tuple(self.factual_outcome.final_state.fruit_queue)):
            raise ValueError(
                'factual queue_after must match final state queue'
            )
        expected_drop_x = (
            self.context.state_analysis.action_drop_x_by_offset[
                self.context.actual_action_offset
            ]
        )
        if not math.isclose(
                drop_result.drop_x,
                expected_drop_x,
                rel_tol=0.0,
                abs_tol=1e-9):
            raise ValueError(
                'factual drop_x must match context actual action'
            )
        if (
                self.factual_outcome.final_state.step_count
                != self.transition_key.step_index + 1):
            raise ValueError(
                'factual outcome must be adjacent to snapshot'
            )
        if (
                self.factual_outcome.physics_result.score_delta
                != (
                    self.factual_outcome.final_state.score
                    - self.snapshot.episode.score
                )):
            raise ValueError(
                'factual score delta must connect snapshot and final state'
            )
        if (
                self.factual_outcome.physics_result.done
                != self.factual_outcome.final_state.done):
            raise ValueError(
                'factual physics done must match final state'
            )
        if (
                self.factual_outcome.final_state.physics_frame
                != (
                    self.snapshot.episode.physics_frame
                    + self.factual_outcome.physics_result.frames_simulated
                )):
            raise ValueError(
                'factual physics frames must connect snapshot and final state'
            )
        if self.factual_outcome.fail_count < self.snapshot.episode.fail_count:
            raise ValueError(
                'factual fail_count must not precede snapshot fail_count'
            )
        if (
                self.factual_outcome.next_fruit_id
                <= drop_result.fruit_id):
            raise ValueError(
                'factual next_fruit_id must follow dropped fruit id'
            )
        alternatives = tuple(
            _strict_int(
                f'alternative_action_offsets[{offset}]',
                action_offset,
                minimum=0,
                maximum=self.context.graph.action_count - 1,
            )
            for offset, action_offset
            in enumerate(tuple(self.alternative_action_offsets))
        )
        if len(alternatives) > 3:
            raise ValueError(
                'history alternative actions must contain at most 3 items'
            )
        if len(set(alternatives)) != len(alternatives):
            raise ValueError(
                'history alternative actions must not contain duplicates'
            )
        if self.context.actual_action_offset in alternatives:
            raise ValueError(
                'history alternatives must differ from actual action'
            )
        object.__setattr__(
            self,
            'alternative_action_offsets',
            alternatives,
        )

    @property
    def actual_action_offset(self):
        return self.context.actual_action_offset


class CounterfactualHistoryRing:
    """按 TransitionKey 保存最近稳定边界、context 和真实 outcome。"""

    def __init__(self, capacity=32):
        self.capacity = _strict_int(
            'capacity',
            capacity,
            minimum=1,
        )
        self._items = OrderedDict()
        self._eviction_count = 0

    def __len__(self):
        return len(self._items)

    @property
    def keys(self):
        return tuple(self._items.keys())

    @property
    def eviction_count(self):
        return self._eviction_count

    def put(self, entry):
        if not isinstance(entry, CounterfactualHistoryEntry):
            raise TypeError(
                'entry must be CounterfactualHistoryEntry'
            )
        key = entry.transition_key
        if key in self._items:
            del self._items[key]
        self._items[key] = entry
        if len(self._items) <= self.capacity:
            return None
        self._eviction_count += 1
        return self._items.popitem(last=False)[1]

    def remember(
            self,
            *,
            context,
            snapshot,
            factual_outcome,
            alternative_action_offsets=()):
        return self.put(CounterfactualHistoryEntry(
            transition_key=context.transition_key,
            context=context,
            snapshot=snapshot,
            factual_outcome=factual_outcome,
            alternative_action_offsets=(
                alternative_action_offsets
            ),
        ))

    def get(self, transition_key, default=None):
        if not isinstance(transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        return self._items.get(transition_key, default)

    def discard_episode(self, worker_id, episode_id):
        worker_id = _strict_int('worker_id', worker_id, minimum=0)
        episode_id = _strict_int('episode_id', episode_id, minimum=0)
        keys = tuple(
            key
            for key in self._items
            if (
                key.worker_id == worker_id
                and key.episode_id == episode_id
            )
        )
        for key in keys:
            del self._items[key]
        return len(keys)

    def clear(self):
        count = len(self._items)
        self._items.clear()
        return count


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualProposal:
    """worker 交给主进程调度器的完整、但尚未绑定 target policy 的候选。"""

    proposal_id: str
    representative_event: AttributionEvent
    budget_key: AttributionEventKey | MergeValueKey
    contributor: Contributor
    context: CausalTransitionContext
    snapshot: EngineSnapshot
    factual_outcome: EngineActionOutcome
    actual_action_offset: int
    alternative_action_offsets: tuple[int, ...]
    trigger_reasons: tuple[str, ...]
    coalition_trace_entries: tuple[
        CounterfactualHistoryEntry, ...
    ]
    coalition_candidate_keys: tuple[TransitionKey, ...]
    utility: float
    confidence: float
    delay: int
    schema_version: int = COUNTERFACTUAL_PROPOSAL_SCHEMA_VERSION

    def __post_init__(self):
        proposal_id = _non_empty_text(
            'proposal_id',
            self.proposal_id,
        )
        object.__setattr__(self, 'proposal_id', proposal_id)
        if not isinstance(
                self.representative_event,
                AttributionEvent):
            raise TypeError(
                'representative_event must be AttributionEvent'
            )
        event = self.representative_event
        if event.status != 'confirmed':
            raise ValueError(
                'representative_event must be confirmed'
            )
        if self.budget_key != event.budget_key:
            raise ValueError(
                'budget_key must match representative event'
            )
        # 同时执行类型和稳定序列化校验。
        stable_budget_key(self.budget_key)
        if not isinstance(self.contributor, Contributor):
            raise TypeError('contributor must be Contributor')
        if self.contributor not in event.contributors:
            raise ValueError(
                'contributor must belong to representative event'
            )
        if not isinstance(self.context, CausalTransitionContext):
            raise TypeError('context must be CausalTransitionContext')
        if self.context.transition_key != self.contributor.transition_key:
            raise ValueError(
                'context must correspond to contributor transition'
            )
        if (
                self.context.actual_action_offset
                != self.contributor.action_offset
                or self.context.actual_action_index
                != self.contributor.action_index):
            raise ValueError(
                'context action mapping must match contributor'
            )
        # 复用 history entry 的完整快照、队列和相邻动作校验。
        CounterfactualHistoryEntry(
            transition_key=self.context.transition_key,
            context=self.context,
            snapshot=self.snapshot,
            factual_outcome=self.factual_outcome,
        )

        actual = _strict_int(
            'actual_action_offset',
            self.actual_action_offset,
            minimum=0,
            maximum=self.context.graph.action_count - 1,
        )
        if actual != self.context.actual_action_offset:
            raise ValueError(
                'actual_action_offset must match context'
            )
        object.__setattr__(self, 'actual_action_offset', actual)
        alternatives = tuple(
            _strict_int(
                f'alternative_action_offsets[{offset}]',
                action_offset,
                minimum=0,
                maximum=self.context.graph.action_count - 1,
            )
            for offset, action_offset
            in enumerate(tuple(self.alternative_action_offsets))
        )
        if not alternatives:
            raise ValueError(
                'alternative_action_offsets must not be empty'
            )
        if len(alternatives) > 3:
            raise ValueError(
                'alternative_action_offsets must contain at most 3 actions'
            )
        if len(set(alternatives)) != len(alternatives):
            raise ValueError(
                'alternative_action_offsets must not contain duplicates'
            )
        if actual in alternatives:
            raise ValueError(
                'alternative actions must differ from actual action'
            )
        object.__setattr__(
            self,
            'alternative_action_offsets',
            alternatives,
        )
        reasons = _canonical_trigger_reasons(self.trigger_reasons)
        object.__setattr__(self, 'trigger_reasons', reasons)

        trace_entries = tuple(self.coalition_trace_entries)
        if not trace_entries:
            raise ValueError(
                'coalition_trace_entries must not be empty'
            )
        if len(trace_entries) > 12:
            raise ValueError(
                'coalition trace must contain at most 12 transitions'
            )
        if any(
                not isinstance(entry, CounterfactualHistoryEntry)
                for entry in trace_entries):
            raise TypeError(
                'coalition_trace_entries must contain history entries'
            )
        trace_keys = tuple(
            entry.transition_key
            for entry in trace_entries
        )
        if trace_keys != tuple(sorted(trace_keys)):
            raise ValueError(
                'coalition trace entries must be sorted by transition key'
            )
        if len(set(trace_keys)) != len(trace_keys):
            raise ValueError(
                'coalition trace entries must not contain duplicates'
            )
        episode_keys = {
            (key.worker_id, key.episode_id)
            for key in trace_keys
        }
        if len(episode_keys) != 1:
            raise ValueError(
                'coalition trace must stay within one episode'
            )
        if any(
                right.step_index != left.step_index + 1
                for left, right in zip(trace_keys, trace_keys[1:])):
            raise ValueError(
                'coalition trace must contain every intermediate step'
            )
        representative_entry = next(
            (
                entry
                for entry in trace_entries
                if entry.transition_key
                == self.context.transition_key
            ),
            None,
        )
        if representative_entry is None:
            raise ValueError(
                'coalition trace must include representative transition'
            )
        if (
                representative_entry.snapshot.checksum
                != self.snapshot.checksum
                or engine_action_outcome_fingerprint(
                    representative_entry.factual_outcome
                )
                != engine_action_outcome_fingerprint(
                    self.factual_outcome
                )
                or representative_entry.actual_action_offset != actual
                or representative_entry.alternative_action_offsets
                != alternatives):
            raise ValueError(
                'representative coalition entry must match proposal source'
            )
        object.__setattr__(
            self,
            'coalition_trace_entries',
            trace_entries,
        )

        candidate_keys = tuple(self.coalition_candidate_keys)
        if any(
                not isinstance(key, TransitionKey)
                for key in candidate_keys):
            raise TypeError(
                'coalition_candidate_keys must contain TransitionKey values'
            )
        if not 1 <= len(candidate_keys) <= 4:
            raise ValueError(
                'coalition_candidate_keys must contain 1 to 4 items'
            )
        if candidate_keys != tuple(sorted(candidate_keys)):
            raise ValueError(
                'coalition candidate keys must be sorted'
            )
        if len(set(candidate_keys)) != len(candidate_keys):
            raise ValueError(
                'coalition candidate keys must not contain duplicates'
            )
        if self.context.transition_key not in candidate_keys:
            raise ValueError(
                'representative transition must be a coalition candidate'
            )
        if any(key not in trace_keys for key in candidate_keys):
            raise ValueError(
                'coalition candidate keys must belong to trace'
            )
        if (
                trace_keys[0] != candidate_keys[0]
                or trace_keys[-1] != candidate_keys[-1]):
            raise ValueError(
                'coalition trace endpoints must be candidate transitions'
            )
        entries_by_key = {
            entry.transition_key: entry
            for entry in trace_entries
        }
        if any(
                not entries_by_key[key].alternative_action_offsets
                for key in candidate_keys):
            raise ValueError(
                'every coalition candidate needs an alternative action'
            )
        object.__setattr__(
            self,
            'coalition_candidate_keys',
            candidate_keys,
        )

        utility = _finite_float(
            'utility',
            self.utility,
            minimum=0.0,
        )
        if utility + 1e-12 < event.utility:
            raise ValueError(
                'proposal utility must cover representative event utility'
            )
        object.__setattr__(self, 'utility', utility)
        confidence = _finite_float(
            'confidence',
            self.confidence,
            minimum=0.0,
            maximum=1.0,
        )
        expected_confidence = min(
            event.link_confidence,
            event.placement_confidence,
        )
        if not math.isclose(
                confidence,
                expected_confidence,
                rel_tol=0.0,
                abs_tol=1e-12):
            raise ValueError(
                'confidence must match representative event evidence'
            )
        if confidence <= 0.0:
            raise ValueError('confidence must be > 0')
        object.__setattr__(self, 'confidence', confidence)
        delay = _strict_int('delay', self.delay, minimum=0)
        if event.delay is None or delay != event.delay:
            raise ValueError(
                'delay must match confirmed representative event'
            )
        object.__setattr__(self, 'delay', delay)
        schema_version = _strict_int(
            'schema_version',
            self.schema_version,
            minimum=1,
        )
        if schema_version != COUNTERFACTUAL_PROPOSAL_SCHEMA_VERSION:
            raise ValueError(
                'unsupported counterfactual proposal schema version'
            )
        object.__setattr__(self, 'schema_version', schema_version)

        expected_id = stable_counterfactual_proposal_id(
            budget_key=self.budget_key,
            representative_event=event,
            contributor=self.contributor,
            context=self.context,
            snapshot=self.snapshot,
            factual_outcome=self.factual_outcome,
            alternative_action_offsets=(
                self.alternative_action_offsets
            ),
            trigger_reasons=self.trigger_reasons,
            coalition_trace_entries=(
                self.coalition_trace_entries
            ),
            coalition_candidate_keys=(
                self.coalition_candidate_keys
            ),
            schema_version=self.schema_version,
        )
        if self.proposal_id != expected_id:
            raise ValueError(
                'proposal_id does not match stable proposal payload'
            )

    @property
    def transition_key(self):
        return self.context.transition_key

    @property
    def attribution_version(self):
        return self.representative_event.attribution_version

    @property
    def coalition_candidate_entries(self):
        selected = set(self.coalition_candidate_keys)
        return tuple(
            entry
            for entry in self.coalition_trace_entries
            if entry.transition_key in selected
        )

    @property
    def shapley_ready(self):
        return len(self.coalition_candidate_keys) >= 2


def should_transfer_counterfactual_proposal(
        proposal,
        *,
        sample_rate=1.0):
    """稳定筛选需要跨进程传输的物理反事实候选。

    完整规则归因和 worker-local 历史不受此门控影响。极少见的高价值合成
    始终进入物理验证；包括连锁与正负规则冲突在内的常规候选按
    ``proposal_id`` 的 SHA-256 做确定性抽样，确保相同轨迹、相同配置在不同
    进程调度下得到同一选择。
    """

    if not isinstance(proposal, CounterfactualProposal):
        raise TypeError('proposal must be CounterfactualProposal')
    rate = _finite_float(
        'sample_rate',
        sample_rate,
        minimum=0.0,
        maximum=1.0,
    )
    if (
            COUNTERFACTUAL_TRANSFER_ALWAYS_REASONS
            .intersection(proposal.trigger_reasons)):
        return True
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.sha256(
        proposal.proposal_id.encode('utf-8')
    ).digest()
    bucket = int.from_bytes(digest[:8], 'big')
    return bucket < int(rate * (1 << 64))


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualProposalBuildStats:
    input_event_count: int
    confirmed_event_count: int
    budget_count: int
    generated_proposal_count: int
    reason_counts: tuple[tuple[str, int], ...]

    def reason_count(self, reason):
        return dict(self.reason_counts).get(str(reason), 0)


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualProposalBuildResult:
    proposals: tuple[CounterfactualProposal, ...]
    stats: CounterfactualProposalBuildStats


@dataclass(frozen=True, slots=True)
class _EvidenceCandidate:
    event: AttributionEvent
    contributor: Contributor
    confidence: float


class CounterfactualProposalBuilder:
    """把 confirmed 事件按预算合并，并只输出文档白名单中的稀疏候选。"""

    def __init__(
            self,
            *,
            ring_size=32,
            random_audit_modulus=DEFAULT_RANDOM_AUDIT_MODULUS):
        self.history = CounterfactualHistoryRing(ring_size)
        self.random_audit_modulus = _strict_int(
            'random_audit_modulus',
            random_audit_modulus,
            minimum=1,
        )
        self._emitted_budgets = set()
        # tracker 的价值包、合成记录和 confirmed 事件不保证在同一个
        # observe_transition() 返回。按 episode 留存小型元数据，才能在延迟确认
        # 时恢复 high-value/chain 触发，并让分步到达的正负证据参与同一判断。
        self._confirmed_events_by_episode = defaultdict(dict)
        self._merge_records_by_episode = defaultdict(dict)
        self._cumulative_reasons = Counter()
        self._build_calls = 0
        self._generated_count = 0

    def remember(self, *, context, snapshot, factual_outcome):
        return self.history.remember(
            context=context,
            snapshot=snapshot,
            factual_outcome=factual_outcome,
            alternative_action_offsets=(
                self._select_alternatives(context)
            ),
        )

    def discard_episode(self, worker_id, episode_id):
        discarded = self.history.discard_episode(
            worker_id,
            episode_id,
        )
        self._emitted_budgets = {
            item
            for item in self._emitted_budgets
            if item[:2] != (int(worker_id), int(episode_id))
        }
        episode_key = int(worker_id), int(episode_id)
        self._confirmed_events_by_episode.pop(episode_key, None)
        self._merge_records_by_episode.pop(episode_key, None)
        return discarded

    def clear(self):
        discarded = self.history.clear()
        self._emitted_budgets.clear()
        self._confirmed_events_by_episode.clear()
        self._merge_records_by_episode.clear()
        return discarded

    @property
    def stats(self):
        return {
            'build_calls': self._build_calls,
            'generated_proposal_count': self._generated_count,
            'history_size': len(self.history),
            'history_eviction_count': self.history.eviction_count,
            'reason_counts': dict(sorted(
                self._cumulative_reasons.items()
            )),
        }

    def build(self, events, *, merge_records=()):
        return self.build_with_stats(
            events,
            merge_records=merge_records,
        ).proposals

    def build_with_stats(self, events, *, merge_records=()):
        events = tuple(events)
        merge_records = tuple(merge_records)
        if any(not isinstance(event, AttributionEvent) for event in events):
            raise TypeError('events must contain AttributionEvent values')
        if any(
                not isinstance(record, MergeLineageRecord)
                for record in merge_records):
            raise TypeError(
                'merge_records must contain MergeLineageRecord values'
            )

        reasons = Counter()
        confirmed = []
        touched_episodes = set()
        for record in merge_records:
            episode_key = record.value_key.episode_key
            touched_episodes.add(episode_key)
            records = self._merge_records_by_episode[episode_key]
            previous = records.get(record.value_key)
            if previous is not None and previous != record:
                raise RuntimeError(
                    'merge record identity was reused with different data'
                )
            records[record.value_key] = record
        for event in events:
            if event.status != 'confirmed':
                reasons['event_not_confirmed'] += 1
                continue
            confirmed.append(event)
            touched_episodes.add(event.episode_key)
            cached_events = self._confirmed_events_by_episode[
                event.episode_key
            ]
            previous = cached_events.get(event.event_id)
            if previous is not None and previous != event:
                raise RuntimeError(
                    'confirmed event identity was reused with different data'
                )
            cached_events[event.event_id] = event

        groups = defaultdict(list)
        for episode_key in touched_episodes:
            for event in self._confirmed_events_by_episode[
                    episode_key].values():
                groups[(
                    episode_key,
                    stable_budget_key(event.budget_key),
                )].append(event)

        signs_by_action = defaultdict(set)
        for cached_events in (
                self._confirmed_events_by_episode.values()):
            for event in cached_events.values():
                for contributor in event.contributors:
                    signs_by_action[(
                        contributor.transition_key,
                        contributor.action_offset,
                    )].add(event.sign)
        conflict_actions = {
            key
            for key, signs in signs_by_action.items()
            if len(signs) > 1
        }

        proposals = []
        for episode_key, budget_string in sorted(groups):
            group = tuple(groups[(episode_key, budget_string)])
            proposal = self._build_group(
                budget_string,
                group,
                tuple(
                    self._merge_records_by_episode[
                        episode_key
                    ].values()
                ),
                conflict_actions,
                reasons,
            )
            if proposal is not None:
                proposals.append(proposal)

        stats = CounterfactualProposalBuildStats(
            input_event_count=len(events),
            confirmed_event_count=len(confirmed),
            budget_count=len(groups),
            generated_proposal_count=len(proposals),
            reason_counts=tuple(sorted(reasons.items())),
        )
        self._build_calls += 1
        self._generated_count += len(proposals)
        self._cumulative_reasons.update(reasons)
        return CounterfactualProposalBuildResult(
            proposals=tuple(proposals),
            stats=stats,
        )

    def _build_group(
            self,
            budget_string,
            events,
            merge_records,
            conflict_actions,
            reasons):
        budget_key = events[0].budget_key
        if any(event.budget_key != budget_key for event in events):
            raise RuntimeError(
                'stable budget key collision across typed budget values'
            )
        episode_key = events[0].episode_key
        emitted_key = (
            episode_key[0],
            episode_key[1],
            budget_string,
        )
        if emitted_key in self._emitted_budgets:
            reasons['duplicate_budget'] += 1
            return None

        candidates = []
        for event in events:
            confidence = min(
                event.link_confidence,
                event.placement_confidence,
            )
            for contributor in event.contributors:
                candidates.append(_EvidenceCandidate(
                    event=event,
                    contributor=contributor,
                    confidence=confidence,
                ))
        if not candidates:
            reasons['no_contributors'] += 1
            return None
        candidate = max(
            candidates,
            key=lambda item: (
                item.contributor.contribution_weight * item.confidence,
                item.confidence,
                item.contributor.contribution_weight,
                item.event.utility,
                stable_event_key(item.event.event_id),
                item.contributor.transition_key,
                item.contributor.action_offset,
                item.contributor.fruit_id,
            ),
        )

        history = self.history.get(candidate.contributor.transition_key)
        if history is None:
            reasons['missing_history'] += 1
            return None
        if (
                history.context.actual_action_offset
                != candidate.contributor.action_offset
                or history.context.actual_action_index
                != candidate.contributor.action_index):
            reasons['history_action_mismatch'] += 1
            return None

        relevant_records = tuple(
            record
            for record in merge_records
            if (
                record.value_key == budget_key
                or any(
                    record.value_key
                    in event.evidence.value_package_keys
                    for event in events
                )
            )
        )
        new_level = max(
            (record.new_level for record in relevant_records),
            default=None,
        )
        chain_depth = max(
            (record.chain_depth for record in relevant_records),
            default=0,
        )
        negative_causes = {
            (
                contributor.transition_key,
                contributor.action_offset,
            )
            for event in events
            if event.sign < 0
            for contributor in event.contributors
        }
        conflict = (
            candidate.contributor.transition_key,
            candidate.contributor.action_offset,
        ) in conflict_actions
        random_audit = self._is_random_audit(
            budget_string,
            candidate,
        )
        middle_placement_confidence = next(
            (
                event.placement_confidence
                for event in events
                if 0.55 <= event.placement_confidence < 0.80
            ),
            candidate.event.placement_confidence,
        )
        trigger_reasons = counterfactual_trigger_reasons(
            new_level=new_level,
            chain_depth=chain_depth,
            possible_blocker_causes=len(negative_causes),
            conflicting_signals=conflict,
            placement_confidence=middle_placement_confidence,
            random_rule_audit=random_audit,
        )
        if not trigger_reasons:
            reasons['no_trigger'] += 1
            return None

        alternatives = history.alternative_action_offsets
        if not alternatives:
            reasons['no_reliable_alternative'] += 1
            return None

        utility = max(
            (
                *(event.utility for event in events),
                *(record.utility for record in relevant_records),
            ),
            default=0.0,
        )
        coalition_trace, coalition_candidate_keys = (
            self._coalition_trace(
                representative=candidate,
                candidates=candidates,
                new_level=new_level,
                utility=utility,
                reasons=reasons,
            )
        )
        proposal_id = stable_counterfactual_proposal_id(
            budget_key=budget_key,
            representative_event=candidate.event,
            contributor=candidate.contributor,
            context=history.context,
            snapshot=history.snapshot,
            factual_outcome=history.factual_outcome,
            alternative_action_offsets=alternatives,
            trigger_reasons=trigger_reasons,
            coalition_trace_entries=coalition_trace,
            coalition_candidate_keys=coalition_candidate_keys,
        )
        proposal = CounterfactualProposal(
            proposal_id=proposal_id,
            representative_event=candidate.event,
            budget_key=budget_key,
            contributor=candidate.contributor,
            context=history.context,
            snapshot=history.snapshot,
            factual_outcome=history.factual_outcome,
            actual_action_offset=(
                history.context.actual_action_offset
            ),
            alternative_action_offsets=alternatives,
            trigger_reasons=trigger_reasons,
            coalition_trace_entries=coalition_trace,
            coalition_candidate_keys=coalition_candidate_keys,
            utility=utility,
            confidence=candidate.confidence,
            delay=candidate.event.delay,
        )
        self._emitted_budgets.add(emitted_key)
        return proposal

    def _coalition_trace(
            self,
            *,
            representative,
            candidates,
            new_level,
            utility,
            reasons):
        representative_key = (
            representative.contributor.transition_key
        )
        representative_entry = self.history.get(representative_key)
        ordinary = (representative_entry,), (representative_key,)
        high_value = (
            (new_level is not None and new_level >= 7)
            or utility >= merge_utility(7)
        )
        if not high_value:
            return ordinary

        best_by_key = {}
        for candidate in candidates:
            key = candidate.contributor.transition_key
            entry = self.history.get(key)
            if entry is None or not entry.alternative_action_offsets:
                continue
            if (
                    entry.context.actual_action_offset
                    != candidate.contributor.action_offset
                    or entry.context.actual_action_index
                    != candidate.contributor.action_index):
                continue
            score = (
                candidate.contributor.contribution_weight
                * candidate.confidence,
                candidate.confidence,
                candidate.contributor.contribution_weight,
                stable_event_key(candidate.event.event_id),
                candidate.contributor.fruit_id,
            )
            current = best_by_key.get(key)
            if current is None or score > current[0]:
                best_by_key[key] = score, entry

        ranked_keys = tuple(
            key
            for key, _value in sorted(
                best_by_key.items(),
                key=lambda item: (
                    item[1][0],
                    item[0],
                ),
                reverse=True,
            )
        )
        selected = [representative_key]
        for key in ranked_keys:
            if key in selected or len(selected) >= 4:
                continue
            trial = tuple(sorted((*selected, key)))
            if (
                    trial[0].worker_id
                    != trial[-1].worker_id
                    or trial[0].episode_id
                    != trial[-1].episode_id
                    or trial[-1].step_index
                    - trial[0].step_index
                    >= 12):
                continue
            selected.append(key)
        candidate_keys = tuple(sorted(selected))
        if len(candidate_keys) < 2:
            reasons['coalition_insufficient_candidates'] += 1
            return ordinary

        trace_entries = []
        for step_index in range(
                candidate_keys[0].step_index,
                candidate_keys[-1].step_index + 1):
            key = TransitionKey(
                worker_id=candidate_keys[0].worker_id,
                episode_id=candidate_keys[0].episode_id,
                step_index=step_index,
            )
            entry = self.history.get(key)
            if entry is None:
                reasons['coalition_incomplete_trace'] += 1
                return ordinary
            trace_entries.append(entry)
        if len(trace_entries) > 12:
            reasons['coalition_trace_too_long'] += 1
            return ordinary
        return tuple(trace_entries), candidate_keys

    def _is_random_audit(self, budget_string, candidate):
        # 唯一且 A 级的机械触发/割点已有高置信规则证据，默认不浪费审计预算。
        if (
                candidate.event.confidence_tier == 'A'
                and len(candidate.event.contributors) == 1
                and candidate.event.event_type
                in _MECHANICALLY_UNIQUE_EVENT_TYPES):
            return False
        payload = (
            f'{budget_string}|'
            f'{stable_event_key(candidate.event.event_id)}|'
            f'{candidate.contributor.transition_key.as_tuple()}'
        ).encode('ascii')
        value = int.from_bytes(
            hashlib.sha256(payload).digest()[:8],
            byteorder='big',
            signed=False,
        )
        return value % self.random_audit_modulus == 0

    @classmethod
    def _select_alternatives(cls, context):
        safety_scores = cls._action_safety_scores(context)
        safest_alternative = max(
            (
                action_offset
                for action_offset in range(len(safety_scores))
                if action_offset != context.actual_action_offset
            ),
            key=lambda action_offset: (
                safety_scores[action_offset],
                -action_offset,
            ),
            default=None,
        )
        # 只公开唯一“最安全替代”，避免它与镜像重复时误退到次安全动作。
        safest_only_scores = tuple(
            (
                safety_scores[action_offset]
                if action_offset == safest_alternative
                else None
            )
            for action_offset in range(len(safety_scores))
        )
        return select_counterfactual_alternatives(
            actual_action_offset=context.actual_action_offset,
            safest_action_scores=safest_only_scores,
            action_count=context.graph.action_count,
            max_alternatives=3,
        )

    @staticmethod
    def _action_safety_scores(context):
        analysis = context.state_analysis
        weights = tuple(
            analysis.queue_decay ** lane.queue_index
            for lane in analysis.queue_lane_analyses
        )
        denominator = sum(weights)
        return tuple(
            sum(
                weight * (
                    0.7 * lane.landing_depths_by_action[action_offset]
                    + 0.3 * bool(
                        lane.safe_action_mask & (1 << action_offset)
                    )
                )
                for weight, lane in zip(
                    weights,
                    analysis.queue_lane_analyses,
                )
            ) / denominator
            for action_offset in range(analysis.action_count)
        )


__all__ = [
    'COUNTERFACTUAL_PROPOSAL_SCHEMA_VERSION',
    'COUNTERFACTUAL_TRANSFER_ALWAYS_REASONS',
    'DEFAULT_RANDOM_AUDIT_MODULUS',
    'CounterfactualHistoryEntry',
    'CounterfactualHistoryRing',
    'CounterfactualProposal',
    'CounterfactualProposalBuildResult',
    'CounterfactualProposalBuildStats',
    'CounterfactualProposalBuilder',
    'should_transfer_counterfactual_proposal',
    'stable_counterfactual_proposal_id',
]
