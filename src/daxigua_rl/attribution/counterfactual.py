"""预算受控的稀疏反事实与局部 Shapley 调度契约。

本模块只负责训练侧的纯数据契约、候选选择、预算和异步调度，不直接 import 或创建
``HeadlessGame``。真实物理分支由调用方注入的模块顶层 runner 执行：

.. code-block:: python

    def replay_counterfactual(task: CounterfactualTask) -> CounterfactualResult:
        ...

runner 和本文所有任务/结果类型都可被 Windows ``spawn`` pickle。这样 EngineSnapshot
恢复与物理重演实现可以独立演进，而 rollout 主线程只做非阻塞 ``submit()`` /
``poll()``，不会等待反事实分支。
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import pickle
import random
import threading
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from operator import index
from typing import Callable

from daxigua.core.state import EngineActionOutcome, EngineSnapshot
from daxigua_rl.graph.schema import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
from daxigua_rl.reward import RewardConfig, merge_utility
from daxigua_rl.training.identity import TransitionKey

from .state_analyzer import StateAnalyzerConfig
from .schema import (
    ANALYSIS_ACTION_COUNT,
    AttributionEvent,
    AttributionEventKey,
    Contributor,
    MergeValueKey,
)


COUNTERFACTUAL_TRIGGER_REASONS = (
    'high_value_merge',
    'multi_stage_chain',
    'ambiguous_blocking',
    'conflicting_signals',
    'middle_placement_confidence',
    'random_rule_audit',
)
COUNTERFACTUAL_BRANCH_STATUSES = (
    'completed',
    'partial',
    'failed',
)
COUNTERFACTUAL_RESULT_STATUSES = (
    'completed',
    'partial',
    'failed',
)
FROZEN_TARGET_POLICY_SCHEMA_VERSION = 1


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


def _code_tuple(name, values, *, allowed=None, require_non_empty=False):
    result = tuple(
        _non_empty_text(f'{name}[{offset}]', value)
        for offset, value in enumerate(tuple(values))
    )
    if require_non_empty and not result:
        raise ValueError(f'{name} must not be empty')
    if len(set(result)) != len(result):
        raise ValueError(f'{name} must not contain duplicates')
    if allowed is not None:
        unknown = tuple(value for value in result if value not in allowed)
        if unknown:
            raise ValueError(
                f'{name} contains unsupported values: {unknown!r}'
            )
    return result


def _fingerprint_dataclass(value):
    payload = {
        field_name: getattr(value, field_name)
        for field_name in value.__dataclass_fields__
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(',', ':'),
        sort_keys=True,
    ).encode('ascii')
    return hashlib.sha256(encoded).hexdigest()[:16]


def _sha256_bytes(value):
    """返回 bytes 的完整 SHA-256；任务 ID 只引用摘要，不复制大模型内容。"""

    if not isinstance(value, bytes):
        raise TypeError('value must be bytes')
    return hashlib.sha256(value).hexdigest()


def engine_action_outcome_fingerprint(outcome):
    """为真实动作结果生成稳定摘要，避免把完整轨迹再次混入任务 ID。

    ``EngineActionOutcome`` 只包含冻结 dataclass、tuple 和基础数值。使用明确 pickle
    协议后，同一 Python/依赖环境中的主进程与 Windows spawn worker 会得到相同摘要；
    runner 仍会对结果本体做逐字段、带几何容差的物理复现比较，摘要不是复现门禁的
    替代品。
    """

    if not isinstance(outcome, EngineActionOutcome):
        raise TypeError('outcome must be EngineActionOutcome')
    payload = pickle.dumps(
        outcome,
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return _sha256_bytes(payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenGNNModelConfig:
    """可在独立进程中精确重建 ``GNNQNetwork`` 的结构配置。"""

    node_feature_dim: int = len(NODE_FEATURE_NAMES)
    edge_feature_dim: int = len(EDGE_FEATURE_NAMES)
    hidden_dim: int = 128
    message_layers: int = 3
    activation: str = 'silu'
    dropout: float = 0.0

    def __post_init__(self):
        for field_name in (
                'node_feature_dim',
                'edge_feature_dim',
                'hidden_dim',
                'message_layers'):
            object.__setattr__(
                self,
                field_name,
                _strict_int(
                    field_name,
                    getattr(self, field_name),
                    minimum=1,
                ),
            )
        activation = _non_empty_text('activation', self.activation)
        if activation not in {'relu', 'silu'}:
            raise ValueError('activation must be relu or silu')
        object.__setattr__(self, 'activation', activation)
        dropout = _finite_float(
            'dropout',
            self.dropout,
            minimum=0.0,
            maximum=1.0,
        )
        if dropout >= 1.0:
            raise ValueError('dropout must be < 1')
        object.__setattr__(self, 'dropout', dropout)

    @property
    def model_kwargs(self):
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenGraphBuilderConfig:
    """冻结策略构图所需的少量归一化配置。

    训练目前使用默认 ``GraphBuilder``，但把这三个值显式冻结可防止未来调整归一化
    参数后，历史反事实任务在另一种图口径上运行。
    """

    velocity_scale: float = 2000.0
    fruit_count_scale: float = 64.0
    connect_global_node: bool = True

    def __post_init__(self):
        for field_name in ('velocity_scale', 'fruit_count_scale'):
            value = _finite_float(
                field_name,
                getattr(self, field_name),
                minimum=0.0,
            )
            if value <= 0.0:
                raise ValueError(f'{field_name} must be > 0')
            object.__setattr__(self, field_name, value)
        if not isinstance(self.connect_global_node, bool):
            raise TypeError('connect_global_node must be bool')

    @property
    def builder_kwargs(self):
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


def _target_policy_fingerprint_payload(
        *,
        schema_version,
        policy_version,
        model_config,
        graph_builder_config,
        state_dict_sha256,
        gamma,
        max_physics_frames,
        stable_frames,
        reward_config,
        state_analyzer_config):
    """生成不含模型 bytes 的规范 JSON 字段。"""

    return {
        'schema_version': schema_version,
        'policy_version': policy_version,
        'model_config': asdict(model_config),
        'graph_builder_config': asdict(graph_builder_config),
        'state_dict_sha256': state_dict_sha256,
        'gamma': gamma,
        'max_physics_frames': max_physics_frames,
        'stable_frames': stable_frames,
        'reward_config': asdict(reward_config),
        'state_analyzer_config': asdict(state_analyzer_config),
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class FrozenTargetPolicyPayload:
    """反事实所有分支共享的冻结 target policy。

    ``state_dict_bytes`` 必须由 CPU tensor 生成；runner 会在使用前再次严格加载、
    检查有限性和模型键集合。``payload_fingerprint`` 只哈希模型摘要与小配置，不会
    把可能数 MB 的 state_dict bytes 拼入 task ID。
    """

    policy_version: str
    model_config: FrozenGNNModelConfig
    graph_builder_config: FrozenGraphBuilderConfig
    state_dict_bytes: bytes
    state_dict_sha256: str
    gamma: float
    max_physics_frames: int
    stable_frames: int
    reward_config: RewardConfig
    state_analyzer_config: StateAnalyzerConfig
    payload_fingerprint: str
    schema_version: int = FROZEN_TARGET_POLICY_SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(
            self,
            'policy_version',
            _non_empty_text('policy_version', self.policy_version),
        )
        if not isinstance(self.model_config, FrozenGNNModelConfig):
            raise TypeError(
                'model_config must be FrozenGNNModelConfig'
            )
        if not isinstance(
                self.graph_builder_config,
                FrozenGraphBuilderConfig):
            raise TypeError(
                'graph_builder_config must be FrozenGraphBuilderConfig'
            )
        if (
                not isinstance(self.state_dict_bytes, bytes)
                or not self.state_dict_bytes):
            raise ValueError(
                'state_dict_bytes must be non-empty bytes'
            )
        state_digest = _non_empty_text(
            'state_dict_sha256',
            self.state_dict_sha256,
        )
        if len(state_digest) != 64 or any(
                character not in '0123456789abcdef'
                for character in state_digest):
            raise ValueError(
                'state_dict_sha256 must be a lowercase SHA-256 hex digest'
            )
        if state_digest != _sha256_bytes(self.state_dict_bytes):
            raise ValueError('state_dict_sha256 does not match bytes')
        object.__setattr__(
            self,
            'state_dict_sha256',
            state_digest,
        )
        gamma = _finite_float(
            'gamma',
            self.gamma,
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(self, 'gamma', gamma)
        for field_name in ('max_physics_frames', 'stable_frames'):
            object.__setattr__(
                self,
                field_name,
                _strict_int(
                    field_name,
                    getattr(self, field_name),
                    minimum=1,
                ),
            )
        if self.stable_frames > self.max_physics_frames:
            raise ValueError(
                'stable_frames must not exceed max_physics_frames'
            )
        if not isinstance(self.reward_config, RewardConfig):
            raise TypeError('reward_config must be RewardConfig')
        if self.reward_config.gamma != self.gamma:
            raise ValueError(
                'reward_config.gamma must exactly match payload gamma'
            )
        if not isinstance(
                self.state_analyzer_config,
                StateAnalyzerConfig):
            raise TypeError(
                'state_analyzer_config must be StateAnalyzerConfig'
            )
        schema_version = _strict_int(
            'schema_version',
            self.schema_version,
            minimum=1,
        )
        if schema_version != FROZEN_TARGET_POLICY_SCHEMA_VERSION:
            raise ValueError(
                'unsupported frozen target policy schema_version'
            )
        object.__setattr__(
            self,
            'schema_version',
            schema_version,
        )

        supplied_fingerprint = _non_empty_text(
            'payload_fingerprint',
            self.payload_fingerprint,
        )
        expected_fingerprint = self.expected_fingerprint()
        if supplied_fingerprint != expected_fingerprint:
            raise ValueError(
                'payload_fingerprint does not match frozen policy payload'
            )
        object.__setattr__(
            self,
            'payload_fingerprint',
            supplied_fingerprint,
        )

    @classmethod
    def create(
            cls,
            *,
            policy_version,
            model_config,
            state_dict_bytes,
            gamma,
            max_physics_frames,
            stable_frames,
            reward_config,
            state_analyzer_config,
            graph_builder_config=None):
        """从已经 CPU 序列化的 state_dict bytes 构造并封印 payload。"""

        if not isinstance(model_config, FrozenGNNModelConfig):
            raise TypeError(
                'model_config must be FrozenGNNModelConfig'
            )
        graph_builder_config = (
            graph_builder_config
            or FrozenGraphBuilderConfig()
        )
        if not isinstance(
                graph_builder_config,
                FrozenGraphBuilderConfig):
            raise TypeError(
                'graph_builder_config must be FrozenGraphBuilderConfig'
            )
        if not isinstance(reward_config, RewardConfig):
            raise TypeError('reward_config must be RewardConfig')
        if not isinstance(
                state_analyzer_config,
                StateAnalyzerConfig):
            raise TypeError(
                'state_analyzer_config must be StateAnalyzerConfig'
            )
        state_dict_sha256 = _sha256_bytes(state_dict_bytes)
        normalized_policy_version = _non_empty_text(
            'policy_version',
            policy_version,
        )
        normalized_gamma = _finite_float(
            'gamma',
            gamma,
            minimum=0.0,
            maximum=1.0,
        )
        normalized_max_frames = _strict_int(
            'max_physics_frames',
            max_physics_frames,
            minimum=1,
        )
        normalized_stable_frames = _strict_int(
            'stable_frames',
            stable_frames,
            minimum=1,
        )
        payload = _target_policy_fingerprint_payload(
            schema_version=FROZEN_TARGET_POLICY_SCHEMA_VERSION,
            policy_version=normalized_policy_version,
            model_config=model_config,
            graph_builder_config=graph_builder_config,
            state_dict_sha256=state_dict_sha256,
            gamma=normalized_gamma,
            max_physics_frames=normalized_max_frames,
            stable_frames=normalized_stable_frames,
            reward_config=reward_config,
            state_analyzer_config=state_analyzer_config,
        )
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('ascii')
        fingerprint = (
            'target-policy-v1:'
            + hashlib.sha256(encoded).hexdigest()
        )
        return cls(
            policy_version=normalized_policy_version,
            model_config=model_config,
            graph_builder_config=graph_builder_config,
            state_dict_bytes=state_dict_bytes,
            state_dict_sha256=state_dict_sha256,
            gamma=normalized_gamma,
            max_physics_frames=normalized_max_frames,
            stable_frames=normalized_stable_frames,
            reward_config=reward_config,
            state_analyzer_config=state_analyzer_config,
            payload_fingerprint=fingerprint,
        )

    def expected_fingerprint(self):
        payload = _target_policy_fingerprint_payload(
            schema_version=self.schema_version,
            policy_version=self.policy_version,
            model_config=self.model_config,
            graph_builder_config=self.graph_builder_config,
            state_dict_sha256=self.state_dict_sha256,
            gamma=self.gamma,
            max_physics_frames=self.max_physics_frames,
            stable_frames=self.stable_frames,
            reward_config=self.reward_config,
            state_analyzer_config=self.state_analyzer_config,
        )
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('ascii')
        return (
            'target-policy-v1:'
            + hashlib.sha256(encoded).hexdigest()
        )

    @property
    def fingerprint(self):
        return self.payload_fingerprint


def _budget_key_payload(budget_key):
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


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualConfig:
    """V1 反事实调度的固定预算与可靠性配置。"""

    counterfactual_version: str = 'counterfactual_v1'
    horizon: int = 10
    horizon_min: int = 8
    horizon_max: int = 12
    cost_ratio: float = 0.08
    cost_hard_limit: float = 0.10
    min_real_steps: int = 256
    cpu_core_ratio: float = 0.25
    queue_capacity: int = 256
    snapshot_ring_size: int = 32
    max_alternatives: int = 3
    max_inflight_per_worker: int = 2
    soft_budget_borrow_priority: float = 10.0
    circuit_breaker_failures: int = 5

    def __post_init__(self):
        object.__setattr__(
            self,
            'counterfactual_version',
            _non_empty_text(
                'counterfactual_version',
                self.counterfactual_version,
            ),
        )
        for field_name in (
                'horizon',
                'horizon_min',
                'horizon_max',
                'min_real_steps',
                'queue_capacity',
                'snapshot_ring_size',
                'max_alternatives',
                'max_inflight_per_worker',
                'circuit_breaker_failures'):
            object.__setattr__(
                self,
                field_name,
                _strict_int(
                    field_name,
                    getattr(self, field_name),
                    minimum=1,
                ),
            )
        if self.horizon_min > self.horizon_max:
            raise ValueError('horizon_min must be <= horizon_max')
        if not self.horizon_min <= self.horizon <= self.horizon_max:
            raise ValueError(
                'horizon must be between horizon_min and horizon_max'
            )
        if self.max_alternatives > 3:
            raise ValueError('max_alternatives must be <= 3')
        if self.max_inflight_per_worker > 2:
            raise ValueError(
                'max_inflight_per_worker must be <= 2'
            )
        for field_name in (
                'cost_ratio',
                'cost_hard_limit',
                'cpu_core_ratio'):
            object.__setattr__(
                self,
                field_name,
                _finite_float(
                    field_name,
                    getattr(self, field_name),
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        if self.cost_ratio <= 0.0:
            raise ValueError('cost_ratio must be > 0')
        if self.cost_hard_limit < self.cost_ratio:
            raise ValueError(
                'cost_hard_limit must be >= cost_ratio'
            )
        if self.cpu_core_ratio <= 0.0:
            raise ValueError('cpu_core_ratio must be > 0')
        object.__setattr__(
            self,
            'soft_budget_borrow_priority',
            _finite_float(
                'soft_budget_borrow_priority',
                self.soft_budget_borrow_priority,
                minimum=0.0,
            ),
        )

    @property
    def fingerprint(self):
        return _fingerprint_dataclass(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalShapleyConfig:
    """极稀疏局部 Shapley 的累计配额。"""

    event_ratio_max: float = 0.0005
    candidate_limit: int = 4
    paired_permutations: int = 4
    minimum_candidates: int = 2
    minimum_utility: float = merge_utility(7)

    def __post_init__(self):
        object.__setattr__(
            self,
            'event_ratio_max',
            _finite_float(
                'event_ratio_max',
                self.event_ratio_max,
                minimum=0.0,
                maximum=1.0,
            ),
        )
        for field_name in (
                'candidate_limit',
                'paired_permutations',
                'minimum_candidates'):
            object.__setattr__(
                self,
                field_name,
                _strict_int(
                    field_name,
                    getattr(self, field_name),
                    minimum=1,
                ),
            )
        if self.minimum_candidates < 2:
            raise ValueError('minimum_candidates must be >= 2')
        if self.candidate_limit > 4:
            raise ValueError('candidate_limit must be <= 4')
        if self.minimum_candidates > self.candidate_limit:
            raise ValueError(
                'minimum_candidates must be <= candidate_limit'
            )
        object.__setattr__(
            self,
            'minimum_utility',
            _finite_float(
                'minimum_utility',
                self.minimum_utility,
                minimum=0.0,
            ),
        )

    @property
    def fingerprint(self):
        return _fingerprint_dataclass(self)


def stable_counterfactual_task_id(
        *,
        budget_key,
        transition_key,
        snapshot_checksum,
        factual_outcome_fingerprint,
        target_policy_fingerprint,
        actual_action_offset,
        alternative_action_offsets,
        trigger_reasons,
        attribution_version,
        config_fingerprint):
    """用显式 JSON 字段生成跨进程、跨 hash seed 稳定的任务 ID。

    真实 outcome 和 target state_dict 都可能很大。ID 只引用各自已经校验的 SHA-256
    指纹，因此任务身份能覆盖完整执行语义，又不会因 JSON 拼接复制大块 bytes。
    """

    if not isinstance(transition_key, TransitionKey):
        raise TypeError('transition_key must be TransitionKey')
    payload = {
        'budget_key': _budget_key_payload(budget_key),
        'transition_key': transition_key.as_tuple(),
        'snapshot_checksum': _non_empty_text(
            'snapshot_checksum',
            snapshot_checksum,
        ),
        'factual_outcome_fingerprint': _non_empty_text(
            'factual_outcome_fingerprint',
            factual_outcome_fingerprint,
        ),
        'target_policy_fingerprint': _non_empty_text(
            'target_policy_fingerprint',
            target_policy_fingerprint,
        ),
        'actual_action_offset': _strict_int(
            'actual_action_offset',
            actual_action_offset,
            minimum=0,
            maximum=ANALYSIS_ACTION_COUNT - 1,
        ),
        'alternative_action_offsets': tuple(
            _strict_int(
                f'alternative_action_offsets[{offset}]',
                action_offset,
                minimum=0,
                maximum=ANALYSIS_ACTION_COUNT - 1,
            )
            for offset, action_offset
            in enumerate(tuple(alternative_action_offsets))
        ),
        'trigger_reasons': tuple(trigger_reasons),
        'attribution_version': _non_empty_text(
            'attribution_version',
            attribution_version,
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
    return 'cf-' + hashlib.sha256(encoded).hexdigest()[:24]


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualTask:
    """一个不阻塞 rollout 的快照分支重演任务。"""

    task_id: str
    budget_key: AttributionEventKey | MergeValueKey
    transition_key: TransitionKey
    snapshot: EngineSnapshot
    factual_outcome: EngineActionOutcome
    factual_outcome_fingerprint: str
    target_policy: FrozenTargetPolicyPayload
    actual_action_offset: int
    alternative_action_offsets: tuple[int, ...]
    trigger_reasons: tuple[str, ...]
    priority: float
    estimated_tokens: int
    horizon: int
    created_real_step: int
    attribution_version: str
    scheduler_config_fingerprint: str
    label_confidence: float = 1.0
    attribution_delay: int = 0

    def __post_init__(self):
        object.__setattr__(
            self,
            'task_id',
            _non_empty_text('task_id', self.task_id),
        )
        _budget_key_payload(self.budget_key)
        if not isinstance(self.transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        if self.budget_key.episode_key != (
                self.transition_key.worker_id,
                self.transition_key.episode_id):
            raise ValueError(
                'budget_key and transition_key must belong to the same '
                'episode'
            )
        if not isinstance(self.snapshot, EngineSnapshot):
            raise TypeError('snapshot must be EngineSnapshot')
        if not self.snapshot.checksum_valid:
            raise ValueError('snapshot checksum is invalid')
        if self.snapshot.episode.step_count != self.transition_key.step_index:
            raise ValueError(
                'snapshot step_count must match transition_key'
            )
        if not isinstance(self.factual_outcome, EngineActionOutcome):
            raise TypeError(
                'factual_outcome must be EngineActionOutcome'
            )
        factual_fingerprint = _non_empty_text(
            'factual_outcome_fingerprint',
            self.factual_outcome_fingerprint,
        )
        expected_factual_fingerprint = (
            engine_action_outcome_fingerprint(self.factual_outcome)
        )
        if factual_fingerprint != expected_factual_fingerprint:
            raise ValueError(
                'factual_outcome_fingerprint does not match outcome'
            )
        object.__setattr__(
            self,
            'factual_outcome_fingerprint',
            factual_fingerprint,
        )
        if not isinstance(
                self.target_policy,
                FrozenTargetPolicyPayload):
            raise TypeError(
                'target_policy must be FrozenTargetPolicyPayload'
            )
        if (
                tuple(self.factual_outcome.drop_result.queue_before)
                != tuple(self.snapshot.episode.fruit_queue)):
            raise ValueError(
                'factual outcome queue_before must match snapshot'
            )
        if (
                self.factual_outcome.final_state.step_count
                != self.transition_key.step_index + 1):
            raise ValueError(
                'factual outcome must describe the adjacent action'
            )

        actual = _strict_int(
            'actual_action_offset',
            self.actual_action_offset,
            minimum=0,
            maximum=ANALYSIS_ACTION_COUNT - 1,
        )
        object.__setattr__(self, 'actual_action_offset', actual)
        alternatives = tuple(
            _strict_int(
                f'alternative_action_offsets[{offset}]',
                action_offset,
                minimum=0,
                maximum=ANALYSIS_ACTION_COUNT - 1,
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
        reasons = _code_tuple(
            'trigger_reasons',
            self.trigger_reasons,
            allowed=COUNTERFACTUAL_TRIGGER_REASONS,
            require_non_empty=True,
        )
        object.__setattr__(self, 'trigger_reasons', reasons)
        object.__setattr__(
            self,
            'priority',
            _finite_float('priority', self.priority, minimum=0.0),
        )
        object.__setattr__(
            self,
            'estimated_tokens',
            _strict_int(
                'estimated_tokens',
                self.estimated_tokens,
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            'horizon',
            _strict_int('horizon', self.horizon, minimum=1),
        )
        expected_tokens = self.horizon * (
            1 + len(self.alternative_action_offsets)
        )
        if self.estimated_tokens != expected_tokens:
            raise ValueError(
                'estimated_tokens must equal horizon times original plus '
                'alternative branch count'
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
        for field_name in (
                'attribution_version',
                'scheduler_config_fingerprint'):
            object.__setattr__(
                self,
                field_name,
                _non_empty_text(
                    field_name,
                    getattr(self, field_name),
                ),
            )
        label_confidence = _finite_float(
            'label_confidence',
            self.label_confidence,
            minimum=0.0,
            maximum=1.0,
        )
        if label_confidence <= 0.0:
            raise ValueError('label_confidence must be > 0')
        object.__setattr__(
            self,
            'label_confidence',
            label_confidence,
        )
        object.__setattr__(
            self,
            'attribution_delay',
            _strict_int(
                'attribution_delay',
                self.attribution_delay,
                minimum=0,
            ),
        )
        expected_id = stable_counterfactual_task_id(
            budget_key=self.budget_key,
            transition_key=self.transition_key,
            snapshot_checksum=self.snapshot.checksum,
            factual_outcome_fingerprint=(
                self.factual_outcome_fingerprint
            ),
            target_policy_fingerprint=(
                self.target_policy.fingerprint
            ),
            actual_action_offset=self.actual_action_offset,
            alternative_action_offsets=(
                self.alternative_action_offsets
            ),
            trigger_reasons=self.trigger_reasons,
            attribution_version=self.attribution_version,
            config_fingerprint=(
                self.scheduler_config_fingerprint
            ),
        )
        if self.task_id != expected_id:
            raise ValueError(
                'task_id does not match stable task payload'
            )

    @property
    def requested_action_offsets(self):
        return (
            self.actual_action_offset,
            *self.alternative_action_offsets,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualBranchResult:
    """一个实际动作或替代动作分支的纯数据结果。"""

    action_offset: int
    status: str
    objective_return: float | None
    simulated_steps: int
    terminated: bool = False
    truncated: bool = False
    early_stopped: bool = False
    failure_reason: str | None = None
    diagnostic_codes: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(
            self,
            'action_offset',
            _strict_int(
                'action_offset',
                self.action_offset,
                minimum=0,
                maximum=ANALYSIS_ACTION_COUNT - 1,
            ),
        )
        if self.status not in COUNTERFACTUAL_BRANCH_STATUSES:
            raise ValueError(
                'status must be completed, partial, or failed'
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
        for field_name in (
                'terminated',
                'truncated',
                'early_stopped'):
            object.__setattr__(
                self,
                field_name,
                bool(getattr(self, field_name)),
            )
        if self.terminated and self.truncated:
            raise ValueError(
                'branch cannot be both terminated and truncated'
            )
        if self.status in {'completed', 'partial'}:
            if self.objective_return is None:
                raise ValueError(
                    'usable branch must provide objective_return'
                )
            object.__setattr__(
                self,
                'objective_return',
                _finite_float(
                    'objective_return',
                    self.objective_return,
                ),
            )
            if self.simulated_steps <= 0:
                raise ValueError(
                    'usable branch must simulate at least one step'
                )
            if self.failure_reason is not None:
                raise ValueError(
                    'usable branch cannot have failure_reason'
                )
        else:
            if self.objective_return is not None:
                raise ValueError(
                    'failed branch cannot provide objective_return'
                )
            object.__setattr__(
                self,
                'failure_reason',
                _non_empty_text(
                    'failure_reason',
                    self.failure_reason,
                ),
            )
        object.__setattr__(
            self,
            'diagnostic_codes',
            _code_tuple(
                'diagnostic_codes',
                self.diagnostic_codes,
            ),
        )

    @property
    def usable(self):
        return (
            self.status in {'completed', 'partial'}
            and self.objective_return is not None
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualResult:
    """一个任务的原动作复现和替代分支结果。"""

    task_id: str
    status: str
    actual_action_offset: int
    original_reproduced: bool
    branches: tuple[CounterfactualBranchResult, ...]
    simulated_steps: int
    failure_reason: str | None = None
    diagnostic_codes: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(
            self,
            'task_id',
            _non_empty_text('task_id', self.task_id),
        )
        if self.status not in COUNTERFACTUAL_RESULT_STATUSES:
            raise ValueError(
                'status must be completed, partial, or failed'
            )
        object.__setattr__(
            self,
            'actual_action_offset',
            _strict_int(
                'actual_action_offset',
                self.actual_action_offset,
                minimum=0,
                maximum=ANALYSIS_ACTION_COUNT - 1,
            ),
        )
        object.__setattr__(
            self,
            'original_reproduced',
            bool(self.original_reproduced),
        )
        branches = tuple(self.branches)
        if any(
                not isinstance(branch, CounterfactualBranchResult)
                for branch in branches):
            raise TypeError(
                'branches must contain CounterfactualBranchResult'
            )
        action_offsets = tuple(
            branch.action_offset
            for branch in branches
        )
        if len(set(action_offsets)) != len(action_offsets):
            raise ValueError(
                'branches must not repeat an action_offset'
            )
        object.__setattr__(
            self,
            'branches',
            tuple(sorted(
                branches,
                key=lambda branch: branch.action_offset,
            )),
        )
        simulated_steps = _strict_int(
            'simulated_steps',
            self.simulated_steps,
            minimum=0,
        )
        if simulated_steps != sum(
                branch.simulated_steps
                for branch in branches):
            raise ValueError(
                'simulated_steps must equal branch step total'
            )
        object.__setattr__(
            self,
            'simulated_steps',
            simulated_steps,
        )
        if self.status == 'completed':
            if not branches or any(
                    branch.status != 'completed'
                    for branch in branches):
                raise ValueError(
                    'completed result requires completed branches'
                )
            if self.failure_reason is not None:
                raise ValueError(
                    'completed result cannot have failure_reason'
                )
        elif self.status == 'failed':
            object.__setattr__(
                self,
                'failure_reason',
                _non_empty_text(
                    'failure_reason',
                    self.failure_reason,
                ),
            )
        elif self.failure_reason is not None:
            object.__setattr__(
                self,
                'failure_reason',
                _non_empty_text(
                    'failure_reason',
                    self.failure_reason,
                ),
            )
        object.__setattr__(
            self,
            'diagnostic_codes',
            _code_tuple(
                'diagnostic_codes',
                self.diagnostic_codes,
            ),
        )

    @property
    def label_ready(self):
        """只有原动作复现成功且至少一个替代分支可用时才允许造标签。"""

        if not self.original_reproduced:
            return False
        actual = next((
            branch
            for branch in self.branches
            if branch.action_offset == self.actual_action_offset
            and branch.usable
        ), None)
        if actual is None:
            return False
        return any(
            branch.usable
            and branch.action_offset != self.actual_action_offset
            for branch in self.branches
        )

    @property
    def return_deltas(self):
        """返回 ``actual - alternative``；无可信复现时返回空 tuple。"""

        if not self.label_ready:
            return ()
        actual = next(
            branch
            for branch in self.branches
            if branch.action_offset == self.actual_action_offset
        )
        return tuple(
            (
                branch.action_offset,
                float(actual.objective_return)
                - float(branch.objective_return),
            )
            for branch in self.branches
            if (
                branch.action_offset != self.actual_action_offset
                and branch.usable
            )
        )


class SnapshotRing:
    """按 TransitionKey 保存最近稳定边界的 worker-local 快照环。"""

    def __init__(self, capacity=32):
        self.capacity = _strict_int(
            'capacity',
            capacity,
            minimum=1,
        )
        self._items = OrderedDict()

    def __len__(self):
        return len(self._items)

    @property
    def keys(self):
        return tuple(self._items.keys())

    def push(self, transition_key, snapshot):
        """写入快照并返回被淘汰的 ``(key, snapshot)``，没有则返回 None。"""

        if not isinstance(transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        if not isinstance(snapshot, EngineSnapshot):
            raise TypeError('snapshot must be EngineSnapshot')
        if not snapshot.checksum_valid:
            raise ValueError('snapshot checksum is invalid')
        if snapshot.episode.step_count != transition_key.step_index:
            raise ValueError(
                'snapshot step_count must match transition_key'
            )
        if transition_key in self._items:
            del self._items[transition_key]
        self._items[transition_key] = snapshot
        if len(self._items) <= self.capacity:
            return None
        return self._items.popitem(last=False)

    def get(self, transition_key, default=None):
        if not isinstance(transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        return self._items.get(transition_key, default)

    def pop(self, transition_key):
        if not isinstance(transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        return self._items.pop(transition_key)

    def clear(self):
        count = len(self._items)
        self._items.clear()
        return count


def counterfactual_trigger_reasons(
        *,
        new_level=None,
        chain_depth=0,
        possible_blocker_causes=0,
        conflicting_signals=False,
        placement_confidence=None,
        random_rule_audit=False):
    """按 V1 白名单返回反事实触发原因；空 tuple 表示不应创建任务。"""

    reasons = []
    if new_level is not None:
        level = _strict_int('new_level', new_level, minimum=1)
        if level >= 7:
            reasons.append('high_value_merge')
    if _strict_int('chain_depth', chain_depth, minimum=0) >= 2:
        reasons.append('multi_stage_chain')
    if _strict_int(
            'possible_blocker_causes',
            possible_blocker_causes,
            minimum=0) >= 2:
        reasons.append('ambiguous_blocking')
    if bool(conflicting_signals):
        reasons.append('conflicting_signals')
    if placement_confidence is not None:
        confidence = _finite_float(
            'placement_confidence',
            placement_confidence,
            minimum=0.0,
            maximum=1.0,
        )
        if 0.55 <= confidence < 0.80:
            reasons.append('middle_placement_confidence')
    if bool(random_rule_audit):
        reasons.append('random_rule_audit')
    return tuple(reasons)


def select_counterfactual_alternatives(
        *,
        actual_action_offset,
        safest_action_scores=None,
        runner_up_action_offset=None,
        action_count=ANALYSIS_ACTION_COUNT,
        max_alternatives=3):
    """依次选择镜像、最安全和冻结策略 runner-up，稳定去重且最多三项。"""

    action_count = _strict_int(
        'action_count',
        action_count,
        minimum=1,
    )
    maximum = action_count - 1
    actual = _strict_int(
        'actual_action_offset',
        actual_action_offset,
        minimum=0,
        maximum=maximum,
    )
    max_alternatives = _strict_int(
        'max_alternatives',
        max_alternatives,
        minimum=1,
        maximum=3,
    )
    selected = []

    def add(action_offset):
        if (
                action_offset is not None
                and action_offset != actual
                and action_offset not in selected
                and len(selected) < max_alternatives):
            selected.append(action_offset)

    add(maximum - actual)

    if safest_action_scores is not None:
        scores = tuple(safest_action_scores)
        if len(scores) != action_count:
            raise ValueError(
                'safest_action_scores length must equal action_count'
            )
        valid_scores = []
        for action_offset, score in enumerate(scores):
            if score is None:
                continue
            valid_scores.append((
                _finite_float(
                    f'safest_action_scores[{action_offset}]',
                    score,
                ),
                -action_offset,
                action_offset,
            ))
        for _score, _negative_offset, action_offset in sorted(
                valid_scores,
                reverse=True):
            if action_offset != actual and action_offset not in selected:
                add(action_offset)
                break

    if runner_up_action_offset is not None:
        add(_strict_int(
            'runner_up_action_offset',
            runner_up_action_offset,
            minimum=0,
            maximum=maximum,
        ))
    return tuple(selected)


_TRIGGER_PRIORITY_WEIGHTS = {
    'high_value_merge': 4.0,
    'multi_stage_chain': 3.0,
    'ambiguous_blocking': 2.5,
    'conflicting_signals': 3.5,
    'middle_placement_confidence': 1.5,
    'random_rule_audit': 0.1,
}


def counterfactual_priority(
        *,
        event_utility,
        trigger_reasons,
        placement_confidence):
    """计算不依赖运行时 hash 或队列时序的稳定优先级。"""

    utility = _finite_float(
        'event_utility',
        event_utility,
        minimum=0.0,
    )
    reasons = _code_tuple(
        'trigger_reasons',
        trigger_reasons,
        allowed=COUNTERFACTUAL_TRIGGER_REASONS,
        require_non_empty=True,
    )
    confidence = _finite_float(
        'placement_confidence',
        placement_confidence,
        minimum=0.0,
        maximum=1.0,
    )
    return (
        utility
        + sum(_TRIGGER_PRIORITY_WEIGHTS[reason] for reason in reasons)
        + confidence
    )


def create_counterfactual_task(
        *,
        budget_key,
        transition_key,
        snapshot,
        factual_outcome,
        target_policy,
        actual_action_offset,
        alternative_action_offsets,
        trigger_reasons,
        event_utility,
        placement_confidence,
        created_real_step,
        attribution_version,
        config=None,
        priority=None,
        label_confidence=None,
        attribution_delay=0):
    """创建稳定任务；无触发原因或无替代动作时返回 ``None``。"""

    config = config or CounterfactualConfig()
    if not isinstance(config, CounterfactualConfig):
        raise TypeError('config must be CounterfactualConfig')
    reasons = tuple(trigger_reasons)
    alternatives = tuple(alternative_action_offsets)[
        :config.max_alternatives
    ]
    if not reasons or not alternatives:
        return None
    if priority is None:
        priority = counterfactual_priority(
            event_utility=event_utility,
            trigger_reasons=reasons,
            placement_confidence=placement_confidence,
        )
    if label_confidence is None:
        label_confidence = placement_confidence
    factual_outcome_fingerprint = engine_action_outcome_fingerprint(
        factual_outcome
    )
    if not isinstance(target_policy, FrozenTargetPolicyPayload):
        raise TypeError(
            'target_policy must be FrozenTargetPolicyPayload'
        )
    task_id = stable_counterfactual_task_id(
        budget_key=budget_key,
        transition_key=transition_key,
        snapshot_checksum=snapshot.checksum,
        factual_outcome_fingerprint=factual_outcome_fingerprint,
        target_policy_fingerprint=target_policy.fingerprint,
        actual_action_offset=actual_action_offset,
        alternative_action_offsets=alternatives,
        trigger_reasons=reasons,
        attribution_version=attribution_version,
        config_fingerprint=config.fingerprint,
    )
    return CounterfactualTask(
        task_id=task_id,
        budget_key=budget_key,
        transition_key=transition_key,
        snapshot=snapshot,
        factual_outcome=factual_outcome,
        factual_outcome_fingerprint=factual_outcome_fingerprint,
        target_policy=target_policy,
        actual_action_offset=actual_action_offset,
        alternative_action_offsets=alternatives,
        trigger_reasons=reasons,
        priority=priority,
        estimated_tokens=config.horizon * (1 + len(alternatives)),
        horizon=config.horizon,
        created_real_step=created_real_step,
        attribution_version=attribution_version,
        scheduler_config_fingerprint=config.fingerprint,
        label_confidence=label_confidence,
        attribution_delay=attribution_delay,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualSubmission:
    """一次非阻塞 admission 的结果。"""

    accepted: bool
    task_id: str | None
    drop_reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self, 'accepted', bool(self.accepted))
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
class CounterfactualTokenDecision:
    """外部 Shapley/审计任务对共享 token 账本的一次操作结果。"""

    accepted: bool
    reservation_id: str
    tokens: int
    drop_reason: str | None = None

    def __post_init__(self):
        if not isinstance(self.accepted, bool):
            raise TypeError('accepted must be bool')
        object.__setattr__(
            self,
            'reservation_id',
            _non_empty_text(
                'reservation_id',
                self.reservation_id,
            ),
        )
        object.__setattr__(
            self,
            'tokens',
            _strict_int('tokens', self.tokens, minimum=0),
        )
        if self.accepted:
            if self.drop_reason is not None:
                raise ValueError(
                    'accepted token decision cannot have drop_reason'
                )
        else:
            object.__setattr__(
                self,
                'drop_reason',
                _non_empty_text(
                    'drop_reason',
                    self.drop_reason,
                ),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class CounterfactualSchedulerStats:
    """全局反事实调度器的不可变统计快照。"""

    real_steps: int
    submitted: int
    accepted: int
    completed: int
    partial: int
    failed: int
    dropped: int
    labels_ready: int
    queued: int
    inflight: int
    queue_evicted: int
    tokens_reserved: int
    tokens_consumed: int
    tokens_refunded: int
    token_overrun: int
    soft_token_limit: float
    hard_token_limit: float
    soft_budget_borrows: int
    consecutive_failures: int
    circuit_open: bool
    drop_reason_counts: tuple[tuple[str, int], ...]
    external_active_reservations: int
    external_reservations_accepted: int
    external_reservations_settled: int
    external_reservations_refunded: int


def _validate_top_level_runner(runner):
    if not callable(runner):
        raise TypeError('runner must be callable')
    name = getattr(runner, '__name__', None)
    qualname = getattr(runner, '__qualname__', None)
    if name == '<lambda>' or (
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


def _counterfactual_runner_entry(runner, task):
    """ProcessPoolExecutor 使用的模块顶层入口。"""

    result = runner(task)
    if not isinstance(result, CounterfactualResult):
        raise TypeError(
            'counterfactual runner must return CounterfactualResult'
        )
    return result


class BudgetedCounterfactualScheduler:
    """全局 token 预算下的非阻塞 spawn 反事实调度器。

    admission 同时受三层限制：

    1. 每 ``min_real_steps`` 个真实投放最多获得一个任务槽位；
    2. 普通任务累计 token 不超过 8% 软预算；
    3. 达到稳定优先级阈值的任务可以借到 10% 硬上限，但绝不越过硬上限。

    任务一经接受会预留 ``estimated_tokens``。完成后按实际
    ``CounterfactualResult.simulated_steps`` 记账并退还未使用 token。队列满时只允许
    更高优先级任务替换最低优先级等待项；被淘汰任务不会生成结果或训练标签。
    """

    def __init__(
            self,
            *,
            worker_count,
            runner,
            config=None,
            executor=None):
        self.config = config or CounterfactualConfig()
        if not isinstance(self.config, CounterfactualConfig):
            raise TypeError('config must be CounterfactualConfig')
        self.worker_count = _strict_int(
            'worker_count',
            worker_count,
            minimum=1,
        )
        self.runner = _validate_top_level_runner(runner)
        self.inflight_limit = (
            self.worker_count
            * self.config.max_inflight_per_worker
        )

        self._owns_executor = executor is None
        if executor is None:
            context = multiprocessing.get_context('spawn')
            executor = ProcessPoolExecutor(
                max_workers=self.worker_count,
                mp_context=context,
            )
        elif not hasattr(executor, 'submit'):
            raise TypeError('executor must provide submit()')
        self._executor = executor

        self._lock = threading.RLock()
        self._queue = {}
        self._inflight = {}
        self._active_budgets = {}
        self._finished_budgets = set()
        self._external_reservations = {}
        self._finished_external_reservations = set()
        self._result_outbox = []
        self._closed = False
        self._close_results = ()
        self._circuit_open = False

        self._real_steps = 0
        self._admission_slots_used = 0
        self._tokens_reserved = 0
        self._tokens_consumed = 0
        self._tokens_refunded = 0
        self._token_overrun = 0
        self._submitted = 0
        self._accepted = 0
        self._completed = 0
        self._partial = 0
        self._failed = 0
        self._dropped = 0
        self._labels_ready = 0
        self._queue_evicted = 0
        self._soft_budget_borrows = 0
        self._consecutive_failures = 0
        self._drop_reason_counts = {}
        self._external_reservations_accepted = 0
        self._external_reservations_settled = 0
        self._external_reservations_refunded = 0

    @property
    def closed(self):
        return self._closed

    @property
    def circuit_open(self):
        return self._circuit_open

    @property
    def active_task_ids(self):
        """返回当前排队或执行中的稳定 task ID，不暴露 future/executor。

        主进程 coordinator 用它清理被优先级淘汰、熔断取消或 runner 异常且没有
        ``CounterfactualResult`` 的 pending 元数据。返回排序 tuple，调用方不能修改
        调度器内部状态。
        """

        with self._lock:
            return tuple(sorted(self._active_budgets.values()))

    @property
    def stats(self):
        with self._lock:
            return CounterfactualSchedulerStats(
                real_steps=self._real_steps,
                submitted=self._submitted,
                accepted=self._accepted,
                completed=self._completed,
                partial=self._partial,
                failed=self._failed,
                dropped=self._dropped,
                labels_ready=self._labels_ready,
                queued=len(self._queue),
                inflight=len(self._inflight),
                queue_evicted=self._queue_evicted,
                tokens_reserved=self._tokens_reserved,
                tokens_consumed=self._tokens_consumed,
                tokens_refunded=self._tokens_refunded,
                token_overrun=self._token_overrun,
                soft_token_limit=self._soft_limit,
                hard_token_limit=self._hard_limit,
                soft_budget_borrows=self._soft_budget_borrows,
                consecutive_failures=self._consecutive_failures,
                circuit_open=self._circuit_open,
                drop_reason_counts=tuple(sorted(
                    self._drop_reason_counts.items()
                )),
                external_active_reservations=len(
                    self._external_reservations
                ),
                external_reservations_accepted=(
                    self._external_reservations_accepted
                ),
                external_reservations_settled=(
                    self._external_reservations_settled
                ),
                external_reservations_refunded=(
                    self._external_reservations_refunded
                ),
            )

    @property
    def _soft_limit(self):
        return self._real_steps * self.config.cost_ratio

    @property
    def _hard_limit(self):
        return self._real_steps * self.config.cost_hard_limit

    def record_real_steps(self, step_count):
        """增加真实 rollout 步数并尝试派发已排队任务。"""

        step_count = _strict_int(
            'step_count',
            step_count,
            minimum=0,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    'counterfactual scheduler is closed'
                )
            self._real_steps += step_count
            self._collect_done_locked()
            self._dispatch_locked()
            return self._real_steps

    def reserve_external_tokens(
            self,
            reservation_id,
            tokens,
            *,
            priority=0.0):
        """为局部 Shapley 等外部物理工作预留共享预算。

        外部工作不占 ``min_real_steps`` proposal 槽位，但与反事实任务共用
        ``tokens_reserved/tokens_consumed`` 以及 8% 软、10% 硬上限。高优先级是否可
        借用硬预算沿用调度器同一阈值。
        """

        reservation_id = _non_empty_text(
            'reservation_id',
            reservation_id,
        )
        tokens = _strict_int('tokens', tokens, minimum=1)
        priority = _finite_float(
            'priority',
            priority,
            minimum=0.0,
        )
        with self._lock:
            self._collect_done_locked()
            if self._closed:
                return self._reject_external_locked(
                    reservation_id,
                    'scheduler_closed',
                )
            if self._circuit_open:
                return self._reject_external_locked(
                    reservation_id,
                    'circuit_open',
                )
            if (
                    reservation_id in self._external_reservations
                    or reservation_id
                    in self._finished_external_reservations):
                return self._reject_external_locked(
                    reservation_id,
                    'duplicate_external_reservation',
                )

            projected_tokens = (
                self._tokens_consumed
                + self._tokens_reserved
                + tokens
            )
            allowed_limit = self._soft_limit
            borrowed = False
            if priority >= self.config.soft_budget_borrow_priority:
                allowed_limit = self._hard_limit
                borrowed = projected_tokens > self._soft_limit
            if projected_tokens > allowed_limit:
                reason = (
                    'external_hard_token_budget'
                    if allowed_limit == self._hard_limit
                    else 'external_soft_token_budget'
                )
                return self._reject_external_locked(
                    reservation_id,
                    reason,
                )

            self._external_reservations[reservation_id] = tokens
            self._tokens_reserved += tokens
            self._external_reservations_accepted += 1
            if borrowed:
                self._soft_budget_borrows += 1
            self._dispatch_locked()
            return CounterfactualTokenDecision(
                accepted=True,
                reservation_id=reservation_id,
                tokens=tokens,
            )

    def settle_external_tokens(self, reservation_id, consumed):
        """按实际消耗结算；actual 不得超过保守预留值。"""

        reservation_id = _non_empty_text(
            'reservation_id',
            reservation_id,
        )
        consumed = _strict_int(
            'consumed',
            consumed,
            minimum=0,
        )
        with self._lock:
            reserved = self._external_reservations.get(
                reservation_id
            )
            if reserved is None:
                reason = (
                    'external_reservation_already_finalized'
                    if reservation_id
                    in self._finished_external_reservations
                    else 'unknown_external_reservation'
                )
                return self._reject_external_locked(
                    reservation_id,
                    reason,
                )
            if consumed > reserved:
                return self._reject_external_locked(
                    reservation_id,
                    'external_consumed_exceeds_reservation',
                )

            del self._external_reservations[reservation_id]
            self._finished_external_reservations.add(
                reservation_id
            )
            self._tokens_reserved -= reserved
            if self._tokens_reserved < 0:
                raise RuntimeError(
                    'counterfactual reserved token accounting underflow'
                )
            self._tokens_consumed += consumed
            self._tokens_refunded += reserved - consumed
            self._external_reservations_settled += 1
            self._dispatch_locked()
            return CounterfactualTokenDecision(
                accepted=True,
                reservation_id=reservation_id,
                tokens=consumed,
            )

    def refund_external_tokens(self, reservation_id):
        """取消尚未执行的外部工作并完整退还 reservation。"""

        reservation_id = _non_empty_text(
            'reservation_id',
            reservation_id,
        )
        with self._lock:
            reserved = self._external_reservations.get(
                reservation_id
            )
            if reserved is None:
                reason = (
                    'external_reservation_already_finalized'
                    if reservation_id
                    in self._finished_external_reservations
                    else 'unknown_external_reservation'
                )
                return self._reject_external_locked(
                    reservation_id,
                    reason,
                )
            self._refund_external_locked(
                reservation_id,
                reason=None,
            )
            self._dispatch_locked()
            return CounterfactualTokenDecision(
                accepted=True,
                reservation_id=reservation_id,
                tokens=reserved,
            )

    def submit(self, task):
        """非阻塞提交任务；不满足条件时只记 drop，不创建伪结果。"""

        with self._lock:
            self._submitted += 1
            self._collect_done_locked()
            if task is None:
                return self._reject_locked(
                    None,
                    'ineligible',
                )
            if not isinstance(task, CounterfactualTask):
                raise TypeError(
                    'task must be CounterfactualTask or None'
                )
            if self._closed:
                return self._reject_locked(
                    task,
                    'scheduler_closed',
                )
            if self._circuit_open:
                return self._reject_locked(
                    task,
                    'circuit_open',
                )
            if (
                    task.scheduler_config_fingerprint
                    != self.config.fingerprint):
                return self._reject_locked(
                    task,
                    'config_fingerprint_mismatch',
                )
            if task.horizon != self.config.horizon:
                return self._reject_locked(
                    task,
                    'horizon_mismatch',
                )
            if task.created_real_step > self._real_steps:
                return self._reject_locked(
                    task,
                    'future_real_step',
                )
            budget_key = task.budget_key
            if (
                    budget_key in self._active_budgets
                    or budget_key in self._finished_budgets):
                return self._reject_locked(
                    task,
                    'duplicate_budget',
                )

            replacement = None
            if len(self._queue) >= self.config.queue_capacity:
                replacement = min(
                    self._queue.values(),
                    key=lambda candidate: (
                        candidate.priority,
                        -candidate.created_real_step,
                        candidate.task_id,
                    ),
                )
                if task.priority <= replacement.priority:
                    return self._reject_locked(
                        task,
                        'queue_full_low_priority',
                    )

            needs_new_slot = replacement is None
            available_slots = (
                self._real_steps // self.config.min_real_steps
            )
            if (
                    needs_new_slot
                    and self._admission_slots_used >= available_slots):
                return self._reject_locked(
                    task,
                    'real_step_gate',
                )

            replacement_tokens = (
                replacement.estimated_tokens
                if replacement is not None
                else 0
            )
            projected_tokens = (
                self._tokens_consumed
                + self._tokens_reserved
                - replacement_tokens
                + task.estimated_tokens
            )
            allowed_limit = self._soft_limit
            borrowed = False
            if (
                    task.priority
                    >= self.config.soft_budget_borrow_priority):
                allowed_limit = self._hard_limit
                borrowed = projected_tokens > self._soft_limit
            if projected_tokens > allowed_limit:
                reason = (
                    'hard_token_budget'
                    if allowed_limit == self._hard_limit
                    else 'soft_token_budget'
                )
                return self._reject_locked(task, reason)

            if replacement is not None:
                self._evict_queued_locked(replacement)
            else:
                self._admission_slots_used += 1
            if borrowed:
                self._soft_budget_borrows += 1

            self._accepted += 1
            self._tokens_reserved += task.estimated_tokens
            self._active_budgets[budget_key] = task.task_id
            self._queue[task.task_id] = task
            self._dispatch_locked()
            return CounterfactualSubmission(
                accepted=True,
                task_id=task.task_id,
            )

    def poll(self):
        """仅处理已经完成的 future，立即返回，不等待正在运行的任务。"""

        with self._lock:
            results = self._collect_done_locked()
            self._dispatch_locked()
            return results

    def drain_results(self):
        """返回并清空全部已完成 runner 结果。drop 不会出现在这里。"""

        with self._lock:
            self._collect_done_locked()
            self._dispatch_locked()
            results = tuple(self._result_outbox)
            self._result_outbox.clear()
            return results

    def drain_label_results(self):
        """仅返回可以安全产生差值标签的结果。"""

        return tuple(
            result
            for result in self.drain_results()
            if result.label_ready
        )

    def close(self, *, wait=True):
        """幂等关闭；等待模式会收集已提交任务，队列项一律 drop。"""

        wait = bool(wait)
        with self._lock:
            if self._closed:
                return self._close_results
            self._closed = True
            for task in tuple(self._queue.values()):
                self._drop_accepted_locked(
                    task,
                    'scheduler_closed',
                )
            self._queue.clear()
            for reservation_id in tuple(
                    self._external_reservations):
                self._refund_external_locked(
                    reservation_id,
                    reason='external_scheduler_closed',
                )

        if self._owns_executor:
            self._executor.shutdown(
                wait=wait,
                cancel_futures=not wait,
            )

        with self._lock:
            if wait:
                self._collect_done_locked()
            for future, task in tuple(self._inflight.items()):
                if future.done():
                    continue
                if future.cancel():
                    self._inflight.pop(future, None)
                    self._drop_accepted_locked(
                        task,
                        'scheduler_closed',
                    )
            self._collect_done_locked()
            self._close_results = tuple(self._result_outbox)
            return self._close_results

    def _reject_locked(self, task, reason):
        self._record_drop_reason_locked(reason)
        return CounterfactualSubmission(
            accepted=False,
            task_id=(
                task.task_id
                if isinstance(task, CounterfactualTask)
                else None
            ),
            drop_reason=reason,
        )

    def _reject_external_locked(self, reservation_id, reason):
        self._record_drop_reason_locked(reason)
        return CounterfactualTokenDecision(
            accepted=False,
            reservation_id=reservation_id,
            tokens=0,
            drop_reason=reason,
        )

    def _refund_external_locked(self, reservation_id, *, reason):
        reserved = self._external_reservations.pop(
            reservation_id
        )
        self._finished_external_reservations.add(reservation_id)
        self._tokens_reserved -= reserved
        if self._tokens_reserved < 0:
            raise RuntimeError(
                'counterfactual reserved token accounting underflow'
            )
        self._tokens_refunded += reserved
        self._external_reservations_refunded += 1
        if reason is not None:
            self._record_drop_reason_locked(reason)
        return reserved

    def _record_drop_reason_locked(self, reason):
        self._dropped += 1
        self._drop_reason_counts[reason] = (
            self._drop_reason_counts.get(reason, 0) + 1
        )

    def _evict_queued_locked(self, task):
        self._queue.pop(task.task_id, None)
        self._queue_evicted += 1
        self._drop_accepted_locked(
            task,
            'queue_priority_evicted',
        )

    def _drop_accepted_locked(self, task, reason):
        self._tokens_reserved -= task.estimated_tokens
        if self._tokens_reserved < 0:
            raise RuntimeError(
                'counterfactual reserved token accounting underflow'
            )
        self._tokens_refunded += task.estimated_tokens
        self._active_budgets.pop(task.budget_key, None)
        self._record_drop_reason_locked(reason)

    def _dispatch_locked(self):
        if self._closed or self._circuit_open:
            return
        while (
                self._queue
                and len(self._inflight) < self.inflight_limit):
            task = max(
                self._queue.values(),
                key=lambda candidate: (
                    candidate.priority,
                    -candidate.created_real_step,
                    candidate.task_id,
                ),
            )
            self._queue.pop(task.task_id)
            try:
                future = self._executor.submit(
                    _counterfactual_runner_entry,
                    self.runner,
                    task,
                )
            except BaseException:
                self._release_failed_task_locked(
                    task,
                    'executor_submit_failure',
                    consumed_tokens=0,
                )
                continue
            self._inflight[future] = task

    def _collect_done_locked(self):
        completed_now = []
        for future, task in tuple(self._inflight.items()):
            # 前一个失败任务可能刚触发熔断，并在 `_open_circuit_locked()` 中取消、
            # 退款且移除了其它 future。外层 tuple 仍含旧快照，必须先跳过，避免对同一
            # reservation 二次退款导致 token 账本下溢。
            if future not in self._inflight:
                continue
            if not future.done():
                continue
            self._inflight.pop(future, None)
            cancelled = getattr(future, 'cancelled', None)
            if callable(cancelled) and cancelled():
                self._drop_accepted_locked(
                    task,
                    'future_cancelled',
                )
                continue
            try:
                result = future.result()
                self._validate_result_for_task(task, result)
            except BaseException:
                self._release_failed_task_locked(
                    task,
                    'runner_failure',
                    consumed_tokens=0,
                )
                continue
            self._release_reservation_locked(
                task,
                result.simulated_steps,
            )
            self._active_budgets.pop(task.budget_key, None)
            self._finished_budgets.add(task.budget_key)
            self._result_outbox.append(result)
            completed_now.append(result)

            if result.status == 'completed':
                self._completed += 1
            elif result.status == 'partial':
                self._partial += 1
            else:
                self._failed += 1
            if result.label_ready:
                self._labels_ready += 1

            if (
                    result.status == 'failed'
                    or not result.original_reproduced):
                self._consecutive_failures += 1
                if (
                        self._consecutive_failures
                        >= self.config.circuit_breaker_failures):
                    self._open_circuit_locked()
            else:
                self._consecutive_failures = 0
        return tuple(completed_now)

    def _validate_result_for_task(self, task, result):
        if not isinstance(result, CounterfactualResult):
            raise TypeError(
                'runner result must be CounterfactualResult'
            )
        if result.task_id != task.task_id:
            raise ValueError('runner result task_id mismatch')
        if result.actual_action_offset != task.actual_action_offset:
            raise ValueError(
                'runner result actual_action_offset mismatch'
            )
        requested = set(task.requested_action_offsets)
        returned = {
            branch.action_offset
            for branch in result.branches
        }
        if not returned.issubset(requested):
            raise ValueError(
                'runner returned an unrequested action branch'
            )
        if (
                result.status == 'completed'
                and returned != requested):
            raise ValueError(
                'completed result must contain every requested branch'
            )
        if any(
                branch.simulated_steps > task.horizon
                for branch in result.branches):
            raise ValueError(
                'branch simulated_steps exceeds task horizon'
            )

    def _release_reservation_locked(self, task, consumed_tokens):
        consumed_tokens = _strict_int(
            'consumed_tokens',
            consumed_tokens,
            minimum=0,
        )
        self._tokens_reserved -= task.estimated_tokens
        if self._tokens_reserved < 0:
            raise RuntimeError(
                'counterfactual reserved token accounting underflow'
            )
        self._tokens_consumed += consumed_tokens
        if consumed_tokens <= task.estimated_tokens:
            self._tokens_refunded += (
                task.estimated_tokens - consumed_tokens
            )
        else:
            self._token_overrun += (
                consumed_tokens - task.estimated_tokens
            )

    def _release_failed_task_locked(
            self,
            task,
            reason,
            *,
            consumed_tokens):
        self._release_reservation_locked(task, consumed_tokens)
        self._active_budgets.pop(task.budget_key, None)
        self._finished_budgets.add(task.budget_key)
        self._failed += 1
        self._consecutive_failures += 1
        self._drop_reason_counts[reason] = (
            self._drop_reason_counts.get(reason, 0) + 1
        )
        if (
                self._consecutive_failures
                >= self.config.circuit_breaker_failures):
            self._open_circuit_locked()

    def _open_circuit_locked(self):
        if self._circuit_open:
            return
        self._circuit_open = True
        for task in tuple(self._queue.values()):
            self._queue.pop(task.task_id, None)
            self._drop_accepted_locked(
                task,
                'circuit_open',
            )
        for future, task in tuple(self._inflight.items()):
            if future.cancel():
                self._inflight.pop(future, None)
                self._drop_accepted_locked(
                    task,
                    'circuit_open',
                )


def local_shapley_candidates(contributors, config=None):
    """按贡献权重选取 2～4 个不同历史动作。"""

    config = config or LocalShapleyConfig()
    if not isinstance(config, LocalShapleyConfig):
        raise TypeError('config must be LocalShapleyConfig')
    contributors = tuple(contributors)
    if any(
            not isinstance(contributor, Contributor)
            for contributor in contributors):
        raise TypeError('contributors must contain Contributor values')
    by_transition = {}
    for contributor in contributors:
        key = contributor.transition_key
        by_transition[key] = (
            by_transition.get(key, 0.0)
            + contributor.contribution_weight
        )
    ordered = tuple(
        transition_key
        for transition_key, _weight in sorted(
            by_transition.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )[:config.candidate_limit]
    )
    if len(ordered) < config.minimum_candidates:
        return ()
    return ordered


def local_shapley_eligible(
        event,
        *,
        has_synergy_ambiguity,
        config=None):
    """只允许最高价值、已确认且存在协同歧义的事件进入局部 Shapley。"""

    config = config or LocalShapleyConfig()
    if not isinstance(config, LocalShapleyConfig):
        raise TypeError('config must be LocalShapleyConfig')
    if not isinstance(event, AttributionEvent):
        raise TypeError('event must be AttributionEvent')
    if (
            event.status != 'confirmed'
            or event.utility < config.minimum_utility
            or not bool(has_synergy_ambiguity)):
        return False
    return bool(local_shapley_candidates(
        event.contributors,
        config=config,
    ))


def paired_shapley_permutations(
        candidates,
        *,
        pair_count=4,
        seed_material=''):
    """稳定生成 ``pair_count`` 个排列及其 reverse 配对。"""

    candidates = tuple(candidates)
    if len(candidates) < 2 or len(candidates) > 4:
        raise ValueError('candidates must contain 2 to 4 items')
    if len(set(candidates)) != len(candidates):
        raise ValueError('candidates must not contain duplicates')
    pair_count = _strict_int(
        'pair_count',
        pair_count,
        minimum=1,
    )
    seed_payload = repr((
        tuple(candidates),
        str(seed_material),
        pair_count,
    )).encode('utf-8')
    seed = int.from_bytes(
        hashlib.sha256(seed_payload).digest()[:8],
        byteorder='big',
        signed=False,
    )
    rng = random.Random(seed)
    pairs = []
    for pair_index in range(pair_count):
        permutation = list(candidates)
        rng.shuffle(permutation)
        if pair_index == 0:
            # 第一对使用规范顺序，便于日志和单元测试复核。
            permutation = list(candidates)
        ordered = tuple(permutation)
        pairs.append((ordered, tuple(reversed(ordered))))
    return tuple(pairs)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalShapleyEstimate:
    """少量配对排列得到的局部 Shapley 估计与效率残差。"""

    candidates: tuple[TransitionKey, ...]
    contributions: tuple[tuple[TransitionKey, float], ...]
    empty_value: float
    full_value: float
    efficiency_residual: float
    evaluated_subset_count: int
    cache_hit_count: int
    permutation_count: int


def estimate_local_shapley(
        candidates,
        evaluator,
        *,
        permutation_pairs=None,
        subset_cache=None):
    """用 subset cache 计算配对排列边际贡献和 efficiency residual。"""

    candidates = tuple(candidates)
    if (
            len(candidates) < 2
            or len(candidates) > 4
            or len(set(candidates)) != len(candidates)
            or any(
                not isinstance(candidate, TransitionKey)
                for candidate in candidates)):
        raise ValueError(
            'candidates must contain 2 to 4 unique TransitionKey values'
        )
    if not callable(evaluator):
        raise TypeError('evaluator must be callable')
    if permutation_pairs is None:
        permutation_pairs = paired_shapley_permutations(candidates)
    permutation_pairs = tuple(permutation_pairs)
    flattened = []
    expected = set(candidates)
    for pair in permutation_pairs:
        if len(pair) != 2:
            raise ValueError(
                'each permutation pair must contain forward and reverse'
            )
        for permutation in pair:
            permutation = tuple(permutation)
            if (
                    len(permutation) != len(candidates)
                    or set(permutation) != expected):
                raise ValueError(
                    'each permutation must contain every candidate once'
                )
            flattened.append(permutation)
    if not flattened:
        raise ValueError('permutation_pairs must not be empty')

    cache = {} if subset_cache is None else subset_cache
    if not isinstance(cache, dict):
        raise TypeError('subset_cache must be dict or None')
    evaluated_count = 0
    cache_hits = 0

    def value_of(subset):
        nonlocal evaluated_count, cache_hits
        key = frozenset(subset)
        if key in cache:
            cache_hits += 1
            return _finite_float('cached subset value', cache[key])
        value = _finite_float('subset evaluator result', evaluator(key))
        cache[key] = value
        evaluated_count += 1
        return value

    totals = {candidate: 0.0 for candidate in candidates}
    for permutation in flattened:
        prefix = frozenset()
        previous_value = value_of(prefix)
        for candidate in permutation:
            next_prefix = prefix.union((candidate,))
            next_value = value_of(next_prefix)
            totals[candidate] += next_value - previous_value
            prefix = next_prefix
            previous_value = next_value

    divisor = float(len(flattened))
    contributions = tuple(
        (candidate, totals[candidate] / divisor)
        for candidate in candidates
    )
    empty_value = value_of(frozenset())
    full_value = value_of(frozenset(candidates))
    residual = (
        full_value
        - empty_value
        - sum(value for _candidate, value in contributions)
    )
    return LocalShapleyEstimate(
        candidates=candidates,
        contributions=contributions,
        empty_value=empty_value,
        full_value=full_value,
        efficiency_residual=residual,
        evaluated_subset_count=evaluated_count,
        cache_hit_count=cache_hits,
        permutation_count=len(flattened),
    )


class CumulativeShapleySelector:
    """保证全程选择比例不超过 ``event_ratio_max`` 的累计门控器。"""

    def __init__(self, config=None):
        self.config = config or LocalShapleyConfig()
        if not isinstance(self.config, LocalShapleyConfig):
            raise TypeError('config must be LocalShapleyConfig')
        self.observed_event_count = 0
        self.selected_event_count = 0

    def consider(self, *, eligible):
        self.observed_event_count += 1
        allowed = math.floor(
            self.observed_event_count
            * self.config.event_ratio_max
            + 1e-12
        )
        if bool(eligible) and self.selected_event_count < allowed:
            self.selected_event_count += 1
            return True
        return False

    @property
    def selected_ratio(self):
        if self.observed_event_count <= 0:
            return 0.0
        return (
            self.selected_event_count
            / self.observed_event_count
        )


__all__ = [
    'COUNTERFACTUAL_BRANCH_STATUSES',
    'COUNTERFACTUAL_RESULT_STATUSES',
    'COUNTERFACTUAL_TRIGGER_REASONS',
    'FROZEN_TARGET_POLICY_SCHEMA_VERSION',
    'BudgetedCounterfactualScheduler',
    'CounterfactualBranchResult',
    'CounterfactualConfig',
    'CounterfactualResult',
    'CounterfactualSchedulerStats',
    'CounterfactualSubmission',
    'CounterfactualTask',
    'CounterfactualTokenDecision',
    'CumulativeShapleySelector',
    'FrozenGNNModelConfig',
    'FrozenGraphBuilderConfig',
    'FrozenTargetPolicyPayload',
    'LocalShapleyConfig',
    'LocalShapleyEstimate',
    'SnapshotRing',
    'counterfactual_priority',
    'counterfactual_trigger_reasons',
    'create_counterfactual_task',
    'engine_action_outcome_fingerprint',
    'estimate_local_shapley',
    'local_shapley_candidates',
    'local_shapley_eligible',
    'paired_shapley_permutations',
    'select_counterfactual_alternatives',
    'stable_counterfactual_task_id',
]
