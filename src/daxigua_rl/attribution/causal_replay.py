"""稀疏因果监督样本、worker 历史上下文和纯内存分层回放。

本模块刻意不依赖 collector、训练循环或主 ``ReplayBuffer``。它只承担三件事：

1. 固定 ``CausalSample`` 的跨进程数据契约；
2. 在 worker 内短期保存可由 ``TransitionKey`` 回查的动作前图和状态分析；
3. 把已确认的 A/B 级 ``AttributionEvent`` 按唯一价值预算合并成规则排序样本，
   并用独立的固定容量内存回放做分层采样。

完整 ``StateAnalysis`` 只存在于短期 context cache，进入因果回放的样本仍只保存
``GraphTensor`` 和必要元数据。这样既能在延迟事件确认时回查历史状态，也不会污染
主 TD replay 或要求随机修改冷磁盘段。
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from numbers import Real
from operator import index

import torch

from daxigua_rl.graph.tensor import GraphTensor
from daxigua_rl.reward import merge_utility
from daxigua_rl.training.identity import TransitionKey

from .schema import (
    ANALYSIS_ACTION_COUNT,
    AttributionEvent,
    AttributionEventKey,
    MergeValueKey,
    StateAnalysis,
)


CAUSAL_SAMPLE_SCHEMA_VERSION = 1
CAUSAL_STRATA = (
    'positive_setup',
    'negative_blocking',
    'counterfactual',
)
CAUSAL_SUPERVISION_KINDS = ('rule', 'counterfactual', 'shapley')


def _strict_integer(name, value, *, minimum=None, maximum=None):
    """读取严格整数，拒绝 bool 和会被静默截断的浮点数。"""

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
    """规范化有限实数，并按需验证闭区间边界。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None and result > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return result


def _non_empty_string(name, value):
    if not isinstance(value, str):
        raise TypeError(f'{name} must be str')
    result = value.strip()
    if not result:
        raise ValueError(f'{name} must not be empty')
    return result


def _optional_string(name, value):
    if value is None:
        return None
    return _non_empty_string(name, value)


def stable_event_key(event_key):
    """把事件身份编码成不依赖 ``repr`` 或随机 hash 的稳定字符串。"""

    if not isinstance(event_key, AttributionEventKey):
        raise TypeError('event_key must be AttributionEventKey')
    return (
        'attribution-event-v1:'
        f'{event_key.worker_id}:{event_key.episode_id}:'
        f'{event_key.event_index}'
    )


def stable_budget_key(budget_key):
    """编码事件或合成价值包身份，供去重、日志和跨进程传输使用。"""

    if isinstance(budget_key, AttributionEventKey):
        return (
            'attribution-budget-v1:event:'
            f'{budget_key.worker_id}:{budget_key.episode_id}:'
            f'{budget_key.event_index}'
        )
    if isinstance(budget_key, MergeValueKey):
        transition_key = budget_key.transition_key
        return (
            'attribution-budget-v1:merge:'
            f'{transition_key.worker_id}:{transition_key.episode_id}:'
            f'{transition_key.step_index}:{budget_key.event_offset}'
        )
    raise TypeError(
        'budget_key must be AttributionEventKey or MergeValueKey'
    )


def graph_schema_fingerprint(graph):
    """返回只描述图 schema 和动作映射的稳定短指纹。

    指纹不包含浮点特征值，因此同一 schema 的不同状态会共享它；训练日志可以据此
    检查因果样本是否混入了不兼容的图构建版本。
    """

    _validate_graph_tensor(graph)
    payload = {
        'node_feature_names': tuple(graph.node_feature_names),
        'edge_feature_names': tuple(graph.edge_feature_names),
        'node_feature_dim': graph.node_feature_dim,
        'edge_feature_dim': graph.edge_feature_dim,
        'action_indices': tuple(
            int(value)
            for value in graph.action_indices.tolist()
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()[:16]


def _validate_index_tensor(name, value, *, dimensions):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'graph.{name} must be torch.Tensor')
    if value.device.type != 'cpu':
        raise ValueError(f'graph.{name} must be stored on CPU')
    if value.ndim != dimensions:
        raise ValueError(
            f'graph.{name} must have {dimensions} dimensions'
        )
    if value.dtype == torch.bool or value.is_floating_point():
        raise TypeError(f'graph.{name} must use an integer dtype')
    if value.requires_grad:
        raise ValueError(f'graph.{name} must not require gradients')


def _validate_feature_tensor(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'graph.{name} must be torch.Tensor')
    if value.device.type != 'cpu':
        raise ValueError(f'graph.{name} must be stored on CPU')
    if value.ndim != 2:
        raise ValueError(f'graph.{name} must be a matrix')
    if not value.is_floating_point():
        raise TypeError(f'graph.{name} must use a floating dtype')
    if value.requires_grad:
        raise ValueError(f'graph.{name} must not require gradients')
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f'graph.{name} must contain only finite values')


def _validate_feature_names(name, values, expected_length):
    if not isinstance(values, tuple):
        raise TypeError(f'graph.{name} must be tuple')
    if len(values) != expected_length:
        raise ValueError(
            f'graph.{name} length must match the feature dimension'
        )
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(
            f'graph.{name} must contain non-empty strings'
        )
    if len(set(values)) != len(values):
        raise ValueError(f'graph.{name} must not contain duplicates')


def _validate_graph_tensor(graph):
    """严格检查因果 replay 中图对象的 shape、索引、设备和有限性。"""

    if not isinstance(graph, GraphTensor):
        raise TypeError(
            f'graph must be GraphTensor, got {type(graph)!r}'
        )

    _validate_feature_tensor('node_features', graph.node_features)
    _validate_feature_tensor('edge_features', graph.edge_features)
    _validate_index_tensor('edge_index', graph.edge_index, dimensions=2)
    _validate_index_tensor(
        'action_node_indices',
        graph.action_node_indices,
        dimensions=1,
    )
    _validate_index_tensor(
        'action_indices',
        graph.action_indices,
        dimensions=1,
    )

    if graph.node_features.shape[0] <= 0:
        raise ValueError('graph must contain at least one node')
    if graph.node_features.shape[1] <= 0:
        raise ValueError('graph must contain node features')
    if graph.edge_features.shape[1] <= 0:
        raise ValueError('graph must contain an edge feature schema')
    if graph.edge_index.shape[0] != 2:
        raise ValueError('graph.edge_index first dimension must equal 2')
    if graph.edge_index.shape[1] != graph.edge_features.shape[0]:
        raise ValueError(
            'graph.edge_index edge count must match graph.edge_features'
        )
    if graph.action_node_indices.numel() <= 0:
        raise ValueError('graph must contain at least one action')
    if (
            graph.action_node_indices.shape[0]
            != graph.action_indices.shape[0]):
        raise ValueError(
            'graph action index tensors must have equal lengths'
        )

    num_nodes = int(graph.node_features.shape[0])
    if graph.edge_index.numel():
        minimum = int(graph.edge_index.min().item())
        maximum = int(graph.edge_index.max().item())
        if minimum < 0 or maximum >= num_nodes:
            raise ValueError('graph.edge_index contains an invalid node index')

    action_nodes = tuple(
        int(value)
        for value in graph.action_node_indices.tolist()
    )
    if any(value < 0 or value >= num_nodes for value in action_nodes):
        raise ValueError(
            'graph.action_node_indices contains an invalid node index'
        )
    if len(set(action_nodes)) != len(action_nodes):
        raise ValueError(
            'graph.action_node_indices must not contain duplicates'
        )

    action_indices = tuple(
        int(value)
        for value in graph.action_indices.tolist()
    )
    if len(set(action_indices)) != len(action_indices):
        raise ValueError('graph.action_indices must not contain duplicates')

    _validate_feature_names(
        'node_feature_names',
        graph.node_feature_names,
        int(graph.node_features.shape[1]),
    )
    _validate_feature_names(
        'edge_feature_names',
        graph.edge_feature_names,
        int(graph.edge_features.shape[1]),
    )


def _graph_storage_stats(graphs):
    """估算一组图引用实际占用的唯一 PyTorch CPU storage。

    多条因果标签经常来自同一历史状态。``CausalSample`` 不复制图张量，因此这里按
    底层 storage 指针去重，而不是把每条 sample 的图字节简单相加。视图共享的大块
    storage 也只会计算一次，结果更接近回放真实常驻张量内存。
    """

    graphs = tuple(graphs)
    unique_graph_objects = {id(graph) for graph in graphs}
    unique_storages = {}
    naive_bytes = 0
    for graph in graphs:
        per_graph_storages = {}
        for tensor in (
                graph.node_features,
                graph.edge_index,
                graph.edge_features,
                graph.action_node_indices,
                graph.action_indices):
            storage = tensor.untyped_storage()
            storage_key = (
                tensor.device.type,
                tensor.device.index,
                storage.data_ptr(),
                storage.nbytes(),
            )
            per_graph_storages[storage_key] = storage.nbytes()
            unique_storages[storage_key] = storage.nbytes()
        naive_bytes += sum(per_graph_storages.values())
    unique_bytes = sum(unique_storages.values())
    return {
        'unique_graph_count': len(unique_graph_objects),
        'unique_graph_storage_count': len(unique_storages),
        'estimated_unique_graph_bytes': unique_bytes,
        'estimated_graph_bytes_without_sharing': naive_bytes,
        'estimated_graph_sharing_saved_bytes': max(
            0,
            naive_bytes - unique_bytes,
        ),
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class CausalSample:
    """一条独立于主 TD replay 的动作对因果监督。

    ``actual_action_offset`` 和 ``comparison_action_offset`` 都是当前图扁平 Q 数组
    的下标。``direction=+1`` 表示实际动作应优于比较动作，``-1`` 表示应更差。
    ``event_key`` 与 ``budget_key`` 使用本模块的稳定字符串编码，不能保存 Python
    对象 ``repr``。
    """

    graph: GraphTensor
    actual_action_offset: int
    comparison_action_offset: int
    direction: int
    target_margin: float
    confidence: float
    cause_type: str
    delay: int
    transition_key: TransitionKey
    attribution_version: str
    supervision_kind: str
    stratum: str
    event_key: str
    budget_key: str
    target_delta: float | None = None
    policy_version: str | None = None
    tracker_config_fingerprint: str | None = None
    analyzer_config_fingerprint: str | None = None
    graph_schema_fingerprint: str | None = None
    schema_version: int = CAUSAL_SAMPLE_SCHEMA_VERSION

    def __post_init__(self):
        _validate_graph_tensor(self.graph)
        action_count = int(self.graph.action_node_indices.shape[0])
        for field_name in (
                'actual_action_offset',
                'comparison_action_offset'):
            object.__setattr__(
                self,
                field_name,
                _strict_integer(
                    field_name,
                    getattr(self, field_name),
                    minimum=0,
                    maximum=action_count - 1,
                ),
            )
        if self.actual_action_offset == self.comparison_action_offset:
            raise ValueError('actual and comparison actions must differ')

        direction = _strict_integer('direction', self.direction)
        if direction not in (-1, 1):
            raise ValueError('direction must be -1 or +1')
        object.__setattr__(self, 'direction', direction)

        target_margin = _finite_float(
            'target_margin',
            self.target_margin,
            minimum=0.0,
        )
        object.__setattr__(self, 'target_margin', target_margin)
        confidence = _finite_float(
            'confidence',
            self.confidence,
            minimum=0.0,
            maximum=1.0,
        )
        if confidence == 0.0:
            raise ValueError('confidence must be positive')
        object.__setattr__(self, 'confidence', confidence)
        object.__setattr__(
            self,
            'cause_type',
            _non_empty_string('cause_type', self.cause_type),
        )
        object.__setattr__(
            self,
            'delay',
            _strict_integer('delay', self.delay, minimum=0),
        )
        if not isinstance(self.transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        object.__setattr__(
            self,
            'attribution_version',
            _non_empty_string(
                'attribution_version',
                self.attribution_version,
            ),
        )

        if self.supervision_kind not in CAUSAL_SUPERVISION_KINDS:
            raise ValueError(
                'supervision_kind must be one of '
                f'{CAUSAL_SUPERVISION_KINDS!r}'
            )
        if self.stratum not in CAUSAL_STRATA:
            raise ValueError(f'stratum must be one of {CAUSAL_STRATA!r}')
        if self.supervision_kind == 'rule':
            if self.stratum == 'counterfactual':
                raise ValueError(
                    'rule supervision cannot use counterfactual stratum'
                )
            if target_margin == 0.0:
                raise ValueError(
                    'rule supervision requires a positive target_margin'
                )
        elif self.stratum != 'counterfactual':
            raise ValueError(
                'counterfactual and shapley supervision must use '
                'counterfactual stratum'
            )

        for field_name in ('event_key', 'budget_key'):
            object.__setattr__(
                self,
                field_name,
                _non_empty_string(
                    field_name,
                    getattr(self, field_name),
                ),
            )

        if self.target_delta is not None:
            target_delta = _finite_float(
                'target_delta',
                self.target_delta,
            )
            object.__setattr__(self, 'target_delta', target_delta)
            if target_delta > 0.0 and direction != 1:
                raise ValueError(
                    'positive target_delta requires direction=+1'
                )
            if target_delta < 0.0 and direction != -1:
                raise ValueError(
                    'negative target_delta requires direction=-1'
                )

        for field_name in (
                'policy_version',
                'tracker_config_fingerprint',
                'analyzer_config_fingerprint',
                'graph_schema_fingerprint'):
            object.__setattr__(
                self,
                field_name,
                _optional_string(
                    field_name,
                    getattr(self, field_name),
                ),
            )

        schema_version = _strict_integer(
            'schema_version',
            self.schema_version,
            minimum=1,
        )
        if schema_version != CAUSAL_SAMPLE_SCHEMA_VERSION:
            raise ValueError(
                'unsupported causal sample schema_version '
                f'{schema_version}; expected '
                f'{CAUSAL_SAMPLE_SCHEMA_VERSION}'
            )
        object.__setattr__(self, 'schema_version', schema_version)

    @property
    def pair_key(self):
        """返回同一状态下无方向动作对的稳定去重键。"""

        low = min(
            self.actual_action_offset,
            self.comparison_action_offset,
        )
        high = max(
            self.actual_action_offset,
            self.comparison_action_offset,
        )
        return self.transition_key, low, high

    @property
    def preferred_action_offset(self):
        """按方向返回该监督认为更优的动作下标。"""

        return (
            self.actual_action_offset
            if self.direction > 0
            else self.comparison_action_offset
        )


class CausalReplayBuffer:
    """固定容量、三 strata 且兼顾 cause type 的纯内存回放。

    采样先轮转 strata，再在各 stratum 内轮转 cause type。即使某一高频原因持续
    写入，低频原因也会在后续 batch 中获得位置。容量满时同样优先从数量最多的
    stratum/cause type 淘汰最旧样本。
    """

    def __init__(self, capacity=20_000, seed=None):
        self.capacity = _strict_integer(
            'capacity',
            capacity,
            minimum=1,
        )
        self._seed = seed
        self._rng = random.Random(seed)
        self._reset_storage()
        self._stratum_cursor = self._rng.randrange(len(CAUSAL_STRATA))
        self._cause_cursors = defaultdict(int)
        self._counterfactual_override_count = 0
        self._ignored_weaker_rule_count = 0
        self._eviction_count = 0
        self._rule_empirical_agreement_count = 0
        self._rule_empirical_disagreement_count = 0

    def _reset_storage(self):
        """初始化主顺序表及所有派生索引。

        ``_items`` 的 key 是只增不减的内部 entry id，因此 OrderedDict 顺序就是
        replay 的从旧到新顺序。其余结构全部是可从 ``_items`` 重建的索引：

        - pair / identity 索引使常规去重与经验优先级查询不再扫描全池；
        - bucket 的 dense id 数组支持 O(1) 随机下标和 swap-remove；
        - bucket 的 OrderedDict 保留最旧 entry，供精确淘汰；
        - 每个 stratum 的 lazy heap 按 ``(-count, oldest_id)`` 找最拥挤且最旧
          的 cause bucket。
        """

        self._items = OrderedDict()
        self._next_entry_id = 0
        self._identity_index = defaultdict(set)
        self._pair_index = defaultdict(set)
        self._pair_empirical_priority_counts = defaultdict(Counter)
        self._pair_rule_preference_counts = defaultdict(Counter)
        self._bucket_entry_ids = defaultdict(list)
        self._bucket_positions = defaultdict(dict)
        self._bucket_order = defaultdict(OrderedDict)
        self._stratum_counts = Counter()
        self._cause_counts = Counter()
        self._cause_type_counts = Counter()
        self._supervision_kind_counts = Counter()
        self._causes_by_stratum = defaultdict(set)
        self._eviction_heaps = {
            stratum: []
            for stratum in CAUSAL_STRATA
        }

    def __len__(self):
        return len(self._items)

    @property
    def is_full(self):
        return len(self) == self.capacity

    @property
    def remaining_capacity(self):
        return max(0, self.capacity - len(self))

    def is_ready(self, batch_size=1):
        """返回是否已有足够样本进行一次指定大小的无放回采样。"""

        batch_size = _strict_integer(
            'batch_size',
            batch_size,
            minimum=1,
        )
        return len(self) >= batch_size

    @staticmethod
    def _identity_key(sample):
        return (
            sample.supervision_kind,
            sample.event_key,
            sample.budget_key,
            sample.transition_key,
            sample.actual_action_offset,
            sample.comparison_action_offset,
            sample.direction,
        )

    @staticmethod
    def _supervision_priority(sample):
        return {
            'rule': 0,
            'counterfactual': 1,
            'shapley': 2,
        }[sample.supervision_kind]

    @staticmethod
    def _bucket_key(sample):
        return sample.stratum, sample.cause_type

    def _push_eviction_heap_entry(self, bucket):
        stratum, cause_type = bucket
        order = self._bucket_order.get(bucket)
        if order:
            heapq.heappush(
                self._eviction_heaps[stratum],
                (-len(order), next(iter(order)), cause_type),
            )
        self._compact_eviction_heap_if_needed(stratum)

    def _compact_eviction_heap_if_needed(self, stratum):
        """限制 lazy heap 的陈旧节点数量，保持长期训练内存有界。"""

        active_causes = len(self._causes_by_stratum.get(stratum, ()))
        heap = self._eviction_heaps[stratum]
        if len(heap) <= max(64, 4 * active_causes):
            return
        rebuilt = []
        for cause_type in self._causes_by_stratum[stratum]:
            bucket = stratum, cause_type
            order = self._bucket_order[bucket]
            rebuilt.append(
                (-len(order), next(iter(order)), cause_type)
            )
        heapq.heapify(rebuilt)
        self._eviction_heaps[stratum] = rebuilt

    def _peek_eviction_heap(self, stratum):
        heap = self._eviction_heaps[stratum]
        while heap:
            negative_count, oldest_id, cause_type = heap[0]
            bucket = stratum, cause_type
            order = self._bucket_order.get(bucket)
            if (
                    order
                    and -negative_count == len(order)
                    and oldest_id == next(iter(order))):
                return negative_count, oldest_id, cause_type
            heapq.heappop(heap)
        raise RuntimeError(
            f'causal replay eviction index is empty for {stratum!r}'
        )

    def _index_entry(self, entry_id, sample):
        identity = self._identity_key(sample)
        pair_key = sample.pair_key
        bucket = self._bucket_key(sample)

        self._identity_index[identity].add(entry_id)
        self._pair_index[pair_key].add(entry_id)
        if sample.supervision_kind == 'rule':
            self._pair_rule_preference_counts[pair_key][
                sample.preferred_action_offset
            ] += 1
        else:
            priority = self._supervision_priority(sample)
            self._pair_empirical_priority_counts[pair_key][priority] += 1

        dense_ids = self._bucket_entry_ids[bucket]
        self._bucket_positions[bucket][entry_id] = len(dense_ids)
        dense_ids.append(entry_id)
        self._bucket_order[bucket][entry_id] = None
        self._stratum_counts[sample.stratum] += 1
        self._cause_counts[bucket] += 1
        self._cause_type_counts[sample.cause_type] += 1
        self._supervision_kind_counts[sample.supervision_kind] += 1
        self._causes_by_stratum[sample.stratum].add(sample.cause_type)
        self._push_eviction_heap_entry(bucket)

    @staticmethod
    def _discard_index_entry(index, key, entry_id):
        entry_ids = index[key]
        entry_ids.remove(entry_id)
        if not entry_ids:
            del index[key]

    @staticmethod
    def _decrement_counter_index(index, key, counter_key):
        counts = index[key]
        counts[counter_key] -= 1
        if counts[counter_key] == 0:
            del counts[counter_key]
        if not counts:
            del index[key]

    def _remove_entry(self, entry_id):
        sample = self._items.pop(entry_id)
        identity = self._identity_key(sample)
        pair_key = sample.pair_key
        bucket = self._bucket_key(sample)

        self._discard_index_entry(
            self._identity_index,
            identity,
            entry_id,
        )
        self._discard_index_entry(
            self._pair_index,
            pair_key,
            entry_id,
        )
        if sample.supervision_kind == 'rule':
            self._decrement_counter_index(
                self._pair_rule_preference_counts,
                pair_key,
                sample.preferred_action_offset,
            )
        else:
            self._decrement_counter_index(
                self._pair_empirical_priority_counts,
                pair_key,
                self._supervision_priority(sample),
            )

        dense_ids = self._bucket_entry_ids[bucket]
        positions = self._bucket_positions[bucket]
        position = positions.pop(entry_id)
        last_id = dense_ids.pop()
        if position < len(dense_ids):
            dense_ids[position] = last_id
            positions[last_id] = position
        del self._bucket_order[bucket][entry_id]

        self._stratum_counts[sample.stratum] -= 1
        if self._stratum_counts[sample.stratum] == 0:
            del self._stratum_counts[sample.stratum]
        self._cause_counts[bucket] -= 1
        self._cause_type_counts[sample.cause_type] -= 1
        if self._cause_type_counts[sample.cause_type] == 0:
            del self._cause_type_counts[sample.cause_type]
        self._supervision_kind_counts[sample.supervision_kind] -= 1
        if self._supervision_kind_counts[sample.supervision_kind] == 0:
            del self._supervision_kind_counts[sample.supervision_kind]

        if self._cause_counts[bucket] == 0:
            del self._cause_counts[bucket]
            del self._bucket_entry_ids[bucket]
            del self._bucket_positions[bucket]
            del self._bucket_order[bucket]
            causes = self._causes_by_stratum[sample.stratum]
            causes.remove(sample.cause_type)
            if not causes:
                del self._causes_by_stratum[sample.stratum]
        self._push_eviction_heap_entry(bucket)
        return sample

    def _append_entry(self, sample):
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        self._items[entry_id] = sample
        self._index_entry(entry_id, sample)
        return entry_id

    def push(self, sample):
        """写入样本；返回样本是否成为当前 replay 的有效内容。

        同一无方向动作对出现更强的反事实或 Shapley 标签时，会删除旧规则标签。
        反过来到达的规则标签会被忽略，避免同一 pair 同时受到重复、可能冲突的监督。
        """

        if not isinstance(sample, CausalSample):
            raise TypeError(
                f'sample must be CausalSample, got {type(sample)!r}'
            )

        pair_key = sample.pair_key
        pair_matches = self._pair_index.get(pair_key, ())
        if sample.supervision_kind != 'rule':
            rule_preferences = self._pair_rule_preference_counts.get(
                pair_key,
                {},
            )
            agreement_count = rule_preferences.get(
                sample.preferred_action_offset,
                0,
            )
            rule_count = sum(rule_preferences.values())
            self._rule_empirical_agreement_count += agreement_count
            self._rule_empirical_disagreement_count += (
                rule_count - agreement_count
            )
        incoming_priority = self._supervision_priority(sample)
        existing_empirical_priorities = (
            self._pair_empirical_priority_counts.get(pair_key, {})
        )

        if (
                sample.supervision_kind == 'rule'
                and existing_empirical_priorities):
            self._ignored_weaker_rule_count += 1
            return False
        if (
                existing_empirical_priorities
                and max(existing_empirical_priorities) > incoming_priority):
            self._ignored_weaker_rule_count += 1
            return False

        remove_entry_ids = set(
            self._identity_index.get(self._identity_key(sample), ())
        )
        if sample.supervision_kind != 'rule':
            for entry_id in pair_matches:
                existing = self._items[entry_id]
                if (
                        self._supervision_priority(existing)
                        <= incoming_priority):
                    remove_entry_ids.add(entry_id)

        removed_rule = any(
            self._items[entry_id].supervision_kind == 'rule'
            for entry_id in remove_entry_ids
        )
        if removed_rule:
            self._counterfactual_override_count += 1
        for entry_id in sorted(remove_entry_ids):
            self._remove_entry(entry_id)

        self._append_entry(sample)
        while len(self._items) > self.capacity:
            self._evict_one()
        return True

    def extend(self, samples):
        """批量写入并返回实际成为有效回放内容的数量。"""

        count = 0
        for sample in samples:
            count += bool(self.push(sample))
        return count

    def _evict_one(self):
        """从最拥挤 stratum/cause 中淘汰最旧值，保护稀有类别。"""

        if not self._items:
            raise RuntimeError('cannot evict from an empty causal replay')
        maximum_stratum_count = max(self._stratum_counts.values())
        crowded_strata = (
            stratum
            for stratum in CAUSAL_STRATA
            if self._stratum_counts.get(stratum, 0)
            == maximum_stratum_count
        )
        _, eviction_id, _ = min(
            self._peek_eviction_heap(stratum)
            for stratum in crowded_strata
        )
        self._remove_entry(eviction_id)
        self._eviction_count += 1

    def sample(self, batch_size):
        """按 stratum/cause 两级轮转进行确定性种子、无放回采样。"""

        batch_size = _strict_integer(
            'batch_size',
            batch_size,
            minimum=1,
        )
        if batch_size > len(self):
            raise ValueError(
                f'cannot sample batch_size={batch_size} from causal '
                f'replay with {len(self)} items'
            )

        result = []
        stratum_order = (
            CAUSAL_STRATA[self._stratum_cursor:]
            + CAUSAL_STRATA[:self._stratum_cursor]
        )
        available_causes = {
            stratum: sorted(
                self._causes_by_stratum.get(stratum, ())
            )
            for stratum in CAUSAL_STRATA
        }
        bucket_draw_states = {}

        while len(result) < batch_size:
            made_progress = False
            for stratum in stratum_order:
                if len(result) >= batch_size:
                    break
                causes = available_causes[stratum]
                if not causes:
                    continue

                cursor = self._cause_cursors[stratum] % len(causes)
                cause = causes[cursor]
                self._cause_cursors[stratum] = cursor + 1
                bucket = stratum, cause
                dense_ids = self._bucket_entry_ids[bucket]
                draw_state = bucket_draw_states.get(bucket)
                if draw_state is None:
                    draw_state = {
                        'remaining': len(dense_ids),
                        'swaps': {},
                    }
                    bucket_draw_states[bucket] = draw_state

                remaining = draw_state['remaining']
                chosen_position = self._rng.randrange(remaining)
                last_position = remaining - 1
                swaps = draw_state['swaps']
                chosen_id = swaps.get(
                    chosen_position,
                    dense_ids[chosen_position],
                )
                if chosen_position != last_position:
                    swaps[chosen_position] = swaps.get(
                        last_position,
                        dense_ids[last_position],
                    )
                swaps.pop(last_position, None)
                draw_state['remaining'] = last_position
                result.append(self._items[chosen_id])
                if last_position == 0:
                    del causes[cursor]
                made_progress = True
            if not made_progress:
                break

        self._stratum_cursor = (
            self._stratum_cursor + 1
        ) % len(CAUSAL_STRATA)
        if len(result) != batch_size:
            raise RuntimeError('causal replay sampler made insufficient progress')
        return tuple(result)

    def clear(self):
        self._reset_storage()
        self._cause_cursors.clear()
        self._stratum_cursor = self._rng.randrange(len(CAUSAL_STRATA))
        self._counterfactual_override_count = 0
        self._ignored_weaker_rule_count = 0
        self._eviction_count = 0
        self._rule_empirical_agreement_count = 0
        self._rule_empirical_disagreement_count = 0

    def to_tuple(self):
        """按当前从旧到新的存储顺序返回只读快照。"""

        return tuple(self._items.values())

    def checkpoint_manifest(self):
        """返回恢复因果回放前必须一致的契约。"""

        return {
            'kind': 'causal-replay-buffer',
            'schema_version': 1,
            'sample_schema_version': CAUSAL_SAMPLE_SCHEMA_VERSION,
            'capacity': self.capacity,
        }

    def validate_checkpoint_manifest(self, manifest):
        """拒绝容量或样本契约不一致的恢复。"""

        expected = self.checkpoint_manifest()
        if not isinstance(manifest, dict):
            raise ValueError(
                'causal replay checkpoint manifest must be a mapping'
            )
        for field_name, expected_value in expected.items():
            if manifest.get(field_name) != expected_value:
                raise ValueError(
                    'causal replay checkpoint manifest mismatch for '
                    f'{field_name}: checkpoint={manifest.get(field_name)!r} '
                    f'current={expected_value!r}'
                )

    def checkpoint_state_dict(self):
        """精确保存稀疏因果样本、轮转游标、统计和独立 RNG。"""

        return {
            'schema_version': 1,
            'items': tuple(self._items.values()),
            'rng_state': self._rng.getstate(),
            'stratum_cursor': self._stratum_cursor,
            'cause_cursors': dict(self._cause_cursors),
            # dense bucket 采用 swap-remove；只保存 items 无法恢复其随机下标
            # 映射。索引状态不改变外部 schema，却是 checkpoint 后采样逐项一致
            # 所必需的内部状态。旧 checkpoint 没有该字段时按旧到新顺序重建。
            'index_state': {
                'schema_version': 1,
                'entry_ids': tuple(self._items),
                'next_entry_id': self._next_entry_id,
                'bucket_entry_ids': tuple(
                    (
                        stratum,
                        cause_type,
                        tuple(self._bucket_entry_ids[
                            (stratum, cause_type)
                        ]),
                    )
                    for stratum, cause_type in sorted(
                        self._bucket_entry_ids
                    )
                ),
            },
            'counterfactual_override_count': (
                self._counterfactual_override_count
            ),
            'ignored_weaker_rule_count': (
                self._ignored_weaker_rule_count
            ),
            'eviction_count': self._eviction_count,
            'rule_empirical_agreement_count': (
                self._rule_empirical_agreement_count
            ),
            'rule_empirical_disagreement_count': (
                self._rule_empirical_disagreement_count
            ),
        }

    @staticmethod
    def _normalize_checkpoint_index_state(items, index_state):
        """校验内部 entry/bucket 映射；None 表示兼容旧 checkpoint。"""

        if index_state is None:
            entry_ids = tuple(range(len(items)))
            return entry_ids, len(entry_ids), None
        if not isinstance(index_state, dict):
            raise ValueError(
                'causal replay index_state must be a mapping'
            )
        if index_state.get('schema_version') != 1:
            raise ValueError(
                'unsupported causal replay index schema version: '
                f'{index_state.get("schema_version")!r}'
            )

        raw_entry_ids = index_state.get('entry_ids')
        if not isinstance(raw_entry_ids, (tuple, list)):
            raise ValueError(
                'causal replay index entry_ids must be a sequence'
            )
        entry_ids = tuple(
            _strict_integer(
                f'index_state.entry_ids[{offset}]',
                entry_id,
                minimum=0,
            )
            for offset, entry_id in enumerate(raw_entry_ids)
        )
        if len(entry_ids) != len(items):
            raise ValueError(
                'causal replay index entry_ids must align with items'
            )
        if any(
                current <= previous
                for previous, current in zip(
                    entry_ids,
                    entry_ids[1:],
                )):
            raise ValueError(
                'causal replay index entry_ids must be strictly increasing'
            )
        next_entry_id = _strict_integer(
            'index_state.next_entry_id',
            index_state.get('next_entry_id'),
            minimum=0,
        )
        if entry_ids and next_entry_id <= entry_ids[-1]:
            raise ValueError(
                'causal replay next_entry_id must exceed all entry ids'
            )

        raw_buckets = index_state.get('bucket_entry_ids')
        if not isinstance(raw_buckets, (tuple, list)):
            raise ValueError(
                'causal replay bucket_entry_ids must be a sequence'
            )
        samples_by_id = dict(zip(entry_ids, items))
        expected_ids = set(entry_ids)
        seen_ids = set()
        bucket_layout = {}
        for offset, record in enumerate(raw_buckets):
            if (
                    not isinstance(record, (tuple, list))
                    or len(record) != 3):
                raise ValueError(
                    'causal replay bucket record must contain '
                    '(stratum, cause_type, entry_ids)'
                )
            stratum, cause_type, raw_bucket_ids = record
            if stratum not in CAUSAL_STRATA:
                raise ValueError(
                    f'unknown causal replay bucket stratum: {stratum!r}'
                )
            cause_type = _non_empty_string(
                f'index_state.bucket_entry_ids[{offset}].cause_type',
                cause_type,
            )
            bucket = stratum, cause_type
            if bucket in bucket_layout:
                raise ValueError(
                    f'duplicate causal replay bucket: {bucket!r}'
                )
            if not isinstance(raw_bucket_ids, (tuple, list)):
                raise ValueError(
                    'causal replay bucket ids must be a sequence'
                )
            bucket_ids = tuple(
                _strict_integer(
                    'causal replay bucket entry id',
                    entry_id,
                    minimum=0,
                )
                for entry_id in raw_bucket_ids
            )
            if not bucket_ids:
                raise ValueError(
                    'causal replay checkpoint cannot contain empty buckets'
                )
            if len(set(bucket_ids)) != len(bucket_ids):
                raise ValueError(
                    'causal replay bucket entry ids must be unique'
                )
            for entry_id in bucket_ids:
                if entry_id not in expected_ids:
                    raise ValueError(
                        'causal replay bucket references an unknown entry'
                    )
                if entry_id in seen_ids:
                    raise ValueError(
                        'causal replay entry appears in multiple buckets'
                    )
                sample = samples_by_id[entry_id]
                if (sample.stratum, sample.cause_type) != bucket:
                    raise ValueError(
                        'causal replay bucket does not match its sample'
                    )
                seen_ids.add(entry_id)
            bucket_layout[bucket] = bucket_ids
        if seen_ids != expected_ids:
            raise ValueError(
                'causal replay bucket index does not cover every item'
            )
        return entry_ids, next_entry_id, bucket_layout

    def _restore_storage(
            self,
            items,
            *,
            entry_ids,
            next_entry_id,
            bucket_layout):
        """从已校验内容重建派生索引，并恢复 dense bucket 的精确顺序。"""

        self._reset_storage()
        for entry_id, sample in zip(entry_ids, items):
            self._items[entry_id] = sample
            self._index_entry(entry_id, sample)
        self._next_entry_id = next_entry_id
        if bucket_layout is None:
            return

        dense_buckets = defaultdict(list)
        positions = defaultdict(dict)
        for bucket, bucket_ids in bucket_layout.items():
            dense_buckets[bucket] = list(bucket_ids)
            positions[bucket] = {
                entry_id: offset
                for offset, entry_id in enumerate(bucket_ids)
            }
        self._bucket_entry_ids = dense_buckets
        self._bucket_positions = positions

    def load_checkpoint_state_dict(self, state):
        """原子校验后恢复因果 replay 的全部可变状态。"""

        if not isinstance(state, dict):
            raise ValueError(
                'causal replay checkpoint state must be a mapping'
            )
        if state.get('schema_version') != 1:
            raise ValueError(
                'unsupported causal replay checkpoint schema version: '
                f'{state.get("schema_version")!r}'
            )
        items = tuple(state.get('items', ()))
        if len(items) > self.capacity:
            raise ValueError(
                'causal replay checkpoint contains more than capacity'
            )
        if any(not isinstance(item, CausalSample) for item in items):
            raise TypeError(
                'causal replay checkpoint items must be CausalSample'
            )
        (
            entry_ids,
            next_entry_id,
            bucket_layout,
        ) = self._normalize_checkpoint_index_state(
            items,
            state.get('index_state'),
        )
        stratum_cursor = _strict_integer(
            'stratum_cursor',
            state.get('stratum_cursor'),
            minimum=0,
            maximum=len(CAUSAL_STRATA) - 1,
        )
        cause_cursors = state.get('cause_cursors', {})
        if not isinstance(cause_cursors, dict):
            raise ValueError('cause_cursors must be a mapping')
        normalized_cursors = {}
        for stratum, cursor in cause_cursors.items():
            if stratum not in CAUSAL_STRATA:
                raise ValueError(
                    f'unknown causal stratum cursor: {stratum!r}'
                )
            normalized_cursors[stratum] = _strict_integer(
                f'cause_cursors[{stratum!r}]',
                cursor,
                minimum=0,
            )
        counters = {}
        for field_name in (
                'counterfactual_override_count',
                'ignored_weaker_rule_count',
                'eviction_count',
                'rule_empirical_agreement_count',
                'rule_empirical_disagreement_count'):
            counters[field_name] = _strict_integer(
                field_name,
                state.get(field_name, 0),
                minimum=0,
            )
        try:
            rng_state = state['rng_state']
            probe_rng = random.Random()
            probe_rng.setstate(rng_state)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                'invalid causal replay RNG checkpoint state'
            ) from exc

        # 派生索引也先在临时对象中完整构建；任一错误都不会破坏当前 replay。
        restored = CausalReplayBuffer(
            capacity=self.capacity,
            seed=self._seed,
        )
        restored._restore_storage(
            items,
            entry_ids=entry_ids,
            next_entry_id=next_entry_id,
            bucket_layout=bucket_layout,
        )
        restored._rng.setstate(rng_state)
        restored._stratum_cursor = stratum_cursor
        restored._cause_cursors = defaultdict(int, normalized_cursors)
        restored._counterfactual_override_count = counters[
            'counterfactual_override_count'
        ]
        restored._ignored_weaker_rule_count = counters[
            'ignored_weaker_rule_count'
        ]
        restored._eviction_count = counters['eviction_count']
        restored._rule_empirical_agreement_count = counters[
            'rule_empirical_agreement_count'
        ]
        restored._rule_empirical_disagreement_count = counters[
            'rule_empirical_disagreement_count'
        ]
        self.__dict__.update(restored.__dict__)

    @property
    def storage_stats(self):
        """返回轻量、可 JSON 序列化的分层存储统计。"""

        return {
            'mode': 'memory',
            'capacity': self.capacity,
            'total_count': len(self),
            'remaining_capacity': self.remaining_capacity,
            'stratum_counts': {
                stratum: self._stratum_counts.get(stratum, 0)
                for stratum in CAUSAL_STRATA
            },
            'cause_type_counts': dict(sorted(
                self._cause_type_counts.items()
            )),
            'supervision_kind_counts': {
                kind: self._supervision_kind_counts.get(kind, 0)
                for kind in CAUSAL_SUPERVISION_KINDS
            },
            'counterfactual_override_count': (
                self._counterfactual_override_count
            ),
            'ignored_weaker_rule_count': self._ignored_weaker_rule_count,
            'eviction_count': self._eviction_count,
            'rule_empirical_agreement_count': (
                self._rule_empirical_agreement_count
            ),
            'rule_empirical_disagreement_count': (
                self._rule_empirical_disagreement_count
            ),
            **_graph_storage_stats(
                item.graph
                for item in self._items.values()
            ),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class CausalTransitionContext:
    """worker 内等待延迟归因回查的一条动作前上下文。"""

    graph: GraphTensor
    state_analysis: StateAnalysis
    actual_action_offset: int
    actual_action_index: int
    policy_version: str | None = None
    graph_schema_fingerprint: str | None = None

    def __post_init__(self):
        _validate_graph_tensor(self.graph)
        if not isinstance(self.state_analysis, StateAnalysis):
            raise TypeError('state_analysis must be StateAnalysis')
        if self.graph.action_count != ANALYSIS_ACTION_COUNT:
            raise ValueError(
                'causal attribution context requires exactly '
                f'{ANALYSIS_ACTION_COUNT} graph actions'
            )
        graph_action_indices = tuple(
            int(value)
            for value in self.graph.action_indices.tolist()
        )
        if graph_action_indices != self.state_analysis.action_indices:
            raise ValueError(
                'graph action mapping must match StateAnalysis.action_indices'
            )

        actual_offset = _strict_integer(
            'actual_action_offset',
            self.actual_action_offset,
            minimum=0,
            maximum=self.graph.action_count - 1,
        )
        actual_index = _strict_integer(
            'actual_action_index',
            self.actual_action_index,
            minimum=0,
        )
        expected_action_index = graph_action_indices[actual_offset]
        if actual_index != expected_action_index:
            raise ValueError(
                'actual_action_index must match graph action mapping'
            )
        object.__setattr__(
            self,
            'actual_action_offset',
            actual_offset,
        )
        object.__setattr__(
            self,
            'actual_action_index',
            actual_index,
        )
        object.__setattr__(
            self,
            'policy_version',
            _optional_string('policy_version', self.policy_version),
        )

        expected_fingerprint = graph_schema_fingerprint(self.graph)
        if self.graph_schema_fingerprint is None:
            object.__setattr__(
                self,
                'graph_schema_fingerprint',
                expected_fingerprint,
            )
        else:
            supplied = _non_empty_string(
                'graph_schema_fingerprint',
                self.graph_schema_fingerprint,
            )
            if supplied != expected_fingerprint:
                raise ValueError(
                    'graph_schema_fingerprint does not match graph schema'
                )
            object.__setattr__(
                self,
                'graph_schema_fingerprint',
                supplied,
            )

    @property
    def transition_key(self):
        return self.state_analysis.transition_key


class RuleCausalContextCache:
    """worker-local、固定容量的 ``TransitionKey -> context`` 历史缓存。"""

    def __init__(self, capacity=512):
        self.capacity = _strict_integer(
            'capacity',
            capacity,
            minimum=1,
        )
        self._contexts = OrderedDict()
        self._eviction_count = 0

    def __len__(self):
        return len(self._contexts)

    def put(self, context):
        if not isinstance(context, CausalTransitionContext):
            raise TypeError(
                'context must be CausalTransitionContext'
            )
        key = context.transition_key
        if key in self._contexts:
            del self._contexts[key]
        self._contexts[key] = context
        while len(self._contexts) > self.capacity:
            self._contexts.popitem(last=False)
            self._eviction_count += 1
        return context

    def remember(
            self,
            *,
            graph,
            state_analysis,
            actual_action_offset,
            actual_action_index,
            policy_version=None,
            graph_schema_fingerprint=None):
        context = CausalTransitionContext(
            graph=graph,
            state_analysis=state_analysis,
            actual_action_offset=actual_action_offset,
            actual_action_index=actual_action_index,
            policy_version=policy_version,
            graph_schema_fingerprint=graph_schema_fingerprint,
        )
        return self.put(context)

    def get(self, transition_key, default=None):
        if not isinstance(transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        return self._contexts.get(transition_key, default)

    def discard_episode(self, worker_id, episode_id):
        worker_id = _strict_integer('worker_id', worker_id, minimum=0)
        episode_id = _strict_integer('episode_id', episode_id, minimum=0)
        keys = tuple(
            key
            for key in self._contexts
            if (
                key.worker_id == worker_id
                and key.episode_id == episode_id
            )
        )
        for key in keys:
            del self._contexts[key]
        return len(keys)

    def clear(self):
        self._contexts.clear()

    def to_tuple(self):
        return tuple(self._contexts.values())

    @property
    def storage_stats(self):
        return {
            'capacity': self.capacity,
            'context_count': len(self),
            'remaining_capacity': max(0, self.capacity - len(self)),
            'eviction_count': self._eviction_count,
        }


# 明确的长名称便于调用方读懂用途；短名称保留给本模块文档中的 context cache。
RuleCausalSampleContextCache = RuleCausalContextCache


@dataclass(frozen=True, slots=True, kw_only=True)
class RuleCausalBuildStats:
    """一次规则样本构建的不可变统计。"""

    input_event_count: int
    eligible_event_count: int
    budget_count: int
    generated_sample_count: int
    reason_counts: tuple[tuple[str, int], ...]

    def reason_count(self, reason):
        reason = _non_empty_string('reason', reason)
        return dict(self.reason_counts).get(reason, 0)

    def to_dict(self):
        return {
            'input_event_count': self.input_event_count,
            'eligible_event_count': self.eligible_event_count,
            'budget_count': self.budget_count,
            'generated_sample_count': self.generated_sample_count,
            'reason_counts': dict(self.reason_counts),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class RuleCausalBuildResult:
    samples: tuple[CausalSample, ...]
    stats: RuleCausalBuildStats


@dataclass(frozen=True, slots=True)
class _ContributorCandidate:
    transition_key: TransitionKey
    action_offset: int
    action_index: int
    contribution_weight: float
    confidence: float
    event: AttributionEvent


class RuleCausalSampleBuilder:
    """把 confirmed A/B 事件按预算合并为规则 Q 排序样本。

    规则比较动作保持保守：

    - 正向铺垫只使用左右镜像动作；中心动作没有不同镜像时跳过；
    - 负向阻挡使用 q0～q3 队列容量加权后严格更安全的动作；
    - 没有历史 context、分析诊断无效或比较动作没有可信优势时只计原因，不造标签。
    """

    def __init__(self, *, context_cache=None, context_capacity=512):
        if context_cache is None:
            context_cache = RuleCausalContextCache(context_capacity)
        if not isinstance(context_cache, RuleCausalContextCache):
            raise TypeError(
                'context_cache must be RuleCausalContextCache'
            )
        self.context_cache = context_cache
        self._build_calls = 0
        self._generated_sample_count = 0
        self._cumulative_reasons = Counter()
        self._last_stats = RuleCausalBuildStats(
            input_event_count=0,
            eligible_event_count=0,
            budget_count=0,
            generated_sample_count=0,
            reason_counts=(),
        )

    def remember_context(self, context):
        return self.context_cache.put(context)

    def remember_transition(self, **kwargs):
        return self.context_cache.remember(**kwargs)

    @property
    def last_stats(self):
        return self._last_stats

    @property
    def stats(self):
        return {
            'build_calls': self._build_calls,
            'generated_sample_count': self._generated_sample_count,
            'reason_counts': dict(sorted(self._cumulative_reasons.items())),
            'context_cache': self.context_cache.storage_stats,
        }

    def build(self, events):
        """构建并仅返回样本 tuple；详细跳过原因保存在 ``last_stats``。"""

        return self.build_with_stats(events).samples

    def build_with_stats(self, events):
        events = tuple(events)
        if any(not isinstance(event, AttributionEvent) for event in events):
            raise TypeError('events must contain AttributionEvent values')

        reasons = Counter()
        confirmed = []
        eligible_event_count = 0
        for event in events:
            if event.status != 'confirmed':
                reasons['event_not_confirmed'] += 1
                continue
            # C 级事件本身不能生成规则标签，但 MERGE_LINEAGE 可能是同一
            # budget 唯一携带非零合成效用的价值包。仍把它留在 budget group
            # 中供读取 utility，后续只从 A/B 事件提取 contributor/comparator。
            confirmed.append(event)
            if event.confidence_tier == 'C':
                reasons['confidence_tier_c'] += 1
                continue
            if not event.contributors:
                reasons['no_contributors'] += 1
                continue
            eligible_event_count += 1

        budget_groups = defaultdict(list)
        for event in confirmed:
            budget_groups[stable_budget_key(event.budget_key)].append(event)

        samples = []
        for budget_string in sorted(budget_groups):
            group_samples = self._build_budget_group(
                budget_string,
                tuple(budget_groups[budget_string]),
                reasons,
            )
            samples.extend(group_samples)

        stats = RuleCausalBuildStats(
            input_event_count=len(events),
            eligible_event_count=eligible_event_count,
            budget_count=len(budget_groups),
            generated_sample_count=len(samples),
            reason_counts=tuple(sorted(reasons.items())),
        )
        self._last_stats = stats
        self._build_calls += 1
        self._generated_sample_count += len(samples)
        self._cumulative_reasons.update(reasons)
        return RuleCausalBuildResult(
            samples=tuple(samples),
            stats=stats,
        )

    def _build_budget_group(self, budget_string, events, reasons):
        signs = {event.sign for event in events}
        if len(signs) != 1:
            reasons['mixed_budget_sign'] += 1
            return ()
        versions = {event.attribution_version for event in events}
        if len(versions) != 1:
            reasons['mixed_attribution_version'] += 1
            return ()
        tracker_fingerprints = {
            event.tracker_config_fingerprint
            for event in events
        }
        if len(tracker_fingerprints) != 1:
            reasons['mixed_tracker_fingerprint'] += 1
            return ()

        utility = max(event.utility for event in events)
        if utility <= 0.0:
            reasons['zero_budget_utility'] += 1
            return ()
        sign = next(iter(signs))
        if sign < 0:
            utility = min(utility, merge_utility(5))

        label_events = tuple(
            event
            for event in events
            if (
                event.confidence_tier != 'C'
                and event.contributors
            )
        )
        if not label_events:
            reasons['no_ab_label_event'] += 1
            return ()

        candidates = self._merge_contributors(label_events)
        if not candidates:
            reasons['no_contributors'] += 1
            return ()

        weight_sum = sum(
            candidate.contribution_weight
            for candidate in candidates
        )
        if weight_sum <= 0.0:
            reasons['zero_contribution_weight'] += 1
            return ()

        result = []
        for candidate in candidates:
            context = self.context_cache.get(candidate.transition_key)
            if context is None:
                reasons['missing_context'] += 1
                continue
            if not context.state_analysis.diagnostics.valid_for_attribution:
                reasons['invalid_context_analysis'] += 1
                continue
            if (
                    context.actual_action_offset != candidate.action_offset
                    or context.actual_action_index != candidate.action_index):
                reasons['context_action_mismatch'] += 1
                continue

            if sign > 0:
                comparison = self._positive_comparison(context)
                stratum = 'positive_setup'
            else:
                comparison = self._negative_comparison(context)
                stratum = 'negative_blocking'
            if comparison is None:
                reasons['no_trustworthy_comparison'] += 1
                continue

            normalized_weight = (
                candidate.contribution_weight / weight_sum
            )
            target_margin = (
                utility
                * normalized_weight
                * candidate.confidence
            )
            if target_margin <= 0.0:
                reasons['zero_target_margin'] += 1
                continue

            event = candidate.event
            result.append(CausalSample(
                graph=context.graph,
                actual_action_offset=candidate.action_offset,
                comparison_action_offset=comparison,
                direction=sign,
                target_margin=target_margin,
                confidence=candidate.confidence,
                cause_type=event.event_type,
                delay=event.delay,
                transition_key=candidate.transition_key,
                attribution_version=event.attribution_version,
                supervision_kind='rule',
                stratum=stratum,
                event_key=stable_event_key(event.event_id),
                budget_key=budget_string,
                policy_version=context.policy_version,
                tracker_config_fingerprint=(
                    event.tracker_config_fingerprint
                ),
                analyzer_config_fingerprint=(
                    context.state_analysis.analyzer_config_fingerprint
                ),
                graph_schema_fingerprint=(
                    context.graph_schema_fingerprint
                ),
            ))
        return tuple(result)

    @staticmethod
    def _merge_contributors(events):
        """按历史动作合并同预算不同标签，防止同一动作重复消耗价值包。"""

        grouped = defaultdict(list)
        for event in events:
            confidence = min(
                event.link_confidence,
                event.placement_confidence,
            )
            for contributor in event.contributors:
                key = (
                    contributor.transition_key,
                    contributor.action_offset,
                    contributor.action_index,
                )
                grouped[key].append((
                    contributor.contribution_weight,
                    confidence,
                    event,
                ))

        candidates = []
        for (
                transition_key,
                action_offset,
                action_index,
        ), evidence in grouped.items():
            contribution_weight = max(
                item[0]
                for item in evidence
            )
            # 优先保留综合置信度和贡献权重更高的事件；最后用 event key
            # 确定性打破平局，避免 set/dict 顺序影响 cause_type。
            selected_weight, confidence, event = max(
                evidence,
                key=lambda item: (
                    item[1],
                    item[0],
                    item[2].event_id,
                ),
            )
            del selected_weight
            candidates.append(_ContributorCandidate(
                transition_key=transition_key,
                action_offset=action_offset,
                action_index=action_index,
                contribution_weight=contribution_weight,
                confidence=confidence,
                event=event,
            ))
        return tuple(sorted(
            candidates,
            key=lambda item: (
                item.transition_key,
                item.action_offset,
                item.action_index,
            ),
        ))

    @staticmethod
    def _positive_comparison(context):
        action_count = context.graph.action_count
        mirror_offset = (
            action_count - 1 - context.actual_action_offset
        )
        if mirror_offset == context.actual_action_offset:
            return None
        return mirror_offset

    @staticmethod
    def _action_safety_scores(analysis):
        """按 analyzer 的 q0～q3 衰减和 0.7/0.3 公式计算逐动作安全度。"""

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
            for action_offset in range(ANALYSIS_ACTION_COUNT)
        )

    @classmethod
    def _negative_comparison(cls, context):
        scores = cls._action_safety_scores(context.state_analysis)
        actual_offset = context.actual_action_offset
        alternatives = tuple(
            offset
            for offset in range(len(scores))
            if offset != actual_offset
        )
        if not alternatives:
            return None
        best_score = max(scores[offset] for offset in alternatives)
        if best_score <= scores[actual_offset] + 1e-12:
            return None

        best_offsets = tuple(
            offset
            for offset in alternatives
            if math.isclose(
                scores[offset],
                best_score,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        mirror_offset = len(scores) - 1 - actual_offset
        if mirror_offset in best_offsets:
            return mirror_offset
        # 距离实际动作更近的安全替代通常需要更小的行为改动；再以 offset
        # 确定性打破平局。
        return min(
            best_offsets,
            key=lambda offset: (
                abs(offset - actual_offset),
                offset,
            ),
        )


__all__ = [
    'CAUSAL_SAMPLE_SCHEMA_VERSION',
    'CAUSAL_STRATA',
    'CAUSAL_SUPERVISION_KINDS',
    'CausalReplayBuffer',
    'CausalSample',
    'CausalTransitionContext',
    'RuleCausalBuildResult',
    'RuleCausalBuildStats',
    'RuleCausalContextCache',
    'RuleCausalSampleBuilder',
    'RuleCausalSampleContextCache',
    'graph_schema_fingerprint',
    'stable_budget_key',
    'stable_event_key',
]
