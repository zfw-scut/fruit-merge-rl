"""动作后结构监督的紧凑、无未来泄漏数据契约。

结构 target 只描述当前动作 ``t`` 已经观测到的结果：

``analysis[t] -> analysis[t + 1]``、本步 ``PhysicsResult`` / ``MergeEvent``
以及本步是否真实 terminal。它不接收后续轨迹、延迟归因事件、target Q 或 episode
回报，因此不能把未来信息泄漏到当前动作监督。

六维值统一限制在 ``[-1, 1]``，不可用维度必须为零并由 6-bit ``valid_mask``
屏蔽。该对象还提供 14-byte float16 replay payload，方便后续冷 replay 在不保存
六个 Python float 对象的情况下落盘。
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from operator import index
from typing import TYPE_CHECKING

from daxigua.core.rules import (
    MAX_FRUIT_LEVEL,
    merge_score,
)
from daxigua.core.state import MergeEvent, PhysicsResult

if TYPE_CHECKING:
    from daxigua_rl.attribution.schema import StateAnalysis


STRUCTURAL_TARGET_SCHEMA_VERSION = 1
STRUCTURAL_TARGET_DIMENSIONS = (
    'top_connected_capacity_delta',
    'recoverability_delta',
    'chain_readiness_delta',
    'new_dead_or_blocked_fruit_risk',
    'sealed_cavity_delta',
    'realized_chain_or_terminal_risk',
)
STRUCTURAL_TARGET_DIMENSION_COUNT = len(STRUCTURAL_TARGET_DIMENSIONS)
STRUCTURAL_TARGET_FULL_VALID_MASK = (
    1 << STRUCTURAL_TARGET_DIMENSION_COUNT
) - 1

_HALF_PAYLOAD_FORMAT = (
    '<BB'
    + 'e' * STRUCTURAL_TARGET_DIMENSION_COUNT
)
STRUCTURAL_TARGET_HALF_PAYLOAD_BYTES = struct.calcsize(
    _HALF_PAYLOAD_FORMAT
)


def _finite_float(name, value):
    if isinstance(value, bool):
        raise TypeError(f'{name} must be a real number')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f'{name} must be a real number') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _strict_integer(name, value, *, minimum=None):
    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer')
    try:
        normalized = index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc
    if minimum is not None and normalized < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return normalized


def _clip_unit(value):
    return max(0.0, min(1.0, float(value)))


def _clip_signed_unit(value):
    return max(-1.0, min(1.0, float(value)))


def _merge_utility(new_level):
    # reward 会经 attribution 包导入 training；延迟到构建监督时再取用，避免
    # ``training.__init__ -> TensorTransition -> structural_targets`` 导入环。
    from daxigua_rl.reward import merge_utility

    return merge_utility(new_level)


@dataclass(frozen=True, slots=True)
class StructuralTarget:
    """六维动作后结构 target 与逐维有效掩码。

    ``valid_mask`` 的第 ``i`` 位对应
    ``STRUCTURAL_TARGET_DIMENSIONS[i]``。无效维度强制为零，避免调用方忘记应用
    mask 时读到未验证值。
    """

    values: tuple[float, ...]
    valid_mask: int
    schema_version: int = STRUCTURAL_TARGET_SCHEMA_VERSION

    def __post_init__(self):
        values = tuple(
            _finite_float(f'values[{offset}]', value)
            for offset, value in enumerate(self.values)
        )
        if len(values) != STRUCTURAL_TARGET_DIMENSION_COUNT:
            raise ValueError(
                'values must contain exactly '
                f'{STRUCTURAL_TARGET_DIMENSION_COUNT} items'
            )
        if any(value < -1.0 or value > 1.0 for value in values):
            raise ValueError('structural target values must lie in [-1, 1]')

        valid_mask = _strict_integer('valid_mask', self.valid_mask)
        if valid_mask < 0 or valid_mask > STRUCTURAL_TARGET_FULL_VALID_MASK:
            raise ValueError(
                'valid_mask contains bits outside the structural target'
            )
        invalid_nonzero = tuple(
            STRUCTURAL_TARGET_DIMENSIONS[offset]
            for offset, value in enumerate(values)
            if not valid_mask & (1 << offset) and value != 0.0
        )
        if invalid_nonzero:
            raise ValueError(
                'invalid structural target dimensions must be zero: '
                + ', '.join(invalid_nonzero)
            )

        schema_version = _strict_integer(
            'schema_version',
            self.schema_version,
            minimum=1,
        )
        if schema_version != STRUCTURAL_TARGET_SCHEMA_VERSION:
            raise ValueError(
                f'unsupported structural target schema: {schema_version}'
            )
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'valid_mask', valid_mask)
        object.__setattr__(self, 'schema_version', schema_version)

    @classmethod
    def empty(cls):
        """返回没有任何可信维度的兼容占位对象。"""

        return cls(
            values=(0.0,) * STRUCTURAL_TARGET_DIMENSION_COUNT,
            valid_mask=0,
        )

    @classmethod
    def from_validity(cls, values, validity):
        """由六个值和六个 bool 构造 bit mask，并清零无效维度。"""

        values = tuple(values)
        validity = tuple(validity)
        if len(validity) != STRUCTURAL_TARGET_DIMENSION_COUNT:
            raise ValueError(
                'validity must contain exactly '
                f'{STRUCTURAL_TARGET_DIMENSION_COUNT} items'
            )
        if any(not isinstance(flag, bool) for flag in validity):
            raise TypeError('validity must contain bool values')
        if len(values) != STRUCTURAL_TARGET_DIMENSION_COUNT:
            raise ValueError(
                'values must contain exactly '
                f'{STRUCTURAL_TARGET_DIMENSION_COUNT} items'
            )
        valid_mask = sum(
            1 << offset
            for offset, flag in enumerate(validity)
            if flag
        )
        return cls(
            values=tuple(
                value if validity[offset] else 0.0
                for offset, value in enumerate(values)
            ),
            valid_mask=valid_mask,
        )

    @property
    def validity(self):
        """按六维固定顺序返回 bool mask。"""

        return tuple(
            bool(self.valid_mask & (1 << offset))
            for offset in range(STRUCTURAL_TARGET_DIMENSION_COUNT)
        )

    @property
    def has_valid_values(self):
        return self.valid_mask != 0

    def is_valid(self, dimension):
        """查询名称或整数下标对应维度是否有效。"""

        if isinstance(dimension, str):
            try:
                dimension = STRUCTURAL_TARGET_DIMENSIONS.index(dimension)
            except ValueError as exc:
                raise KeyError(dimension) from exc
        dimension = _strict_integer('dimension', dimension)
        if not 0 <= dimension < STRUCTURAL_TARGET_DIMENSION_COUNT:
            raise IndexError('structural target dimension out of range')
        return bool(self.valid_mask & (1 << dimension))

    def to_half_bytes(self):
        """编码为 ``schema + mask + 6*float16`` 的 14-byte payload。"""

        return struct.pack(
            _HALF_PAYLOAD_FORMAT,
            self.schema_version,
            self.valid_mask,
            *self.values,
        )

    @classmethod
    def from_half_bytes(cls, payload):
        """从 :meth:`to_half_bytes` 产生的紧凑 payload 恢复。"""

        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError('payload must be bytes-like')
        payload = bytes(payload)
        if len(payload) != STRUCTURAL_TARGET_HALF_PAYLOAD_BYTES:
            raise ValueError(
                'structural target half payload must contain exactly '
                f'{STRUCTURAL_TARGET_HALF_PAYLOAD_BYTES} bytes'
            )
        schema_version, valid_mask, *values = struct.unpack(
            _HALF_PAYLOAD_FORMAT,
            payload,
        )
        return cls(
            values=tuple(values),
            valid_mask=valid_mask,
            schema_version=schema_version,
        )


def _validate_analysis_pair(
        previous_analysis: StateAnalysis | None,
        next_analysis: StateAnalysis | None):
    # 同上，类型只在真正构建 target 时检查；模块导入阶段不能反向初始化
    # attribution 包。
    from daxigua_rl.attribution.schema import StateAnalysis

    for name, analysis in (
            ('previous_analysis', previous_analysis),
            ('next_analysis', next_analysis)):
        if analysis is not None and not isinstance(analysis, StateAnalysis):
            raise TypeError(f'{name} must be StateAnalysis or None')
    if previous_analysis is None or next_analysis is None:
        return False

    previous_key = previous_analysis.transition_key
    next_key = next_analysis.transition_key
    if (
            previous_key.worker_id != next_key.worker_id
            or previous_key.episode_id != next_key.episode_id
            or previous_key.step_index + 1 != next_key.step_index):
        raise ValueError(
            'next_analysis must be the immediately following state in '
            'the same worker and episode'
        )
    if (
            next_analysis.incoming_transition_key is not None
            and next_analysis.incoming_transition_key != previous_key):
        raise ValueError(
            'next_analysis.incoming_transition_key must match '
            'previous_analysis'
        )
    has_transition_provenance = (
        next_analysis.incoming_transition_key == previous_key
    )
    if (
            previous_analysis.analyzer_config_fingerprint
            != next_analysis.analyzer_config_fingerprint):
        raise ValueError(
            'adjacent analyses must use the same analyzer config'
        )

    return has_transition_provenance and all(
        analysis.diagnostics.stable_boundary
        and analysis.diagnostics.valid_for_attribution
        and not analysis.diagnostics.degraded
        for analysis in (previous_analysis, next_analysis)
    )


def _validated_merge_events(merge_events, *, expected_score_delta=None):
    try:
        merge_events = tuple(merge_events)
    except TypeError as exc:
        raise TypeError('merge_events must be iterable') from exc
    if any(not isinstance(event, MergeEvent) for event in merge_events):
        raise TypeError('merge_events must contain MergeEvent values')

    new_fruit_ids = []
    score_total = 0
    for event_offset, event in enumerate(merge_events):
        new_level = _strict_integer(
            f'merge_events[{event_offset}].new_level',
            event.new_level,
        )
        # 同时验证等级处于 Reward V2 可合成范围。
        _merge_utility(new_level)
        score_delta = _strict_integer(
            f'merge_events[{event_offset}].score_delta',
            event.score_delta,
        )
        if score_delta != merge_score(new_level):
            raise ValueError(
                'merge event score_delta does not match its new_level'
            )
        source_ids = tuple(
            _strict_integer(
                (
                    f'merge_events[{event_offset}]'
                    f'.source_ids[{source_offset}]'
                ),
                source_id,
                minimum=1,
            )
            for source_offset, source_id in enumerate(event.source_ids)
        )
        if len(source_ids) != 2 or source_ids[0] == source_ids[1]:
            raise ValueError(
                'merge event must contain two distinct source_ids'
            )
        new_fruit_id = _strict_integer(
            f'merge_events[{event_offset}].new_fruit_id',
            event.new_fruit_id,
            minimum=1,
        )
        if new_fruit_id in source_ids:
            raise ValueError(
                'merge event new_fruit_id must differ from source_ids'
            )
        new_fruit_ids.append(new_fruit_id)
        score_total += score_delta

    if len(set(new_fruit_ids)) != len(new_fruit_ids):
        raise ValueError(
            'merge_events contains duplicate new_fruit_id values'
        )
    if expected_score_delta is not None:
        expected_score_delta = _strict_integer(
            'physics_result.score_delta',
            expected_score_delta,
        )
        if score_total != expected_score_delta:
            raise ValueError(
                'physics_result.score_delta must equal merge event scores'
            )
    return merge_events


def _resolve_action_evidence(
        *,
        physics_result,
        merge_events,
        terminated):
    """规范本步物理证据；没有证据时返回无效而不是猜测零结果。"""

    if physics_result is not None:
        if not isinstance(physics_result, PhysicsResult):
            raise TypeError('physics_result must be PhysicsResult or None')
        if merge_events is not None:
            raise ValueError(
                'provide physics_result or merge_events, not both'
            )
        if (
                not isinstance(physics_result.done, bool)
                or not isinstance(physics_result.truncated, bool)):
            raise TypeError(
                'physics_result done/truncated flags must be bool'
            )
        if physics_result.done and physics_result.truncated:
            raise ValueError(
                'physics_result cannot be both done and truncated'
            )
        resolved_terminated = physics_result.done
        if terminated is not None:
            if not isinstance(terminated, bool):
                raise TypeError('terminated must be bool or None')
            if terminated != resolved_terminated:
                raise ValueError(
                    'terminated must match physics_result.done'
                )
        events = _validated_merge_events(
            physics_result.merge_events,
            expected_score_delta=physics_result.score_delta,
        )
        return events, resolved_terminated, True

    if terminated is not None and not isinstance(terminated, bool):
        raise TypeError('terminated must be bool or None')
    if merge_events is not None and terminated is None:
        raise ValueError(
            'explicit merge_events require an explicit terminated flag'
        )
    if merge_events is None:
        # ``terminated=True`` 自身足以确认终局风险；非终局标志却不能证明
        # “没有发生合成”，因此不能据此制造零连锁标签。
        if terminated is True:
            return (), True, True
        return (), False, False
    events = _validated_merge_events(merge_events)
    return events, bool(terminated), True


def _is_dead_or_blocked(fruit):
    return (
        fruit.reachable_action_count == 0
        and not fruit.partner_reachable
    )


def _new_dead_or_blocked_risk(previous_analysis, next_analysis):
    """量化本步新形成的死果或投放路径损失，不使用水果年龄。"""

    previous_by_id = {
        fruit.fruit_id: fruit
        for fruit in previous_analysis.fruit_analyses
    }
    action_count = max(1, next_analysis.action_count)
    maximum_risk = 0.0
    for next_fruit in next_analysis.fruit_analyses:
        previous_fruit = previous_by_id.get(next_fruit.fruit_id)
        now_dead = _is_dead_or_blocked(next_fruit)
        if previous_fruit is None:
            # 新投放或新合成水果只有立即成为死果时才计风险；不能把正常加入
            # 场景本身当作负监督。
            severity = 1.0 if now_dead else 0.0
        else:
            lost_fraction = max(
                0.0,
                (
                    previous_fruit.reachable_action_count
                    - next_fruit.reachable_action_count
                )
                / action_count,
            )
            newly_dead = (
                now_dead
                and not _is_dead_or_blocked(previous_fruit)
            )
            severity = 1.0 if newly_dead else lost_fraction
        if severity <= 0.0:
            continue

        # 低级水果和深埋水果更难处理；仍给大型水果保留至少 25% 权重。
        level_weight = 1.0 - 0.75 * (
            (next_fruit.level - 1) / (MAX_FRUIT_LEVEL - 1)
        )
        burial_weight = 0.5 + 0.5 * next_fruit.burial_depth
        partner_weight = (
            0.5
            if next_fruit.partner_reachable
            else 1.0
        )
        maximum_risk = max(
            maximum_risk,
            severity
            * level_weight
            * burial_weight
            * partner_weight,
        )
    return _clip_unit(maximum_risk)


def _sealed_cavity_ratio(analysis):
    return _clip_unit(sum(
        region.area_ratio
        for region in analysis.free_space_regions
        if region.sealed
    ))


def _realized_chain_signal(merge_events, terminated):
    """同一动作内由 merge ID 谱系计算连锁深度和效用。"""

    if terminated:
        # 单一复用维度中，真实终局风险优先于同一步可能发生的正合成。
        return -1.0

    depth_by_fruit_id = {}
    lineage_utility_by_fruit_id = {}
    max_depth = 0
    max_chain_utility = 0.0
    for event in merge_events:
        parent_depth = max(
            (
                depth_by_fruit_id.get(source_id, 0)
                for source_id in event.source_ids
            ),
            default=0,
        )
        depth = parent_depth + 1
        depth_by_fruit_id[event.new_fruit_id] = depth
        # 两个 source 的已实现谱系互不重叠：水果被 merge 消费后不能再次作为
        # 另一条分支的 source。因此这里可以把两侧祖先效用与本次合成相加，
        # 得到“以这个新水果结束的连通 merge 谱系”效用。
        lineage_utility = (
            sum(
                lineage_utility_by_fruit_id.get(source_id, 0.0)
                for source_id in event.source_ids
            )
            + _merge_utility(event.new_level)
        )
        lineage_utility_by_fruit_id[event.new_fruit_id] = (
            lineage_utility
        )
        max_depth = max(max_depth, depth)
        if depth >= 2:
            max_chain_utility = max(
                max_chain_utility,
                lineage_utility,
            )

    # 两次互不相干的同一步合成不冒充连锁；必须存在新 fruit ID 被后续合成消费。
    if max_depth < 2:
        return 0.0
    # 效用只取同一连通谱系。与该链无 fruit-ID 关系的同时合成不能抬高标签。
    utility_score = max_chain_utility / (
        max_chain_utility + _merge_utility(5)
    )
    depth_score = min(1.0, max_depth / 4.0)
    return _clip_unit(
        0.5 * utility_score
        + 0.5 * depth_score
    )


def build_structural_target(
        previous_analysis: StateAnalysis | None,
        next_analysis: StateAnalysis | None,
        *,
        physics_result: PhysicsResult | None = None,
        merge_events=None,
        terminated: bool | None = None) -> StructuralTarget:
    """从当前动作已完成的证据构造一步结构监督。

    前五维只在相邻 analysis 均稳定、可归因且未降级时有效。最后一维独立依赖
    当前动作的物理/合成/terminal 证据，因此即使 terminal 没有 post-action
    analysis，也可以保留明确的终局风险。
    """

    analysis_valid = _validate_analysis_pair(
        previous_analysis,
        next_analysis,
    )
    events, resolved_terminated, action_evidence_valid = (
        _resolve_action_evidence(
            physics_result=physics_result,
            merge_events=merge_events,
            terminated=terminated,
        )
    )

    values = [0.0] * STRUCTURAL_TARGET_DIMENSION_COUNT
    valid_mask = 0
    if analysis_valid:
        values[0] = _clip_signed_unit(
            next_analysis.top_connected_capacity
            - previous_analysis.top_connected_capacity
        )
        values[1] = _clip_signed_unit(
            next_analysis.recoverability
            - previous_analysis.recoverability
        )
        values[2] = _clip_signed_unit(
            next_analysis.chain_readiness
            - previous_analysis.chain_readiness
        )
        values[3] = _new_dead_or_blocked_risk(
            previous_analysis,
            next_analysis,
        )
        values[4] = _clip_signed_unit(
            _sealed_cavity_ratio(next_analysis)
            - _sealed_cavity_ratio(previous_analysis)
        )
        valid_mask |= (1 << 5) - 1

    if action_evidence_valid:
        values[5] = _realized_chain_signal(
            events,
            resolved_terminated,
        )
        valid_mask |= 1 << 5

    return StructuralTarget(
        values=tuple(values),
        valid_mask=valid_mask,
    )


__all__ = [
    'STRUCTURAL_TARGET_DIMENSIONS',
    'STRUCTURAL_TARGET_DIMENSION_COUNT',
    'STRUCTURAL_TARGET_FULL_VALID_MASK',
    'STRUCTURAL_TARGET_HALF_PAYLOAD_BYTES',
    'STRUCTURAL_TARGET_SCHEMA_VERSION',
    'StructuralTarget',
    'build_structural_target',
]
