"""Reward V2：指数合成效用与状态 potential shaping。

游戏规则层继续维护原始 score；训练奖励只消费真实 ``MergeEvent`` 和
``StateAnalysis``，避免把存活时间、最高水果高度或同一次连锁重复计分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from operator import index

from daxigua.core.rules import MAX_FRUIT_LEVEL, MIN_FRUIT_LEVEL, merge_score
from daxigua.core.state import MergeEvent, PhysicsResult

from .attribution.schema import StateAnalysis


# 训练日志和可视化只从这里读取稳定字段顺序。
REWARD_BREAKDOWN_FIELDS = (
    'total',
    'task_reward',
    'potential_shaping_reward',
    'terminal_penalty',
    'previous_potential',
    'next_potential',
    'potential_delta',
    'previous_top_connected_capacity',
    'next_top_connected_capacity',
    'previous_recoverability',
    'next_recoverability',
    'previous_chain_readiness',
    'next_chain_readiness',
    'merge_event_count',
)


def _finite_float(name, value, *, minimum=None, maximum=None):
    """读取有限实数，并拒绝 bool 之类容易误配的值。"""

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


def _strict_integer(name, value, *, minimum=None):
    """读取严格整数，避免浮点截断或 bool 污染事件一致性检查。"""

    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer')
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardConfig:
    """Reward V2 的固定公式参数。

    ``gamma`` 必须与 DQN target 的折扣因子相同；训练入口只从同一个参数构造两者。
    ``terminal_penalty`` 默认关闭，仅保留为短跑证明确有必要时的小额显式开关。
    """

    gamma: float = 0.99
    lambda_phi: float = 0.5
    capacity_weight: float = 0.6
    recoverability_weight: float = 0.3
    chain_readiness_weight: float = 0.1
    terminal_penalty: float = 0.0

    def __post_init__(self):
        for field_name in (
                'gamma',
                'lambda_phi',
                'capacity_weight',
                'recoverability_weight',
                'chain_readiness_weight'):
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

        weight_sum = (
            self.capacity_weight
            + self.recoverability_weight
            + self.chain_readiness_weight
        )
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError('potential weights must sum to 1')

        terminal_penalty = _finite_float(
            'terminal_penalty',
            self.terminal_penalty,
            maximum=0.0,
        )
        object.__setattr__(self, 'terminal_penalty', terminal_penalty)


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardBreakdown:
    """一次环境 step 的 Reward V2 明细。"""

    total: float
    task_reward: float
    potential_shaping_reward: float
    terminal_penalty: float
    previous_potential: float
    next_potential: float
    potential_delta: float
    previous_top_connected_capacity: float
    next_top_connected_capacity: float
    previous_recoverability: float
    next_recoverability: float
    previous_chain_readiness: float
    next_chain_readiness: float
    merge_event_count: int

    def __post_init__(self):
        for field_name in REWARD_BREAKDOWN_FIELDS:
            value = getattr(self, field_name)
            if field_name == 'merge_event_count':
                if isinstance(value, bool):
                    raise TypeError('merge_event_count must be an integer')
                try:
                    count = index(value)
                except TypeError as exc:
                    raise TypeError('merge_event_count must be an integer') from exc
                if count < 0:
                    raise ValueError('merge_event_count must be non-negative')
                object.__setattr__(self, field_name, count)
            else:
                object.__setattr__(
                    self,
                    field_name,
                    _finite_float(field_name, value),
                )

        reconstructed = (
            self.task_reward
            + self.potential_shaping_reward
            + self.terminal_penalty
        )
        if not math.isclose(
                self.total,
                reconstructed,
                rel_tol=0.0,
                abs_tol=1e-9):
            raise ValueError('reward breakdown components must reconstruct total')

    def to_dict(self):
        """转换成普通 dict，供 collector、CSV 和终端日志统一消费。"""

        return {
            field_name: getattr(self, field_name)
            for field_name in REWARD_BREAKDOWN_FIELDS
        }


def merge_utility(new_level):
    """返回合成到等级 ``L`` 的指数任务效用 ``2**((L-2)/2)``。"""

    new_level = _strict_integer('new_level', new_level)

    minimum_merge_level = MIN_FRUIT_LEVEL + 1
    if new_level < minimum_merge_level or new_level > MAX_FRUIT_LEVEL:
        raise ValueError(
            f'new_level must be in [{minimum_merge_level}, {MAX_FRUIT_LEVEL}]'
        )
    return float(2.0 ** ((new_level - 2) / 2.0))


def compute_state_potential(analysis, config=None):
    """由完整状态分析计算 ``Phi(s)``。"""

    if not isinstance(analysis, StateAnalysis):
        raise TypeError('analysis must be StateAnalysis')
    config = config or RewardConfig()
    if not isinstance(config, RewardConfig):
        raise TypeError('config must be RewardConfig')

    potential = (
        config.capacity_weight * analysis.top_connected_capacity
        + config.recoverability_weight * analysis.recoverability
        + config.chain_readiness_weight * analysis.chain_readiness
    )
    # 三个输入和归一化权重都已校验；容忍浮点求和的极小边界误差。
    return max(0.0, min(1.0, float(potential)))


def compute_reward(
        previous_analysis,
        next_analysis,
        physics_result,
        config=None):
    """计算 Reward V2，并返回 ``(reward, RewardBreakdown)``。

    真实终止允许 ``next_analysis=None`` 且强制下一 potential 为零；普通 transition
    和物理截断必须提供相邻的下一状态分析。该函数不运行 ``StateAnalyzer``。
    """

    config = config or RewardConfig()
    if not isinstance(config, RewardConfig):
        raise TypeError('config must be RewardConfig')
    if not isinstance(previous_analysis, StateAnalysis):
        raise TypeError('previous_analysis must be StateAnalysis')
    if not isinstance(physics_result, PhysicsResult):
        raise TypeError('physics_result must be PhysicsResult')
    if physics_result.done and physics_result.truncated:
        raise ValueError('physics_result cannot be both done and truncated')

    if next_analysis is not None:
        if not isinstance(next_analysis, StateAnalysis):
            raise TypeError('next_analysis must be StateAnalysis or None')
        _validate_adjacent_analyses(previous_analysis, next_analysis)
    elif not physics_result.done:
        raise ValueError('non-terminal reward requires next_analysis')

    merge_events = _validate_merge_events(physics_result)
    task_reward = sum(
        merge_utility(event.new_level)
        for event in merge_events
    )

    previous_potential = compute_state_potential(previous_analysis, config)
    if physics_result.done:
        next_potential = 0.0
        next_capacity = 0.0
        next_recoverability = 0.0
        next_chain_readiness = 0.0
    else:
        next_potential = compute_state_potential(next_analysis, config)
        next_capacity = float(next_analysis.top_connected_capacity)
        next_recoverability = float(next_analysis.recoverability)
        next_chain_readiness = float(next_analysis.chain_readiness)

    potential_delta = config.gamma * next_potential - previous_potential
    potential_shaping_reward = config.lambda_phi * potential_delta
    terminal_penalty = (
        config.terminal_penalty
        if physics_result.done
        else 0.0
    )
    total = task_reward + potential_shaping_reward + terminal_penalty

    breakdown = RewardBreakdown(
        total=float(total),
        task_reward=float(task_reward),
        potential_shaping_reward=float(potential_shaping_reward),
        terminal_penalty=float(terminal_penalty),
        previous_potential=float(previous_potential),
        next_potential=float(next_potential),
        potential_delta=float(potential_delta),
        previous_top_connected_capacity=float(
            previous_analysis.top_connected_capacity
        ),
        next_top_connected_capacity=next_capacity,
        previous_recoverability=float(previous_analysis.recoverability),
        next_recoverability=next_recoverability,
        previous_chain_readiness=float(previous_analysis.chain_readiness),
        next_chain_readiness=next_chain_readiness,
        merge_event_count=len(merge_events),
    )
    return float(total), breakdown


def _validate_adjacent_analyses(previous_analysis, next_analysis):
    """拒绝跨 worker、跨 episode、跳步或分析口径漂移。"""

    previous_key = previous_analysis.transition_key
    next_key = next_analysis.transition_key
    if (
            previous_key.worker_id != next_key.worker_id
            or previous_key.episode_id != next_key.episode_id
            or previous_key.step_index + 1 != next_key.step_index):
        raise ValueError(
            'next_analysis must identify the immediately following state '
            'in the same worker and episode'
        )
    if (
            next_analysis.incoming_transition_key is not None
            and next_analysis.incoming_transition_key != previous_key):
        raise ValueError(
            'next_analysis.incoming_transition_key must match previous analysis'
        )
    if (
            previous_analysis.analyzer_config_fingerprint
            != next_analysis.analyzer_config_fingerprint):
        raise ValueError('adjacent analyses must use the same analyzer config')


def _validate_merge_events(physics_result):
    """检查任务价值所依赖的事件流完整且没有重复。"""

    merge_events = tuple(physics_result.merge_events)
    if any(not isinstance(event, MergeEvent) for event in merge_events):
        raise TypeError('physics_result.merge_events must contain MergeEvent values')

    new_fruit_ids = []
    score_delta_total = 0
    for event_offset, event in enumerate(merge_events):
        # merge_utility 同时验证等级范围。
        merge_utility(event.new_level)
        expected_score = merge_score(event.new_level)
        event_score = _strict_integer(
            f'merge_events[{event_offset}].score_delta',
            event.score_delta,
        )
        if event_score != expected_score:
            raise ValueError(
                'merge event score_delta does not match the game rule '
                f'for level {event.new_level}'
            )
        source_ids = tuple(
            _strict_integer(
                f'merge_events[{event_offset}].source_ids[{source_offset}]',
                source_id,
                minimum=1,
            )
            for source_offset, source_id in enumerate(event.source_ids)
        )
        if len(source_ids) != 2 or source_ids[0] == source_ids[1]:
            raise ValueError('merge event must contain two distinct source_ids')
        new_fruit_id = _strict_integer(
            f'merge_events[{event_offset}].new_fruit_id',
            event.new_fruit_id,
            minimum=1,
        )
        if new_fruit_id in source_ids:
            raise ValueError('merge event new_fruit_id must differ from source_ids')
        new_fruit_ids.append(new_fruit_id)
        score_delta_total += event_score

    if len(set(new_fruit_ids)) != len(new_fruit_ids):
        raise ValueError('merge_events contains duplicate new_fruit_id values')
    physics_score_delta = _strict_integer(
        'physics_result.score_delta',
        physics_result.score_delta,
    )
    if score_delta_total != physics_score_delta:
        raise ValueError(
            'physics_result.score_delta must equal the sum of merge event scores'
        )
    return merge_events
