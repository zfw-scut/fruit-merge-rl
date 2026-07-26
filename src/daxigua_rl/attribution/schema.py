"""完整状态归因使用的只读数据契约。

本模块只描述分析结果“长什么样”，不负责计算可达性、支撑关系或连锁结构。
所有集合都规范化为 tuple，所有数据类都可哈希侧安全地跨 Windows spawn/pickle
边界传递；训练 worker 后续可以在本地保存这些对象，而不污染主 ReplayBuffer。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from operator import index

from daxigua.core.rules import (
    MAX_FRUIT_LEVEL,
    MIN_FRUIT_LEVEL,
    SPAWN_FRUIT_MAX_LEVEL,
    SPAWN_FRUIT_MIN_LEVEL,
    dropped_fruit_physics_radius,
)
from daxigua_rl.training.identity import TransitionKey


ANALYSIS_ACTION_COUNT = 15
QUEUE_LOOKAHEAD_COUNT = 4
FULL_ACTION_MASK = (1 << ANALYSIS_ACTION_COUNT) - 1
STATE_ANALYSIS_SCHEMA_VERSION = 1
LANDING_DEPTH_WEIGHT = 0.7
SAFE_ACTION_WEIGHT = 0.3
DEFAULT_QUEUE_DECAY = 0.5

SUPPORT_BOUNDARIES = ('floor', 'left_wall', 'right_wall')
SUPPORT_RELATIONS = ('supports', 'wall_constraint', 'caps', 'bridges')


def _integer(name, value, *, minimum=0):
    """读取严格整数，避免把 bool 或 1.5 静默变成合法 ID。"""

    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer, got bool')
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc
    if result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return result


def _finite_float(name, value, *, minimum=None):
    """规范化有限浮点数，并按需检查下界。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return result


def _unit_float(name, value):
    """读取一个闭区间 ``[0, 1]`` 内的有限比例。"""

    result = _finite_float(name, value)
    if result > 1.0:
        raise ValueError(f'{name} must be in [0, 1]')
    if result < 0.0:
        raise ValueError(f'{name} must be in [0, 1]')
    return result


def _boolean(name, value):
    """只接受真正的 bool，防止诊断标志被整数污染。"""

    if not isinstance(value, bool):
        raise TypeError(f'{name} must be bool')
    return value


def _action_mask(name, value):
    """读取固定为 15 位、按 action offset 编位的动作掩码。"""

    result = _integer(name, value)
    if result & ~FULL_ACTION_MASK:
        raise ValueError(
            f'{name} contains bits outside {ANALYSIS_ACTION_COUNT} actions'
        )
    return result


def _fruit_level(name, value, *, spawn_only=False):
    """读取规则允许的场上等级或新投放等级。"""

    minimum = SPAWN_FRUIT_MIN_LEVEL if spawn_only else MIN_FRUIT_LEVEL
    maximum = SPAWN_FRUIT_MAX_LEVEL if spawn_only else MAX_FRUIT_LEVEL
    result = _integer(name, value, minimum=minimum)
    if result > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return result


def _id_tuple(name, values, *, own_id=None, minimum_length=0, sort_values=True):
    """把水果 ID 集合规范化为无重复 tuple。"""

    result = tuple(
        _integer(f'{name}[{offset}]', value, minimum=1)
        for offset, value in enumerate(values)
    )
    if len(result) < minimum_length:
        raise ValueError(f'{name} must contain at least {minimum_length} items')
    if len(set(result)) != len(result):
        raise ValueError(f'{name} must not contain duplicate fruit IDs')
    if own_id is not None and own_id in result:
        raise ValueError(f'{name} must not contain the owning fruit ID')
    return tuple(sorted(result)) if sort_values else result


def _fixed_float_tuple(name, values, *, unit=False):
    """规范化一个与 15 个 action offset 对齐的浮点 tuple。"""

    result = tuple(
        (_unit_float if unit else _finite_float)(f'{name}[{offset}]', value)
        for offset, value in enumerate(values)
    )
    if len(result) != ANALYSIS_ACTION_COUNT:
        raise ValueError(
            f'{name} must contain exactly {ANALYSIS_ACTION_COUNT} items'
        )
    return result


def _strictly_increasing(name, values):
    """确保 action offset 从左到右对应唯一投放列。"""

    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f'{name} must be strictly increasing')
    return values


def _blocker_matrix(name, values):
    """规范化 15 条动作路径各自的阻挡水果 ID。"""

    result = tuple(
        _id_tuple(
            f'{name}[{action_offset}]',
            blocker_ids,
            sort_values=False,
        )
        for action_offset, blocker_ids in enumerate(values)
    )
    if len(result) != ANALYSIS_ACTION_COUNT:
        raise ValueError(
            f'{name} must contain exactly {ANALYSIS_ACTION_COUNT} items'
        )
    return result


def _code_tuple(name, values):
    """规范化短诊断码；诊断正文不应进入高频状态对象。"""

    result = []
    for offset, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f'{name}[{offset}] must be str')
        value = value.strip()
        if not value:
            raise ValueError(f'{name}[{offset}] must not be empty')
        result.append(value)
    if len(set(result)) != len(result):
        raise ValueError(f'{name} must not contain duplicates')
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True, kw_only=True)
class FruitAnalysis:
    """一颗场上水果在当前稳定边界的静态分析结果。

    ``physics_radius`` 是该对象当前 shape 的真实半径；
    ``probe_physics_radius`` 是一个未来直接投放的同级水果用于可达性膨胀的半径。
    两者在合成生成水果上可能不同，不能相互推导。
    """

    fruit_id: int
    level: int
    physics_radius: float
    probe_physics_radius: float
    reachable_action_mask: int
    reachable_action_count: int
    top_visible_ratio: float
    top_blocker_ids_by_action: tuple[tuple[int, ...], ...]
    partner_ids: tuple[int, ...]
    partner_reachable: bool
    support_parent_ids: tuple[int, ...]
    supported_child_ids: tuple[int, ...]
    burial_depth: float
    inversion_count: int
    connected_region_id: int | None
    reachable_partner_ids: tuple[int, ...] = ()
    critical_blocker_ids: tuple[int, ...] = ()
    inversion_blocker_ids: tuple[int, ...] = ()

    def __post_init__(self):
        fruit_id = _integer('fruit_id', self.fruit_id, minimum=1)
        object.__setattr__(self, 'fruit_id', fruit_id)
        object.__setattr__(self, 'level', _fruit_level('level', self.level))
        object.__setattr__(
            self,
            'physics_radius',
            _finite_float('physics_radius', self.physics_radius, minimum=0.0),
        )
        object.__setattr__(
            self,
            'probe_physics_radius',
            _finite_float(
                'probe_physics_radius',
                self.probe_physics_radius,
                minimum=0.0,
            ),
        )
        if self.physics_radius == 0.0:
            raise ValueError('physics_radius must be positive')
        if self.probe_physics_radius == 0.0:
            raise ValueError('probe_physics_radius must be positive')
        expected_probe_radius = float(dropped_fruit_physics_radius(self.level))
        if not math.isclose(
                self.probe_physics_radius,
                expected_probe_radius,
                rel_tol=0.0,
                abs_tol=1e-9):
            raise ValueError(
                'probe_physics_radius must equal the direct-drop radius '
                'for this level'
            )

        mask = _action_mask('reachable_action_mask', self.reachable_action_mask)
        count = _integer('reachable_action_count', self.reachable_action_count)
        if count != mask.bit_count():
            raise ValueError(
                'reachable_action_count must equal '
                'reachable_action_mask.bit_count()'
            )
        object.__setattr__(self, 'reachable_action_mask', mask)
        object.__setattr__(self, 'reachable_action_count', count)
        object.__setattr__(
            self,
            'top_visible_ratio',
            _unit_float('top_visible_ratio', self.top_visible_ratio),
        )
        object.__setattr__(
            self,
            'top_blocker_ids_by_action',
            _blocker_matrix(
                'top_blocker_ids_by_action',
                self.top_blocker_ids_by_action,
            ),
        )

        for field_name in (
                'partner_ids',
                'support_parent_ids',
                'supported_child_ids',
                'reachable_partner_ids',
                'critical_blocker_ids',
                'inversion_blocker_ids'):
            object.__setattr__(
                self,
                field_name,
                _id_tuple(
                    field_name,
                    getattr(self, field_name),
                    own_id=fruit_id,
                ),
            )

        object.__setattr__(
            self,
            'partner_reachable',
            _boolean('partner_reachable', self.partner_reachable),
        )
        if self.reachable_partner_ids and not self.partner_reachable:
            raise ValueError(
                'reachable_partner_ids requires partner_reachable=True'
            )
        if not set(self.reachable_partner_ids).issubset(self.partner_ids):
            raise ValueError('reachable_partner_ids must be a subset of partner_ids')
        path_blocker_ids = {
            blocker_id
            for blockers in self.top_blocker_ids_by_action
            for blocker_id in blockers
        }
        if not set(self.critical_blocker_ids).issubset(path_blocker_ids):
            raise ValueError(
                'critical_blocker_ids must come from top_blocker_ids_by_action'
            )

        object.__setattr__(
            self,
            'burial_depth',
            _unit_float('burial_depth', self.burial_depth),
        )
        inversion_count = _integer('inversion_count', self.inversion_count)
        if inversion_count != len(self.inversion_blocker_ids):
            raise ValueError(
                'inversion_count must equal len(inversion_blocker_ids)'
            )
        object.__setattr__(self, 'inversion_count', inversion_count)

        if self.connected_region_id is not None:
            object.__setattr__(
                self,
                'connected_region_id',
                _integer('connected_region_id', self.connected_region_id),
            )

    @property
    def reachable_fraction(self):
        """返回 15 个动作中仍可接近本水果的比例。"""

        return self.reachable_action_count / ANALYSIS_ACTION_COUNT


@dataclass(frozen=True, slots=True, kw_only=True)
class SupportEdge:
    """一条稳定约束边，方向统一为 supporter -> supported fruit。"""

    supported_fruit_id: int
    relation: str
    supporter_fruit_id: int | None = None
    boundary: str | None = None
    confidence: float = 1.0

    def __post_init__(self):
        supported_id = _integer(
            'supported_fruit_id',
            self.supported_fruit_id,
            minimum=1,
        )
        object.__setattr__(self, 'supported_fruit_id', supported_id)

        if (self.supporter_fruit_id is None) == (self.boundary is None):
            raise ValueError(
                'exactly one of supporter_fruit_id and boundary must be set'
            )
        if self.supporter_fruit_id is not None:
            supporter_id = _integer(
                'supporter_fruit_id',
                self.supporter_fruit_id,
                minimum=1,
            )
            if supporter_id == supported_id:
                raise ValueError('support edge must not reference itself')
            object.__setattr__(self, 'supporter_fruit_id', supporter_id)

        if self.boundary is not None:
            if self.boundary not in SUPPORT_BOUNDARIES:
                raise ValueError(
                    f'boundary must be one of {SUPPORT_BOUNDARIES!r}'
                )

        if self.relation not in SUPPORT_RELATIONS:
            raise ValueError(
                f'relation must be one of {SUPPORT_RELATIONS!r}'
            )
        if self.boundary == 'floor' and self.relation != 'supports':
            raise ValueError('floor edges must use relation="supports"')
        if self.boundary in {'left_wall', 'right_wall'}:
            if self.relation != 'wall_constraint':
                raise ValueError(
                    'wall edges must use relation="wall_constraint"'
                )
        if self.supporter_fruit_id is not None:
            if self.relation == 'wall_constraint':
                raise ValueError(
                    'fruit support edges cannot use relation="wall_constraint"'
                )

        object.__setattr__(
            self,
            'confidence',
            _unit_float('confidence', self.confidence),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ContactInfluenceEdge:
    """前一动作中 source 对 target 产生的压缩物理影响证据。"""

    source_fruit_id: int
    target_fruit_id: int
    contact_count: int
    displacement_x: float
    displacement_y: float
    max_impulse: float | None = None
    first_contact_frame: int | None = None
    last_contact_frame: int | None = None
    on_merge_path: bool = False

    def __post_init__(self):
        source_id = _integer(
            'source_fruit_id',
            self.source_fruit_id,
            minimum=1,
        )
        target_id = _integer(
            'target_fruit_id',
            self.target_fruit_id,
            minimum=1,
        )
        if source_id == target_id:
            raise ValueError('contact influence edge must not reference itself')
        object.__setattr__(self, 'source_fruit_id', source_id)
        object.__setattr__(self, 'target_fruit_id', target_id)
        object.__setattr__(
            self,
            'contact_count',
            _integer('contact_count', self.contact_count, minimum=1),
        )
        object.__setattr__(
            self,
            'displacement_x',
            _finite_float('displacement_x', self.displacement_x),
        )
        object.__setattr__(
            self,
            'displacement_y',
            _finite_float('displacement_y', self.displacement_y),
        )
        if self.max_impulse is not None:
            object.__setattr__(
                self,
                'max_impulse',
                _finite_float('max_impulse', self.max_impulse, minimum=0.0),
            )
        for field_name in ('first_contact_frame', 'last_contact_frame'):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _integer(field_name, value),
                )
        if (
                self.first_contact_frame is not None
                and self.last_contact_frame is not None
                and self.first_contact_frame > self.last_contact_frame):
            raise ValueError(
                'first_contact_frame must not exceed last_contact_frame'
            )
        if (
                (self.first_contact_frame is None)
                != (self.last_contact_frame is None)):
            raise ValueError(
                'first_contact_frame and last_contact_frame must be '
                'provided together'
            )
        object.__setattr__(
            self,
            'on_merge_path',
            _boolean('on_merge_path', self.on_merge_path),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PartnerComponent:
    """同等级候选伙伴组成的单状态局部连通分量。"""

    component_id: int
    level: int
    fruit_ids: tuple[int, ...]
    reachable_action_mask: int
    top_connected: bool
    connected_region_id: int | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            'component_id',
            _integer('component_id', self.component_id),
        )
        object.__setattr__(self, 'level', _fruit_level('level', self.level))
        object.__setattr__(
            self,
            'fruit_ids',
            _id_tuple('fruit_ids', self.fruit_ids, minimum_length=1),
        )
        object.__setattr__(
            self,
            'reachable_action_mask',
            _action_mask('reachable_action_mask', self.reachable_action_mask),
        )
        object.__setattr__(
            self,
            'top_connected',
            _boolean('top_connected', self.top_connected),
        )
        if self.connected_region_id is not None:
            object.__setattr__(
                self,
                'connected_region_id',
                _integer('connected_region_id', self.connected_region_id),
            )

    @property
    def signature(self):
        """返回不依赖 Python hash 随机化的确定性结构签名。"""

        return self.level, self.fruit_ids


@dataclass(frozen=True, slots=True, kw_only=True)
class ChainMotif:
    """一个可被未来投放触发的局部连锁合成结构。"""

    motif_type: str
    fruit_ids: tuple[int, ...]
    levels: tuple[int, ...]
    depth: int
    trigger_action_mask: int
    compatible_queue_indices: tuple[int, ...]
    readiness: float

    def __post_init__(self):
        if not isinstance(self.motif_type, str):
            raise TypeError('motif_type must be str')
        motif_type = self.motif_type.strip()
        if not motif_type:
            raise ValueError('motif_type must not be empty')
        object.__setattr__(self, 'motif_type', motif_type)

        fruit_ids = _id_tuple(
            'fruit_ids',
            self.fruit_ids,
            minimum_length=2,
            sort_values=False,
        )
        levels = tuple(
            _fruit_level(f'levels[{offset}]', level)
            for offset, level in enumerate(self.levels)
        )
        if len(levels) != len(fruit_ids):
            raise ValueError('levels must align one-to-one with fruit_ids')
        if motif_type == 'merge_pair':
            if len(fruit_ids) != 2 or len(set(levels)) != 1:
                raise ValueError(
                    'merge_pair must contain exactly two same-level fruits'
                )
            ordered_members = tuple(sorted(zip(fruit_ids, levels)))
            fruit_ids = tuple(item[0] for item in ordered_members)
            levels = tuple(item[1] for item in ordered_members)
        object.__setattr__(self, 'fruit_ids', fruit_ids)
        object.__setattr__(self, 'levels', levels)
        object.__setattr__(
            self,
            'depth',
            _integer('depth', self.depth, minimum=1),
        )
        object.__setattr__(
            self,
            'trigger_action_mask',
            _action_mask('trigger_action_mask', self.trigger_action_mask),
        )

        queue_indices = tuple(
            _integer(
                f'compatible_queue_indices[{offset}]',
                queue_index,
            )
            for offset, queue_index in enumerate(self.compatible_queue_indices)
        )
        if len(set(queue_indices)) != len(queue_indices):
            raise ValueError('compatible_queue_indices must not contain duplicates')
        if any(index_value >= QUEUE_LOOKAHEAD_COUNT for index_value in queue_indices):
            raise ValueError(
                'compatible_queue_indices contains an unavailable queue slot'
            )
        object.__setattr__(
            self,
            'compatible_queue_indices',
            tuple(sorted(queue_indices)),
        )
        object.__setattr__(
            self,
            'readiness',
            _unit_float('readiness', self.readiness),
        )

    @property
    def signature(self):
        """返回由类型和有序成员构成的确定性 motif 签名。"""

        return self.motif_type, self.fruit_ids, self.levels


@dataclass(frozen=True, slots=True, kw_only=True)
class QueueLaneAnalysis:
    """队列一个槽位在 15 个动作列上的投放容量分析。"""

    queue_index: int
    level: int
    physics_radius: float
    drop_x_by_action: tuple[float, ...]
    landing_depths_by_action: tuple[float, ...]
    safe_action_mask: int
    safe_action_count: int
    blocker_ids_by_action: tuple[tuple[int, ...], ...]
    capacity: float

    def __post_init__(self):
        queue_index = _integer('queue_index', self.queue_index)
        if queue_index >= QUEUE_LOOKAHEAD_COUNT:
            raise ValueError(
                f'queue_index must be < {QUEUE_LOOKAHEAD_COUNT}'
            )
        object.__setattr__(self, 'queue_index', queue_index)
        object.__setattr__(
            self,
            'level',
            _fruit_level('level', self.level, spawn_only=True),
        )
        radius = _finite_float(
            'physics_radius',
            self.physics_radius,
            minimum=0.0,
        )
        if radius == 0.0:
            raise ValueError('physics_radius must be positive')
        expected_radius = float(dropped_fruit_physics_radius(self.level))
        if not math.isclose(
                radius,
                expected_radius,
                rel_tol=0.0,
                abs_tol=1e-9):
            raise ValueError(
                'queue physics_radius must equal the direct-drop radius '
                'for this level'
            )
        object.__setattr__(self, 'physics_radius', radius)
        drop_x_by_action = _fixed_float_tuple(
            'drop_x_by_action',
            self.drop_x_by_action,
        )
        object.__setattr__(
            self,
            'drop_x_by_action',
            _strictly_increasing(
                'drop_x_by_action',
                drop_x_by_action,
            ),
        )
        object.__setattr__(
            self,
            'landing_depths_by_action',
            _fixed_float_tuple(
                'landing_depths_by_action',
                self.landing_depths_by_action,
                unit=True,
            ),
        )

        mask = _action_mask('safe_action_mask', self.safe_action_mask)
        count = _integer('safe_action_count', self.safe_action_count)
        if count != mask.bit_count():
            raise ValueError(
                'safe_action_count must equal safe_action_mask.bit_count()'
            )
        object.__setattr__(self, 'safe_action_mask', mask)
        object.__setattr__(self, 'safe_action_count', count)
        object.__setattr__(
            self,
            'blocker_ids_by_action',
            _blocker_matrix(
                'blocker_ids_by_action',
                self.blocker_ids_by_action,
            ),
        )
        object.__setattr__(
            self,
            'capacity',
            _unit_float('capacity', self.capacity),
        )
        if not math.isclose(
                self.capacity,
                self.computed_capacity,
                rel_tol=0.0,
                abs_tol=1e-9):
            raise ValueError(
                'capacity must equal 0.7 * mean_landing_depth '
                '+ 0.3 * safe_fraction'
            )

    @property
    def safe_fraction(self):
        """返回仍被判定安全的动作比例。"""

        return self.safe_action_count / ANALYSIS_ACTION_COUNT

    @property
    def mean_landing_depth(self):
        """返回 15 个动作列的平均可深入程度。"""

        return sum(self.landing_depths_by_action) / ANALYSIS_ACTION_COUNT

    @property
    def computed_capacity(self):
        """按 Reward V2 固定公式计算本槽位容量。"""

        return (
            LANDING_DEPTH_WEIGHT * self.mean_landing_depth
            + SAFE_ACTION_WEIGHT * self.safe_fraction
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StateAnalysisDiagnostics:
    """状态分析的有效性、降级和轻量性能诊断。"""

    stable_boundary: bool
    valid_for_attribution: bool
    physics_frame: int
    analysis_seconds: float = 0.0
    degraded: bool = False
    warning_codes: tuple[str, ...] = ()
    approximation_flags: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(
            self,
            'stable_boundary',
            _boolean('stable_boundary', self.stable_boundary),
        )
        object.__setattr__(
            self,
            'valid_for_attribution',
            _boolean('valid_for_attribution', self.valid_for_attribution),
        )
        object.__setattr__(
            self,
            'degraded',
            _boolean('degraded', self.degraded),
        )
        if self.valid_for_attribution and not self.stable_boundary:
            raise ValueError(
                'an unstable boundary cannot be valid for attribution'
            )
        object.__setattr__(
            self,
            'physics_frame',
            _integer('physics_frame', self.physics_frame),
        )
        object.__setattr__(
            self,
            'analysis_seconds',
            _finite_float(
                'analysis_seconds',
                self.analysis_seconds,
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            'warning_codes',
            _code_tuple('warning_codes', self.warning_codes),
        )
        object.__setattr__(
            self,
            'approximation_flags',
            _code_tuple(
                'approximation_flags',
                self.approximation_flags,
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StateAnalysis:
    """一个动作前稳定边界的完整、只读状态分析快照。

    ``transition_key.step_index=t`` 表示这是动作 ``t`` 执行前的边界。若对象还
    携带前一动作的接触证据，``incoming_transition_key`` 必须是同一 episode 的
    ``t-1``；初始边界没有 incoming key。动作 mask 的第 a 位始终按
    ``action_offset=a`` 编位，真实环境动作号保存在 ``action_indices[a]``。
    """

    transition_key: TransitionKey
    action_indices: tuple[int, ...]
    action_drop_x_by_offset: tuple[float, ...]
    queue_lane_analyses: tuple[QueueLaneAnalysis, ...]
    top_connected_capacity: float
    recoverability: float
    chain_readiness: float
    diagnostics: StateAnalysisDiagnostics
    analyzer_config_fingerprint: str
    queue_decay: float = DEFAULT_QUEUE_DECAY
    incoming_transition_key: TransitionKey | None = None
    fruit_analyses: tuple[FruitAnalysis, ...] = ()
    support_edges: tuple[SupportEdge, ...] = ()
    contact_influence_edges: tuple[ContactInfluenceEdge, ...] = ()
    partner_components: tuple[PartnerComponent, ...] = ()
    chain_motifs: tuple[ChainMotif, ...] = ()
    schema_version: int = STATE_ANALYSIS_SCHEMA_VERSION

    def __post_init__(self):
        if not isinstance(self.transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        if (
                self.incoming_transition_key is not None
                and not isinstance(self.incoming_transition_key, TransitionKey)):
            raise TypeError('incoming_transition_key must be TransitionKey or None')

        action_indices = tuple(
            _integer(f'action_indices[{offset}]', action_index)
            for offset, action_index in enumerate(self.action_indices)
        )
        if len(action_indices) != ANALYSIS_ACTION_COUNT:
            raise ValueError(
                f'action_indices must contain exactly {ANALYSIS_ACTION_COUNT} items'
            )
        if len(set(action_indices)) != len(action_indices):
            raise ValueError('action_indices must not contain duplicates')
        object.__setattr__(self, 'action_indices', action_indices)
        action_drop_x = _fixed_float_tuple(
            'action_drop_x_by_offset',
            self.action_drop_x_by_offset,
        )
        object.__setattr__(
            self,
            'action_drop_x_by_offset',
            _strictly_increasing(
                'action_drop_x_by_offset',
                action_drop_x,
            ),
        )

        typed_collections = (
            ('fruit_analyses', FruitAnalysis, lambda item: item.fruit_id),
            (
                'support_edges',
                SupportEdge,
                lambda item: (
                    item.supported_fruit_id,
                    item.boundary or '',
                    item.supporter_fruit_id or 0,
                    item.relation,
                ),
            ),
            (
                'contact_influence_edges',
                ContactInfluenceEdge,
                lambda item: (item.source_fruit_id, item.target_fruit_id),
            ),
            (
                'partner_components',
                PartnerComponent,
                lambda item: item.component_id,
            ),
            ('chain_motifs', ChainMotif, lambda item: item.signature),
            (
                'queue_lane_analyses',
                QueueLaneAnalysis,
                lambda item: item.queue_index,
            ),
        )
        for field_name, item_type, sort_key in typed_collections:
            items = tuple(getattr(self, field_name))
            if any(not isinstance(item, item_type) for item in items):
                raise TypeError(
                    f'{field_name} must contain only {item_type.__name__}'
                )
            object.__setattr__(
                self,
                field_name,
                tuple(sorted(items, key=sort_key)),
            )

        if not isinstance(self.diagnostics, StateAnalysisDiagnostics):
            raise TypeError('diagnostics must be StateAnalysisDiagnostics')
        for field_name in (
                'top_connected_capacity',
                'recoverability',
                'chain_readiness'):
            object.__setattr__(
                self,
                field_name,
                _unit_float(field_name, getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            'queue_decay',
            _unit_float('queue_decay', self.queue_decay),
        )

        if not isinstance(self.analyzer_config_fingerprint, str):
            raise TypeError('analyzer_config_fingerprint must be str')
        fingerprint = self.analyzer_config_fingerprint.strip()
        if not fingerprint:
            raise ValueError('analyzer_config_fingerprint must not be empty')
        object.__setattr__(self, 'analyzer_config_fingerprint', fingerprint)

        version = _integer('schema_version', self.schema_version, minimum=1)
        if version != STATE_ANALYSIS_SCHEMA_VERSION:
            raise ValueError(
                'unsupported StateAnalysis schema_version '
                f'{version}; expected {STATE_ANALYSIS_SCHEMA_VERSION}'
            )
        object.__setattr__(self, 'schema_version', version)

        self._validate_time_semantics()
        self._validate_queue_layout()
        self._validate_current_state_references()

    def _validate_time_semantics(self):
        """确保接触证据不会被错误挂到当前或未来动作上。"""

        incoming = self.incoming_transition_key
        current = self.transition_key
        if incoming is None:
            if self.contact_influence_edges:
                raise ValueError(
                    'contact_influence_edges requires incoming_transition_key'
                )
            return
        if (
                incoming.worker_id != current.worker_id
                or incoming.episode_id != current.episode_id
                or incoming.step_index + 1 != current.step_index):
            raise ValueError(
                'incoming_transition_key must identify the immediately '
                'preceding action in the same worker and episode'
            )

    def _validate_queue_layout(self):
        """检查 q0-q3 和 action offset 的一一对应关系。"""

        expected_indices = tuple(range(QUEUE_LOOKAHEAD_COUNT))
        actual_indices = tuple(
            lane.queue_index
            for lane in self.queue_lane_analyses
        )
        if actual_indices != expected_indices:
            raise ValueError(
                f'queue_lane_analyses must contain q0-q'
                f'{QUEUE_LOOKAHEAD_COUNT - 1} exactly once'
            )

        q0_drop_x = self.queue_lane_analyses[0].drop_x_by_action
        if any(
                not math.isclose(
                    q0_x,
                    action_x,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for q0_x, action_x in zip(
                    q0_drop_x,
                    self.action_drop_x_by_offset,
                )):
            raise ValueError(
                'q0 drop_x_by_action must match action_drop_x_by_offset'
            )
        if not math.isclose(
                self.top_connected_capacity,
                self.computed_top_connected_capacity,
                rel_tol=0.0,
                abs_tol=1e-9):
            raise ValueError(
                'top_connected_capacity must equal the queue-decayed '
                'q0-q3 lane capacity'
            )

    def _validate_current_state_references(self):
        """验证只应指向当前棋盘水果的静态关系。"""

        fruit_by_id = {}
        for fruit in self.fruit_analyses:
            if fruit.fruit_id in fruit_by_id:
                raise ValueError('fruit_analyses contains duplicate fruit_id')
            fruit_by_id[fruit.fruit_id] = fruit
        fruit_ids = set(fruit_by_id)

        def require_current_ids(name, values):
            missing = set(values) - fruit_ids
            if missing:
                raise ValueError(
                    f'{name} references missing current fruit IDs '
                    f'{tuple(sorted(missing))!r}'
                )

        for fruit in self.fruit_analyses:
            for field_name in (
                    'partner_ids',
                    'support_parent_ids',
                    'supported_child_ids',
                    'reachable_partner_ids',
                    'critical_blocker_ids',
                    'inversion_blocker_ids'):
                require_current_ids(
                    f'fruit {fruit.fruit_id} {field_name}',
                    getattr(fruit, field_name),
                )
            for action_offset, blockers in enumerate(
                    fruit.top_blocker_ids_by_action):
                require_current_ids(
                    f'fruit {fruit.fruit_id} blockers at action '
                    f'{action_offset}',
                    blockers,
                )
                if fruit.fruit_id in blockers:
                    raise ValueError(
                        'a fruit cannot be its own top-path blocker'
                    )

            for partner_id in fruit.partner_ids:
                if fruit_by_id[partner_id].level != fruit.level:
                    raise ValueError('partner_ids must reference same-level fruits')
            for blocker_id in fruit.inversion_blocker_ids:
                if fruit_by_id[blocker_id].level <= fruit.level:
                    raise ValueError(
                        'inversion_blocker_ids must reference higher-level fruits'
                    )

        support_keys = set()
        fruit_parent_ids = {fruit_id: set() for fruit_id in fruit_ids}
        fruit_child_ids = {fruit_id: set() for fruit_id in fruit_ids}
        for edge in self.support_edges:
            require_current_ids(
                'support edge supported_fruit_id',
                (edge.supported_fruit_id,),
            )
            if edge.supporter_fruit_id is not None:
                require_current_ids(
                    'support edge supporter_fruit_id',
                    (edge.supporter_fruit_id,),
                )
            edge_key = (
                edge.supporter_fruit_id,
                edge.boundary,
                edge.supported_fruit_id,
                edge.relation,
            )
            if edge_key in support_keys:
                raise ValueError('support_edges contains a duplicate relation')
            support_keys.add(edge_key)
            if (
                    edge.supporter_fruit_id is not None
                    and edge.relation == 'supports'):
                fruit_parent_ids[edge.supported_fruit_id].add(
                    edge.supporter_fruit_id
                )
                fruit_child_ids[edge.supporter_fruit_id].add(
                    edge.supported_fruit_id
                )

        for fruit in self.fruit_analyses:
            if set(fruit.support_parent_ids) != fruit_parent_ids[fruit.fruit_id]:
                raise ValueError(
                    'support_parent_ids must match canonical support_edges'
                )
            if set(fruit.supported_child_ids) != fruit_child_ids[fruit.fruit_id]:
                raise ValueError(
                    'supported_child_ids must match canonical support_edges'
                )

        contact_pairs = set()
        for edge in self.contact_influence_edges:
            pair = edge.source_fruit_id, edge.target_fruit_id
            if pair in contact_pairs:
                raise ValueError(
                    'contact_influence_edges must be compressed per directed pair'
                )
            contact_pairs.add(pair)

        component_ids = set()
        component_members = set()
        for component in self.partner_components:
            if component.component_id in component_ids:
                raise ValueError('partner_components contains duplicate component_id')
            component_ids.add(component.component_id)
            require_current_ids('partner component', component.fruit_ids)
            if component_members.intersection(component.fruit_ids):
                raise ValueError(
                    'a fruit cannot belong to multiple partner components'
                )
            component_members.update(component.fruit_ids)
            if any(
                    fruit_by_id[fruit_id].level != component.level
                    for fruit_id in component.fruit_ids):
                raise ValueError(
                    'partner component level must match all member fruits'
                )

        motif_signatures = set()
        for motif in self.chain_motifs:
            if motif.signature in motif_signatures:
                raise ValueError('chain_motifs contains a duplicate signature')
            motif_signatures.add(motif.signature)
            require_current_ids('chain motif', motif.fruit_ids)
            actual_levels = tuple(
                fruit_by_id[fruit_id].level
                for fruit_id in motif.fruit_ids
            )
            if motif.levels != actual_levels:
                raise ValueError(
                    'chain motif levels must match its ordered fruit_ids'
                )

        for lane in self.queue_lane_analyses:
            for action_offset, blockers in enumerate(lane.blocker_ids_by_action):
                require_current_ids(
                    f'queue q{lane.queue_index} blockers at action '
                    f'{action_offset}',
                    blockers,
                )

    @property
    def action_count(self):
        """当前 V1 固定动作数量。"""

        return ANALYSIS_ACTION_COUNT

    @property
    def computed_top_connected_capacity(self):
        """按当前 ``queue_decay`` 聚合 q0-q3 的容量。"""

        weights = tuple(
            self.queue_decay ** queue_index
            for queue_index in range(QUEUE_LOOKAHEAD_COUNT)
        )
        return (
            sum(
                weight * lane.capacity
                for weight, lane in zip(weights, self.queue_lane_analyses)
            )
            / sum(weights)
        )

    @property
    def episode_key(self):
        """返回 worker 内唯一 episode 身份。"""

        return (
            self.transition_key.worker_id,
            self.transition_key.episode_id,
        )

    def get_fruit(self, fruit_id):
        """按 ID 查找当前水果；不存在时返回 ``None``。"""

        fruit_id = _integer('fruit_id', fruit_id, minimum=1)
        return next(
            (
                fruit
                for fruit in self.fruit_analyses
                if fruit.fruit_id == fruit_id
            ),
            None,
        )


__all__ = [
    'ANALYSIS_ACTION_COUNT',
    'DEFAULT_QUEUE_DECAY',
    'FULL_ACTION_MASK',
    'LANDING_DEPTH_WEIGHT',
    'QUEUE_LOOKAHEAD_COUNT',
    'SAFE_ACTION_WEIGHT',
    'STATE_ANALYSIS_SCHEMA_VERSION',
    'SUPPORT_BOUNDARIES',
    'SUPPORT_RELATIONS',
    'ChainMotif',
    'ContactInfluenceEdge',
    'FruitAnalysis',
    'PartnerComponent',
    'QueueLaneAnalysis',
    'StateAnalysis',
    'StateAnalysisDiagnostics',
    'SupportEdge',
]
