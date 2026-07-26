"""动作前稳定边界的完整静态状态分析器。

本模块把 ``GameState`` 转换为只读 ``StateAnalysis``，供后续 Reward V2 和
``AttributionTracker`` 共用。V1 有意使用两种互补近似：

- 15 条解析竖直投放列负责落点、单果可达性和可回溯 blocker；
- 最小水果探针的规范网格负责顶部连通自由空间和封闭空腔。

这里不推进 Pymunk、不修改环境，也不把分析结果写入主 ReplayBuffer。动态滚动、碰撞
影响链和跨步事件确认由后续 tracker/稀疏反事实处理。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from bisect import bisect_left, bisect_right
from collections import Counter, deque
from dataclasses import asdict, dataclass
from numbers import Real
from operator import index

from daxigua.core.rules import (
    MAX_FRUIT_LEVEL,
    MIN_FRUIT_LEVEL,
    SPAWN_FRUIT_MAX_LEVEL,
    SPAWN_FRUIT_MIN_LEVEL,
    dropped_fruit_physics_radius,
    fruit_radius,
    merged_fruit_physics_radius,
)
from daxigua.core.state import ActionCandidate, BoardGeometry, FruitState, GameState
from daxigua_rl.training.identity import TransitionKey

from .schema import (
    ANALYSIS_ACTION_COUNT,
    DEFAULT_QUEUE_DECAY,
    LANDING_DEPTH_WEIGHT,
    QUEUE_LOOKAHEAD_COUNT,
    SAFE_ACTION_WEIGHT,
    ChainMotif,
    ContactInfluenceEdge,
    FreeSpaceRegionAnalysis,
    FruitAnalysis,
    PartnerComponent,
    QueueLaneAnalysis,
    StateAnalysis,
    StateAnalysisDiagnostics,
    SupportEdge,
)


_APPROXIMATION_FLAGS = (
    'canonical_level1_grid_v1',
    'pair_midpoint_ladder_v1',
    'static_support_graph_v1',
    'vertical_drop_columns_v1',
)


def _finite(name, value, *, minimum=None, maximum=None):
    """读取有限数值并给配置错误提供明确字段名。"""

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
    """读取严格整数，避免 ``bool`` 被当成 0/1 配置。"""

    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer')
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc
    if minimum is not None and result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return result


def _clip_unit(value):
    """把几何近似产生的微小越界裁剪到 ``[0, 1]``。"""

    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True, slots=True, kw_only=True)
class StateAnalyzerConfig:
    """控制静态分析精度、容差和结构搜索预算。

    默认值针对 400x800 棋盘和几十颗水果设置。配置会进入确定性 fingerprint，
    因而训练记录可以区分不同分析口径。
    """

    action_count: int = ANALYSIS_ACTION_COUNT
    queue_decay: float = DEFAULT_QUEUE_DECAY
    safe_landing_depth: float = 0.25
    obstacle_clearance: float = 1.0
    landing_tie_tolerance: float = 1.0
    contact_gap_tolerance: float = 2.0
    wall_gap_tolerance: float = 2.0
    support_normal_y_min: float = 0.25
    grid_cell_size: float = 8.0
    grid_min_cavity_cells: int = 4
    partner_distance_factor: float = 3.0
    ladder_distance_factor: float = 2.5
    max_motifs: int = 32
    max_critical_blockers: int = 4

    def __post_init__(self):
        action_count = _strict_integer(
            'action_count',
            self.action_count,
            minimum=1,
        )
        if action_count != ANALYSIS_ACTION_COUNT:
            raise ValueError(
                f'V1 StateAnalyzer requires exactly '
                f'{ANALYSIS_ACTION_COUNT} actions'
            )
        object.__setattr__(self, 'action_count', action_count)

        for field_name in (
                'queue_decay',
                'safe_landing_depth',
                'support_normal_y_min'):
            object.__setattr__(
                self,
                field_name,
                _finite(
                    field_name,
                    getattr(self, field_name),
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
        for field_name in (
                'obstacle_clearance',
                'landing_tie_tolerance',
                'contact_gap_tolerance',
                'wall_gap_tolerance'):
            object.__setattr__(
                self,
                field_name,
                _finite(
                    field_name,
                    getattr(self, field_name),
                    minimum=0.0,
                ),
            )
        object.__setattr__(
            self,
            'grid_cell_size',
            _finite(
                'grid_cell_size',
                self.grid_cell_size,
                minimum=1.0,
            ),
        )
        for field_name in (
                'partner_distance_factor',
                'ladder_distance_factor'):
            value = _finite(
                field_name,
                getattr(self, field_name),
                minimum=1.0,
            )
            object.__setattr__(self, field_name, value)
        for field_name in (
                'grid_min_cavity_cells',
                'max_motifs',
                'max_critical_blockers'):
            object.__setattr__(
                self,
                field_name,
                _strict_integer(
                    field_name,
                    getattr(self, field_name),
                    minimum=1,
                ),
            )

    @property
    def fingerprint(self):
        """返回不依赖 Python hash 随机化的配置摘要。"""

        encoded = json.dumps(
            asdict(self),
            ensure_ascii=True,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        return f'sha256:{hashlib.sha256(encoded).hexdigest()}'


def drop_x_positions_for_level(
        geometry,
        level,
        action_count=ANALYSIS_ACTION_COUNT):
    """按游戏真实显示半径生成某等级的合法离散投放横坐标。

    物理碰撞使用 ``dropped_fruit_physics_radius``，但合法鼠标范围由显示半径决定；
    q1～q3 因此不能复用 q0 的横坐标。
    """

    if not isinstance(geometry, BoardGeometry):
        raise TypeError('geometry must be BoardGeometry')
    action_count = _strict_integer(
        'action_count',
        action_count,
        minimum=1,
    )
    level = _strict_integer('level', level, minimum=MIN_FRUIT_LEVEL)
    if level > MAX_FRUIT_LEVEL:
        raise ValueError(f'level must be <= {MAX_FRUIT_LEVEL}')

    radius = float(fruit_radius(level))
    left = float(geometry.wall_width) + radius + 2.0
    right = float(geometry.width - geometry.wall_width) - radius - 2.0
    if right < left or (action_count > 1 and right == left):
        raise ValueError('board is too narrow for the requested fruit level')
    if action_count == 1:
        return ((left + right) / 2.0,)
    step = (right - left) / (action_count - 1)
    return tuple(left + step * offset for offset in range(action_count))


@dataclass(frozen=True, slots=True)
class _FruitReachability:
    """构造最终 ``FruitAnalysis`` 前使用的内部几何结果。"""

    reachable_action_mask: int
    top_visible_ratio: float
    blockers_by_action: tuple[tuple[int, ...], ...]
    critical_blocker_ids: tuple[int, ...]
    inversion_blocker_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _GridResult:
    """规范自由空间网格的公开区域和水果邻接映射。"""

    regions: tuple[FreeSpaceRegionAnalysis, ...]
    fruit_region_ids: dict[int, int | None]


class StateAnalyzer:
    """把一个动作前 ``GameState`` 分析为完整只读快照。"""

    def __init__(self, config=None):
        if config is None:
            config = StateAnalyzerConfig()
        if not isinstance(config, StateAnalyzerConfig):
            raise TypeError('config must be StateAnalyzerConfig')
        self.config = config
        self._config_fingerprint = config.fingerprint

    def analyze(
            self,
            state,
            action_candidates,
            transition_key,
            *,
            stable_boundary=None,
            incoming_transition_key=None,
            contact_influence_edges=()):
        """分析一个动作执行前边界，不推进或修改游戏状态。

        ``transition_key.step_index`` 必须等于 ``state.step_count``。调用方若掌握
        ``PhysicsResult.stable``，可通过 ``stable_boundary`` 显式传入；否则使用
        所有 ``FruitState.stable`` 的合取作为保守判断。
        """

        started_at = time.perf_counter()
        candidates = self._validate_inputs(
            state,
            action_candidates,
            transition_key,
            stable_boundary,
        )
        contact_edges = tuple(contact_influence_edges)
        if any(
                not isinstance(edge, ContactInfluenceEdge)
                for edge in contact_edges):
            raise TypeError(
                'contact_influence_edges must contain '
                'ContactInfluenceEdge values'
            )
        fruits = tuple(sorted(state.board_fruits, key=lambda item: item.fruit_id))
        fruit_by_id = {fruit.fruit_id: fruit for fruit in fruits}

        if stable_boundary is None:
            stable_boundary = all(fruit.stable for fruit in fruits)

        # q0 采用环境原样给出的动作顺序；其它槽位按各自显示半径重新离散化。
        q0_drop_x = tuple(float(candidate.drop_x) for candidate in candidates)
        column_cache = self._build_column_cache(
            state,
            fruits,
            q0_drop_x,
        )
        lane_analyses = tuple(
            self._analyze_queue_lane(
                state,
                queue_index,
                int(level),
                (
                    q0_drop_x
                    if queue_index == 0
                    else column_cache[int(level)][0]
                ),
                column_cache[int(level)][1],
            )
            for queue_index, level in enumerate(state.fruit_queue)
        )

        # 同等级水果共享 15 条已排序投放列，避免每颗目标重新扫描全场；
        # blocker 判断仍针对每个目标独立派生。
        reachability = {
            fruit.fruit_id: self._analyze_fruit_reachability(
                fruit,
                column_cache[fruit.level][1],
                fruit_by_id,
            )
            for fruit in fruits
        }

        grid = self._analyze_free_space(state, fruits, q0_drop_x)
        support_edges = self._build_support_edges(
            state,
            fruits,
            fruit_by_id,
            reachability,
        )
        support_parents, support_children = self._support_caches(
            fruits,
            support_edges,
        )
        partner_ids, reachable_partner_ids = self._partner_graph(
            fruits,
            reachability,
        )
        partner_components = self._partner_components(
            fruits,
            reachability,
            partner_ids,
            grid.fruit_region_ids,
        )
        chain_motifs = self._chain_motifs(
            state,
            fruits,
            reachability,
            partner_ids,
        )

        fruit_analyses = tuple(
            self._fruit_analysis(
                state,
                fruit,
                reachability[fruit.fruit_id],
                partner_ids[fruit.fruit_id],
                reachable_partner_ids[fruit.fruit_id],
                support_parents[fruit.fruit_id],
                support_children[fruit.fruit_id],
                grid.fruit_region_ids[fruit.fruit_id],
            )
            for fruit in fruits
        )
        recoverability = self._recoverability(fruit_analyses)
        chain_readiness = max(
            (motif.readiness for motif in chain_motifs),
            default=0.0,
        )
        weights = tuple(
            self.config.queue_decay ** queue_index
            for queue_index in range(QUEUE_LOOKAHEAD_COUNT)
        )
        top_connected_capacity = (
            sum(
                weight * lane.capacity
                for weight, lane in zip(weights, lane_analyses)
            )
            / sum(weights)
        )

        warning_codes = ()
        if not stable_boundary:
            warning_codes = ('unstable_boundary',)
        diagnostics = StateAnalysisDiagnostics(
            stable_boundary=stable_boundary,
            valid_for_attribution=stable_boundary,
            physics_frame=state.physics_frame,
            analysis_seconds=max(0.0, time.perf_counter() - started_at),
            degraded=not stable_boundary,
            warning_codes=warning_codes,
            approximation_flags=_APPROXIMATION_FLAGS,
        )

        return StateAnalysis(
            transition_key=transition_key,
            incoming_transition_key=incoming_transition_key,
            action_indices=tuple(
                candidate.action_index
                for candidate in candidates
            ),
            action_drop_x_by_offset=q0_drop_x,
            fruit_analyses=fruit_analyses,
            support_edges=support_edges,
            contact_influence_edges=contact_edges,
            partner_components=partner_components,
            chain_motifs=chain_motifs,
            free_space_regions=grid.regions,
            queue_lane_analyses=lane_analyses,
            queue_decay=self.config.queue_decay,
            top_connected_capacity=top_connected_capacity,
            recoverability=recoverability,
            chain_readiness=chain_readiness,
            diagnostics=diagnostics,
            analyzer_config_fingerprint=self._config_fingerprint,
        )

    def _validate_inputs(
            self,
            state,
            action_candidates,
            transition_key,
            stable_boundary):
        """在昂贵分析前拒绝时间语义或布局不一致的输入。"""

        if not isinstance(state, GameState):
            raise TypeError('state must be GameState')
        if not isinstance(transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        if transition_key.step_index != state.step_count:
            raise ValueError(
                'transition_key.step_index must equal state.step_count'
            )
        if stable_boundary is not None and not isinstance(stable_boundary, bool):
            raise TypeError('stable_boundary must be bool or None')
        if not isinstance(state.geometry, BoardGeometry):
            raise TypeError('state.geometry must be BoardGeometry')
        geometry = state.geometry
        for field_name in (
                'width',
                'height',
                'spawn_y',
                'wall_width',
                'floor_y'):
            minimum = 1 if field_name in {'width', 'height'} else 0
            _strict_integer(
                f'state.geometry.{field_name}',
                getattr(geometry, field_name),
                minimum=minimum,
            )
        if geometry.width <= 2 * geometry.wall_width:
            raise ValueError('board width must contain a playable interior')
        if not 0 <= geometry.spawn_y < geometry.floor_y <= geometry.height:
            raise ValueError(
                'geometry must satisfy 0 <= spawn_y < floor_y <= height'
            )

        queue = tuple(state.fruit_queue)
        if len(queue) != QUEUE_LOOKAHEAD_COUNT:
            raise ValueError(
                f'state.fruit_queue must contain q0-q'
                f'{QUEUE_LOOKAHEAD_COUNT - 1}'
            )
        for queue_index, level in enumerate(queue):
            level = _strict_integer(
                f'state.fruit_queue[{queue_index}]',
                level,
                minimum=SPAWN_FRUIT_MIN_LEVEL,
            )
            if level > SPAWN_FRUIT_MAX_LEVEL:
                raise ValueError(
                    f'state.fruit_queue[{queue_index}] must be <= '
                    f'{SPAWN_FRUIT_MAX_LEVEL}'
                )

        fruits = tuple(state.board_fruits)
        if any(not isinstance(fruit, FruitState) for fruit in fruits):
            raise TypeError('state.board_fruits must contain FruitState values')
        fruit_ids = []
        for fruit_offset, fruit in enumerate(fruits):
            fruit_id = _strict_integer(
                f'state.board_fruits[{fruit_offset}].fruit_id',
                fruit.fruit_id,
                minimum=1,
            )
            fruit_ids.append(fruit_id)
            level = _strict_integer(
                f'state.board_fruits[{fruit_offset}].level',
                fruit.level,
                minimum=MIN_FRUIT_LEVEL,
            )
            if level > MAX_FRUIT_LEVEL:
                raise ValueError(
                    f'state.board_fruits[{fruit_offset}].level must be '
                    f'<= {MAX_FRUIT_LEVEL}'
                )
            for field_name in ('x', 'y', 'radius', 'physics_radius'):
                minimum = (
                    1e-12
                    if field_name in {'radius', 'physics_radius'}
                    else None
                )
                _finite(
                    f'state.board_fruits[{fruit_offset}].{field_name}',
                    getattr(fruit, field_name),
                    minimum=minimum,
                )
            if not isinstance(fruit.stable, bool):
                raise TypeError(
                    f'state.board_fruits[{fruit_offset}].stable must be bool'
                )
        fruit_ids = tuple(fruit_ids)
        if len(fruit_ids) != len(set(fruit_ids)):
            raise ValueError('state.board_fruits contains duplicate fruit_id')
        if state.fruit_count != len(fruits):
            raise ValueError(
                'state.fruit_count must equal len(state.board_fruits)'
            )

        candidates = tuple(action_candidates)
        if len(candidates) != self.config.action_count:
            raise ValueError(
                f'action_candidates must contain exactly '
                f'{self.config.action_count} items'
            )
        if any(
                not isinstance(candidate, ActionCandidate)
                for candidate in candidates):
            raise TypeError(
                'action_candidates must contain ActionCandidate values'
            )
        if len({candidate.action_index for candidate in candidates}) != len(
                candidates):
            raise ValueError('action candidate indices must be unique')
        drop_x = tuple(float(candidate.drop_x) for candidate in candidates)
        if any(
                left >= right
                for left, right in zip(drop_x, drop_x[1:])):
            raise ValueError(
                'action candidates must be ordered by increasing drop_x'
            )

        q0_level = int(queue[0])
        expected_x = drop_x_positions_for_level(
            state.geometry,
            q0_level,
            self.config.action_count,
        )
        expected_display_radius = float(fruit_radius(q0_level))
        expected_physics_radius = float(
            dropped_fruit_physics_radius(q0_level)
        )
        for offset, (candidate, expected_drop_x) in enumerate(
                zip(candidates, expected_x)):
            if candidate.current_level != q0_level:
                raise ValueError(
                    f'action candidate {offset} current_level must match q0'
                )
            if not math.isclose(
                    candidate.current_radius,
                    expected_display_radius,
                    rel_tol=0.0,
                    abs_tol=1e-9):
                raise ValueError(
                    f'action candidate {offset} has wrong display radius'
                )
            if not math.isclose(
                    candidate.current_physics_radius,
                    expected_physics_radius,
                    rel_tol=0.0,
                    abs_tol=1e-9):
                raise ValueError(
                    f'action candidate {offset} has wrong physics radius'
                )
            if not math.isclose(
                    candidate.drop_x,
                    expected_drop_x,
                    rel_tol=0.0,
                    abs_tol=1e-9):
                raise ValueError(
                    f'action candidate {offset} has wrong legal drop_x'
                )
        return candidates

    def _build_column_cache(self, state, fruits, q0_drop_x):
        """按实际出现的等级预计算 15 条投放列及其有序交点。"""

        levels = {
            int(level)
            for level in state.fruit_queue
        }
        levels.update(fruit.level for fruit in fruits)
        q0_level = int(state.fruit_queue[0])
        spawn_y = float(state.geometry.spawn_y)
        cache = {}
        for level in sorted(levels):
            drop_x = (
                q0_drop_x
                if level == q0_level
                else drop_x_positions_for_level(
                    state.geometry,
                    level,
                    self.config.action_count,
                )
            )
            probe_radius = float(dropped_fruit_physics_radius(level))
            cache[level] = (
                tuple(drop_x),
                tuple(
                    self._column_hits(
                        action_x,
                        probe_radius,
                        spawn_y,
                        fruits,
                    )
                    for action_x in drop_x
                ),
            )
        return cache

    def _column_hits(self, drop_x, probe_radius, spawn_y, fruits):
        """返回从 ``spawn_y`` 向下的竖直射线所遇水果接触面。"""

        hits = []
        for fruit in fruits:
            combined_radius = (
                probe_radius
                + fruit.physics_radius
                + self.config.obstacle_clearance
            )
            delta_x = abs(drop_x - fruit.x)
            if delta_x > combined_radius:
                continue
            vertical_offset = math.sqrt(
                max(0.0, combined_radius * combined_radius - delta_x * delta_x)
            )
            top_contact_y = fruit.y - vertical_offset
            bottom_contact_y = fruit.y + vertical_offset
            if bottom_contact_y < spawn_y:
                # 整段圆柱截面都在射线起点上方，未来落果向下运动不会碰到它。
                continue
            hits.append((
                max(float(spawn_y), top_contact_y),
                fruit.fruit_id,
            ))
        return tuple(sorted(hits, key=lambda item: (item[0], item[1])))

    def _analyze_queue_lane(
            self,
            state,
            queue_index,
            level,
            drop_x_by_action,
            hits_by_action):
        """估计一个队列槽位从顶部落下时的首碰深度。"""

        probe_radius = float(dropped_fruit_physics_radius(level))
        spawn_y = float(state.geometry.spawn_y)
        floor_center_y = float(state.geometry.floor_y) - probe_radius
        usable_depth = max(1e-9, floor_center_y - spawn_y)
        landing_depths = []
        blockers_by_action = []
        safe_mask = 0

        for action_offset, hits in enumerate(hits_by_action):
            first_y = min(
                (hits[0][0] if hits else floor_center_y),
                floor_center_y,
            )
            landing_y = max(spawn_y, min(floor_center_y, first_y))
            landing_depth = _clip_unit(
                (landing_y - spawn_y) / usable_depth
            )
            landing_depths.append(landing_depth)
            if landing_depth >= self.config.safe_landing_depth:
                safe_mask |= 1 << action_offset

            if hits and hits[0][0] <= floor_center_y + (
                    self.config.landing_tie_tolerance):
                blockers = tuple(
                    fruit_id
                    for contact_y, fruit_id in hits
                    if abs(contact_y - hits[0][0])
                    <= self.config.landing_tie_tolerance
                    and contact_y <= floor_center_y
                    + self.config.landing_tie_tolerance
                )
            else:
                blockers = ()
            blockers_by_action.append(blockers)

        mean_depth = sum(landing_depths) / self.config.action_count
        safe_count = safe_mask.bit_count()
        capacity = (
            LANDING_DEPTH_WEIGHT * mean_depth
            + SAFE_ACTION_WEIGHT
            * (safe_count / self.config.action_count)
        )
        return QueueLaneAnalysis(
            queue_index=queue_index,
            level=level,
            physics_radius=probe_radius,
            drop_x_by_action=tuple(drop_x_by_action),
            landing_depths_by_action=tuple(landing_depths),
            safe_action_mask=safe_mask,
            safe_action_count=safe_count,
            blocker_ids_by_action=tuple(blockers_by_action),
            capacity=capacity,
        )

    def _analyze_fruit_reachability(
            self,
            target,
            hits_by_action,
            fruit_by_id):
        """判断一个同级探针是否能沿各动作列首先接触目标水果。"""

        mask = 0
        eligible_count = 0
        blockers_by_action = []

        for action_offset, hits in enumerate(hits_by_action):
            target_hits = tuple(
                contact_y
                for contact_y, fruit_id in hits
                if fruit_id == target.fruit_id
            )
            if not target_hits:
                blockers_by_action.append(())
                continue

            eligible_count += 1
            target_contact_y = target_hits[0]
            blockers = tuple(
                fruit_id
                for contact_y, fruit_id in hits
                if fruit_id != target.fruit_id
                and contact_y
                < target_contact_y - self.config.landing_tie_tolerance
            )
            blockers_by_action.append(blockers)
            if not blockers:
                mask |= 1 << action_offset

        blocked_paths = tuple(
            (action_offset, blockers)
            for action_offset, blockers in enumerate(blockers_by_action)
            if blockers
        )
        # ``critical`` 表示目标已经失去最后入口时的最小静态割集。目标仍有
        # 任一开放列时，逐列 blocker 已足够描述渐进损失，不能提前把某颗水果
        # 升格为最终封口者并生成 caps 关系。
        if mask == 0 and blocked_paths:
            _, critical = min(
                blocked_paths,
                key=lambda item: (
                    len(item[1]),
                    item[0],
                    item[1],
                ),
            )
            critical_blocker_ids = tuple(
                critical[:self.config.max_critical_blockers]
            )
        else:
            critical_blocker_ids = ()

        inversion_blocker_ids = tuple(sorted({
            blocker_id
            for blockers in blockers_by_action
            for blocker_id in blockers
            if (
                fruit_by_id[blocker_id].level > target.level
                and fruit_by_id[blocker_id].y < target.y
            )
        }))
        top_visible_ratio = (
            mask.bit_count() / eligible_count
            if eligible_count
            else 0.0
        )
        return _FruitReachability(
            reachable_action_mask=mask,
            top_visible_ratio=top_visible_ratio,
            blockers_by_action=tuple(blockers_by_action),
            critical_blocker_ids=critical_blocker_ids,
            inversion_blocker_ids=inversion_blocker_ids,
        )

    def _build_support_edges(
            self,
            state,
            fruits,
            fruit_by_id,
            reachability):
        """从静态接触几何构造边界、支撑、盖压和桥接关系。"""

        edges = []
        tolerance = self.config.wall_gap_tolerance
        geometry = state.geometry

        for fruit in fruits:
            floor_gap = (
                float(geometry.floor_y) - fruit.physics_radius - fruit.y
            )
            if floor_gap <= tolerance:
                edges.append(SupportEdge(
                    boundary='floor',
                    supported_fruit_id=fruit.fruit_id,
                    relation='supports',
                    confidence=self._gap_confidence(floor_gap, tolerance),
                ))

            left_gap = (
                fruit.x
                - (float(geometry.wall_width) + fruit.physics_radius)
            )
            if left_gap <= tolerance:
                edges.append(SupportEdge(
                    boundary='left_wall',
                    supported_fruit_id=fruit.fruit_id,
                    relation='wall_constraint',
                    confidence=self._gap_confidence(left_gap, tolerance),
                ))
            right_gap = (
                float(geometry.width - geometry.wall_width)
                - fruit.physics_radius
                - fruit.x
            )
            if right_gap <= tolerance:
                edges.append(SupportEdge(
                    boundary='right_wall',
                    supported_fruit_id=fruit.fruit_id,
                    relation='wall_constraint',
                    confidence=self._gap_confidence(right_gap, tolerance),
                ))

        support_parents = {fruit.fruit_id: [] for fruit in fruits}
        for left_offset, left in enumerate(fruits):
            for right in fruits[left_offset + 1:]:
                delta_x = right.x - left.x
                delta_y = right.y - left.y
                distance = math.hypot(delta_x, delta_y)
                gap = distance - left.physics_radius - right.physics_radius
                if gap > self.config.contact_gap_tolerance:
                    continue
                if math.isclose(delta_y, 0.0, abs_tol=1e-9):
                    continue

                lower, upper = (
                    (right, left)
                    if right.y > left.y
                    else (left, right)
                )
                vertical_fraction = abs(delta_y) / max(distance, 1e-9)
                if vertical_fraction < self.config.support_normal_y_min:
                    continue
                edges.append(SupportEdge(
                    supporter_fruit_id=lower.fruit_id,
                    supported_fruit_id=upper.fruit_id,
                    relation='supports',
                    confidence=self._gap_confidence(
                        gap,
                        self.config.contact_gap_tolerance,
                    ),
                ))
                support_parents[upper.fruit_id].append(lower.fruit_id)

        # 一个上方水果同时落在左右两个下方支点上时，额外保留桥接语义；
        # canonical parent/child 缓存仍只由 ``supports`` 边生成。
        for upper_id, parent_ids in support_parents.items():
            upper = fruit_by_id[upper_id]
            left_parents = tuple(
                parent_id
                for parent_id in parent_ids
                if fruit_by_id[parent_id].x < upper.x
            )
            right_parents = tuple(
                parent_id
                for parent_id in parent_ids
                if fruit_by_id[parent_id].x > upper.x
            )
            if left_parents and right_parents:
                for parent_id in sorted(set(left_parents + right_parents)):
                    edges.append(SupportEdge(
                        supporter_fruit_id=parent_id,
                        supported_fruit_id=upper_id,
                        relation='bridges',
                        confidence=0.8,
                    ))

        # 只有解析路径中真实出现的关键上方 blocker 才记为盖压，避免把“视觉上
        # 在附近”误当作封路机制。
        for target in fruits:
            for blocker_id in reachability[
                    target.fruit_id].critical_blocker_ids:
                blocker = fruit_by_id[blocker_id]
                if blocker.y >= target.y:
                    continue
                edges.append(SupportEdge(
                    supporter_fruit_id=blocker_id,
                    supported_fruit_id=target.fruit_id,
                    relation='caps',
                    confidence=0.8,
                ))
        return tuple(edges)

    @staticmethod
    def _gap_confidence(gap, tolerance):
        """把接触间隙映射成保守但非零的几何置信度。"""

        if gap <= 0.0 or tolerance <= 0.0:
            return 1.0
        return _clip_unit(1.0 - 0.5 * gap / tolerance)

    @staticmethod
    def _support_caches(fruits, edges):
        """从 canonical ``supports`` 边反向生成冗余查询缓存。"""

        parents = {fruit.fruit_id: set() for fruit in fruits}
        children = {fruit.fruit_id: set() for fruit in fruits}
        for edge in edges:
            if (
                    edge.relation != 'supports'
                    or edge.supporter_fruit_id is None):
                continue
            parents[edge.supported_fruit_id].add(edge.supporter_fruit_id)
            children[edge.supporter_fruit_id].add(edge.supported_fruit_id)
        return (
            {
                fruit_id: tuple(sorted(parent_ids))
                for fruit_id, parent_ids in parents.items()
            },
            {
                fruit_id: tuple(sorted(child_ids))
                for fruit_id, child_ids in children.items()
            },
        )

    def _partner_graph(self, fruits, reachability):
        """构造同等级、局部且存在接触或共享入口的无向伙伴图。"""

        partners = {fruit.fruit_id: set() for fruit in fruits}
        reachable_partners = {fruit.fruit_id: set() for fruit in fruits}
        for left_offset, left in enumerate(fruits):
            for right in fruits[left_offset + 1:]:
                if left.level != right.level:
                    continue
                distance = math.hypot(
                    right.x - left.x,
                    right.y - left.y,
                )
                combined_radius = (
                    left.physics_radius + right.physics_radius
                )
                touching = (
                    distance
                    <= combined_radius + self.config.contact_gap_tolerance
                )
                shared_mask = (
                    reachability[left.fruit_id].reachable_action_mask
                    & reachability[right.fruit_id].reachable_action_mask
                )
                locally_reachable = (
                    bool(shared_mask)
                    and distance
                    <= self.config.partner_distance_factor * combined_radius
                )
                if not touching and not locally_reachable:
                    continue
                partners[left.fruit_id].add(right.fruit_id)
                partners[right.fruit_id].add(left.fruit_id)
                if shared_mask:
                    reachable_partners[left.fruit_id].add(right.fruit_id)
                    reachable_partners[right.fruit_id].add(left.fruit_id)

        return (
            {
                fruit_id: tuple(sorted(values))
                for fruit_id, values in partners.items()
            },
            {
                fruit_id: tuple(sorted(values))
                for fruit_id, values in reachable_partners.items()
            },
        )

    @staticmethod
    def _partner_components(
            fruits,
            reachability,
            partner_ids,
            fruit_region_ids):
        """按伙伴图连通分量生成确定性组件。"""

        fruit_by_id = {fruit.fruit_id: fruit for fruit in fruits}
        visited = set()
        raw_components = []
        for fruit_id in sorted(fruit_by_id):
            if fruit_id in visited or not partner_ids[fruit_id]:
                continue
            stack = [fruit_id]
            members = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                members.append(current)
                stack.extend(
                    partner_id
                    for partner_id in reversed(partner_ids[current])
                    if partner_id not in visited
                )
            raw_components.append(tuple(sorted(members)))

        components = []
        for component_id, members in enumerate(sorted(raw_components)):
            mask = 0
            region_ids = set()
            for fruit_id in members:
                mask |= reachability[fruit_id].reachable_action_mask
                region_id = fruit_region_ids[fruit_id]
                if region_id is not None:
                    region_ids.add(region_id)
            connected_region_id = (
                next(iter(region_ids))
                if len(region_ids) == 1
                else None
            )
            components.append(PartnerComponent(
                component_id=component_id,
                level=fruit_by_id[members[0]].level,
                fruit_ids=members,
                reachable_action_mask=mask,
                top_connected=bool(mask),
                connected_region_id=connected_region_id,
            ))
        return tuple(components)

    def _chain_motifs(
            self,
            state,
            fruits,
            reachability,
            partner_ids):
        """从伙伴边和邻近的下一等级水果提取轻量连锁 motif。"""

        fruit_by_id = {fruit.fruit_id: fruit for fruit in fruits}
        motifs = []
        for left in fruits:
            for right_id in partner_ids[left.fruit_id]:
                if left.fruit_id >= right_id:
                    continue
                right = fruit_by_id[right_id]
                shared_mask = (
                    reachability[left.fruit_id].reachable_action_mask
                    & reachability[right_id].reachable_action_mask
                )
                combined_radius = (
                    left.physics_radius + right.physics_radius
                )
                distance = math.hypot(
                    right.x - left.x,
                    right.y - left.y,
                )
                touching = (
                    distance
                    <= combined_radius + self.config.contact_gap_tolerance
                )
                # 对已经接触的 pair，能够到达任一成员的动作都可能成为机械
                # 触发；尚未接触的 pair 则必须共享同一局部入口。
                trigger_mask = (
                    shared_mask
                    if shared_mask
                    else (
                        reachability[
                            left.fruit_id].reachable_action_mask
                        | reachability[right_id].reachable_action_mask
                        if touching
                        else 0
                    )
                )
                compatible_queue_indices = tuple(
                    queue_index
                    for queue_index, level in enumerate(state.fruit_queue)
                    if level == left.level
                )
                # Chain readiness 只描述当前 q0-q3 能实际触发的结构。单纯贴近但
                # 已与顶部隔绝的同级水果仍保留 partner 关系，不获得正 K。
                if not trigger_mask or not compatible_queue_indices:
                    continue
                queue_score = max(
                    (
                        self.config.queue_decay ** queue_index
                        for queue_index in compatible_queue_indices
                    ),
                    default=0.0,
                )
                max_distance = (
                    self.config.partner_distance_factor * combined_radius
                )
                if distance <= combined_radius:
                    proximity = 1.0
                else:
                    proximity = _clip_unit(
                        (max_distance - distance)
                        / max(1e-9, max_distance - combined_radius)
                    )
                trigger_fraction = (
                    trigger_mask.bit_count() / self.config.action_count
                )
                pair_readiness = _clip_unit(
                    0.45 * proximity
                    + 0.35 * trigger_fraction
                    + 0.20 * queue_score
                )
                pair_ids = tuple(sorted((left.fruit_id, right_id)))
                motifs.append(ChainMotif(
                    motif_type='merge_pair',
                    fruit_ids=pair_ids,
                    levels=(left.level, left.level),
                    depth=1,
                    trigger_action_mask=trigger_mask,
                    compatible_queue_indices=compatible_queue_indices,
                    readiness=pair_readiness,
                ))

                if left.level >= MAX_FRUIT_LEVEL:
                    continue
                midpoint_x = (left.x + right.x) / 2.0
                midpoint_y = (left.y + right.y) / 2.0
                next_level = left.level + 1
                ladder_candidates = tuple(
                    fruit
                    for fruit in fruits
                    if fruit.level == next_level
                )
                if not ladder_candidates:
                    continue
                next_fruit = min(
                    ladder_candidates,
                    key=lambda fruit: (
                        math.hypot(
                            fruit.x - midpoint_x,
                            fruit.y - midpoint_y,
                        ),
                        fruit.fruit_id,
                    ),
                )
                expected_merged_radius = float(
                    merged_fruit_physics_radius(next_level)
                )
                ladder_scale = (
                    expected_merged_radius + next_fruit.physics_radius
                )
                ladder_distance = math.hypot(
                    next_fruit.x - midpoint_x,
                    next_fruit.y - midpoint_y,
                )
                ladder_limit = (
                    self.config.ladder_distance_factor * ladder_scale
                )
                if ladder_distance > ladder_limit:
                    continue
                ladder_proximity = _clip_unit(
                    1.0 - ladder_distance / max(1e-9, ladder_limit)
                )
                ladder_readiness = _clip_unit(
                    0.55 * pair_readiness
                    + 0.30 * ladder_proximity
                    + 0.15 * queue_score
                )
                motifs.append(ChainMotif(
                    motif_type='level_ladder',
                    fruit_ids=pair_ids + (next_fruit.fruit_id,),
                    levels=(left.level, left.level, next_level),
                    depth=2,
                    trigger_action_mask=trigger_mask,
                    compatible_queue_indices=compatible_queue_indices,
                    readiness=ladder_readiness,
                ))

        motifs.sort(
            key=lambda motif: (
                -motif.readiness,
                motif.signature,
            )
        )
        return tuple(motifs[:self.config.max_motifs])

    def _fruit_analysis(
            self,
            state,
            fruit,
            reachability,
            partner_ids,
            reachable_partner_ids,
            support_parent_ids,
            supported_child_ids,
            connected_region_id):
        """组合一颗水果的几何、关系和可恢复性输入字段。"""

        depth_denominator = max(
            1e-9,
            float(state.geometry.floor_y - state.geometry.spawn_y),
        )
        burial_depth = _clip_unit(
            (fruit.y - state.geometry.spawn_y) / depth_denominator
        )

        # 只要仍有直接入口，就存在未来投放形成同级伙伴的局部路径；这与
        # ``bool(partner_ids)`` 不同，也避免把开放区域中的孤立水果误判成埋死。
        partner_reachable = bool(
            reachability.reachable_action_mask
            or reachable_partner_ids
        )
        return FruitAnalysis(
            fruit_id=fruit.fruit_id,
            level=fruit.level,
            physics_radius=fruit.physics_radius,
            probe_physics_radius=dropped_fruit_physics_radius(fruit.level),
            reachable_action_mask=reachability.reachable_action_mask,
            reachable_action_count=(
                reachability.reachable_action_mask.bit_count()
            ),
            top_visible_ratio=reachability.top_visible_ratio,
            top_blocker_ids_by_action=reachability.blockers_by_action,
            partner_ids=partner_ids,
            partner_reachable=partner_reachable,
            support_parent_ids=support_parent_ids,
            supported_child_ids=supported_child_ids,
            burial_depth=burial_depth,
            inversion_count=len(reachability.inversion_blocker_ids),
            connected_region_id=connected_region_id,
            reachable_partner_ids=reachable_partner_ids,
            critical_blocker_ids=reachability.critical_blocker_ids,
            inversion_blocker_ids=reachability.inversion_blocker_ids,
        )

    @staticmethod
    def _recoverability(fruit_analyses):
        """按低等级加权的永久占位负担计算 ``R(s)``。"""

        weighted_burden = 0.0
        total_weight = 0.0
        for fruit in fruit_analyses:
            if fruit.level >= MAX_FRUIT_LEVEL:
                continue
            weight = 2.0 ** (-(fruit.level - 1) / 2.0)
            total_weight += weight
            weighted_burden += (
                weight
                * (1.0 - fruit.reachable_fraction)
                * (0.0 if fruit.partner_reachable else 1.0)
                * fruit.burial_depth
            )
        if total_weight == 0.0:
            return 1.0
        burden = _clip_unit(weighted_burden / total_weight)
        return 1.0 - burden

    def _analyze_free_space(self, state, fruits, q0_drop_x):
        """用最小水果探针网格分割顶部连通空间和封闭空腔。"""

        geometry = state.geometry
        probe_radius = float(
            dropped_fruit_physics_radius(MIN_FRUIT_LEVEL)
        )
        x_min = float(geometry.wall_width) + probe_radius
        x_max = float(geometry.width - geometry.wall_width) - probe_radius
        y_min = float(geometry.spawn_y)
        y_max = float(geometry.floor_y) - probe_radius
        if x_max <= x_min or y_max <= y_min:
            raise ValueError('board has no legal free-space probe domain')

        x_values = self._grid_axis(
            x_min,
            x_max,
            self.config.grid_cell_size,
        )
        y_values = self._grid_axis(
            y_min,
            y_max,
            self.config.grid_cell_size,
        )
        column_count = len(x_values)
        row_count = len(y_values)
        cell_count = column_count * row_count
        owners = [None] * cell_count

        # 把每个圆障碍按探针半径和半个栅格对角线膨胀。后者让网格判断
        # 保守：不会仅因采样点恰好落在窄缝中就把不可穿越通道判为开放。
        half_diagonal = self.config.grid_cell_size / math.sqrt(2.0)
        for fruit in fruits:
            inflated_radius = (
                fruit.physics_radius
                + probe_radius
                + self.config.obstacle_clearance
                + half_diagonal
            )
            min_column = max(
                0,
                bisect_left(x_values, fruit.x - inflated_radius),
            )
            max_column = min(
                column_count,
                bisect_right(x_values, fruit.x + inflated_radius),
            )
            min_row = max(
                0,
                bisect_left(y_values, fruit.y - inflated_radius),
            )
            max_row = min(
                row_count,
                bisect_right(y_values, fruit.y + inflated_radius),
            )
            radius_squared = inflated_radius * inflated_radius
            for row in range(min_row, max_row):
                delta_y_squared = (y_values[row] - fruit.y) ** 2
                for column in range(min_column, max_column):
                    if (
                            (x_values[column] - fruit.x) ** 2
                            + delta_y_squared
                            <= radius_squared):
                        cell_index = row * column_count + column
                        if owners[cell_index] is None:
                            owners[cell_index] = {fruit.fruit_id}
                        else:
                            owners[cell_index].add(fruit.fruit_id)

        labels = [-1] * cell_count
        for cell_index, cell_owners in enumerate(owners):
            if cell_owners:
                labels[cell_index] = -2

        components = []
        for seed in range(cell_count):
            if labels[seed] != -1:
                continue
            component_id = seed
            labels[seed] = component_id
            queue = deque((seed,))
            members = []
            while queue:
                current = queue.popleft()
                members.append(current)
                row, column = divmod(current, column_count)
                for neighbor in self._grid_neighbors(
                        row,
                        column,
                        row_count,
                        column_count):
                    if labels[neighbor] != -1:
                        continue
                    labels[neighbor] = component_id
                    queue.append(neighbor)
            components.append((component_id, tuple(members)))

        component_by_id = {
            component_id: members
            for component_id, members in components
        }
        top_connected_ids = {
            labels[column]
            for column in range(column_count)
            if labels[column] >= 0
        }
        emitted_ids = {
            component_id
            for component_id, members in components
            if (
                component_id in top_connected_ids
                or len(members) >= self.config.grid_min_cavity_cells
            )
        }

        action_mask_by_region = Counter()
        for action_offset, drop_x in enumerate(q0_drop_x):
            column = self._nearest_axis_index(x_values, drop_x)
            component_id = labels[column]
            if (
                    component_id >= 0
                    and component_id in top_connected_ids
                    and component_id in emitted_ids):
                action_mask_by_region[component_id] |= 1 << action_offset

        region_boundary_ids = {
            component_id: set()
            for component_id in emitted_ids
        }
        fruit_region_contacts = {
            fruit.fruit_id: Counter()
            for fruit in fruits
        }
        for component_id in emitted_ids:
            for cell_index in component_by_id[component_id]:
                row, column = divmod(cell_index, column_count)
                for neighbor in self._grid_neighbors(
                        row,
                        column,
                        row_count,
                        column_count):
                    if not owners[neighbor]:
                        continue
                    for fruit_id in owners[neighbor]:
                        region_boundary_ids[component_id].add(fruit_id)
                        fruit_region_contacts[fruit_id][component_id] += 1

        regions = []
        for component_id, members in components:
            if component_id not in emitted_ids:
                continue
            rows = []
            columns = []
            for cell_index in members:
                row, column = divmod(cell_index, column_count)
                rows.append(row)
                columns.append(column)
            normalized_x = tuple(
                _clip_unit(
                    (x_values[column] - x_min) / (x_max - x_min)
                )
                for column in columns
            )
            normalized_y = tuple(
                _clip_unit(
                    (y_values[row] - y_min) / (y_max - y_min)
                )
                for row in rows
            )
            top_connected = component_id in top_connected_ids
            regions.append(FreeSpaceRegionAnalysis(
                region_id=component_id,
                top_connected=top_connected,
                reachable_action_mask=(
                    int(action_mask_by_region[component_id])
                    if top_connected
                    else 0
                ),
                cell_count=len(members),
                area_ratio=len(members) / cell_count,
                centroid_x=sum(normalized_x) / len(normalized_x),
                centroid_y=sum(normalized_y) / len(normalized_y),
                min_x=min(normalized_x),
                max_x=max(normalized_x),
                min_y=min(normalized_y),
                max_y=max(normalized_y),
                boundary_fruit_ids=tuple(
                    sorted(region_boundary_ids[component_id])
                ),
                touches_left_wall=0 in columns,
                touches_right_wall=column_count - 1 in columns,
                touches_floor=row_count - 1 in rows,
            ))

        top_connected_by_id = {
            region.region_id: region.top_connected
            for region in regions
        }
        fruit_region_ids = {}
        for fruit in fruits:
            contacts = fruit_region_contacts[fruit.fruit_id]
            if not contacts:
                fruit_region_ids[fruit.fruit_id] = None
                continue
            fruit_region_ids[fruit.fruit_id] = max(
                contacts,
                key=lambda component_id: (
                    top_connected_by_id[component_id],
                    contacts[component_id],
                    -component_id,
                ),
            )
        return _GridResult(
            regions=tuple(regions),
            fruit_region_ids=fruit_region_ids,
        )

    @staticmethod
    def _grid_axis(start, end, target_cell_size):
        """生成包含两个端点且近似指定间距的对称坐标轴。"""

        interval_count = max(
            1,
            int(math.ceil((end - start) / target_cell_size)),
        )
        step = (end - start) / interval_count
        return tuple(
            start + step * offset
            for offset in range(interval_count + 1)
        )

    @staticmethod
    def _nearest_axis_index(values, target):
        """在递增轴上返回离目标最近的坐标索引。"""

        right = bisect_left(values, target)
        if right <= 0:
            return 0
        if right >= len(values):
            return len(values) - 1
        left = right - 1
        if abs(values[left] - target) <= abs(values[right] - target):
            return left
        return right

    @staticmethod
    def _grid_neighbors(row, column, row_count, column_count):
        """按 4 邻域返回确定性相邻索引。"""

        if row > 0:
            yield (row - 1) * column_count + column
        if column > 0:
            yield row * column_count + column - 1
        if column + 1 < column_count:
            yield row * column_count + column + 1
        if row + 1 < row_count:
            yield (row + 1) * column_count + column


__all__ = [
    'StateAnalyzer',
    'StateAnalyzerConfig',
    'drop_x_positions_for_level',
]
