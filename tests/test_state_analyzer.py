"""StateAnalyzer 的人工几何场景测试。

这些测试只构造公开 ``GameState`` 快照，不依赖 Pymunk 的落果与稳定过程。这样既能把
几何预期写成可复核的测试 oracle，也避免不同物理模式造成非确定性。
"""

from __future__ import annotations

import math
import unittest

from daxigua.core.rules import (
    dropped_fruit_physics_radius,
    fruit_radius,
)
from daxigua.core.state import (
    ActionCandidate,
    BoardGeometry,
    FruitState,
    GameState,
)
from daxigua_rl.attribution import (
    ANALYSIS_ACTION_COUNT,
    FULL_ACTION_MASK,
    StateAnalyzer,
    StateAnalyzerConfig,
    drop_x_positions_for_level,
)
from daxigua_rl.attribution.state_analyzer import _bounded_mean
from daxigua_rl.training.identity import TransitionKey


GEOMETRY = BoardGeometry(
    width=400,
    height=800,
    spawn_y=180,
    wall_width=20,
    floor_y=780,
)


class StableGeometryPrimitiveTest(unittest.TestCase):
    """覆盖只在退化单行/单列区域出现的浮点边界回归。"""

    def test_bounded_mean_of_repeated_coordinate_stays_on_coordinate(self):
        coordinate = 21.0 / 211.0

        self.assertEqual(
            _bounded_mean((coordinate, coordinate, coordinate)),
            coordinate,
        )


def _fruit(
        fruit_id,
        level,
        x,
        y,
        *,
        physics_radius=None,
        stable=True):
    """用真实规则半径构造一颗静止水果。"""

    display_radius = float(fruit_radius(level))
    actual_radius = float(
        dropped_fruit_physics_radius(level)
        if physics_radius is None
        else physics_radius
    )
    return FruitState(
        fruit_id=fruit_id,
        level=level,
        radius=display_radius,
        physics_radius=actual_radius,
        x=float(x),
        y=float(y),
        vx=0.0,
        vy=0.0,
        angle=0.0,
        angular_velocity=0.0,
        age_frames=0,
        stable=stable,
        distance_to_left_wall=float(
            x - (GEOMETRY.wall_width + actual_radius)
        ),
        distance_to_right_wall=float(
            (GEOMETRY.width - GEOMETRY.wall_width - actual_radius) - x
        ),
        distance_to_floor=float(
            (GEOMETRY.floor_y - actual_radius) - y
        ),
        distance_to_danger_line=float(
            (y - actual_radius) - GEOMETRY.spawn_y
        ),
    )


def _state(
        fruits=(),
        *,
        queue=(1, 2, 3, 4),
        step_count=0,
        physics_frame=120):
    """构造字段内部一致的稳定游戏状态。"""

    fruits = tuple(fruits)
    max_level = max((fruit.level for fruit in fruits), default=0)
    highest_top = min(
        (fruit.y - fruit.physics_radius for fruit in fruits),
        default=GEOMETRY.height,
    )
    playable_area = GEOMETRY.width * (GEOMETRY.height - GEOMETRY.spawn_y)
    occupied_area = sum(
        math.pi * fruit.physics_radius ** 2
        for fruit in fruits
    )
    return GameState(
        board_fruits=fruits,
        fruit_queue=tuple(queue),
        score=0,
        last_score=0,
        step_count=step_count,
        physics_frame=physics_frame,
        done=False,
        geometry=GEOMETRY,
        max_height=(
            0.0
            if not fruits
            else float(GEOMETRY.height - highest_top)
        ),
        fruit_count=len(fruits),
        max_level=max_level,
        empty_space_ratio=max(
            0.0,
            min(1.0, 1.0 - occupied_area / playable_area),
        ),
    )


def _actions(level=1, *, first_index=0):
    """按游戏公开规则生成当前规范数量的动作候选。"""

    positions = drop_x_positions_for_level(
        GEOMETRY,
        level,
        action_count=ANALYSIS_ACTION_COUNT,
    )
    left = positions[0]
    right = positions[-1]
    display_radius = float(fruit_radius(level))
    physics_radius = float(dropped_fruit_physics_radius(level))
    return tuple(
        ActionCandidate(
            action_index=first_index + offset,
            drop_x=drop_x,
            normalized_drop_x=(
                0.0
                if right == left
                else (drop_x - left) / (right - left)
            ),
            current_level=level,
            current_radius=display_radius,
            current_physics_radius=physics_radius,
        )
        for offset, drop_x in enumerate(positions)
    )


def _analyze(state, *, actions=None, stable_boundary=True):
    """用一组偏细但仍足够快的确定性配置分析状态。"""

    analyzer = StateAnalyzer(
        StateAnalyzerConfig(
            grid_cell_size=8.0,
        )
    )
    return analyzer.analyze(
        state,
        _actions(state.fruit_queue[0]) if actions is None else actions,
        TransitionKey(0, 0, state.step_count),
        stable_boundary=stable_boundary,
    )


def _mask(*offsets):
    """把动作 offset 集合转换为当前规范位数的掩码。"""

    return sum(1 << offset for offset in offsets)


def _mirror_mask(mask):
    """左右翻转固定宽度的动作掩码。"""

    result = 0
    for offset in range(ANALYSIS_ACTION_COUNT):
        if mask & (1 << offset):
            result |= 1 << (ANALYSIS_ACTION_COUNT - 1 - offset)
    return result


def _mirror_fruit(fruit):
    """围绕棋盘中线镜像水果，同时保持稳定性和 ID。"""

    return _fruit(
        fruit.fruit_id,
        fruit.level,
        GEOMETRY.width - fruit.x,
        fruit.y,
        physics_radius=fruit.physics_radius,
        stable=fruit.stable,
    )


class StateAnalyzerEmptyBoardTest(unittest.TestCase):
    """验证空盘是容量和可恢复性的规范基线。"""

    def test_empty_board_has_full_capacity_for_all_four_queue_slots(self):
        analysis = _analyze(_state())

        self.assertEqual(len(analysis.queue_lane_analyses), 4)
        self.assertEqual(
            analysis.action_indices,
            tuple(range(ANALYSIS_ACTION_COUNT)),
        )
        self.assertEqual(
            analysis.action_drop_x_by_offset,
            tuple(action.drop_x for action in _actions()),
        )
        for queue_index, lane in enumerate(analysis.queue_lane_analyses):
            level = queue_index + 1
            self.assertEqual(lane.queue_index, queue_index)
            self.assertEqual(lane.level, level)
            self.assertEqual(
                lane.drop_x_by_action,
                drop_x_positions_for_level(GEOMETRY, level),
            )
            self.assertEqual(lane.safe_action_mask, FULL_ACTION_MASK)
            self.assertEqual(lane.safe_action_count, ANALYSIS_ACTION_COUNT)
            self.assertTrue(
                all(
                    math.isclose(depth, 1.0, abs_tol=1e-12)
                    for depth in lane.landing_depths_by_action
                )
            )
            self.assertTrue(
                all(not blockers for blockers in lane.blocker_ids_by_action)
            )
            self.assertAlmostEqual(lane.capacity, 1.0)

        self.assertAlmostEqual(analysis.top_connected_capacity, 1.0)
        self.assertAlmostEqual(analysis.recoverability, 1.0)
        self.assertAlmostEqual(analysis.chain_readiness, 0.0)
        self.assertEqual(analysis.fruit_analyses, ())
        self.assertTrue(analysis.diagnostics.valid_for_attribution)
        self.assertEqual(len(analysis.free_space_regions), 1)
        self.assertTrue(analysis.free_space_regions[0].top_connected)
        self.assertEqual(
            analysis.free_space_regions[0].reachable_action_mask,
            FULL_ACTION_MASK,
        )
        self.assertAlmostEqual(
            analysis.top_connected_free_space_ratio,
            1.0,
        )
        self.assertEqual(analysis.sealed_cavity_count, 0)

    def test_drop_positions_use_each_level_display_radius(self):
        level_one = drop_x_positions_for_level(GEOMETRY, 1)
        level_four = drop_x_positions_for_level(GEOMETRY, 4)

        self.assertEqual(len(level_one), ANALYSIS_ACTION_COUNT)
        self.assertEqual(level_one[0], 42.0)
        self.assertEqual(level_one[-1], 358.0)
        self.assertGreater(level_four[0], level_one[0])
        self.assertLess(level_four[-1], level_one[-1])
        self.assertTrue(
            all(
                left < right
                for left, right in zip(level_four, level_four[1:])
            )
        )


class StateAnalyzerReachabilityTest(unittest.TestCase):
    """验证解析竖直通道的可达、阻挡和层级倒置语义。"""

    def test_floor_level_one_fruit_is_reachable_from_nearby_columns(self):
        target = _fruit(1, 1, 200, 760)

        analysis = _analyze(_state((target,)))
        fruit = analysis.get_fruit(1)

        self.assertIsNotNone(fruit)
        self.assertEqual(
            fruit.reachable_action_mask,
            _mask(8, 9, 10, 11, 12),
        )
        self.assertEqual(fruit.reachable_action_count, 5)
        self.assertAlmostEqual(fruit.top_visible_ratio, 1.0)
        self.assertTrue(fruit.partner_reachable)
        self.assertEqual(fruit.inversion_count, 0)
        self.assertTrue(
            all(
                not blockers
                for blockers in fruit.top_blocker_ids_by_action
            )
        )

    def test_higher_fruit_blocks_low_target_and_is_recorded_as_inversion(self):
        target = _fruit(1, 1, 200, 740)
        blocker = _fruit(2, 3, 200, 500)

        analysis = _analyze(_state((target, blocker)))
        fruit = analysis.get_fruit(1)

        self.assertEqual(fruit.reachable_action_mask, 0)
        self.assertEqual(fruit.reachable_action_count, 0)
        self.assertFalse(fruit.partner_reachable)
        self.assertIn(2, fruit.critical_blocker_ids)
        self.assertEqual(fruit.inversion_blocker_ids, (2,))
        self.assertEqual(fruit.inversion_count, 1)
        self.assertLess(analysis.recoverability, 1.0)
        self.assertTrue(
            any(2 in blockers for blockers in fruit.top_blocker_ids_by_action)
        )
        self.assertIn(
            (2, None, 1, 'caps'),
            {
                (
                    edge.supporter_fruit_id,
                    edge.boundary,
                    edge.supported_fruit_id,
                    edge.relation,
                )
                for edge in analysis.support_edges
            },
        )

    def test_queue_lane_records_all_tied_first_contact_blockers(self):
        left = _fruit(1, 1, 170, 500)
        right = _fruit(2, 1, 230, 500)

        analysis = _analyze(_state((right, left)))
        q0 = analysis.queue_lane_analyses[0]

        self.assertEqual(q0.blocker_ids_by_action[10], (1, 2))
        self.assertLess(q0.landing_depths_by_action[10], 1.0)

    def test_partial_path_loss_does_not_create_final_cap_evidence(self):
        target = _fruit(1, 1, 200, 740)
        partial_blocker = _fruit(2, 2, 170, 500)

        analysis = _analyze(_state((target, partial_blocker)))
        fruit = analysis.get_fruit(1)

        self.assertEqual(fruit.reachable_action_mask, _mask(12))
        self.assertEqual(fruit.critical_blocker_ids, ())
        self.assertNotIn(
            (2, 1, 'caps'),
            {
                (
                    edge.supporter_fruit_id,
                    edge.supported_fruit_id,
                    edge.relation,
                )
                for edge in analysis.support_edges
            },
        )

    def test_fruit_entirely_above_spawn_line_is_not_a_downward_blocker(self):
        target = _fruit(1, 1, 200, 760)
        above_spawn = _fruit(2, 1, 200, 100)

        analysis = _analyze(_state((target, above_spawn)))
        fruit = analysis.get_fruit(1)

        self.assertEqual(
            fruit.reachable_action_mask,
            _mask(8, 9, 10, 11, 12),
        )
        self.assertTrue(
            all(
                2 not in blockers
                for blockers in fruit.top_blocker_ids_by_action
            )
        )
        self.assertTrue(
            all(
                2 not in blockers
                for blockers
                in analysis.queue_lane_analyses[0].blocker_ids_by_action
            )
        )


class StateAnalyzerStructureTest(unittest.TestCase):
    """验证支撑、伙伴、motif 和规范自由空间输出。"""

    def test_floor_wall_and_fruit_support_edges_have_canonical_direction(self):
        lower = _fruit(1, 2, 120, 750)
        upper = _fruit(2, 1, 120, 698)
        wall_fruit = _fruit(3, 1, 40, 600)

        analysis = _analyze(_state((upper, wall_fruit, lower)))
        edge_keys = {
            (
                edge.supporter_fruit_id,
                edge.boundary,
                edge.supported_fruit_id,
                edge.relation,
            )
            for edge in analysis.support_edges
        }

        self.assertIn((None, 'floor', 1, 'supports'), edge_keys)
        self.assertIn((None, 'left_wall', 3, 'wall_constraint'), edge_keys)
        self.assertIn((1, None, 2, 'supports'), edge_keys)
        self.assertEqual(analysis.get_fruit(1).supported_child_ids, (2,))
        self.assertEqual(analysis.get_fruit(2).support_parent_ids, (1,))

    def test_same_level_pair_builds_symmetric_component_and_merge_motif(self):
        left = _fruit(1, 1, 160, 760)
        right = _fruit(2, 1, 200, 760)

        analysis = _analyze(_state((right, left), queue=(1, 4, 3, 2)))
        left_analysis = analysis.get_fruit(1)
        right_analysis = analysis.get_fruit(2)

        self.assertEqual(left_analysis.partner_ids, (2,))
        self.assertEqual(right_analysis.partner_ids, (1,))
        self.assertTrue(left_analysis.partner_reachable)
        self.assertTrue(right_analysis.partner_reachable)
        self.assertEqual(len(analysis.partner_components), 1)
        component = analysis.partner_components[0]
        self.assertEqual(component.level, 1)
        self.assertEqual(component.fruit_ids, (1, 2))
        self.assertNotEqual(component.reachable_action_mask, 0)

        pair_motifs = tuple(
            motif
            for motif in analysis.chain_motifs
            if motif.motif_type == 'merge_pair'
        )
        self.assertEqual(len(pair_motifs), 1)
        self.assertEqual(pair_motifs[0].fruit_ids, (1, 2))
        self.assertIn(0, pair_motifs[0].compatible_queue_indices)
        self.assertGreater(pair_motifs[0].readiness, 0.0)
        self.assertAlmostEqual(
            analysis.chain_readiness,
            max(motif.readiness for motif in analysis.chain_motifs),
        )

    def test_nearby_next_level_fruit_extends_pair_into_ladder_motif(self):
        left = _fruit(1, 1, 160, 760)
        right = _fruit(2, 1, 200, 760)
        # 下一级水果放在 pair 侧上方，既属于局部阶梯，又不把 pair 的
        # 所有顶部触发列完全盖住。
        next_level = _fruit(3, 2, 260, 700)

        analysis = _analyze(_state((next_level, right, left)))
        ladders = tuple(
            motif
            for motif in analysis.chain_motifs
            if motif.motif_type == 'level_ladder'
        )

        self.assertEqual(len(ladders), 1)
        self.assertEqual(ladders[0].fruit_ids, (1, 2, 3))
        self.assertEqual(ladders[0].levels, (1, 1, 2))
        self.assertEqual(ladders[0].depth, 2)
        self.assertGreater(ladders[0].readiness, 0.0)

    def test_inaccessible_pair_without_queue_trigger_has_no_positive_motif(self):
        fruits = (
            _fruit(1, 1, 180, 740),
            _fruit(2, 1, 220, 740),
            _fruit(3, 3, 180, 500),
            _fruit(4, 3, 220, 500),
        )

        analysis = _analyze(_state(fruits, queue=(4, 4, 4, 4)))

        self.assertEqual(analysis.get_fruit(1).reachable_action_mask, 0)
        self.assertEqual(analysis.get_fruit(2).reachable_action_mask, 0)
        self.assertEqual(analysis.get_fruit(1).partner_ids, (2,))
        self.assertFalse(analysis.chain_motifs)
        self.assertAlmostEqual(analysis.chain_readiness, 0.0)

    def test_free_space_regions_obey_reference_and_topology_invariants(self):
        fruits = (
            _fruit(1, 1, 90, 760),
            _fruit(2, 2, 200, 750),
            _fruit(3, 3, 300, 738),
        )

        analysis = _analyze(_state(fruits))
        fruit_ids = {fruit.fruit_id for fruit in fruits}
        region_ids = {region.region_id for region in analysis.free_space_regions}

        self.assertTrue(analysis.free_space_regions)
        self.assertTrue(
            any(region.top_connected for region in analysis.free_space_regions)
        )
        self.assertLessEqual(
            sum(region.area_ratio for region in analysis.free_space_regions),
            1.0 + 1e-9,
        )
        for region in analysis.free_space_regions:
            self.assertGreater(region.cell_count, 0)
            self.assertTrue(set(region.boundary_fruit_ids) <= fruit_ids)
            self.assertLessEqual(region.min_x, region.centroid_x)
            self.assertLessEqual(region.centroid_x, region.max_x)
            self.assertLessEqual(region.min_y, region.centroid_y)
            self.assertLessEqual(region.centroid_y, region.max_y)
            if region.sealed:
                self.assertEqual(region.reachable_action_mask, 0)
            else:
                self.assertNotEqual(region.reachable_action_mask, 0)
        for fruit in analysis.fruit_analyses:
            if fruit.connected_region_id is not None:
                self.assertIn(fruit.connected_region_id, region_ids)

    def test_ring_cavity_opens_after_removing_its_top_obstacle(self):
        ring_fruits = tuple(
            _fruit(
                offset + 1,
                1,
                200 + 70 * math.cos(2 * math.pi * offset / 8),
                500 + 70 * math.sin(2 * math.pi * offset / 8),
            )
            for offset in range(8)
        )

        sealed = _analyze(_state(ring_fruits))
        enclosed_regions = tuple(
            region
            for region in sealed.free_space_regions
            if (
                region.sealed
                and set(region.boundary_fruit_ids) == set(range(1, 9))
            )
        )

        self.assertEqual(sealed.sealed_cavity_count, 1)
        self.assertEqual(len(enclosed_regions), 1)
        self.assertEqual(enclosed_regions[0].reachable_action_mask, 0)
        self.assertGreater(enclosed_regions[0].cell_count, 4)
        self.assertGreater(sealed.sealed_cavity_ratio, 0.0)

        opened = _analyze(_state(tuple(
            fruit
            for fruit in ring_fruits
            if fruit.fruit_id != 7
        )))
        self.assertEqual(opened.sealed_cavity_count, 0)
        self.assertAlmostEqual(opened.sealed_cavity_ratio, 0.0)

    def test_single_row_cavity_centroid_stays_inside_bounding_box(self):
        """退化成单行的区域不能因均值舍入在包围盒外崩溃。"""

        fruits = tuple(
            _fruit(
                offset + 1,
                1,
                200 + 70 * math.cos(2 * math.pi * offset / 8),
                300 + 50 * math.sin(2 * math.pi * offset / 8),
            )
            for offset in range(8)
        )

        analysis = _analyze(_state(fruits))
        cavities = tuple(
            region
            for region in analysis.free_space_regions
            if region.sealed
        )

        self.assertEqual(len(cavities), 1)
        cavity = cavities[0]
        self.assertEqual(cavity.cell_count, 5)
        self.assertEqual(cavity.min_y, cavity.max_y)
        self.assertEqual(cavity.centroid_y, cavity.min_y)


class StateAnalyzerValidationAndSymmetryTest(unittest.TestCase):
    """验证边界输入会失败，并检查左右镜像不改变状态价值。"""

    def test_rejects_wrong_action_count_step_key_and_q0_metadata(self):
        state = _state(step_count=3)
        analyzer = StateAnalyzer()
        actions = _actions(1)

        with self.assertRaises(ValueError):
            StateAnalyzerConfig(action_count=ANALYSIS_ACTION_COUNT - 1)
        with self.assertRaises(ValueError):
            analyzer.analyze(
                state,
                actions[:-1],
                TransitionKey(0, 0, 3),
            )
        with self.assertRaises(ValueError):
            analyzer.analyze(
                state,
                actions,
                TransitionKey(0, 0, 2),
            )
        with self.assertRaises(ValueError):
            analyzer.analyze(
                state,
                _actions(2),
                TransitionKey(0, 0, 3),
            )

    def test_unstable_boundary_is_retained_only_as_degraded_diagnostics(self):
        state = _state((_fruit(1, 1, 200, 760, stable=False),))

        analysis = _analyze(state, stable_boundary=False)

        self.assertFalse(analysis.diagnostics.stable_boundary)
        self.assertFalse(analysis.diagnostics.valid_for_attribution)
        self.assertTrue(analysis.diagnostics.degraded)
        self.assertIn(
            'unstable_boundary',
            analysis.diagnostics.warning_codes,
        )

    def test_left_right_mirror_reverses_masks_and_lane_outputs(self):
        fruits = (
            _fruit(1, 1, 120, 760),
            _fruit(2, 3, 250, 710),
            _fruit(3, 2, 300, 750),
        )
        original = _analyze(_state(fruits))
        mirrored = _analyze(
            _state(tuple(_mirror_fruit(fruit) for fruit in fruits))
        )

        self.assertAlmostEqual(
            original.top_connected_capacity,
            mirrored.top_connected_capacity,
            places=12,
        )
        self.assertAlmostEqual(
            original.recoverability,
            mirrored.recoverability,
            places=12,
        )
        for original_lane, mirrored_lane in zip(
                original.queue_lane_analyses,
                mirrored.queue_lane_analyses):
            for original_depth, mirrored_depth in zip(
                    original_lane.landing_depths_by_action,
                    reversed(mirrored_lane.landing_depths_by_action)):
                self.assertAlmostEqual(
                    original_depth,
                    mirrored_depth,
                    places=14,
                )
            self.assertEqual(
                original_lane.blocker_ids_by_action,
                tuple(reversed(mirrored_lane.blocker_ids_by_action)),
            )
            self.assertEqual(
                original_lane.safe_action_mask,
                _mirror_mask(mirrored_lane.safe_action_mask),
            )

        for fruit_id in (1, 2, 3):
            original_fruit = original.get_fruit(fruit_id)
            mirrored_fruit = mirrored.get_fruit(fruit_id)
            self.assertEqual(
                original_fruit.reachable_action_mask,
                _mirror_mask(mirrored_fruit.reachable_action_mask),
            )
            self.assertEqual(
                original_fruit.top_blocker_ids_by_action,
                tuple(
                    reversed(mirrored_fruit.top_blocker_ids_by_action)
                ),
            )
            self.assertAlmostEqual(
                original_fruit.burial_depth,
                mirrored_fruit.burial_depth,
            )


if __name__ == '__main__':
    unittest.main()
