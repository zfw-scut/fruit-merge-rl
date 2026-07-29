"""StateAnalysis 到 GNN 图的结构增强专项测试。

这些测试直接构造稳定 ``GameState`` 并复用正式 ``StateAnalyzer``。测试重点不是
重复验证几何算法，而是确保 GraphBuilder：

- 在没有 analysis 时保持旧接口并把所有新增结构列留为 0；
- 严格对齐状态、队列和动作列；
- 把 motif mask 拆成逐 action 特征，而不是编码 mask 整数；
- 只把当前边界可见的水果、q0-q3 和静态关系写入图。
"""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

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
    StateAnalyzer,
    drop_x_positions_for_level,
)
from daxigua_rl.graph import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    FeatureAblationConfig,
    GraphAblator,
    GraphBuilder,
)
from daxigua_rl.training.identity import TransitionKey


GEOMETRY = BoardGeometry(
    width=400,
    height=800,
    spawn_y=180,
    wall_width=20,
    floor_y=780,
)

STRUCTURE_NODE_FEATURES = (
    'reachable_action_fraction',
    'top_visible_ratio',
    'partner_reachable',
    'partner_count',
    'reachable_partner_count',
    'support_parent_count',
    'supported_child_count',
    'burial_depth',
    'inversion_count',
    'critical_blocker_count',
    'connected_to_top_space',
    'q0_landing_depth',
    'q0_is_safe',
    'q0_blocker_count',
    'has_state_analysis',
    'analysis_valid',
    'analysis_degraded',
    'top_connected_capacity',
    'recoverability',
    'chain_readiness',
    'top_connected_free_space_ratio',
    'sealed_cavity_ratio',
    'sealed_cavity_count',
    'motif_is_merge_pair',
    'motif_is_level_ladder',
    'motif_base_level',
    'motif_member_count',
    'motif_depth',
    'motif_readiness',
    'motif_trigger_action_fraction',
    'motif_current_queue_compatible',
    'motif_future_queue_compatible',
    'motif_future_queue_weight',
)

STRUCTURE_EDGE_FEATURES = (
    'is_contact_relation',
    'is_support_relation',
    'is_caps_relation',
    'is_bridges_relation',
    'is_reachable_partner_relation',
    'is_critical_blocker_relation',
    'is_inversion_blocker_relation',
    'structure_confidence',
    'motif_role_pair_member',
    'motif_role_chain_target',
    'motif_stage',
    'motif_trigger_now',
    'motif_future_queue',
    'motif_preserve',
    'motif_break_risk',
    'motif_relation_strength',
    'action_reaches_fruit',
    'is_q0_first_blocker',
)


def _fruit(fruit_id, level, x, y):
    """构造一颗使用直接投放物理半径的静止水果。"""

    physics_radius = float(dropped_fruit_physics_radius(level))
    return FruitState(
        fruit_id=fruit_id,
        level=level,
        radius=float(fruit_radius(level)),
        physics_radius=physics_radius,
        x=float(x),
        y=float(y),
        vx=0.0,
        vy=0.0,
        angle=0.0,
        angular_velocity=0.0,
        age_frames=0,
        stable=True,
        distance_to_left_wall=float(
            x - (GEOMETRY.wall_width + physics_radius)
        ),
        distance_to_right_wall=float(
            GEOMETRY.width
            - GEOMETRY.wall_width
            - physics_radius
            - x
        ),
        distance_to_floor=float(
            GEOMETRY.floor_y - physics_radius - y
        ),
        distance_to_danger_line=float(
            y - physics_radius - GEOMETRY.spawn_y
        ),
    )


def _state(fruits=(), *, queue=(1, 2, 3, 4), step_count=0):
    """构造字段内部一致的稳定游戏状态。"""

    fruits = tuple(fruits)
    highest_top = min(
        (
            fruit.y - fruit.physics_radius
            for fruit in fruits
        ),
        default=GEOMETRY.height,
    )
    playable_area = (
        GEOMETRY.width
        * (GEOMETRY.height - GEOMETRY.spawn_y)
    )
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
        physics_frame=120,
        done=False,
        geometry=GEOMETRY,
        max_height=(
            0.0
            if not fruits
            else float(GEOMETRY.height - highest_top)
        ),
        fruit_count=len(fruits),
        max_level=max(
            (fruit.level for fruit in fruits),
            default=0,
        ),
        empty_space_ratio=max(
            0.0,
            min(1.0, 1.0 - occupied_area / playable_area),
        ),
    )


def _actions(level=1, *, action_count=ANALYSIS_ACTION_COUNT):
    """按当前游戏规则生成从左到右排列的候选动作。"""

    positions = drop_x_positions_for_level(
        GEOMETRY,
        level,
        action_count=action_count,
    )
    left = positions[0]
    right = positions[-1]
    return tuple(
        ActionCandidate(
            action_index=action_offset,
            drop_x=drop_x,
            normalized_drop_x=(
                0.0
                if right == left
                else (drop_x - left) / (right - left)
            ),
            current_level=level,
            current_radius=float(fruit_radius(level)),
            current_physics_radius=float(
                dropped_fruit_physics_radius(level)
            ),
        )
        for action_offset, drop_x in enumerate(positions)
    )


def _analyze(state):
    """为测试状态生成与其 step 对齐的规范 21 列分析。"""

    actions = _actions(state.fruit_queue[0])
    analysis = StateAnalyzer().analyze(
        state,
        actions,
        TransitionKey(
            worker_id=0,
            episode_id=0,
            step_index=state.step_count,
        ),
        stable_boundary=True,
    )
    return actions, analysis


def _node_value(graph, node_index, feature_name):
    """按名称读取一个节点特征，避免测试依赖固定列号。"""

    return graph.node_features[node_index][
        NODE_FEATURE_NAMES.index(feature_name)
    ]


def _edge_value(graph, edge_offset, feature_name):
    """按名称读取一条边特征。"""

    return graph.edge_features[edge_offset][
        EDGE_FEATURE_NAMES.index(feature_name)
    ]


def _fruit_node_by_id(graph, fruit_id):
    return next(
        node_offset
        for node_offset, ref in enumerate(graph.node_refs)
        if (
            ref.node_type == 'board_fruit'
            and ref.source_id == fruit_id
        )
    )


def _edge_offset(graph, source_node, target_node, edge_type):
    return next(
        edge_offset
        for edge_offset, ref in enumerate(graph.edge_refs)
        if (
            ref.source_node == source_node
            and ref.target_node == target_node
            and ref.edge_type == edge_type
        )
    )


class StructureGraphCompatibilityTest(unittest.TestCase):
    """验证旧接口、对齐拒绝和少动作环境兼容。"""

    def test_without_analysis_keeps_new_structure_columns_zero(self):
        state = _state((_fruit(1, 1, 200, 760),))
        actions = _actions()

        graph = GraphBuilder().build(state, actions)

        self.assertFalse(
            any(
                ref.node_type == 'chain_motif'
                for ref in graph.node_refs
            )
        )
        for feature_name in STRUCTURE_NODE_FEATURES:
            feature_index = NODE_FEATURE_NAMES.index(feature_name)
            self.assertTrue(
                all(
                    row[feature_index] == 0.0
                    for row in graph.node_features
                ),
                feature_name,
            )
        for feature_name in STRUCTURE_EDGE_FEATURES:
            feature_index = EDGE_FEATURE_NAMES.index(feature_name)
            self.assertTrue(
                all(
                    row[feature_index] == 0.0
                    for row in graph.edge_features
                ),
                feature_name,
            )

    def test_rejects_analysis_from_another_step_or_queue(self):
        state = _state(step_count=3)
        actions, analysis = _analyze(state)
        builder = GraphBuilder()

        wrong_step_state = _state(step_count=4)
        with self.assertRaisesRegex(ValueError, 'step_index'):
            builder.build(
                wrong_step_state,
                actions,
                state_analysis=analysis,
            )

        wrong_queue_state = _state(
            queue=(4, 3, 2, 1),
            step_count=3,
        )
        with self.assertRaisesRegex(ValueError, 'queue lanes'):
            builder.build(
                wrong_queue_state,
                actions,
                state_analysis=analysis,
            )

        wrong_action_ids = tuple(
            replace(
                action,
                action_index=action.action_index + 100,
            )
            for action in actions
        )
        with self.assertRaisesRegex(ValueError, 'full action layout'):
            builder.build(
                state,
                wrong_action_ids,
                state_analysis=analysis,
            )

    def test_seven_policy_actions_map_to_nearest_analysis_columns(self):
        """测试用 7 动作策略仍应复用规范 21 列，而不是错用 action_index。"""

        state = _state((_fruit(1, 1, 200, 760),))
        _analysis_actions, analysis = _analyze(state)
        policy_actions = _actions(action_count=7)

        graph = GraphBuilder().build(
            state,
            policy_actions,
            state_analysis=analysis,
        )

        analyzed_positions = analysis.action_drop_x_by_offset
        nearest_offsets = tuple(
            min(
                range(len(analyzed_positions)),
                key=lambda offset: (
                    abs(action.drop_x - analyzed_positions[offset]),
                    offset,
                ),
            )
            for action in policy_actions
        )
        q0_lane = analysis.queue_lane_analyses[0]
        for action_offset, node_index in enumerate(
                graph.action_node_indices):
            analysis_offset = nearest_offsets[action_offset]
            self.assertAlmostEqual(
                _node_value(
                    graph,
                    node_index,
                    'q0_landing_depth',
                ),
                q0_lane.landing_depths_by_action[analysis_offset],
            )


class StructureGraphFeatureTest(unittest.TestCase):
    """验证节点摘要和水果显式关系。"""

    def test_fruit_action_and_global_nodes_receive_current_analysis(self):
        state = _state((
            _fruit(3, 2, 260, 700),
            _fruit(2, 1, 200, 760),
            _fruit(1, 1, 160, 760),
        ))
        actions, analysis = _analyze(state)

        graph = GraphBuilder().build(
            state,
            actions,
            state_analysis=analysis,
        )

        fruit_node = _fruit_node_by_id(graph, 1)
        analyzed_fruit = analysis.get_fruit(1)
        self.assertAlmostEqual(
            _node_value(
                graph,
                fruit_node,
                'reachable_action_fraction',
            ),
            analyzed_fruit.reachable_fraction,
        )
        self.assertAlmostEqual(
            _node_value(graph, fruit_node, 'burial_depth'),
            analyzed_fruit.burial_depth,
        )

        global_node = next(
            node_offset
            for node_offset, ref in enumerate(graph.node_refs)
            if ref.node_type == 'global'
        )
        self.assertEqual(
            _node_value(graph, global_node, 'has_state_analysis'),
            1.0,
        )
        self.assertAlmostEqual(
            _node_value(
                graph,
                global_node,
                'top_connected_capacity',
            ),
            analysis.top_connected_capacity,
        )
        self.assertAlmostEqual(
            _node_value(graph, global_node, 'recoverability'),
            analysis.recoverability,
        )
        self.assertAlmostEqual(
            _node_value(graph, global_node, 'chain_readiness'),
            analysis.chain_readiness,
        )

        q0_lane = analysis.queue_lane_analyses[0]
        for action_offset, action_node in enumerate(
                graph.action_node_indices):
            self.assertAlmostEqual(
                _node_value(
                    graph,
                    action_node,
                    'q0_landing_depth',
                ),
                q0_lane.landing_depths_by_action[action_offset],
            )
            self.assertEqual(
                _node_value(graph, action_node, 'q0_is_safe'),
                float(
                    bool(
                        q0_lane.safe_action_mask
                        & (1 << action_offset)
                    )
                ),
            )

    def test_fruit_edges_encode_contact_partner_support_and_blockers(self):
        builder = GraphBuilder()

        # 同级接触 pair 同时覆盖 contact 与 reachable_partner。
        pair_state = _state((
            _fruit(1, 1, 180, 760),
            _fruit(2, 1, 220, 760),
        ))
        actions, pair_analysis = _analyze(pair_state)
        pair_graph = builder.build(
            pair_state,
            actions,
            state_analysis=pair_analysis,
        )
        left_node = _fruit_node_by_id(pair_graph, 1)
        right_node = _fruit_node_by_id(pair_graph, 2)
        pair_edge = _edge_offset(
            pair_graph,
            left_node,
            right_node,
            'board_fruit_to_board_fruit',
        )
        self.assertEqual(
            _edge_value(
                pair_graph,
                pair_edge,
                'is_contact_relation',
            ),
            1.0,
        )
        self.assertEqual(
            _edge_value(
                pair_graph,
                pair_edge,
                'is_reachable_partner_relation',
            ),
            1.0,
        )

        # 上方高级水果封死低级水果，canonical 方向必须是 blocker -> victim。
        blocked_state = _state((
            _fruit(1, 1, 200, 740),
            _fruit(2, 3, 200, 500),
        ))
        actions, blocked_analysis = _analyze(blocked_state)
        blocked_graph = builder.build(
            blocked_state,
            actions,
            state_analysis=blocked_analysis,
        )
        blocker_node = _fruit_node_by_id(blocked_graph, 2)
        victim_node = _fruit_node_by_id(blocked_graph, 1)
        blocker_edge = _edge_offset(
            blocked_graph,
            blocker_node,
            victim_node,
            'board_fruit_to_board_fruit',
        )
        self.assertEqual(
            _edge_value(
                blocked_graph,
                blocker_edge,
                'is_caps_relation',
            ),
            1.0,
        )
        self.assertEqual(
            _edge_value(
                blocked_graph,
                blocker_edge,
                'is_critical_blocker_relation',
            ),
            1.0,
        )
        self.assertEqual(
            _edge_value(
                blocked_graph,
                blocker_edge,
                'is_inversion_blocker_relation',
            ),
            1.0,
        )
        reverse_edge = _edge_offset(
            blocked_graph,
            victim_node,
            blocker_node,
            'board_fruit_to_board_fruit',
        )
        self.assertEqual(
            _edge_value(
                blocked_graph,
                reverse_edge,
                'is_critical_blocker_relation',
            ),
            0.0,
        )

        # 一个上方水果同时由左右两颗水果支撑时，supports 和 bridges 均保留。
        bridge_state = _state(
            (
                _fruit(1, 2, 170, 740),
                _fruit(2, 2, 230, 740),
                _fruit(3, 1, 200, 700),
            ),
            queue=(4, 4, 4, 4),
        )
        actions, bridge_analysis = _analyze(bridge_state)
        bridge_graph = builder.build(
            bridge_state,
            actions,
            state_analysis=bridge_analysis,
        )
        upper_node = _fruit_node_by_id(bridge_graph, 3)
        for parent_id in (1, 2):
            parent_node = _fruit_node_by_id(
                bridge_graph,
                parent_id,
            )
            support_edge = _edge_offset(
                bridge_graph,
                parent_node,
                upper_node,
                'board_fruit_to_board_fruit',
            )
            self.assertEqual(
                _edge_value(
                    bridge_graph,
                    support_edge,
                    'is_support_relation',
                ),
                1.0,
            )
            self.assertEqual(
                _edge_value(
                    bridge_graph,
                    support_edge,
                    'is_bridges_relation',
                ),
                1.0,
            )


class ChainMotifGraphTest(unittest.TestCase):
    """验证 motif 虚拟节点及其水果、队列、动作边。"""

    def setUp(self):
        self.state = _state(
            (
                _fruit(3, 2, 260, 700),
                _fruit(2, 1, 200, 760),
                _fruit(1, 1, 160, 760),
            ),
            # q0 和 q2 均可触发一级 pair；q2 同时提供“先保护再触发”的信号。
            queue=(1, 4, 1, 3),
        )
        self.actions, self.analysis = _analyze(self.state)
        self.graph = GraphBuilder().build(
            self.state,
            self.actions,
            state_analysis=self.analysis,
        )

    def test_each_analysis_motif_becomes_one_virtual_node(self):
        motif_nodes = tuple(
            (node_offset, ref)
            for node_offset, ref in enumerate(self.graph.node_refs)
            if ref.node_type == 'chain_motif'
        )

        self.assertEqual(
            len(motif_nodes),
            len(self.analysis.chain_motifs),
        )
        for node_offset, ref in motif_nodes:
            motif = self.analysis.chain_motifs[ref.source_index]
            self.assertAlmostEqual(
                _node_value(
                    self.graph,
                    node_offset,
                    'motif_readiness',
                ),
                motif.readiness,
            )
            self.assertAlmostEqual(
                _node_value(
                    self.graph,
                    node_offset,
                    'motif_trigger_action_fraction',
                ),
                (
                    motif.trigger_action_mask.bit_count()
                    / ANALYSIS_ACTION_COUNT
                ),
            )

    def test_member_and_queue_edges_encode_role_and_visible_horizon(self):
        ladder_offset = next(
            motif_offset
            for motif_offset, motif in enumerate(
                self.analysis.chain_motifs)
            if motif.motif_type == 'level_ladder'
        )
        motif = self.analysis.chain_motifs[ladder_offset]
        motif_node = next(
            node_offset
            for node_offset, ref in enumerate(self.graph.node_refs)
            if (
                ref.node_type == 'chain_motif'
                and ref.source_index == ladder_offset
            )
        )

        for fruit_id, level in zip(motif.fruit_ids, motif.levels):
            fruit_node = _fruit_node_by_id(self.graph, fruit_id)
            edge_offset = _edge_offset(
                self.graph,
                fruit_node,
                motif_node,
                'board_fruit_to_chain_motif',
            )
            self.assertEqual(
                _edge_value(
                    self.graph,
                    edge_offset,
                    'motif_role_pair_member',
                ),
                float(level == motif.levels[0]),
            )
            self.assertEqual(
                _edge_value(
                    self.graph,
                    edge_offset,
                    'motif_role_chain_target',
                ),
                float(level != motif.levels[0]),
            )

        queue_nodes = {
            ref.source_index: node_offset
            for node_offset, ref in enumerate(self.graph.node_refs)
            if ref.node_type == 'queue_fruit'
        }
        for queue_index in motif.compatible_queue_indices:
            edge_offset = _edge_offset(
                self.graph,
                queue_nodes[queue_index],
                motif_node,
                'queue_fruit_to_chain_motif',
            )
            self.assertEqual(
                _edge_value(
                    self.graph,
                    edge_offset,
                    'motif_trigger_now',
                ),
                float(queue_index == 0),
            )
            self.assertEqual(
                _edge_value(
                    self.graph,
                    edge_offset,
                    'motif_future_queue',
                ),
                float(queue_index > 0),
            )

    def test_trigger_mask_is_decoded_per_action_edge(self):
        for motif_offset, motif in enumerate(
                self.analysis.chain_motifs):
            motif_node = next(
                node_offset
                for node_offset, ref in enumerate(self.graph.node_refs)
                if (
                    ref.node_type == 'chain_motif'
                    and ref.source_index == motif_offset
                )
            )
            for action_offset, action_node in enumerate(
                    self.graph.action_node_indices):
                edge_offset = _edge_offset(
                    self.graph,
                    action_node,
                    motif_node,
                    'action_to_chain_motif',
                )
                expected = float(
                    0 in motif.compatible_queue_indices
                    and bool(
                        motif.trigger_action_mask
                        & (1 << action_offset)
                    )
                )
                self.assertEqual(
                    _edge_value(
                        self.graph,
                        edge_offset,
                        'motif_trigger_now',
                    ),
                    expected,
                )
                self.assertGreaterEqual(
                    _edge_value(
                        self.graph,
                        edge_offset,
                        'motif_break_risk',
                    ),
                    0.0,
                )
                self.assertLessEqual(
                    _edge_value(
                        self.graph,
                        edge_offset,
                        'motif_break_risk',
                    ),
                    1.0,
                )

    def test_structure_ablation_removes_motif_topology_and_masks_content(self):
        masked = GraphAblator.from_preset(
            'no_structure_analysis'
        ).apply(self.graph)

        motif_count = sum(
            ref.node_type == 'chain_motif'
            for ref in self.graph.node_refs
        )
        self.assertGreater(motif_count, 0)
        self.assertEqual(
            masked.num_nodes,
            self.graph.num_nodes - motif_count,
        )
        self.assertFalse(any(
            ref.node_type == 'chain_motif'
            for ref in masked.node_refs
        ))
        self.assertFalse(any(
            'chain_motif' in ref.edge_type
            for ref in masked.edge_refs
        ))
        self.assertTrue(all(
            0 <= source < masked.num_nodes
            and 0 <= target < masked.num_nodes
            for source, target in masked.edge_index
        ))
        self.assertEqual(
            tuple(
                masked.node_refs[index].node_type
                for index in masked.action_node_indices
            ),
            ('action',) * len(masked.action_node_indices),
        )
        for feature_name in (
                *STRUCTURE_NODE_FEATURES,
                'is_chain_motif_node'):
            feature_index = NODE_FEATURE_NAMES.index(feature_name)
            self.assertTrue(
                all(
                    row[feature_index] == 0.0
                    for row in masked.node_features
                ),
                feature_name,
            )
        for feature_name in (
                *STRUCTURE_EDGE_FEATURES,
                'is_board_chain_motif_edge',
                'is_queue_chain_motif_edge',
                'is_action_chain_motif_edge'):
            feature_index = EDGE_FEATURE_NAMES.index(feature_name)
            self.assertTrue(
                all(
                    row[feature_index] == 0.0
                    for row in masked.edge_features
                ),
                feature_name,
            )

    def test_feature_only_motif_mask_clears_generic_level_and_queue_index(self):
        masked = GraphAblator(FeatureAblationConfig(
            disabled_node_feature_groups=('chain_motif',),
            disabled_edge_feature_groups=('chain_motif',),
        )).apply(self.graph)

        motif_nodes = tuple(
            index
            for index, ref in enumerate(masked.node_refs)
            if ref.node_type == 'chain_motif'
        )
        self.assertTrue(motif_nodes)
        self.assertTrue(all(
            _node_value(masked, index, 'level') == 0.0
            for index in motif_nodes
        ))

        queue_motif_edges = tuple(
            index
            for index, ref in enumerate(masked.edge_refs)
            if ref.edge_type == 'queue_fruit_to_chain_motif'
        )
        self.assertTrue(queue_motif_edges)
        self.assertTrue(all(
            _edge_value(masked, index, 'queue_index') == 0.0
            for index in queue_motif_edges
        ))


if __name__ == '__main__':
    unittest.main()
