"""StateAnalysis 只读契约、引用约束和跨进程序列化测试。"""

from __future__ import annotations

import multiprocessing
import pickle
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import FrozenInstanceError, replace

from daxigua.core.rules import (
    dropped_fruit_physics_radius,
    merged_fruit_physics_radius,
)
from daxigua_rl.attribution import (
    ANALYSIS_ACTION_COUNT,
    FULL_ACTION_MASK,
    ChainMotif,
    ContactInfluenceEdge,
    FruitAnalysis,
    PartnerComponent,
    QueueLaneAnalysis,
    StateAnalysis,
    StateAnalysisDiagnostics,
    SupportEdge,
)
from daxigua_rl.training.identity import TransitionKey


ACTION_X = tuple(30.0 + 24.0 * offset for offset in range(ANALYSIS_ACTION_COUNT))
EMPTY_BLOCKERS = tuple(() for _ in range(ANALYSIS_ACTION_COUNT))


def _spawn_roundtrip(value):
    """由 spawn 子进程执行，验证类型定义位于可重新导入的模块顶层。"""

    return pickle.loads(pickle.dumps(value))


def _fruit(
        fruit_id,
        level,
        *,
        partner_ids=(),
        reachable_partner_ids=(),
        support_parent_ids=(),
        supported_child_ids=(),
        inversion_blocker_ids=(),
        top_blockers=EMPTY_BLOCKERS,
        critical_blocker_ids=(),
        physics_radius=None):
    """构造引用一致的测试水果分析。"""

    return FruitAnalysis(
        fruit_id=fruit_id,
        level=level,
        physics_radius=(
            dropped_fruit_physics_radius(level)
            if physics_radius is None
            else physics_radius
        ),
        probe_physics_radius=dropped_fruit_physics_radius(level),
        reachable_action_mask=FULL_ACTION_MASK,
        reachable_action_count=ANALYSIS_ACTION_COUNT,
        top_visible_ratio=0.75,
        top_blocker_ids_by_action=top_blockers,
        partner_ids=partner_ids,
        partner_reachable=bool(reachable_partner_ids),
        support_parent_ids=support_parent_ids,
        supported_child_ids=supported_child_ids,
        burial_depth=0.25,
        inversion_count=len(inversion_blocker_ids),
        connected_region_id=0,
        reachable_partner_ids=reachable_partner_ids,
        critical_blocker_ids=critical_blocker_ids,
        inversion_blocker_ids=inversion_blocker_ids,
    )


def _lane(queue_index, *, blockers=EMPTY_BLOCKERS):
    """构造一个带有槽位专属横坐标的队列容量分析。"""

    drop_x = tuple(value + queue_index for value in ACTION_X)
    return QueueLaneAnalysis(
        queue_index=queue_index,
        level=queue_index + 1,
        physics_radius=dropped_fruit_physics_radius(queue_index + 1),
        drop_x_by_action=drop_x,
        landing_depths_by_action=(0.8,) * ANALYSIS_ACTION_COUNT,
        safe_action_mask=FULL_ACTION_MASK,
        safe_action_count=ANALYSIS_ACTION_COUNT,
        blocker_ids_by_action=blockers,
        capacity=0.86,
    )


def _valid_state(*, with_contact=True):
    """构造一份覆盖所有 schema 类型的最小有效状态。"""

    blockers_for_one = ((3,),) + EMPTY_BLOCKERS[1:]
    fruit_one = _fruit(
        1,
        1,
        partner_ids=(2,),
        reachable_partner_ids=(2,),
        supported_child_ids=(2,),
        inversion_blocker_ids=(3,),
        top_blockers=blockers_for_one,
        critical_blocker_ids=(3,),
    )
    fruit_two = _fruit(
        2,
        1,
        partner_ids=(1,),
        reachable_partner_ids=(1,),
        support_parent_ids=(1,),
    )
    fruit_three = _fruit(
        3,
        3,
        physics_radius=merged_fruit_physics_radius(3),
    )

    lane_blockers = ((3,),) + EMPTY_BLOCKERS[1:]
    lanes = (
        _lane(2),
        _lane(0, blockers=lane_blockers),
        _lane(3),
        _lane(1),
    )
    incoming_key = TransitionKey(2, 5, 0) if with_contact else None
    transition_key = TransitionKey(2, 5, 1 if with_contact else 0)
    contact_edges = (
        ContactInfluenceEdge(
            source_fruit_id=99,
            target_fruit_id=1,
            contact_count=2,
            displacement_x=1.5,
            displacement_y=-2.0,
            max_impulse=4.0,
            first_contact_frame=10,
            last_contact_frame=12,
            on_merge_path=True,
        ),
    ) if with_contact else ()

    return StateAnalysis(
        transition_key=transition_key,
        incoming_transition_key=incoming_key,
        action_indices=tuple(range(100, 100 + ANALYSIS_ACTION_COUNT)),
        action_drop_x_by_offset=ACTION_X,
        fruit_analyses=(fruit_three, fruit_two, fruit_one),
        support_edges=(
            SupportEdge(
                supporter_fruit_id=1,
                supported_fruit_id=2,
                relation='supports',
            ),
            SupportEdge(
                boundary='floor',
                supported_fruit_id=1,
                relation='supports',
            ),
            SupportEdge(
                boundary='floor',
                supported_fruit_id=3,
                relation='supports',
            ),
            SupportEdge(
                boundary='left_wall',
                supported_fruit_id=3,
                relation='wall_constraint',
            ),
        ),
        contact_influence_edges=contact_edges,
        partner_components=(
            PartnerComponent(
                component_id=0,
                level=1,
                fruit_ids=(2, 1),
                reachable_action_mask=FULL_ACTION_MASK,
                top_connected=True,
                connected_region_id=0,
            ),
        ),
        chain_motifs=(
            ChainMotif(
                motif_type='merge_pair',
                fruit_ids=(1, 2),
                levels=(1, 1),
                depth=1,
                trigger_action_mask=0b101,
                compatible_queue_indices=(2, 0),
                readiness=0.7,
            ),
        ),
        queue_lane_analyses=lanes,
        top_connected_capacity=0.86,
        recoverability=0.6,
        chain_readiness=0.7,
        diagnostics=StateAnalysisDiagnostics(
            stable_boundary=True,
            valid_for_attribution=True,
            physics_frame=120,
            analysis_seconds=0.003,
            warning_codes=('approximate_path',),
            approximation_flags=('grid_v1',),
        ),
        analyzer_config_fingerprint='sha256:test-state-analyzer-v1',
    )


class AttributionSchemaConstructionTest(unittest.TestCase):
    """验证字段语义、深只读和确定性规范化。"""

    def test_valid_state_normalizes_all_unordered_collections(self):
        state = _valid_state()

        self.assertEqual(
            tuple(fruit.fruit_id for fruit in state.fruit_analyses),
            (1, 2, 3),
        )
        self.assertEqual(
            tuple(lane.queue_index for lane in state.queue_lane_analyses),
            (0, 1, 2, 3),
        )
        self.assertEqual(
            state.partner_components[0].fruit_ids,
            (1, 2),
        )
        self.assertEqual(
            state.chain_motifs[0].compatible_queue_indices,
            (0, 2),
        )
        self.assertEqual(state.action_count, ANALYSIS_ACTION_COUNT)
        self.assertEqual(state.episode_key, (2, 5))
        self.assertEqual(state.get_fruit(2).level, 1)
        self.assertIsNone(state.get_fruit(50))

    def test_nested_lists_become_tuples_and_objects_are_frozen(self):
        fruit = FruitAnalysis(
            fruit_id=1,
            level=2,
            physics_radius=30,
            probe_physics_radius=dropped_fruit_physics_radius(2),
            reachable_action_mask=0,
            reachable_action_count=0,
            top_visible_ratio=0,
            top_blocker_ids_by_action=[[] for _ in range(ANALYSIS_ACTION_COUNT)],
            partner_ids=[],
            partner_reachable=True,
            support_parent_ids=[],
            supported_child_ids=[],
            burial_depth=1,
            inversion_count=0,
            connected_region_id=None,
        )

        self.assertIsInstance(fruit.top_blocker_ids_by_action, tuple)
        self.assertTrue(
            all(
                isinstance(blockers, tuple)
                for blockers in fruit.top_blocker_ids_by_action
            )
        )
        self.assertEqual(fruit.partner_ids, ())
        self.assertTrue(fruit.partner_reachable)
        with self.assertRaises(FrozenInstanceError):
            fruit.level = 3

    def test_probe_radius_and_each_queue_slot_drop_layout_are_independent(self):
        state = _valid_state()
        fruit = state.get_fruit(3)

        self.assertNotEqual(fruit.physics_radius, fruit.probe_physics_radius)
        self.assertEqual(
            state.action_drop_x_by_offset,
            state.queue_lane_analyses[0].drop_x_by_action,
        )
        self.assertNotEqual(
            state.queue_lane_analyses[0].drop_x_by_action,
            state.queue_lane_analyses[3].drop_x_by_action,
        )

    def test_component_and_motif_signatures_do_not_use_random_hashes(self):
        state = _valid_state()
        reversed_pair = replace(
            state.chain_motifs[0],
            fruit_ids=(2, 1),
            levels=(1, 1),
        )

        self.assertEqual(
            state.partner_components[0].signature,
            (1, (1, 2)),
        )
        self.assertEqual(
            state.chain_motifs[0].signature,
            ('merge_pair', (1, 2), (1, 1)),
        )
        self.assertEqual(reversed_pair.signature, state.chain_motifs[0].signature)


class AttributionSchemaInvariantTest(unittest.TestCase):
    """验证廉价不变量会尽早拒绝损坏的分析结果。"""

    def test_mask_count_overflow_and_action_array_length_are_rejected(self):
        fruit = _valid_state().fruit_analyses[0]

        with self.assertRaises(ValueError):
            replace(fruit, reachable_action_count=0)
        with self.assertRaises(ValueError):
            replace(
                fruit,
                reachable_action_mask=1 << ANALYSIS_ACTION_COUNT,
                reachable_action_count=1,
            )
        with self.assertRaises(ValueError):
            replace(
                fruit,
                top_blocker_ids_by_action=EMPTY_BLOCKERS[:-1],
            )

        lane = _valid_state().queue_lane_analyses[0]
        with self.assertRaises(ValueError):
            replace(lane, safe_action_count=0)
        with self.assertRaises(ValueError):
            replace(lane, drop_x_by_action=ACTION_X[:-1])
        with self.assertRaises(ValueError):
            replace(
                lane,
                drop_x_by_action=(ACTION_X[0], ACTION_X[0]) + ACTION_X[2:],
            )

    def test_lane_and_top_capacity_must_match_reward_v2_formula(self):
        state = _valid_state()
        lane = state.queue_lane_analyses[0]

        self.assertAlmostEqual(lane.computed_capacity, 0.86)
        self.assertAlmostEqual(state.computed_top_connected_capacity, 0.86)
        with self.assertRaises(ValueError):
            replace(lane, capacity=0.85)
        with self.assertRaises(ValueError):
            replace(state, top_connected_capacity=0.85)
        with self.assertRaises(ValueError):
            replace(state, queue_decay=1.1)

    def test_fruit_and_queue_levels_follow_game_rule_bounds(self):
        fruit = _valid_state().fruit_analyses[0]
        lane = _valid_state().queue_lane_analyses[0]

        with self.assertRaises(ValueError):
            replace(fruit, level=12)
        with self.assertRaises(ValueError):
            replace(lane, level=5)
        with self.assertRaises(ValueError):
            replace(
                fruit,
                probe_physics_radius=fruit.probe_physics_radius + 1,
            )
        with self.assertRaises(ValueError):
            replace(lane, physics_radius=lane.physics_radius + 1)

    def test_non_finite_ranges_duplicate_ids_and_self_references_are_rejected(self):
        fruit = _valid_state().fruit_analyses[0]

        with self.assertRaises(ValueError):
            replace(fruit, top_visible_ratio=float('nan'))
        with self.assertRaises(ValueError):
            replace(fruit, burial_depth=1.01)
        with self.assertRaises(ValueError):
            replace(fruit, partner_ids=(2, 2))
        with self.assertRaises(ValueError):
            replace(fruit, critical_blocker_ids=(1,))
        with self.assertRaises(ValueError):
            replace(fruit, critical_blocker_ids=(2,))
        with self.assertRaises(ValueError):
            SupportEdge(
                supporter_fruit_id=1,
                supported_fruit_id=1,
                relation='supports',
            )

    def test_support_cache_must_match_canonical_support_edges(self):
        state = _valid_state()

        with self.assertRaises(ValueError):
            replace(
                state,
                support_edges=tuple(
                    edge
                    for edge in state.support_edges
                    if edge.supporter_fruit_id != 1
                ),
            )

    def test_static_references_must_exist_but_consumed_contact_ids_may_not(self):
        state = _valid_state()
        self.assertEqual(
            state.contact_influence_edges[0].source_fruit_id,
            99,
        )

        broken_fruit = replace(
            state.fruit_analyses[0],
            partner_ids=(2, 88),
            reachable_partner_ids=(2,),
        )
        with self.assertRaises(ValueError):
            replace(
                state,
                fruit_analyses=(broken_fruit,) + state.fruit_analyses[1:],
            )

    def test_inversion_blockers_must_be_higher_level_and_counted_exactly(self):
        fruit = _valid_state().fruit_analyses[0]

        with self.assertRaises(ValueError):
            replace(fruit, inversion_count=0)
        invalid_blocker = replace(
            fruit,
            inversion_blocker_ids=(2,),
            inversion_count=1,
        )
        state = _valid_state()
        with self.assertRaises(ValueError):
            replace(
                state,
                fruit_analyses=(invalid_blocker,) + state.fruit_analyses[1:],
            )

    def test_state_boundary_and_incoming_transition_have_fixed_time_semantics(self):
        state = _valid_state()

        with self.assertRaises(ValueError):
            replace(
                state,
                incoming_transition_key=TransitionKey(2, 5, 1),
            )
        with self.assertRaises(ValueError):
            replace(state, incoming_transition_key=None)
        with self.assertRaises(ValueError):
            replace(
                state,
                incoming_transition_key=TransitionKey(3, 5, 0),
            )

    def test_unstable_boundary_cannot_be_used_for_attribution(self):
        with self.assertRaises(ValueError):
            StateAnalysisDiagnostics(
                stable_boundary=False,
                valid_for_attribution=True,
                physics_frame=1,
            )

        diagnostic = StateAnalysisDiagnostics(
            stable_boundary=False,
            valid_for_attribution=False,
            physics_frame=1,
            degraded=True,
            warning_codes=('physics_truncated',),
        )
        self.assertFalse(diagnostic.valid_for_attribution)

    def test_contact_frame_window_requires_both_endpoints(self):
        edge = _valid_state().contact_influence_edges[0]

        with self.assertRaises(ValueError):
            replace(edge, last_contact_frame=None)
        with self.assertRaises(ValueError):
            replace(edge, first_contact_frame=None)

    def test_q0_layout_and_all_four_queue_slots_are_required(self):
        state = _valid_state()

        with self.assertRaises(ValueError):
            replace(
                state,
                action_drop_x_by_offset=tuple(
                    value + 0.5
                    for value in state.action_drop_x_by_offset
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                state,
                queue_lane_analyses=state.queue_lane_analyses[:-1],
            )


class AttributionSchemaSerializationTest(unittest.TestCase):
    """验证 schema 可安全进入 pickle 和 Windows spawn 边界。"""

    def test_pickle_roundtrip_preserves_full_state(self):
        state = _valid_state()

        restored = pickle.loads(pickle.dumps(state))

        self.assertEqual(restored, state)
        self.assertIsInstance(restored.fruit_analyses, tuple)
        self.assertIsInstance(
            restored.fruit_analyses[0].top_blocker_ids_by_action,
            tuple,
        )

    def test_spawn_process_roundtrip_preserves_full_state(self):
        state = _valid_state()
        context = multiprocessing.get_context('spawn')

        with ProcessPoolExecutor(
                max_workers=1,
                mp_context=context) as executor:
            restored = executor.submit(_spawn_roundtrip, state).result(timeout=30)

        self.assertEqual(restored, state)


if __name__ == '__main__':
    unittest.main()
