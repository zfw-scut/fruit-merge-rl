"""一步结构 target、TensorTransition 与 n-step 语义专项测试。"""

from __future__ import annotations

import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from attribution_fixtures import (
    FruitSpec,
    make_analysis,
)
from daxigua.core.rules import merge_score
from daxigua.core.state import MergeEvent, PhysicsResult
from daxigua_rl.attribution.schema import FreeSpaceRegionAnalysis
from daxigua_rl.graph.tensor import GraphTensor
from daxigua_rl.training.identity import TransitionKey
from daxigua_rl.training.n_step import NStepTransitionAccumulator
from daxigua_rl.training.replay_buffer import ReplayBuffer
from daxigua_rl.training.structural_targets import (
    STRUCTURAL_TARGET_DIMENSIONS,
    STRUCTURAL_TARGET_FULL_VALID_MASK,
    STRUCTURAL_TARGET_HALF_PAYLOAD_BYTES,
    StructuralTarget,
    build_structural_target,
)
from daxigua_rl.training.tensor_transition import TensorTransition


def _graph(action_count=3):
    return GraphTensor(
        node_features=torch.zeros(
            (action_count, 1),
            dtype=torch.float16,
        ),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_features=torch.empty((0, 1), dtype=torch.float16),
        action_node_indices=torch.arange(
            action_count,
            dtype=torch.long,
        ),
        action_indices=torch.arange(
            action_count,
            dtype=torch.long,
        ),
        node_feature_names=('feature',),
        edge_feature_names=('edge_feature',),
    )


def _transition(
        graph,
        next_graph,
        reward,
        *,
        target=None,
        terminated=False):
    return TensorTransition(
        graph=graph,
        action_offset=0,
        reward=reward,
        next_graph=next_graph,
        terminated=terminated,
        truncated=False,
        structural_target=target,
    )


def _sealed_region(*, region_id=99, area_ratio=0.2):
    return FreeSpaceRegionAnalysis(
        region_id=region_id,
        top_connected=False,
        reachable_action_mask=0,
        cell_count=8,
        area_ratio=area_ratio,
        centroid_x=0.5,
        centroid_y=0.5,
        min_x=0.2,
        max_x=0.8,
        min_y=0.2,
        max_y=0.8,
    )


def _healthy_analysis_pair():
    key = TransitionKey(0, 1, 7)
    next_key = TransitionKey(0, 1, 8)
    previous = make_analysis(
        (
            FruitSpec(
                fruit_id=1,
                level=1,
                reachable_mask=(1 << 15) - 1,
                partner_reachable=True,
                burial_depth=0.3,
            ),
        ),
        key=key,
    )
    following = make_analysis(
        (
            FruitSpec(
                fruit_id=1,
                level=1,
                reachable_mask=0,
                partner_reachable=False,
                burial_depth=0.8,
            ),
        ),
        key=next_key,
        incoming_key=key,
    )
    return previous, replace(
        following,
        # 该专项只需要一个 0.2 的封闭区；不保留 analyzer 为全空棋盘生成的
        # 1.0 开放区，否则会违反 schema 的总面积上限。
        free_space_regions=(_sealed_region(),),
    )


def _chain_events():
    first = MergeEvent(
        new_level=2,
        x=100.0,
        y=500.0,
        score_delta=merge_score(2),
        source_ids=(10, 11),
        new_fruit_id=20,
    )
    second = MergeEvent(
        new_level=3,
        x=102.0,
        y=502.0,
        score_delta=merge_score(3),
        source_ids=(20, 12),
        new_fruit_id=21,
    )
    return first, second


class StructuralTargetContractTest(unittest.TestCase):
    def test_mask_validation_pickle_and_half_payload(self):
        target = StructuralTarget(
            values=(0.1, -0.2, 0.3, 0.4, -0.5, 0.6),
            valid_mask=STRUCTURAL_TARGET_FULL_VALID_MASK,
        )

        self.assertEqual(len(STRUCTURAL_TARGET_DIMENSIONS), 6)
        self.assertEqual(target.validity, (True,) * 6)
        self.assertTrue(target.is_valid('recoverability_delta'))
        self.assertTrue(target.is_valid(5))
        self.assertEqual(pickle.loads(pickle.dumps(target)), target)

        payload = target.to_half_bytes()
        self.assertEqual(
            len(payload),
            STRUCTURAL_TARGET_HALF_PAYLOAD_BYTES,
        )
        self.assertEqual(len(payload), 14)
        restored = StructuralTarget.from_half_bytes(payload)
        self.assertEqual(restored.valid_mask, target.valid_mask)
        for actual, expected in zip(restored.values, target.values):
            self.assertAlmostEqual(actual, expected, places=3)

        with self.assertRaisesRegex(
                ValueError,
                'invalid structural target dimensions'):
            StructuralTarget(
                values=(0.0, 0.1, 0.0, 0.0, 0.0, 0.0),
                valid_mask=1,
            )
        with self.assertRaises(ValueError):
            StructuralTarget(
                values=(0.0,) * 5,
                valid_mask=0,
            )
        with self.assertRaises(ValueError):
            StructuralTarget(
                values=(0.0,) * 6,
                valid_mask=1 << 6,
            )

    def test_builder_uses_only_adjacent_action_evidence(self):
        previous, following = _healthy_analysis_pair()
        events = _chain_events()
        physics_result = PhysicsResult(
            frames_simulated=30,
            stable=True,
            done=False,
            truncated=False,
            score_delta=sum(event.score_delta for event in events),
            merge_events=events,
        )

        target = build_structural_target(
            previous,
            following,
            physics_result=physics_result,
        )

        self.assertEqual(
            target.valid_mask,
            STRUCTURAL_TARGET_FULL_VALID_MASK,
        )
        self.assertAlmostEqual(
            target.values[0],
            (
                following.top_connected_capacity
                - previous.top_connected_capacity
            ),
        )
        self.assertAlmostEqual(
            target.values[1],
            following.recoverability - previous.recoverability,
        )
        self.assertAlmostEqual(
            target.values[2],
            following.chain_readiness - previous.chain_readiness,
        )
        self.assertGreater(target.values[3], 0.0)
        self.assertAlmostEqual(target.values[4], 0.2)
        self.assertGreater(target.values[5], 0.0)

        # 显式 merge 流与 PhysicsResult 走同一纯函数语义。
        explicit = build_structural_target(
            previous,
            following,
            merge_events=events,
            terminated=False,
        )
        self.assertEqual(explicit, target)

        # 仅知道“没有终局”不能反推出“没有连锁”；未提供 merge 证据时最后
        # 一维必须保持无效。
        missing_merge_evidence = build_structural_target(
            previous,
            following,
            terminated=False,
        )
        self.assertEqual(missing_merge_evidence.valid_mask, (1 << 5) - 1)

        # 即使 step 编号相邻，缺少 incoming key 也不能把两份 analysis 当成
        # 已证明属于同一动作；独立的本步物理结果仍然可以保留。
        unlinked = replace(
            following,
            incoming_transition_key=None,
        )
        unlinked_target = build_structural_target(
            previous,
            unlinked,
            physics_result=physics_result,
        )
        self.assertEqual(unlinked_target.valid_mask, 1 << 5)

        future = make_analysis(
            (),
            key=TransitionKey(0, 1, 9),
        )
        with self.assertRaisesRegex(ValueError, 'immediately following'):
            build_structural_target(
                previous,
                future,
                physics_result=physics_result,
            )
        with self.assertRaisesRegex(ValueError, 'must match'):
            build_structural_target(
                previous,
                following,
                physics_result=physics_result,
                terminated=True,
            )
        with self.assertRaisesRegex(ValueError, 'not both'):
            build_structural_target(
                previous,
                following,
                physics_result=physics_result,
                merge_events=events,
            )

    def test_unrelated_merge_does_not_inflate_realized_chain_signal(self):
        chain_events = _chain_events()
        unrelated = MergeEvent(
            new_level=10,
            x=300.0,
            y=600.0,
            score_delta=merge_score(10),
            source_ids=(30, 31),
            new_fruit_id=32,
        )

        chain_only = build_structural_target(
            None,
            None,
            merge_events=chain_events,
            terminated=False,
        )
        with_unrelated = build_structural_target(
            None,
            None,
            merge_events=(*chain_events, unrelated),
            terminated=False,
        )

        self.assertGreater(chain_only.values[5], 0.0)
        self.assertEqual(
            with_unrelated.values[5],
            chain_only.values[5],
        )

    def test_degraded_analysis_masks_geometry_but_keeps_action_outcome(self):
        previous, _following = _healthy_analysis_pair()
        degraded = make_analysis(
            (),
            key=TransitionKey(0, 1, 8),
            incoming_key=previous.transition_key,
            valid=False,
        )
        physics_result = PhysicsResult(
            frames_simulated=30,
            stable=True,
            done=False,
            truncated=False,
            score_delta=0,
            merge_events=(),
        )

        target = build_structural_target(
            previous,
            degraded,
            physics_result=physics_result,
        )

        self.assertEqual(target.valid_mask, 1 << 5)
        self.assertEqual(target.values, (0.0,) * 6)

    def test_terminal_without_next_analysis_keeps_only_terminal_risk(self):
        previous, _following = _healthy_analysis_pair()
        physics_result = PhysicsResult(
            frames_simulated=30,
            stable=True,
            done=True,
            truncated=False,
            score_delta=0,
            merge_events=(),
        )

        target = build_structural_target(
            previous,
            None,
            physics_result=physics_result,
        )

        self.assertEqual(target.valid_mask, 1 << 5)
        self.assertEqual(target.values[:5], (0.0,) * 5)
        self.assertEqual(target.values[5], -1.0)

    def test_already_dead_fruit_is_not_repeated_as_new_risk(self):
        key = TransitionKey(1, 2, 3)
        dead = FruitSpec(
            fruit_id=7,
            level=1,
            reachable_mask=0,
            partner_reachable=False,
            burial_depth=0.9,
        )
        previous = make_analysis((dead,), key=key)
        following = make_analysis(
            (dead,),
            key=TransitionKey(1, 2, 4),
            incoming_key=key,
        )

        target = build_structural_target(
            previous,
            following,
            merge_events=(),
            terminated=False,
        )

        self.assertTrue(target.is_valid(3))
        self.assertEqual(target.values[3], 0.0)


class TensorTransitionStructuralTargetTest(unittest.TestCase):
    def test_optional_target_is_backward_compatible_and_pickle_safe(self):
        graph = _graph()
        legacy = _transition(graph, graph, 1.0)
        self.assertIsNone(legacy.structural_target)

        # 模拟旧版本 pickle 的实例状态：其中不存在后来新增的字段。dataclass
        # 类默认值应使反序列化结果仍可读取为 None。
        object.__delattr__(legacy, 'structural_target')
        legacy_restored = pickle.loads(pickle.dumps(legacy))
        self.assertIsNone(legacy_restored.structural_target)

        target = StructuralTarget(
            values=(0.1, 0.0, 0.0, 0.0, 0.0, -1.0),
            valid_mask=(1 << 0) | (1 << 5),
        )
        transition = _transition(
            graph,
            None,
            2.0,
            target=target,
            terminated=True,
        )
        restored = pickle.loads(pickle.dumps(transition))
        self.assertEqual(restored.structural_target, target)
        self.assertEqual(restored.bootstrap_steps, 1)

        with self.assertRaisesRegex(TypeError, 'StructuralTarget or None'):
            _transition(
                graph,
                graph,
                1.0,
                target='not-a-target',
            )

    def test_n_step_preserves_each_window_start_target_without_adding(self):
        graphs = tuple(_graph() for _ in range(3))
        targets = tuple(
            StructuralTarget(
                values=(value, 0.0, 0.0, 0.0, 0.0, 0.0),
                valid_mask=1,
            )
            for value in (0.1, 0.2, 0.3)
        )
        accumulator = NStepTransitionAccumulator(
            n_step=3,
            gamma=0.5,
        )
        accumulator.append(_transition(
            graphs[0],
            graphs[1],
            1.0,
            target=targets[0],
        ))
        accumulator.append(_transition(
            graphs[1],
            graphs[2],
            2.0,
            target=targets[1],
        ))

        emitted = accumulator.append(_transition(
            graphs[2],
            None,
            4.0,
            target=targets[2],
            terminated=True,
        ))

        self.assertEqual(
            tuple(item.bootstrap_steps for item in emitted),
            (3, 2, 1),
        )
        self.assertEqual(
            tuple(item.structural_target for item in emitted),
            targets,
        )
        self.assertIs(emitted[0].structural_target, targets[0])
        self.assertEqual(
            tuple(item.reward for item in emitted),
            (3.0, 4.0, 4.0),
        )


class ReplaySegmentStructuralTargetTest(unittest.TestCase):
    @staticmethod
    def _load_payload(replay, payload):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'segment.pt'
            torch.save(payload, path)
            return replay._load_segment({
                'path': str(path),
                'count': len(payload.get('records', ())),
            })

    def test_version_three_round_trip_uses_compact_payload(self):
        graph = _graph()
        target = StructuralTarget(
            values=(0.5, -0.25, 0.125, 0.75, -0.5, 1.0),
            valid_mask=STRUCTURAL_TARGET_FULL_VALID_MASK,
        )
        replay = ReplayBuffer(capacity=4)
        payload = replay._pack_segment((
            _transition(graph, graph, 1.0, target=target),
            _transition(
                graph,
                graph,
                2.0,
                target=StructuralTarget.empty(),
            ),
            _transition(graph, graph, 3.0),
        ))

        self.assertEqual(payload['version'], 3)
        self.assertEqual(len(payload['records'][0]), 8)
        packed_target = payload['records'][0][7]
        self.assertIsInstance(packed_target, bytes)
        self.assertEqual(
            len(packed_target),
            STRUCTURAL_TARGET_HALF_PAYLOAD_BYTES,
        )
        self.assertEqual(packed_target, target.to_half_bytes())
        self.assertIsNone(payload['records'][2][7])

        restored = self._load_payload(replay, payload)
        self.assertEqual(restored[0].structural_target, target)
        self.assertEqual(
            restored[1].structural_target,
            StructuralTarget.empty(),
        )
        self.assertIsNone(restored[2].structural_target)
        self.assertTrue(
            restored[0].graph is restored[1].graph
            is restored[2].graph
        )

    def test_hybrid_cold_write_and_sample_preserve_target(self):
        graph = _graph()
        target = StructuralTarget(
            values=(0.5, -0.25, 0.125, 0.75, -0.5, 1.0),
            valid_mask=STRUCTURAL_TARGET_FULL_VALID_MASK,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            replay = ReplayBuffer(
                capacity=3,
                seed=7,
                hot_capacity=1,
                cold_dir=Path(tmp_dir) / 'cold',
                segment_size=1,
                cold_cache_size=1,
                cold_sample_ratio=1.0,
                cold_cache_refresh_interval=1,
            )
            replay.push(
                _transition(graph, graph, 1.0, target=target)
            )
            replay.push(_transition(graph, graph, 2.0))

            self.assertEqual(
                replay.storage_stats['cold_segment_count'],
                1,
            )
            sampled = replay.sample(1)[0]
            self.assertEqual(sampled.structural_target, target)

            segment_path = next(
                (Path(tmp_dir) / 'cold').glob('segment_*.pt')
            )
            disk_payload = torch.load(
                segment_path,
                map_location='cpu',
                weights_only=False,
            )
            self.assertEqual(disk_payload['version'], 3)
            self.assertEqual(
                len(disk_payload['records'][0][7]),
                STRUCTURAL_TARGET_HALF_PAYLOAD_BYTES,
            )

    def test_version_three_rejects_corrupt_structural_payload(self):
        graph = _graph()
        target = StructuralTarget(
            values=(0.5, 0.0, 0.0, 0.0, 0.0, -1.0),
            valid_mask=(1 << 0) | (1 << 5),
        )
        replay = ReplayBuffer(capacity=4)
        original = replay._pack_segment((
            _transition(graph, graph, 1.0, target=target),
        ))
        good = original['records'][0][7]
        corrupt_payloads = {
            'wrong_type': 'not-bytes',
            'truncated': good[:-1],
            'unknown_schema': bytes((99,)) + good[1:],
            'mask_out_of_range': good[:1] + bytes((1 << 6,)) + good[2:],
            'masked_nonzero': good[:1] + bytes((0,)) + good[2:],
        }

        for name, corrupt in corrupt_payloads.items():
            with self.subTest(name=name):
                record = (
                    *original['records'][0][:-1],
                    corrupt,
                )
                payload = {
                    **original,
                    'records': (record,),
                }
                with self.assertRaisesRegex(
                        ValueError,
                        (
                            'version 3 replay record 0 has invalid '
                            'structural target payload'
                        )):
                    self._load_payload(replay, payload)

    def test_version_three_rejects_malformed_record_and_graph_index(self):
        graph = _graph()
        replay = ReplayBuffer(capacity=4)
        original = replay._pack_segment((
            _transition(graph, graph, 1.0),
        ))

        malformed = {
            **original,
            'records': (original['records'][0][:-1],),
        }
        with self.assertRaisesRegex(
                ValueError,
                (
                    'version 3 replay record must contain 8 fields '
                    r'\(record 0\)'
                )):
            self._load_payload(replay, malformed)

        invalid_graph_record = (
            -1,
            *original['records'][0][1:],
        )
        invalid_graph = {
            **original,
            'records': (invalid_graph_record,),
        }
        with self.assertRaisesRegex(
                ValueError,
                'record 0 contains an invalid graph index'):
            self._load_payload(replay, invalid_graph)


if __name__ == '__main__':
    unittest.main()
