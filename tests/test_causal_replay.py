"""因果样本契约、规则构建器和分层内存回放专项测试。"""

from __future__ import annotations

import math
import pickle
import time
import unittest
from dataclasses import replace

import torch

from attribution_fixtures import FruitSpec, make_analysis, make_transition
from daxigua.core.rules import merge_score
from daxigua.core.state import MergeEvent
from daxigua_rl.attribution.causal_replay import (
    CAUSAL_SAMPLE_SCHEMA_VERSION,
    CausalReplayBuffer,
    CausalSample,
    CausalTransitionContext,
    RuleCausalContextCache,
    RuleCausalSampleBuilder,
    graph_schema_fingerprint,
    stable_budget_key,
    stable_event_key,
)
from daxigua_rl.attribution.tracker import AttributionTracker
from daxigua_rl.attribution.schema import (
    ANALYSIS_ACTION_COUNT,
    AttributionEvent,
    AttributionEventKey,
    AttributionEvidence,
    Contributor,
    MergeValueKey,
)
from daxigua_rl.graph.tensor import GraphTensor
from daxigua_rl.reward import merge_utility
from daxigua_rl.training.identity import TransitionKey


def _graph(*, action_indices=None):
    action_indices = (
        tuple(range(ANALYSIS_ACTION_COUNT))
        if action_indices is None
        else tuple(action_indices)
    )
    return GraphTensor(
        node_features=torch.arange(
            ANALYSIS_ACTION_COUNT * 2,
            dtype=torch.float32,
        ).reshape(ANALYSIS_ACTION_COUNT, 2),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_features=torch.empty((0, 1), dtype=torch.float32),
        action_node_indices=torch.arange(
            ANALYSIS_ACTION_COUNT,
            dtype=torch.long,
        ),
        action_indices=torch.tensor(
            action_indices,
            dtype=torch.long,
        ),
        node_feature_names=('x', 'y'),
        edge_feature_names=('distance',),
    )


def _context(
        key,
        *,
        action_offset=0,
        analysis=None,
        policy_version='policy-7'):
    analysis = analysis or make_analysis((), key=key)
    return CausalTransitionContext(
        graph=_graph(action_indices=analysis.action_indices),
        state_analysis=analysis,
        actual_action_offset=action_offset,
        actual_action_index=analysis.action_indices[action_offset],
        policy_version=policy_version,
    )


def _safer_analysis(key, *, safest_offset=14, actual_offset=0):
    """构造只有一个明显安全动作的合法 StateAnalysis。"""

    analysis = make_analysis((), key=key)
    lanes = []
    for lane in analysis.queue_lane_analyses:
        depths = [0.05] * ANALYSIS_ACTION_COUNT
        depths[actual_offset] = 0.0
        depths[safest_offset] = 1.0
        safe_mask = 1 << safest_offset
        capacity = (
            0.7 * (sum(depths) / ANALYSIS_ACTION_COUNT)
            + 0.3 * (safe_mask.bit_count() / ANALYSIS_ACTION_COUNT)
        )
        lanes.append(replace(
            lane,
            landing_depths_by_action=tuple(depths),
            safe_action_mask=safe_mask,
            safe_action_count=1,
            capacity=capacity,
        ))
    weights = tuple(
        analysis.queue_decay ** lane.queue_index
        for lane in lanes
    )
    capacity = (
        sum(weight * lane.capacity for weight, lane in zip(weights, lanes))
        / sum(weights)
    )
    return replace(
        analysis,
        queue_lane_analyses=tuple(lanes),
        top_connected_capacity=capacity,
    )


def _contributor(
        key,
        *,
        action_offset=0,
        weight=1.0,
        fruit_id=1):
    return Contributor(
        transition_key=key,
        action_offset=action_offset,
        action_index=action_offset,
        fruit_id=fruit_id,
        evidence_type='test',
        raw_evidence_weight=weight,
        contribution_weight=weight,
        role='material',
    )


def _event(
        *,
        key,
        event_index=0,
        event_type='DIRECT_TRIGGER',
        sign=1,
        status='confirmed',
        utility=4.0,
        link_confidence=0.95,
        placement_confidence=0.90,
        contributors=None,
        budget_key=None,
        detected_step=0,
        resolved_step=2,
        attribution_version='causal_attribution_v1',
        tracker_fingerprint='tracker-abc'):
    contributors = (
        (_contributor(key),)
        if contributors is None
        else tuple(contributors)
    )
    event_key = AttributionEventKey(
        key.worker_id,
        key.episode_id,
        event_index,
    )
    if budget_key is None:
        budget_key = event_key
    pending = status == 'pending'
    return AttributionEvent(
        event_id=event_key,
        episode_key=(key.worker_id, key.episode_id),
        attribution_version=attribution_version,
        tracker_config_fingerprint=tracker_fingerprint,
        detected_step=detected_step,
        resolved_step=None if pending else resolved_step,
        event_type=event_type,
        status=status,
        sign=sign,
        target_fruit_ids=(99,),
        contributors=contributors,
        utility=utility,
        link_confidence=link_confidence,
        placement_confidence=placement_confidence,
        evidence=AttributionEvidence(reason_codes=('test',)),
        budget_key=budget_key,
        resolution_reason=None if pending else 'test_confirmed',
        delay=None if pending else resolved_step - detected_step,
    )


def _sample(
        *,
        step=0,
        stratum='positive_setup',
        supervision_kind='rule',
        cause_type='DIRECT_TRIGGER',
        actual=0,
        comparison=14,
        direction=1,
        event_suffix='0',
        target_delta=None):
    key = TransitionKey(0, 0, step)
    return CausalSample(
        graph=_graph(),
        actual_action_offset=actual,
        comparison_action_offset=comparison,
        direction=direction,
        target_margin=1.0,
        confidence=0.9,
        cause_type=cause_type,
        delay=2,
        transition_key=key,
        attribution_version='causal_attribution_v1',
        supervision_kind=supervision_kind,
        stratum=stratum,
        event_key=f'event:{event_suffix}',
        budget_key=f'budget:{event_suffix}',
        target_delta=target_delta,
        policy_version='policy-7',
        tracker_config_fingerprint='tracker-abc',
        analyzer_config_fingerprint='analyzer-abc',
        graph_schema_fingerprint=graph_schema_fingerprint(_graph()),
    )


def _fast_sample_clone(
        template,
        *,
        step,
        cause_type,
        event_suffix):
    """压力测试专用的已验证样本克隆，避免重复校验共享 GraphTensor。"""

    replacements = {
        'transition_key': TransitionKey(0, 0, step),
        'cause_type': cause_type,
        'event_key': f'event:{event_suffix}',
        'budget_key': f'budget:{event_suffix}',
    }
    sample = object.__new__(CausalSample)
    for field_name in CausalSample.__slots__:
        object.__setattr__(
            sample,
            field_name,
            replacements.get(field_name, getattr(template, field_name)),
        )
    return sample


class CausalSampleContractTests(unittest.TestCase):

    def test_contract_is_frozen_slotted_pickle_safe_and_versioned(self):
        sample = _sample()

        self.assertFalse(hasattr(sample, '__dict__'))
        self.assertEqual(
            sample.schema_version,
            CAUSAL_SAMPLE_SCHEMA_VERSION,
        )
        with self.assertRaises((AttributeError, TypeError)):
            sample.direction = -1
        restored = pickle.loads(pickle.dumps(sample))
        self.assertEqual(restored.transition_key, sample.transition_key)
        self.assertEqual(restored.event_key, sample.event_key)
        self.assertTrue(torch.equal(
            restored.graph.node_features,
            sample.graph.node_features,
        ))

    def test_rejects_invalid_actions_direction_finite_values_and_version(self):
        sample = _sample()

        with self.assertRaises(ValueError):
            replace(sample, comparison_action_offset=0)
        with self.assertRaises(ValueError):
            replace(sample, actual_action_offset=ANALYSIS_ACTION_COUNT)
        with self.assertRaises(ValueError):
            replace(sample, direction=0)
        with self.assertRaises(ValueError):
            replace(sample, target_margin=float('nan'))
        with self.assertRaises(ValueError):
            replace(sample, confidence=0.0)
        with self.assertRaises(ValueError):
            replace(sample, schema_version=999)

    def test_rejects_non_finite_or_malformed_graph(self):
        sample = _sample()
        bad_features = sample.graph.node_features.clone()
        bad_features[0, 0] = float('nan')
        with self.assertRaises(ValueError):
            replace(
                sample,
                graph=replace(
                    sample.graph,
                    node_features=bad_features,
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                sample,
                graph=replace(
                    sample.graph,
                    action_node_indices=torch.tensor(
                        [99] * ANALYSIS_ACTION_COUNT,
                        dtype=torch.long,
                    ),
                ),
            )

    def test_supervision_kind_stratum_and_delta_direction_are_consistent(self):
        with self.assertRaises(ValueError):
            _sample(stratum='counterfactual')
        with self.assertRaises(ValueError):
            _sample(
                supervision_kind='counterfactual',
                stratum='positive_setup',
            )
        with self.assertRaises(ValueError):
            _sample(
                supervision_kind='counterfactual',
                stratum='counterfactual',
                target_delta=-1.0,
                direction=1,
            )

    def test_stable_keys_do_not_depend_on_object_repr(self):
        event_key = AttributionEventKey(2, 3, 4)
        merge_key = MergeValueKey(TransitionKey(2, 3, 5), 6)

        self.assertEqual(
            stable_event_key(event_key),
            'attribution-event-v1:2:3:4',
        )
        self.assertEqual(
            stable_budget_key(event_key),
            'attribution-budget-v1:event:2:3:4',
        )
        self.assertEqual(
            stable_budget_key(merge_key),
            'attribution-budget-v1:merge:2:3:5:6',
        )


class CausalReplayBufferTests(unittest.TestCase):

    def test_fixed_capacity_preserves_rare_strata_on_flood(self):
        replay = CausalReplayBuffer(capacity=3, seed=11)
        replay.extend((
            _sample(step=0, event_suffix='p0'),
            _sample(step=1, event_suffix='p1'),
            _sample(step=2, event_suffix='p2'),
        ))
        replay.push(_sample(
            step=3,
            stratum='negative_blocking',
            direction=-1,
            cause_type='REACHABILITY_SEALED',
            event_suffix='n',
        ))
        replay.push(_sample(
            step=4,
            stratum='counterfactual',
            supervision_kind='counterfactual',
            cause_type='COUNTERFACTUAL_RETURN',
            target_delta=1.0,
            event_suffix='cf',
        ))

        self.assertEqual(len(replay), 3)
        self.assertEqual(
            replay.storage_stats['stratum_counts'],
            {
                'positive_setup': 1,
                'negative_blocking': 1,
                'counterfactual': 1,
            },
        )
        self.assertEqual(replay.storage_stats['eviction_count'], 2)

    def test_sampling_is_seed_deterministic_and_balances_strata_and_causes(self):
        values = (
            _sample(step=0, cause_type='A', event_suffix='a0'),
            _sample(step=1, cause_type='A', event_suffix='a1'),
            _sample(step=2, cause_type='B', event_suffix='b'),
            _sample(
                step=3,
                stratum='negative_blocking',
                direction=-1,
                cause_type='N',
                event_suffix='n',
            ),
            _sample(
                step=4,
                stratum='counterfactual',
                supervision_kind='counterfactual',
                cause_type='CF',
                target_delta=1.0,
                event_suffix='cf',
            ),
        )
        first = CausalReplayBuffer(capacity=10, seed=123)
        second = CausalReplayBuffer(capacity=10, seed=123)
        first.extend(values)
        second.extend(values)

        first_batch = first.sample(5)
        second_batch = second.sample(5)
        self.assertEqual(
            tuple(item.event_key for item in first_batch),
            tuple(item.event_key for item in second_batch),
        )
        self.assertEqual(
            {item.stratum for item in first_batch},
            {
                'positive_setup',
                'negative_blocking',
                'counterfactual',
            },
        )
        self.assertIn('A', {item.cause_type for item in first_batch})
        self.assertIn('B', {item.cause_type for item in first_batch})

    def test_counterfactual_replaces_same_unordered_pair_rule(self):
        replay = CausalReplayBuffer(capacity=10, seed=0)
        rule = _sample(event_suffix='rule')
        counterfactual = _sample(
            supervision_kind='counterfactual',
            stratum='counterfactual',
            actual=14,
            comparison=0,
            direction=-1,
            target_delta=-2.0,
            event_suffix='cf',
        )

        self.assertTrue(replay.push(rule))
        self.assertTrue(replay.push(counterfactual))
        self.assertEqual(replay.to_tuple(), (counterfactual,))
        self.assertEqual(
            replay.storage_stats['counterfactual_override_count'],
            1,
        )
        self.assertEqual(
            replay.storage_stats['rule_empirical_agreement_count'],
            1,
        )
        self.assertEqual(
            replay.storage_stats['rule_empirical_disagreement_count'],
            0,
        )
        self.assertFalse(replay.push(rule))
        self.assertEqual(replay.to_tuple(), (counterfactual,))
        self.assertEqual(
            replay.storage_stats['ignored_weaker_rule_count'],
            1,
        )

    def test_pair_indexes_preserve_multi_rule_override_and_priority_semantics(
            self):
        replay = CausalReplayBuffer(capacity=10, seed=4)
        positive_rule = _sample(
            event_suffix='positive-rule',
        )
        negative_rule = _sample(
            stratum='negative_blocking',
            direction=-1,
            event_suffix='negative-rule',
        )
        counterfactual = _sample(
            supervision_kind='counterfactual',
            stratum='counterfactual',
            actual=14,
            comparison=0,
            direction=1,
            target_delta=2.0,
            event_suffix='counterfactual',
        )
        shapley = replace(
            counterfactual,
            supervision_kind='shapley',
            event_key='event:shapley',
            budget_key='budget:shapley',
        )

        replay.extend((positive_rule, negative_rule))
        self.assertTrue(replay.push(counterfactual))
        self.assertEqual(replay.to_tuple(), (counterfactual,))
        self.assertEqual(
            replay.storage_stats['rule_empirical_agreement_count'],
            1,
        )
        self.assertEqual(
            replay.storage_stats['rule_empirical_disagreement_count'],
            1,
        )
        self.assertTrue(replay.push(shapley))
        self.assertEqual(replay.to_tuple(), (shapley,))
        self.assertFalse(replay.push(counterfactual))
        self.assertFalse(replay.push(positive_rule))
        self.assertEqual(replay.to_tuple(), (shapley,))
        self.assertEqual(
            replay.storage_stats['ignored_weaker_rule_count'],
            2,
        )

    def test_extend_readiness_stats_tuple_and_clear(self):
        replay = CausalReplayBuffer(capacity=4, seed=0)
        first = _sample(step=0, event_suffix='0')
        # dataclasses.replace 保留同一个 GraphTensor 引用，模拟同一图产生多个
        # 稀疏因果标签时的真实内存共享。
        second = replace(
            first,
            transition_key=TransitionKey(0, 0, 1),
            event_key='event:1',
            budget_key='budget:1',
        )
        values = first, second
        self.assertEqual(replay.extend(values), 2)
        self.assertTrue(replay.is_ready(2))
        self.assertFalse(replay.is_ready(3))
        self.assertEqual(replay.to_tuple(), values)
        self.assertEqual(
            replay.storage_stats['supervision_kind_counts']['rule'],
            2,
        )
        self.assertEqual(
            replay.storage_stats['unique_graph_count'],
            1,
        )
        self.assertGreater(
            replay.storage_stats['estimated_graph_sharing_saved_bytes'],
            0,
        )
        with self.assertRaises(ValueError):
            replay.sample(3)
        replay.clear()
        self.assertEqual(len(replay), 0)
        self.assertEqual(replay.to_tuple(), ())
        self.assertEqual(
            replay.storage_stats['estimated_unique_graph_bytes'],
            0,
        )

    def test_checkpoint_round_trip_restores_items_rng_and_sampling_cursors(self):
        values = (
            _sample(step=0, cause_type='A', event_suffix='a'),
            _sample(
                step=1,
                stratum='negative_blocking',
                direction=-1,
                cause_type='B',
                event_suffix='b',
            ),
            _sample(
                step=2,
                stratum='counterfactual',
                supervision_kind='counterfactual',
                cause_type='CF',
                target_delta=1.0,
                event_suffix='cf',
            ),
        )
        source = CausalReplayBuffer(capacity=8, seed=17)
        source.extend(values)
        source.sample(2)
        state = source.checkpoint_state_dict()

        restored = CausalReplayBuffer(capacity=8, seed=999)
        restored.validate_checkpoint_manifest(
            source.checkpoint_manifest()
        )
        restored.load_checkpoint_state_dict(state)

        self.assertEqual(restored.to_tuple(), source.to_tuple())
        self.assertEqual(restored.storage_stats, source.storage_stats)
        self.assertEqual(
            tuple(item.event_key for item in restored.sample(3)),
            tuple(item.event_key for item in source.sample(3)),
        )

    def test_checkpoint_restores_dense_sampling_index_after_swap_removals(self):
        source = CausalReplayBuffer(capacity=6, seed=29)
        source.extend(tuple(
            _sample(
                step=step,
                cause_type='A' if step % 2 == 0 else 'B',
                event_suffix=f'rule-{step}',
            )
            for step in range(6)
        ))
        source.push(_sample(
            step=6,
            stratum='negative_blocking',
            direction=-1,
            cause_type='N',
            event_suffix='negative',
        ))
        source.push(_sample(
            step=7,
            stratum='counterfactual',
            supervision_kind='counterfactual',
            cause_type='CF',
            target_delta=1.0,
            event_suffix='counterfactual',
        ))
        source.sample(4)

        state = source.checkpoint_state_dict()
        restored = CausalReplayBuffer(capacity=6, seed=999)
        restored.load_checkpoint_state_dict(state)

        for _ in range(8):
            self.assertEqual(
                tuple(item.event_key for item in restored.sample(4)),
                tuple(item.event_key for item in source.sample(4)),
            )

    def test_legacy_checkpoint_without_index_state_remains_loadable(self):
        source = CausalReplayBuffer(capacity=5, seed=31)
        source.extend(tuple(
            _sample(step=step, event_suffix=f'legacy-{step}')
            for step in range(5)
        ))
        state = source.checkpoint_state_dict()
        del state['index_state']

        restored = CausalReplayBuffer(capacity=5, seed=0)
        restored.load_checkpoint_state_dict(state)

        self.assertEqual(restored.to_tuple(), source.to_tuple())
        self.assertEqual(len(restored.sample(5)), 5)

    def test_capacity_20k_push_eviction_and_sampling_stress(self):
        """正式容量下的热路径不能退化为逐样本全池扫描。"""

        replay = CausalReplayBuffer(capacity=20_000, seed=43)
        templates = (
            _sample(event_suffix='positive-template'),
            _sample(
                stratum='negative_blocking',
                direction=-1,
                event_suffix='negative-template',
            ),
            _sample(
                stratum='counterfactual',
                supervision_kind='counterfactual',
                target_delta=1.0,
                event_suffix='counterfactual-template',
            ),
        )
        started = time.perf_counter()
        for step in range(22_000):
            template = templates[step % len(templates)]
            replay.push(_fast_sample_clone(
                template,
                step=step,
                cause_type=f'CAUSE_{step % 8}',
                event_suffix=f'stress-{step}',
            ))
        for _ in range(128):
            batch = replay.sample(32)
            self.assertEqual(len({item.event_key for item in batch}), 32)
        elapsed = time.perf_counter() - started

        self.assertEqual(len(replay), 20_000)
        self.assertEqual(replay.storage_stats['eviction_count'], 2_000)
        self.assertLess(
            elapsed,
            15.0,
            '20k indexed causal replay hot path regressed: '
            f'{elapsed:.3f}s',
        )


class RuleCausalContextCacheTests(unittest.TestCase):

    def test_context_validates_graph_analysis_and_actual_action_identity(self):
        key = TransitionKey(3, 4, 5)
        context = _context(key, action_offset=2)

        self.assertEqual(context.transition_key, key)
        self.assertEqual(context.actual_action_index, 2)
        self.assertEqual(
            context.graph_schema_fingerprint,
            graph_schema_fingerprint(context.graph),
        )
        with self.assertRaises(ValueError):
            replace(context, actual_action_index=9)
        with self.assertRaises(ValueError):
            replace(
                context,
                graph=_graph(action_indices=tuple(range(1, 16))),
            )

    def test_cache_evicts_oldest_and_discards_episode(self):
        cache = RuleCausalContextCache(capacity=2)
        first = _context(TransitionKey(0, 1, 0))
        second = _context(TransitionKey(0, 1, 1))
        third = _context(TransitionKey(1, 2, 0))

        cache.put(first)
        cache.put(second)
        cache.put(third)
        self.assertIsNone(cache.get(first.transition_key))
        self.assertIs(cache.get(second.transition_key), second)
        self.assertEqual(cache.storage_stats['eviction_count'], 1)
        self.assertEqual(cache.discard_episode(0, 1), 1)
        self.assertEqual(cache.to_tuple(), (third,))


class RuleCausalSampleBuilderTests(unittest.TestCase):

    def test_real_tracker_direct_merge_builds_one_budgeted_rule(self):
        key = TransitionKey(7, 3, 0)
        tracker = AttributionTracker()
        tracker.begin_episode(worker_id=7, episode_id=3)
        transition = make_transition(
            worker_id=7,
            episode_id=3,
            step_index=0,
            previous_specs=(FruitSpec(1, level=1),),
            next_specs=(FruitSpec(3, level=2),),
            drop_fruit_id=2,
            merge_events=(MergeEvent(
                new_level=2,
                x=200.0,
                y=500.0,
                score_delta=merge_score(2),
                source_ids=(1, 2),
                new_fruit_id=3,
            ),),
        )
        attribution = tracker.observe_transition(transition)
        builder = RuleCausalSampleBuilder()
        builder.remember_transition(
            graph=_graph(
                action_indices=(
                    transition.previous_analysis.action_indices
                ),
            ),
            state_analysis=transition.previous_analysis,
            actual_action_offset=0,
            actual_action_index=0,
        )

        samples = builder.build(
            attribution.created_events
            + attribution.resolved_events
        )

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].transition_key, key)
        self.assertEqual(samples[0].cause_type, 'DIRECT_TRIGGER')
        self.assertAlmostEqual(samples[0].target_margin, 0.65)

    def test_positive_budget_is_merged_once_and_uses_mirror_comparator(self):
        key = TransitionKey(0, 2, 4)
        budget_key = MergeValueKey(key, 0)
        contributor = _contributor(key)
        lineage = _event(
            key=key,
            event_index=0,
            event_type='MERGE_LINEAGE',
            utility=4.0,
            link_confidence=1.0,
            placement_confidence=0.80,
            contributors=(contributor,),
            budget_key=budget_key,
        )
        trigger = _event(
            key=key,
            event_index=1,
            event_type='DIRECT_TRIGGER',
            utility=0.0,
            link_confidence=0.95,
            placement_confidence=0.90,
            contributors=(contributor,),
            budget_key=budget_key,
        )
        builder = RuleCausalSampleBuilder()
        builder.remember_context(_context(key))

        result = builder.build_with_stats((lineage, trigger))

        self.assertEqual(result.stats.budget_count, 1)
        self.assertEqual(result.stats.generated_sample_count, 1)
        sample = result.samples[0]
        self.assertEqual(
            sample.comparison_action_offset,
            ANALYSIS_ACTION_COUNT - 1,
        )
        self.assertEqual(sample.direction, 1)
        self.assertEqual(sample.cause_type, 'DIRECT_TRIGGER')
        self.assertAlmostEqual(sample.target_margin, 4.0 * 0.90)
        self.assertEqual(sample.budget_key, stable_budget_key(budget_key))
        self.assertEqual(
            sample.tracker_config_fingerprint,
            trigger.tracker_config_fingerprint,
        )
        self.assertEqual(sample.policy_version, 'policy-7')

    def test_c_lineage_can_supply_unique_utility_to_b_realization(self):
        """C 级谱系不直接监督，但不能丢掉共享 budget 的唯一价值包。"""

        key = TransitionKey(0, 2, 5)
        budget_key = MergeValueKey(key, 0)
        lineage = _event(
            key=key,
            event_index=0,
            event_type='MERGE_LINEAGE',
            utility=4.0,
            link_confidence=1.0,
            placement_confidence=0.50,
            budget_key=budget_key,
        )
        realization = _event(
            key=key,
            event_index=1,
            event_type='REALIZED_LADDER',
            utility=0.0,
            link_confidence=0.85,
            placement_confidence=0.65,
            budget_key=budget_key,
        )
        builder = RuleCausalSampleBuilder()
        builder.remember_context(_context(key))

        result = builder.build_with_stats((lineage, realization))

        self.assertEqual(len(result.samples), 1)
        self.assertEqual(
            result.samples[0].cause_type,
            'REALIZED_LADDER',
        )
        self.assertAlmostEqual(
            result.samples[0].target_margin,
            4.0 * 0.65,
        )
        self.assertEqual(
            result.stats.reason_count('confidence_tier_c'),
            1,
        )

    def test_c_event_alone_never_creates_rule_supervision(self):
        key = TransitionKey(0, 2, 6)
        event = _event(
            key=key,
            event_type='MERGE_LINEAGE',
            link_confidence=1.0,
            placement_confidence=0.50,
            utility=8.0,
        )
        builder = RuleCausalSampleBuilder()
        builder.remember_context(_context(key))

        self.assertEqual(builder.build((event,)), ())
        self.assertEqual(
            builder.last_stats.reason_count('confidence_tier_c'),
            1,
        )
        self.assertEqual(
            builder.last_stats.reason_count('no_ab_label_event'),
            1,
        )

    def test_multiple_contributors_share_one_unique_budget(self):
        first_key = TransitionKey(0, 3, 1)
        second_key = TransitionKey(0, 3, 2)
        contributors = (
            _contributor(
                first_key,
                action_offset=0,
                weight=0.25,
                fruit_id=1,
            ),
            _contributor(
                second_key,
                action_offset=1,
                weight=0.75,
                fruit_id=2,
            ),
        )
        event = _event(
            key=first_key,
            contributors=contributors,
            utility=8.0,
        )
        builder = RuleCausalSampleBuilder()
        builder.remember_context(_context(first_key, action_offset=0))
        builder.remember_context(_context(second_key, action_offset=1))

        samples = builder.build((event,))

        self.assertEqual(len(samples), 2)
        margins = {
            sample.transition_key: sample.target_margin
            for sample in samples
        }
        self.assertAlmostEqual(
            margins[first_key],
            8.0 * 0.25 * 0.90,
        )
        self.assertAlmostEqual(
            margins[second_key],
            8.0 * 0.75 * 0.90,
        )
        self.assertAlmostEqual(sum(margins.values()), 8.0 * 0.90)

    def test_negative_uses_strictly_safer_action_and_clips_utility(self):
        key = TransitionKey(1, 0, 3)
        analysis = _safer_analysis(key, safest_offset=14)
        event = _event(
            key=key,
            event_type='REACHABILITY_SEALED',
            sign=-1,
            utility=100.0,
            link_confidence=0.85,
            placement_confidence=0.80,
        )
        builder = RuleCausalSampleBuilder()
        builder.remember_context(_context(key, analysis=analysis))

        samples = builder.build((event,))

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.stratum, 'negative_blocking')
        self.assertEqual(sample.direction, -1)
        self.assertEqual(sample.comparison_action_offset, 14)
        self.assertLessEqual(
            sample.target_margin,
            merge_utility(5),
        )
        self.assertAlmostEqual(
            sample.target_margin,
            merge_utility(5) * 0.80,
        )

    def test_c_pending_missing_context_and_no_comparator_are_skipped(self):
        missing_key = TransitionKey(0, 4, 0)
        equal_safety_key = TransitionKey(0, 4, 1)
        builder = RuleCausalSampleBuilder()
        builder.remember_context(_context(equal_safety_key))
        events = (
            _event(
                key=missing_key,
                event_index=0,
            ),
            _event(
                key=missing_key,
                event_index=1,
                placement_confidence=0.30,
            ),
            _event(
                key=missing_key,
                event_index=2,
                status='pending',
            ),
            _event(
                key=equal_safety_key,
                event_index=3,
                event_type='REACHABILITY_SEALED',
                sign=-1,
            ),
        )

        result = builder.build_with_stats(events)

        self.assertEqual(result.samples, ())
        self.assertEqual(result.stats.reason_count('missing_context'), 1)
        self.assertEqual(
            result.stats.reason_count('confidence_tier_c'),
            1,
        )
        self.assertEqual(
            result.stats.reason_count('event_not_confirmed'),
            1,
        )
        self.assertEqual(
            result.stats.reason_count('no_trustworthy_comparison'),
            1,
        )

    def test_center_positive_action_without_distinct_mirror_is_skipped(self):
        key = TransitionKey(0, 5, 0)
        event = _event(
            key=key,
            contributors=(_contributor(
                key,
                action_offset=ANALYSIS_ACTION_COUNT // 2,
            ),),
        )
        builder = RuleCausalSampleBuilder()
        builder.remember_context(_context(
            key,
            action_offset=ANALYSIS_ACTION_COUNT // 2,
        ))

        self.assertEqual(builder.build((event,)), ())
        self.assertEqual(
            builder.last_stats.reason_count(
                'no_trustworthy_comparison'
            ),
            1,
        )

    def test_invalid_analysis_is_not_used_for_rule_labels(self):
        key = TransitionKey(0, 6, 0)
        invalid = make_analysis((), key=key, valid=False)
        builder = RuleCausalSampleBuilder()
        builder.remember_context(_context(key, analysis=invalid))

        self.assertEqual(builder.build((_event(key=key),)), ())
        self.assertEqual(
            builder.last_stats.reason_count('invalid_context_analysis'),
            1,
        )

    def test_mixed_versions_in_one_budget_are_rejected(self):
        key = TransitionKey(0, 7, 0)
        budget_key = MergeValueKey(key, 0)
        builder = RuleCausalSampleBuilder()
        builder.remember_context(_context(key))
        events = (
            _event(
                key=key,
                event_index=0,
                budget_key=budget_key,
                attribution_version='v1',
            ),
            _event(
                key=key,
                event_index=1,
                budget_key=budget_key,
                attribution_version='v2',
            ),
        )

        self.assertEqual(builder.build(events), ())
        self.assertEqual(
            builder.last_stats.reason_count(
                'mixed_attribution_version'
            ),
            1,
        )


if __name__ == '__main__':
    unittest.main()
