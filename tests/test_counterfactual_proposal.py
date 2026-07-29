"""worker-local 反事实 proposal 契约与触发测试。"""

from __future__ import annotations

import pickle
import unittest
from dataclasses import replace

from daxigua.core.engine import HeadlessGame
from daxigua_rl.attribution.counterfactual_proposal import (
    CounterfactualProposal,
    CounterfactualProposalBuilder,
    should_transfer_counterfactual_proposal,
    stable_counterfactual_proposal_id,
)
from daxigua_rl.attribution.schema import MergeLineageRecord, MergeValueKey
from daxigua_rl.reward import merge_utility
from daxigua_rl.training import TransitionKey

from test_causal_replay import (
    _contributor,
    _context,
    _event,
    _safer_analysis,
)


def _game_entry(key, *, action_offset=0, game=None, analysis=None):
    game = game or HeadlessGame(seed=key.step_index)
    if key.step_index == 0:
        game.reset(fruit_queue=(1, 1, 1, 1))
    snapshot = game.capture_snapshot()
    candidates = game.get_action_candidates(15)
    outcome = game.execute_action(
        candidates[action_offset].drop_x,
        max_frames=300,
        stable_frames=6,
    )
    context = _context(
        key,
        action_offset=action_offset,
        analysis=analysis,
    )
    return game, context, snapshot, outcome


class CounterfactualProposalContractTest(unittest.TestCase):

    def test_delayed_middle_confidence_event_uses_historical_entry(self):
        key = TransitionKey(2, 3, 0)
        _game, context, snapshot, outcome = _game_entry(
            key,
            analysis=_safer_analysis(key, safest_offset=14),
        )
        event = _event(
            key=key,
            placement_confidence=0.65,
            detected_step=0,
            resolved_step=2,
        )
        builder = CounterfactualProposalBuilder(ring_size=32)
        builder.remember(
            context=context,
            snapshot=snapshot,
            factual_outcome=outcome,
        )

        result = builder.build_with_stats((event,))

        self.assertEqual(len(result.proposals), 1)
        proposal = result.proposals[0]
        self.assertIsInstance(proposal, CounterfactualProposal)
        self.assertEqual(proposal.transition_key, key)
        self.assertIs(proposal.context, context)
        self.assertIs(proposal.snapshot, snapshot)
        self.assertIs(proposal.factual_outcome, outcome)
        self.assertEqual(
            proposal.trigger_reasons,
            ('middle_placement_confidence',),
        )
        # 21 列下 mirror=20 与最安全动作 14 不同，两者都应稳定保留。
        self.assertEqual(
            proposal.alternative_action_offsets,
            (20, 14),
        )
        self.assertEqual(proposal.delay, 2)
        self.assertEqual(
            proposal.proposal_id,
            stable_counterfactual_proposal_id(
                budget_key=proposal.budget_key,
                representative_event=proposal.representative_event,
                contributor=proposal.contributor,
                context=proposal.context,
                snapshot=proposal.snapshot,
                factual_outcome=proposal.factual_outcome,
                alternative_action_offsets=(
                    proposal.alternative_action_offsets
                ),
                trigger_reasons=proposal.trigger_reasons,
            ),
        )
        restored = pickle.loads(pickle.dumps(proposal))
        self.assertEqual(restored.proposal_id, proposal.proposal_id)
        with self.assertRaises(ValueError):
            replace(proposal, actual_action_offset=1)
        with self.assertRaises(ValueError):
            replace(
                proposal,
                alternative_action_offsets=(14, 14),
            )
        with self.assertRaises(ValueError):
            replace(
                proposal,
                factual_outcome=replace(
                    proposal.factual_outcome,
                    drop_result=replace(
                        proposal.factual_outcome.drop_result,
                        dropped_level=2,
                    ),
                ),
            )

    def test_high_level_merge_trigger_and_budget_are_deduplicated(self):
        key = TransitionKey(0, 1, 0)
        _game, context, snapshot, outcome = _game_entry(key)
        budget_key = MergeValueKey(key, 0)
        event = _event(
            key=key,
            budget_key=budget_key,
            utility=merge_utility(7),
        )
        record = MergeLineageRecord(
            value_key=budget_key,
            source_fruit_ids=(1, 2),
            new_fruit_id=3,
            new_level=7,
            utility=merge_utility(7),
            root_material_weights=((1, 0.5), (2, 0.5)),
            chain_depth=1,
        )
        builder = CounterfactualProposalBuilder()
        builder.remember(
            context=context,
            snapshot=snapshot,
            factual_outcome=outcome,
        )

        first = builder.build_with_stats(
            (event,),
            merge_records=(record,),
        )
        second = builder.build_with_stats(
            (event,),
            merge_records=(record,),
        )

        self.assertEqual(len(first.proposals), 1)
        self.assertIn(
            'high_value_merge',
            first.proposals[0].trigger_reasons,
        )
        self.assertEqual(first.proposals[0].budget_key, budget_key)
        self.assertEqual(second.proposals, ())
        self.assertEqual(
            second.stats.reason_count('duplicate_budget'),
            1,
        )

    def test_transfer_sampling_is_stable_and_keeps_critical_proposals(self):
        ordinary_key = TransitionKey(0, 2, 0)
        _game, context, snapshot, outcome = _game_entry(ordinary_key)
        ordinary_builder = CounterfactualProposalBuilder()
        ordinary_builder.remember(
            context=context,
            snapshot=snapshot,
            factual_outcome=outcome,
        )
        ordinary = ordinary_builder.build((_event(
            key=ordinary_key,
            placement_confidence=0.65,
        ),))[0]

        self.assertFalse(should_transfer_counterfactual_proposal(
            ordinary,
            sample_rate=0.0,
        ))
        self.assertTrue(should_transfer_counterfactual_proposal(
            ordinary,
            sample_rate=1.0,
        ))
        decisions = tuple(
            should_transfer_counterfactual_proposal(
                ordinary,
                sample_rate=0.0625,
            )
            for _ in range(4)
        )
        self.assertEqual(len(set(decisions)), 1)

        critical_key = TransitionKey(0, 3, 0)
        _game, context, snapshot, outcome = _game_entry(critical_key)
        budget_key = MergeValueKey(critical_key, 0)
        critical_builder = CounterfactualProposalBuilder()
        critical_builder.remember(
            context=context,
            snapshot=snapshot,
            factual_outcome=outcome,
        )
        critical = critical_builder.build(
            (_event(
                key=critical_key,
                budget_key=budget_key,
                utility=merge_utility(7),
            ),),
            merge_records=(MergeLineageRecord(
                value_key=budget_key,
                source_fruit_ids=(1, 2),
                new_fruit_id=3,
                new_level=7,
                utility=merge_utility(7),
                root_material_weights=((1, 0.5), (2, 0.5)),
                chain_depth=1,
            ),),
        )[0]
        self.assertTrue(should_transfer_counterfactual_proposal(
            critical,
            sample_rate=0.0,
        ))

        with self.assertRaises(ValueError):
            should_transfer_counterfactual_proposal(
                ordinary,
                sample_rate=1.01,
            )

    def test_ring_expiry_skips_instead_of_fabricating_proposal(self):
        first_key = TransitionKey(0, 4, 0)
        game, first_context, first_snapshot, first_outcome = (
            _game_entry(first_key)
        )
        second_key = TransitionKey(0, 4, 1)
        game, second_context, second_snapshot, second_outcome = (
            _game_entry(second_key, game=game)
        )
        builder = CounterfactualProposalBuilder(ring_size=1)
        builder.remember(
            context=first_context,
            snapshot=first_snapshot,
            factual_outcome=first_outcome,
        )
        evicted = builder.remember(
            context=second_context,
            snapshot=second_snapshot,
            factual_outcome=second_outcome,
        )

        result = builder.build_with_stats((_event(
            key=first_key,
            placement_confidence=0.65,
        ),))

        self.assertEqual(evicted.transition_key, first_key)
        self.assertEqual(builder.history.keys, (second_key,))
        self.assertEqual(result.proposals, ())
        self.assertEqual(
            result.stats.reason_count('missing_history'),
            1,
        )

    def test_same_historical_action_positive_negative_conflict_triggers_both(self):
        key = TransitionKey(1, 2, 0)
        _game, context, snapshot, outcome = _game_entry(key)
        positive = _event(
            key=key,
            event_index=0,
            placement_confidence=0.90,
        )
        negative = _event(
            key=key,
            event_index=1,
            event_type='REACHABILITY_SEALED',
            sign=-1,
            placement_confidence=0.90,
        )
        builder = CounterfactualProposalBuilder()
        builder.remember(
            context=context,
            snapshot=snapshot,
            factual_outcome=outcome,
        )

        proposals = builder.build((positive, negative))

        self.assertEqual(len(proposals), 2)
        self.assertTrue(all(
            'conflicting_signals' in proposal.trigger_reasons
            for proposal in proposals
        ))
        self.assertEqual(
            len({proposal.proposal_id for proposal in proposals}),
            2,
        )

    def test_delayed_confirmation_retains_merge_trigger_metadata(self):
        key = TransitionKey(4, 6, 0)
        _game, context, snapshot, outcome = _game_entry(key)
        budget_key = MergeValueKey(key, 0)
        event = _event(
            key=key,
            budget_key=budget_key,
            utility=merge_utility(7),
        )
        record = MergeLineageRecord(
            value_key=budget_key,
            source_fruit_ids=(1, 2),
            new_fruit_id=3,
            new_level=7,
            utility=merge_utility(7),
            root_material_weights=((1, 0.5), (2, 0.5)),
            chain_depth=2,
        )
        builder = CounterfactualProposalBuilder()
        builder.remember(
            context=context,
            snapshot=snapshot,
            factual_outcome=outcome,
        )

        detected = builder.build_with_stats(
            (),
            merge_records=(record,),
        )
        resolved = builder.build_with_stats((event,))

        self.assertEqual(detected.proposals, ())
        self.assertEqual(len(resolved.proposals), 1)
        self.assertEqual(
            resolved.proposals[0].trigger_reasons[:2],
            ('high_value_merge', 'multi_stage_chain'),
        )

    def test_conflict_and_budget_ranking_span_separate_build_calls(self):
        keys = (
            TransitionKey(5, 7, 0),
            TransitionKey(5, 7, 1),
        )
        game = None
        entries = []
        builder = CounterfactualProposalBuilder()
        for key in keys:
            game, context, snapshot, outcome = _game_entry(
                key,
                game=game,
            )
            builder.remember(
                context=context,
                snapshot=snapshot,
                factual_outcome=outcome,
            )
            entries.append((context, snapshot, outcome))

        positive = _event(
            key=keys[0],
            event_index=0,
            placement_confidence=0.90,
        )
        negative = _event(
            key=keys[0],
            event_index=1,
            event_type='REACHABILITY_SEALED',
            sign=-1,
            placement_confidence=0.90,
        )
        self.assertEqual(builder.build((positive,)), ())
        conflicts = builder.build((negative,))
        self.assertEqual(len(conflicts), 2)
        self.assertTrue(all(
            'conflicting_signals' in proposal.trigger_reasons
            for proposal in conflicts
        ))

        ranked_builder = CounterfactualProposalBuilder()
        for context, snapshot, outcome in entries:
            ranked_builder.remember(
                context=context,
                snapshot=snapshot,
                factual_outcome=outcome,
            )
        shared_budget = MergeValueKey(keys[-1], 9)
        strongest = _event(
            key=keys[0],
            event_index=2,
            budget_key=shared_budget,
            placement_confidence=0.90,
        )
        ambiguous = _event(
            key=keys[1],
            event_index=3,
            budget_key=shared_budget,
            link_confidence=0.80,
            placement_confidence=0.65,
        )
        self.assertEqual(ranked_builder.build((strongest,)), ())
        ranked = ranked_builder.build((ambiguous,))

        self.assertEqual(len(ranked), 1)
        self.assertEqual(
            ranked[0].contributor.transition_key,
            keys[0],
        )
        self.assertIn(
            'middle_placement_confidence',
            ranked[0].trigger_reasons,
        )

    def test_high_value_multi_contributor_keeps_contiguous_coalition_trace(self):
        keys = tuple(
            TransitionKey(3, 5, step_index)
            for step_index in range(3)
        )
        game = None
        entries = []
        builder = CounterfactualProposalBuilder()
        for key in keys:
            game, context, snapshot, outcome = _game_entry(
                key,
                game=game,
            )
            builder.remember(
                context=context,
                snapshot=snapshot,
                factual_outcome=outcome,
            )
            entries.append((context, snapshot, outcome))

        budget_key = MergeValueKey(keys[-1], 0)
        event = _event(
            key=keys[-1],
            budget_key=budget_key,
            utility=merge_utility(7),
            contributors=(
                _contributor(
                    keys[0],
                    weight=0.5,
                    fruit_id=1,
                ),
                _contributor(
                    keys[-1],
                    weight=0.5,
                    fruit_id=2,
                ),
            ),
        )
        record = MergeLineageRecord(
            value_key=budget_key,
            source_fruit_ids=(1, 2),
            new_fruit_id=3,
            new_level=7,
            utility=merge_utility(7),
            root_material_weights=((1, 0.5), (2, 0.5)),
            chain_depth=2,
        )

        proposal = builder.build(
            (event,),
            merge_records=(record,),
        )[0]

        self.assertTrue(proposal.shapley_ready)
        self.assertEqual(
            proposal.coalition_candidate_keys,
            (keys[0], keys[-1]),
        )
        self.assertEqual(
            tuple(
                entry.transition_key
                for entry in proposal.coalition_trace_entries
            ),
            keys,
        )
        self.assertEqual(
            tuple(
                entry.transition_key
                for entry in proposal.coalition_candidate_entries
            ),
            (keys[0], keys[-1]),
        )
        self.assertTrue(all(
            entry.alternative_action_offsets
            for entry in proposal.coalition_candidate_entries
        ))
        restored = pickle.loads(pickle.dumps(proposal))
        self.assertEqual(
            restored.coalition_candidate_keys,
            proposal.coalition_candidate_keys,
        )

        with self.assertRaises(ValueError):
            replace(
                proposal,
                coalition_trace_entries=(
                    proposal.coalition_trace_entries[0],
                    proposal.coalition_trace_entries[-1],
                ),
            )
        with self.assertRaises(ValueError):
            replace(
                proposal,
                coalition_candidate_keys=(
                    keys[-1],
                    keys[0],
                ),
            )


if __name__ == '__main__':
    unittest.main()
