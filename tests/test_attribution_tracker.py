"""Lineage, event budget and delayed attribution lifecycle tests."""

from __future__ import annotations

import math
import pickle
import unittest
from dataclasses import replace

from daxigua.core.engine import HeadlessGame
from daxigua.core.state import MergeEvent
from daxigua_rl.attribution import (
    ANALYSIS_ACTION_COUNT,
    ATTRIBUTION_EVENT_TYPES,
    NEGATIVE_ATTRIBUTION_EVENT_TYPES,
    POSITIVE_ATTRIBUTION_EVENT_TYPES,
    AttributionTracker,
    AttributionTrackerConfig,
    SupportEdge,
)
from daxigua_rl.env import DaxiguaEnv, DaxiguaEnvConfig
from daxigua_rl.graph import GraphBuilder
from daxigua_rl.training import ReplayBuffer, RolloutCollector

from tests.attribution_fixtures import (
    ContactInfluenceEdge,
    FruitSpec,
    make_transition,
    merge_pair_motif,
)


def _events(result, event_type):
    return tuple(
        event
        for event in result.created_events + result.resolved_events
        if event.event_type == event_type
    )


def _merge(
        new_level,
        source_ids,
        new_fruit_id):
    from daxigua.core.rules import merge_score

    return MergeEvent(
        new_level=new_level,
        x=200.0,
        y=500.0,
        score_delta=merge_score(new_level),
        source_ids=tuple(source_ids),
        new_fruit_id=new_fruit_id,
    )


class AttributionSchemaAndLineageTest(unittest.TestCase):
    def test_event_type_catalog_covers_all_specified_positive_and_negative_types(self):
        self.assertEqual(len(POSITIVE_ATTRIBUTION_EVENT_TYPES), 13)
        self.assertEqual(len(NEGATIVE_ATTRIBUTION_EVENT_TYPES), 15)
        self.assertEqual(
            set(ATTRIBUTION_EVENT_TYPES),
            set(POSITIVE_ATTRIBUTION_EVENT_TYPES)
            | set(NEGATIVE_ATTRIBUTION_EVENT_TYPES),
        )

    def test_single_merge_registers_drop_before_merge_and_uses_one_value_package(self):
        tracker = AttributionTracker()
        tracker.begin_episode(worker_id=7, episode_id=3)
        transition = make_transition(
            worker_id=7,
            episode_id=3,
            step_index=0,
            previous_specs=(FruitSpec(1, level=1),),
            next_specs=(FruitSpec(3, level=2),),
            drop_fruit_id=2,
            merge_events=(_merge(2, (1, 2), 3),),
        )

        result = tracker.observe_transition(transition)

        self.assertEqual(len(result.merge_records), 1)
        merge_record = result.merge_records[0]
        self.assertEqual(merge_record.source_fruit_ids, (1, 2))
        self.assertEqual(
            merge_record.root_material_weights,
            ((1, 0.5), (2, 0.5)),
        )
        self.assertEqual(
            tracker.lineage_for(3).parent_fruit_ids,
            (1, 2),
        )
        primary = _events(result, 'MERGE_LINEAGE')
        direct = _events(result, 'DIRECT_TRIGGER')
        self.assertEqual(len(primary), 1)
        self.assertEqual(len(direct), 1)
        self.assertAlmostEqual(primary[0].utility, 1.0)
        self.assertEqual(primary[0].budget_key, merge_record.value_key)
        self.assertEqual(direct[0].budget_key, merge_record.value_key)
        self.assertEqual(direct[0].utility, 0.0)
        self.assertLess(primary[0].placement_confidence, 0.8)
        self.assertEqual(
            primary[0].attribution_version,
            tracker.config.attribution_version,
        )
        self.assertEqual(
            primary[0].tracker_config_fingerprint,
            tracker.config.fingerprint,
        )

        duplicated_value = replace(direct[0], utility=primary[0].utility)
        corrupted_events = tuple(
            duplicated_value if event is direct[0] else event
            for event in result.created_events
        )
        with self.assertRaises(ValueError):
            replace(result, created_events=corrupted_events)
        with self.assertRaises(ValueError):
            replace(
                result,
                created_events=result.created_events
                + (result.created_events[0],),
            )
        with self.assertRaises(ValueError):
            replace(
                result,
                lineage_records=result.lineage_records
                + (result.lineage_records[0],),
            )
        with self.assertRaises(ValueError):
            replace(
                result,
                merge_records=result.merge_records
                + (result.merge_records[0],),
            )

    def test_transition_action_identity_must_match_analysis_offset(self):
        with self.assertRaises(ValueError):
            make_transition(
                worker_id=0,
                episode_id=0,
                step_index=0,
                previous_specs=(),
                next_specs=(FruitSpec(1),),
                drop_fruit_id=1,
                action_offset=0,
                action_index=1,
            )
        valid = make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(),
            next_specs=(FruitSpec(1),),
            drop_fruit_id=1,
        )
        with self.assertRaises(ValueError):
            replace(
                valid,
                drop_result=replace(
                    valid.drop_result,
                    drop_x=valid.drop_result.drop_x + 1.0,
                ),
            )

    def test_multilevel_chain_preserves_all_roots_and_does_not_copy_chain_value(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        transition = make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(
                FruitSpec(1, level=1),
                FruitSpec(4, level=2),
            ),
            next_specs=(FruitSpec(5, level=3),),
            drop_fruit_id=2,
            merge_events=(
                _merge(2, (1, 2), 3),
                _merge(3, (3, 4), 5),
            ),
        )

        result = tracker.observe_transition(transition)

        self.assertEqual(len(result.merge_records), 2)
        self.assertEqual(result.chain_merge_count, 1)
        self.assertEqual(
            tracker.lineage_for(5).root_material_weights,
            ((1, 0.25), (2, 0.25), (4, 0.5)),
        )
        self.assertEqual(tracker.lineage_for(5).chain_depth, 2)
        primary = _events(result, 'MERGE_LINEAGE')
        chain = _events(result, 'CHAIN_TRIGGER')
        self.assertEqual(len(primary), 2)
        self.assertEqual(len(chain), 1)
        self.assertAlmostEqual(
            sum(event.utility for event in primary),
            1.0 + math.sqrt(2.0),
        )
        self.assertEqual(chain[0].utility, 0.0)
        self.assertIn(chain[0].budget_key, {
            event.budget_key
            for event in primary
        })
        self.assertEqual(result.max_transition_chain_depth, 2)

    def test_lineage_depth_does_not_inflate_single_step_chain_metric(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        first = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(
                FruitSpec(1, level=1),
                FruitSpec(4, level=2),
            ),
            next_specs=(
                FruitSpec(3, level=2),
                FruitSpec(4, level=2),
            ),
            drop_fruit_id=2,
            merge_events=(_merge(2, (1, 2), 3),),
        ))
        second = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=1,
            previous_specs=(
                FruitSpec(3, level=2),
                FruitSpec(4, level=2),
            ),
            next_specs=(
                FruitSpec(5, level=1),
                FruitSpec(6, level=3),
            ),
            drop_fruit_id=5,
            merge_events=(_merge(3, (3, 4), 6),),
        ))

        self.assertEqual(first.max_transition_chain_depth, 1)
        self.assertEqual(second.merge_records[0].chain_depth, 2)
        self.assertEqual(second.max_transition_chain_depth, 1)
        self.assertEqual(second.chain_merge_count, 0)

    def test_unrelated_drop_does_not_claim_an_old_fruit_chain(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        transition = make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(
                FruitSpec(1, level=1),
                FruitSpec(2, level=1),
                FruitSpec(4, level=2),
            ),
            next_specs=(
                FruitSpec(5, level=3),
                FruitSpec(10, level=1),
            ),
            drop_fruit_id=10,
            merge_events=(
                _merge(2, (1, 2), 3),
                _merge(3, (3, 4), 5),
            ),
        )

        result = tracker.observe_transition(transition)

        self.assertEqual(len(_events(result, 'MERGE_LINEAGE')), 2)
        self.assertFalse(_events(result, 'DIRECT_TRIGGER'))
        self.assertFalse(_events(result, 'MECHANICAL_TRIGGER'))
        self.assertFalse(_events(result, 'CHAIN_TRIGGER'))

    def test_future_merge_source_is_rejected(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        transition = make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(FruitSpec(1),),
            next_specs=(FruitSpec(2),),
            drop_fruit_id=2,
            merge_events=(_merge(2, (1, 99), 3),),
        )

        with self.assertRaises(ValueError):
            tracker.observe_transition(transition)

    def test_mechanical_trigger_requires_an_explicit_contact_path(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        contact = ContactInfluenceEdge(
            source_fruit_id=3,
            target_fruit_id=1,
            contact_count=1,
            displacement_x=0.0,
            displacement_y=2.0,
            max_impulse=10.0,
            first_contact_frame=1,
            last_contact_frame=1,
            on_merge_path=True,
        )
        transition = make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(FruitSpec(1), FruitSpec(2)),
            next_specs=(FruitSpec(3), FruitSpec(4, level=2)),
            drop_fruit_id=3,
            merge_events=(_merge(2, (1, 2), 4),),
            contact_edges=(contact,),
        )

        result = tracker.observe_transition(transition)

        mechanical = _events(result, 'MECHANICAL_TRIGGER')
        self.assertEqual(len(mechanical), 1)
        self.assertEqual(mechanical[0].contributors[0].fruit_id, 3)
        self.assertEqual(
            mechanical[0].contributors[0].role,
            'mechanical_trigger',
        )
        self.assertEqual(mechanical[0].utility, 0.0)

    def test_event_and_tracker_are_pickle_safe_mid_episode(self):
        tracker = AttributionTracker()
        tracker.begin_episode(2, 4)
        sealed = make_transition(
            worker_id=2,
            episode_id=4,
            step_index=0,
            previous_specs=(FruitSpec(1, reachable_mask=0b111),),
            next_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
            ),
            drop_fruit_id=2,
        )
        first = tracker.observe_transition(sealed)
        restored = pickle.loads(pickle.dumps(tracker))
        self.assertEqual(
            pickle.loads(pickle.dumps(first.created_events[0])),
            first.created_events[0],
        )

        next_transition = make_transition(
            worker_id=2,
            episode_id=4,
            step_index=1,
            previous_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
            ),
            next_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
                FruitSpec(3),
            ),
            drop_fruit_id=3,
        )
        original_result = tracker.observe_transition(next_transition)
        restored_result = restored.observe_transition(next_transition)
        self.assertEqual(
            original_result.created_events,
            restored_result.created_events,
        )
        self.assertEqual(
            original_result.resolved_events,
            restored_result.resolved_events,
        )
        self.assertEqual(
            original_result.pending_event_count,
            restored_result.pending_event_count,
        )


class PendingReachabilityLifecycleTest(unittest.TestCase):
    @staticmethod
    def _seal_transition(worker_id=0, episode_id=0):
        return make_transition(
            worker_id=worker_id,
            episode_id=episode_id,
            step_index=0,
            previous_specs=(FruitSpec(1, reachable_mask=0b111),),
            next_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                    burial_depth=0.8,
                ),
                FruitSpec(2),
            ),
            drop_fruit_id=2,
        )

    def test_recovery_within_three_steps_cancels_then_merge_realizes_rescue(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        seal = self._seal_transition()
        first = tracker.observe_transition(seal)
        self.assertEqual(first.pending_event_count, 1)

        recovered_specs = (
            FruitSpec(
                1,
                reachable_mask=0b1,
                partner_reachable=True,
                blocker_ids=(2,),
                burial_depth=0.7,
            ),
            FruitSpec(2),
            FruitSpec(3),
        )
        recovery = make_transition(
            worker_id=0,
            episode_id=0,
            step_index=1,
            previous_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                    burial_depth=0.8,
                ),
                FruitSpec(2),
            ),
            next_specs=recovered_specs,
            drop_fruit_id=3,
        )
        second = tracker.observe_transition(recovery)
        self.assertEqual(len(second.cancelled_events), 1)
        self.assertEqual(
            second.cancelled_events[0].resolution_reason,
            'reachability_recovered',
        )

        merge_transition = make_transition(
            worker_id=0,
            episode_id=0,
            step_index=2,
            previous_specs=recovered_specs,
            next_specs=(
                FruitSpec(2),
                FruitSpec(3),
                FruitSpec(5, level=2),
            ),
            drop_fruit_id=4,
            merge_events=(_merge(2, (1, 4), 5),),
        )
        third = tracker.observe_transition(merge_transition)
        rescued = _events(third, 'FRUIT_RESCUED')
        self.assertEqual(len(rescued), 1)
        self.assertEqual(rescued[0].utility, 0.0)
        self.assertEqual(
            rescued[0].evidence.primary_event_key,
            first.created_events[0].event_id,
        )

    def test_twelve_unreachable_boundaries_confirm_once_without_time_scaling(self):
        tracker = AttributionTracker(AttributionTrackerConfig(
            transient_stable_steps=3,
            burial_confirm_steps=12,
        ))
        tracker.begin_episode(0, 0)
        first = tracker.observe_transition(self._seal_transition())
        primary = first.created_events[0]
        current_specs = [
            FruitSpec(
                1,
                reachable_mask=0,
                partner_reachable=False,
                blocker_ids=(2,),
                burial_depth=0.8,
            ),
            FruitSpec(2),
        ]

        result = None
        for step_index in range(1, 12):
            new_id = step_index + 2
            next_specs = tuple(current_specs + [FruitSpec(new_id)])
            result = tracker.observe_transition(make_transition(
                worker_id=0,
                episode_id=0,
                step_index=step_index,
                previous_specs=tuple(current_specs),
                next_specs=next_specs,
                drop_fruit_id=new_id,
            ))
            current_specs.append(FruitSpec(new_id))
            if step_index < 11:
                self.assertEqual(result.pending_event_count, 1)
                self.assertFalse(result.confirmed_events)

        confirmed = tuple(
            event
            for event in result.resolved_events
            if event.event_id == primary.event_id
        )
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(
            confirmed[0].evidence.stable_unreachable_steps,
            12,
        )
        self.assertEqual(confirmed[0].utility, primary.utility)
        self.assertEqual(result.pending_event_count, 0)
        self.assertEqual(
            len(_events(result, 'DEAD_LOW_FRUIT_CONFIRMED')),
            1,
        )

        new_id = 14
        later = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=12,
            previous_specs=tuple(current_specs),
            next_specs=tuple(current_specs + [FruitSpec(new_id)]),
            drop_fruit_id=new_id,
        ))
        self.assertFalse(_events(later, 'DEAD_LOW_FRUIT_CONFIRMED'))
        self.assertFalse(later.resolved_events)

    def test_terminal_confirms_original_incident_but_truncation_cancels_it(self):
        terminal_tracker = AttributionTracker()
        terminal_tracker.begin_episode(0, 0)
        first = terminal_tracker.observe_transition(
            self._seal_transition()
        )
        terminal = terminal_tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=1,
            previous_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
            ),
            next_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
                FruitSpec(3),
            ),
            drop_fruit_id=3,
            terminated=True,
        ))
        confirmed = terminal.resolved_events[0]
        self.assertEqual(confirmed.event_id, first.created_events[0].event_id)
        self.assertEqual(
            confirmed.resolution_reason,
            'terminal_still_buried',
        )
        self.assertEqual(
            confirmed.contributors[0].transition_key.step_index,
            0,
        )

        truncated_tracker = AttributionTracker()
        truncated_tracker.begin_episode(0, 1)
        truncated_tracker.observe_transition(
            self._seal_transition(episode_id=1)
        )
        truncated = truncated_tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=1,
            step_index=1,
            previous_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
            ),
            next_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
                FruitSpec(3),
            ),
            drop_fruit_id=3,
            truncated=True,
            valid_next=False,
        ))
        self.assertEqual(truncated.interrupted_pending_count, 1)
        self.assertEqual(len(truncated.cancelled_events), 1)
        self.assertEqual(
            truncated.cancelled_events[0].resolution_reason,
            'truncated',
        )
        self.assertFalse(truncated.confirmed_events)

    def test_pending_merge_resolution_precedes_terminal_or_truncation(self):
        def run_boundary(*, episode_id, terminated, truncated):
            tracker = AttributionTracker()
            tracker.begin_episode(0, episode_id)
            first = tracker.observe_transition(make_transition(
                worker_id=0,
                episode_id=episode_id,
                step_index=0,
                previous_specs=(
                    FruitSpec(1, reachable_mask=0b111),
                    FruitSpec(5, reachable_mask=0b111),
                ),
                next_specs=(
                    FruitSpec(
                        1,
                        reachable_mask=0,
                        partner_reachable=False,
                        blocker_ids=(2,),
                    ),
                    FruitSpec(
                        5,
                        reachable_mask=0,
                        partner_reachable=False,
                        blocker_ids=(2,),
                    ),
                    FruitSpec(2),
                ),
                drop_fruit_id=2,
            ))
            boundary = tracker.observe_transition(make_transition(
                worker_id=0,
                episode_id=episode_id,
                step_index=1,
                previous_specs=(
                    FruitSpec(
                        1,
                        reachable_mask=0,
                        partner_reachable=False,
                        blocker_ids=(2,),
                    ),
                    FruitSpec(
                        5,
                        reachable_mask=0,
                        partner_reachable=False,
                        blocker_ids=(2,),
                    ),
                    FruitSpec(2),
                ),
                next_specs=(
                    FruitSpec(4, level=2),
                    FruitSpec(
                        5,
                        reachable_mask=0,
                        partner_reachable=False,
                        blocker_ids=(2,),
                    ),
                    FruitSpec(2),
                ),
                drop_fruit_id=3,
                merge_events=(_merge(2, (1, 3), 4),),
                terminated=terminated,
                truncated=truncated,
                valid_next=not truncated,
            ))
            return first, boundary

        terminal_first, terminal = run_boundary(
            episode_id=0,
            terminated=True,
            truncated=False,
        )
        terminal_by_id = {
            event.event_id: event
            for event in terminal.resolved_events
        }
        terminal_pending = {
            event.target_fruit_ids[0]: event
            for event in terminal_first.created_events
        }
        self.assertEqual(
            terminal_by_id[
                terminal_pending[1].event_id
            ].resolution_reason,
            'entered_merge_lineage',
        )
        self.assertEqual(
            terminal_by_id[
                terminal_pending[5].event_id
            ].resolution_reason,
            'terminal_still_buried',
        )

        truncated_first, truncated = run_boundary(
            episode_id=1,
            terminated=False,
            truncated=True,
        )
        truncated_by_id = {
            event.event_id: event
            for event in truncated.resolved_events
        }
        truncated_pending = {
            event.target_fruit_ids[0]: event
            for event in truncated_first.created_events
        }
        self.assertEqual(
            truncated_by_id[
                truncated_pending[1].event_id
            ].resolution_reason,
            'entered_merge_lineage',
        )
        self.assertEqual(
            truncated_by_id[
                truncated_pending[5].event_id
            ].resolution_reason,
            'truncated',
        )
        self.assertEqual(truncated.interrupted_pending_count, 1)
        self.assertEqual(len(_events(truncated, 'MERGE_LINEAGE')), 1)

    def test_terminal_post_action_analysis_can_cancel_a_recovered_incident(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        tracker.observe_transition(self._seal_transition())
        terminal = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=1,
            previous_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
            ),
            next_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0b1,
                    partner_reachable=True,
                ),
                FruitSpec(2),
                FruitSpec(3),
            ),
            drop_fruit_id=3,
            terminated=True,
        ))

        self.assertFalse(terminal.confirmed_events)
        self.assertEqual(len(terminal.cancelled_events), 1)
        self.assertEqual(
            terminal.cancelled_events[0].resolution_reason,
            'reachability_recovered',
        )
        self.assertIn(
            'terminal_post_action_geometry_used',
            terminal.diagnostic_codes,
        )

    def test_invalid_terminal_geometry_cannot_resolve_pending_evidence(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        tracker.observe_transition(self._seal_transition())
        terminal = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=1,
            previous_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
            ),
            next_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0b1,
                    partner_reachable=True,
                ),
                FruitSpec(2),
                FruitSpec(3),
            ),
            drop_fruit_id=3,
            terminated=True,
            valid_next=False,
        ))

        self.assertFalse(terminal.confirmed_events)
        self.assertEqual(len(terminal.cancelled_events), 1)
        self.assertEqual(
            terminal.cancelled_events[0].resolution_reason,
            'terminal_geometry_untrusted',
        )
        self.assertIn(
            'terminal_geometry_skipped',
            terminal.diagnostic_codes,
        )

    def test_terminal_new_support_cut_is_attributed_from_final_geometry(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        terminal = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(
                FruitSpec(1, reachable_mask=0b111),
            ),
            next_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                    blocker_ids=(2,),
                ),
                FruitSpec(2),
            ),
            drop_fruit_id=2,
            terminated=True,
            next_support_edges=(
                SupportEdge(
                    supporter_fruit_id=2,
                    supported_fruit_id=1,
                    relation='caps',
                ),
            ),
        ))

        confirmed_seals = tuple(
            event
            for event in terminal.resolved_events
            if event.event_type == 'REACHABILITY_SEALED'
        )
        self.assertEqual(len(confirmed_seals), 1)
        support_events = _events(
            terminal,
            'TERMINAL_SUPPORT_CREATED',
        )
        self.assertEqual(len(support_events), 1)
        self.assertEqual(support_events[0].status, 'confirmed')
        self.assertEqual(
            support_events[0].resolution_reason,
            'actual_terminal_support_cut',
        )
        self.assertEqual(support_events[0].utility, 0.0)
        self.assertEqual(
            support_events[0].budget_key,
            confirmed_seals[0].budget_key,
        )
        duplicated_negative_value = replace(
            support_events[0],
            utility=confirmed_seals[0].utility,
        )
        with self.assertRaises(ValueError):
            replace(
                terminal,
                created_events=tuple(
                    duplicated_negative_value
                    if event.event_id == support_events[0].event_id
                    else event
                    for event in terminal.created_events
                ),
            )

    def test_progressive_lane_loss_shares_responsibility_across_actions(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        masks = (0b111, 0b110, 0b100, 0)
        specs = [FruitSpec(1, reachable_mask=masks[0])]
        result = None
        for step_index in range(3):
            drop_id = step_index + 2
            previous_specs = tuple(specs)
            next_target = FruitSpec(
                1,
                reachable_mask=masks[step_index + 1],
                partner_reachable=masks[step_index + 1] != 0,
                blocker_ids=tuple(range(2, drop_id + 1)),
                burial_depth=0.5 + 0.05 * (step_index + 1),
            )
            next_specs = (
                next_target,
                *tuple(
                    spec
                    for spec in specs
                    if spec.fruit_id != 1
                ),
                FruitSpec(drop_id),
            )
            result = tracker.observe_transition(make_transition(
                worker_id=0,
                episode_id=0,
                step_index=step_index,
                previous_specs=previous_specs,
                next_specs=next_specs,
                drop_fruit_id=drop_id,
            ))
            specs = list(next_specs)

        pending = _events(result, 'REACHABILITY_SEALED')[0]
        self.assertEqual(len(pending.contributors), 3)
        self.assertEqual(
            {
                contributor.transition_key.step_index
                for contributor in pending.contributors
            },
            {0, 1, 2},
        )
        self.assertAlmostEqual(
            sum(
                contributor.contribution_weight
                for contributor in pending.contributors
            ),
            1.0,
        )
        self.assertTrue(all(
            contributor.contribution_weight > 0.0
            for contributor in pending.contributors
        ))

    def test_partial_lane_recovery_only_removes_the_restored_bits(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        masks = (0b111, 0b100, 0b101, 0)
        specs = [FruitSpec(1, reachable_mask=masks[0])]
        result = None
        for step_index in range(3):
            drop_id = step_index + 2
            next_mask = masks[step_index + 1]
            next_specs = (
                FruitSpec(
                    1,
                    reachable_mask=next_mask,
                    partner_reachable=next_mask != 0,
                    blocker_ids=tuple(range(2, drop_id + 1)),
                ),
                *tuple(
                    spec
                    for spec in specs
                    if spec.fruit_id != 1
                ),
                FruitSpec(drop_id),
            )
            result = tracker.observe_transition(make_transition(
                worker_id=0,
                episode_id=0,
                step_index=step_index,
                previous_specs=tuple(specs),
                next_specs=next_specs,
                drop_fruit_id=drop_id,
            ))
            specs = list(next_specs)

        pending = _events(result, 'REACHABILITY_SEALED')[0]
        self.assertEqual(
            {
                contributor.transition_key.step_index
                for contributor in pending.contributors
            },
            {0, 2},
        )
        self.assertEqual(
            pending.evidence.lost_action_mask,
            0b111,
        )

    def test_born_buried_uses_drop_action_as_primary_incident(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        result = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(),
            next_specs=(
                FruitSpec(
                    1,
                    reachable_mask=0,
                    partner_reachable=False,
                ),
            ),
            drop_fruit_id=1,
        ))

        event = _events(result, 'BORN_BURIED')[0]
        self.assertEqual(event.status, 'pending')
        self.assertEqual(event.contributors[0].fruit_id, 1)
        self.assertEqual(event.budget_key, event.event_id)
        self.assertEqual(event.confidence_tier, 'B')


class RealizedStructureAndCollectorTest(unittest.TestCase):
    def test_adjacency_only_becomes_positive_after_real_merge(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        motif = merge_pair_motif(1, 2)
        setup = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(FruitSpec(1),),
            next_specs=(FruitSpec(1), FruitSpec(2)),
            drop_fruit_id=2,
            next_motifs=(motif,),
        ))
        self.assertFalse(_events(setup, 'REALIZED_ADJACENCY'))

        realized = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=1,
            previous_specs=(FruitSpec(1), FruitSpec(2)),
            next_specs=(FruitSpec(3), FruitSpec(4, level=2)),
            drop_fruit_id=3,
            merge_events=(_merge(2, (1, 2), 4),),
            previous_motifs=(motif,),
        ))
        adjacency = _events(realized, 'REALIZED_ADJACENCY')
        self.assertEqual(len(adjacency), 1)
        self.assertEqual(adjacency[0].detected_step, 0)
        self.assertEqual(adjacency[0].resolved_step, 1)
        self.assertEqual(adjacency[0].delay, 1)
        self.assertEqual(adjacency[0].utility, 0.0)

    def test_broken_motif_origin_cannot_claim_a_later_merge(self):
        tracker = AttributionTracker()
        tracker.begin_episode(0, 0)
        motif = merge_pair_motif(1, 2)
        tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=0,
            previous_specs=(FruitSpec(1),),
            next_specs=(FruitSpec(1), FruitSpec(2)),
            drop_fruit_id=2,
            next_motifs=(motif,),
        ))
        disappeared = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=1,
            previous_specs=(FruitSpec(1), FruitSpec(2)),
            next_specs=(
                FruitSpec(1),
                FruitSpec(2),
                FruitSpec(3),
            ),
            drop_fruit_id=3,
            previous_motifs=(motif,),
        ))
        self.assertFalse(_events(disappeared, 'CHAIN_MOTIF_BROKEN'))

        tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=2,
            previous_specs=(
                FruitSpec(1),
                FruitSpec(2),
                FruitSpec(3),
            ),
            next_specs=(
                FruitSpec(1),
                FruitSpec(2),
                FruitSpec(3),
                FruitSpec(4),
            ),
            drop_fruit_id=4,
            next_motifs=(motif,),
        ))
        realized = tracker.observe_transition(make_transition(
            worker_id=0,
            episode_id=0,
            step_index=3,
            previous_specs=(
                FruitSpec(1),
                FruitSpec(2),
                FruitSpec(3),
                FruitSpec(4),
            ),
            next_specs=(
                FruitSpec(3),
                FruitSpec(4),
                FruitSpec(5),
                FruitSpec(6, level=2),
            ),
            drop_fruit_id=5,
            merge_events=(_merge(2, (1, 2), 6),),
            previous_motifs=(motif,),
        ))

        self.assertFalse(_events(realized, 'REALIZED_ADJACENCY'))

    def test_event_identity_isolated_by_episode_and_worker(self):
        first_tracker = AttributionTracker()
        first_tracker.begin_episode(0, 0)
        first = first_tracker.observe_transition(
            PendingReachabilityLifecycleTest._seal_transition()
        ).created_events[0]
        first_tracker.finalize_episode('manual_reset')
        first_tracker.begin_episode(0, 1)
        second = first_tracker.observe_transition(
            PendingReachabilityLifecycleTest._seal_transition(
                episode_id=1,
            )
        ).created_events[0]

        other_tracker = AttributionTracker()
        other_tracker.begin_episode(1, 0)
        other = other_tracker.observe_transition(
            PendingReachabilityLifecycleTest._seal_transition(
                worker_id=1,
            )
        ).created_events[0]

        self.assertEqual(first.target_fruit_ids, second.target_fruit_ids)
        self.assertEqual(first.target_fruit_ids, other.target_fruit_ids)
        self.assertNotEqual(first.event_id, second.event_id)
        self.assertNotEqual(first.event_id, other.event_id)
        self.assertEqual(first.episode_key, (0, 0))
        self.assertEqual(second.episode_key, (0, 1))
        self.assertEqual(other.episode_key, (1, 0))

    def test_collector_calls_tracker_once_per_transition_without_changing_replay(self):
        game = HeadlessGame(gravity=(0, 0), seed=0)
        game.stable_velocity_epsilon = 1000.0
        env = DaxiguaEnv(
            config=DaxiguaEnvConfig(
                action_count=ANALYSIS_ACTION_COUNT,
                max_physics_frames=1,
                stable_frames=1,
            ),
            game=game,
        )
        replay = ReplayBuffer(capacity=4, seed=0)
        collector = RolloutCollector(
            env=env,
            graph_builder=GraphBuilder(),
            replay_buffer=replay,
            worker_id=4,
            seed=0,
        )

        stats = collector.collect_steps(2, epsilon=1.0)

        self.assertEqual(stats.attribution_tracker_calls, 2)
        self.assertGreaterEqual(stats.attribution_tracker_seconds, 0.0)
        self.assertEqual(len(replay), 2)
        self.assertFalse(hasattr(replay.to_tuple()[0], 'state_analysis'))
        self.assertEqual(
            stats.attribution_pending_event_count,
            collector.attribution_tracker.pending_event_count,
        )
        first_close = collector.close()
        self.assertEqual(collector.close(), first_close)
        with self.assertRaises(RuntimeError):
            collector.collect_steps(1, epsilon=1.0)


if __name__ == '__main__':
    unittest.main()
