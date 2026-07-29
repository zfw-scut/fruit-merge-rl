"""局部 Shapley 连续真实轨迹、物理门禁与样本回灌测试。"""

from __future__ import annotations

import multiprocessing
import pickle
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from daxigua.core.engine import HeadlessGame
from daxigua_rl.attribution.causal_replay import (
    CausalTransitionContext,
)
from daxigua_rl.attribution.counterfactual import (
    LocalShapleyConfig,
)
from daxigua_rl.attribution.counterfactual_proposal import (
    CounterfactualHistoryEntry,
    CounterfactualProposal,
    stable_counterfactual_proposal_id,
)
from daxigua_rl.attribution.local_shapley_runner import (
    create_local_shapley_task,
    local_shapley_result_to_causal_samples,
    run_local_shapley_task,
)
from daxigua_rl.attribution.schema import (
    ANALYSIS_ACTION_COUNT,
    AttributionEvent,
    AttributionEventKey,
    AttributionEvidence,
    Contributor,
)
from daxigua_rl.attribution.state_analyzer import StateAnalyzer
from daxigua_rl.graph import GraphBuilder
from daxigua_rl.graph.tensor import graph_to_tensor
from daxigua_rl.reward import merge_utility
from daxigua_rl.training.identity import TransitionKey

from test_counterfactual_runner import _zero_target_payload


def _history_entry(
        game,
        payload,
        *,
        actual_action_offset,
        comparison_action_offset):
    snapshot = game.capture_snapshot()
    state = game.get_state()
    candidates = tuple(
        game.get_action_candidates(ANALYSIS_ACTION_COUNT)
    )
    key = TransitionKey(0, 0, state.step_count)
    analysis = StateAnalyzer(
        config=payload.state_analyzer_config
    ).analyze(
        state,
        candidates,
        key,
        stable_boundary=True,
    )
    graph = graph_to_tensor(
        GraphBuilder().build(state, candidates)
    )
    context = CausalTransitionContext(
        graph=graph,
        state_analysis=analysis,
        actual_action_offset=actual_action_offset,
        actual_action_index=int(
            graph.action_indices[actual_action_offset].item()
        ),
        policy_version='local-shapley-online-v1',
    )
    factual_outcome = game.execute_action(
        candidates[actual_action_offset].drop_x,
        max_frames=payload.max_physics_frames,
        stable_frames=payload.stable_frames,
    )
    return CounterfactualHistoryEntry(
        transition_key=key,
        context=context,
        snapshot=snapshot,
        factual_outcome=factual_outcome,
        alternative_action_offsets=(comparison_action_offset,),
    )


def _build_two_step_proposal(*, event_index=1):
    payload = _zero_target_payload(max_physics_frames=360)
    game = HeadlessGame(seed=123)
    game.reset(seed=123, fruit_queue=(1, 1, 2, 3))
    entries = (
        _history_entry(
            game,
            payload,
            actual_action_offset=1,
            comparison_action_offset=13,
        ),
        _history_entry(
            game,
            payload,
            actual_action_offset=1,
            comparison_action_offset=7,
        ),
    )
    event_id = AttributionEventKey(0, 0, event_index)
    contributors = tuple(
        Contributor(
            transition_key=entry.transition_key,
            action_offset=entry.actual_action_offset,
            action_index=entry.context.actual_action_index,
            fruit_id=entry.factual_outcome.drop_result.fruit_id,
            evidence_type='local_shapley_test',
            raw_evidence_weight=weight,
            contribution_weight=weight,
            role='material',
        )
        for entry, weight in zip(entries, (0.55, 0.45))
    )
    event = AttributionEvent(
        event_id=event_id,
        episode_key=(0, 0),
        attribution_version='local-shapley-attribution-v1',
        tracker_config_fingerprint='local-shapley-tracker-v1',
        detected_step=0,
        resolved_step=2,
        event_type='CHAIN_TRIGGER',
        status='confirmed',
        sign=1,
        target_fruit_ids=tuple(
            entry.factual_outcome.drop_result.fruit_id
            for entry in entries
        ),
        contributors=contributors,
        utility=merge_utility(7),
        link_confidence=0.95,
        placement_confidence=0.72,
        evidence=AttributionEvidence(
            reason_codes=('multi_stage_chain',),
        ),
        budget_key=event_id,
        resolution_reason='local_shapley_test',
    )
    representative = contributors[-1]
    representative_entry = entries[-1]
    proposal_id = stable_counterfactual_proposal_id(
        budget_key=event_id,
        representative_event=event,
        contributor=representative,
        context=representative_entry.context,
        snapshot=representative_entry.snapshot,
        factual_outcome=representative_entry.factual_outcome,
        alternative_action_offsets=(
            representative_entry.alternative_action_offsets
        ),
        trigger_reasons=('multi_stage_chain',),
        coalition_trace_entries=entries,
        coalition_candidate_keys=tuple(
            entry.transition_key
            for entry in entries
        ),
    )
    proposal = CounterfactualProposal(
        proposal_id=proposal_id,
        representative_event=event,
        budget_key=event_id,
        contributor=representative,
        context=representative_entry.context,
        snapshot=representative_entry.snapshot,
        factual_outcome=representative_entry.factual_outcome,
        actual_action_offset=(
            representative_entry.actual_action_offset
        ),
        alternative_action_offsets=(
            representative_entry.alternative_action_offsets
        ),
        trigger_reasons=('multi_stage_chain',),
        coalition_trace_entries=entries,
        coalition_candidate_keys=tuple(
            entry.transition_key
            for entry in entries
        ),
        utility=event.utility,
        confidence=min(
            event.link_confidence,
            event.placement_confidence,
        ),
        delay=event.delay,
    )
    return payload, proposal


def _tamper_first_trace_outcome(proposal):
    first, second = proposal.coalition_trace_entries
    tampered_outcome = replace(
        first.factual_outcome,
        fail_count=first.factual_outcome.fail_count + 1,
    )
    tampered_first = replace(
        first,
        factual_outcome=tampered_outcome,
    )
    trace = tampered_first, second
    proposal_id = stable_counterfactual_proposal_id(
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
        coalition_trace_entries=trace,
        coalition_candidate_keys=(
            proposal.coalition_candidate_keys
        ),
    )
    return replace(
        proposal,
        proposal_id=proposal_id,
        coalition_trace_entries=trace,
    )


def _replay_report(
        *,
        reproduction_status,
        mismatch_codes,
        maxima=(0.0, 0.0, 0.0, 0.0, 0.0)):
    return SimpleNamespace(
        matches=False,
        reproduction_status=reproduction_status,
        mismatch_codes=tuple(mismatch_codes),
        max_merge_event_position_error=maxima[0],
        max_fruit_position_error=maxima[1],
        max_linear_velocity_error=maxima[2],
        max_orientation_error=maxima[3],
        max_angular_velocity_error=maxima[4],
    )


class LocalShapleyPhysicalRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload, cls.proposal = _build_two_step_proposal()
        cls.config = LocalShapleyConfig(
            event_ratio_max=1.0,
            paired_permutations=4,
        )
        cls.task = create_local_shapley_task(
            cls.proposal,
            cls.payload,
            config=cls.config,
        )

    def test_task_is_frozen_stable_and_pickle_safe(self):
        restored = pickle.loads(pickle.dumps(self.task))

        self.assertEqual(restored.task_id, self.task.task_id)
        self.assertEqual(
            restored.target_policy.fingerprint,
            self.task.target_policy.fingerprint,
        )
        self.assertEqual(
            restored.candidate_keys,
            self.task.candidate_keys,
        )
        self.assertEqual(self.task.horizon, 2)
        self.assertEqual(self.task.estimated_tokens, 8)
        self.assertEqual(
            tuple(
                offset
                for _key, offset
                in self.task.comparison_action_offsets
            ),
            (13, 7),
        )

    def test_grand_trace_four_subsets_efficiency_and_samples(self):
        result = run_local_shapley_task(self.task)

        self.assertTrue(result.grand_reproduced)
        self.assertTrue(result.label_ready)
        self.assertEqual(result.status, 'completed')
        self.assertEqual(result.evaluated_subset_count, 4)
        self.assertEqual(len(result.subset_results), 4)
        self.assertEqual(
            {
                subset.member_keys
                for subset in result.subset_results
            },
            {
                (),
                (self.task.candidate_keys[0],),
                (self.task.candidate_keys[1],),
                self.task.candidate_keys,
            },
        )
        self.assertLessEqual(
            abs(result.efficiency_residual),
            result.efficiency_tolerance,
        )
        self.assertEqual(result.simulated_steps, 8)

        samples = local_shapley_result_to_causal_samples(
            self.task,
            result,
        )
        self.assertEqual(len(samples), 2)
        self.assertEqual(
            {sample.transition_key for sample in samples},
            set(self.task.candidate_keys),
        )
        contribution_map = dict(result.contributions)
        for sample in samples:
            self.assertEqual(sample.cause_type, 'LOCAL_SHAPLEY')
            self.assertEqual(sample.supervision_kind, 'shapley')
            self.assertEqual(sample.stratum, 'counterfactual')
            self.assertEqual(
                sample.target_delta,
                contribution_map[sample.transition_key],
            )
            self.assertEqual(
                sample.tracker_config_fingerprint,
                'local-shapley-tracker-v1',
            )

    def test_any_tampered_factual_step_rejects_all_labels(self):
        tampered = _tamper_first_trace_outcome(self.proposal)
        task = create_local_shapley_task(
            tampered,
            self.payload,
            config=self.config,
        )

        result = run_local_shapley_task(task)

        self.assertEqual(result.status, 'failed')
        self.assertFalse(result.grand_reproduced)
        self.assertFalse(result.label_ready)
        self.assertEqual(
            result.failure_reason,
            'grand_reproduction_mismatch',
        )
        self.assertEqual(
            result.reproduction_outcome,
            'semantic_divergence_drop',
        )
        self.assertEqual(
            local_shapley_result_to_causal_samples(task, result),
            (),
        )

    def test_numeric_jitter_has_distinct_outcome_and_rejects_all_labels(self):
        report = _replay_report(
            reproduction_status='numeric_jitter_drop',
            mismatch_codes=(
                'merge_event_position',
                'fruit_position',
                'fruit_velocity',
                'fruit_angle',
            ),
            maxima=(0.11, 0.22, 0.33, 0.44, 0.55),
        )
        with patch.object(
                HeadlessGame,
                'compare_action_outcomes',
                return_value=report):
            result = run_local_shapley_task(self.task)

        self.assertEqual(result.status, 'failed')
        self.assertFalse(result.grand_reproduced)
        self.assertFalse(result.label_ready)
        self.assertEqual(
            result.failure_reason,
            'grand_reproduction_numeric_jitter',
        )
        self.assertEqual(
            result.reproduction_outcome,
            'numeric_jitter_drop',
        )
        self.assertEqual(
            result.diagnostic_codes,
            (
                'grand_mismatch_merge_event_position',
                'grand_mismatch_fruit_position',
                'grand_mismatch_fruit_velocity',
                'grand_mismatch_fruit_angle',
            ),
        )
        self.assertEqual(
            (
                result.replay_max_merge_event_position_error,
                result.replay_max_fruit_position_error,
                result.replay_max_linear_velocity_error,
                result.replay_max_orientation_error,
                result.replay_max_angular_velocity_error,
            ),
            (0.11, 0.22, 0.33, 0.44, 0.55),
        )
        self.assertEqual(
            local_shapley_result_to_causal_samples(
                self.task,
                result,
            ),
            (),
        )

    def test_spawn_worker_runs_real_physics_task(self):
        context = multiprocessing.get_context('spawn')
        started = time.perf_counter()
        with ProcessPoolExecutor(
                max_workers=1,
                mp_context=context) as executor:
            result = executor.submit(
                run_local_shapley_task,
                self.task,
            ).result(timeout=40.0)
        elapsed = time.perf_counter() - started

        self.assertTrue(result.label_ready)
        self.assertEqual(result.evaluated_subset_count, 4)
        self.assertLess(elapsed, 40.0)


if __name__ == '__main__':
    unittest.main()
