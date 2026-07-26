"""Deterministic builders shared by AttributionTracker tests."""

from __future__ import annotations

from dataclasses import dataclass, replace

from daxigua.core.rules import (
    dropped_fruit_physics_radius,
    fruit_radius,
    merge_score,
)
from daxigua.core.state import (
    ActionCandidate,
    BoardGeometry,
    DropResult,
    FruitState,
    GameState,
    PhysicsResult,
)
from daxigua_rl.attribution import (
    ANALYSIS_ACTION_COUNT,
    FULL_ACTION_MASK,
    ChainMotif,
    ContactInfluenceEdge,
    FruitAnalysis,
    StateAnalyzer,
    StateAnalyzerConfig,
    drop_x_positions_for_level,
)
from daxigua_rl.attribution.tracker import TrackerTransitionInput
from daxigua_rl.training import TransitionKey


GEOMETRY = BoardGeometry(
    width=400,
    height=800,
    spawn_y=180,
    wall_width=20,
    floor_y=780,
)
QUEUE = (1, 2, 3, 4)
_ANALYZER = StateAnalyzer(StateAnalyzerConfig(grid_cell_size=32.0))


@dataclass(frozen=True, slots=True)
class FruitSpec:
    fruit_id: int
    level: int = 1
    reachable_mask: int = FULL_ACTION_MASK
    partner_reachable: bool = True
    burial_depth: float = 0.5
    blocker_ids: tuple[int, ...] = ()
    inversion_blocker_ids: tuple[int, ...] = ()
    y: float | None = None


def make_fruit_state(spec):
    physics_radius = float(dropped_fruit_physics_radius(spec.level))
    y = (
        float(spec.y)
        if spec.y is not None
        else GEOMETRY.spawn_y
        + spec.burial_depth * (GEOMETRY.floor_y - GEOMETRY.spawn_y)
    )
    x = 60.0 + 25.0 * (spec.fruit_id % 11)
    return FruitState(
        fruit_id=spec.fruit_id,
        level=spec.level,
        radius=float(fruit_radius(spec.level)),
        physics_radius=physics_radius,
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        angle=0.0,
        angular_velocity=0.0,
        age_frames=10,
        stable=True,
        distance_to_left_wall=max(0.0, x - physics_radius),
        distance_to_right_wall=max(
            0.0,
            GEOMETRY.width - x - physics_radius,
        ),
        distance_to_floor=max(
            0.0,
            GEOMETRY.floor_y - y - physics_radius,
        ),
        distance_to_danger_line=max(0.0, y - GEOMETRY.spawn_y),
    )


def make_game_state(
        specs,
        *,
        step_index,
        done=False,
        queue=QUEUE):
    fruits = tuple(
        make_fruit_state(spec)
        for spec in sorted(specs, key=lambda item: item.fruit_id)
    )
    max_level = max((fruit.level for fruit in fruits), default=0)
    return GameState(
        board_fruits=fruits,
        fruit_queue=tuple(queue),
        score=0,
        last_score=0,
        step_count=step_index,
        physics_frame=step_index * 10,
        done=bool(done),
        geometry=GEOMETRY,
        max_height=0.0,
        fruit_count=len(fruits),
        max_level=max_level,
        empty_space_ratio=0.8,
    )


def make_actions(level=1):
    positions = drop_x_positions_for_level(
        GEOMETRY,
        level,
        action_count=ANALYSIS_ACTION_COUNT,
    )
    left = positions[0]
    right = positions[-1]
    return tuple(
        ActionCandidate(
            action_index=offset,
            drop_x=drop_x,
            normalized_drop_x=(drop_x - left) / (right - left),
            current_level=level,
            current_radius=float(fruit_radius(level)),
            current_physics_radius=float(
                dropped_fruit_physics_radius(level)
            ),
        )
        for offset, drop_x in enumerate(positions)
    )


def make_fruit_analysis(spec):
    blockers_by_action = tuple(
        ()
        if spec.reachable_mask & (1 << action_offset)
        else tuple(spec.blocker_ids)
        for action_offset in range(ANALYSIS_ACTION_COUNT)
    )
    return FruitAnalysis(
        fruit_id=spec.fruit_id,
        level=spec.level,
        physics_radius=float(dropped_fruit_physics_radius(spec.level)),
        probe_physics_radius=float(
            dropped_fruit_physics_radius(spec.level)
        ),
        reachable_action_mask=spec.reachable_mask,
        reachable_action_count=spec.reachable_mask.bit_count(),
        top_visible_ratio=(
            spec.reachable_mask.bit_count() / ANALYSIS_ACTION_COUNT
        ),
        top_blocker_ids_by_action=blockers_by_action,
        partner_ids=(),
        partner_reachable=spec.partner_reachable,
        support_parent_ids=(),
        supported_child_ids=(),
        burial_depth=spec.burial_depth,
        inversion_count=len(spec.inversion_blocker_ids),
        connected_region_id=None,
        reachable_partner_ids=(),
        critical_blocker_ids=(
            tuple(spec.blocker_ids)
            if spec.reachable_mask == 0
            else ()
        ),
        inversion_blocker_ids=tuple(spec.inversion_blocker_ids),
    )


def make_analysis(
        specs,
        *,
        key,
        incoming_key=None,
        valid=True,
        motifs=(),
        support_edges=(),
        contact_edges=()):
    empty_state = make_game_state(
        (),
        step_index=key.step_index,
    )
    base = _ANALYZER.analyze(
        empty_state,
        make_actions(),
        key,
        stable_boundary=bool(valid),
        incoming_transition_key=incoming_key,
        contact_influence_edges=tuple(contact_edges),
    )
    diagnostics = base.diagnostics
    if not valid:
        diagnostics = replace(
            diagnostics,
            stable_boundary=False,
            valid_for_attribution=False,
            degraded=True,
            warning_codes=('unstable_boundary',),
        )
    return replace(
        base,
        fruit_analyses=tuple(
            make_fruit_analysis(spec)
            for spec in sorted(specs, key=lambda item: item.fruit_id)
        ),
        chain_motifs=tuple(motifs),
        chain_readiness=max(
            (motif.readiness for motif in motifs),
            default=0.0,
        ),
        support_edges=tuple(support_edges),
        contact_influence_edges=tuple(contact_edges),
        diagnostics=diagnostics,
    )


def merge_pair_motif(left_id, right_id, *, level=1, readiness=0.8):
    return ChainMotif(
        motif_type='merge_pair',
        fruit_ids=tuple(sorted((left_id, right_id))),
        levels=(level, level),
        depth=1,
        trigger_action_mask=FULL_ACTION_MASK,
        compatible_queue_indices=(0,),
        readiness=readiness,
    )


def make_transition(
        *,
        worker_id,
        episode_id,
        step_index,
        previous_specs,
        next_specs,
        drop_fruit_id,
        action_offset=0,
        action_index=0,
        merge_events=(),
        terminated=False,
        truncated=False,
        valid_next=True,
        previous_motifs=(),
        next_motifs=(),
        previous_support_edges=(),
        next_support_edges=(),
        contact_edges=()):
    key = TransitionKey(worker_id, episode_id, step_index)
    next_key = TransitionKey(worker_id, episode_id, step_index + 1)
    previous_state = make_game_state(
        previous_specs,
        step_index=step_index,
    )
    next_state = make_game_state(
        next_specs,
        step_index=step_index + 1,
        done=terminated,
    )
    previous_analysis = make_analysis(
        previous_specs,
        key=key,
        motifs=previous_motifs,
        support_edges=previous_support_edges,
    )
    next_analysis = make_analysis(
        next_specs,
        key=next_key,
        incoming_key=key,
        valid=valid_next,
        motifs=next_motifs,
        support_edges=next_support_edges,
        contact_edges=contact_edges,
    )
    score_delta = sum(
        merge_score(event.new_level)
        for event in merge_events
    )
    physics_result = PhysicsResult(
        frames_simulated=1,
        stable=not truncated,
        done=terminated,
        truncated=truncated,
        score_delta=score_delta,
        merge_events=tuple(merge_events),
    )
    drop_level = QUEUE[0]
    return TrackerTransitionInput(
        transition_key=key,
        action_offset=action_offset,
        action_index=action_index,
        previous_state=previous_state,
        next_state=next_state,
        previous_analysis=previous_analysis,
        next_analysis=next_analysis,
        drop_result=DropResult(
            dropped_level=drop_level,
            drop_x=(
                previous_analysis.action_drop_x_by_offset[
                    action_offset
                ]
            ),
            fruit_id=drop_fruit_id,
            queue_before=QUEUE,
            queue_after=QUEUE,
        ),
        physics_result=physics_result,
    )


__all__ = [
    'ContactInfluenceEdge',
    'FruitSpec',
    'GEOMETRY',
    'QUEUE',
    'make_analysis',
    'make_game_state',
    'make_transition',
    'merge_pair_motif',
]
