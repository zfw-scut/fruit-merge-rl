"""Worker-local causal attribution tracker.

The tracker consumes one already-executed transition at a time.  It never
advances physics, runs ``StateAnalyzer`` or writes the main replay buffer.
Its durable state is deliberately limited to lineage, compressed structural
origins and unresolved incidents so a long rollout does not retain complete
``GameState``/``StateAnalysis`` snapshots.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from numbers import Real
from operator import index

from daxigua.core.rules import (
    MAX_FRUIT_LEVEL,
    MIN_FRUIT_LEVEL,
    SPAWN_FRUIT_MAX_LEVEL,
)
from daxigua.core.state import (
    DropResult,
    GameState,
    MergeEvent,
    PhysicsResult,
)
from daxigua_rl.reward import merge_utility
from daxigua_rl.training.identity import TransitionKey

from .schema import (
    ANALYSIS_ACTION_COUNT,
    AttributionEvent,
    AttributionEventKey,
    AttributionEvidence,
    AttributionStepResult,
    Contributor,
    FruitLineageRecord,
    MergeLineageRecord,
    MergeValueKey,
    StateAnalysis,
    StructureOriginRecord,
)


def _strict_integer(name, value, *, minimum=0, maximum=None):
    if isinstance(value, bool):
        raise TypeError(f'{name} must be an integer')
    try:
        result = index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc
    if result < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None and result > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return result


def _finite_float(name, value, *, minimum=None, maximum=None):
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


@dataclass(frozen=True, slots=True, kw_only=True)
class AttributionTrackerConfig:
    """Fixed thresholds and evidence weights for Attribution V1."""

    attribution_version: str = 'causal_attribution_v1'
    transient_stable_steps: int = 3
    burial_confirm_steps: int = 12
    lane_loss_weight: float = 1.0
    top_occlusion_weight: float = 1.0
    critical_contact_weight: float = 1.0
    downward_displacement_weight: float = 1.0
    burial_improvement_threshold: float = 0.05
    significant_displacement_pixels: float = 8.0
    pair_scatter_distance_pixels: float = 12.0
    low_level_max: int = SPAWN_FRUIT_MAX_LEVEL
    large_blocker_min_level: int = 5
    common_blocker_target_count: int = 2
    structure_memory_steps: int = 64
    max_structure_events_per_step: int = 8
    cavity_match_threshold: float = 0.55
    lineage_placement_confidence: float = 0.50
    direct_placement_confidence: float = 0.65
    realized_placement_confidence: float = 0.65
    seal_placement_confidence: float = 0.75

    def __post_init__(self):
        if not isinstance(self.attribution_version, str):
            raise TypeError('attribution_version must be str')
        version = self.attribution_version.strip()
        if not version:
            raise ValueError('attribution_version must not be empty')
        object.__setattr__(self, 'attribution_version', version)

        for field_name in (
                'transient_stable_steps',
                'burial_confirm_steps',
                'low_level_max',
                'large_blocker_min_level',
                'common_blocker_target_count',
                'structure_memory_steps',
                'max_structure_events_per_step'):
            minimum = 1
            maximum = None
            if field_name in {'low_level_max', 'large_blocker_min_level'}:
                maximum = MAX_FRUIT_LEVEL
            object.__setattr__(
                self,
                field_name,
                _strict_integer(
                    field_name,
                    getattr(self, field_name),
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        if self.burial_confirm_steps < self.transient_stable_steps:
            raise ValueError(
                'burial_confirm_steps must be >= transient_stable_steps'
            )

        for field_name in (
                'lane_loss_weight',
                'top_occlusion_weight',
                'critical_contact_weight',
                'downward_displacement_weight',
                'burial_improvement_threshold',
                'significant_displacement_pixels',
                'pair_scatter_distance_pixels'):
            object.__setattr__(
                self,
                field_name,
                _finite_float(
                    field_name,
                    getattr(self, field_name),
                    minimum=0.0,
                ),
            )
        for field_name in (
                'cavity_match_threshold',
                'lineage_placement_confidence',
                'direct_placement_confidence',
                'realized_placement_confidence',
                'seal_placement_confidence'):
            object.__setattr__(
                self,
                field_name,
                _finite_float(
                    field_name,
                    getattr(self, field_name),
                    minimum=0.0,
                    maximum=1.0,
                ),
            )

    @property
    def fingerprint(self):
        payload = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(',', ':'),
        ).encode('utf-8')
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackerTransitionInput:
    """All immutable evidence for one already-executed environment action."""

    transition_key: TransitionKey
    action_offset: int
    action_index: int
    previous_state: GameState
    next_state: GameState
    previous_analysis: StateAnalysis
    next_analysis: StateAnalysis | None
    drop_result: DropResult
    physics_result: PhysicsResult

    def __post_init__(self):
        if not isinstance(self.transition_key, TransitionKey):
            raise TypeError('transition_key must be TransitionKey')
        for field_name in ('action_offset', 'action_index'):
            object.__setattr__(
                self,
                field_name,
                _strict_integer(field_name, getattr(self, field_name)),
            )
        if not isinstance(self.previous_state, GameState):
            raise TypeError('previous_state must be GameState')
        if not isinstance(self.next_state, GameState):
            raise TypeError('next_state must be GameState')
        if not isinstance(self.previous_analysis, StateAnalysis):
            raise TypeError('previous_analysis must be StateAnalysis')
        if not isinstance(self.drop_result, DropResult):
            raise TypeError('drop_result must be DropResult')
        if not isinstance(self.physics_result, PhysicsResult):
            raise TypeError('physics_result must be PhysicsResult')
        if self.physics_result.done and self.physics_result.truncated:
            raise ValueError('transition cannot be both terminated and truncated')

        key = self.transition_key
        if self.previous_state.step_count != key.step_index:
            raise ValueError(
                'previous_state.step_count must equal transition step_index'
            )
        if self.next_state.step_count != key.step_index + 1:
            raise ValueError(
                'next_state.step_count must equal transition step_index + 1'
            )
        if self.previous_analysis.transition_key != key:
            raise ValueError(
                'previous_analysis key must equal transition_key'
            )
        if self.action_offset >= len(
                self.previous_analysis.action_indices):
            raise ValueError(
                'action_offset must index previous_analysis actions'
            )
        if (
                self.previous_analysis.action_indices[self.action_offset]
                != self.action_index):
            raise ValueError(
                'action_index must match previous_analysis at action_offset'
            )
        expected_drop_x = (
            self.previous_analysis.action_drop_x_by_offset[
                self.action_offset
            ]
        )
        if not math.isclose(
                float(self.drop_result.drop_x),
                expected_drop_x,
                rel_tol=0.0,
                abs_tol=1e-6):
            raise ValueError(
                'drop_result.drop_x must match previous_analysis at '
                'action_offset'
            )
        if (
                not self.physics_result.done
                and not isinstance(self.next_analysis, StateAnalysis)):
            raise TypeError(
                'non-terminal transition requires next_analysis'
            )
        if self.next_analysis is not None:
            if not isinstance(self.next_analysis, StateAnalysis):
                raise TypeError(
                    'next_analysis must be StateAnalysis or None'
                )
            expected_next_key = TransitionKey(
                key.worker_id,
                key.episode_id,
                key.step_index + 1,
            )
            if self.next_analysis.transition_key != expected_next_key:
                raise ValueError(
                    'next_analysis must be the adjacent episode boundary'
                )
            if self.next_analysis.incoming_transition_key != key:
                raise ValueError(
                    'next_analysis incoming key must equal transition_key'
                )

        if not self.drop_result.queue_before:
            raise ValueError('drop_result.queue_before must not be empty')
        if self.drop_result.dropped_level != self.drop_result.queue_before[0]:
            raise ValueError(
                'drop_result level must equal queue_before q0'
            )
        if tuple(self.previous_state.fruit_queue) != tuple(
                self.drop_result.queue_before):
            raise ValueError(
                'drop_result.queue_before must equal previous fruit_queue'
            )
        if tuple(self.next_state.fruit_queue) != tuple(
                self.drop_result.queue_after):
            raise ValueError(
                'drop_result.queue_after must equal next fruit_queue'
            )
        if bool(self.next_state.done) != bool(self.physics_result.done):
            raise ValueError(
                'next_state.done must match physics_result.done'
            )

    @property
    def terminated(self):
        return bool(self.physics_result.done)

    @property
    def truncated(self):
        return bool(self.physics_result.truncated)

    @property
    def stable(self):
        return bool(self.physics_result.stable)

    @property
    def merge_events(self):
        return tuple(self.physics_result.merge_events)


@dataclass(frozen=True, slots=True)
class _ContributorSeed:
    transition_key: TransitionKey
    action_offset: int
    action_index: int
    fruit_id: int
    evidence_type: str
    raw_weight: float
    role: str


@dataclass(frozen=True, slots=True)
class _PathLossEvidence:
    transition_key: TransitionKey
    action_offset: int
    action_index: int
    fruit_id: int
    lost_action_mask: int
    top_occlusion: float
    critical_contacts: float
    downward_displacement: float
    contact_link: bool

    def score(self, config):
        confidence = 0.95 if self.contact_link else 0.85
        evidence = (
            config.lane_loss_weight
            * (self.lost_action_mask.bit_count() / ANALYSIS_ACTION_COUNT)
            + config.top_occlusion_weight * self.top_occlusion
            + config.critical_contact_weight * self.critical_contacts
            + config.downward_displacement_weight
            * self.downward_displacement
        )
        return confidence * max(evidence, 1e-9)


@dataclass(slots=True)
class _PendingIncident:
    event: AttributionEvent
    target_level: int
    unreachable_observations: int
    reference_burial_depth: float
    classification_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RescueRecord:
    origin: StructureOriginRecord
    primary_event_key: AttributionEventKey


class AttributionTracker:
    """Maintain exact lineage and delayed structural attribution per worker."""

    _ROLE_PRIORITY = {
        'mechanical_trigger': 9,
        'trigger': 8,
        'rescuer': 7,
        'path_opener': 6,
        'path_blocker': 5,
        'motif_creator': 4,
        'motif_breaker': 3,
        'support': 2,
        'material': 1,
        'victim_drop': 0,
    }

    def __init__(self, config=None):
        self.config = config or AttributionTrackerConfig()
        if not isinstance(self.config, AttributionTrackerConfig):
            raise TypeError(
                'config must be AttributionTrackerConfig'
            )
        self._episode_key = None
        self._episode_active = False
        self._expected_step = 0
        self._next_event_index = 0
        self._lineage = {}
        self._active_fruit_ids = set()
        self._merge_history = []
        self._pending = {}
        self._path_loss_history = defaultdict(list)
        self._structure_origins = {}
        self._recovery_origins = defaultdict(dict)
        self._rescues = {}

    @property
    def episode_key(self):
        return self._episode_key

    @property
    def episode_active(self):
        return self._episode_active

    @property
    def pending_event_count(self):
        return len(self._pending)

    @property
    def pending_events(self):
        return tuple(
            incident.event
            for _fruit_id, incident in sorted(self._pending.items())
        )

    @property
    def merge_history(self):
        return tuple(self._merge_history)

    def lineage_for(self, fruit_id):
        fruit_id = _strict_integer('fruit_id', fruit_id, minimum=1)
        return self._lineage.get(fruit_id)

    def begin_episode(self, worker_id, episode_id):
        """Start an empty episode after the previous one was finalized."""

        if self._episode_active:
            raise RuntimeError(
                'active attribution episode must be finalized first'
            )
        self._episode_key = (
            _strict_integer('worker_id', worker_id),
            _strict_integer('episode_id', episode_id),
        )
        self._episode_active = True
        self._expected_step = 0
        self._next_event_index = 0
        self._lineage.clear()
        self._active_fruit_ids.clear()
        self._merge_history.clear()
        self._pending.clear()
        self._path_loss_history.clear()
        self._structure_origins.clear()
        self._recovery_origins.clear()
        self._rescues.clear()

    def finalize_episode(self, reason='worker_shutdown'):
        """Cancel unresolved incidents without turning them into negatives."""

        if not self._episode_active:
            return ()
        reason = str(reason).strip()
        if not reason:
            raise ValueError('reason must not be empty')
        resolved_step = max(0, self._expected_step - 1)
        resolved = tuple(
            self._resolve_pending_event(
                incident,
                status='cancelled',
                resolved_step=resolved_step,
                reason=reason,
            )
            for _fruit_id, incident in sorted(self._pending.items())
        )
        self._pending.clear()
        self._episode_active = False
        return resolved

    def observe_transition(self, transition):
        """Consume one transition and return immutable event deltas."""

        started_at = time.perf_counter()
        if not isinstance(transition, TrackerTransitionInput):
            raise TypeError(
                'transition must be TrackerTransitionInput'
            )
        self._validate_sequence(transition)

        created_events = []
        resolved_events = []
        lineage_records = []
        diagnostic_codes = []

        self._seed_or_validate_previous_lineage(transition)
        drop_lineage = self._register_drop(transition)
        lineage_records.append(drop_lineage)

        merge_records, merge_lineages = self._apply_merge_stream(
            transition
        )
        lineage_records.extend(merge_lineages)
        self._merge_history.extend(merge_records)
        created_events.extend(self._merge_attribution_events(
            transition,
            merge_records,
        ))

        consumed_ids = {
            source_id
            for record in merge_records
            for source_id in record.source_fruit_ids
        }
        for fruit_id in sorted(consumed_ids):
            incident = self._pending.pop(fruit_id, None)
            if incident is not None:
                resolved_events.append(self._resolve_pending_event(
                    incident,
                    status='cancelled',
                    resolved_step=transition.transition_key.step_index,
                    reason='entered_merge_lineage',
                ))

        created_events.extend(self._realize_structures(
            transition,
            merge_records,
        ))

        interrupted_count = 0
        valid_geometry = (
            transition.next_analysis is not None
            and transition.previous_analysis.diagnostics.valid_for_attribution
            and transition.next_analysis.diagnostics.valid_for_attribution
        )

        if transition.truncated:
            diagnostic_codes.append('truncated_geometry_skipped')
            interrupted_count = len(self._pending)
            for _fruit_id, incident in sorted(self._pending.items()):
                resolved_events.append(self._resolve_pending_event(
                    incident,
                    status='cancelled',
                    resolved_step=transition.transition_key.step_index,
                    reason='truncated',
                ))
            self._pending.clear()
        elif transition.terminated:
            if valid_geometry:
                pending_updates, recovery_events = self._update_pending(
                    transition
                )
                resolved_events.extend(pending_updates)
                created_events.extend(recovery_events)
                new_pending, immediate_events = (
                    self._detect_geometry_changes(transition)
                )
                created_events.extend(new_pending)
                created_events.extend(immediate_events)
                created_events.extend(
                    self._detect_terminal_support_events(transition)
                )
                diagnostic_codes.append(
                    'terminal_post_action_geometry_used'
                )
            else:
                diagnostic_codes.append('terminal_geometry_skipped')
                diagnostic_codes.append(
                    'terminal_support_unavailable_without_final_analysis'
                )
            confirmed, ancillary = self._resolve_terminal_pending(
                transition,
                geometry_trusted=valid_geometry,
            )
            resolved_events.extend(confirmed)
            created_events.extend(ancillary)
        elif valid_geometry:
            pending_updates, recovery_events = self._update_pending(
                transition
            )
            resolved_events.extend(pending_updates)
            created_events.extend(recovery_events)

            new_pending, immediate_events = self._detect_geometry_changes(
                transition
            )
            created_events.extend(new_pending)
            created_events.extend(immediate_events)
            self._update_structure_origins(transition)
        else:
            diagnostic_codes.append('invalid_geometry_skipped')

        self._prune_structure_memory(
            transition.transition_key.step_index
        )
        self._expected_step += 1
        if transition.terminated or transition.truncated:
            self._episode_active = False

        return AttributionStepResult(
            transition_key=transition.transition_key,
            created_events=tuple(created_events),
            resolved_events=tuple(resolved_events),
            lineage_records=tuple(lineage_records),
            merge_records=tuple(merge_records),
            pending_event_count=len(self._pending),
            interrupted_pending_count=interrupted_count,
            tracker_seconds=time.perf_counter() - started_at,
            diagnostic_codes=tuple(diagnostic_codes),
        )

    def _validate_sequence(self, transition):
        if not self._episode_active or self._episode_key is None:
            raise RuntimeError(
                'begin_episode() must be called before observe_transition()'
            )
        key = transition.transition_key
        if (key.worker_id, key.episode_id) != self._episode_key:
            raise ValueError(
                'transition belongs to a different worker or episode'
            )
        if key.step_index != self._expected_step:
            raise ValueError(
                'transition step is out of sequence: '
                f'{key.step_index} != {self._expected_step}'
            )

    def _seed_or_validate_previous_lineage(self, transition):
        previous_by_id = {
            fruit.fruit_id: fruit
            for fruit in transition.previous_analysis.fruit_analyses
        }
        previous_ids = set(previous_by_id)
        if self._expected_step == 0:
            for fruit_id, fruit in sorted(previous_by_id.items()):
                self._lineage[fruit_id] = FruitLineageRecord(
                    episode_key=self._episode_key,
                    fruit_id=fruit_id,
                    level=fruit.level,
                    source_kind='preexisting',
                    root_material_weights=((fruit_id, 1.0),),
                )
            self._active_fruit_ids = set(previous_ids)
            return
        if previous_ids != self._active_fruit_ids:
            raise RuntimeError(
                'lineage active fruit IDs are out of sync with '
                'previous StateAnalysis'
            )

    def _register_drop(self, transition):
        drop = transition.drop_result
        fruit_id = _strict_integer(
            'drop_result.fruit_id',
            drop.fruit_id,
            minimum=1,
        )
        if fruit_id in self._lineage:
            raise ValueError(
                f'dropped fruit ID {fruit_id} already exists in lineage'
            )
        record = FruitLineageRecord(
            episode_key=self._episode_key,
            fruit_id=fruit_id,
            level=drop.dropped_level,
            source_kind='drop',
            created_transition_key=transition.transition_key,
            action_offset=transition.action_offset,
            action_index=transition.action_index,
            root_material_weights=((fruit_id, 1.0),),
        )
        self._lineage[fruit_id] = record
        self._active_fruit_ids.add(fruit_id)
        return record

    def _apply_merge_stream(self, transition):
        merge_records = []
        lineage_records = []
        seen_new_ids = set()
        for event_offset, merge_event in enumerate(
                transition.merge_events):
            if not isinstance(merge_event, MergeEvent):
                raise TypeError(
                    'physics merge_events must contain MergeEvent values'
                )
            source_ids = tuple(merge_event.source_ids)
            if (
                    len(source_ids) != 2
                    or source_ids[0] == source_ids[1]):
                raise ValueError(
                    'merge event must have two distinct source IDs'
                )
            if any(
                    source_id not in self._active_fruit_ids
                    for source_id in source_ids):
                raise ValueError(
                    'merge event references inactive or future source fruit'
                )
            new_id = _strict_integer(
                'merge_event.new_fruit_id',
                merge_event.new_fruit_id,
                minimum=1,
            )
            if (
                    new_id in self._lineage
                    or new_id in seen_new_ids
                    or new_id in source_ids):
                raise ValueError(
                    'merge event new fruit ID must be globally new '
                    'within the episode'
                )
            parents = tuple(self._lineage[source_id] for source_id in source_ids)
            expected_level = parents[0].level + 1
            if (
                    parents[0].level != parents[1].level
                    or merge_event.new_level != expected_level):
                raise ValueError(
                    'merge lineage levels do not match two equal parents'
                )

            root_weights = defaultdict(float)
            for parent in parents:
                for root_id, weight in parent.root_material_weights:
                    root_weights[root_id] += 0.5 * weight
            normalized_roots = tuple(sorted(root_weights.items()))
            value_key = MergeValueKey(
                transition.transition_key,
                event_offset,
            )
            chain_depth = max(
                parent.chain_depth
                for parent in parents
            ) + 1
            lineage = FruitLineageRecord(
                episode_key=self._episode_key,
                fruit_id=new_id,
                level=merge_event.new_level,
                source_kind='merge',
                created_transition_key=transition.transition_key,
                action_offset=transition.action_offset,
                action_index=transition.action_index,
                parent_fruit_ids=source_ids,
                merge_value_key=value_key,
                root_material_weights=normalized_roots,
                chain_depth=chain_depth,
            )
            merge_record = MergeLineageRecord(
                value_key=value_key,
                source_fruit_ids=source_ids,
                new_fruit_id=new_id,
                new_level=merge_event.new_level,
                utility=merge_utility(merge_event.new_level),
                root_material_weights=normalized_roots,
                chain_depth=chain_depth,
            )
            for source_id in source_ids:
                self._active_fruit_ids.remove(source_id)
            self._active_fruit_ids.add(new_id)
            self._lineage[new_id] = lineage
            seen_new_ids.add(new_id)
            lineage_records.append(lineage)
            merge_records.append(merge_record)

        actual_next_ids = {
            fruit.fruit_id
            for fruit in transition.next_state.board_fruits
        }
        if actual_next_ids != self._active_fruit_ids:
            raise RuntimeError(
                'merge lineage result does not match next GameState fruit IDs'
            )
        return tuple(merge_records), tuple(lineage_records)

    def _merge_attribution_events(self, transition, merge_records):
        events = []
        created_this_step = set()
        dropped_id = transition.drop_result.fruit_id
        next_analysis = transition.next_analysis
        first_direct = False
        first_mechanical_path = ()
        if merge_records:
            first_roots = {
                root_id
                for root_id, _weight
                in merge_records[0].root_material_weights
            }
            first_direct = dropped_id in first_roots
            if not first_direct and next_analysis is not None:
                first_mechanical_path = self._contact_path(
                    next_analysis,
                    dropped_id,
                    set(merge_records[0].source_fruit_ids),
                    require_merge_path=True,
                )
        for event_offset, record in enumerate(merge_records):
            root_ids = {
                root_id
                for root_id, _weight in record.root_material_weights
            }
            direct = event_offset == 0 and first_direct
            chained = any(
                source_id in created_this_step
                for source_id in record.source_fruit_ids
            )
            seeds = self._material_contributor_seeds(record)
            if direct:
                seeds.append(_ContributorSeed(
                    transition.transition_key,
                    transition.action_offset,
                    transition.action_index,
                    dropped_id,
                    'direct_trigger',
                    0.75,
                    'trigger',
                ))

            reason_codes = ['actual_merge']
            if direct:
                reason_codes.append('direct_trigger')
            if chained:
                reason_codes.append('chain_trigger')
            primary = self._create_event(
                event_type='MERGE_LINEAGE',
                status='confirmed',
                detected_step=transition.transition_key.step_index,
                resolved_step=transition.transition_key.step_index,
                target_fruit_ids=record.source_fruit_ids,
                contributor_seeds=seeds,
                utility=record.utility,
                link_confidence=1.0,
                placement_confidence=(
                    self.config.direct_placement_confidence
                    if direct
                    else self.config.lineage_placement_confidence
                ),
                evidence=AttributionEvidence(
                    reason_codes=tuple(reason_codes),
                    value_package_keys=(record.value_key,),
                    source_fruit_ids=record.source_fruit_ids,
                ),
                budget_key=record.value_key,
                resolution_reason='actual_merge',
            )
            events.append(primary)

            if direct:
                events.append(self._create_event(
                    event_type='DIRECT_TRIGGER',
                    status='confirmed',
                    detected_step=transition.transition_key.step_index,
                    resolved_step=transition.transition_key.step_index,
                    target_fruit_ids=record.source_fruit_ids,
                    contributor_seeds=(_ContributorSeed(
                        transition.transition_key,
                        transition.action_offset,
                        transition.action_index,
                        dropped_id,
                        'direct_trigger',
                        1.0,
                        'trigger',
                    ),),
                    utility=0.0,
                    link_confidence=1.0,
                    placement_confidence=(
                        self.config.direct_placement_confidence
                    ),
                    evidence=AttributionEvidence(
                        reason_codes=('shared_merge_value_package',),
                        value_package_keys=(record.value_key,),
                        source_fruit_ids=record.source_fruit_ids,
                    ),
                    budget_key=record.value_key,
                    resolution_reason='actual_merge',
                ))

            if chained and (first_direct or first_mechanical_path):
                events.append(self._create_event(
                    event_type='CHAIN_TRIGGER',
                    status='confirmed',
                    detected_step=transition.transition_key.step_index,
                    resolved_step=transition.transition_key.step_index,
                    target_fruit_ids=record.source_fruit_ids,
                    contributor_seeds=(_ContributorSeed(
                        transition.transition_key,
                        transition.action_offset,
                        transition.action_index,
                        dropped_id,
                        (
                            'chain_trigger'
                            if first_direct
                            else 'mechanical_chain_trigger'
                        ),
                        1.0,
                        (
                            'trigger'
                            if first_direct
                            else 'mechanical_trigger'
                        ),
                    ),),
                    utility=0.0,
                    link_confidence=(
                        1.0 if first_direct else 0.92
                    ),
                    placement_confidence=(
                        self.config.direct_placement_confidence
                    ),
                    evidence=AttributionEvidence(
                        reason_codes=('shared_merge_value_package',),
                        value_package_keys=(record.value_key,),
                        source_fruit_ids=record.source_fruit_ids,
                    ),
                    budget_key=record.value_key,
                    resolution_reason='actual_chain',
                ))

            if event_offset == 0 and first_mechanical_path:
                path = first_mechanical_path
                if path:
                    events.append(self._create_event(
                        event_type='MECHANICAL_TRIGGER',
                        status='confirmed',
                        detected_step=transition.transition_key.step_index,
                        resolved_step=transition.transition_key.step_index,
                        target_fruit_ids=record.source_fruit_ids,
                        contributor_seeds=(_ContributorSeed(
                            transition.transition_key,
                            transition.action_offset,
                            transition.action_index,
                            dropped_id,
                            'contact_merge_path',
                            1.0,
                            'mechanical_trigger',
                        ),),
                        utility=0.0,
                        link_confidence=0.92,
                        placement_confidence=(
                            self.config.realized_placement_confidence
                        ),
                        evidence=AttributionEvidence(
                            reason_codes=(
                                'shared_merge_value_package',
                                'unique_contact_path',
                            ),
                            value_package_keys=(record.value_key,),
                            source_fruit_ids=record.source_fruit_ids,
                            contact_path_fruit_ids=path,
                        ),
                        budget_key=record.value_key,
                        resolution_reason='actual_merge',
                    ))
            created_this_step.add(record.new_fruit_id)
        return events

    def _material_contributor_seeds(self, merge_record):
        seeds = []
        for root_id, material_weight in merge_record.root_material_weights:
            root = self._lineage[root_id]
            if (
                    root.source_kind != 'drop'
                    or root.created_transition_key is None):
                continue
            seeds.append(_ContributorSeed(
                root.created_transition_key,
                root.action_offset,
                root.action_index,
                root_id,
                'merge_lineage',
                material_weight,
                'material',
            ))
        return seeds

    def _update_pending(self, transition):
        resolved = []
        ancillary = []
        next_analysis = transition.next_analysis
        for fruit_id, incident in tuple(sorted(self._pending.items())):
            next_fruit = next_analysis.get_fruit(fruit_id)
            if next_fruit is None:
                resolved.append(self._resolve_pending_event(
                    incident,
                    status='cancelled',
                    resolved_step=transition.transition_key.step_index,
                    reason='fruit_disappeared_without_merge',
                ))
                del self._pending[fruit_id]
                continue
            if (
                    next_fruit.reachable_action_count > 0
                    or next_fruit.partner_reachable):
                resolved_event = self._resolve_pending_event(
                    incident,
                    status='cancelled',
                    resolved_step=transition.transition_key.step_index,
                    reason='reachability_recovered',
                )
                resolved.append(resolved_event)
                self._rescues[fruit_id] = _RescueRecord(
                    origin=self._origin_from_transition(
                        transition,
                        'FRUIT_RESCUED',
                        f'rescue:{fruit_id}',
                        (fruit_id,),
                    ),
                    primary_event_key=incident.event.event_id,
                )
                del self._pending[fruit_id]
                continue

            burial_improvement = (
                incident.reference_burial_depth
                - next_fruit.burial_depth
            )
            if (
                    burial_improvement
                    >= self.config.burial_improvement_threshold):
                incident.unreachable_observations = 1
                incident.reference_burial_depth = next_fruit.burial_depth
                continue

            incident.unreachable_observations += 1
            if (
                    incident.unreachable_observations
                    >= self.config.burial_confirm_steps):
                confirmed = self._resolve_pending_event(
                    incident,
                    status='confirmed',
                    resolved_step=transition.transition_key.step_index,
                    reason='burial_confirm_window',
                )
                resolved.append(confirmed)
                ancillary.extend(self._negative_ancillary_events(
                    incident,
                    confirmed,
                ))
                del self._pending[fruit_id]
        return tuple(resolved), tuple(ancillary)

    def _resolve_terminal_pending(
            self,
            transition,
            *,
            geometry_trusted):
        resolved = []
        ancillary = []
        terminal_analysis = transition.next_analysis
        for fruit_id, incident in tuple(sorted(self._pending.items())):
            terminal_fruit = (
                terminal_analysis.get_fruit(fruit_id)
                if geometry_trusted
                and terminal_analysis is not None
                else None
            )
            if (
                    geometry_trusted
                    and terminal_fruit is not None
                    and terminal_fruit.reachable_action_count == 0
                    and not terminal_fruit.partner_reachable):
                confirmed = self._resolve_pending_event(
                    incident,
                    status='confirmed',
                    resolved_step=transition.transition_key.step_index,
                    reason='terminal_still_buried',
                )
                resolved.append(confirmed)
                ancillary.extend(self._negative_ancillary_events(
                    incident,
                    confirmed,
                ))
            else:
                resolved.append(self._resolve_pending_event(
                    incident,
                    status='cancelled',
                    resolved_step=transition.transition_key.step_index,
                    reason=(
                        'terminal_state_not_confirmed'
                        if geometry_trusted
                        else 'terminal_geometry_untrusted'
                    ),
                ))
            del self._pending[fruit_id]
        return tuple(resolved), tuple(ancillary)

    def _detect_terminal_support_events(self, transition):
        """用真实终局后的新支撑割点生成低频终局责任事件。"""

        previous = transition.previous_analysis
        next_analysis = transition.next_analysis
        if next_analysis is None:
            return ()
        previous_keys = {
            self._support_key(edge)
            for edge in previous.support_edges
        }
        new_terminal_edges = tuple(
            edge
            for edge in next_analysis.support_edges
            if (
                edge.relation in {'bridges', 'caps'}
                and self._support_key(edge) not in previous_keys
            )
        )
        if not new_terminal_edges:
            return ()

        previous_by_id = {
            fruit.fruit_id: fruit
            for fruit in previous.fruit_analyses
        }
        newly_unreachable = tuple(sorted(
            fruit.fruit_id
            for fruit in next_analysis.fruit_analyses
            if (
                fruit.fruit_id in previous_by_id
                and previous_by_id[
                    fruit.fruit_id
                ].reachable_action_count > 0
                and fruit.reachable_action_count == 0
            )
        ))
        capacity_loss = max(
            0.0,
            previous.top_connected_capacity
            - next_analysis.top_connected_capacity,
        )
        if not newly_unreachable and capacity_loss <= 0.0:
            return ()

        support_ids = tuple(sorted({
            fruit_id
            for edge in new_terminal_edges
            for fruit_id in (
                edge.supporter_fruit_id,
                edge.supported_fruit_id,
            )
            if fruit_id is not None
        }))
        target_ids = tuple(sorted(set(
            support_ids + newly_unreachable
        )))
        if not target_ids:
            return ()

        event_id = self._allocate_event_key()
        primary_incident = next(
            (
                self._pending[fruit_id]
                for fruit_id in newly_unreachable
                if fruit_id in self._pending
            ),
            None,
        )
        shared_budget_key = (
            primary_incident.event.budget_key
            if primary_incident is not None
            else event_id
        )
        event = self._create_event(
            event_type='TERMINAL_SUPPORT_CREATED',
            status='confirmed',
            detected_step=transition.transition_key.step_index,
            resolved_step=transition.transition_key.step_index,
            target_fruit_ids=target_ids,
            contributor_seeds=(_ContributorSeed(
                transition.transition_key,
                transition.action_offset,
                transition.action_index,
                transition.drop_result.fruit_id,
                'terminal_support_cut',
                max(
                    capacity_loss,
                    len(newly_unreachable)
                    / ANALYSIS_ACTION_COUNT,
                    1e-6,
                ),
                'path_blocker',
            ),),
            utility=(
                0.0
                if primary_incident is not None
                else min(
                    merge_utility(6),
                    merge_utility(4)
                    * (
                        1.0
                        + capacity_loss
                        + len(newly_unreachable)
                        / ANALYSIS_ACTION_COUNT
                    ),
                )
            ),
            link_confidence=0.85,
            placement_confidence=(
                self.config.realized_placement_confidence
            ),
            evidence=AttributionEvidence(
                reason_codes=(
                    'actual_terminal',
                    'new_support_cut',
                    'post_action_terminal_analysis',
                    *(
                        ('shared_negative_incident',)
                        if primary_incident is not None
                        else ()
                    ),
                ),
                blocker_fruit_ids=support_ids,
                previous_reachable_count=(
                    max(
                        previous_by_id[
                            fruit_id
                        ].reachable_action_count
                        for fruit_id in newly_unreachable
                    )
                    if newly_unreachable
                    else None
                ),
                next_reachable_count=(
                    0 if newly_unreachable else None
                ),
                structure_key=(
                    'terminal-support:'
                    + ','.join(
                        self._support_key(edge)
                        for edge in new_terminal_edges
                    )
                ),
                primary_event_key=(
                    primary_incident.event.event_id
                    if primary_incident is not None
                    else None
                ),
            ),
            budget_key=shared_budget_key,
            event_id=event_id,
            resolution_reason='actual_terminal_support_cut',
        )
        return (event,)

    def _detect_geometry_changes(self, transition):
        previous = transition.previous_analysis
        next_analysis = transition.next_analysis
        next_by_id = {
            fruit.fruit_id: fruit
            for fruit in next_analysis.fruit_analyses
        }
        previous_by_id = {
            fruit.fruit_id: fruit
            for fruit in previous.fruit_analyses
        }
        current_drop_id = transition.drop_result.fruit_id
        new_merge_ids = {
            event.new_fruit_id
            for event in transition.merge_events
        }

        for fruit_id in sorted(set(previous_by_id) & set(next_by_id)):
            before = previous_by_id[fruit_id]
            after = next_by_id[fruit_id]
            gained_mask = (
                after.reachable_action_mask
                & ~before.reachable_action_mask
            )
            lost_mask = (
                before.reachable_action_mask
                & ~after.reachable_action_mask
            )
            if gained_mask:
                remaining_history = []
                for item in self._path_loss_history.get(fruit_id, ()):
                    remaining_mask = (
                        item.lost_action_mask & ~gained_mask
                    )
                    if remaining_mask:
                        remaining_history.append(replace(
                            item,
                            lost_action_mask=remaining_mask,
                        ))
                if remaining_history:
                    self._path_loss_history[fruit_id] = (
                        remaining_history
                    )
                else:
                    self._path_loss_history.pop(fruit_id, None)
                self._record_recovery_origins(
                    transition,
                    before,
                    after,
                )
            if lost_mask:
                evidence = self._path_loss_evidence(
                    transition,
                    before,
                    after,
                    lost_mask,
                )
                history = self._path_loss_history[fruit_id]
                history.append(evidence)
                if len(history) > ANALYSIS_ACTION_COUNT:
                    del history[:-ANALYSIS_ACTION_COUNT]

        pending_events = []
        for fruit_id in sorted(set(previous_by_id) & set(next_by_id)):
            before = previous_by_id[fruit_id]
            after = next_by_id[fruit_id]
            if (
                    before.reachable_action_count <= 0
                    or after.reachable_action_count != 0
                    or fruit_id in self._pending):
                continue
            event = self._create_seal_incident(
                transition,
                before,
                after,
                new_merge_ids,
            )
            pending_events.append(event)

        dropped_after = next_by_id.get(current_drop_id)
        if (
                dropped_after is not None
                and dropped_after.reachable_action_count == 0
                and not dropped_after.partner_reachable
                and current_drop_id not in self._pending):
            event = self._create_born_buried_incident(
                transition,
                dropped_after,
            )
            pending_events.append(event)

        immediate_events = list(self._detect_cavity_and_roof_events(
            transition
        ))
        immediate_events.extend(self._detect_broken_motifs(
            transition
        ))
        return tuple(pending_events), tuple(immediate_events)

    def _create_seal_incident(
            self,
            transition,
            before,
            after,
            new_merge_ids):
        history = tuple(self._path_loss_history.get(before.fruit_id, ()))
        if not history:
            history = (
                self._path_loss_evidence(
                    transition,
                    before,
                    after,
                    before.reachable_action_mask,
                ),
            )
        seeds = tuple(
            _ContributorSeed(
                item.transition_key,
                item.action_offset,
                item.action_index,
                item.fruit_id,
                'progressive_path_loss',
                item.score(self.config),
                'path_blocker',
            )
            for item in history
        )
        classifications = self._seal_classifications(
            transition,
            before,
            after,
            new_merge_ids,
        )
        lost_mask = 0
        for item in history:
            lost_mask |= item.lost_action_mask
        # 仅凭实际轨迹中的唯一剩余通道，尚未证明其它投放动作是安全替代；
        # 因此不把该规则抬到需要明确替代动作证据的 A 级。
        placement_confidence = self.config.seal_placement_confidence
        utility = self._negative_incident_utility(
            level=after.level,
            lost_action_count=lost_mask.bit_count(),
            partner_lost=(
                before.partner_reachable
                and not after.partner_reachable
            ),
            capacity_loss=max(
                0.0,
                transition.previous_analysis.top_connected_capacity
                - transition.next_analysis.top_connected_capacity,
            ),
        )
        event_id = self._allocate_event_key()
        event = self._create_event(
            event_type='REACHABILITY_SEALED',
            status='pending',
            detected_step=transition.transition_key.step_index,
            target_fruit_ids=(after.fruit_id,),
            contributor_seeds=seeds,
            utility=utility,
            link_confidence=0.85,
            placement_confidence=placement_confidence,
            evidence=AttributionEvidence(
                reason_codes=classifications,
                blocker_fruit_ids=after.critical_blocker_ids,
                lost_action_mask=lost_mask,
                previous_reachable_count=before.reachable_action_count,
                next_reachable_count=after.reachable_action_count,
                previous_partner_reachable=before.partner_reachable,
                next_partner_reachable=after.partner_reachable,
                previous_burial_depth=before.burial_depth,
                next_burial_depth=after.burial_depth,
                stable_unreachable_steps=1,
            ),
            budget_key=event_id,
            event_id=event_id,
        )
        self._pending[after.fruit_id] = _PendingIncident(
            event=event,
            target_level=after.level,
            unreachable_observations=1,
            reference_burial_depth=after.burial_depth,
            classification_types=classifications,
        )
        self._path_loss_history.pop(after.fruit_id, None)
        return event

    def _create_born_buried_incident(self, transition, fruit):
        event_id = self._allocate_event_key()
        utility = self._negative_incident_utility(
            level=fruit.level,
            lost_action_count=ANALYSIS_ACTION_COUNT,
            partner_lost=True,
            capacity_loss=max(
                0.0,
                transition.previous_analysis.top_connected_capacity
                - transition.next_analysis.top_connected_capacity,
            ),
        )
        event = self._create_event(
            event_type='BORN_BURIED',
            status='pending',
            detected_step=transition.transition_key.step_index,
            target_fruit_ids=(fruit.fruit_id,),
            contributor_seeds=(_ContributorSeed(
                transition.transition_key,
                transition.action_offset,
                transition.action_index,
                transition.drop_result.fruit_id,
                'born_without_top_path',
                1.0,
                'victim_drop',
            ),),
            utility=utility,
            link_confidence=0.90,
            placement_confidence=(
                self.config.seal_placement_confidence
            ),
            evidence=AttributionEvidence(
                reason_codes=('born_buried',),
                blocker_fruit_ids=fruit.critical_blocker_ids,
                previous_reachable_count=None,
                next_reachable_count=0,
                next_partner_reachable=fruit.partner_reachable,
                next_burial_depth=fruit.burial_depth,
                stable_unreachable_steps=1,
            ),
            budget_key=event_id,
            event_id=event_id,
        )
        self._pending[fruit.fruit_id] = _PendingIncident(
            event=event,
            target_level=fruit.level,
            unreachable_observations=1,
            reference_burial_depth=fruit.burial_depth,
            classification_types=(),
        )
        return event

    def _path_loss_evidence(
            self,
            transition,
            before,
            after,
            lost_mask):
        new_critical = (
            set(after.critical_blocker_ids)
            - set(before.critical_blocker_ids)
        )
        contact_path = self._contact_path(
            transition.next_analysis,
            transition.drop_result.fruit_id,
            {before.fruit_id},
        )
        return _PathLossEvidence(
            transition_key=transition.transition_key,
            action_offset=transition.action_offset,
            action_index=transition.action_index,
            fruit_id=transition.drop_result.fruit_id,
            lost_action_mask=lost_mask,
            top_occlusion=max(
                0.0,
                before.top_visible_ratio - after.top_visible_ratio,
            ),
            critical_contacts=min(
                1.0,
                len(new_critical) / max(1, len(after.critical_blocker_ids)),
            ),
            downward_displacement=max(
                0.0,
                after.burial_depth - before.burial_depth,
            ),
            contact_link=bool(contact_path),
        )

    def _seal_classifications(
            self,
            transition,
            before,
            after,
            new_merge_ids):
        classifications = []
        if before.reachable_action_count == 1:
            classifications.append('LAST_LANE_BLOCKED')
        if before.partner_reachable and not after.partner_reachable:
            classifications.append('PARTNER_ISOLATED')
        new_inversions = (
            set(after.inversion_blocker_ids)
            - set(before.inversion_blocker_ids)
        )
        if new_inversions and not after.partner_reachable:
            classifications.append('INVERSION_CREATED')
        if set(after.critical_blocker_ids) & set(new_merge_ids):
            classifications.append('MERGE_EXPANSION_BLOCK')

        previous_state = {
            fruit.fruit_id: fruit
            for fruit in transition.previous_state.board_fruits
        }
        next_state = {
            fruit.fruit_id: fruit
            for fruit in transition.next_state.board_fruits
        }
        if before.fruit_id in previous_state and before.fruit_id in next_state:
            displacement_y = (
                next_state[before.fruit_id].y
                - previous_state[before.fruit_id].y
            )
            if (
                    displacement_y
                    >= self.config.significant_displacement_pixels
                    and self._contact_path(
                        transition.next_analysis,
                        transition.drop_result.fruit_id,
                        {before.fruit_id},
                    )):
                classifications.append('PUSHED_BURIED')

        wall_constrained = any(
            edge.supported_fruit_id == before.fruit_id
            and edge.boundary in {'left_wall', 'right_wall'}
            for edge in (
                transition.previous_analysis.support_edges
                + transition.next_analysis.support_edges
            )
        )
        capped = any(
            edge.supported_fruit_id == before.fruit_id
            and edge.relation == 'caps'
            and edge.supporter_fruit_id in after.critical_blocker_ids
            for edge in transition.next_analysis.support_edges
        )
        if wall_constrained and capped:
            classifications.append('CORNER_CAPPED')

        next_fruit_by_id = {
            fruit.fruit_id: fruit
            for fruit in transition.next_analysis.fruit_analyses
        }
        blocker_target_counts = defaultdict(int)
        for target in transition.next_analysis.fruit_analyses:
            for blocker_id in target.critical_blocker_ids:
                blocker_target_counts[blocker_id] += 1
        if any(
                blocker_target_counts[blocker_id]
                >= self.config.common_blocker_target_count
                and next_fruit_by_id[blocker_id].level
                >= self.config.large_blocker_min_level
                for blocker_id in after.critical_blocker_ids):
            classifications.append('LARGE_BLOCKER_CREATED')
        return tuple(sorted(set(classifications)))

    def _negative_ancillary_events(self, incident, confirmed_event):
        event_types = list(incident.classification_types)
        if incident.target_level <= self.config.low_level_max:
            event_types.append('DEAD_LOW_FRUIT_CONFIRMED')
        events = []
        for event_type in sorted(set(event_types)):
            if event_type in {
                    'REACHABILITY_SEALED',
                    'BORN_BURIED'}:
                continue
            evidence = replace(
                confirmed_event.evidence,
                primary_event_key=confirmed_event.event_id,
                reason_codes=tuple(sorted(set(
                    confirmed_event.evidence.reason_codes
                    + ('shared_negative_incident',)
                ))),
            )
            events.append(self._create_event(
                event_type=event_type,
                status='confirmed',
                detected_step=confirmed_event.detected_step,
                resolved_step=confirmed_event.resolved_step,
                target_fruit_ids=confirmed_event.target_fruit_ids,
                contributor_seeds=self._seeds_from_contributors(
                    confirmed_event.contributors
                ),
                utility=0.0,
                link_confidence=confirmed_event.link_confidence,
                placement_confidence=(
                    confirmed_event.placement_confidence
                ),
                evidence=evidence,
                budget_key=confirmed_event.budget_key,
                resolution_reason='shared_negative_incident',
            ))
        return tuple(events)

    def _resolve_pending_event(
            self,
            incident,
            *,
            status,
            resolved_step,
            reason):
        evidence = replace(
            incident.event.evidence,
            stable_unreachable_steps=incident.unreachable_observations,
            reason_codes=tuple(sorted(set(
                incident.event.evidence.reason_codes
                + (
                    (
                        'transient_window_passed'
                        if incident.unreachable_observations
                        >= self.config.transient_stable_steps
                        else 'transient_window_not_passed'
                    ),
                )
            ))),
        )
        return replace(
            incident.event,
            status=status,
            resolved_step=resolved_step,
            resolution_reason=reason,
            delay=resolved_step - incident.event.detected_step,
            evidence=evidence,
        )

    def _record_recovery_origins(
            self,
            transition,
            before,
            after):
        fruit_id = before.fruit_id
        if before.reachable_action_count == 0:
            self._recovery_origins[fruit_id][
                'BLOCKER_CLEARED'
            ] = self._origin_from_transition(
                transition,
                'BLOCKER_CLEARED',
                f'blocker-cleared:{fruit_id}',
                (fruit_id,),
            )
        if before.inversion_count > 0 and after.inversion_count == 0:
            self._recovery_origins[fruit_id][
                'INVERSION_RESOLVED'
            ] = self._origin_from_transition(
                transition,
                'INVERSION_RESOLVED',
                f'inversion-resolved:{fruit_id}',
                (fruit_id,),
            )
    def _update_structure_origins(self, transition):
        previous = transition.previous_analysis
        next_analysis = transition.next_analysis
        previous_motifs = {
            self._motif_key(motif): motif
            for motif in previous.chain_motifs
        }
        created_fruit_ids = {
            transition.drop_result.fruit_id,
            *(
                event.new_fruit_id
                for event in transition.merge_events
            ),
        }
        for motif in next_analysis.chain_motifs:
            key = self._motif_key(motif)
            has_mechanism_link = bool(
                set(motif.fruit_ids) & created_fruit_ids
                or self._contact_path(
                    next_analysis,
                    transition.drop_result.fruit_id,
                    set(motif.fruit_ids),
                )
            )
            if key not in previous_motifs and has_mechanism_link:
                self._structure_origins[key] = (
                    self._origin_from_transition(
                        transition,
                        motif.motif_type,
                        key,
                        motif.fruit_ids,
                        evidence_weight=max(motif.readiness, 1e-6),
                    )
                )

        previous_support = {
            self._support_key(edge)
            for edge in previous.support_edges
        }
        for edge in next_analysis.support_edges:
            key = self._support_key(edge)
            if key in previous_support:
                continue
            targets = tuple(
                fruit_id
                for fruit_id in (
                    edge.supporter_fruit_id,
                    edge.supported_fruit_id,
                )
                if fruit_id is not None
            )
            self._structure_origins[key] = (
                self._origin_from_transition(
                    transition,
                    'support',
                    key,
                    targets,
                    evidence_weight=edge.confidence,
                )
            )

        dropped_id = transition.drop_result.fruit_id
        dropped = next_analysis.get_fruit(dropped_id)
        if (
                dropped is not None
                and dropped.reachable_action_count > 0
                and any(
                    edge.supported_fruit_id == dropped_id
                    and edge.boundary in {'left_wall', 'right_wall'}
                    for edge in next_analysis.support_edges)):
            key = f'wall-anchor:{dropped_id}'
            self._structure_origins[key] = (
                self._origin_from_transition(
                    transition,
                    'wall_anchor',
                    key,
                    (dropped_id,),
                )
            )

        previous_pairs = self._reachable_partner_pairs(previous)
        for pair in sorted(
                self._reachable_partner_pairs(next_analysis)
                - previous_pairs):
            key = self._partner_key(pair)
            self._structure_origins[key] = (
                self._origin_from_transition(
                    transition,
                    'partner_connected',
                    key,
                    pair,
                )
            )
        self._invalidate_structure_origins(next_analysis)

    def _invalidate_structure_origins(self, analysis):
        """移除已不再成立的结构，避免很久以后误领真实合成。"""

        active_keys = {
            self._motif_key(motif)
            for motif in analysis.chain_motifs
        }
        active_keys.update(
            self._support_key(edge)
            for edge in analysis.support_edges
        )
        active_keys.update(
            self._partner_key(pair)
            for pair in self._reachable_partner_pairs(analysis)
        )
        active_keys.update(
            f'wall-anchor:{fruit.fruit_id}'
            for fruit in analysis.fruit_analyses
            if (
                fruit.reachable_action_count > 0
                and any(
                    edge.supported_fruit_id == fruit.fruit_id
                    and edge.boundary in {'left_wall', 'right_wall'}
                    for edge in analysis.support_edges
                )
            )
        )
        self._structure_origins = {
            key: origin
            for key, origin in self._structure_origins.items()
            if key in active_keys
        }

    def _realize_structures(self, transition, merge_records):
        if not merge_records:
            return ()
        events = []
        previous = transition.previous_analysis
        merge_by_source_pair = {
            tuple(sorted(record.source_fruit_ids)): record
            for record in merge_records
        }

        for pair, merge_record in merge_by_source_pair.items():
            motif = next((
                motif
                for motif in previous.chain_motifs
                if (
                    motif.motif_type == 'merge_pair'
                    and tuple(sorted(motif.fruit_ids)) == pair
                )
            ), None)
            if motif is not None:
                origin = self._structure_origins.pop(
                    self._motif_key(motif),
                    None,
                )
                if origin is not None:
                    events.append(self._realized_event(
                        'REALIZED_ADJACENCY',
                        origin,
                        merge_record,
                    ))

            partner_origin = self._structure_origins.pop(
                self._partner_key(pair),
                None,
            )
            if partner_origin is not None:
                events.append(self._realized_event(
                    'PARTNER_CONNECTED',
                    partner_origin,
                    merge_record,
                ))

            pair_set = set(pair)
            for edge in previous.support_edges:
                edge_ids = {
                    fruit_id
                    for fruit_id in (
                        edge.supporter_fruit_id,
                        edge.supported_fruit_id,
                    )
                    if fruit_id is not None
                }
                if edge_ids != pair_set:
                    continue
                origin = self._structure_origins.pop(
                    self._support_key(edge),
                    None,
                )
                if origin is not None:
                    events.append(self._realized_event(
                        'SUPPORT_PATH_REALIZED',
                        origin,
                        merge_record,
                    ))

            for source_id in pair:
                wall_origin = self._structure_origins.pop(
                    f'wall-anchor:{source_id}',
                    None,
                )
                if wall_origin is not None:
                    events.append(self._realized_event(
                        'WALL_ANCHOR_REALIZED',
                        wall_origin,
                        merge_record,
                    ))

                rescue = self._rescues.pop(source_id, None)
                if rescue is not None:
                    events.append(self._realized_event(
                        'FRUIT_RESCUED',
                        rescue.origin,
                        merge_record,
                        primary_event_key=rescue.primary_event_key,
                    ))
                recovery_types = self._recovery_origins.pop(
                    source_id,
                    {},
                )
                for event_type, origin in sorted(
                        recovery_types.items()):
                    events.append(self._realized_event(
                        event_type,
                        origin,
                        merge_record,
                    ))

        for motif in previous.chain_motifs:
            if motif.motif_type != 'level_ladder':
                continue
            pair = tuple(sorted(motif.fruit_ids[:2]))
            first = merge_by_source_pair.get(pair)
            if first is None:
                continue
            third_id = motif.fruit_ids[2]
            second = next((
                record
                for record in merge_records
                if set(record.source_fruit_ids)
                == {first.new_fruit_id, third_id}
            ), None)
            if second is None:
                continue
            origin = self._structure_origins.pop(
                self._motif_key(motif),
                None,
            )
            if origin is not None:
                events.append(self._realized_event(
                    'REALIZED_LADDER',
                    origin,
                    second,
                ))
        return tuple(events)

    def _realized_event(
            self,
            event_type,
            origin,
            merge_record,
            *,
            primary_event_key=None):
        return self._create_event(
            event_type=event_type,
            status='confirmed',
            detected_step=origin.transition_key.step_index,
            resolved_step=merge_record.value_key.transition_key.step_index,
            target_fruit_ids=origin.target_fruit_ids,
            contributor_seeds=(_ContributorSeed(
                origin.transition_key,
                origin.action_offset,
                origin.action_index,
                origin.fruit_id,
                origin.structure_type,
                origin.evidence_weight,
                (
                    'rescuer'
                    if event_type == 'FRUIT_RESCUED'
                    else 'path_opener'
                    if event_type in {
                        'BLOCKER_CLEARED',
                        'CORRIDOR_OPENED_USED',
                        'INVERSION_RESOLVED',
                        'PARTNER_CONNECTED',
                    }
                    else 'support'
                    if event_type in {
                        'WALL_ANCHOR_REALIZED',
                        'SUPPORT_PATH_REALIZED',
                    }
                    else 'motif_creator'
                ),
            ),),
            utility=0.0,
            link_confidence=0.85,
            placement_confidence=(
                self.config.realized_placement_confidence
            ),
            evidence=AttributionEvidence(
                reason_codes=(
                    'realized_by_actual_merge',
                    'shared_merge_value_package',
                ),
                value_package_keys=(merge_record.value_key,),
                primary_event_key=primary_event_key,
                source_fruit_ids=merge_record.source_fruit_ids,
                structure_key=origin.structure_key,
            ),
            budget_key=merge_record.value_key,
            resolution_reason='realized_by_actual_merge',
        )

    def _detect_cavity_and_roof_events(self, transition):
        previous = transition.previous_analysis
        next_analysis = transition.next_analysis
        previous_sealed = tuple(
            region
            for region in previous.free_space_regions
            if region.sealed
        )
        new_regions = []
        for region in next_analysis.free_space_regions:
            if not region.sealed:
                continue
            best_match = max(
                (
                    self._region_match_score(region, candidate)
                    for candidate in previous_sealed
                ),
                default=0.0,
            )
            if best_match < self.config.cavity_match_threshold:
                new_regions.append(region)

        events = []
        if new_regions:
            targets = tuple(sorted({
                fruit_id
                for region in new_regions
                for fruit_id in region.boundary_fruit_ids
            }))
            if targets:
                event_id = self._allocate_event_key()
                events.append(self._create_event(
                    event_type='SEALED_CAVITY_CREATED',
                    status='confirmed',
                    detected_step=transition.transition_key.step_index,
                    resolved_step=transition.transition_key.step_index,
                    target_fruit_ids=targets,
                    contributor_seeds=(_ContributorSeed(
                        transition.transition_key,
                        transition.action_offset,
                        transition.action_index,
                        transition.drop_result.fruit_id,
                        'new_unmatched_sealed_region',
                        sum(region.area_ratio for region in new_regions),
                        'path_blocker',
                    ),),
                    utility=min(
                        merge_utility(5),
                        sum(region.area_ratio for region in new_regions),
                    ),
                    link_confidence=0.68,
                    placement_confidence=0.55,
                    evidence=AttributionEvidence(
                        reason_codes=(
                            'region_geometry_match',
                            'structural_event_only',
                        ),
                        structure_key=(
                            'sealed-cavity:'
                            + ','.join(str(item) for item in targets)
                        ),
                    ),
                    budget_key=event_id,
                    event_id=event_id,
                    resolution_reason='stable_geometry_change',
                ))

        previous_bridges = {
            self._support_key(edge)
            for edge in previous.support_edges
            if edge.relation == 'bridges'
        }
        new_bridges = tuple(
            edge
            for edge in next_analysis.support_edges
            if (
                edge.relation == 'bridges'
                and self._support_key(edge) not in previous_bridges
            )
        )
        capacity_loss = max(
            0.0,
            previous.top_connected_capacity
            - next_analysis.top_connected_capacity,
        )
        if new_bridges and (new_regions or capacity_loss > 0.0):
            targets = tuple(sorted({
                fruit_id
                for edge in new_bridges
                for fruit_id in (
                    edge.supporter_fruit_id,
                    edge.supported_fruit_id,
                )
                if fruit_id is not None
            }))
            if targets:
                event_id = self._allocate_event_key()
                events.append(self._create_event(
                    event_type='ROOF_BRIDGE_CREATED',
                    status='confirmed',
                    detected_step=transition.transition_key.step_index,
                    resolved_step=transition.transition_key.step_index,
                    target_fruit_ids=targets,
                    contributor_seeds=(_ContributorSeed(
                        transition.transition_key,
                        transition.action_offset,
                        transition.action_index,
                        transition.drop_result.fruit_id,
                        'new_bridge_with_space_loss',
                        max(capacity_loss, 1e-6),
                        'path_blocker',
                    ),),
                    utility=min(merge_utility(5), capacity_loss),
                    link_confidence=0.70,
                    placement_confidence=0.55,
                    evidence=AttributionEvidence(
                        reason_codes=(
                            'bridge_relation',
                            'top_connected_space_loss',
                        ),
                        structure_key=(
                            'roof-bridge:'
                            + ','.join(str(item) for item in targets)
                        ),
                    ),
                    budget_key=event_id,
                    event_id=event_id,
                    resolution_reason='stable_geometry_change',
                ))
        return tuple(events)

    def _detect_broken_motifs(self, transition):
        previous = transition.previous_analysis
        next_analysis = transition.next_analysis
        next_keys = {
            self._motif_key(motif)
            for motif in next_analysis.chain_motifs
        }
        consumed = {
            source_id
            for event in transition.merge_events
            for source_id in event.source_ids
        }
        next_ids = {
            fruit.fruit_id
            for fruit in next_analysis.fruit_analyses
        }
        previous_state = {
            fruit.fruit_id: fruit
            for fruit in transition.previous_state.board_fruits
        }
        next_state = {
            fruit.fruit_id: fruit
            for fruit in transition.next_state.board_fruits
        }
        events = []
        for motif in previous.chain_motifs:
            if len(events) >= self.config.max_structure_events_per_step:
                break
            key = self._motif_key(motif)
            if (
                    key in next_keys
                    or set(motif.fruit_ids) & consumed
                    or not set(motif.fruit_ids).issubset(next_ids)):
                continue

            contact_path = self._contact_path(
                next_analysis,
                transition.drop_result.fruit_id,
                set(motif.fruit_ids),
            )
            # ChainMotif 本身受 q0-q3 队列条件约束；没有接触链时，它的
            # 消失可能只是队列推进，不能归因给当前投放。
            if not contact_path:
                continue
            event_type = 'CHAIN_MOTIF_BROKEN'
            if (
                    motif.motif_type == 'merge_pair'
                    and len(motif.fruit_ids) == 2):
                left_id, right_id = motif.fruit_ids
                old_distance = math.hypot(
                    previous_state[left_id].x - previous_state[right_id].x,
                    previous_state[left_id].y - previous_state[right_id].y,
                )
                new_distance = math.hypot(
                    next_state[left_id].x - next_state[right_id].x,
                    next_state[left_id].y - next_state[right_id].y,
                )
                if (
                        new_distance - old_distance
                        >= self.config.pair_scatter_distance_pixels):
                    event_type = 'PAIR_SCATTERED'

            event_id = self._allocate_event_key()
            events.append(self._create_event(
                event_type=event_type,
                status='confirmed',
                detected_step=transition.transition_key.step_index,
                resolved_step=transition.transition_key.step_index,
                target_fruit_ids=motif.fruit_ids,
                contributor_seeds=(_ContributorSeed(
                    transition.transition_key,
                    transition.action_offset,
                    transition.action_index,
                    transition.drop_result.fruit_id,
                    'contact_scatter',
                    max(motif.readiness, 1e-6),
                    'motif_breaker',
                ),),
                utility=min(merge_utility(5), motif.readiness),
                link_confidence=0.90,
                placement_confidence=(
                    self.config.realized_placement_confidence
                ),
                evidence=AttributionEvidence(
                    reason_codes=('contact_path',),
                    contact_path_fruit_ids=contact_path,
                    structure_key=key,
                ),
                budget_key=event_id,
                event_id=event_id,
                resolution_reason='stable_geometry_change',
            ))
        return tuple(events)

    def _create_event(
            self,
            *,
            event_type,
            status,
            detected_step,
            target_fruit_ids,
            contributor_seeds,
            utility,
            link_confidence,
            placement_confidence,
            evidence,
            budget_key,
            event_id=None,
            resolved_step=None,
            resolution_reason=None):
        if event_id is None:
            event_id = self._allocate_event_key()
        return AttributionEvent(
            event_id=event_id,
            episode_key=self._episode_key,
            attribution_version=self.config.attribution_version,
            tracker_config_fingerprint=self.config.fingerprint,
            detected_step=detected_step,
            resolved_step=resolved_step,
            event_type=event_type,
            status=status,
            sign=1 if event_type in {
                'MERGE_LINEAGE',
                'DIRECT_TRIGGER',
                'MECHANICAL_TRIGGER',
                'CHAIN_TRIGGER',
                'REALIZED_ADJACENCY',
                'REALIZED_LADDER',
                'WALL_ANCHOR_REALIZED',
                'SUPPORT_PATH_REALIZED',
                'PARTNER_CONNECTED',
                'FRUIT_RESCUED',
                'CORRIDOR_OPENED_USED',
                'INVERSION_RESOLVED',
                'BLOCKER_CLEARED',
            } else -1,
            target_fruit_ids=tuple(target_fruit_ids),
            contributors=self._normalize_contributors(contributor_seeds),
            utility=utility,
            link_confidence=link_confidence,
            placement_confidence=placement_confidence,
            evidence=evidence,
            budget_key=budget_key,
            resolution_reason=resolution_reason,
        )

    def _normalize_contributors(self, seeds):
        aggregate = {}
        for seed in seeds:
            if not isinstance(seed, _ContributorSeed):
                raise TypeError(
                    'contributor seeds must be _ContributorSeed values'
                )
            key = (
                seed.transition_key,
                seed.action_offset,
                seed.action_index,
                seed.fruit_id,
            )
            if key not in aggregate:
                aggregate[key] = {
                    'weight': 0.0,
                    'types': set(),
                    'role': seed.role,
                }
            entry = aggregate[key]
            entry['weight'] += max(0.0, float(seed.raw_weight))
            entry['types'].add(seed.evidence_type)
            if (
                    self._ROLE_PRIORITY[seed.role]
                    > self._ROLE_PRIORITY[entry['role']]):
                entry['role'] = seed.role
        if not aggregate:
            return ()
        total = sum(entry['weight'] for entry in aggregate.values())
        if total <= 0.0:
            total = float(len(aggregate))
            for entry in aggregate.values():
                entry['weight'] = 1.0
        contributors = []
        for key, entry in sorted(aggregate.items()):
            transition_key, action_offset, action_index, fruit_id = key
            contributors.append(Contributor(
                transition_key=transition_key,
                action_offset=action_offset,
                action_index=action_index,
                fruit_id=fruit_id,
                evidence_type='+'.join(sorted(entry['types'])),
                raw_evidence_weight=entry['weight'],
                contribution_weight=entry['weight'] / total,
                role=entry['role'],
            ))
        return tuple(contributors)

    @staticmethod
    def _seeds_from_contributors(contributors):
        return tuple(
            _ContributorSeed(
                contributor.transition_key,
                contributor.action_offset,
                contributor.action_index,
                contributor.fruit_id,
                contributor.evidence_type,
                contributor.raw_evidence_weight,
                contributor.role,
            )
            for contributor in contributors
        )

    def _allocate_event_key(self):
        if self._episode_key is None:
            raise RuntimeError('no active attribution episode')
        key = AttributionEventKey(
            self._episode_key[0],
            self._episode_key[1],
            self._next_event_index,
        )
        self._next_event_index += 1
        return key

    def _origin_from_transition(
            self,
            transition,
            structure_type,
            structure_key,
            target_fruit_ids,
            evidence_weight=1.0):
        return StructureOriginRecord(
            structure_type=structure_type,
            structure_key=structure_key,
            transition_key=transition.transition_key,
            action_offset=transition.action_offset,
            action_index=transition.action_index,
            fruit_id=transition.drop_result.fruit_id,
            target_fruit_ids=tuple(target_fruit_ids),
            evidence_weight=max(float(evidence_weight), 1e-9),
        )

    def _prune_structure_memory(self, current_step):
        cutoff = current_step - self.config.structure_memory_steps
        self._structure_origins = {
            key: origin
            for key, origin in self._structure_origins.items()
            if origin.transition_key.step_index >= cutoff
        }
        for fruit_id, origins in tuple(self._recovery_origins.items()):
            kept = {
                event_type: origin
                for event_type, origin in origins.items()
                if origin.transition_key.step_index >= cutoff
            }
            if kept:
                self._recovery_origins[fruit_id] = kept
            else:
                del self._recovery_origins[fruit_id]
        self._rescues = {
            fruit_id: rescue
            for fruit_id, rescue in self._rescues.items()
            if rescue.origin.transition_key.step_index >= cutoff
        }

    @staticmethod
    def _motif_key(motif):
        return (
            f'motif:{motif.motif_type}:'
            + ','.join(str(item) for item in motif.fruit_ids)
            + ':'
            + ','.join(str(item) for item in motif.levels)
        )

    @staticmethod
    def _support_key(edge):
        source = (
            f'fruit-{edge.supporter_fruit_id}'
            if edge.supporter_fruit_id is not None
            else f'boundary-{edge.boundary}'
        )
        return (
            f'support:{edge.relation}:{source}:'
            f'target-{edge.supported_fruit_id}'
        )

    @staticmethod
    def _partner_key(pair):
        pair = tuple(sorted(pair))
        return f'partner:{pair[0]}:{pair[1]}'

    @staticmethod
    def _reachable_partner_pairs(analysis):
        return {
            tuple(sorted((fruit.fruit_id, partner_id)))
            for fruit in analysis.fruit_analyses
            for partner_id in fruit.reachable_partner_ids
            if fruit.fruit_id < partner_id
        }

    @staticmethod
    def _contact_path(
            analysis,
            source_id,
            target_ids,
            *,
            require_merge_path=False):
        if analysis is None:
            return ()
        target_ids = set(target_ids)
        adjacency = defaultdict(set)
        for edge in analysis.contact_influence_edges:
            if require_merge_path and not edge.on_merge_path:
                continue
            adjacency[edge.source_fruit_id].add(edge.target_fruit_id)
        queue = deque(((source_id, (source_id,)),))
        visited = {source_id}
        while queue:
            current, path = queue.popleft()
            if current in target_ids and current != source_id:
                return path
            for neighbor in sorted(adjacency[current]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + (neighbor,)))
        return ()

    @staticmethod
    def _region_match_score(left, right):
        intersection_width = max(
            0.0,
            min(left.max_x, right.max_x)
            - max(left.min_x, right.min_x),
        )
        intersection_height = max(
            0.0,
            min(left.max_y, right.max_y)
            - max(left.min_y, right.min_y),
        )
        intersection = intersection_width * intersection_height
        left_area = max(
            1e-9,
            (left.max_x - left.min_x) * (left.max_y - left.min_y),
        )
        right_area = max(
            1e-9,
            (right.max_x - right.min_x) * (right.max_y - right.min_y),
        )
        bbox_iou = intersection / max(
            1e-9,
            left_area + right_area - intersection,
        )
        left_boundary = set(left.boundary_fruit_ids)
        right_boundary = set(right.boundary_fruit_ids)
        boundary_union = left_boundary | right_boundary
        boundary_jaccard = (
            len(left_boundary & right_boundary) / len(boundary_union)
            if boundary_union
            else 1.0
        )
        centroid_distance = math.hypot(
            left.centroid_x - right.centroid_x,
            left.centroid_y - right.centroid_y,
        )
        centroid_score = max(
            0.0,
            1.0 - centroid_distance / math.sqrt(2.0),
        )
        area_score = (
            min(left.area_ratio, right.area_ratio)
            / max(left.area_ratio, right.area_ratio, 1e-9)
        )
        return (
            0.35 * bbox_iou
            + 0.35 * boundary_jaccard
            + 0.20 * centroid_score
            + 0.10 * area_score
        )

    @staticmethod
    def _negative_incident_utility(
            *,
            level,
            lost_action_count,
            partner_lost,
            capacity_loss):
        low_level_weight = 2.0 ** (-(level - MIN_FRUIT_LEVEL) / 2.0)
        raw = (
            lost_action_count / ANALYSIS_ACTION_COUNT
            + 0.5 * low_level_weight
            + 0.25 * float(partner_lost)
            + max(0.0, capacity_loss)
        )
        return min(merge_utility(5), float(raw))


__all__ = [
    'AttributionTracker',
    'AttributionTrackerConfig',
    'TrackerTransitionInput',
]
