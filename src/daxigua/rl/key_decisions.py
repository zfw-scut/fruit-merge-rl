"""低侵入训练事实采集器与可插拔关键决策选择边界。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from daxigua.simulator import (
    BatchDropResult,
    BatchMergeEvents,
    BatchPhysicsResult,
)

from .config import DecisionDataConfig
from .decision_data import (
    ActionSelectionBatch,
    AsyncDecisionArchive,
    CompositeDecisionSink,
    DecisionFactBatch,
    DecisionSelectionBatch,
    FACT_PRODUCER_VERSION,
    GpuDecisionBuffer,
)
from .observations import TensorState
from .replay import GpuReplayBuffer, ReplayStateReference, ReplayWriteTicket


@dataclass(frozen=True, slots=True)
class DecisionPreContext:
    """选择器可在物理推进前读取、但不得长期保留视图的事实。"""

    decision_ids: torch.Tensor
    episode_ids: torch.Tensor
    environment_rows: torch.Tensor
    current: TensorState
    action_selection: ActionSelectionBatch
    transition_start: int
    policy_version: int


@dataclass(frozen=True, slots=True)
class DecisionPostContext:
    """选择器在事实物理结果可用后接收的只读上下文。"""

    pre: DecisionPreContext
    prepared: object
    result: object
    next_state: TensorState
    rewards: torch.Tensor
    stages: torch.Tensor
    episode_finished: torch.Tensor


class DecisionSelector(Protocol):
    """未来事件规则或模型选择器需要实现的两阶段接口。"""

    active: bool

    def prepare(self, context: DecisionPreContext):
        ...

    def select(self, context: DecisionPostContext) -> DecisionSelectionBatch:
        ...


class EmptyDecisionSelector:
    """默认空选择器；确保只配置框架不会改变训练热路径。"""

    active = False

    def __init__(self, capacity):
        self.capacity = int(capacity)

    def prepare(self, context):
        return None

    def select(self, context):
        return DecisionSelectionBatch.empty(
            self.capacity, device=context.rewards.device
        )


@dataclass(frozen=True, slots=True)
class StagedDecisionBatch:
    """一次物理推进期间短暂保留的设备端决策事实。"""

    pre_context: DecisionPreContext
    replay_reference: ReplayStateReference
    pre_sidecar: object
    prepared_selector_state: object


def _select_optional(value, rows):
    return None if value is None else value.index_select(0, rows)


def _select_drop(drop, rows):
    return BatchDropResult(
        dropped_levels=drop.dropped_levels.index_select(0, rows),
        drop_x=drop.drop_x.index_select(0, rows),
        physics_radius=drop.physics_radius.index_select(0, rows),
        fruit_ids=drop.fruit_ids.index_select(0, rows),
        queue_before=drop.queue_before.index_select(0, rows),
        queue_after=drop.queue_after.index_select(0, rows),
    )


def _select_merge_events(events, rows):
    return BatchMergeEvents(
        count=events.count.index_select(0, rows),
        source_levels=events.source_levels.index_select(0, rows),
        new_levels=events.new_levels.index_select(0, rows),
        positions=events.positions.index_select(0, rows),
        score_deltas=events.score_deltas.index_select(0, rows),
        source_ids=events.source_ids.index_select(0, rows),
        new_fruit_ids=events.new_fruit_ids.index_select(0, rows),
    )


def _select_physics(physics, rows):
    return BatchPhysicsResult(
        frames_simulated=physics.frames_simulated.index_select(0, rows),
        stable=physics.stable.index_select(0, rows),
        done=physics.done.index_select(0, rows),
        truncated=physics.truncated.index_select(0, rows),
        score_delta=physics.score_delta.index_select(0, rows),
        merge_events=_select_merge_events(physics.merge_events, rows),
        settle_timeout=_select_optional(physics.settle_timeout, rows),
        fast_forwarded_frames=_select_optional(
            physics.fast_forwarded_frames, rows
        ),
        collision_substeps=_select_optional(
            physics.collision_substeps, rows
        ),
        action_effects=(
            None
            if physics.action_effects is None
            else physics.action_effects.index_select(rows)
        ),
    )


class KeyDecisionCollector:
    """把训练事实发布到设备内存和异步归档，不定义关键条件。"""

    def __init__(
            self,
            config,
            *,
            replay,
            simulator,
            run_dir,
            selector=None,
            extra_sinks=()):
        if not isinstance(config, DecisionDataConfig):
            raise TypeError('config must be DecisionDataConfig')
        if not isinstance(replay, GpuReplayBuffer):
            raise TypeError('replay must be GpuReplayBuffer')
        self.config = config
        self.replay = replay
        self.simulator = simulator
        self.run_dir = Path(run_dir)
        self.run_id = self.run_dir.name
        self.selector = selector or EmptyDecisionSelector(
            config.max_candidates_per_step
        )
        self.configured = bool(config.enabled)
        self.active = bool(
            self.configured and getattr(self.selector, 'active', True)
        )
        if self.active and not replay.state_references_enabled:
            raise ValueError(
                'active decision collection requires versioned Replay references'
            )
        self.gpu_buffer = None
        sinks = list(extra_sinks)
        if self.active and config.gpu_retention_capacity > 0:
            self.gpu_buffer = GpuDecisionBuffer(
                config.gpu_retention_capacity
            )
            sinks.append(self.gpu_buffer)
        self.archive = None
        if self.active and config.archive_enabled:
            self.archive = AsyncDecisionArchive(
                self.run_dir / config.archive_subdirectory,
                shard_records=config.archive_shard_records,
                queue_size=config.archive_queue_size,
                max_storage_bytes=config.max_archive_bytes,
            )
            sinks.append(self.archive)
        self.sink = CompositeDecisionSink(sinks)
        self._episode_ids = None
        self._candidate_slots = None
        self._valid_records = None
        self._invalid_replay_references = None
        if self.active:
            self._episode_ids = torch.full(
                (simulator.num_envs,),
                -1,
                dtype=torch.int64,
                device=simulator.device,
            )
            self._candidate_slots = torch.zeros(
                (), dtype=torch.int64, device=simulator.device
            )
            self._valid_records = torch.zeros_like(self._candidate_slots)
            self._invalid_replay_references = torch.zeros_like(
                self._candidate_slots
            )
        self._submitted_batches = 0
        self._closed = False

    def stage_pre(
            self,
            *,
            current,
            action_selection,
            ticket,
            environment_rows,
            transition_start,
            policy_version):
        if not self.active:
            return None
        if not isinstance(action_selection, ActionSelectionBatch):
            raise TypeError('action_selection must be ActionSelectionBatch')
        if not isinstance(ticket, ReplayWriteTicket):
            raise TypeError('ticket must be ReplayWriteTicket')
        batch_size = current.batch_size
        if ticket.count != batch_size:
            raise ValueError('replay ticket must align with the actor batch')
        environment_rows = torch.as_tensor(
            environment_rows, dtype=torch.int64, device=current.device
        ).flatten()
        if environment_rows.shape != (batch_size,):
            raise ValueError('environment_rows must align with actor batch')
        decision_ids = torch.arange(
            int(transition_start),
            int(transition_start) + batch_size,
            dtype=torch.int64,
            device=current.device,
        )
        previous_episode_ids = self._episode_ids.index_select(
            0, environment_rows
        )
        episode_ids = torch.where(
            previous_episode_ids < 0,
            decision_ids,
            previous_episode_ids,
        )
        self._episode_ids.index_copy_(0, environment_rows, episode_ids)
        pre_context = DecisionPreContext(
            decision_ids=decision_ids,
            episode_ids=episode_ids,
            environment_rows=environment_rows,
            current=current,
            action_selection=action_selection,
            transition_start=int(transition_start),
            policy_version=int(policy_version),
        )
        prepared = self.selector.prepare(pre_context)
        return StagedDecisionBatch(
            pre_context=pre_context,
            replay_reference=ticket.reference,
            pre_sidecar=self.simulator.export_decision_sidecar(
                environment_rows
            ),
            prepared_selector_state=prepared,
        )

    def observe_post(
            self,
            staged,
            *,
            result,
            next_state,
            rewards,
            stages,
            episode_finished):
        if staged is None:
            return None
        pre = staged.pre_context
        post_context = DecisionPostContext(
            pre=pre,
            prepared=staged.prepared_selector_state,
            result=result,
            next_state=next_state,
            rewards=rewards,
            stages=stages,
            episode_finished=episode_finished,
        )
        selection = self.selector.select(post_context)
        if not isinstance(selection, DecisionSelectionBatch):
            raise TypeError('selector must return DecisionSelectionBatch')
        if selection.capacity != self.config.max_candidates_per_step:
            raise ValueError('selector returned an unexpected fixed capacity')
        batch_size = pre.current.batch_size
        safe_rows = selection.rows.clamp(0, batch_size - 1)
        in_range = selection.rows.eq(safe_rows)
        valid = selection.valid_mask & in_range
        environment_rows = pre.environment_rows.index_select(0, safe_rows)
        replay_reference = staged.replay_reference.index_select(safe_rows)
        replay_state = self.replay.gather_current(replay_reference)
        valid &= replay_state.valid
        selected_actions = ActionSelectionBatch(
            actions=pre.action_selection.actions.index_select(0, safe_rows),
            greedy_actions=(
                pre.action_selection.greedy_actions.index_select(
                    0, safe_rows
                )
            ),
            explore_mask=pre.action_selection.explore_mask.index_select(
                0, safe_rows
            ),
            q_values=pre.action_selection.q_values.index_select(0, safe_rows),
            uncertainty=(
                None
                if pre.action_selection.uncertainty is None
                else pre.action_selection.uncertainty.index_select(
                    0, safe_rows
                )
            ),
            active_learning_mask=(
                None
                if pre.action_selection.active_learning_mask is None
                else pre.action_selection.active_learning_mask.index_select(
                    0, safe_rows
                )
            ),
        )
        policy_versions = torch.full_like(
            selection.rows, pre.policy_version, dtype=torch.int64
        )
        missing_ids = torch.full_like(selection.rows, -1, dtype=torch.int64)
        facts = DecisionFactBatch(
            run_id=self.run_id,
            producer_version=FACT_PRODUCER_VERSION,
            information_scope='agent_observation',
            decision_ids=pre.decision_ids.index_select(0, safe_rows),
            episode_ids=pre.episode_ids.index_select(0, safe_rows),
            segment_ids=missing_ids,
            plan_ids=missing_ids.clone(),
            environment_rows=environment_rows,
            replay_reference=replay_reference,
            policy_versions=policy_versions,
            priorities=selection.priorities,
            reason_bits=selection.reason_bits,
            valid_mask=valid,
            action_selection=selected_actions,
            rewards=rewards.index_select(0, safe_rows),
            stages=stages.index_select(0, safe_rows),
            current=replay_state.state,
            next_state=next_state.index_select(safe_rows),
            pre_sidecar=staged.pre_sidecar.index_select(safe_rows),
            drop=_select_drop(result.drop, environment_rows),
            physics=_select_physics(result.physics, environment_rows),
        )
        self._candidate_slots += selection.valid_mask.to(torch.int64).sum()
        self._valid_records += valid.to(torch.int64).sum()
        self._invalid_replay_references += (
            selection.valid_mask & ~replay_state.valid
        ).to(torch.int64).sum()
        self.sink.submit(facts)
        self._submitted_batches += 1
        self.mark_episode_finished(
            pre.environment_rows, episode_finished
        )
        return facts

    def mark_episode_finished(self, environment_rows, finished_mask):
        if not self.active:
            return
        environment_rows = torch.as_tensor(
            environment_rows,
            dtype=torch.int64,
            device=self._episode_ids.device,
        ).flatten()
        finished_mask = torch.as_tensor(
            finished_mask,
            dtype=torch.bool,
            device=self._episode_ids.device,
        ).flatten()
        current = self._episode_ids.index_select(0, environment_rows)
        current = torch.where(
            finished_mask,
            torch.full_like(current, -1),
            current,
        )
        self._episode_ids.index_copy_(0, environment_rows, current)

    def on_env_reset(self, full_reset_mask):
        if not self.active:
            return
        mask = torch.as_tensor(
            full_reset_mask,
            dtype=torch.bool,
            device=self._episode_ids.device,
        )
        self._episode_ids.masked_fill_(mask, -1)

    def flush(self):
        if self.active and not self._closed:
            self.sink.flush()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.active:
            self.sink.close()

    def metrics(self):
        result = {
            'configured': self.configured,
            'active': self.active,
        }
        if not self.active:
            return result
        result.update({
            'submitted_batches': self._submitted_batches,
            'candidate_slots': int(self._candidate_slots.item()),
            'valid_records': int(self._valid_records.item()),
            'invalid_replay_references': int(
                self._invalid_replay_references.item()
            ),
        })
        result.update(self.sink.metrics())
        return result
