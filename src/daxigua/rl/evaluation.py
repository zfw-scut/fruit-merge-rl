"""与训练 Replay 完全隔离的 30/120 FPS greedy 评估。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import statistics
import time

import torch

from daxigua.simulator import SimulatorConfig, TensorVectorSimulator

from .observations import TensorState


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    physics_fps: int
    episodes: int
    mean_score: float
    median_score: float
    score_std: float
    mean_drops: float
    median_drops: float
    mean_max_level: float
    mean_merges: float
    mean_final_fruit_count: float
    danger_drop_rate: float
    settle_timeout_rate: float
    timeout_episodes: int
    elapsed_seconds: float
    env_steps_per_second: float

    def to_dict(self):
        return asdict(self)


def _percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _state_rows_to_cpu(state, rows):
    selected = state.index_select(rows)
    return {
        name: getattr(selected, name).detach().cpu()
        for name in selected.__dataclass_fields__
        if name != 'physics_fps'
    }


def _save_trajectory_trace(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate_policy(
        model,
        *,
        physics_fps,
        episodes,
        parallel_envs,
        device,
        seed_base,
        max_fruits=64,
        max_episode_drops=1000,
        trajectory_output_path=None,
        trajectory_episodes=0):
    if physics_fps == 30:
        simulator_config = SimulatorConfig.training_fast(
            max_fruits=max_fruits
        )
    elif physics_fps == 120:
        simulator_config = SimulatorConfig.high_fidelity_fast(
            max_fruits=max_fruits
        )
    else:
        raise ValueError('evaluation physics_fps must be 30 or 120')
    batch_size = min(int(parallel_envs), int(episodes))
    simulator = TensorVectorSimulator(
        batch_size, config=simulator_config, device=device
    )
    simulator.reset(seeds=int(seed_base))
    was_training = model.training
    model.eval()
    scores = []
    drops = []
    max_levels = []
    merge_counts = []
    final_fruit_counts = []
    danger_drop_counts = []
    settle_timeout_counts = []
    completed_action_counts = torch.zeros(
        simulator_config.action_count, dtype=torch.int64
    )
    episode_merges = torch.zeros(
        batch_size, dtype=torch.int64, device=simulator.device
    )
    episode_danger_drops = torch.zeros_like(episode_merges)
    episode_settle_timeouts = torch.zeros_like(episode_merges)
    episode_action_counts = torch.zeros(
        batch_size,
        simulator_config.action_count,
        dtype=torch.int64,
        device=simulator.device,
    )
    timeout_episodes = 0
    transitions = 0
    trajectory_count = min(
        max(0, int(trajectory_episodes)), batch_size, int(episodes)
    )
    recording = torch.zeros(
        batch_size, dtype=torch.bool, device=simulator.device
    )
    recording[:trajectory_count] = True
    trajectory_frames = []
    trajectory_outcomes = []
    started = time.perf_counter()
    while len(scores) < episodes:
        observation = simulator.observe()
        state = TensorState.from_observation(
            observation, physics_fps=physics_fps
        )
        actions = model(state).argmax(dim=1)
        recording_rows = torch.nonzero(recording, as_tuple=False).flatten()
        recorded_state = None
        recorded_actions = None
        if recording_rows.numel() > 0:
            recorded_state = _state_rows_to_cpu(state, recording_rows)
            recorded_actions = actions.index_select(
                0, recording_rows
            ).detach().cpu()
        result = simulator.step(actions)
        transitions += batch_size
        observation = result.observation
        episode_merges += result.physics.merge_events.count
        episode_danger_drops += (
            observation.over_danger_line
            | (observation.danger_progress > 0.0)
        ).to(torch.int64)
        if result.physics.settle_timeout is not None:
            episode_settle_timeouts += result.physics.settle_timeout.to(
                torch.int64
            )
        episode_action_counts.scatter_add_(
            1,
            actions.unsqueeze(1),
            torch.ones(
                batch_size,
                1,
                dtype=torch.int64,
                device=simulator.device,
            ),
        )
        time_limit = observation.step_count >= max_episode_drops
        finished = result.physics.done | time_limit
        if recording_rows.numel() > 0:
            recorded_terminal = finished.index_select(
                0, recording_rows
            )
            trajectory_frames.append({
                'environment_ids': recording_rows.detach().cpu(),
                'state': recorded_state,
                'actions': recorded_actions,
                'rewards': (
                    result.physics.score_delta.index_select(
                        0, recording_rows
                    ).to(torch.float32) / 66.0
                ).detach().cpu(),
                'terminal': recorded_terminal.detach().cpu(),
            })
            completed_rows = recording_rows[recorded_terminal]
            if completed_rows.numel() > 0:
                trajectory_outcomes.append({
                    'environment_ids': completed_rows.detach().cpu(),
                    'scores': observation.score.index_select(
                        0, completed_rows
                    ).detach().cpu(),
                    'drops': observation.step_count.index_select(
                        0, completed_rows
                    ).detach().cpu(),
                    'max_levels': observation.max_level.index_select(
                        0, completed_rows
                    ).detach().cpu(),
                    'timeout': time_limit.index_select(
                        0, completed_rows
                    ).detach().cpu(),
                })
                recording[completed_rows] = False
        if bool(finished.any().item()):
            indices = torch.nonzero(finished, as_tuple=False).flatten()
            remaining = episodes - len(scores)
            indices = indices[:remaining]
            scores.extend(
                int(value) for value in observation.score[indices].tolist()
            )
            drops.extend(
                int(value)
                for value in observation.step_count[indices].tolist()
            )
            max_levels.extend(
                int(value)
                for value in observation.max_level[indices].tolist()
            )
            merge_counts.extend(
                int(value) for value in episode_merges[indices].tolist()
            )
            final_fruit_counts.extend(
                int(value)
                for value in observation.fruit_count[indices].tolist()
            )
            danger_drop_counts.extend(
                int(value)
                for value in episode_danger_drops[indices].tolist()
            )
            settle_timeout_counts.extend(
                int(value)
                for value in episode_settle_timeouts[indices].tolist()
            )
            completed_action_counts += episode_action_counts[
                indices
            ].sum(dim=0).detach().cpu()
            timeout_episodes += int(
                (time_limit[indices] & ~result.physics.done[indices])
                .sum().item()
            )
            simulator.reset(finished)
            episode_merges[finished] = 0
            episode_danger_drops[finished] = 0
            episode_settle_timeouts[finished] = 0
            episode_action_counts[finished] = 0
    elapsed = time.perf_counter() - started
    if was_training:
        model.train()
    summary = EvaluationSummary(
        physics_fps=physics_fps,
        episodes=len(scores),
        mean_score=statistics.fmean(scores),
        median_score=statistics.median(scores),
        score_std=statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        mean_drops=statistics.fmean(drops),
        median_drops=statistics.median(drops),
        mean_max_level=statistics.fmean(max_levels),
        mean_merges=statistics.fmean(merge_counts),
        mean_final_fruit_count=statistics.fmean(final_fruit_counts),
        danger_drop_rate=(
            sum(danger_drop_counts) / max(1, sum(drops))
        ),
        settle_timeout_rate=(
            sum(settle_timeout_counts) / max(1, sum(drops))
        ),
        timeout_episodes=timeout_episodes,
        elapsed_seconds=elapsed,
        env_steps_per_second=transitions / max(elapsed, 1e-9),
    )
    details = {
        'scores': scores,
        'drops': drops,
        'max_levels': max_levels,
        'merge_counts': merge_counts,
        'final_fruit_counts': final_fruit_counts,
        'danger_drop_counts': danger_drop_counts,
        'settle_timeout_counts': settle_timeout_counts,
        'action_counts': completed_action_counts.tolist(),
        'action_distribution': (
            completed_action_counts.to(torch.float64)
            / completed_action_counts.sum().clamp_min(1)
        ).tolist(),
        'score_p10': _percentile(scores, 0.10),
        'score_p25': _percentile(scores, 0.25),
        'score_p75': _percentile(scores, 0.75),
        'score_p90': _percentile(scores, 0.90),
        'score_p95': _percentile(scores, 0.95),
        'recorded_trajectory_episodes': trajectory_count,
    }
    if trajectory_output_path is not None:
        _save_trajectory_trace(trajectory_output_path, {
            'format_version': 1,
            'physics_fps': physics_fps,
            'reward_scale': 'score_delta / 66',
            'state_semantics': 'decision_boundary_before_drop',
            'episodes': trajectory_count,
            'frames': trajectory_frames,
            'outcomes': trajectory_outcomes,
        })
    return summary, details
