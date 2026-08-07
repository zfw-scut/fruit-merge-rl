"""与训练 Replay 隔离的批量 greedy 评估和关键事件回放。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import random
import statistics
import time

import torch

from daxigua.simulator import SimulatorConfig, TensorVectorSimulator

from .observations import TensorState


_EVALUATION_SEED_STRIDE = 1_000_003


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


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _simulator_config(physics_fps, max_fruits):
    if physics_fps == 30:
        return SimulatorConfig.training_fast(max_fruits=max_fruits)
    if physics_fps == 120:
        return SimulatorConfig.high_fidelity_fast(max_fruits=max_fruits)
    raise ValueError('evaluation physics_fps must be 30 or 120')


def _episode_seeds(seed_base, episodes):
    return torch.arange(episodes, dtype=torch.int64).mul_(
        _EVALUATION_SEED_STRIDE
    ).add_(int(seed_base))


def _score_histogram(scores, bin_width):
    maximum = max(scores, default=0)
    bins = max(1, maximum // bin_width + 1)
    counts = [0] * bins
    for score in scores:
        counts[min(bins - 1, max(0, score // bin_width))] += 1
    total = max(1, len(scores))
    return {
        'bin_width': int(bin_width),
        'edges': [index * bin_width for index in range(bins + 1)],
        'counts': counts,
        'density': [count / total for count in counts],
    }


def _event_counts(events, batch_size, level_count=12):
    event_slots = events.source_levels.shape[1]
    valid = (
        torch.arange(event_slots, device=events.count.device)[None, :]
        < events.count[:, None]
    )
    source = events.source_levels.clamp(0, level_count - 1)
    created = events.new_levels.clamp(0, level_count - 1)
    source_counts = torch.zeros(
        batch_size, level_count, dtype=torch.int64, device=events.count.device
    )
    created_counts = torch.zeros_like(source_counts)
    source_counts.scatter_add_(1, source, valid.to(torch.int64))
    created_counts.scatter_add_(1, created, valid.to(torch.int64))
    peak = torch.where(
        valid,
        torch.maximum(events.source_levels, events.new_levels),
        torch.zeros_like(events.source_levels),
    ).amax(dim=1)
    return source_counts, created_counts, peak


def _final_level_counts(observation, level_count=12):
    counts = torch.zeros(
        observation.levels.shape[0],
        level_count,
        dtype=torch.int64,
        device=observation.levels.device,
    )
    counts.scatter_add_(
        1,
        observation.levels.clamp(0, level_count - 1),
        observation.active.to(torch.int64),
    )
    return counts


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
        trajectory_episodes=0,
        episode_index_output_path=None,
        critical_event_min_level=9,
        score_bin_width=500):
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError('episodes must be positive')
    simulator_config = _simulator_config(physics_fps, max_fruits)
    batch_size = min(int(parallel_envs), episodes)
    simulator = TensorVectorSimulator(
        batch_size, config=simulator_config, device=device
    )
    all_seeds = _episode_seeds(seed_base, episodes)
    episode_ids = torch.arange(
        batch_size, dtype=torch.int64, device=simulator.device
    )
    simulator.reset(seeds=all_seeds[:batch_size].to(simulator.device))
    next_episode_id = batch_size
    was_training = model.training
    model.eval()

    output_scores = torch.zeros(episodes, dtype=torch.int64)
    output_drops = torch.zeros(episodes, dtype=torch.int64)
    output_max_levels = torch.zeros(episodes, dtype=torch.int64)
    output_merge_counts = torch.zeros(episodes, dtype=torch.int64)
    output_final_counts = torch.zeros(episodes, dtype=torch.int64)
    output_danger_counts = torch.zeros(episodes, dtype=torch.int64)
    output_settle_counts = torch.zeros(episodes, dtype=torch.int64)
    output_source_counts = torch.zeros(episodes, 12, dtype=torch.int32)
    output_created_counts = torch.zeros(episodes, 12, dtype=torch.int32)
    output_final_level_counts = torch.zeros(episodes, 12, dtype=torch.int16)
    output_historical_peak = torch.zeros(episodes, dtype=torch.int16)
    output_first_high_step = torch.full((episodes,), -1, dtype=torch.int32)
    output_max_merges_per_drop = torch.zeros(episodes, dtype=torch.int16)
    output_actions = torch.full(
        (episodes, max_episode_drops), 255, dtype=torch.uint8
    )
    output_score_deltas = torch.zeros(
        episodes, max_episode_drops, dtype=torch.int16
    )
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
    episode_source_counts = torch.zeros(
        batch_size, 12, dtype=torch.int64, device=simulator.device
    )
    episode_created_counts = torch.zeros_like(episode_source_counts)
    episode_historical_peak = torch.zeros_like(episode_merges)
    episode_first_high_step = torch.full_like(episode_merges, -1)
    episode_max_merges_per_drop = torch.zeros_like(episode_merges)
    episode_actions = torch.full(
        (batch_size, max_episode_drops),
        255,
        dtype=torch.uint8,
        device=simulator.device,
    )
    episode_score_deltas = torch.zeros(
        batch_size,
        max_episode_drops,
        dtype=torch.int16,
        device=simulator.device,
    )

    timeout_episodes = 0
    completed = 0
    transitions = 0
    trajectory_count = min(
        max(0, int(trajectory_episodes)), batch_size, episodes
    )
    recording = torch.zeros(
        batch_size, dtype=torch.bool, device=simulator.device
    )
    recording[:trajectory_count] = True
    trajectory_frames = []
    trajectory_outcomes = []
    started = time.perf_counter()

    while completed < episodes:
        observation = simulator.observe()
        state = TensorState.from_observation(
            observation, physics_fps=physics_fps
        )
        actions = model(state).argmax(dim=1)
        active_rows = episode_ids >= 0
        step_indices = observation.step_count.clamp_max(
            max_episode_drops - 1
        )
        rows = torch.nonzero(active_rows, as_tuple=False).flatten()
        episode_actions[rows, step_indices[rows]] = actions[rows].to(torch.uint8)

        recording_rows = torch.nonzero(
            recording & active_rows, as_tuple=False
        ).flatten()
        recorded_state = None
        recorded_actions = None
        if recording_rows.numel() > 0:
            recorded_state = _state_rows_to_cpu(state, recording_rows)
            recorded_actions = actions.index_select(
                0, recording_rows
            ).detach().cpu()

        if bool(active_rows.all().item()) or simulator.device.type != 'cuda':
            result = simulator.step(actions)
        else:
            result = simulator.step_masked(actions, active_rows)
        transitions += int(active_rows.sum().item())
        observation = result.observation
        events = result.physics.merge_events
        event_source, event_created, event_peak = _event_counts(
            events, batch_size
        )
        episode_source_counts += event_source
        episode_created_counts += event_created
        episode_merges += events.count
        episode_historical_peak = torch.maximum(
            episode_historical_peak,
            torch.maximum(observation.max_level, event_peak),
        )
        episode_max_merges_per_drop = torch.maximum(
            episode_max_merges_per_drop, events.count
        )
        high_now = event_peak >= int(critical_event_min_level)
        episode_first_high_step = torch.where(
            (episode_first_high_step < 0) & high_now,
            observation.step_count,
            episode_first_high_step,
        )
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
            active_rows.to(torch.int64).unsqueeze(1),
        )
        episode_score_deltas[rows, step_indices[rows]] = (
            result.physics.score_delta[rows].to(torch.int16)
        )

        time_limit = observation.step_count >= max_episode_drops
        finished = active_rows & (result.physics.done | time_limit)
        if recording_rows.numel() > 0:
            recorded_terminal = finished.index_select(0, recording_rows)
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

        if not bool(finished.any().item()):
            continue
        finished_rows = torch.nonzero(finished, as_tuple=False).flatten()
        finished_ids = episode_ids[finished_rows].detach().cpu()
        destination = finished_ids.to(torch.int64)
        final_levels = _final_level_counts(observation)

        output_scores[destination] = observation.score[finished_rows].cpu()
        output_drops[destination] = observation.step_count[finished_rows].cpu()
        output_max_levels[destination] = observation.max_level[finished_rows].cpu()
        output_merge_counts[destination] = episode_merges[finished_rows].cpu()
        output_final_counts[destination] = observation.fruit_count[finished_rows].cpu()
        output_danger_counts[destination] = episode_danger_drops[finished_rows].cpu()
        output_settle_counts[destination] = episode_settle_timeouts[finished_rows].cpu()
        output_source_counts[destination] = episode_source_counts[finished_rows].to(torch.int32).cpu()
        output_created_counts[destination] = episode_created_counts[finished_rows].to(torch.int32).cpu()
        output_final_level_counts[destination] = final_levels[finished_rows].to(torch.int16).cpu()
        output_historical_peak[destination] = episode_historical_peak[finished_rows].to(torch.int16).cpu()
        output_first_high_step[destination] = episode_first_high_step[finished_rows].to(torch.int32).cpu()
        output_max_merges_per_drop[destination] = episode_max_merges_per_drop[finished_rows].to(torch.int16).cpu()
        output_actions[destination] = episode_actions[finished_rows].cpu()
        output_score_deltas[destination] = episode_score_deltas[finished_rows].cpu()
        completed_action_counts += episode_action_counts[
            finished_rows
        ].sum(dim=0).detach().cpu()
        timeout_episodes += int(
            (time_limit[finished_rows] & ~result.physics.done[finished_rows])
            .sum().item()
        )
        completed += int(finished_rows.numel())
        if completed >= episodes:
            break

        reset_count = int(finished_rows.numel())
        assign_count = min(reset_count, episodes - next_episode_id)
        replacement_ids = torch.full(
            (reset_count,), -1, dtype=torch.int64, device=simulator.device
        )
        if assign_count:
            replacement_ids[:assign_count] = torch.arange(
                next_episode_id,
                next_episode_id + assign_count,
                device=simulator.device,
            )
            reset_seeds = all_seeds[
                next_episode_id:next_episode_id + assign_count
            ].to(simulator.device)
            if assign_count < reset_count:
                reset_seeds = torch.cat((
                    reset_seeds,
                    torch.arange(
                        reset_count - assign_count,
                        device=simulator.device,
                        dtype=torch.int64,
                    ).add_(seed_base + 9_000_000_000),
                ))
            next_episode_id += assign_count
        else:
            reset_seeds = torch.arange(
                reset_count,
                device=simulator.device,
                dtype=torch.int64,
            ).add_(seed_base + 9_000_000_000)
        simulator.reset(finished, seeds=reset_seeds)
        episode_ids[finished_rows] = replacement_ids
        for tensor in (
                episode_merges,
                episode_danger_drops,
                episode_settle_timeouts,
                episode_historical_peak,
                episode_max_merges_per_drop):
            tensor[finished_rows] = 0
        episode_first_high_step[finished_rows] = -1
        episode_action_counts[finished_rows] = 0
        episode_source_counts[finished_rows] = 0
        episode_created_counts[finished_rows] = 0
        episode_actions[finished_rows] = 255
        episode_score_deltas[finished_rows] = 0

    elapsed = time.perf_counter() - started
    if was_training:
        model.train()
    scores = output_scores.tolist()
    drops = output_drops.tolist()
    max_levels = output_max_levels.tolist()
    merge_counts = output_merge_counts.tolist()
    final_fruit_counts = output_final_counts.tolist()
    danger_drop_counts = output_danger_counts.tolist()
    settle_timeout_counts = output_settle_counts.tolist()
    summary = EvaluationSummary(
        physics_fps=physics_fps,
        episodes=episodes,
        mean_score=statistics.fmean(scores),
        median_score=statistics.median(scores),
        score_std=statistics.pstdev(scores) if episodes > 1 else 0.0,
        mean_drops=statistics.fmean(drops),
        median_drops=statistics.median(drops),
        mean_max_level=statistics.fmean(max_levels),
        mean_merges=statistics.fmean(merge_counts),
        mean_final_fruit_count=statistics.fmean(final_fruit_counts),
        danger_drop_rate=sum(danger_drop_counts) / max(1, sum(drops)),
        settle_timeout_rate=sum(settle_timeout_counts) / max(1, sum(drops)),
        timeout_episodes=timeout_episodes,
        elapsed_seconds=elapsed,
        env_steps_per_second=transitions / max(elapsed, 1e-9),
    )
    l11_created = output_created_counts[:, 11]
    l11_removed = output_source_counts[:, 11]
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
        'score_histogram': _score_histogram(scores, int(score_bin_width)),
        'created_l11_episodes': int((l11_created > 0).sum().item()),
        'removed_l11_episodes': int((l11_removed > 0).sum().item()),
        'created_not_removed_l11_episodes': int(
            ((l11_created > 0) & (l11_removed == 0)).sum().item()
        ),
        'high_score_without_l11_episodes': int(
            ((output_scores >= 7000) & (l11_created == 0)).sum().item()
        ),
        'recorded_trajectory_episodes': trajectory_count,
    }
    if trajectory_output_path is not None:
        _atomic_torch_save(trajectory_output_path, {
            'format_version': 1,
            'physics_fps': physics_fps,
            'reward_scale': 'score_delta / 66',
            'state_semantics': 'decision_boundary_before_drop',
            'episodes': trajectory_count,
            'frames': trajectory_frames,
            'outcomes': trajectory_outcomes,
        })
    if episode_index_output_path is not None:
        _atomic_torch_save(episode_index_output_path, {
            'format_version': 2,
            'physics_fps': int(physics_fps),
            'seed_base': int(seed_base),
            'seed_stride': _EVALUATION_SEED_STRIDE,
            'max_episode_drops': int(max_episode_drops),
            'critical_event_min_level': int(critical_event_min_level),
            'seeds': all_seeds,
            'actions': output_actions,
            'score_deltas': output_score_deltas,
            'scores': output_scores,
            'drops': output_drops,
            'max_levels': output_max_levels,
            'historical_peak_levels': output_historical_peak,
            'merge_counts': output_merge_counts,
            'source_level_merge_counts': output_source_counts,
            'created_level_counts': output_created_counts,
            'final_level_counts': output_final_level_counts,
            'first_high_event_steps': output_first_high_step,
            'max_merges_per_drop': output_max_merges_per_drop,
        })
    return summary, details


def select_critical_episodes(index, limit, *, score_bin_width=500, seed=0):
    """从紧凑评估索引中分层选择关键事件局。"""

    scores = index['scores'].to(torch.int64)
    created = index['created_level_counts'].to(torch.int64)
    source = index['source_level_merge_counts'].to(torch.int64)
    final_counts = index['final_level_counts'].to(torch.int64)
    episode_count = int(scores.numel())
    limit = min(max(0, int(limit)), episode_count)
    selected = []
    reasons = {}

    def add(indices, reason, quota):
        candidates = [
            int(value) for value in indices
            if int(value) not in reasons
        ]
        candidates.sort(key=lambda item: (-int(scores[item]), item))
        for item in candidates[:max(0, quota)]:
            if len(selected) >= limit:
                return
            selected.append(item)
            reasons[item] = reason

    l11_created = created[:, 11] > 0
    l11_removed = source[:, 11] > 0
    duplicated_high = (final_counts[:, 8:] >= 2).any(dim=1)
    quota = max(4, limit // 8) if limit else 0
    add(torch.nonzero(l11_removed).flatten().tolist(), 'L11 已消除', quota)
    add(
        torch.nonzero(l11_created & ~l11_removed).flatten().tolist(),
        'L11 已生成但未消除',
        quota,
    )
    add(
        torch.nonzero((scores >= 10_000)).flatten().tolist(),
        '一万分以上',
        quota,
    )
    add(
        torch.nonzero((scores >= 7000) & ~l11_created).flatten().tolist(),
        '高分但未生成 L11',
        quota,
    )
    add(
        torch.nonzero(duplicated_high).flatten().tolist(),
        '终局遗留同级高阶水果',
        quota,
    )

    generator = random.Random(int(seed))
    bins = {}
    for episode_id, score in enumerate(scores.tolist()):
        if episode_id not in reasons:
            bins.setdefault(score // int(score_bin_width), []).append(episode_id)
    while len(selected) < limit and bins:
        progressed = False
        for bin_id in sorted(bins):
            candidates = bins[bin_id]
            if not candidates:
                continue
            item = candidates.pop(generator.randrange(len(candidates)))
            selected.append(item)
            reasons[item] = f'分数区间 {bin_id * score_bin_width}+'
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    selected_tensor = torch.tensor(selected, dtype=torch.int64)
    return selected_tensor, [reasons[item] for item in selected]


@torch.inference_mode()
def replay_critical_episodes(
        model,
        *,
        episode_index_path,
        output_path,
        selected_episodes,
        device,
        max_fruits=64,
        score_bin_width=500,
        selection_seed=0):
    """按记录动作复现代表局，并保存逐决策状态、Q 值和合成事件。"""

    index = torch.load(episode_index_path, map_location='cpu', weights_only=False)
    selected, reasons = select_critical_episodes(
        index,
        selected_episodes,
        score_bin_width=score_bin_width,
        seed=selection_seed,
    )
    if selected.numel() == 0:
        _atomic_torch_save(output_path, {
            'format_version': 2, 'episodes': 0, 'frames': []
        })
        return {'episodes': 0, 'mismatched_episodes': 0}

    physics_fps = int(index['physics_fps'])
    simulator = TensorVectorSimulator(
        int(selected.numel()),
        config=_simulator_config(physics_fps, max_fruits),
        device=device,
    )
    seeds = index['seeds'].index_select(0, selected).to(simulator.device)
    expected_actions = index['actions'].index_select(0, selected)
    expected_deltas = index['score_deltas'].index_select(0, selected)
    expected_drops = index['drops'].index_select(0, selected)
    expected_scores = index['scores'].index_select(0, selected)
    simulator.reset(seeds=seeds)
    was_training = model.training
    model.eval()
    active = torch.ones(
        selected.numel(), dtype=torch.bool, device=simulator.device
    )
    mismatch = torch.zeros_like(active)
    frames = []
    maximum_steps = int(expected_drops.max().item())
    for step in range(maximum_steps):
        rows = torch.nonzero(active, as_tuple=False).flatten()
        if rows.numel() == 0:
            break
        state = TensorState.from_observation(
            simulator.observe(), physics_fps=physics_fps
        )
        q_values = model(state)
        actions = torch.zeros(
            selected.numel(), dtype=torch.int64, device=simulator.device
        )
        actions[rows] = expected_actions[rows.cpu(), step].to(
            simulator.device, dtype=torch.int64
        )
        if simulator.device.type == 'cuda':
            result = simulator.step_masked(actions, active)
        else:
            result = simulator.step(actions)
        actual_delta = result.physics.score_delta.index_select(0, rows)
        wanted_delta = expected_deltas[rows.cpu(), step].to(
            simulator.device, dtype=actual_delta.dtype
        )
        mismatch[rows] |= actual_delta != wanted_delta
        events = result.physics.merge_events
        frames.append({
            'step': step,
            'selected_rows': rows.detach().cpu(),
            'state': _state_rows_to_cpu(state, rows),
            'q_values': q_values.index_select(0, rows).detach().cpu(),
            'actions': actions.index_select(0, rows).detach().cpu(),
            'score_deltas': actual_delta.detach().cpu(),
            'merge_event_count': events.count.index_select(0, rows).cpu(),
            'merge_source_levels': events.source_levels.index_select(0, rows).cpu(),
            'merge_new_levels': events.new_levels.index_select(0, rows).cpu(),
            'merge_positions': events.positions.index_select(0, rows).cpu(),
            'merge_source_ids': events.source_ids.index_select(0, rows).cpu(),
            'merge_new_fruit_ids': events.new_fruit_ids.index_select(0, rows).cpu(),
        })
        should_finish = (
            torch.tensor(step + 1, device=simulator.device)
            >= expected_drops.to(simulator.device)
        )
        mismatch |= active & result.physics.done & ~should_finish
        active &= ~should_finish

    final_scores = simulator.observe().score.detach().cpu()
    mismatch |= (final_scores != expected_scores).to(simulator.device)
    if was_training:
        model.train()
    payload = {
        'format_version': 2,
        'physics_fps': physics_fps,
        'state_semantics': 'decision_boundary_before_recorded_drop',
        'episodes': int(selected.numel()),
        'episode_indices': selected,
        'selection_reasons': reasons,
        'seeds': seeds.detach().cpu(),
        'expected_scores': expected_scores,
        'replayed_scores': final_scores,
        'expected_drops': expected_drops,
        'replay_mismatch': mismatch.detach().cpu(),
        'frames': frames,
    }
    _atomic_torch_save(output_path, payload)
    return {
        'episodes': int(selected.numel()),
        'mismatched_episodes': int(mismatch.sum().item()),
    }
