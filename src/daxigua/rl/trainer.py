"""第一版 GNN-DQN 的单进程 GPU 正式训练主链。"""

from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time

import torch

from daxigua.simulator import (
    SimulatorConfig,
    SpatialRewardComputer,
    SpatialRewardConfig,
    TensorVectorSimulator,
)

from .checkpoint import (
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
    write_artifact_manifest,
)
from .autoscale import AdaptiveScaleController
from .action_effects import build_action_effect_targets
from .config import TrainingConfig
from .evaluation import evaluate_policy, replay_critical_episodes
from .event_analysis import render_evaluation_event_analysis
from .decision_data import ActionSelectionBatch
from .key_decisions import KeyDecisionCollector
from .learner import DqnLearner
from .model import BaselineGnnDqn
from .monitoring import DashboardPublisher, ResourceSampler
from .observations import TensorState
from .replay import GpuReplayBuffer


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def epsilon_at_transition(dqn_config, transition, total_transitions):
    """按显式折线或兼容旧配置的线性计划计算探索率。"""

    transition = max(0, int(transition))
    schedule = dqn_config.epsilon_schedule
    if schedule:
        if transition >= schedule[-1][0]:
            return schedule[-1][1]
        for (left_step, left_value), (right_step, right_value) in zip(
                schedule, schedule[1:]):
            if left_step <= transition <= right_step:
                progress = (
                    (transition - left_step) / (right_step - left_step)
                )
                return left_value + progress * (right_value - left_value)
        return schedule[0][1]
    decay_steps = max(
        1,
        int(total_transitions * dqn_config.epsilon_decay_fraction),
    )
    progress = min(1.0, transition / decay_steps)
    return (
        dqn_config.epsilon_start
        + progress
        * (dqn_config.epsilon_end - dqn_config.epsilon_start)
    )


def active_learning_probability(dqn_config, epsilon):
    """按 epsilon 所处阶段给出独立的主动学习分支概率。"""

    epsilon = float(epsilon)
    start = dqn_config.active_learning_start_epsilon
    full = dqn_config.active_learning_full_epsilon
    maximum = dqn_config.active_learning_max_probability
    if epsilon >= start:
        return 0.0
    if epsilon <= full or start == full:
        return maximum
    return maximum * (start - epsilon) / (start - full)


def _descending_ordinal_ranks(values):
    """逐行生成从 1 开始的降序名次；并列时动作下标小者优先。"""

    order = torch.argsort(values, dim=1, descending=True, stable=True)
    positions = torch.arange(
        1, values.shape[1] + 1, dtype=torch.int64, device=values.device
    ).expand_as(order)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, positions)
    return ranks


def ranked_active_learning_candidates(q_values, uncertainty, top_k):
    """返回排名融合后的前 K 动作及两种原始名次。"""

    action_count = q_values.shape[1]
    if not 0 < int(top_k) <= action_count:
        raise ValueError('top_k must be in [1, action_count]')
    value_ranks = _descending_ordinal_ranks(q_values.float())
    uncertainty_ranks = _descending_ordinal_ranks(uncertainty.float())
    combined = value_ranks + uncertainty_ranks
    worst_rank = torch.maximum(value_ranks, uncertainty_ranks)
    action_indices = torch.arange(
        action_count, dtype=torch.int64, device=q_values.device
    ).expand_as(value_ranks)
    base = action_count + 1
    # 整数进位只负责精确实现约定的字典序，不引入可调权重或量纲。
    ordering_key = (
        ((combined * base + worst_rank) * base + value_ranks) * base
        + action_indices
    )
    candidates = torch.argsort(
        ordering_key, dim=1, descending=False, stable=True
    )[:, :int(top_k)]
    return candidates, value_ranks, uncertainty_ranks


def rank_correlation_from_sums(values):
    """从 n,sum(x),sum(y),sum(x²),sum(y²),sum(xy) 计算相关系数。"""

    count, sum_x, sum_y, sum_x2, sum_y2, sum_xy = values
    if count < 2.0:
        return None
    numerator = count * sum_xy - sum_x * sum_y
    denominator_squared = (
        (count * sum_x2 - sum_x * sum_x)
        * (count * sum_y2 - sum_y * sum_y)
    )
    if denominator_squared <= 0.0:
        return None
    return max(-1.0, min(1.0, numerator / math.sqrt(denominator_squared)))


def _git_identity(project_root):
    def run(*arguments):
        try:
            return subprocess.check_output(
                ('git',) + arguments,
                cwd=project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        'commit': run('rev-parse', 'HEAD'),
        'branch': run('branch', '--show-current'),
        'dirty': bool(run('status', '--short')),
    }


def _lower_unreachable_prewarm_targets(targets, survived_drops, failed):
    """把终止轨迹的预投放目标降到其最后一个可存活投放数。"""

    reachable = (survived_drops.to(targets.dtype) - 1).clamp_min(0)
    return torch.where(failed, torch.minimum(targets, reachable), targets)


def _bounded_stage_thresholds(quantiles, pilot_max_drops):
    """把删失局长分位限制在统计窗口的 25%/50%/75%。"""

    caps = tuple(
        max(1, pilot_max_drops * numerator // 4)
        for numerator in (1, 2, 3)
    )
    return tuple(
        max(1, min(int(value), cap))
        for value, cap in zip(quantiles, caps, strict=True)
    )


class _TensorMetricAccumulator:
    def __init__(self):
        self.sums = {}
        self.count = 0

    def add(self, metrics):
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                detached = value.detach().float()
                self.sums[name] = self.sums.get(name, 0.0) + detached
        self.count += 1

    def flush(self):
        if self.count == 0:
            return {}
        values = {
            name: float((total / self.count).item())
            for name, total in self.sums.items()
        }
        self.sums.clear()
        self.count = 0
        return values


class BaselineTrainer:
    """30 FPS 独占训练；评估模拟器从不写入 Replay。"""

    CHECKPOINT_MILESTONES = (
        1_000_000,
        5_000_000,
        10_000_000,
        16_000_000,
        20_000_000,
        24_000_000,
        25_000_000,
        50_000_000,
        100_000_000,
    )

    def __init__(
            self,
            config,
            *,
            project_root=None,
            decision_selector=None,
            decision_sinks=()):
        if not isinstance(config, TrainingConfig):
            raise TypeError('config must be TrainingConfig')
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available for formal training')
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type != 'cuda' and config.active_envs != config.max_envs:
            raise ValueError('CPU smoke requires active_envs == max_envs')

        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.run_dir = Path(config.run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.run_dir / 'evaluations').mkdir(exist_ok=True)
        (self.run_dir / 'analysis').mkdir(exist_ok=True)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

        self.simulator_config = SimulatorConfig.training_fast(
            max_fruits=config.model.max_fruits,
            action_count=config.model.action_count,
            queue_length=config.model.queue_length,
            use_cuda_extension=self.device.type == 'cuda',
            track_action_effects=config.model.action_effect_enabled,
        )
        self.simulator = TensorVectorSimulator(
            config.max_envs,
            config=self.simulator_config,
            device=self.device,
        )
        self.reward_computer = (
            SpatialRewardComputer(
                self.simulator_config,
                device=self.device,
                reward_config=SpatialRewardConfig(
                    queue_weights=config.reward.queue_weights,
                    reward_scale=config.reward.reward_scale,
                    reference_mode=(
                        'best_no_merge'
                        if config.reward.kind == 'spatial_v2_1'
                        else 'empty_average'
                    ),
                ),
            )
            if config.reward.kind in ('spatial_v2', 'spatial_v2_1')
            else None
        )
        self.active_envs = config.active_envs
        base_model = BaselineGnnDqn(
            config.model,
            board_width=self.simulator_config.board_width,
            board_height=self.simulator_config.board_height,
            spawn_y=self.simulator_config.spawn_y,
            wall_width=self.simulator_config.wall_width,
            gravity_y=self.simulator_config.gravity_y,
        ).to(self.device)
        self.learner = DqnLearner(base_model, config.dqn)
        self.model = self.learner.online_model
        decision_collection_active = bool(
            config.decision_data.enabled
            and decision_selector is not None
            and getattr(decision_selector, 'active', True)
        )
        self.replay = GpuReplayBuffer(
            config.replay.capacity,
            max_fruits=config.model.max_fruits,
            queue_length=config.model.queue_length,
            device=self.device,
            physics_fps=self.simulator_config.physics_fps,
            seed=config.seed + 1,
            enable_state_references=decision_collection_active,
            policy_head_count=config.model.policy_head_count,
            bootstrap_probability=config.dqn.bootstrap_probability,
            action_effects_enabled=config.model.action_effect_enabled,
        )
        self.branch_simulator = None
        self.branch_replay = None
        self.branch_generator = None
        self.branch_replay_training_threshold = 0
        if config.branch_learning.enabled:
            branch = config.branch_learning
            self.branch_generator = torch.Generator(device=self.device)
            self.branch_generator.manual_seed(config.seed + 3)
            self.branch_replay_training_threshold = branch.replay_warmup
            self.branch_simulator = TensorVectorSimulator(
                branch.simulator_batch_size,
                config=self.simulator_config,
                device=self.device,
            )
            self.branch_replay = GpuReplayBuffer(
                branch.replay_capacity,
                max_fruits=config.model.max_fruits,
                queue_length=config.model.queue_length,
                device=self.device,
                physics_fps=self.simulator_config.physics_fps,
                seed=config.seed + 2,
                policy_head_count=config.model.policy_head_count,
                bootstrap_probability=config.dqn.bootstrap_probability,
                action_effects_enabled=True,
            )
        self.key_decision_collector = KeyDecisionCollector(
            config.decision_data,
            replay=self.replay,
            simulator=self.simulator,
            run_dir=self.run_dir,
            selector=decision_selector,
            extra_sinks=decision_sinks,
        )
        self.dashboard = DashboardPublisher(config.dashboard, self.run_dir)
        self.resource_sampler = ResourceSampler(os.getpid())
        self.scale_controller = AdaptiveScaleController(
            config.autoscale,
            initial_envs=config.active_envs,
            maximum_envs=config.max_envs,
        )
        self.training_metrics = _TensorMetricAccumulator()
        self.reward_metrics = _TensorMetricAccumulator()
        self.recent_scores = deque(maxlen=4096)
        self.recent_drops = deque(maxlen=4096)
        self.metric_window_scores = []
        self.metric_window_drops = []
        self.transitions = 0
        self.branch_transitions = 0
        self.branch_source_states = 0
        self.simulated_transitions = 0
        self.episodes = 0
        self.update_credit = 0.0
        self.best_accurate_score = float('-inf')
        self.last_fast_eval_score = None
        self.last_accurate_eval_score = None
        self.last_eval_created_density = [None] * 12
        self.best_training_score = 0
        self.completed_accurate_milestones = set()
        self.completed_checkpoint_milestones = set()
        self.action_counts = torch.zeros(
            config.model.action_count,
            dtype=torch.int64,
            device=self.device,
        )
        self.actor_metric_sums = torch.zeros(
            3, dtype=torch.float32, device=self.device
        )
        self.actor_rank_correlation_sums = torch.zeros(
            6, dtype=torch.float32, device=self.device
        )
        self.actor_metric_decisions = 0
        self.stage_thresholds = (16, 64, 128)
        self.stop_requested = False
        self.stop_reason = None
        self._write_run_identity()
        self._write_run_status(
            'initializing', '训练进程已创建，正在初始化'
        )

    def request_stop(self, reason='requested'):
        self.stop_requested = True
        self.stop_reason = str(reason)
        self.dashboard.event(
            'stop_requested', '收到安全停止请求', reason=self.stop_reason
        )

    def _write_run_identity(self):
        identity = {
            'training_config': self.config.to_dict(),
            'git': _git_identity(self.project_root),
            'torch_version': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'device': str(self.device),
            'device_name': (
                torch.cuda.get_device_name(self.device)
                if self.device.type == 'cuda'
                else 'cpu'
            ),
            'pid': os.getpid(),
            'created_at': time.time(),
        }
        (self.run_dir / 'run_identity.json').write_text(
            json.dumps(identity, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    def _write_run_status(self, phase, message, **values):
        path = self.run_dir / 'run_status.json'
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        payload = {
            'phase': str(phase),
            'completion_message': str(message),
            'status_timestamp': time.time(),
            **self._progress(),
            **values,
        }
        temporary.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        os.replace(temporary, path)
        return payload

    def _active_rows(self):
        return torch.arange(
            self.active_envs, dtype=torch.int64, device=self.device
        )

    def _enabled_mask(self, extra=None):
        mask = torch.zeros(
            self.config.max_envs, dtype=torch.bool, device=self.device
        )
        mask[:self.active_envs] = True
        if extra is not None:
            mask &= extra
        return mask

    def _step(self, actions, enabled_mask=None):
        if enabled_mask is None or bool(enabled_mask.all().item()):
            return self.simulator.step(actions)
        if self.device.type != 'cuda':
            raise RuntimeError('masked stepping is only available on CUDA')
        return self.simulator.step_masked(actions, enabled_mask)

    def _random_actions(self):
        return torch.randint(
            self.config.model.action_count,
            (self.config.max_envs,),
            device=self.device,
        )

    def _reset_finished(self, finished):
        if bool(finished.any().item()):
            self.key_decision_collector.on_env_reset(finished)
            observation = self.simulator.reset(finished)
            if self.reward_computer is not None:
                self.reward_computer.reset_rows(finished, observation)

    def _initialize_reward(self):
        if self.reward_computer is not None:
            self.reward_computer.initialize(self.simulator.observe())

    def _compute_rewards(self, result, *, record_metrics=False):
        if self.reward_computer is None:
            return result.physics.score_delta[:self.active_envs].to(
                torch.float32
            ) / self.config.reward.score_divisor
        reward_step = self.reward_computer.step(
            result, batch_size=self.active_envs
        )
        if record_metrics:
            self.reward_metrics.add(reward_step.scalar_metrics())
        return reward_step.reward

    def _build_action_effect_targets(
            self,
            current,
            next_state,
            result,
            current_fruit_count,
            current_danger_progress):
        if not self.config.model.action_effect_enabled:
            return None
        return build_action_effect_targets(
            current,
            next_state,
            result,
            board_width=self.simulator_config.board_width,
            board_height=self.simulator_config.board_height,
            gravity_y=self.simulator_config.gravity_y,
            max_physics_frames=self.simulator_config.max_physics_frames,
            current_fruit_count=current_fruit_count,
            current_danger_progress=current_danger_progress,
        )

    @torch.no_grad()
    def estimate_stage_thresholds(self):
        if self.device.type != 'cuda':
            self.stage_thresholds = (8, 16, 32)
            return self.stage_thresholds
        pilot_envs = min(self.active_envs, self.config.stage_pilot_envs)
        pilot_max_drops = min(
            self.config.max_episode_drops,
            self.config.stage_pilot_max_drops,
        )
        pilot_mask = torch.zeros(
            self.config.max_envs, dtype=torch.bool, device=self.device
        )
        pilot_mask[:pilot_envs] = True
        self.dashboard.event(
            'pilot_started',
            '开始随机局长阶段标定',
            pilot_envs=pilot_envs,
            active_envs=self.active_envs,
            pilot_max_drops=pilot_max_drops,
        )
        self.simulator.reset(pilot_mask, seeds=self.config.seed + 100)
        finished = torch.zeros(
            self.config.max_envs, dtype=torch.bool, device=self.device
        )
        lengths = torch.zeros(
            self.config.max_envs, dtype=torch.int64, device=self.device
        )
        for _ in range(pilot_max_drops):
            enabled = pilot_mask & ~finished
            if not bool(enabled.any().item()):
                break
            result = self._step(self._random_actions(), enabled)
            time_limit = result.observation.step_count >= self.config.max_episode_drops
            newly_finished = enabled & (result.physics.done | time_limit)
            lengths[newly_finished] = result.observation.step_count[newly_finished]
            finished |= newly_finished
        active_lengths = lengths[:pilot_envs].to(torch.float32)
        active_lengths = torch.where(
            active_lengths > 0,
            active_lengths,
            torch.full_like(active_lengths, pilot_max_drops),
        )
        quantiles = torch.quantile(
            active_lengths,
            torch.tensor((0.25, 0.50, 0.75), device=self.device),
        ).to(torch.int64)
        self.stage_thresholds = _bounded_stage_thresholds(
            quantiles.tolist(), pilot_max_drops
        )
        self.simulator.reset(self._enabled_mask(), seeds=self.config.seed + 200)
        self.dashboard.event(
            'pilot_finished',
            '随机局长阶段标定完成',
            thresholds=self.stage_thresholds,
            pilot_envs=pilot_envs,
            pilot_max_drops=pilot_max_drops,
        )
        return self.stage_thresholds

    @torch.no_grad()
    def stagger_initial_states(self):
        if self.device.type != 'cuda':
            return
        q1, q2, q3 = self.stage_thresholds
        upper = max(q3 + 1, min(self.config.max_episode_drops, q3 * 2))
        bounds = ((0, q1), (q1, q2), (q2, q3), (q3, upper))
        targets = torch.zeros(
            self.config.max_envs, dtype=torch.int64, device=self.device
        )
        for stage, (low, high) in enumerate(bounds):
            if stage >= self.active_envs:
                continue
            rows = torch.arange(
                stage, self.active_envs, 4, device=self.device
            )
            if rows.numel() == 0:
                continue
            high = max(low + 1, high)
            targets[rows] = torch.randint(
                low, high, (rows.numel(),), device=self.device
            )
        original_targets = targets.clone()
        for _ in range(self.config.max_episode_drops * 3):
            pending = self._enabled_mask(
                self.simulator.step_count < targets
            )
            if not bool(pending.any().item()):
                adjusted = int(
                    (targets[:self.active_envs]
                     != original_targets[:self.active_envs]).sum().item()
                )
                self.dashboard.event(
                    'warmup_states_staggered',
                    '分散预热起始状态构造完成',
                    adjusted_unreachable_targets=adjusted,
                )
                return
            result = self._step(self._random_actions(), pending)
            failed = pending & result.physics.done
            targets = _lower_unreachable_prewarm_targets(
                targets,
                result.observation.step_count,
                failed,
            )
            self._reset_finished(failed)
        raise RuntimeError('could not construct dispersed warmup states')

    def _classify_stages(self, observation):
        rows = self._active_rows()
        step_count = observation.step_count.index_select(0, rows)
        danger = (
            observation.over_danger_line.index_select(0, rows)
            | (observation.danger_progress.index_select(0, rows) > 0)
        )
        q1, q2, _q3 = self.stage_thresholds
        stages = torch.zeros_like(step_count)
        stages = torch.where(step_count >= q1, torch.ones_like(stages), stages)
        stages = torch.where(step_count >= q2, torch.full_like(stages, 2), stages)
        stages = torch.where(danger, torch.full_like(stages, 3), stages)
        return stages

    @torch.no_grad()
    def fill_warmup_replay(self):
        target = self.config.replay.warmup_transitions
        ratios = self.config.replay.warmup_stage_ratios
        quotas = [int(target * ratio) for ratio in ratios]
        quotas[-1] += target - sum(quotas)
        counts = [0, 0, 0, 0]
        self.dashboard.event('warmup_started', '开始分阶段预热 Replay')
        self._initialize_reward()
        max_rounds = max(
            self.config.max_episode_drops * 20,
            math.ceil(target / self.active_envs) * 100,
        )
        for _ in range(max_rounds):
            if sum(counts) >= target:
                break
            observation = self.simulator.observe()
            current = TensorState.from_observation(
                observation,
                physics_fps=self.simulator_config.physics_fps,
                rows=self._active_rows(),
            )
            current_fruit_count = current.active.sum(dim=1)
            current_danger_progress = current.danger_progress.clone()
            stages = self._classify_stages(observation)
            admit = torch.zeros(
                self.active_envs, dtype=torch.bool, device=self.device
            )
            for stage in range(4):
                remaining = quotas[stage] - counts[stage]
                if remaining <= 0:
                    continue
                candidates = torch.nonzero(
                    stages == stage, as_tuple=False
                ).flatten()[:remaining]
                admit[candidates] = True
                counts[stage] += int(candidates.numel())
            ticket = self.replay.begin_append(current, mask=admit) \
                if bool(admit.any().item()) else None
            full_actions = self._random_actions()
            result = self._step(full_actions, self._enabled_mask())
            self.simulated_transitions += self.active_envs
            next_state = TensorState.from_observation(
                result.observation,
                physics_fps=self.simulator_config.physics_fps,
                rows=self._active_rows(),
            )
            active_actions = full_actions[:self.active_envs]
            rewards = self._compute_rewards(result)
            terminals = result.physics.done[:self.active_envs]
            if ticket is not None:
                action_effects = self._build_action_effect_targets(
                    current,
                    next_state,
                    result,
                    current_fruit_count,
                    current_danger_progress,
                )
                self.replay.finish_append(
                    ticket,
                    next_state,
                    active_actions,
                    rewards,
                    terminals,
                    stages,
                    action_effects,
                )
            timeout = (
                result.observation.step_count >= self.config.max_episode_drops
            )
            self._reset_finished(
                self._enabled_mask() & (result.physics.done | timeout)
            )
        if sum(counts) < target:
            raise RuntimeError(
                f'warmup stage quotas were not filled: {counts} / {quotas}'
            )
        if self.transitions == 0:
            self.transitions = len(self.replay)
        self.dashboard.event(
            'warmup_finished',
            '分阶段预热 Replay 完成',
            counts=counts,
            replay_size=len(self.replay),
        )

    def epsilon(self):
        return epsilon_at_transition(
            self.config.dqn,
            self.transitions,
            self.config.total_transitions,
        )

    @torch.no_grad()
    def _select_action_batch(self, state, epsilon):
        self.model.eval()
        output = self.model(state, True)
        q_values = output.q_values
        centered_heads = output.head_q_values - output.head_q_values.mean(
            dim=2, keepdim=True
        )
        uncertainty = centered_heads.float().std(dim=1, correction=0)
        greedy = q_values.argmax(dim=1)
        random_actions = torch.randint(
            self.config.model.action_count,
            (state.batch_size,),
            device=self.device,
        )
        active_probability = 0.0
        if (
                self.config.dqn.active_learning_enabled
                and self.config.model.policy_head_count > 1):
            active_probability = active_learning_probability(
                self.config.dqn, epsilon
            )
        branch_draw = torch.rand(state.batch_size, device=self.device)
        active_learning = branch_draw < active_probability
        explore = (
            (branch_draw >= active_probability)
            & (branch_draw < active_probability + epsilon)
        )
        selected_value_ranks = torch.zeros(
            state.batch_size, dtype=torch.int64, device=self.device
        )
        selected_uncertainty_ranks = torch.zeros_like(selected_value_ranks)
        active_actions = greedy
        if active_probability > 0.0:
            candidates, value_ranks, uncertainty_ranks = (
                ranked_active_learning_candidates(
                    q_values,
                    uncertainty,
                    self.config.dqn.active_learning_top_k,
                )
            )
            candidate_offsets = torch.randint(
                candidates.shape[1],
                (state.batch_size,),
                device=self.device,
            )
            active_actions = candidates.gather(
                1, candidate_offsets.unsqueeze(1)
            ).squeeze(1)
            selected_value_ranks = value_ranks.gather(
                1, active_actions.unsqueeze(1)
            ).squeeze(1)
            selected_uncertainty_ranks = uncertainty_ranks.gather(
                1, active_actions.unsqueeze(1)
            ).squeeze(1)
        actions = torch.where(explore, random_actions, greedy)
        actions = torch.where(active_learning, active_actions, actions)
        return ActionSelectionBatch(
            actions=actions,
            greedy_actions=greedy,
            explore_mask=explore,
            q_values=q_values,
            uncertainty=uncertainty,
            active_learning_mask=active_learning,
            selected_value_ranks=selected_value_ranks,
            selected_uncertainty_ranks=selected_uncertainty_ranks,
        )

    @torch.no_grad()
    def _select_actions(self, state, epsilon):
        """保留现有内部/测试调用的动作 Tensor 兼容接口。"""

        return self._select_action_batch(state, epsilon).actions

    def _branch_target_at(self, parent_transitions):
        branch = self.config.branch_learning
        if not branch.enabled:
            return 0
        horizon = self.config.total_transitions - branch.start_transition
        eligible = min(
            horizon,
            max(0, int(parent_transitions) - branch.start_transition),
        )
        raw_target = branch.transition_budget * eligible // horizon
        return (
            raw_target // branch.simulator_batch_size
            * branch.simulator_batch_size
        )

    @torch.no_grad()
    def _collect_branch_transitions(
            self,
            *,
            current_stages,
            action_selection,
            parent_transitions_after_step):
        """从未修改的父状态旁路执行一次主动动作并写入独立 Replay。"""

        if self.branch_simulator is None or self.branch_replay is None:
            return 0
        branch = self.config.branch_learning
        target = self._branch_target_at(parent_transitions_after_step)
        pending = target - self.branch_transitions
        if pending <= 0:
            return 0
        if pending % branch.simulator_batch_size != 0:
            raise RuntimeError('branch scheduler produced a partial simulator batch')
        source_count = (
            branch.simulator_batch_size // branch.actions_per_state
        )
        if source_count > self.active_envs:
            raise RuntimeError('branch simulator requires too many parent states')
        pending_source_count = pending // branch.actions_per_state
        if pending_source_count > self.active_envs:
            raise RuntimeError(
                'branch schedule selected one parent state more than once '
                'inside a decision batch'
            )
        pending_source_rows = torch.randperm(
            self.active_envs,
            device=self.device,
            generator=self.branch_generator,
        )[:pending_source_count]
        destination_rows = torch.arange(
            branch.simulator_batch_size,
            dtype=torch.int64,
            device=self.device,
        )
        produced = 0
        while produced < pending:
            source_start = produced // branch.actions_per_state
            source_rows = pending_source_rows[
                source_start:source_start + source_count
            ]
            parent_actions = action_selection.actions.index_select(
                0, source_rows
            )
            ranked, _value_ranks, _uncertainty_ranks = (
                ranked_active_learning_candidates(
                    action_selection.q_values.index_select(0, source_rows),
                    action_selection.uncertainty.index_select(0, source_rows),
                    self.config.model.action_count,
                )
            )
            alternatives = ranked[
                ranked.ne(parent_actions.unsqueeze(1))
            ].view(source_count, self.config.model.action_count - 1)
            selected_actions = alternatives[
                :, :branch.actions_per_state
            ]
            selected_value_ranks = _value_ranks.gather(
                1, selected_actions
            ).reshape(-1).to(torch.float32)
            selected_uncertainty_ranks = _uncertainty_ranks.gather(
                1, selected_actions
            ).reshape(-1).to(torch.float32)
            self.actor_rank_correlation_sums += torch.stack((
                torch.ones_like(selected_value_ranks),
                selected_value_ranks,
                selected_uncertainty_ranks,
                selected_value_ranks.square(),
                selected_uncertainty_ranks.square(),
                selected_value_ranks * selected_uncertainty_ranks,
            )).sum(dim=1)
            expanded_sources = source_rows.unsqueeze(1).expand(
                -1, branch.actions_per_state
            ).reshape(-1)
            branch_actions = selected_actions.reshape(-1)
            self.branch_simulator.copy_rows_from(
                self.simulator,
                expanded_sources,
                destination_rows,
                validate_rows=False,
            )
            branch_observation = self.branch_simulator.observe()
            branch_current = TensorState.from_observation(
                branch_observation,
                physics_fps=self.simulator_config.physics_fps,
                clone=True,
            )
            current_fruit_count = branch_current.active.sum(dim=1)
            current_danger_progress = branch_current.danger_progress.clone()
            ticket = self.branch_replay.begin_append(branch_current)
            result = self.branch_simulator.step(branch_actions)
            branch_next = TensorState.from_observation(
                result.observation,
                physics_fps=self.simulator_config.physics_fps,
            )
            rewards = result.physics.score_delta.to(
                torch.float32
            ) / self.config.reward.score_divisor
            stages = current_stages.index_select(
                0, source_rows
            ).unsqueeze(1).expand(
                -1, branch.actions_per_state
            ).reshape(-1)
            action_effects = self._build_action_effect_targets(
                branch_current,
                branch_next,
                result,
                current_fruit_count,
                current_danger_progress,
            )
            self.branch_replay.finish_append(
                ticket,
                branch_next,
                branch_actions,
                rewards,
                result.physics.done,
                stages,
                action_effects,
            )
            produced += branch.simulator_batch_size
            self.branch_transitions += branch.simulator_batch_size
            self.branch_source_states += source_count
            self.simulated_transitions += branch.simulator_batch_size
        return produced

    @torch.no_grad()
    def _accumulate_actor_metrics(self, selection):
        active = selection.active_learning_mask
        active_changed = active & selection.actions.ne(
            selection.greedy_actions
        )
        self.actor_metric_sums[0] += active.sum()
        self.actor_metric_sums[1] += selection.explore_mask.sum()
        self.actor_metric_sums[2] += active_changed.sum()
        if (
                selection.selected_value_ranks is not None
                and selection.selected_uncertainty_ranks is not None):
            mask = active.to(torch.float32)
            value_rank = selection.selected_value_ranks.to(torch.float32)
            uncertainty_rank = (
                selection.selected_uncertainty_ranks.to(torch.float32)
            )
            self.actor_rank_correlation_sums += torch.stack((
                mask,
                mask * value_rank,
                mask * uncertainty_rank,
                mask * value_rank.square(),
                mask * uncertainty_rank.square(),
                mask * value_rank * uncertainty_rank,
            )).sum(dim=1)
        self.actor_metric_decisions += selection.batch_size

    def _episode_finished(self, result):
        return (
            result.physics.done[:self.active_envs]
            | (
                result.observation.step_count[:self.active_envs]
                >= self.config.max_episode_drops
            )
        )

    def _record_episodes(self, result):
        finished = self._episode_finished(result)
        if bool(finished.any().item()):
            finished_scores = [
                int(value)
                for value in result.observation.score[:self.active_envs][finished]
                .tolist()
            ]
            finished_drops = [
                int(value)
                for value in result.observation.step_count[:self.active_envs][finished]
                .tolist()
            ]
            self.recent_scores.extend(finished_scores)
            self.recent_drops.extend(finished_drops)
            self.metric_window_scores.extend(finished_scores)
            self.metric_window_drops.extend(finished_drops)
            self.best_training_score = max(
                self.best_training_score, max(finished_scores)
            )
            self.episodes += int(finished.sum().item())
        full_finished = torch.zeros(
            self.config.max_envs, dtype=torch.bool, device=self.device
        )
        full_finished[:self.active_envs] = finished
        self._reset_finished(full_finished)

    @torch.no_grad()
    def _set_active_envs(self, target_envs):
        target_envs = int(target_envs)
        if not 0 < target_envs <= self.config.max_envs:
            raise ValueError('autoscale target is outside allocated environments')
        if target_envs <= self.active_envs:
            self.active_envs = target_envs
            return
        if self.device.type != 'cuda':
            raise RuntimeError('dynamic environment scaling requires CUDA')
        previous = self.active_envs
        new_rows = torch.zeros(
            self.config.max_envs, dtype=torch.bool, device=self.device
        )
        new_rows[previous:target_envs] = True
        self.simulator.reset(
            new_rows,
            seeds=self.config.seed + self.transitions + target_envs,
        )
        self.key_decision_collector.on_env_reset(new_rows)
        targets = torch.zeros(
            self.config.max_envs, dtype=torch.int64, device=self.device
        )
        upper = max(1, self.stage_thresholds[1])
        targets[previous:target_envs] = torch.randint(
            0,
            upper,
            (target_envs - previous,),
            device=self.device,
        )
        for _ in range(max(1, upper * 3)):
            pending = new_rows & (self.simulator.step_count < targets)
            if not bool(pending.any().item()):
                self.active_envs = target_envs
                self._initialize_reward()
                return
            result = self._step(self._random_actions(), pending)
            failed = pending & result.physics.done
            self._reset_finished(failed)
            self.simulated_transitions += int(pending.sum().item())
        raise RuntimeError('autoscale pre-roll did not reach its target stages')

    def _progress(self):
        branch_budget = self.config.branch_learning.transition_budget
        return {
            'transitions': self.transitions,
            'total_transitions': self.config.total_transitions,
            'progress_fraction': min(
                1.0, self.transitions / self.config.total_transitions
            ),
            'simulated_transitions': self.simulated_transitions,
            'branch_transitions': self.branch_transitions,
            'branch_total_transitions': branch_budget,
            'branch_progress_fraction': (
                0.0
                if branch_budget <= 0
                else min(1.0, self.branch_transitions / branch_budget)
            ),
            'branch_source_states': self.branch_source_states,
            'episodes': self.episodes,
            'updates': self.learner.update_count,
            'active_envs': self.active_envs,
            'best_accurate_score': self.best_accurate_score,
            'last_fast_eval_score': self.last_fast_eval_score,
            'last_accurate_eval_score': self.last_accurate_eval_score,
            'best_training_score': self.best_training_score,
        }

    def save_checkpoint(self, name, *, extra=None):
        path = self.run_dir / 'checkpoints' / name
        return save_checkpoint_atomic(
            path,
            learner=self.learner,
            training_config=self.config,
            progress=self._progress(),
            replay_metadata=self.replay.metadata(),
            extra={
                'replay_resume_policy': 'rebuild',
                'branch_replay_resume_policy': (
                    'rebuild_from_future_branches'
                ),
                'branch_generator_state': (
                    None
                    if self.branch_generator is None
                    else self.branch_generator.get_state()
                ),
                'branch_replay_metadata': (
                    None
                    if self.branch_replay is None
                    else self.branch_replay.metadata()
                ),
                'decision_data': self.key_decision_collector.metrics(),
                **(extra or {}),
            },
        )

    def resume(self, checkpoint_path):
        checkpoint = load_checkpoint(
            checkpoint_path, map_location=self.device
        )
        self.learner.load_state_dict(checkpoint['learner'])
        restore_rng_state(checkpoint['rng_state'])
        progress = checkpoint['progress']
        self.transitions = int(progress['transitions'])
        self.branch_transitions = int(
            progress.get('branch_transitions', 0)
        )
        self.branch_source_states = int(
            progress.get('branch_source_states', 0)
        )
        if self.branch_replay is not None:
            self.branch_replay_training_threshold = (
                self.config.branch_learning.learner_batch_size
            )
            branch_generator_state = checkpoint.get('extra', {}).get(
                'branch_generator_state'
            )
            if branch_generator_state is not None:
                self.branch_generator.set_state(
                    branch_generator_state.detach().cpu()
                )
        self.simulated_transitions = int(progress['simulated_transitions'])
        self.episodes = int(progress['episodes'])
        self.best_accurate_score = float(progress['best_accurate_score'])
        self.last_fast_eval_score = progress.get('last_fast_eval_score')
        self.last_accurate_eval_score = progress.get('last_accurate_eval_score')
        self.best_training_score = int(progress.get('best_training_score', 0))
        self.dashboard.event(
            'resumed',
            '已恢复模型与优化器；主 Replay 重新预热，主动 Replay 从后续旁路样本重建',
            branch_replay_training_threshold=(
                self.branch_replay_training_threshold
            ),
        )

    def _append_jsonl(self, name, payload):
        with (self.run_dir / name).open(
                'a', encoding='utf-8', buffering=1
        ) as handle:
            handle.write(json.dumps(
                _json_safe(payload), ensure_ascii=False
            ) + '\n')

    def _evaluate(
            self,
            physics_fps,
            episodes,
            transition,
            *,
            trajectory_output_path=None,
            trajectory_episodes=0,
            episode_index_output_path=None):
        summary, details = evaluate_policy(
            self.model,
            physics_fps=physics_fps,
            episodes=episodes,
            parallel_envs=min(
                self.config.evaluation.parallel_envs, episodes
            ),
            device=self.device,
            seed_base=self.config.evaluation.seed_base,
            max_fruits=self.config.model.max_fruits,
            max_episode_drops=self.config.evaluation.max_episode_drops,
            trajectory_output_path=trajectory_output_path,
            trajectory_episodes=trajectory_episodes,
            episode_index_output_path=episode_index_output_path,
            critical_event_min_level=(
                self.config.analysis.critical_event_min_level
            ),
            score_bin_width=self.config.analysis.score_bin_width,
        )
        payload = {
            'transition': transition,
            **summary.to_dict(),
            **{key: value for key, value in details.items() if key != 'scores'},
        }
        self._append_jsonl('evaluations/metrics.jsonl', payload)
        self.last_eval_created_density = list(
            details.get(
                'created_level_density_per_1000_drops', [None] * 12
            )
        )
        if physics_fps == 30:
            self.last_fast_eval_score = summary.mean_score
        else:
            self.last_accurate_eval_score = summary.mean_score
            if summary.mean_score > self.best_accurate_score:
                self.best_accurate_score = summary.mean_score
                self.save_checkpoint(
                    'best.pt',
                    extra={'best_evaluation': payload},
                )
        self.dashboard.event(
            'evaluation_finished',
            f'{physics_fps} FPS greedy 评估完成',
            mean_score=summary.mean_score,
            episodes=summary.episodes,
        )
        self.dashboard.snapshot_curves(wait=True, timeout=30.0)
        return payload

    @staticmethod
    def _state_to_cpu_dict(state):
        return {
            name: getattr(state, name).detach().cpu()
            for name in state.__dataclass_fields__
            if name != 'physics_fps'
        }

    def _export_transition_sample(self):
        sample_count = min(
            self.config.analysis.transition_sample_size, len(self.replay)
        )
        if sample_count <= 0:
            return None
        chunks = []
        remaining = sample_count
        while remaining > 0:
            chunk_size = min(
                remaining,
                self.config.analysis.transition_chunk_size,
                len(self.replay),
            )
            batch = self.replay.sample(chunk_size)
            chunks.append({
                'current': self._state_to_cpu_dict(batch.current),
                'actions': batch.action.detach().cpu(),
                'rewards': batch.reward.detach().cpu(),
                'next_state': self._state_to_cpu_dict(batch.next_state),
                'terminal': batch.terminal.detach().cpu(),
                'stage': batch.stage.detach().cpu(),
            })
            remaining -= chunk_size
        path = self.run_dir / 'analysis' / 'transition_sample.pt'
        temporary = path.with_suffix('.pt.tmp')
        torch.save({
            'format_version': 1,
            'physics_fps': self.simulator_config.physics_fps,
            'reward_config': self.config.to_dict()['reward'],
            'sample_count': sample_count,
            'replay_metadata': self.replay.metadata(),
            'chunks': chunks,
        }, temporary)
        os.replace(temporary, path)
        self.dashboard.event(
            'analysis_sample_exported',
            '已导出带阶段标签的均匀 Replay 样本',
            sample_count=sample_count,
            path=str(path),
        )
        return path

    def _maybe_evaluate(self, previous_transitions):
        interval = self.config.evaluation.fast_interval_transitions
        if (
                previous_transitions // interval
                < self.transitions // interval):
            self._evaluate(
                30,
                self.config.evaluation.periodic_episodes,
                self.transitions,
            )
        for milestone in self.config.evaluation.accurate_milestones:
            if (
                    previous_transitions < milestone <= self.transitions
                    and milestone not in self.completed_accurate_milestones):
                self._evaluate(
                    120,
                    self.config.evaluation.periodic_episodes,
                    self.transitions,
                )
                self.completed_accurate_milestones.add(milestone)

    def _maybe_milestone_checkpoint(self, previous_transitions):
        for milestone in self.CHECKPOINT_MILESTONES:
            if (
                    previous_transitions < milestone <= self.transitions
                    and milestone not in self.completed_checkpoint_milestones):
                self.save_checkpoint(f'transition_{milestone:09d}.pt')
                self.completed_checkpoint_milestones.add(milestone)

    def _publish_metrics(
            self,
            *,
            started,
            window_started,
            window_transitions,
            window_branch_transitions,
            window_updates,
            stage_seconds):
        now = time.perf_counter()
        window = max(now - window_started, 1e-9)
        uptime = now - started
        remaining = max(0, self.config.total_transitions - self.transitions)
        speed = window_transitions / window
        actor_values = torch.cat((
            self.actor_metric_sums,
            self.actor_rank_correlation_sums,
        )).detach().cpu().tolist()
        actor_decisions = max(1, self.actor_metric_decisions)
        active_count = actor_values[0]
        active_changed_count = actor_values[2]
        selected_rank_correlation = rank_correlation_from_sums(
            actor_values[3:]
        )
        payload = {
            'phase': 'training',
            **self._progress(),
            'epsilon': self.epsilon(),
            'replay_size': len(self.replay),
            'replay_capacity': self.replay.capacity,
            'branch_replay_size': (
                0 if self.branch_replay is None else len(self.branch_replay)
            ),
            'branch_replay_capacity': (
                0
                if self.branch_replay is None
                else self.branch_replay.capacity
            ),
            'env_steps_per_second': speed,
            'branch_steps_per_second': window_branch_transitions / window,
            'updates_per_second': window_updates / window,
            'learner_samples_per_second': (
                window_updates * (
                    self.config.replay.batch_size
                    + (
                        self.config.branch_learning.learner_batch_size
                        if (
                            self.branch_replay is not None
                            and len(self.branch_replay)
                            >= self.branch_replay_training_threshold
                        ) else 0
                    )
                ) / window
            ),
            'uptime_seconds': uptime,
            'eta_seconds': remaining / max(speed, 1e-9),
            'training_mean_score': (
                sum(self.recent_scores) / len(self.recent_scores)
                if self.recent_scores else None
            ),
            'training_rolling_mean_score': (
                sum(self.recent_scores) / len(self.recent_scores)
                if self.recent_scores else None
            ),
            'training_rolling_max_score': (
                max(self.recent_scores) if self.recent_scores else None
            ),
            'training_window_mean_score': (
                sum(self.metric_window_scores) / len(self.metric_window_scores)
                if self.metric_window_scores else None
            ),
            'training_window_max_score': (
                max(self.metric_window_scores)
                if self.metric_window_scores else None
            ),
            'training_window_episodes': len(self.metric_window_scores),
            'training_mean_drops': (
                sum(self.recent_drops) / len(self.recent_drops)
                if self.recent_drops else None
            ),
            'training_window_mean_drops': (
                sum(self.metric_window_drops) / len(self.metric_window_drops)
                if self.metric_window_drops else None
            ),
            'last_fast_eval_score': self.last_fast_eval_score,
            'last_accurate_eval_score': self.last_accurate_eval_score,
            'active_learning_action_fraction': (
                active_count / actor_decisions
            ),
            'epsilon_explore_action_fraction': (
                actor_values[1] / actor_decisions
            ),
            'active_learning_effective_action_fraction': (
                active_changed_count / actor_decisions
            ),
            'active_learning_greedy_overlap_rate': (
                None
                if active_count <= 0.0
                else 1.0 - active_changed_count / active_count
            ),
            'active_selected_rank_correlation': selected_rank_correlation,
            **stage_seconds,
            **self.training_metrics.flush(),
            **self.reward_metrics.flush(),
            **{
                f'eval_created_l{level}_per_1000': (
                    self.last_eval_created_density[level]
                )
                for level in range(7, 12)
            },
            **{
                f'decision_data_{name}': value
                for name, value
                in self.key_decision_collector.metrics().items()
            },
        }
        self.actor_metric_sums.zero_()
        self.actor_rank_correlation_sums.zero_()
        self.actor_metric_decisions = 0
        action_counts = self.action_counts.detach().cpu()
        payload['action_counts'] = action_counts.tolist()
        payload['action_distribution'] = (
            action_counts.to(torch.float64)
            / action_counts.sum().clamp_min(1)
        ).tolist()
        self.action_counts.zero_()
        resources = self.resource_sampler.sample()
        decision = self.scale_controller.observe(resources, speed)
        if decision is not None:
            previous_envs = self.active_envs
            if decision.action in ('trial', 'rollback'):
                self._set_active_envs(decision.target_envs)
            self.dashboard.event(
                f'autoscale_{decision.action}',
                decision.reason,
                previous_envs=previous_envs,
                target_envs=self.active_envs,
            )
            payload['active_envs'] = self.active_envs
            payload['autoscale_action'] = decision.action
            payload['autoscale_reason'] = decision.reason
        payload.update({
            f'trainer_{name}': value for name, value in resources.items()
        })
        self.dashboard.publish(payload)
        self._append_jsonl('metrics.jsonl', {'timestamp': time.time(), **payload})
        self.metric_window_scores.clear()
        self.metric_window_drops.clear()
        return payload

    def run(self, *, final_evaluation=True):
        started = time.perf_counter()
        deadline = (
            started
            + self.config.max_wall_seconds
            - self.config.finalization_reserve_seconds
            if self.config.max_wall_seconds > 0
            else float('inf')
        )
        self.dashboard.event(
            'training_started',
            (
                '空间奖励30 FPS GNN-DQN训练启动'
                if self.config.reward.kind in ('spatial_v2', 'spatial_v2_1')
                else '纯分数基线 30 FPS GNN-DQN 训练启动'
            ),
            reward_kind=self.config.reward.kind,
            dashboard=f'http://{self.config.dashboard.host}:{self.config.dashboard.port}',
        )
        self._write_run_status('training', '训练正在进行')
        if len(self.replay) < self.config.replay.warmup_transitions:
            self.estimate_stage_thresholds()
            self.stagger_initial_states()
            self.fill_warmup_replay()
        if (
                self.reward_computer is not None
                and not self.reward_computer.initialized):
            self._initialize_reward()

        window_started = time.perf_counter()
        window_transitions = 0
        window_branch_transitions = 0
        window_updates = 0
        last_log = window_started
        last_heartbeat = window_started
        last_checkpoint = window_started
        stage_seconds = {
            'actor_seconds': 0.0,
            'physics_seconds': 0.0,
            'reward_seconds': 0.0,
            'learner_seconds': 0.0,
            'decision_data_seconds': 0.0,
            'branch_seconds': 0.0,
        }
        try:
            while (
                    self.transitions < self.config.total_transitions
                    and time.perf_counter() < deadline
                    and not self.stop_requested):
                previous_transitions = self.transitions
                observation = self.simulator.observe()
                current = TensorState.from_observation(
                    observation,
                    physics_fps=self.simulator_config.physics_fps,
                    rows=self._active_rows(),
                )
                current_fruit_count = current.active.sum(dim=1)
                current_danger_progress = current.danger_progress.clone()
                current_stages = self._classify_stages(observation)
                actor_started = time.perf_counter()
                action_selection = self._select_action_batch(
                    current, self.epsilon()
                )
                self._accumulate_actor_metrics(action_selection)
                active_actions = action_selection.actions
                self.action_counts += torch.bincount(
                    active_actions,
                    minlength=self.config.model.action_count,
                )
                full_actions = torch.zeros(
                    self.config.max_envs,
                    dtype=torch.int64,
                    device=self.device,
                )
                full_actions[:self.active_envs] = active_actions
                branch_started = time.perf_counter()
                produced_branches = self._collect_branch_transitions(
                    current_stages=current_stages,
                    action_selection=action_selection,
                    parent_transitions_after_step=min(
                        self.config.total_transitions,
                        previous_transitions + self.active_envs,
                    ),
                )
                stage_seconds['branch_seconds'] += (
                    time.perf_counter() - branch_started
                )
                window_branch_transitions += produced_branches
                ticket = self.replay.begin_append(current)
                stage_seconds['actor_seconds'] += (
                    time.perf_counter() - actor_started
                )
                staged_decisions = None
                if self.key_decision_collector.active:
                    collection_started = time.perf_counter()
                    staged_decisions = self.key_decision_collector.stage_pre(
                        current=current,
                        action_selection=action_selection,
                        ticket=ticket,
                        environment_rows=self._active_rows(),
                        transition_start=previous_transitions,
                        policy_version=self.learner.update_count,
                    )
                    stage_seconds['decision_data_seconds'] += (
                        time.perf_counter() - collection_started
                    )

                physics_started = time.perf_counter()
                result = self._step(full_actions, self._enabled_mask())
                stage_seconds['physics_seconds'] += (
                    time.perf_counter() - physics_started
                )
                next_state = TensorState.from_observation(
                    result.observation,
                    physics_fps=self.simulator_config.physics_fps,
                    rows=self._active_rows(),
                )
                reward_started = time.perf_counter()
                rewards = self._compute_rewards(
                    result, record_metrics=True
                )
                stage_seconds['reward_seconds'] += (
                    time.perf_counter() - reward_started
                )
                terminals = result.physics.done[:self.active_envs]
                action_effects = self._build_action_effect_targets(
                    current,
                    next_state,
                    result,
                    current_fruit_count,
                    current_danger_progress,
                )
                self.replay.finish_append(
                    ticket,
                    next_state,
                    active_actions,
                    rewards,
                    terminals,
                    current_stages,
                    action_effects,
                )
                if staged_decisions is not None:
                    collection_started = time.perf_counter()
                    self.key_decision_collector.observe_post(
                        staged_decisions,
                        result=result,
                        next_state=next_state,
                        rewards=rewards,
                        stages=current_stages,
                        episode_finished=self._episode_finished(result),
                    )
                    stage_seconds['decision_data_seconds'] += (
                        time.perf_counter() - collection_started
                    )
                self.transitions += self.active_envs
                self.simulated_transitions += self.active_envs
                window_transitions += self.active_envs
                self._record_episodes(result)

                self.update_credit += (
                    self.active_envs
                    * self.config.dqn.utd_ratio
                    / self.config.replay.batch_size
                )
                updates = int(self.update_credit)
                self.update_credit -= updates
                learner_started = time.perf_counter()
                for _ in range(updates):
                    use_branch = bool(
                        self.branch_replay is not None
                        and len(self.branch_replay)
                        >= self.branch_replay_training_threshold
                    )
                    metrics = self.learner.update(
                        self.replay,
                        self.config.replay.batch_size,
                        branch_replay=(
                            self.branch_replay if use_branch else None
                        ),
                        branch_batch_size=(
                            self.config.branch_learning.learner_batch_size
                            if use_branch else 0
                        ),
                        branch_loss_weight=(
                            self.config.branch_learning.loss_weight
                            if use_branch else 0.0
                        ),
                    )
                    self.training_metrics.add(metrics)
                stage_seconds['learner_seconds'] += (
                    time.perf_counter() - learner_started
                )
                window_updates += updates

                self._maybe_milestone_checkpoint(previous_transitions)
                self._maybe_evaluate(previous_transitions)
                now = time.perf_counter()
                if (
                        now - last_heartbeat
                        >= self.config.dashboard.publish_interval_seconds):
                    heartbeat_window = max(now - window_started, 1e-9)
                    self.dashboard.publish({
                        'phase': 'training',
                        **self._progress(),
                        'epsilon': self.epsilon(),
                        'replay_size': len(self.replay),
                        'replay_capacity': self.replay.capacity,
                        'branch_transitions': self.branch_transitions,
                        'branch_total_transitions': (
                            self.config.branch_learning.transition_budget
                        ),
                        'branch_replay_size': (
                            0
                            if self.branch_replay is None
                            else len(self.branch_replay)
                        ),
                        'env_steps_per_second': (
                            window_transitions / heartbeat_window
                        ),
                        'branch_steps_per_second': (
                            window_branch_transitions / heartbeat_window
                        ),
                        'updates_per_second': (
                            window_updates / heartbeat_window
                        ),
                        'uptime_seconds': now - started,
                    })
                    last_heartbeat = now
                if now - last_checkpoint >= self.config.checkpoint_interval_seconds:
                    self.save_checkpoint('latest.pt')
                    last_checkpoint = now
                if now - last_log >= self.config.log_interval_seconds:
                    self._publish_metrics(
                        started=started,
                        window_started=window_started,
                        window_transitions=window_transitions,
                        window_branch_transitions=window_branch_transitions,
                        window_updates=window_updates,
                        stage_seconds=stage_seconds,
                    )
                    window_started = time.perf_counter()
                    window_transitions = 0
                    window_branch_transitions = 0
                    window_updates = 0
                    stage_seconds = {
                        'actor_seconds': 0.0,
                        'physics_seconds': 0.0,
                        'reward_seconds': 0.0,
                        'learner_seconds': 0.0,
                        'decision_data_seconds': 0.0,
                        'branch_seconds': 0.0,
                    }
                    last_log = window_started

            self.save_checkpoint('latest.pt')
            if final_evaluation:
                index_30fps = (
                    self.run_dir / 'analysis' / 'final_eval_30fps_index.pt'
                )
                index_120fps = (
                    self.run_dir / 'analysis' / 'final_eval_120fps_index.pt'
                )
                self._evaluate(
                    30,
                    self.config.evaluation.final_episodes,
                    self.transitions,
                    episode_index_output_path=index_30fps,
                )
                self._evaluate(
                    120,
                    self.config.evaluation.final_episodes,
                    self.transitions,
                    episode_index_output_path=index_120fps,
                )
                replay_summary = replay_critical_episodes(
                    self.model,
                    episode_index_path=index_30fps,
                    output_path=(
                        self.run_dir
                        / 'analysis'
                        / 'critical_event_trajectories_30fps.pt'
                    ),
                    selected_episodes=(
                        self.config.analysis.critical_event_episodes
                    ),
                    device=self.device,
                    max_fruits=self.config.model.max_fruits,
                    score_bin_width=self.config.analysis.score_bin_width,
                    selection_seed=self.config.seed + 700,
                )
                self.dashboard.event(
                    'critical_replay_finished',
                    '关键事件局分层回放完成',
                    **replay_summary,
                )
                event_plot = render_evaluation_event_analysis(
                    self.run_dir,
                    index_30fps,
                    index_120fps,
                    score_bin_width=self.config.analysis.score_bin_width,
                )
                self.dashboard.plot('evaluation_event_analysis', event_plot)
            self._export_transition_sample()
            self.key_decision_collector.close()
            final_path = self.save_checkpoint('final.pt')
            loaded = load_checkpoint(final_path, map_location='cpu')
            if int(loaded['progress']['transitions']) != self.transitions:
                raise RuntimeError('final checkpoint round-trip validation failed')
            completed = (
                self.transitions >= self.config.total_transitions
                and (
                    not self.config.branch_learning.enabled
                    or self.branch_transitions
                    >= self.config.branch_learning.transition_budget
                )
                and not self.stop_requested
            )
            terminal_phase = 'completed' if completed else 'stopped'
            terminal_message = (
                (
                    '训练、最终评估和产物校验均已正常完成'
                    if final_evaluation else
                    '训练和产物校验已正常完成（按参数跳过最终评估）'
                )
                if completed
                else f'训练已安全停止：{self.stop_reason or "达到运行时间预算"}'
            )
            manifest_path = self.run_dir / 'artifact_manifest.json'
            terminal_payload = self._write_run_status(
                terminal_phase,
                terminal_message,
                manifest=str(manifest_path),
                completed_at=time.time(),
                stop_reason=self.stop_reason,
            )
            self.dashboard.publish(terminal_payload)
            self.dashboard.event(
                'training_finished' if completed else 'training_stopped',
                terminal_message,
                manifest=str(manifest_path),
            )
            self.dashboard.snapshot_curves(wait=True, timeout=30.0)
            write_artifact_manifest(
                self.run_dir,
                tuple(
                    path
                    for pattern in ('*.json', '*.jsonl', '*.pt', '*.png')
                    for path in self.run_dir.rglob(pattern)
                    if '.mplconfig' not in path.parts
                ),
                metadata=self._progress(),
            )
            return self._progress()
        except BaseException as error:
            failure = {
                'timestamp': time.time(),
                'error_type': type(error).__name__,
                'message': str(error),
                'progress': self._progress(),
            }
            (self.run_dir / 'failure_latest.json').write_text(
                json.dumps(failure, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
            failure_status = self._write_run_status(
                'failed',
                f'训练异常结束：{type(error).__name__}: {error}',
                failed_at=time.time(),
                error_type=type(error).__name__,
            )
            self.dashboard.publish(failure_status)
            try:
                self.save_checkpoint('failure_last.pt', extra=failure)
            except BaseException:
                pass
            self.dashboard.event(
                'training_failed', str(error), error_type=type(error).__name__
            )
            raise
        finally:
            try:
                self.key_decision_collector.close()
            except BaseException:
                pass
            self.resource_sampler.close()
            self.dashboard.close()
