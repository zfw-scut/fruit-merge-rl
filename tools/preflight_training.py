"""云服务器正式训练前的 CUDA、更新、评估和 checkpoint 门禁。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from daxigua.rl.action_effects import build_action_effect_targets
from daxigua.rl.checkpoint import (
    initialize_learner_weights,
    load_checkpoint,
    save_checkpoint_atomic,
)
from daxigua.rl.config import TrainingConfig
from daxigua.rl.curves import render_training_curve_snapshot
from daxigua.rl.evaluation import evaluate_policy
from daxigua.rl.learner import DqnLearner
from daxigua.rl.model import BaselineGnnDqn
from daxigua.rl.observations import TensorState
from daxigua.rl.replay import GpuReplayBuffer
from daxigua.rl.trainer import (
    ranked_active_learning_candidates,
    training_simulator_config,
)
from daxigua.rl.viewer import load_viewer_model
from daxigua.simulator import (
    SpatialRewardComputer,
    SpatialRewardConfig,
    TensorVectorSimulator,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=Path,
        default=PROJECT_ROOT / 'configs' / 'gnn_dqn_reward_v2_1.toml',
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-envs', type=int, default=32)
    parser.add_argument('--smoke-batch-size', type=int, default=16)
    parser.add_argument('--evaluation-episodes', type=int, default=8)
    parser.add_argument('--evaluation-max-drops', type=int, default=16)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        '--init-checkpoint', type=Path,
        help='在预检更新前验证weights-only迁移来源',
    )
    initialization.add_argument(
        '--prewarm-checkpoint', type=Path,
        help='验证只用于预热场景生成的teacher，不初始化待训练learner',
    )
    parser.add_argument(
        '--disable-compile',
        action='store_true',
        help='本地Windows缺少Triton时仅关闭模型编译，不改变正式配置',
    )
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_ROOT / 'runs' / 'preflight' / 'gnn_dqn.json',
    )
    return parser.parse_args()


def run_preflight(args):
    config = TrainingConfig.from_toml(args.config)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    simulator_config = training_simulator_config(config, device)
    simulator = TensorVectorSimulator(
        args.smoke_envs,
        config=simulator_config,
        device=device,
    )
    reward_computer = (
        SpatialRewardComputer(
            simulator_config,
            device=device,
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
    model = BaselineGnnDqn(config.model).to(device)
    learner = DqnLearner(
        model,
        replace(
            config.dqn,
            target_update_interval=1,
            use_bfloat16=(config.dqn.use_bfloat16 and device.type == 'cuda'),
            compile_model=(
                config.dqn.compile_model and not args.disable_compile
            ),
        ),
    )
    initialization = None
    stage_pilot_teacher = None
    if args.init_checkpoint is not None:
        initialization = initialize_learner_weights(
            learner,
            args.init_checkpoint,
            expected_model_config=config.to_dict()['model'],
            map_location=device,
        )
    elif args.prewarm_checkpoint is not None:
        if config.stage_pilot_policy_epsilon is None:
            raise ValueError(
                'prewarm checkpoint requires stage_pilot_policy_epsilon'
            )
        loaded_teacher = load_viewer_model(
            args.prewarm_checkpoint, device=device
        )
        if loaded_teacher.model_config != config.model:
            raise ValueError(
                'prewarm checkpoint model config does not match target model'
            )
        with torch.inference_mode():
            teacher_q_values = loaded_teacher.model(
                TensorState.from_observation(
                    simulator.observe(),
                    physics_fps=simulator_config.physics_fps,
                )
            )
        if not bool(torch.isfinite(teacher_q_values).all().item()):
            raise FloatingPointError(
                'prewarm teacher produced non-finite Q values'
            )
        stage_pilot_teacher = {
            'kind': 'prewarm_teacher_only',
            'source_checkpoint': str(loaded_teacher.checkpoint_path),
            'source_checkpoint_sha256': loaded_teacher.checkpoint_sha256,
            'source_progress': loaded_teacher.progress,
            'q_shape': list(teacher_q_values.shape),
            'training_learner_random_initialization_preserved': True,
        }
        del loaded_teacher
    elif config.stage_pilot_policy_epsilon is not None:
        raise ValueError(
            'this config requires --prewarm-checkpoint; the teacher is not '
            'a weights-only initialization source'
        )
    model = learner.online_model
    replay = GpuReplayBuffer(
        max(args.smoke_envs * 2, args.smoke_batch_size * 2),
        max_fruits=config.model.max_fruits,
        device=device,
        physics_fps=simulator_config.physics_fps,
        policy_head_count=config.model.policy_head_count,
        bootstrap_probability=config.dqn.bootstrap_probability,
        action_effects_enabled=config.model.action_effect_enabled,
    )
    observation = simulator.observe()
    if reward_computer is not None:
        reward_computer.initialize(observation)
    current = TensorState.from_observation(
        observation, physics_fps=simulator_config.physics_fps
    )
    with torch.inference_mode():
        q_values = model(current)
    if q_values.shape != (args.smoke_envs, 21):
        raise RuntimeError('model Q output shape is invalid')
    if not bool(torch.isfinite(q_values).all().item()):
        raise FloatingPointError('model produced non-finite Q values')
    actions = q_values.argmax(dim=1)
    current_fruit_count = current.active.sum(dim=1)
    current_danger_progress = current.danger_progress.clone()
    ticket = replay.begin_append(current)
    result = simulator.step(actions)
    reward_step = (
        reward_computer.step(result)
        if reward_computer is not None
        else None
    )
    rewards = (
        reward_step.reward
        if reward_step is not None
        else result.physics.score_delta.to(torch.float32)
        / config.reward.score_divisor
    )
    next_state = TensorState.from_observation(
        result.observation, physics_fps=simulator_config.physics_fps
    )
    action_effects = (
        build_action_effect_targets(
            current,
            next_state,
            result,
            board_width=simulator_config.board_width,
            board_height=simulator_config.board_height,
            gravity_y=simulator_config.gravity_y,
            max_physics_frames=simulator_config.max_physics_frames,
            current_fruit_count=current_fruit_count,
            current_danger_progress=current_danger_progress,
        )
        if config.model.action_effect_enabled
        else None
    )
    replay.finish_append(
        ticket,
        next_state,
        actions,
        rewards,
        result.physics.done,
        action_effects=action_effects,
    )
    while len(replay) < args.smoke_batch_size:
        replay.append(
            current,
            actions,
            torch.zeros(args.smoke_envs, device=device),
            next_state,
            torch.zeros(
                args.smoke_envs, dtype=torch.bool, device=device
            ),
            action_effects=action_effects,
        )
    learner_metrics = learner.update(replay, args.smoke_batch_size)
    if not bool(torch.isfinite(learner_metrics['loss']).item()):
        raise FloatingPointError('learner update produced non-finite loss')

    branch_replay = None
    branch_metrics = None
    branch_transition_count = 0
    projected_branch_replay_memory = 0
    if config.branch_learning.enabled:
        branch = config.branch_learning
        source_count = max(
            1,
            min(
                args.smoke_envs,
                args.smoke_batch_size,
            ) // branch.actions_per_state,
        )
        branch_batch_size = source_count * branch.actions_per_state
        branch_simulator = TensorVectorSimulator(
            branch_batch_size,
            config=simulator_config,
            device=device,
        )
        source_rows = torch.arange(
            source_count, dtype=torch.int64, device=device
        )
        expanded_sources = source_rows.unsqueeze(1).expand(
            -1, branch.actions_per_state
        ).reshape(-1)
        branch_simulator.copy_rows_from(
            simulator,
            expanded_sources,
        )
        with torch.inference_mode():
            source_output = model(next_state.index_select(source_rows), True)
            source_uncertainty = (
                source_output.head_q_values
                - source_output.head_q_values.mean(dim=2, keepdim=True)
            ).float().std(dim=1, correction=0)
            candidates, _value_ranks, _uncertainty_ranks = (
                ranked_active_learning_candidates(
                    source_output.q_values,
                    source_uncertainty,
                    config.model.action_count,
                )
            )
            parent_actions = source_output.q_values.argmax(dim=1)
            alternatives = candidates[
                candidates.ne(parent_actions.unsqueeze(1))
            ].view(source_count, config.model.action_count - 1)
            branch_actions = alternatives[
                :, :branch.actions_per_state
            ].reshape(-1)
        branch_current = TensorState.from_observation(
            branch_simulator.observe(),
            physics_fps=simulator_config.physics_fps,
            clone=True,
        )
        branch_fruit_count = branch_current.active.sum(dim=1)
        branch_danger_progress = branch_current.danger_progress.clone()
        branch_result = branch_simulator.step(branch_actions)
        branch_next = TensorState.from_observation(
            branch_result.observation,
            physics_fps=simulator_config.physics_fps,
        )
        branch_effects = build_action_effect_targets(
            branch_current,
            branch_next,
            branch_result,
            board_width=simulator_config.board_width,
            board_height=simulator_config.board_height,
            gravity_y=simulator_config.gravity_y,
            max_physics_frames=simulator_config.max_physics_frames,
            current_fruit_count=branch_fruit_count,
            current_danger_progress=branch_danger_progress,
        )
        branch_rewards = branch_result.physics.score_delta.to(
            torch.float32
        ) / config.reward.score_divisor
        branch_learner_batch_size = min(
            branch.learner_batch_size,
            args.smoke_batch_size,
        )
        branch_replay = GpuReplayBuffer(
            max(branch_batch_size * 2, branch_learner_batch_size * 2),
            max_fruits=config.model.max_fruits,
            device=device,
            physics_fps=simulator_config.physics_fps,
            policy_head_count=config.model.policy_head_count,
            bootstrap_probability=config.dqn.bootstrap_probability,
            action_effects_enabled=True,
        )
        while len(branch_replay) < branch_learner_batch_size:
            branch_replay.append(
                branch_current,
                branch_actions,
                branch_rewards,
                branch_next,
                branch_result.physics.done,
                action_effects=branch_effects,
            )
        branch_metrics = learner.update(
            replay,
            args.smoke_batch_size,
            branch_replay=branch_replay,
            branch_batch_size=branch_learner_batch_size,
            branch_loss_weight=branch.loss_weight,
        )
        if not bool(torch.isfinite(branch_metrics['loss']).item()):
            raise FloatingPointError(
                'branch learner update produced non-finite loss'
            )
        branch_transition_count = branch_batch_size
        projected_branch_replay_memory = int(
            branch_replay.memory_bytes
            * branch.replay_capacity
            / branch_replay.capacity
        )

    smoke_dir = args.output.parent / '_checkpoint_smoke'
    smoke_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = smoke_dir / 'round_trip.pt'
    smoke_config = replace(
        config,
        device=str(device),
        max_envs=args.smoke_envs,
        active_envs=args.smoke_envs,
    )
    save_checkpoint_atomic(
        checkpoint_path,
        learner=learner,
        training_config=smoke_config,
        progress={'transitions': args.smoke_envs, 'updates': 1},
        replay_metadata=replay.metadata(),
    )
    loaded = load_checkpoint(checkpoint_path, map_location='cpu')
    if loaded['progress']['updates'] != 1:
        raise RuntimeError('checkpoint round-trip lost progress')

    curve_snapshot = None
    if (
            config.dashboard.enabled
            and config.dashboard.curve_snapshot_enabled):
        (smoke_dir / 'metrics.jsonl').write_text(
            json.dumps({
                'transitions': args.smoke_envs,
                'training_rolling_mean_score': 100.0,
                'training_window_mean_score': 100.0,
                'training_window_max_score': 120.0,
                'loss': float(learner_metrics['loss'].item()),
                'mean_abs_td_error': 0.1,
                'env_steps_per_second': 1.0,
                'learner_samples_per_second': 1.0,
            }) + '\n',
            encoding='utf-8',
        )
        curve_snapshot = render_training_curve_snapshot(smoke_dir)
        curve_path = smoke_dir / 'plots' / 'training_curves.png'
        if curve_path.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
            raise RuntimeError('curve snapshot PNG validation failed')

    evaluations = {}
    for physics_fps in (30, 120):
        summary, _details = evaluate_policy(
            model,
            physics_fps=physics_fps,
            episodes=args.evaluation_episodes,
            parallel_envs=args.evaluation_episodes,
            device=device,
            seed_base=config.evaluation.seed_base,
            max_fruits=config.model.max_fruits,
            max_episode_drops=args.evaluation_max_drops,
        )
        evaluations[str(physics_fps)] = summary.to_dict()
    shutil.rmtree(smoke_dir)
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        device_name = torch.cuda.get_device_name(device)
    else:
        peak_memory = 0.0
        device_name = 'cpu'
    projected_parent_replay_memory = int(
        replay.memory_bytes * config.replay.capacity / replay.capacity
    )
    return {
        'ready': True,
        'timestamp': time.time(),
        'device': str(device),
        'device_name': device_name,
        'torch_version': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'smoke_envs': args.smoke_envs,
        'smoke_batch_size': args.smoke_batch_size,
        'q_shape': list(q_values.shape),
        'learner_loss': float(learner_metrics['loss'].item()),
        'dqn_loss': float(learner_metrics['dqn_loss'].item()),
        'auxiliary_loss': (
            float(learner_metrics['aux_loss_total'].item())
            if 'aux_loss_total' in learner_metrics else None
        ),
        'policy_disagreement': float(
            learner_metrics['policy_disagreement'].item()
        ),
        'branch_learning_enabled': config.branch_learning.enabled,
        'branch_smoke_transitions': branch_transition_count,
        'branch_replay_memory_bytes': (
            0 if branch_replay is None else branch_replay.memory_bytes
        ),
        'projected_branch_replay_memory_bytes': (
            projected_branch_replay_memory
        ),
        'branch_composite_loss': (
            None
            if branch_metrics is None
            else float(branch_metrics['loss'].item())
        ),
        'branch_dqn_loss': (
            None
            if branch_metrics is None
            else float(branch_metrics['branch_dqn_loss'].item())
        ),
        'branch_auxiliary_loss': (
            None
            if branch_metrics is None
            else float(branch_metrics['branch_aux_loss_total'].item())
        ),
        'branch_sample_fraction': (
            None
            if branch_metrics is None
            else float(branch_metrics['branch_sample_fraction'].item())
        ),
        'configured_branch_sample_fraction': (
            0.0
            if not config.branch_learning.enabled
            else config.branch_learning.learner_batch_size / (
                config.replay.batch_size
                + config.branch_learning.learner_batch_size
            )
        ),
        'configured_branch_effective_loss_fraction': (
            0.0
            if not config.branch_learning.enabled
            else config.branch_learning.loss_weight / (
                1.0 + config.branch_learning.loss_weight
            )
        ),
        'target_synced': learner_metrics['target_synced'],
        'replay_memory_bytes': replay.memory_bytes,
        'projected_parent_replay_memory_bytes': (
            projected_parent_replay_memory
        ),
        'projected_total_replay_memory_bytes': (
            projected_parent_replay_memory
            + projected_branch_replay_memory
        ),
        'peak_cuda_memory_mb': peak_memory,
        'evaluations': evaluations,
        'checkpoint_round_trip': True,
        'curve_snapshot': curve_snapshot,
        'training_physics_fps': simulator_config.physics_fps,
        'initialization': initialization,
        'stage_pilot_teacher': stage_pilot_teacher,
        'evaluation_physics_fps': [30, 120],
        'accurate_replay_writes': 0,
        'reward_kind': config.reward.kind,
        'reward_scale': config.reward.reward_scale,
        'spatial_reward_mean': (
            float(reward_step.reward.mean().item())
            if reward_step is not None else None
        ),
        'spatial_reward_finite': bool(torch.isfinite(rewards).all().item()),
    }


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_preflight(args)
    except BaseException as error:
        report = {
            'ready': False,
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
        }
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
