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

from daxigua.rl.checkpoint import load_checkpoint, save_checkpoint_atomic
from daxigua.rl.config import TrainingConfig
from daxigua.rl.curves import render_training_curve_snapshot
from daxigua.rl.evaluation import evaluate_policy
from daxigua.rl.learner import DqnLearner
from daxigua.rl.model import BaselineGnnDqn
from daxigua.rl.observations import TensorState
from daxigua.rl.replay import GpuReplayBuffer
from daxigua.simulator import SimulatorConfig, TensorVectorSimulator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=Path,
        default=PROJECT_ROOT / 'configs' / 'gnn_dqn_baseline.toml',
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-envs', type=int, default=32)
    parser.add_argument('--smoke-batch-size', type=int, default=16)
    parser.add_argument('--evaluation-episodes', type=int, default=8)
    parser.add_argument('--evaluation-max-drops', type=int, default=16)
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
    simulator_config = SimulatorConfig.training_fast(
        max_fruits=config.model.max_fruits,
        use_cuda_extension=device.type == 'cuda',
    )
    simulator = TensorVectorSimulator(
        args.smoke_envs,
        config=simulator_config,
        device=device,
    )
    model = BaselineGnnDqn(config.model).to(device)
    learner = DqnLearner(
        model,
        replace(
            config.dqn,
            target_update_interval=1,
            use_bfloat16=(config.dqn.use_bfloat16 and device.type == 'cuda'),
        ),
    )
    model = learner.online_model
    replay = GpuReplayBuffer(
        max(args.smoke_envs * 2, args.smoke_batch_size * 2),
        max_fruits=config.model.max_fruits,
        device=device,
        physics_fps=simulator_config.physics_fps,
    )
    observation = simulator.observe()
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
    ticket = replay.begin_append(current)
    result = simulator.step(actions)
    next_state = TensorState.from_observation(
        result.observation, physics_fps=simulator_config.physics_fps
    )
    replay.finish_append(
        ticket,
        next_state,
        actions,
        result.physics.score_delta.to(torch.float32) / 66.0,
        result.physics.done,
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
        )
    learner_metrics = learner.update(replay, args.smoke_batch_size)
    if not bool(torch.isfinite(learner_metrics['loss']).item()):
        raise FloatingPointError('learner update produced non-finite loss')

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
        'target_synced': learner_metrics['target_synced'],
        'replay_memory_bytes': replay.memory_bytes,
        'peak_cuda_memory_mb': peak_memory,
        'evaluations': evaluations,
        'checkpoint_round_trip': True,
        'curve_snapshot': curve_snapshot,
        'training_physics_fps': 30,
        'evaluation_physics_fps': [30, 120],
        'accurate_replay_writes': 0,
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
