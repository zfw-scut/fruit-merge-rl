"""随机抽取 CUDA 环境，录制连续投放并生成浏览器回放。"""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from daxigua.simulator import (
    SimulatorConfig,
    TensorVectorSimulator,
    write_replay_fragment,
    write_replay_html,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--warmup-drops', type=int, default=4)
    parser.add_argument('--record-drops', type=int, default=12)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--frame-stride', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260804)
    parser.add_argument('--max-fruits', type=int, default=64)
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'recordings' / 'cuda-physics-replay.html',
    )
    parser.add_argument(
        '--trace-output',
        type=Path,
        default=PROJECT_ROOT / 'recordings' / 'cuda-physics-trace.pt',
    )
    parser.add_argument('--fragment-output', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available')
    if args.samples <= 0 or args.samples > args.num_envs:
        raise ValueError('samples must be in [1, num_envs]')
    if args.warmup_drops < 0 or args.record_drops <= 0:
        raise ValueError('warmup-drops must be non-negative and record-drops positive')

    config = SimulatorConfig(max_fruits=args.max_fruits)
    simulator = TensorVectorSimulator(
        args.num_envs, config=config, device='cuda'
    )
    simulator.reset(seeds=args.seed)
    generator = torch.Generator(device=simulator.device)
    generator.manual_seed(args.seed)

    for _ in range(args.warmup_drops):
        actions = torch.randint(
            config.action_count,
            (args.num_envs,),
            dtype=torch.int64,
            device=simulator.device,
            generator=generator,
        )
        result = simulator.step(actions)
        reset_mask = result.physics.done | result.physics.truncated
        if bool(reset_mask.any().item()):
            simulator.reset(reset_mask)

    env_indices = torch.randperm(
        args.num_envs, device=simulator.device, generator=generator
    )[:args.samples]
    traces = []
    selected_reset_counts = [0 for _ in range(args.samples)]
    selected_settle_timeout_counts = [0 for _ in range(args.samples)]
    physics_frame_totals = [0 for _ in range(args.samples)]
    score_delta_totals = [0 for _ in range(args.samples)]
    for _ in range(args.record_drops):
        actions = torch.randint(
            config.action_count,
            (args.num_envs,),
            dtype=torch.int64,
            device=simulator.device,
            generator=generator,
        )
        result, step_trace = simulator.step_with_trace(
            actions,
            env_indices,
            frame_stride=args.frame_stride,
        )
        step_trace = step_trace.cpu()
        traces.append(step_trace)
        for row, env_index in enumerate(step_trace.env_indices.tolist()):
            physics_frame_totals[row] += int(
                result.physics.frames_simulated[env_index].item()
            )
            score_delta_totals[row] += int(
                result.physics.score_delta[env_index].item()
            )
            selected_reset_counts[row] += int(
                result.physics.done[env_index].item()
                or result.physics.truncated[env_index].item()
            )
            selected_settle_timeout_counts[row] += int(
                result.physics.settle_timeout[env_index].item()
            )
        reset_mask = result.physics.done | result.physics.truncated
        if bool(reset_mask.any().item()):
            simulator.reset(reset_mask)

    trace_sequence = tuple(traces)
    output_path = write_replay_html(
        args.output,
        trace_sequence,
        config,
        title='CUDA 连续随机投放物理回放',
    )
    fragment_path = (
        write_replay_fragment(args.fragment_output, trace_sequence, config)
        if args.fragment_output is not None
        else None
    )
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'format_version': 3,
            'steps': [
                {
                    field_name: getattr(step_trace, field_name)
                    for field_name in step_trace.__dataclass_fields__
                }
                for step_trace in trace_sequence
            ],
        },
        args.trace_output,
    )

    selected = trace_sequence[0].env_indices.tolist()
    report = {
        'output': str(output_path.resolve()),
        'trace_output': str(args.trace_output.resolve()),
        'fragment_output': (
            str(fragment_path.resolve()) if fragment_path else None
        ),
        'seed': args.seed,
        'warmup_drops': args.warmup_drops,
        'record_drops': args.record_drops,
        'sampled_envs': selected,
        'actions_by_drop': [step.actions.tolist() for step in trace_sequence],
        'physics_frame_totals': physics_frame_totals,
        'score_delta_totals': score_delta_totals,
        'episode_resets': selected_reset_counts,
        'settle_timeouts': selected_settle_timeout_counts,
        'record_counts_by_drop': [
            step.record_counts.tolist() for step in trace_sequence
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
