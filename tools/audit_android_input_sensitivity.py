"""在真实 Tensor/CUDA 长局状态上检查 Android 输入的时间与槽位敏感性。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch  # noqa: E402

from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.viewer import load_viewer_model, viewer_simulator_config  # noqa: E402
from daxigua.simulator import TensorVectorSimulator  # noqa: E402


DEFAULT_CHECKPOINT = (
    ROOT
    / 'runs'
    / 'cloud_rtx5090_auxiliary_action_structured_128m_to_120fps_seed20260812_16m'
    / 'checkpoints'
    / 'best.pt'
)
FRUIT_FIELDS = (
    'positions',
    'velocities',
    'angular_velocities',
    'levels',
    'physics_radii',
    'age_frames',
    'active',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42_000_000)
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def shift_age(state, frames):
    ages = state.age_frames.clone()
    shifted = (ages + int(frames)).clamp_min(0)
    ages[state.active] = shifted[state.active]
    return replace(state, age_frames=ages)


def shift_danger(state, frames):
    progress = (
        state.danger_progress + float(frames) / 240.0
    ).clamp(0.0, 1.0)
    return replace(state, danger_progress=progress)


def compact_slots(state):
    replacements = {}
    active_slots = torch.nonzero(state.active[0], as_tuple=False).flatten()
    count = int(active_slots.numel())
    for name in FRUIT_FIELDS:
        value = getattr(state, name)
        packed = torch.zeros_like(value)
        if count:
            packed[0, :count] = value[0].index_select(0, active_slots)
        replacements[name] = packed
    return replace(state, **replacements)


def new_stat():
    return {
        'states': 0,
        'action_flips': 0,
        'maximum_q_absolute_delta': 0.0,
        'sum_step_maximum_q_delta': 0.0,
    }


def update_stat(stat, reference, candidate):
    delta = float((reference - candidate).abs().max().item())
    stat['states'] += 1
    stat['action_flips'] += int(
        reference.argmax(dim=1).item() != candidate.argmax(dim=1).item()
    )
    stat['maximum_q_absolute_delta'] = max(
        stat['maximum_q_absolute_delta'], delta
    )
    stat['sum_step_maximum_q_delta'] += delta


def finalize(stat):
    states = max(1, stat.pop('states'))
    total_delta = stat.pop('sum_step_maximum_q_delta')
    stat['evaluated_states'] = states
    stat['action_flip_rate'] = stat['action_flips'] / states
    stat['mean_step_maximum_q_delta'] = total_delta / states
    return stat


@torch.inference_mode()
def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError('--steps must be positive')
    loaded = load_viewer_model(args.checkpoint, device=args.device)
    config = replace(
        viewer_simulator_config(120, loaded.model_config, loaded.device),
        drop_fast_forward=False,
    )
    simulator = TensorVectorSimulator(1, config=config, device=loaded.device)
    simulator.reset(seeds=args.seed)
    variants = {
        **{
            f'age_{frames:+d}_physics_frames': (
                lambda state, frames=frames: shift_age(state, frames)
            )
            for frames in (-10, -5, -1, 1, 5, 10)
        },
        **{
            f'danger_{frames:+d}_physics_frames': (
                lambda state, frames=frames: shift_danger(state, frames)
            )
            for frames in (-10, -5, -1, 1, 5, 10)
        },
        'legacy_compacted_slots': compact_slots,
    }
    stats = {name: new_stat() for name in variants}
    steps = 0
    while steps < args.steps:
        observation = simulator.observe()
        state = TensorState.from_observation(observation, physics_fps=120)
        variant_states = tuple(transform(state) for transform in variants.values())
        all_q_values = loaded.model(TensorState.cat((state, *variant_states)))
        reference = all_q_values[:1]
        for index, name in enumerate(variants, start=1):
            update_stat(stats[name], reference, all_q_values[index:index + 1])
        action = reference.argmax(dim=1)
        result = simulator.step(action)
        steps += 1
        if bool(result.physics.done[0]) or bool(result.physics.truncated[0]):
            break

    payload = {
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'seed': args.seed,
        'rollout_steps': steps,
        'score': int(simulator.score[0].item()),
        'physics_fps': 120,
        'variants': {name: finalize(stat) for name, stat in stats.items()},
        'scope': (
            'This measures policy sensitivity on canonical Tensor/CUDA states; '
            'it does not by itself attribute the Java rollout score gap.'
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
    print(encoded)


if __name__ == '__main__':
    main()
