"""在 Tensor/CUDA 真实长局状态上逐步校验 Android ONNX 动作。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.viewer import load_viewer_model, viewer_simulator_config  # noqa: E402
from daxigua.simulator import TensorVectorSimulator  # noqa: E402


RUN = (
    ROOT
    / 'runs'
    / 'sab-full-fall-t120-16m-r1_seed20260813'
)
INPUT_NAMES = (
    'positions', 'velocities', 'angular_velocities', 'levels',
    'physics_radii', 'age_frames', 'active', 'fruit_queue',
    'danger_progress', 'over_danger_line',
)


def ort_inputs(observation):
    values = (
        observation.positions,
        observation.velocities,
        observation.angular_velocities,
        observation.levels,
        observation.physics_radii,
        observation.age_frames,
        observation.active.to(torch.int64),
        observation.fruit_queue,
        observation.danger_progress,
        observation.over_danger_line.to(torch.int64),
    )
    return {
        name: value.detach().cpu().numpy()
        for name, value in zip(INPUT_NAMES, values)
    }


@torch.inference_mode()
def main():
    loaded = load_viewer_model(RUN / 'checkpoints' / 'final.pt', device='cuda')
    config = replace(
        viewer_simulator_config(120, loaded.model_config, loaded.device),
        drop_fast_forward=False,
    )
    simulator = TensorVectorSimulator(1, config=config, device=loaded.device)
    simulator.reset(seeds=42_000_000)
    session = ort.InferenceSession(
        str(ROOT / 'runs/mobile_export/sab-ff120/sab_ff120.onnx'),
        providers=('CPUExecutionProvider',),
    )
    mismatch_steps = []
    maximum_error = 0.0
    error_sum = 0.0
    steps = 0
    while steps < 1000:
        observation = simulator.observe()
        state = TensorState.from_observation(observation, physics_fps=120)
        expected = loaded.model(state).detach().cpu().numpy()
        actual = session.run(('q_values',), ort_inputs(observation))[0]
        error = float(np.max(np.abs(expected - actual)))
        maximum_error = max(maximum_error, error)
        error_sum += error
        expected_action = int(expected.argmax(axis=1)[0])
        actual_action = int(actual.argmax(axis=1)[0])
        if expected_action != actual_action:
            mismatch_steps.append((steps, expected_action, actual_action, error))
        result = simulator.step(
            torch.tensor([expected_action], device=loaded.device)
        )
        steps += 1
        if bool(result.physics.done[0]) or bool(result.physics.truncated[0]):
            break
    print({
        'steps': steps,
        'score': int(simulator.score[0]),
        'argmax_mismatch_count': len(mismatch_steps),
        'first_mismatches': mismatch_steps[:10],
        'maximum_absolute_error': maximum_error,
        'mean_step_maximum_error': error_sum / steps,
    })


if __name__ == '__main__':
    main()
