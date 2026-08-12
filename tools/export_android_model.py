"""将当前定长张量 GNN-DQN checkpoint 导出为 Android ONNX 模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from daxigua.rl.observations import TensorState  # noqa: E402
from daxigua.rl.viewer import load_viewer_model  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / 'runs'
    / 'cloud_rtx5090_auxiliary_action_structured_128m_to_120fps_seed20260812_16m'
    / 'checkpoints'
    / 'best.pt'
)
INPUT_NAMES = (
    'positions',
    'velocities',
    'angular_velocities',
    'levels',
    'physics_radii',
    'age_frames',
    'active',
    'fruit_queue',
    'danger_progress',
    'over_danger_line',
)


class AndroidPolicy(torch.nn.Module):
    """只导出 greedy 决策所需的 Q 值，不携带辅助预测头。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
            self,
            positions,
            velocities,
            angular_velocities,
            levels,
            physics_radii,
            age_frames,
            active,
            fruit_queue,
            danger_progress,
            over_danger_line):
        state = TensorState(
            positions=positions,
            velocities=velocities,
            angular_velocities=angular_velocities,
            levels=levels,
            physics_radii=physics_radii,
            age_frames=age_frames,
            active=active.to(torch.bool),
            fruit_queue=fruit_queue,
            danger_progress=danger_progress,
            over_danger_line=over_danger_line.to(torch.bool),
            physics_fps=120.0,
        )
        return self.model(state)


def parse_args():
    parser = argparse.ArgumentParser(
        description='将 SAB-T120 checkpoint 导出并校验为 Android ONNX。'
    )
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        '--output-dir', type=Path,
        default=PROJECT_ROOT / 'runs' / 'mobile_export' / 'sab-t120',
    )
    parser.add_argument('--opset', type=int, default=18)
    return parser.parse_args()


def sample_inputs(max_fruits=64):
    generator = torch.Generator().manual_seed(20260812)
    positions = torch.zeros(1, max_fruits, 2, dtype=torch.float32)
    velocities = torch.zeros_like(positions)
    angular_velocities = torch.zeros(1, max_fruits, dtype=torch.float32)
    levels = torch.zeros(1, max_fruits, dtype=torch.int64)
    physics_radii = torch.zeros(1, max_fruits, dtype=torch.float32)
    age_frames = torch.zeros(1, max_fruits, dtype=torch.int64)
    active = torch.zeros(1, max_fruits, dtype=torch.int64)
    active_count = 12
    active[0, :active_count] = 1
    levels[0, :active_count] = torch.randint(
        1, 9, (active_count,), generator=generator
    )
    positions[0, :active_count, 0] = torch.linspace(70.0, 490.0, active_count)
    positions[0, :active_count, 1] = torch.linspace(1030.0, 650.0, active_count)
    physics_radii[0, :active_count] = torch.tensor(
        (20, 30, 40, 45, 55, 70, 70, 100, 20, 30, 40, 45),
        dtype=torch.float32,
    )
    velocities[0, :active_count] = torch.randn(
        active_count, 2, generator=generator
    ) * 5.0
    angular_velocities[0, :active_count] = torch.randn(
        active_count, generator=generator
    ) * 0.2
    age_frames[0, :active_count] = torch.arange(20, 20 + active_count)
    fruit_queue = torch.tensor(((3, 1, 5, 2),), dtype=torch.int64)
    danger_progress = torch.tensor((0.25,), dtype=torch.float32)
    over_danger_line = torch.tensor((0,), dtype=torch.int64)
    return (
        positions,
        velocities,
        angular_velocities,
        levels,
        physics_radii,
        age_frames,
        active,
        fruit_queue,
        danger_progress,
        over_danger_line,
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def export(checkpoint, output_dir, *, opset=18):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'sab_t120.onnx'
    metadata_path = output_dir / 'sab_t120.metadata.json'

    loaded = load_viewer_model(checkpoint, device='cpu')
    policy = AndroidPolicy(loaded.model).eval()
    inputs = sample_inputs(loaded.model_config.max_fruits)
    with torch.inference_mode():
        expected = policy(*inputs).detach().cpu().numpy()

    torch.onnx.export(
        policy,
        inputs,
        model_path,
        input_names=list(INPUT_NAMES),
        output_names=['q_values'],
        opset_version=int(opset),
        dynamo=False,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(
        str(model_path), providers=('CPUExecutionProvider',)
    )
    ort_inputs = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(INPUT_NAMES, inputs)
    }
    actual = session.run(('q_values',), ort_inputs)[0]
    max_error = float(np.max(np.abs(expected - actual)))
    pytorch_action = int(expected.argmax(axis=1)[0])
    onnx_action = int(actual.argmax(axis=1)[0])
    if pytorch_action != onnx_action:
        raise RuntimeError(
            f'ONNX argmax mismatch: PyTorch={pytorch_action}, ONNX={onnx_action}'
        )
    if max_error > 1e-4:
        raise RuntimeError(f'ONNX maximum absolute error is too large: {max_error}')

    metadata = {
        'format_version': 1,
        'model_alias': 'SAB-T120',
        'formal_model_id': 'structured-128m-to-120fps-transfer-r1',
        'checkpoint': str(loaded.checkpoint_path),
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'onnx_file': model_path.name,
        'onnx_sha256': sha256_file(model_path),
        'onnx_size_bytes': model_path.stat().st_size,
        'opset': int(opset),
        'physics_fps': 120,
        'max_fruits': loaded.model_config.max_fruits,
        'action_count': loaded.model_config.action_count,
        'queue_length': loaded.model_config.queue_length,
        'inputs': {
            item.name: {
                'shape': list(item.shape),
                'dtype': item.type,
            }
            for item in session.get_inputs()
        },
        'output': {
            'name': session.get_outputs()[0].name,
            'shape': list(session.get_outputs()[0].shape),
            'dtype': session.get_outputs()[0].type,
        },
        'validation': {
            'sample_count': 1,
            'pytorch_argmax': pytorch_action,
            'onnx_argmax': onnx_action,
            'max_absolute_error': max_error,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return metadata_path, metadata


def main():
    args = parse_args()
    metadata_path, metadata = export(
        args.checkpoint, args.output_dir, opset=args.opset
    )
    print(json.dumps({
        'model': str(metadata_path.parent / metadata['onnx_file']),
        'metadata': str(metadata_path),
        'onnx_sha256': metadata['onnx_sha256'],
        'max_absolute_error': metadata['validation']['max_absolute_error'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
