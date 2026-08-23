"""基线训练的原子 checkpoint 与身份清单。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch

from daxigua.simulator.config import PHYSICS_IDENTITY


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _torch_rng_state():
    state = {
        'python': random.getstate(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    # `load_checkpoint(..., map_location='cuda')` 会把 payload 中包括 RNG
    # sidecar 在内的所有张量映射到 CUDA；PyTorch 的 RNG 恢复接口仍要求
    # CPU ByteTensor，因此在接口边界显式归一化设备。
    torch.set_rng_state(state['torch_cpu'].detach().cpu())
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            item.detach().cpu() for item in state['torch_cuda']
        ])


def save_checkpoint_atomic(
        path,
        *,
        learner,
        training_config,
        progress,
        replay_metadata,
        extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    payload = {
        'format_version': 1,
        'physics_identity': PHYSICS_IDENTITY,
        'learner': learner.state_dict(),
        'training_config': training_config.to_dict(),
        'progress': dict(progress),
        'replay_metadata': dict(replay_metadata),
        'rng_state': _torch_rng_state(),
        'extra': dict(extra or {}),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(path, *, map_location='cpu'):
    return torch.load(
        Path(path), map_location=map_location, weights_only=False
    )


def require_matching_physics_identity(checkpoint):
    """阻止完整恢复跨越不兼容的模拟器物理域。"""

    source_identity = checkpoint.get('physics_identity')
    if source_identity != PHYSICS_IDENTITY:
        rendered = source_identity or 'legacy_unspecified'
        raise ValueError(
            'checkpoint physics identity does not match current simulator: '
            f'{rendered!r} != {PHYSICS_IDENTITY!r}; start a new run with '
            'weights-only initialization instead of resume'
        )
    return source_identity


def initialize_learner_weights(
        learner,
        checkpoint_path,
        *,
        expected_model_config,
        map_location='cpu'):
    """只继承在线模型权重，并显式重建目标网络。

    该入口用于物理域或其它训练分布迁移。优化器、更新计数、Replay、RNG
    和训练进度都不得从来源 checkpoint 继承。
    """

    if learner.update_count != 0 or learner.optimizer.state:
        raise RuntimeError(
            'weights-only initialization requires a fresh learner'
        )
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(
        checkpoint_path, map_location=map_location
    )
    source_training_config = checkpoint.get('training_config')
    if not isinstance(source_training_config, dict):
        raise ValueError('source checkpoint has no training_config')
    source_model_config = source_training_config.get('model')
    expected_model_config = dict(expected_model_config)
    if source_model_config != expected_model_config:
        raise ValueError(
            'source checkpoint model config does not match target model config'
        )
    learner_state = checkpoint.get('learner')
    if (
            not isinstance(learner_state, dict)
            or 'online_model' not in learner_state):
        raise ValueError('source checkpoint has no online model weights')
    from .model import load_compatible_model_state_dict

    load_compatible_model_state_dict(
        learner.online_module, learner_state['online_model'], strict=True
    )
    learner.target_module.load_state_dict(
        learner.online_module.state_dict(), strict=True
    )
    learner.update_count = 0
    if learner.optimizer.state:
        raise RuntimeError(
            'weights-only initialization restored optimizer state'
        )
    source_progress_payload = checkpoint.get('progress', {})
    source_progress = {
        name: int(source_progress_payload.get(name, 0))
        for name in (
            'transitions', 'branch_transitions', 'branch_source_states',
            'updates', 'episodes',
        )
    }
    return {
        'kind': 'weights_only',
        'source_checkpoint': str(checkpoint_path),
        'source_checkpoint_sha256': sha256_file(checkpoint_path),
        'source_training_physics_fps': int(
            source_training_config.get('training_physics_fps', 30)
        ),
        'source_physics_identity': checkpoint.get(
            'physics_identity', 'legacy_unspecified'
        ),
        'target_physics_identity': PHYSICS_IDENTITY,
        'source_progress': source_progress,
        'source_model_config': source_model_config,
        'reset_components': [
            'target_model_from_online',
            'optimizer',
            'replay',
            'rng',
            'training_progress',
            'epsilon_schedule_progress',
        ],
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifact_manifest(run_dir, paths, *, metadata=None):
    run_dir = Path(run_dir)
    manifest = {
        'metadata': dict(metadata or {}),
        'files': [],
    }
    for path in paths:
        path = Path(path)
        if not path.exists() or not path.is_file():
            continue
        manifest['files'].append({
            'path': path.relative_to(run_dir).as_posix(),
            'bytes': path.stat().st_size,
            'sha256': sha256_file(path),
        })
    output = run_dir / 'artifact_manifest.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(
        json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    os.replace(temporary, output)
    return output
