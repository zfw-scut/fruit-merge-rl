"""基线训练的原子 checkpoint 与身份清单。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random

import torch


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
