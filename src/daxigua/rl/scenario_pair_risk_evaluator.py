"""场景实验室中的轻量水果对堵塞风险推理适配层。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock
import time

import torch

from daxigua.simulator.scenario_lab_service import validate_scenario

from .pair_risk import PairRiskModel, PairRiskModelConfig


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(device):
    if str(device) == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolved = torch.device(device)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available for pair-risk inference')
    return resolved


class ScenarioPairRiskEvaluator:
    """把一个场景展开为全部同级 L7～L11 水果对并批量预测。"""

    def __init__(self, checkpoint_path, *, device='cpu'):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f'pair-risk checkpoint not found: {self.checkpoint_path}'
            )
        self.device = _resolve_device(device)
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if checkpoint.get('model_type') != (
                'pair_conditioned_deep_sets_risk_v1'):
            raise ValueError('unsupported pair-risk checkpoint')
        self.model = PairRiskModel(
            PairRiskModelConfig(**checkpoint['model_config'])
        ).to(self.device)
        self.model.load_state_dict(
            checkpoint['model_state_dict'], strict=True
        )
        self.model.eval()
        dataset = checkpoint.get('dataset_manifest') or {}
        self.forecast_horizon = int(dataset.get('forecast_horizon', 24))
        self.identity = {
            'checkpoint': self.checkpoint_path.name,
            'checkpoint_path': str(self.checkpoint_path),
            'checkpoint_sha256': _checkpoint_sha256(
                self.checkpoint_path
            ),
            'device': str(self.device),
            'model_type': checkpoint['model_type'],
            'forecast_horizon': self.forecast_horizon,
            'levels': [7, 8, 9, 10, 11],
        }
        self._lock = Lock()

    @staticmethod
    def _candidate_pairs(scene):
        fruits = scene['fruits']
        return [
            (first, second)
            for first in range(len(fruits))
            for second in range(first + 1, len(fruits))
            if fruits[first]['level'] == fruits[second]['level']
            and 7 <= fruits[first]['level'] <= 11
        ]

    def _columns(self, scene, candidates):
        config = self.model.config
        count = len(candidates)
        capacity = int(config.max_fruits)
        fruit_count = len(scene['fruits'])
        if fruit_count > capacity:
            raise ValueError(
                'scene fruit count exceeds pair-risk model capacity'
            )

        positions = torch.zeros(
            (capacity, 2), dtype=torch.float32, device=self.device
        )
        levels = torch.zeros(
            capacity, dtype=torch.int64, device=self.device
        )
        radii = torch.zeros(
            capacity, dtype=torch.float32, device=self.device
        )
        ages = torch.zeros(
            capacity, dtype=torch.float32, device=self.device
        )
        active = torch.zeros(
            capacity, dtype=torch.bool, device=self.device
        )
        if fruit_count:
            fruits = scene['fruits']
            positions[:fruit_count] = torch.tensor(
                [(fruit['x'], fruit['y']) for fruit in fruits],
                dtype=torch.float32,
                device=self.device,
            )
            levels[:fruit_count] = torch.tensor(
                [fruit['level'] for fruit in fruits],
                dtype=torch.int64,
                device=self.device,
            )
            radii[:fruit_count] = torch.tensor(
                [fruit['physics_radius'] for fruit in fruits],
                dtype=torch.float32,
                device=self.device,
            )
            # age_frames 是物理帧数；先恢复为时间，再换算到训练帧率。
            age_scale = float(config.physics_fps) / float(scene['fps'])
            ages[:fruit_count] = torch.tensor(
                [fruit['age_frames'] * age_scale for fruit in fruits],
                dtype=torch.float32,
                device=self.device,
            )
            active[:fruit_count] = True

        batch_shape = (count, capacity)
        return {
            'positions': positions.unsqueeze(0).expand(count, -1, -1),
            'levels': levels.unsqueeze(0).expand(batch_shape),
            'physics_radii': radii.unsqueeze(0).expand(batch_shape),
            'age_frames': ages.unsqueeze(0).expand(batch_shape),
            'active': active.unsqueeze(0).expand(batch_shape),
            'fruit_queue': torch.tensor(
                scene['queue'], dtype=torch.int64, device=self.device
            ).unsqueeze(0).expand(count, -1),
            'danger_progress': torch.full(
                (count,), float(scene['danger_progress']),
                dtype=torch.float32, device=self.device,
            ),
            'over_danger_line': torch.full(
                (count,), bool(scene['over_danger_line']),
                dtype=torch.bool, device=self.device,
            ),
            'pair_slot_i': torch.tensor(
                [pair[0] for pair in candidates],
                dtype=torch.int64,
                device=self.device,
            ),
            'pair_slot_j': torch.tensor(
                [pair[1] for pair in candidates],
                dtype=torch.int64,
                device=self.device,
            ),
        }

    @torch.inference_mode()
    def evaluate(self, scene):
        scene = validate_scenario(scene)
        candidates = self._candidate_pairs(scene)
        started = time.perf_counter()
        if not candidates:
            probabilities = []
        else:
            with self._lock:
                logits = self.model(self._columns(scene, candidates))
                probabilities = torch.sigmoid(
                    logits.float()
                ).detach().cpu().tolist()
        fruits = scene['fruits']
        pairs = []
        for (first, second), probability in zip(
                candidates, probabilities):
            fruit_i = fruits[first]
            fruit_j = fruits[second]
            pairs.append({
                'fruit_id_i': int(fruit_i['id']),
                'fruit_id_j': int(fruit_j['id']),
                'level': int(fruit_i['level']),
                'probability': round(float(probability), 6),
            })
        pairs.sort(key=lambda item: item['probability'], reverse=True)
        return {
            'format_version': 1,
            'semantics': 'onset_within_forecast_horizon',
            'forecast_horizon': self.forecast_horizon,
            'inference_ms': round(
                (time.perf_counter() - started) * 1000.0, 3
            ),
            'eligible_levels': [7, 8, 9, 10, 11],
            'pair_count': len(pairs),
            'pairs': pairs,
            'model': self.identity,
        }


__all__ = ['ScenarioPairRiskEvaluator']
