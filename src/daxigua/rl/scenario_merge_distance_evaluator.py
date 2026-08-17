"""在场景实验室中执行逐水果合成步距预测。"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from threading import Lock
import time

import torch

from daxigua.simulator.scenario_lab_service import validate_scenario

from .merge_distance import MergeDistanceConfig, MergeDistancePredictor
from .scenario_model_evaluator import tensor_state_from_scene
from .viewer import resolve_viewer_device


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _class_interval(class_index, horizons):
    horizons = tuple(int(value) for value in horizons)
    if class_index < len(horizons):
        upper = horizons[class_index]
        lower = 1 if class_index == 0 else horizons[class_index - 1] + 1
        return lower, upper
    return None, None


def _class_label(class_index, config):
    if class_index == config.tail_class:
        return f'>{config.horizons[-1]} 次'
    if class_index == config.terminal_unmerged_class:
        return '终局前未再合成'
    lower, upper = _class_interval(class_index, config.horizons)
    if lower == upper:
        return f'{lower} 次'
    return f'{lower}–{upper} 次'


class ScenarioMergeDistanceEvaluator:
    """对当前场景逐水果推理，不改变实时物理世界。"""

    def __init__(self, checkpoint_path, *, device='auto'):
        checkpoint_path = Path(checkpoint_path).resolve()
        payload = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False
        )
        if payload.get('purpose') != 'merge_distance_predictor':
            raise ValueError('checkpoint is not a merge distance predictor')
        self.device = resolve_viewer_device(device)
        self.config = MergeDistanceConfig.from_dict(payload['model_config'])
        self.model = MergeDistancePredictor(
            self.config, **payload['geometry_config']
        )
        self.model.load_state_dict(payload['model_state'], strict=True)
        self.model.to(self.device).eval()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        self.checkpoint = payload
        self._lock = Lock()

    @property
    def identity(self):
        metrics = self.checkpoint.get('metrics') or {}
        summary_metrics = (
            metrics.get('test') or metrics.get('validation') or metrics
        )
        return {
            'checkpoint': self.checkpoint_path.name,
            'checkpoint_sha256': self.checkpoint_sha256[:12],
            'device': str(self.device),
            'parameter_count': sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
            'class_count': self.config.merge_class_count,
            'horizons': list(self.config.horizons),
            'exact_bin_accuracy': summary_metrics.get('exact_bin_accuracy'),
            'adjacent_bin_accuracy': summary_metrics.get(
                'adjacent_bin_accuracy'
            ),
        }

    @torch.inference_mode()
    def evaluate(
            self,
            scene,
            *,
            danger_progress=0.0,
            over_danger_line=None):
        scene = validate_scenario(scene)
        danger_progress = float(danger_progress)
        if not math.isfinite(danger_progress):
            raise ValueError('danger_progress must be finite')
        danger_progress = max(0.0, min(1.0, danger_progress))
        state = tensor_state_from_scene(
            scene,
            device=self.device,
            capacity=self.config.max_fruits,
            danger_progress=danger_progress,
            over_danger_line=over_danger_line,
        )
        with self._lock:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            probabilities = self.model(state).probabilities[0]
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - started) * 1000.0
            probabilities = probabilities[:len(scene['fruits'])].float().cpu()
        if not bool(torch.isfinite(probabilities).all().item()):
            raise FloatingPointError(
                'merge distance predictor produced non-finite probabilities'
            )

        predictions = []
        for fruit, distribution in zip(scene['fruits'], probabilities):
            class_index = int(distribution.argmax().item())
            lower, upper = _class_interval(
                class_index, self.config.horizons
            )
            predictions.append({
                'fruit_id': fruit['id'],
                'level': fruit['level'],
                'class_index': class_index,
                'label': _class_label(class_index, self.config),
                'min_steps': lower,
                'max_steps': upper,
                'is_tail': class_index == self.config.tail_class,
                'is_terminal_unmerged': (
                    class_index == self.config.terminal_unmerged_class
                ),
                'confidence': round(float(distribution[class_index]), 6),
                'eventual_merge_probability': round(float(
                    1.0 - distribution[
                        self.config.terminal_unmerged_class
                    ]
                ), 6),
                'probabilities': [
                    round(float(value), 6) for value in distribution.tolist()
                ],
            })
        return {
            'format_version': 1,
            'fruit_count': len(predictions),
            'inference_ms': round(inference_ms, 3),
            'predictions': predictions,
            'model': self.identity,
            'message': '合成步距预测只读取当前场景，不修改实时世界。',
        }


__all__ = ['ScenarioMergeDistanceEvaluator']
