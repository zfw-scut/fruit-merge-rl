"""在场景实验室中对手工状态执行单次模型推理。"""

from __future__ import annotations

import math
from threading import Lock
import time

import torch

from daxigua.core import fruit_radius
from daxigua.simulator.config import SimulatorConfig
from daxigua.simulator.scenario_lab_service import validate_scenario

from .observations import TensorState
from .viewer import LoadedViewerModel


def _state_from_scene(
        scene,
        loaded,
        *,
        danger_progress=0.0,
        over_danger_line=None):
    """把已规范化的场景转换成基线 GNN-DQN 的单批状态。"""

    if not isinstance(loaded, LoadedViewerModel):
        raise TypeError('loaded must be LoadedViewerModel')
    device = loaded.device
    capacity = loaded.model_config.max_fruits
    fruit_count = len(scene['fruits'])
    if fruit_count > capacity:
        raise ValueError('scenario fruit count exceeds model capacity')

    positions = torch.zeros(
        (1, capacity, 2), dtype=torch.float32, device=device
    )
    velocities = torch.zeros_like(positions)
    angular_velocities = torch.zeros(
        (1, capacity), dtype=torch.float32, device=device
    )
    levels = torch.zeros(
        (1, capacity), dtype=torch.int64, device=device
    )
    physics_radii = torch.zeros(
        (1, capacity), dtype=torch.float32, device=device
    )
    age_frames = torch.zeros(
        (1, capacity), dtype=torch.int64, device=device
    )
    active = torch.zeros(
        (1, capacity), dtype=torch.bool, device=device
    )
    if fruit_count:
        slots = slice(0, fruit_count)
        positions[0, slots] = torch.tensor(
            [(fruit['x'], fruit['y']) for fruit in scene['fruits']],
            dtype=torch.float32,
            device=device,
        )
        velocities[0, slots] = torch.tensor(
            [(fruit['vx'], fruit['vy']) for fruit in scene['fruits']],
            dtype=torch.float32,
            device=device,
        )
        angular_velocities[0, slots] = torch.tensor(
            [fruit['angular_velocity'] for fruit in scene['fruits']],
            dtype=torch.float32,
            device=device,
        )
        levels[0, slots] = torch.tensor(
            [fruit['level'] for fruit in scene['fruits']],
            dtype=torch.int64,
            device=device,
        )
        physics_radii[0, slots] = torch.tensor(
            [fruit['physics_radius'] for fruit in scene['fruits']],
            dtype=torch.float32,
            device=device,
        )
        age_frames[0, slots] = torch.tensor(
            [fruit['age_frames'] for fruit in scene['fruits']],
            dtype=torch.int64,
            device=device,
        )
        active[0, slots] = True

    geometry = SimulatorConfig()
    inferred_over_danger_line = any(
        fruit['y'] - fruit['physics_radius'] < geometry.spawn_y
        for fruit in scene['fruits']
    )
    if over_danger_line is None:
        over_danger_line = inferred_over_danger_line
    return TensorState(
        positions=positions,
        velocities=velocities,
        angular_velocities=angular_velocities,
        levels=levels,
        physics_radii=physics_radii,
        age_frames=age_frames,
        active=active,
        fruit_queue=torch.tensor(
            [scene['queue']], dtype=torch.int64, device=device
        ),
        danger_progress=torch.tensor(
            [danger_progress], dtype=torch.float32, device=device
        ),
        over_danger_line=torch.tensor(
            [over_danger_line], dtype=torch.bool, device=device
        ),
        physics_fps=float(scene['fps']),
    )


def _drop_x(action, level, action_count):
    geometry = SimulatorConfig()
    radius = float(fruit_radius(int(level)))
    left = geometry.wall_width + radius + 2.0
    right = geometry.board_width - geometry.wall_width - radius - 2.0
    return left + (right - left) * int(action) / max(1, action_count - 1)


class ScenarioModelEvaluator:
    """对浏览器提交的状态做一次 greedy Q 网络推理，不改变物理世界。"""

    def __init__(self, loaded):
        if not isinstance(loaded, LoadedViewerModel):
            raise TypeError('loaded must be LoadedViewerModel')
        self.loaded = loaded
        self.device = loaded.device
        self._lock = Lock()

    @property
    def identity(self):
        progress = self.loaded.progress
        transitions = progress.get('transitions')
        if transitions is None:
            transitions = progress.get('total_transitions')
        return {
            'checkpoint': self.loaded.checkpoint_path.name,
            'checkpoint_sha256': self.loaded.checkpoint_sha256[:12],
            'device': str(self.loaded.device),
            'training_transitions': transitions,
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
        state = _state_from_scene(
            scene,
            self.loaded,
            danger_progress=danger_progress,
            over_danger_line=over_danger_line,
        )
        with self._lock:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            q_values = self.loaded.model(state)[0].float()
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            inference_ms = (time.perf_counter() - started) * 1000.0
            q_cpu = q_values.detach().cpu()
        if not bool(torch.isfinite(q_cpu).all().item()):
            raise FloatingPointError('model produced non-finite Q values')

        action = int(q_cpu.argmax().item())
        queue = list(scene['queue'])
        values = [round(float(value), 6) for value in q_cpu.tolist()]
        return {
            'format_version': 1,
            'policy': 'greedy',
            'action': action,
            'drop_x': round(_drop_x(
                action,
                queue[0],
                self.loaded.model_config.action_count,
            ), 3),
            'current_level': queue[0],
            'queue': queue,
            'fruit_count': len(scene['fruits']),
            'over_danger_line': bool(state.over_danger_line[0].item()),
            'danger_progress': round(danger_progress, 6),
            'q_values': values,
            'selected_q': round(float(q_cpu[action]), 6),
            'q_min': round(float(q_cpu.min()), 6),
            'q_mean': round(float(q_cpu.mean()), 6),
            'q_max': round(float(q_cpu.max()), 6),
            'inference_ms': round(inference_ms, 3),
            'model': self.identity,
            'message': (
                '模型仅对当前手工场景执行一次推理；未投放水果，也未修改实时世界。'
            ),
        }


__all__ = ['ScenarioModelEvaluator']
