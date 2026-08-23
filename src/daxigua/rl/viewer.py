"""加载基线 checkpoint，并生成可在浏览器观看的模型完整对局。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import time

import torch

from daxigua.core import fruit_name, fruit_radius
from daxigua.simulator import (
    SimulatorConfig,
    TensorVectorSimulator,
    save_trace_archive,
    trace_to_payload,
    write_replay_payload_html,
)

from .checkpoint import load_checkpoint, sha256_file
from .config import ModelConfig
from .model import BaselineGnnDqn, load_compatible_model_state_dict
from .observations import TensorState


@dataclass(frozen=True, slots=True)
class LoadedViewerModel:
    checkpoint_path: Path
    checkpoint_sha256: str
    model: BaselineGnnDqn
    model_config: ModelConfig
    progress: dict
    device: torch.device


@dataclass(frozen=True, slots=True)
class ViewerEpisode:
    traces: tuple
    decisions: tuple[dict, ...]
    summary: dict
    simulator_config: SimulatorConfig


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def resolve_viewer_device(device='auto'):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resolved = torch.device(device)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA viewer device is unavailable')
    if resolved.type == 'cuda' and resolved.index is None:
        resolved = torch.device('cuda', torch.cuda.current_device())
    return resolved


def viewer_simulator_config(physics_fps, model_config, device):
    common = {
        'max_fruits': model_config.max_fruits,
        'action_count': model_config.action_count,
        'queue_length': model_config.queue_length,
        'use_cuda_extension': torch.device(device).type == 'cuda',
    }
    if int(physics_fps) == 30:
        return SimulatorConfig.training_fast(**common)
    if int(physics_fps) == 120:
        return SimulatorConfig.high_fidelity_fast(**common)
    raise ValueError('viewer physics_fps must be 30 or 120')


def load_viewer_model(checkpoint_path, *, device='auto'):
    """按 checkpoint 内冻结配置重建 eager 在线网络。"""

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
    resolved_device = resolve_viewer_device(device)
    checkpoint = load_checkpoint(
        checkpoint_path, map_location=resolved_device
    )
    if checkpoint.get('format_version') != 1:
        raise ValueError('unsupported checkpoint format_version')
    try:
        model_values = checkpoint['training_config']['model']
        online_state = checkpoint['learner']['online_model']
    except (KeyError, TypeError) as error:
        raise ValueError(
            'checkpoint misses training model config or online weights'
        ) from error
    model_config = ModelConfig(**model_values)
    geometry = viewer_simulator_config(
        120, model_config, resolved_device
    )
    model = BaselineGnnDqn(
        model_config,
        board_width=geometry.board_width,
        board_height=geometry.board_height,
        spawn_y=geometry.spawn_y,
        wall_width=geometry.wall_width,
        gravity_y=geometry.gravity_y,
    ).to(resolved_device)
    load_compatible_model_state_dict(model, online_state, strict=True)
    model.eval()
    return LoadedViewerModel(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=sha256_file(checkpoint_path),
        model=model,
        model_config=model_config,
        progress=dict(checkpoint.get('progress') or {}),
        device=resolved_device,
    )


def _drop_x(config, current_level, action):
    radius = float(fruit_radius(int(current_level)))
    left = config.wall_width + radius + 2.0
    right = config.board_width - config.wall_width - radius - 2.0
    fraction = int(action) / max(1, config.action_count - 1)
    return left + fraction * (right - left)


@torch.inference_mode()
def run_viewer_episode(
        loaded,
        *,
        physics_fps=120,
        seed=20260805,
        max_drops=1000,
        frame_stride=2):
    """用 greedy 策略运行一局，并记录真实逐帧物理与决策信息。"""

    if not isinstance(loaded, LoadedViewerModel):
        raise TypeError('loaded must be LoadedViewerModel')
    if int(max_drops) <= 0:
        raise ValueError('max_drops must be positive')
    if int(frame_stride) <= 0:
        raise ValueError('frame_stride must be positive')
    if loaded.device.type != 'cuda':
        raise RuntimeError(
            'the first full-frame model viewer requires CUDA; '
            'CPU inference export can be added after the baseline is trained'
        )
    config = viewer_simulator_config(
        physics_fps, loaded.model_config, loaded.device
    )
    simulator = TensorVectorSimulator(
        1, config=config, device=loaded.device
    )
    simulator.reset(seeds=int(seed))
    warmup_state = TensorState.from_observation(
        simulator.observe(), physics_fps=config.physics_fps
    )
    loaded.model(warmup_state)
    torch.cuda.synchronize(loaded.device)
    trace_rows = torch.tensor(
        [0], dtype=torch.int64, device=loaded.device
    )
    traces = []
    decisions = []
    inference_times_ms = []
    terminated = False
    truncated = False
    started = time.perf_counter()
    for drop_index in range(int(max_drops)):
        observation = simulator.observe()
        state = TensorState.from_observation(
            observation, physics_fps=config.physics_fps
        )
        inference_started = time.perf_counter()
        q_values = loaded.model(state)[0].float()
        action = int(q_values.argmax().item())
        q_cpu = q_values.detach().cpu()
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        inference_times_ms.append(inference_ms)
        if not bool(torch.isfinite(q_cpu).all().item()):
            raise FloatingPointError('viewer model produced non-finite Q values')
        queue = [int(value) for value in state.fruit_queue[0].tolist()]
        decisions.append({
            'drop': drop_index + 1,
            'action': action,
            'drop_x': round(_drop_x(config, queue[0], action), 3),
            'current_level': queue[0],
            'current_fruit': fruit_name(queue[0]),
            'queue': queue,
            'q_values': [round(float(value), 6) for value in q_cpu.tolist()],
            'selected_q': round(float(q_cpu[action]), 6),
            'q_min': round(float(q_cpu.min()), 6),
            'q_mean': round(float(q_cpu.mean()), 6),
            'q_max': round(float(q_cpu.max()), 6),
            'danger_progress': round(
                float(state.danger_progress[0].item()), 6
            ),
            'score_before': int(observation.score[0].item()),
            'fruit_count_before': int(observation.fruit_count[0].item()),
            'inference_ms': round(inference_ms, 3),
        })
        actions = torch.tensor(
            [action], dtype=torch.int64, device=loaded.device
        )
        result, trace = simulator.step_with_trace(
            actions, trace_rows, frame_stride=int(frame_stride)
        )
        traces.append(trace.cpu())
        terminated = bool(result.physics.done[0].item())
        truncated = bool(result.physics.truncated[0].item())
        if terminated or truncated:
            break

    final = simulator.observe()
    elapsed_seconds = time.perf_counter() - started
    final_score = int(final.score[0].item())
    summary = {
        'seed': int(seed),
        'physics_fps': config.physics_fps,
        'frame_stride': int(frame_stride),
        'drops': len(traces),
        'score': final_score,
        'max_level': int(final.max_level[0].item()),
        'fruit_count': int(final.fruit_count[0].item()),
        'terminated': terminated,
        'truncated': truncated,
        'capped': not terminated and not truncated,
        'elapsed_seconds': elapsed_seconds,
        'mean_inference_ms': (
            statistics.fmean(inference_times_ms)
            if inference_times_ms else 0.0
        ),
        'max_inference_ms': max(inference_times_ms, default=0.0),
    }
    return ViewerEpisode(
        traces=tuple(traces),
        decisions=tuple(decisions),
        summary=summary,
        simulator_config=config,
    )


def viewer_episode_payload(episode, loaded):
    if not isinstance(episode, ViewerEpisode):
        raise TypeError('episode must be ViewerEpisode')
    payload = trace_to_payload(
        episode.traces, episode.simulator_config, compact=True
    )
    clip = payload['clips'][0]
    if len(clip['drop_summaries']) != len(episode.decisions):
        raise RuntimeError('decision count does not match replay drops')
    for drop_summary, decision in zip(
            clip['drop_summaries'], episode.decisions):
        drop_summary['decision'] = decision
    payload['model_viewer'] = {
        'checkpoint': loaded.checkpoint_path.name,
        'checkpoint_sha256': loaded.checkpoint_sha256,
        'checkpoint_progress': _json_safe(loaded.progress),
        'device': str(loaded.device),
        'policy': 'greedy',
        'action_count': loaded.model_config.action_count,
        'episode': episode.summary,
    }
    return payload


def write_viewer_episode(
        output_path,
        episode,
        loaded,
        *,
        trace_path=None,
        title='GNN-DQN 模型游玩'):
    """写出自包含游戏页面，并可选保存可重新渲染的物理追踪。"""

    payload = viewer_episode_payload(episode, loaded)
    output = write_replay_payload_html(
        output_path,
        payload,
        title=title,
        use_textures=True,
        compress_payload=True,
    )
    if trace_path is not None:
        save_trace_archive(trace_path, episode.traces)
    return output


__all__ = [
    'LoadedViewerModel',
    'ViewerEpisode',
    'load_viewer_model',
    'resolve_viewer_device',
    'run_viewer_episode',
    'viewer_episode_payload',
    'viewer_simulator_config',
    'write_viewer_episode',
]
