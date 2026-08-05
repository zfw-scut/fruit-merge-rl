"""单进程候选规模的端到端训练性能测量。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import torch

from daxigua.simulator import SimulatorConfig, TensorVectorSimulator

from .config import DqnConfig, ModelConfig
from .learner import DqnLearner
from .model import BaselineGnnDqn
from .observations import TensorState
from .replay import GpuReplayBuffer


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkResult:
    num_envs: int
    batch_size: int
    measured_steps: int
    pure_physics_env_steps_per_second: float
    actor_states_per_second: float
    learner_samples_per_second: float
    end_to_end_env_steps_per_second: float
    end_to_end_updates_per_second: float
    peak_memory_allocated_mb: float
    peak_memory_reserved_mb: float
    projected_peak_memory_mb: float
    total_memory_mb: float
    projected_replay_capacity: int
    replay_memory_mb: float
    benchmark_replay_memory_mb: float
    device_name: str
    use_bfloat16: bool
    compile_model: bool
    profile_trace: str | None

    def to_dict(self):
        return asdict(self)


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def _random_transition(simulator, replay):
    current = TensorState.from_observation(
        simulator.observe(), physics_fps=simulator.config.physics_fps
    )
    actions = torch.randint(
        simulator.config.action_count,
        (simulator.num_envs,),
        device=simulator.device,
    )
    ticket = replay.begin_append(current)
    result = simulator.step(actions)
    next_state = TensorState.from_observation(
        result.observation, physics_fps=simulator.config.physics_fps
    )
    replay.finish_append(
        ticket,
        next_state,
        actions,
        result.physics.score_delta.to(torch.float32) / 66.0,
        result.physics.done,
    )
    simulator.reset(result.physics.done)


def benchmark_training_candidate(
        *,
        num_envs,
        batch_size,
        device='cuda',
        measured_steps=3,
        pre_roll_steps=8,
        use_bfloat16=True,
        compile_model=False,
        profiler_output=None):
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    model_config = ModelConfig()
    simulator_config = SimulatorConfig.training_fast(
        max_fruits=model_config.max_fruits,
        use_cuda_extension=device.type == 'cuda',
    )
    simulator = TensorVectorSimulator(
        num_envs, config=simulator_config, device=device
    )
    base_model = BaselineGnnDqn(model_config).to(device)
    learner = DqnLearner(
        base_model,
        DqnConfig(
            use_bfloat16=bool(use_bfloat16),
            compile_model=bool(compile_model),
        ),
    )
    model = learner.online_model
    replay = GpuReplayBuffer(
        max(num_envs * 4, batch_size * 4),
        max_fruits=model_config.max_fruits,
        device=device,
        physics_fps=simulator_config.physics_fps,
    )

    if device.type == 'cuda':
        stage = torch.arange(num_envs, device=device).remainder(4)
        targets = stage * int(pre_roll_steps)
        for step in range(max(0, int(targets.max().item()))):
            enabled = targets > step
            actions = torch.randint(21, (num_envs,), device=device)
            result = simulator.step_masked(actions, enabled)
            simulator.reset(enabled & result.physics.done)
    else:
        for _ in range(pre_roll_steps):
            actions = torch.randint(21, (num_envs,), device=device)
            result = simulator.step(actions)
            simulator.reset(result.physics.done)
    while len(replay) < batch_size:
        _random_transition(simulator, replay)

    state = TensorState.from_observation(
        simulator.observe(), physics_fps=simulator_config.physics_fps
    )
    model.eval()
    with torch.inference_mode():
        model(state)
    learner.update(replay, batch_size)
    _sync(device)
    _sync(device)
    actor_started = time.perf_counter()
    for _ in range(measured_steps):
        model(state)
    _sync(device)
    actor_elapsed = time.perf_counter() - actor_started

    actions = torch.randint(21, (num_envs,), device=device)
    _sync(device)
    physics_started = time.perf_counter()
    for _ in range(measured_steps):
        result = simulator.step(actions)
        simulator.reset(result.physics.done)
    _sync(device)
    physics_elapsed = time.perf_counter() - physics_started

    _sync(device)
    learner_started = time.perf_counter()
    learner.update(replay, batch_size)
    _sync(device)
    learner_elapsed = time.perf_counter() - learner_started

    update_credit = 0.0
    full_updates = 0
    profile_context = nullcontext(None)
    if profiler_output is not None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile_context = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        )
    _sync(device)
    full_started = time.perf_counter()
    with profile_context as profiler:
        for _ in range(measured_steps):
            current = TensorState.from_observation(
                simulator.observe(), physics_fps=simulator_config.physics_fps
            )
            model.eval()
            with torch.inference_mode():
                selected = model(current).argmax(dim=1)
            ticket = replay.begin_append(current)
            result = simulator.step(selected)
            next_state = TensorState.from_observation(
                result.observation, physics_fps=simulator_config.physics_fps
            )
            replay.finish_append(
                ticket,
                next_state,
                selected,
                result.physics.score_delta.to(torch.float32) / 66.0,
                result.physics.done,
            )
            simulator.reset(result.physics.done)
            update_credit += num_envs / batch_size
            updates = int(update_credit)
            update_credit -= updates
            for _ in range(updates):
                learner.update(replay, batch_size)
            full_updates += updates
        _sync(device)
    full_elapsed = time.perf_counter() - full_started
    profile_path = None
    if profiler_output is not None:
        profile_path = Path(profiler_output).resolve()
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(profile_path))
    if device.type == 'cuda':
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2
        total_memory = (
            torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        )
        device_name = torch.cuda.get_device_name(device)
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0
        total_memory = 0.0
        device_name = 'cpu'
    benchmark_replay_mb = replay.memory_bytes / 1024 ** 2
    projected_replay_capacity = min(
        2_097_152,
        max(1_048_576, 256 * num_envs),
    )
    replay_bytes_per_transition = replay.memory_bytes / replay.capacity
    projected_replay_mb = (
        replay_bytes_per_transition * projected_replay_capacity / 1024 ** 2
    )
    projected_peak_mb = (
        peak_reserved
        + max(0.0, projected_replay_mb - benchmark_replay_mb)
    )
    transitions = num_envs * measured_steps
    return PipelineBenchmarkResult(
        num_envs=num_envs,
        batch_size=batch_size,
        measured_steps=measured_steps,
        pure_physics_env_steps_per_second=(
            transitions / max(physics_elapsed, 1e-9)
        ),
        actor_states_per_second=(
            transitions / max(actor_elapsed, 1e-9)
        ),
        learner_samples_per_second=(
            batch_size / max(learner_elapsed, 1e-9)
        ),
        end_to_end_env_steps_per_second=(
            transitions / max(full_elapsed, 1e-9)
        ),
        end_to_end_updates_per_second=(
            full_updates / max(full_elapsed, 1e-9)
        ),
        peak_memory_allocated_mb=peak_allocated,
        peak_memory_reserved_mb=peak_reserved,
        projected_peak_memory_mb=projected_peak_mb,
        total_memory_mb=total_memory,
        projected_replay_capacity=projected_replay_capacity,
        replay_memory_mb=projected_replay_mb,
        benchmark_replay_memory_mb=benchmark_replay_mb,
        device_name=device_name,
        use_bfloat16=bool(use_bfloat16),
        compile_model=bool(compile_model),
        profile_trace=str(profile_path) if profile_path is not None else None,
    )
