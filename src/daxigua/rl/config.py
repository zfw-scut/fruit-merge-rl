"""第一版基线模型和训练系统的冻结配置。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True, slots=True)
class ModelConfig:
    max_fruits: int = 64
    action_count: int = 21
    queue_length: int = 4
    hidden_dim: int = 128
    edge_hidden_dim: int = 128
    message_layers: int = 3
    queue_hidden_dim: int = 64
    queue_layers: int = 1
    level_embedding_dim: int = 16
    max_neighbors: int = 12
    nearest_neighbors: int = 4
    motion_neighbors: int = 2
    vertical_neighbors_per_direction: int = 2
    action_key_fruits: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        integer_names = (
            'max_fruits', 'action_count', 'queue_length', 'hidden_dim',
            'edge_hidden_dim', 'message_layers', 'queue_hidden_dim',
            'queue_layers', 'level_embedding_dim', 'max_neighbors',
            'nearest_neighbors', 'motion_neighbors',
            'vertical_neighbors_per_direction', 'action_key_fruits',
        )
        for name in integer_names:
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.queue_length != 4:
            raise ValueError('queue_length must be 4')
        if self.action_count != 21:
            raise ValueError('action_count must be 21')
        if self.max_neighbors > self.max_fruits:
            raise ValueError('max_neighbors cannot exceed max_fruits')
        if self.action_key_fruits > self.max_fruits:
            raise ValueError('action_key_fruits cannot exceed max_fruits')
        if self.dropout != 0.0:
            raise ValueError('the first baseline keeps dropout disabled')


@dataclass(frozen=True, slots=True)
class DqnConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-4
    huber_delta: float = 1.0
    grad_clip_norm: float = 10.0
    target_update_interval: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.40
    utd_ratio: float = 1.0
    use_bfloat16: bool = True
    fused_adam: bool = True
    compile_model: bool = False
    compile_mode: str = 'reduce-overhead'

    def __post_init__(self):
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError('gamma must be in [0, 1]')
        if self.learning_rate <= 0.0:
            raise ValueError('learning_rate must be positive')
        if self.huber_delta <= 0.0 or self.grad_clip_norm <= 0.0:
            raise ValueError('loss and clipping thresholds must be positive')
        if self.target_update_interval <= 0:
            raise ValueError('target_update_interval must be positive')
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError('epsilon values are invalid')
        if not 0.0 < self.epsilon_decay_fraction <= 1.0:
            raise ValueError('epsilon_decay_fraction must be in (0, 1]')
        if self.utd_ratio <= 0.0:
            raise ValueError('utd_ratio must be positive')
        if self.compile_mode not in ('default', 'reduce-overhead', 'max-autotune'):
            raise ValueError('unsupported torch.compile mode')


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    capacity: int = 1_048_576
    batch_size: int = 512
    warmup_transitions: int = 262_144
    warmup_stage_ratios: tuple[float, float, float, float] = (
        0.30, 0.30, 0.25, 0.15
    )

    def __post_init__(self):
        if self.capacity <= 0 or self.batch_size <= 0:
            raise ValueError('replay sizes must be positive')
        if self.batch_size > self.capacity:
            raise ValueError('batch_size cannot exceed replay capacity')
        if not self.batch_size <= self.warmup_transitions <= self.capacity:
            raise ValueError('warmup_transitions must be inside replay capacity')
        if len(self.warmup_stage_ratios) != 4:
            raise ValueError('warmup_stage_ratios must contain four stages')
        if abs(sum(self.warmup_stage_ratios) - 1.0) > 1e-6:
            raise ValueError('warmup stage ratios must sum to one')


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    fast_interval_transitions: int = 2_097_152
    accurate_milestones: tuple[int, ...] = (
        10_000_000, 25_000_000, 50_000_000, 100_000_000
    )
    periodic_episodes: int = 512
    final_episodes: int = 4096
    parallel_envs: int = 512
    max_episode_drops: int = 1000
    seed_base: int = 32_000_000


@dataclass(frozen=True, slots=True)
class AnalysisExportConfig:
    transition_sample_size: int = 131_072
    transition_chunk_size: int = 8192
    trajectory_episodes: int = 512

    def __post_init__(self):
        if self.transition_sample_size < 0:
            raise ValueError('transition_sample_size cannot be negative')
        if self.transition_chunk_size <= 0:
            raise ValueError('transition_chunk_size must be positive')
        if self.trajectory_episodes < 0:
            raise ValueError('trajectory_episodes cannot be negative')


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    enabled: bool = True
    host: str = '127.0.0.1'
    port: int = 8765
    publish_interval_seconds: float = 1.0
    resource_interval_seconds: float = 2.0
    history_size: int = 3600


@dataclass(frozen=True, slots=True)
class AutoScaleConfig:
    enabled: bool = True
    candidate_envs: tuple[int, ...] = (
        1024, 2048, 4096, 8192, 16384, 32768
    )
    low_gpu_utilization: float = 80.0
    low_cpu_utilization: float = 75.0
    max_memory_utilization: float = 85.0
    observation_seconds: float = 120.0
    trial_seconds: float = 120.0
    cooldown_seconds: float = 600.0
    minimum_throughput_gain: float = 0.08
    rollback_loss: float = 0.05


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    run_dir: str = 'runs/gnn_dqn_baseline'
    device: str = 'cuda'
    seed: int = 20260805
    max_envs: int = 4096
    active_envs: int = 4096
    total_transitions: int = 50_000_000
    max_wall_seconds: float = 0.0
    finalization_reserve_seconds: float = 1800.0
    log_interval_seconds: float = 10.0
    checkpoint_interval_seconds: float = 1800.0
    max_episode_drops: int = 1000
    stage_pilot_envs: int = 256
    stage_pilot_max_drops: int = 128
    model: ModelConfig = field(default_factory=ModelConfig)
    dqn: DqnConfig = field(default_factory=DqnConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    analysis: AnalysisExportConfig = field(default_factory=AnalysisExportConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    autoscale: AutoScaleConfig = field(default_factory=AutoScaleConfig)

    def __post_init__(self):
        if self.max_envs <= 0 or self.active_envs <= 0:
            raise ValueError('environment counts must be positive')
        if self.active_envs > self.max_envs:
            raise ValueError('active_envs cannot exceed max_envs')
        if self.total_transitions <= 0:
            raise ValueError('total_transitions must be positive')
        if self.max_wall_seconds < 0.0:
            raise ValueError('max_wall_seconds cannot be negative')
        if self.stage_pilot_envs <= 0:
            raise ValueError('stage_pilot_envs must be positive')
        if self.stage_pilot_max_drops <= 0:
            raise ValueError('stage_pilot_max_drops must be positive')
        if self.model.max_fruits != 64:
            raise ValueError('the first baseline is frozen at 64 fruit slots')

    @classmethod
    def from_toml(cls, path):
        with Path(path).open('rb') as handle:
            data = tomllib.load(handle)
        root = dict(data.get('training', {}))
        return cls(
            **root,
            model=ModelConfig(**data.get('model', {})),
            dqn=DqnConfig(**data.get('dqn', {})),
            replay=ReplayConfig(**data.get('replay', {})),
            evaluation=EvaluationConfig(**data.get('evaluation', {})),
            analysis=AnalysisExportConfig(**data.get('analysis', {})),
            dashboard=DashboardConfig(**data.get('dashboard', {})),
            autoscale=AutoScaleConfig(**data.get('autoscale', {})),
        )

    def to_dict(self):
        return asdict(self)
