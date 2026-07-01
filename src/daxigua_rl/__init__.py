"""RL package for automation and training.

The game package must not import this package. Automation code should depend on
stable interfaces exposed by daxigua, never the other way around.
"""

from .env import DaxiguaEnv, DaxiguaEnvConfig
from .reward import RewardBreakdown, RewardConfig, compute_reward
from .graph import (
    FeatureAblationConfig,
    FeatureMask,
    GraphAblator,
    GraphBuilder,
    GraphBuilderConfig,
    GraphData,
    get_ablation_preset,
)
from .training import ReplayBuffer


__all__ = [
    'DaxiguaEnv',
    'DaxiguaEnvConfig',
    'FeatureAblationConfig',
    'FeatureMask',
    'GraphAblator',
    'GraphBuilder',
    'GraphBuilderConfig',
    'GraphData',
    'ReplayBuffer',
    'RewardBreakdown',
    'RewardConfig',
    'compute_reward',
    'get_ablation_preset',
]
