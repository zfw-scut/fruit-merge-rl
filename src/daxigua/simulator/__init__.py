"""高吞吐批量物理模拟器。

``daxigua.core`` 继续只依赖 Python 标准库；只有显式导入本包时才
需要 PyTorch。
"""

from .config import SimulatorConfig
from .reward import GameScoreReward, RewardComputer, ZeroReward
from .replay import trace_to_payload, write_replay_fragment, write_replay_html
from .types import (
    BatchDropResult,
    BatchMergeEvents,
    BatchObservation,
    BatchPhysicsResult,
    BatchSimulationTrace,
    BatchStepResult,
)
from .vector import SingleEnvAdapter, TensorVectorSimulator, VectorEnv

__all__ = [
    'BatchDropResult',
    'BatchMergeEvents',
    'BatchObservation',
    'BatchPhysicsResult',
    'BatchSimulationTrace',
    'BatchStepResult',
    'GameScoreReward',
    'RewardComputer',
    'SimulatorConfig',
    'SingleEnvAdapter',
    'TensorVectorSimulator',
    'trace_to_payload',
    'VectorEnv',
    'write_replay_fragment',
    'write_replay_html',
    'ZeroReward',
]
