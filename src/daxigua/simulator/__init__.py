"""高吞吐批量物理模拟器。

``daxigua.core`` 继续只依赖 Python 标准库；只有显式导入本包时才
需要 PyTorch。
"""

from .config import SimulatorConfig
from .reward import GameScoreReward, RewardComputer, ZeroReward
from .spatial_reward import (
    AccessibleSpaceBatch,
    AccessibleSpaceCalculator,
    SpatialRewardComputer,
    SpatialRewardConfig,
    SpatialRewardDiagnostics,
    SpatialRewardStep,
    build_standard_compensation_table,
    diagnose_spatial_reward,
)
from .replay import (
    DEFAULT_FRUIT_TEXTURE_DIR,
    load_fruit_texture_data_urls,
    load_trace_archive,
    save_trace_archive,
    trace_to_payload,
    write_replay_catalog,
    write_replay_fragment,
    write_replay_html,
    write_replay_payload_html,
)
from .scenario_lab import (
    fruit_specs,
    render_scenario_lab_html,
    write_scenario_lab_html,
)
from .scenario_lab_server import ScenarioLabServer
from .scenario_lab_service import ScenarioLabEvaluator, validate_scenario
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
    'AccessibleSpaceBatch',
    'AccessibleSpaceCalculator',
    'GameScoreReward',
    'DEFAULT_FRUIT_TEXTURE_DIR',
    'load_fruit_texture_data_urls',
    'load_trace_archive',
    'RewardComputer',
    'ScenarioLabEvaluator',
    'ScenarioLabServer',
    'SimulatorConfig',
    'SingleEnvAdapter',
    'SpatialRewardComputer',
    'SpatialRewardConfig',
    'SpatialRewardDiagnostics',
    'SpatialRewardStep',
    'build_standard_compensation_table',
    'diagnose_spatial_reward',
    'fruit_specs',
    'save_trace_archive',
    'render_scenario_lab_html',
    'TensorVectorSimulator',
    'trace_to_payload',
    'VectorEnv',
    'write_replay_fragment',
    'write_replay_html',
    'write_replay_payload_html',
    'write_replay_catalog',
    'write_scenario_lab_html',
    'validate_scenario',
    'ZeroReward',
]
