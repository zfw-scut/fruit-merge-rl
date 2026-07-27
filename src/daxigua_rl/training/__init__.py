"""强化学习训练侧组件入口。

本包只放训练系统自己的概念，例如经验记录、回放池和更新器。
游戏本体不得 import 本包；`daxigua_rl` 内部后续训练代码可以使用这里的结构。

注意：`RolloutCollector` 和 `DQNTrainer` 依赖 PyTorch。为了避免普通
`daxigua_rl` 导入被 torch 依赖拖住，这里使用 `__getattr__` 懒加载相关对象。
"""

from .replay_buffer import ReplayBuffer


__all__ = [
    'CounterfactualCoordinator',
    'CounterfactualCoordinatorActivityStats',
    'CounterfactualCoordinatorPoll',
    'CounterfactualCoordinatorStats',
    'CounterfactualCoordinatorSubmission',
    'DQNTrainer',
    'DQNTrainerConfig',
    'DQNTrainStats',
    'EpsilonGreedyPolicy',
    'NStepTransitionAccumulator',
    'ParallelCollectHandle',
    'ParallelRolloutCollector',
    'ReplayBuffer',
    'RolloutCollector',
    'RolloutStats',
    'TensorTransition',
    'TransitionKey',
    'WorkerAttributionFinalization',
    'recommended_counterfactual_worker_count',
]


def __getattr__(name):
    """懒加载依赖 torch 的训练组件。"""

    if name in {
            'CounterfactualCoordinator',
            'CounterfactualCoordinatorActivityStats',
            'CounterfactualCoordinatorPoll',
            'CounterfactualCoordinatorStats',
            'CounterfactualCoordinatorSubmission',
            'recommended_counterfactual_worker_count'}:
        from .counterfactual_coordinator import (
            CounterfactualCoordinator,
            CounterfactualCoordinatorActivityStats,
            CounterfactualCoordinatorPoll,
            CounterfactualCoordinatorStats,
            CounterfactualCoordinatorSubmission,
            recommended_counterfactual_worker_count,
        )

        exports = {
            'CounterfactualCoordinator': CounterfactualCoordinator,
            'CounterfactualCoordinatorActivityStats': (
                CounterfactualCoordinatorActivityStats
            ),
            'CounterfactualCoordinatorPoll': (
                CounterfactualCoordinatorPoll
            ),
            'CounterfactualCoordinatorStats': (
                CounterfactualCoordinatorStats
            ),
            'CounterfactualCoordinatorSubmission': (
                CounterfactualCoordinatorSubmission
            ),
            'recommended_counterfactual_worker_count': (
                recommended_counterfactual_worker_count
            ),
        }
        return exports[name]

    if name in {'EpsilonGreedyPolicy', 'RolloutCollector', 'RolloutStats'}:
        from .collector import EpsilonGreedyPolicy, RolloutCollector, RolloutStats

        exports = {
            'EpsilonGreedyPolicy': EpsilonGreedyPolicy,
            'RolloutCollector': RolloutCollector,
            'RolloutStats': RolloutStats,
        }
        return exports[name]

    if name in {
            'ParallelCollectHandle',
            'ParallelRolloutCollector',
            'WorkerAttributionFinalization'}:
        from .parallel_collector import (
            ParallelCollectHandle,
            ParallelRolloutCollector,
            WorkerAttributionFinalization,
        )

        exports = {
            'ParallelCollectHandle': ParallelCollectHandle,
            'ParallelRolloutCollector': ParallelRolloutCollector,
            'WorkerAttributionFinalization': (
                WorkerAttributionFinalization
            ),
        }
        return exports[name]

    if name in {'DQNTrainer', 'DQNTrainerConfig', 'DQNTrainStats'}:
        from .dqn import DQNTrainer, DQNTrainerConfig, DQNTrainStats

        exports = {
            'DQNTrainer': DQNTrainer,
            'DQNTrainerConfig': DQNTrainerConfig,
            'DQNTrainStats': DQNTrainStats,
        }
        return exports[name]

    if name == 'NStepTransitionAccumulator':
        from .n_step import NStepTransitionAccumulator

        return NStepTransitionAccumulator

    if name == 'TensorTransition':
        from .tensor_transition import TensorTransition

        return TensorTransition

    if name == 'TransitionKey':
        from .identity import TransitionKey

        return TransitionKey

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
