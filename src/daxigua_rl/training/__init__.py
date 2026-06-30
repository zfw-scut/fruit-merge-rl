"""强化学习训练侧数据结构入口。

本包只放训练系统自己的概念，例如经验记录、回放池和更新器。
游戏本体不得 import 本包；`daxigua_rl` 内部后续训练代码可以使用这里的结构。

注意：`RolloutCollector` 依赖 PyTorch 模型前向。为了避免普通 `daxigua_rl`
导入被 torch 依赖拖住，这里使用 `__getattr__` 懒加载 collector 相关对象。
"""

from .transition import Transition
from .replay_buffer import ReplayBuffer


__all__ = [
    'EpsilonGreedyPolicy',
    'ReplayBuffer',
    'RolloutCollector',
    'RolloutStats',
    'Transition',
]


def __getattr__(name):
    """懒加载依赖 torch 的 rollout 采集组件。"""

    if name in {'EpsilonGreedyPolicy', 'RolloutCollector', 'RolloutStats'}:
        from .collector import EpsilonGreedyPolicy, RolloutCollector, RolloutStats

        exports = {
            'EpsilonGreedyPolicy': EpsilonGreedyPolicy,
            'RolloutCollector': RolloutCollector,
            'RolloutStats': RolloutStats,
        }
        return exports[name]

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
