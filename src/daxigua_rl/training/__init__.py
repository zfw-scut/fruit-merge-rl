"""强化学习训练侧数据结构入口。

本包只放训练系统自己的概念，例如经验记录、回放池和更新器。
游戏本体不得 import 本包；`daxigua_rl` 内部后续训练代码可以使用这里的结构。
"""

from .transition import Transition


__all__ = [
    'Transition',
]
