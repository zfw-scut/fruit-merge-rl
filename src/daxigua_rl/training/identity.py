"""并行 rollout 和状态归因共用的稳定轨迹身份。"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index


@dataclass(frozen=True, order=True)
class TransitionKey:
    """一次训练 run 内唯一标识一个动作前状态和动作。"""

    worker_id: int
    episode_id: int
    step_index: int

    def __post_init__(self):
        """规范化整数类型并拒绝会产生歧义的负编号。"""

        object.__setattr__(self, 'worker_id', index(self.worker_id))
        object.__setattr__(self, 'episode_id', index(self.episode_id))
        object.__setattr__(self, 'step_index', index(self.step_index))

        if self.worker_id < 0:
            raise ValueError('worker_id must be non-negative')
        if self.episode_id < 0:
            raise ValueError('episode_id must be non-negative')
        if self.step_index < 0:
            raise ValueError('step_index must be non-negative')

    def as_tuple(self):
        """返回文档约定的 `(worker_id, episode_id, step_index)`。"""

        return self.worker_id, self.episode_id, self.step_index
