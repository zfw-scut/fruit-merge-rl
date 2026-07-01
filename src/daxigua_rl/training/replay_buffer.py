"""DQN 使用的固定容量经验回放池。

ReplayBuffer 的职责很单一：

1. 保存 rollout 过程中产生的经验对象。
2. 容量满后覆盖最旧经验。
3. 训练时随机采样一批经验。

它不负责：

- 构建图。
- 选择动作。
- 计算 reward。
- 把图转换成 PyTorch tensor。
- 执行 DQN loss 或 optimizer step。

这样可以让采样、存储和训练更新保持分离，后续如果要替换成优先经验回放
或多进程采样，也能尽量少影响其它模块。
"""

from __future__ import annotations

import random

class ReplayBuffer:
    """固定容量环形 Replay Buffer。

    buffer 只负责保存和采样经验对象，不关心对象内部是 `Transition` 还是
    `TensorTransition`。这样训练主链路可以保存张量化经验，调试链路仍然可以
    保存框架无关经验。
    """

    def __init__(self, capacity=100_000, seed=None):
        """创建经验回放池。

        参数：
        - `capacity`: 最大保存多少条 transition。默认十万条。
        - `seed`: buffer 自己的随机种子，用于复现 sample 结果。
        """

        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be positive')

        self.capacity = capacity

        # 独立随机源避免和环境随机数、模型初始化随机数互相影响。
        self._rng = random.Random(seed)

        # `_items` 存储真实 transition。未满时按插入顺序 append；
        # 满了以后通过 `_next_index` 环形覆盖最旧数据。
        self._items = []
        self._next_index = 0

    def __len__(self):
        """返回当前已保存的 transition 数量。"""

        return len(self._items)

    @property
    def is_full(self):
        """ReplayBuffer 是否已经达到最大容量。"""

        return len(self) == self.capacity

    @property
    def remaining_capacity(self):
        """距离满容量还剩多少位置。"""

        return self.capacity - len(self)

    def clear(self):
        """清空所有经验，并重置写入位置。"""

        self._items.clear()
        self._next_index = 0

    def push(self, transition):
        """写入一条经验对象。

        如果 buffer 未满，直接追加到末尾。
        如果 buffer 已满，覆盖当前最旧位置，并把写入指针向后移动一格。
        """

        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._next_index] = transition

        self._next_index = (self._next_index + 1) % self.capacity

    def extend(self, transitions):
        """批量写入 transition，并返回实际写入数量。"""

        count = 0
        for transition in transitions:
            self.push(transition)
            count += 1
        return count

    def sample(self, batch_size):
        """随机无放回采样一批 transition。

        返回值是原始经验对象元组，而不是已经拼好的 GraphBatch。
        GraphBatch 拼接由 DQN trainer 在拿到当前 sample 后完成。
        """

        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if batch_size > len(self):
            raise ValueError(
                f'cannot sample batch_size={batch_size} from replay buffer with {len(self)} items'
            )

        return tuple(self._rng.sample(self._items, batch_size))

    def is_ready(self, batch_size):
        """当前经验数量是否足够采样一个 batch。"""

        return len(self) >= int(batch_size)

    def to_tuple(self):
        """按时间顺序返回当前保存的所有 transition。

        这个方法主要用于调试和测试。训练采样应使用 `sample()`。
        """

        if len(self._items) < self.capacity:
            return tuple(self._items)

        # buffer 满了以后，`_next_index` 指向下一次要覆盖的位置，
        # 也就是当前最旧 transition 的位置。
        return tuple(self._items[self._next_index:] + self._items[:self._next_index])
