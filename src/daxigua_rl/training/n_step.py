"""按 worker 顺序把单步经验聚合为 n-step DQN 经验。

`NStepTransitionAccumulator` 只处理已经完成构图和 reward 计算的
`TensorTransition`。它必须由每个 rollout worker 独立、长期持有：

- deque 会跨多次 ``collect_steps()`` 调用保留；
- 不允许把多个 worker 或 episode 的随机 replay 样本事后拼接；
- terminal/truncated 会立即收口当前 episode 的全部尾部经验。

本模块不写 ReplayBuffer，也不知道 collector 的进程模型。调用方只需把每个真实环境
step 的单步 transition 按顺序传给 :meth:`append`，再把返回的零到多条聚合经验写入
主 replay。
"""

from __future__ import annotations

import math
from collections import deque
from operator import index

from .tensor_transition import TensorTransition


class NStepTransitionAccumulator:
    """把有序单步 transition 转换成固定上限的 n-step return。

    正常连续轨迹在积累满 ``n_step`` 条后发射一条经验，并保留最后
    ``n_step - 1`` 条作为下一个滑动窗口的前缀。遇到 episode 边界时立即发射所有
    缩短尾部，例如 n=3 时依次发射 3-step、2-step、1-step。

    truncated 是采集 episode 的边界，但不是 MDP terminal。因此 truncated 尾部仍
    保存最终可信 ``next_graph``，后续 DQN 会用 ``gamma ** bootstrap_steps``
    bootstrap；真实 terminal 尾部则没有 ``next_graph``。
    """

    def __init__(self, *, n_step=3, gamma=0.99):
        if isinstance(n_step, bool):
            raise TypeError('n_step must be an integer')
        try:
            n_step = index(n_step)
        except TypeError as exc:
            raise TypeError('n_step must be an integer') from exc
        if n_step <= 0:
            raise ValueError('n_step must be positive')

        if isinstance(gamma, bool):
            raise TypeError('gamma must be a real number')
        try:
            gamma = float(gamma)
        except (TypeError, ValueError) as exc:
            raise TypeError('gamma must be a real number') from exc
        if not math.isfinite(gamma) or gamma < 0.0 or gamma > 1.0:
            raise ValueError('gamma must be finite and in [0, 1]')

        self.n_step = n_step
        self.gamma = gamma
        self._pending = deque()

    def __len__(self):
        """返回尚未形成完整起点样本的单步 transition 数量。"""

        return len(self._pending)

    @property
    def pending_count(self):
        """``len(accumulator)`` 的显式指标别名。"""

        return len(self._pending)

    def append(self, transition):
        """追加一条单步经验，返回本次新形成的 n-step 经验。

        普通 step 最多返回一条；terminal/truncated 可能一次返回多条 episode 尾部。
        输入必须是原始单步经验，防止无意中对已经聚合的 reward 再聚合一次。
        """

        if not isinstance(transition, TensorTransition):
            raise TypeError(
                'transition must be TensorTransition, '
                f'got {type(transition)!r}'
            )
        if transition.bootstrap_steps != 1:
            raise ValueError(
                'NStepTransitionAccumulator only accepts one-step '
                'TensorTransition inputs'
            )

        self._pending.append(transition)
        if transition.done:
            return self._flush_all()

        if len(self._pending) < self.n_step:
            return ()

        emitted = self._aggregate_prefix(self.n_step)
        self._pending.popleft()
        return (emitted,)

    def extend(self, transitions):
        """顺序追加多条单步经验并合并返回结果。"""

        emitted = []
        for transition in transitions:
            emitted.extend(self.append(transition))
        return tuple(emitted)

    def flush(self):
        """在人工 reset/shutdown 前把当前非终止尾部按短 horizon 收口。

        正常采集调用边界不应调用本方法，因为 deque 必须跨 collect 调用保留。仅当调用
        方确定即将丢弃当前有序轨迹时使用；非终止尾部会从最后已知 ``next_graph``
        bootstrap。
        """

        return self._flush_all()

    def clear(self):
        """丢弃未发射尾部并返回丢弃数量。

        正式训练不应静默调用；该接口主要用于异常恢复和测试。
        """

        count = len(self._pending)
        self._pending.clear()
        return count

    def _flush_all(self):
        emitted = []
        while self._pending:
            emitted.append(self._aggregate_prefix(
                min(self.n_step, len(self._pending))
            ))
            self._pending.popleft()
        return tuple(emitted)

    def _aggregate_prefix(self, length):
        """把 deque 最前面的 ``length`` 条经验聚合成一条。"""

        window = tuple(
            self._pending[offset]
            for offset in range(length)
        )
        final = window[-1]
        discounted_reward = sum(
            (self.gamma ** offset) * transition.reward
            for offset, transition in enumerate(window)
        )
        return TensorTransition(
            graph=window[0].graph,
            action_offset=window[0].action_offset,
            reward=discounted_reward,
            next_graph=final.next_graph,
            terminated=final.terminated,
            truncated=final.truncated,
            bootstrap_steps=length,
            # 结构 target 是动作后的一步监督，绝不能像 reward 一样跨窗口求和。
            # 每条 n-step 经验只继承其起始动作对应的 target。
            structural_target=window[0].structural_target,
        )


__all__ = ['NStepTransitionAccumulator']
