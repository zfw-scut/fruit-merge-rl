"""DQN 训练使用的张量化经验记录。

`TensorTransition` 是当前训练主链路唯一使用的经验结构。它保存已经转好的
CPU `GraphTensor`，方便 replay buffer 采样后直接拼成 `GraphBatch`。
"""

from __future__ import annotations

from dataclasses import dataclass

from daxigua_rl.graph.tensor import GraphTensor


@dataclass(frozen=True)
class TensorTransition:
    """一次环境动作产生的一条张量化训练经验。

    约定：
    - replay buffer 长期保存 CPU `GraphTensor`；
    - 图特征由 `RolloutCollector` 固定为 float16，以降低常驻内存；
    - 训练时再把 collate 后的 `GraphBatch` 搬到模型设备；
    - 训练入口不再保留旧 GraphData transition 兼容路径。
    """

    # 当前状态图，已经是 PyTorch 张量格式。
    graph: GraphTensor

    # 被选择动作在当前候选动作列表中的位置，同时也是当前图 Q 值中的下标。
    action_offset: int

    # 执行动作后的即时奖励。
    reward: float

    # 下一状态图。terminal/truncated transition 可以为 None。
    next_graph: GraphTensor | None

    # 游戏规则意义上的终止。
    terminated: bool

    # 环境流程意义上的截断。
    truncated: bool

    def __post_init__(self):
        """做轻量一致性检查，避免训练时才发现图和动作错位。"""

        object.__setattr__(self, 'action_offset', int(self.action_offset))
        object.__setattr__(self, 'reward', float(self.reward))
        object.__setattr__(self, 'terminated', bool(self.terminated))
        object.__setattr__(self, 'truncated', bool(self.truncated))

        if not isinstance(self.graph, GraphTensor):
            raise TypeError(f'graph must be GraphTensor, got {type(self.graph)!r}')
        if self.next_graph is not None and not isinstance(self.next_graph, GraphTensor):
            raise TypeError(f'next_graph must be GraphTensor or None, got {type(self.next_graph)!r}')

        if self.action_count <= 0:
            raise ValueError('graph must contain at least one action node')
        if int(self.graph.action_indices.shape[0]) != self.action_count:
            raise ValueError('graph.action_indices length must match graph.action_node_indices length')
        if self.action_offset < 0 or self.action_offset >= self.action_count:
            raise IndexError(
                f'action_offset out of range: {self.action_offset}, action_count={self.action_count}'
            )

        if self.next_graph is None:
            if not self.done:
                raise ValueError('non-terminal transition must provide next_graph')
            return

        if not self.done and self.next_action_count <= 0:
            raise ValueError('non-terminal next_graph must contain at least one action node')

    @property
    def action_count(self):
        """当前状态下候选动作数量。"""

        return int(self.graph.action_node_indices.shape[0])

    @property
    def next_action_count(self):
        """下一状态下候选动作数量；terminal 且无 next_graph 时返回 0。"""

        if self.next_graph is None:
            return 0
        return int(self.next_graph.action_node_indices.shape[0])

    @property
    def action_node_index(self):
        """被选择动作对应的 action 节点在当前图中的行号。"""

        return int(self.graph.action_node_indices[self.action_offset].item())

    @property
    def action_index(self):
        """被选择动作的环境动作编号。"""

        return int(self.graph.action_indices[self.action_offset].item())

    @property
    def done(self):
        """是否到达 episode 边界。"""

        return self.terminated or self.truncated

    @property
    def can_bootstrap(self):
        """DQN target 是否可以读取下一状态 Q 值。"""

        return self.next_graph is not None and not self.done
