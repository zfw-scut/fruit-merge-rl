"""DQN 训练使用的张量化经验记录。

`TensorTransition` 是当前训练主链路唯一使用的经验结构。它保存已经转好的
CPU `GraphTensor`，方便 replay buffer 采样后直接拼成 `GraphBatch`。
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index

from daxigua_rl.graph.tensor import GraphTensor

from .structural_targets import StructuralTarget


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

    # 下一状态图。只有真实 terminal transition 可以为 None；
    # truncated transition 仍需保存可信 final observation 供 bootstrap。
    next_graph: GraphTensor | None

    # 游戏规则意义上的终止。
    terminated: bool

    # 环境流程意义上的截断。
    truncated: bool

    # 当前 reward 中实际累计了多少个连续环境步。
    #
    # 单步经验保持为 1；3-step accumulator 产生的经验通常为 3，episode
    # 尾部则可能为 1 或 2。DQN bootstrap 必须使用 gamma ** bootstrap_steps，
    # 不能把缩短的 terminal/truncated 尾部仍按固定 gamma ** 3 折扣。
    bootstrap_steps: int = 1

    # 当前动作的一步结构监督。它不随 n-step reward 累计；聚合经验始终保留
    # 窗口起始动作自己的 target。None 保持旧 replay/调用方完全兼容。
    structural_target: StructuralTarget | None = None

    def __post_init__(self):
        """做轻量一致性检查，避免训练时才发现图和动作错位。"""

        object.__setattr__(self, 'action_offset', int(self.action_offset))
        object.__setattr__(self, 'reward', float(self.reward))
        object.__setattr__(self, 'terminated', bool(self.terminated))
        object.__setattr__(self, 'truncated', bool(self.truncated))
        if (
                self.structural_target is not None
                and not isinstance(
                    self.structural_target,
                    StructuralTarget,
                )):
            raise TypeError(
                'structural_target must be StructuralTarget or None'
            )
        if isinstance(self.bootstrap_steps, bool):
            raise TypeError('bootstrap_steps must be an integer')
        try:
            bootstrap_steps = index(self.bootstrap_steps)
        except TypeError as exc:
            raise TypeError(
                'bootstrap_steps must be an integer'
            ) from exc
        if bootstrap_steps <= 0:
            raise ValueError('bootstrap_steps must be positive')
        object.__setattr__(self, 'bootstrap_steps', bootstrap_steps)

        if not isinstance(self.graph, GraphTensor):
            raise TypeError(f'graph must be GraphTensor, got {type(self.graph)!r}')
        if self.next_graph is not None and not isinstance(self.next_graph, GraphTensor):
            raise TypeError(f'next_graph must be GraphTensor or None, got {type(self.next_graph)!r}')
        if self.terminated and self.truncated:
            raise ValueError(
                'transition cannot be both terminated and truncated'
            )

        if self.action_count <= 0:
            raise ValueError('graph must contain at least one action node')
        if int(self.graph.action_indices.shape[0]) != self.action_count:
            raise ValueError('graph.action_indices length must match graph.action_node_indices length')
        if self.action_offset < 0 or self.action_offset >= self.action_count:
            raise IndexError(
                f'action_offset out of range: {self.action_offset}, action_count={self.action_count}'
            )

        if self.next_graph is None:
            if not self.terminated:
                raise ValueError('non-terminated transition must provide next_graph')
            return

        if not self.terminated and self.next_action_count <= 0:
            raise ValueError('non-terminated next_graph must contain at least one action node')

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

        return self.next_graph is not None and not self.terminated
