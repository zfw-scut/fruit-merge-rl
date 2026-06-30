"""DQN 训练使用的一条经验记录。

`Transition` 是强化学习训练侧的最小数据单元，用来表达：

    当前状态图 -> 执行动作 -> 得到奖励 -> 进入下一状态图

它不包含神经网络张量，不依赖 PyTorch。这样 replay buffer 可以先保存
框架无关的 `GraphData`，等真正采样训练 batch 时再转换成模型需要的张量。
"""

from __future__ import annotations

from dataclasses import dataclass

from daxigua_rl.graph.schema import GraphData


@dataclass(frozen=True)
class Transition:
    """一次环境动作产生的一条训练经验。

    典型 DQN 更新会用到：

    - `graph`: 当前状态 s。
    - `action_offset`: 当前动作 a 在 `q_values` 里的下标。
    - `reward`: 执行动作后得到的即时奖励 r。
    - `next_graph`: 下一状态 s'。
    - `terminated` / `truncated`: 是否到达 episode 边界。

    注意：这里有意使用 `action_offset`，而不是只保存 `ActionCandidate.action_index`。
    因为模型输出是一个一维张量 `q_values`，训练 loss 需要直接读取：

        q_values[transition.action_offset]

    `ActionCandidate.action_index` 仍然可以通过 `transition.action_index` 属性读取，
    主要用于调试、日志和未来动作映射检查。
    """

    # 当前状态图。它必须包含 action 节点，因为训练时要从这些节点输出 Q 值。
    graph: GraphData

    # 被选择动作在当前候选动作列表中的位置，同时也是 q_values 中的下标。
    # 例如 action_offset=7 表示使用 q_values[7] 参与 TD loss。
    action_offset: int

    # 执行动作后得到的即时奖励。当前环境里默认来自 score_delta 和失败惩罚。
    reward: float

    # 下一状态图。非终止 transition 必须有 next_graph，DQN 需要用它计算 bootstrap 项。
    # 如果游戏已经结束，next_graph 可以为 None，因为 terminal target 不再读取下一状态 Q 值。
    next_graph: GraphData | None

    # 游戏规则意义上的终止，例如水果越过死亡线导致游戏结束。
    terminated: bool

    # 环境流程意义上的截断，例如本次 step 达到最大物理帧数仍未稳定。
    truncated: bool

    def __post_init__(self):
        """做轻量一致性检查，尽早发现训练数据错位。"""

        # dataclass(frozen=True) 默认不允许赋值；
        # 这里用 object.__setattr__ 做基础类型归一化，外部传入 numpy/int 等也能稳定使用。
        object.__setattr__(self, 'action_offset', int(self.action_offset))
        object.__setattr__(self, 'reward', float(self.reward))
        object.__setattr__(self, 'terminated', bool(self.terminated))
        object.__setattr__(self, 'truncated', bool(self.truncated))

        if not isinstance(self.graph, GraphData):
            raise TypeError(f'graph must be GraphData, got {type(self.graph)!r}')

        if self.next_graph is not None and not isinstance(self.next_graph, GraphData):
            raise TypeError(f'next_graph must be GraphData or None, got {type(self.next_graph)!r}')

        if self.action_count <= 0:
            raise ValueError('graph must contain at least one action node')

        if len(self.graph.action_indices) != self.action_count:
            raise ValueError('graph.action_indices length must match graph.action_node_indices length')

        if self.action_offset < 0 or self.action_offset >= self.action_count:
            raise IndexError(
                f'action_offset out of range: {self.action_offset}, action_count={self.action_count}'
            )

        # 非终止经验必须提供 next_graph，否则 DQN 无法计算
        # reward + gamma * max_a' Q(next_state, a')。
        if self.next_graph is None:
            if not self.done:
                raise ValueError('non-terminal transition must provide next_graph')
            return

        # 如果 next_graph 存在且不是终止边界，至少要包含一个下一步候选动作。
        # 否则 target network 无法从下一状态读取 max Q。
        if not self.done and self.next_action_count <= 0:
            raise ValueError('non-terminal next_graph must contain at least one action node')

    @property
    def action_count(self):
        """当前状态下候选动作数量，也就是当前 Q 网络输出长度。"""

        return len(self.graph.action_node_indices)

    @property
    def next_action_count(self):
        """下一状态下候选动作数量；terminal 且无 next_graph 时返回 0。"""

        if self.next_graph is None:
            return 0
        return len(self.next_graph.action_node_indices)

    @property
    def action_node_index(self):
        """被选择动作对应的 action 节点在 `graph.node_features` 中的行号。"""

        return self.graph.action_node_indices[self.action_offset]

    @property
    def action_index(self):
        """被选择动作的环境动作编号。

        目前 `action_index` 与 `action_offset` 通常相同，但未来如果候选动作筛选、
        重排或动作空间扩展，它们可能不再相等。因此训练 loss 使用 `action_offset`，
        日志和动作映射检查可以使用这个属性。
        """

        return self.graph.action_indices[self.action_offset]

    @property
    def done(self):
        """是否到达 episode 边界。"""

        return self.terminated or self.truncated

    @property
    def can_bootstrap(self):
        """DQN target 是否可以读取下一状态 Q 值。

        第一版约定：只要 `terminated` 或 `truncated` 为 True，就不 bootstrap。
        如果后续引入普通时间限制型 truncation，可以在训练逻辑里重新定义策略。
        """

        return self.next_graph is not None and not self.done
