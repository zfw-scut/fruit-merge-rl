"""面向合成大西瓜结构信息的 GNN-Q 网络。

模型继续保持最简单的外部契约：

``forward(graph) -> q_values``

其中 ``q_values`` 始终是一维张量。需要训练结构辅助任务时，可改用
``forward_with_aux(graph)``，它会额外返回每个动作的六维结构预测。两条入口共享
同一次图编码和消息传播，不会为了辅助头重复计算 GNN。

图仍然使用统一的节点/边张量，但消息传递会显式利用边表示生成 relation gate 和
attention gate。Q 值采用 dueling 分解：每张图从 ``is_global_node`` 特征列定位
唯一 global 节点并预测状态值，再对该图自己的动作 advantage 做中心化。
"""

from typing import NamedTuple

import torch
from torch import nn

from daxigua_rl.graph.schema import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    GraphData,
)
from daxigua_rl.graph.tensor import GraphBatch, GraphTensor, graph_to_tensor


# 固定顺序是辅助监督的数据契约。后续 trainer 可以按列选择回归或分类损失，
# 不需要依赖字典迭代顺序。
STRUCTURE_PREDICTION_NAMES = (
    'top_connected_capacity_delta',
    'recoverability_delta',
    'chain_readiness_delta',
    'new_dead_or_blocked_fruit_risk',
    'sealed_cavity_delta',
    'realized_chain_or_terminal_risk',
)


class StructureAwareQOutput(NamedTuple):
    """``forward_with_aux`` 的稳定返回结构。

    ``q_values`` 的 shape 为 ``[total_action_count]``；
    ``structure_predictions`` 的 shape 为 ``[total_action_count, 6]``。
    GraphBatch 仍使用 ``graph.action_slices`` 把扁平动作维还原到每张图。
    """

    q_values: torch.Tensor
    structure_predictions: torch.Tensor

    @property
    def prediction_names(self):
        """返回六个辅助预测列的固定名称。"""

        return STRUCTURE_PREDICTION_NAMES


def _activation(name):
    """根据名称创建激活层。"""

    if name == 'relu':
        return nn.ReLU()
    if name == 'silu':
        return nn.SiLU()
    raise ValueError(f'unsupported activation: {name}')


def _mlp(input_dim, hidden_dim, output_dim, activation='silu', dropout=0.0):
    """创建项目内统一使用的两层 MLP。"""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        _activation(activation),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class MessagePassingLayer(nn.Module):
    """带 edge-conditioned relation gate/attention 的消息传递层。

    对每条 ``src -> dst`` 边：

    1. ``message_mlp`` 根据源节点、目标节点和边表示生成候选消息；
    2. ``relation_gate`` 仅由边表示生成逐通道门，学习不同关系应开放哪些通道；
    3. ``attention_gate`` 根据完整三元组生成标量权重，学习同类关系中的重要边；
    4. 按目标节点做加权平均，再通过残差和 LayerNorm 更新节点。

    sigmoid attention 配合加权平均不要求额外的 scatter-softmax 扩展，因而能在
    CPU rollout worker、CUDA trainer 和空边图上使用同一实现。
    """

    def __init__(self, hidden_dim, activation='silu', dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        message_input_dim = self.hidden_dim * 3
        self.message_mlp = _mlp(
            message_input_dim,
            self.hidden_dim,
            self.hidden_dim,
            activation,
            dropout,
        )
        self.relation_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(message_input_dim, self.hidden_dim),
            _activation(activation),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.update_mlp = _mlp(
            self.hidden_dim * 2,
            self.hidden_dim,
            self.hidden_dim,
            activation,
            dropout,
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_hidden, edge_index, edge_hidden):
        """传播一层消息并返回 ``[num_nodes, hidden_dim]``。"""

        if node_hidden.dim() != 2:
            raise ValueError(
                'node_hidden must have shape [num_nodes, hidden_dim]'
            )
        if edge_hidden.dim() != 2:
            raise ValueError(
                'edge_hidden must have shape [num_edges, hidden_dim]'
            )
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError('edge_index must have shape [2, num_edges]')
        if node_hidden.shape[1] != self.hidden_dim:
            raise ValueError(
                f'expected node hidden dim {self.hidden_dim}, '
                f'got {node_hidden.shape[1]}'
            )
        if edge_hidden.shape[1] != self.hidden_dim:
            raise ValueError(
                f'expected edge hidden dim {self.hidden_dim}, '
                f'got {edge_hidden.shape[1]}'
            )
        if edge_index.shape[1] != edge_hidden.shape[0]:
            raise ValueError(
                'edge_index and edge_hidden must contain the same '
                'number of edges'
            )

        if edge_index.numel() == 0:
            aggregated = torch.zeros_like(node_hidden)
        else:
            source_index = edge_index[0]
            target_index = edge_index[1]
            source_hidden = node_hidden[source_index]
            target_hidden = node_hidden[target_index]
            message_input = torch.cat(
                (source_hidden, target_hidden, edge_hidden),
                dim=-1,
            )

            messages = self.message_mlp(message_input)
            relation_weights = self.relation_gate(edge_hidden)
            attention_weights = self.attention_gate(message_input)
            weighted_messages = (
                messages
                * relation_weights
                * attention_weights
            )

            aggregated = torch.zeros_like(node_hidden)
            aggregated.index_add_(
                0,
                target_index,
                weighted_messages,
            )

            # 用 attention 权重而不是裸入度归一化。这样被 gate 抑制的边不会同时
            # 把其它有效消息按完整入度缩小。
            attention_sum = torch.zeros(
                node_hidden.shape[0],
                1,
                dtype=node_hidden.dtype,
                device=node_hidden.device,
            )
            attention_sum.index_add_(
                0,
                target_index,
                attention_weights,
            )
            epsilon = torch.finfo(node_hidden.dtype).eps
            aggregated = aggregated / attention_sum.clamp_min(epsilon)

        update = self.update_mlp(
            torch.cat((node_hidden, aggregated), dim=-1)
        )
        return self.norm(
            node_hidden + self.dropout(update)
        )


class GNNQNetwork(nn.Module):
    """输入状态图，输出所有候选动作的 dueling Q 值。

    构造参数保持与旧模型相同，因此训练脚本、rollout worker 和反事实 runner
    不需要修改。``q_head`` 这个历史属性也继续保留；在 dueling 结构中它承担
    global state-value 分支，动作 advantage 由 ``advantage_head`` 负责。
    """

    def __init__(
            self,
            node_feature_dim=None,
            edge_feature_dim=None,
            hidden_dim=128,
            message_layers=3,
            activation='silu',
            dropout=0.0):
        super().__init__()

        self.node_feature_dim = (
            node_feature_dim
            if node_feature_dim is not None
            else len(NODE_FEATURE_NAMES)
        )
        self.edge_feature_dim = (
            edge_feature_dim
            if edge_feature_dim is not None
            else len(EDGE_FEATURE_NAMES)
        )
        self.hidden_dim = int(hidden_dim)
        self.message_layers = int(message_layers)

        if self.node_feature_dim <= 0:
            raise ValueError('node_feature_dim must be positive')
        if self.edge_feature_dim <= 0:
            raise ValueError('edge_feature_dim must be positive')
        if self.hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')
        if self.message_layers < 0:
            raise ValueError('message_layers must be non-negative')

        self.node_encoder = _mlp(
            self.node_feature_dim,
            self.hidden_dim,
            self.hidden_dim,
            activation,
            dropout,
        )
        self.edge_encoder = _mlp(
            self.edge_feature_dim,
            self.hidden_dim,
            self.hidden_dim,
            activation,
            dropout,
        )
        self.layers = nn.ModuleList(
            MessagePassingLayer(
                self.hidden_dim,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(self.message_layers)
        )

        # 保留 q_head 名称以兼容既有测试和调试代码。dueling 结构里它读取每张图
        # 唯一 global 节点，输出 V(s)。
        self.q_head = _mlp(
            self.hidden_dim,
            self.hidden_dim,
            1,
            activation,
            dropout,
        )
        self.advantage_head = _mlp(
            self.hidden_dim,
            self.hidden_dim,
            1,
            activation,
            dropout,
        )

        # 辅助头同时读取 action 表示和同图 global 表示，使它既能看到落点局部
        # 结构，也能判断该变化相对整个局面的意义。
        self.structure_head = _mlp(
            self.hidden_dim * 2,
            self.hidden_dim,
            len(STRUCTURE_PREDICTION_NAMES),
            activation,
            dropout,
        )

    @property
    def value_head(self):
        """``q_head`` 的语义化只读别名，不重复注册同一个子模块。"""

        return self.q_head

    def forward(self, graph):
        """返回一维 Q 值，保持原有调用 API。"""

        (
            action_hidden,
            action_slices,
            global_hidden,
            _action_global_hidden,
        ) = self._encode_readout_context(graph)
        return self._dueling_q_values(
            action_hidden=action_hidden,
            action_slices=action_slices,
            global_hidden=global_hidden,
        )

    def forward_with_aux(self, graph):
        """返回 Q 值和每动作六维结构预测。

        六列与 ``StructuralTarget`` 的列顺序一致，并统一通过 tanh 映射到
        ``[-1, 1]``。风险列的有效 target 位于 ``[0, 1]``；封闭空腔变化量需要
        表达改善和恶化；最后一列用正值表达真实连锁、``-1`` 表达真实终局。
        """

        (
            action_hidden,
            action_slices,
            global_hidden,
            action_global_hidden,
        ) = self._encode_readout_context(graph)
        q_values = self._dueling_q_values(
            action_hidden=action_hidden,
            action_slices=action_slices,
            global_hidden=global_hidden,
        )

        raw_structure = self.structure_head(
            torch.cat(
                (action_hidden, action_global_hidden),
                dim=-1,
            )
        )
        structure_predictions = torch.tanh(raw_structure)

        return StructureAwareQOutput(
            q_values=q_values,
            structure_predictions=structure_predictions,
        )

    def _encode_readout_context(self, graph):
        """只执行一次图编码，并准备 Q/辅助头共用的读出表示。"""

        # 训练主链路的图通常仍在 CPU。先在原设备上按 schema 定位 global
        # 节点，再整体搬到 CUDA，避免每次 forward 为读取一个索引触发 GPU -> CPU
        # 同步。调用者若直接传入 GPU GraphTensor，仍支持同一逻辑。
        graph_tensor = self._as_tensor(graph)
        self._validate_graph_tensor(graph_tensor)
        (
            action_slices,
            global_node_indices,
        ) = self._locate_graph_global_nodes(graph_tensor)
        graph_tensor = self._move_tensor_to_model(graph_tensor)
        global_node_indices = global_node_indices.to(
            device=graph_tensor.node_features.device,
        )

        node_hidden = self.node_encoder(graph_tensor.node_features)
        edge_hidden = self.edge_encoder(graph_tensor.edge_features)
        for layer in self.layers:
            node_hidden = layer(
                node_hidden,
                graph_tensor.edge_index,
                edge_hidden,
            )

        action_hidden = node_hidden[
            graph_tensor.action_node_indices
        ]
        (
            global_hidden,
            action_global_hidden,
        ) = self._read_graph_global_hidden(
            graph_tensor,
            node_hidden,
            action_slices,
            global_node_indices,
        )
        return (
            action_hidden,
            action_slices,
            global_hidden,
            action_global_hidden,
        )

    def _dueling_q_values(
            self,
            *,
            action_hidden,
            action_slices,
            global_hidden):
        """从读出表示计算按图中心化的 dueling Q。"""

        state_values = self.q_head(global_hidden).squeeze(-1)
        advantages = self.advantage_head(action_hidden).squeeze(-1)
        return self._combine_dueling_values(
            state_values,
            advantages,
            action_slices,
        )

    def _as_tensor(self, graph):
        """把 GraphData 转成张量，但暂时保留调用方所在设备。"""

        first_parameter = next(self.parameters())
        dtype = first_parameter.dtype

        if isinstance(graph, GraphTensor):
            return graph
        if isinstance(graph, GraphBatch):
            return graph
        if isinstance(graph, GraphData):
            return graph_to_tensor(
                graph,
                dtype=dtype,
            )
        raise TypeError(f'unsupported graph type: {type(graph)!r}')

    def _move_tensor_to_model(self, graph):
        """把已经检查过的图移动到模型参数所在 device/dtype。"""

        first_parameter = next(self.parameters())
        return graph.to(
            device=first_parameter.device,
            dtype=first_parameter.dtype,
        )

    def _validate_graph_tensor(self, graph):
        """检查模型依赖的张量形状和 schema 元数据。"""

        if graph.node_features.dim() != 2:
            raise ValueError(
                'node_features must have shape '
                '[num_nodes, node_feature_dim]'
            )
        if graph.edge_features.dim() != 2:
            raise ValueError(
                'edge_features must have shape '
                '[num_edges, edge_feature_dim]'
            )
        if graph.edge_index.dim() != 2 or graph.edge_index.shape[0] != 2:
            raise ValueError(
                'edge_index must have shape [2, num_edges]'
            )
        if graph.edge_index.shape[1] != graph.edge_features.shape[0]:
            raise ValueError(
                'edge_index and edge_features must contain the same '
                'number of edges'
            )
        if graph.node_feature_dim != self.node_feature_dim:
            raise ValueError(
                f'expected node_feature_dim={self.node_feature_dim}, '
                f'got {graph.node_feature_dim}'
            )
        if graph.edge_feature_dim != self.edge_feature_dim:
            raise ValueError(
                f'expected edge_feature_dim={self.edge_feature_dim}, '
                f'got {graph.edge_feature_dim}'
            )
        if len(graph.node_feature_names) != graph.node_feature_dim:
            raise ValueError(
                'node_feature_names must match node feature dimension'
            )
        if len(graph.edge_feature_names) != graph.edge_feature_dim:
            raise ValueError(
                'edge_feature_names must match edge feature dimension'
            )
        if graph.action_count == 0:
            raise ValueError(
                'graph must contain at least one action node'
            )

    def _locate_graph_global_nodes(self, graph):
        """在图原设备上按 schema 定位每张图唯一 global 节点。"""

        global_feature_name = 'is_global_node'
        matching_columns = tuple(
            index
            for index, name in enumerate(graph.node_feature_names)
            if name == global_feature_name
        )
        if len(matching_columns) != 1:
            raise ValueError(
                'node feature schema must contain exactly one '
                'is_global_node column'
            )
        global_column = matching_columns[0]

        if isinstance(graph, GraphBatch):
            action_slices = tuple(graph.action_slices)
            node_slices = tuple(graph.node_slices)
            if len(action_slices) != len(node_slices):
                raise ValueError(
                    'GraphBatch action_slices and node_slices must align'
                )
        else:
            action_slices = ((0, graph.action_count),)
            node_slices = ((0, graph.num_nodes),)

        global_indices = []
        expected_action_start = 0
        for graph_index, (
                (action_start, action_end),
                (node_start, node_end),
        ) in enumerate(zip(action_slices, node_slices)):
            if (
                    action_start != expected_action_start
                    or action_end <= action_start):
                raise ValueError(
                    'action_slices must be contiguous and each graph '
                    f'must contain an action; invalid graph {graph_index}'
                )
            expected_action_start = action_end
            if node_start < 0 or node_end <= node_start:
                raise ValueError(
                    f'graph {graph_index} must contain at least one node'
                )

            flags = graph.node_features[
                node_start:node_end,
                global_column,
            ]
            local_indices = torch.nonzero(
                flags > 0.5,
                as_tuple=False,
            ).flatten()
            if local_indices.numel() != 1:
                raise ValueError(
                    f'graph {graph_index} must contain exactly one '
                    'global node selected by is_global_node'
                )
            global_indices.append(
                local_indices[0] + node_start
            )

        if expected_action_start != graph.action_count:
            raise ValueError(
                'action_slices must cover all action nodes exactly once'
            )

        return (
            action_slices,
            torch.stack(global_indices).to(dtype=torch.long),
        )

    @staticmethod
    def _read_graph_global_hidden(
            graph,
            node_hidden,
            action_slices,
            global_node_indices):
        """读取 global 表示，并把每张图的表示广播到自己的动作。"""

        global_hidden = node_hidden[global_node_indices]
        action_global_chunks = tuple(
            global_hidden[graph_index].unsqueeze(0).expand(
                action_end - action_start,
                -1,
            )
            for graph_index, (action_start, action_end) in enumerate(
                action_slices
            )
        )
        action_global_hidden = torch.cat(
            action_global_chunks,
            dim=0,
        )
        if action_global_hidden.shape[0] != graph.action_count:
            raise ValueError(
                'action_slices do not match the flattened action count'
            )

        return (
            global_hidden,
            action_global_hidden,
        )

    @staticmethod
    def _combine_dueling_values(
            state_values,
            advantages,
            action_slices):
        """逐图中心化 advantage，禁止 GraphBatch 样本互相泄漏基线。"""

        q_chunks = []
        for graph_index, (action_start, action_end) in enumerate(
                action_slices):
            graph_advantages = advantages[action_start:action_end]
            if graph_advantages.numel() == 0:
                raise ValueError(
                    f'graph {graph_index} has no action advantages'
                )
            centered_advantages = (
                graph_advantages
                - graph_advantages.mean()
            )
            q_chunks.append(
                state_values[graph_index]
                + centered_advantages
            )
        return torch.cat(q_chunks, dim=0)
