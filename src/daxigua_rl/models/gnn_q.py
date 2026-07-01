"""GNN-Q 模型。

第一版模型目标很朴素：验证 `GraphData -> q_values` 的前向链路。
它使用统一图 message passing，不区分真正的异构图算子；节点类型和边类型
由 GraphBuilder 写入 one-hot 特征，模型通过这些特征自行学习不同语义。
"""

import torch
from torch import nn

from daxigua_rl.graph.schema import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, GraphData
from daxigua_rl.graph.tensor import GraphBatch, GraphTensor, graph_to_tensor


def _activation(name):
    """根据名称创建激活层。

    目前只暴露少量选择，避免模型配置还没稳定时引入太多分支。
    """

    # ReLU 简单、稳定，是很多 DQN baseline 的常见选择。
    if name == 'relu':
        return nn.ReLU()

    # SiLU 在连续特征任务里通常更平滑，这里作为默认激活函数。
    if name == 'silu':
        return nn.SiLU()

    # 明确报错可以尽早发现拼写错误，例如把 `silu` 写成 `SiLU`。
    raise ValueError(f'unsupported activation: {name}')


def _mlp(input_dim, hidden_dim, output_dim, activation='silu', dropout=0.0):
    """创建两层 MLP。

    本文件里的节点编码、边编码、message 生成、节点更新都复用这个结构。
    """

    return nn.Sequential(
        # 先把输入特征映射到统一隐藏维度，方便后续 message passing 处理。
        nn.Linear(input_dim, hidden_dim),

        # 非线性激活用于提升表达能力，否则多层线性层仍然等价于一层线性层。
        _activation(activation),

        # dropout 默认关闭；后续训练时如果过拟合明显，可以通过参数打开。
        nn.Dropout(dropout),

        # 输出维度由调用者决定，例如编码器输出 hidden_dim，Q head 最终输出 1。
        nn.Linear(hidden_dim, output_dim),
    )


class MessagePassingLayer(nn.Module):
    """一层 mean aggregation 的消息传递。

    对每条有向边 `src -> dst`：
    - 使用源节点、目标节点和边特征共同生成 message。
    - 按目标节点对 message 做 mean 聚合。
    - 用残差和 LayerNorm 更新节点表示。
    """

    def __init__(self, hidden_dim, activation='silu', dropout=0.0):
        super().__init__()

        # 每条边的 message 同时看三部分信息：
        # 1. source 节点当前隐藏表示；
        # 2. target 节点当前隐藏表示；
        # 3. 这条边自己的隐藏表示。
        # 因此输入维度是 hidden_dim * 3。
        self.message_mlp = _mlp(hidden_dim * 3, hidden_dim, hidden_dim, activation, dropout)

        # target 节点收到邻居消息后，用「自身旧表示 + 聚合消息」计算更新量。
        # 输入维度是 hidden_dim * 2。
        self.update_mlp = _mlp(hidden_dim * 2, hidden_dim, hidden_dim, activation, dropout)

        # LayerNorm 用来稳定多层 message passing，降低隐藏表示数值漂移。
        self.norm = nn.LayerNorm(hidden_dim)

        # 残差分支上的 dropout，默认不启用。
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_hidden, edge_index, edge_hidden):
        """执行一次消息传递并返回更新后的节点表示。

        参数约定：
        - node_hidden: [num_nodes, hidden_dim]
        - edge_index: [2, num_edges]
        - edge_hidden: [num_edges, hidden_dim]
        返回：
        - updated_node_hidden: [num_nodes, hidden_dim]
        """

        if edge_index.numel() == 0:
            # 没有边时，所有节点都收不到邻居消息。
            # 这里使用全零聚合结果，保证后面的 update_mlp 仍然能处理节点自身信息。
            aggregated = torch.zeros_like(node_hidden)
        else:
            # source_index/target_index 都是一维 LongTensor，长度等于边数量。
            # 对第 i 条边来说：source_index[i] -> target_index[i]。
            source_index = edge_index[0]
            target_index = edge_index[1]

            # 根据边索引取出每条边两端节点的当前隐藏表示。
            # source_hidden、target_hidden 的 shape 都是 [num_edges, hidden_dim]。
            source_hidden = node_hidden[source_index]
            target_hidden = node_hidden[target_index]

            # 把边两端节点表示和边表示拼接起来，作为 message_mlp 的输入。
            # message_input shape: [num_edges, hidden_dim * 3]。
            message_input = torch.cat((source_hidden, target_hidden, edge_hidden), dim=-1)

            # 为每条边生成一条 message，shape: [num_edges, hidden_dim]。
            messages = self.message_mlp(message_input)

            # aggregated 用来按 target 节点累加收到的 message。
            # 初始全零，随后 index_add_ 会把 messages 加到对应目标节点行上。
            aggregated = torch.zeros_like(node_hidden)
            aggregated.index_add_(0, target_index, messages)

            # 记录每个 target 节点收到了多少条入边 message。
            # 后面除以 counts，就从 sum aggregation 变成 mean aggregation。
            counts = torch.zeros(
                node_hidden.shape[0],
                1,
                dtype=node_hidden.dtype,
                device=node_hidden.device,
            )
            counts.index_add_(0, target_index, torch.ones_like(messages[:, :1]))

            # clamp_min(1.0) 防止没有入边的节点除以 0。
            # 没有入边的节点 aggregated 原本就是 0，除以 1 后仍然是 0。
            aggregated = aggregated / counts.clamp_min(1.0)

        # 节点更新时同时看自己的旧表示和从邻居聚合来的消息。
        update_input = torch.cat((node_hidden, aggregated), dim=-1)
        update = self.update_mlp(update_input)

        # 残差连接保留旧表示，update 只负责补充新信息；
        # LayerNorm 让多层堆叠时数值范围更稳定。
        return self.norm(node_hidden + self.dropout(update))


class GNNQNetwork(nn.Module):
    """输入一张状态图，输出每个候选动作的 Q 值。

    当前模型不是完整训练算法，只负责 Q 网络的前向计算。
    后续 DQN/Double DQN、经验回放、目标网络等训练逻辑会在更外层实现。
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

        # 如果调用者没有显式传入维度，就使用当前 schema 中定义的完整特征维度。
        # 这样 GraphBuilder 的默认输出可以直接送入模型。
        self.node_feature_dim = node_feature_dim or len(NODE_FEATURE_NAMES)
        self.edge_feature_dim = edge_feature_dim or len(EDGE_FEATURE_NAMES)
        self.hidden_dim = hidden_dim
        self.message_layers = message_layers

        # 节点编码器：把原始节点特征映射到 hidden_dim。
        # 输入 shape: [num_nodes, node_feature_dim]
        # 输出 shape: [num_nodes, hidden_dim]
        self.node_encoder = _mlp(self.node_feature_dim, hidden_dim, hidden_dim, activation, dropout)

        # 边编码器：把原始边特征映射到 hidden_dim。
        # 输入 shape: [num_edges, edge_feature_dim]
        # 输出 shape: [num_edges, hidden_dim]
        self.edge_encoder = _mlp(self.edge_feature_dim, hidden_dim, hidden_dim, activation, dropout)

        # 连续堆叠多层 message passing。
        # 层数越多，action 节点理论上能整合越远的图结构信息；
        # 但层数太深也可能带来过平滑或训练不稳定，当前默认先用 3 层。
        self.layers = nn.ModuleList(
            MessagePassingLayer(hidden_dim, activation=activation, dropout=dropout)
            for _ in range(message_layers)
        )

        # Q head 只作用在 action 节点的最终隐藏表示上。
        # 每个 action 节点输出一个标量，表示该候选动作的 Q(s, a)。
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph):
        """返回 shape 为 `[action_count]` 的 Q 值张量。

        输入可以是：
        - `GraphData`：调试友好，内部会临时转成 Tensor；
        - `GraphTensor`：训练友好，推荐训练循环中提前转换好后复用。
        """

        # 统一转换成 GraphTensor，并确保张量和模型处于同一个 device/dtype。
        graph_tensor = self._ensure_tensor(graph)

        # 在真正进入神经网络前做形状检查，避免 PyTorch 报出难读的矩阵乘法错误。
        self._validate_graph_tensor(graph_tensor)

        # 原始特征先编码成统一隐藏维度。
        node_hidden = self.node_encoder(graph_tensor.node_features)
        edge_hidden = self.edge_encoder(graph_tensor.edge_features)

        # 逐层传播图结构信息。
        # 每一层都会让节点从入边邻居那里接收一次信息。
        for layer in self.layers:
            node_hidden = layer(node_hidden, graph_tensor.edge_index, edge_hidden)

        # 图里包含真实水果、队列水果、边界、全局节点等多种节点；
        # 但 DQN 最终只需要比较候选动作，所以这里仅读取 action 节点。
        action_hidden = node_hidden[graph_tensor.action_node_indices]

        # 单图时输出 [action_count]；GraphBatch 时输出 [total_action_count]。
        # batch 中每张图的动作区间由 GraphBatch.action_slices 记录。
        return self.q_head(action_hidden).squeeze(-1)

    def _ensure_tensor(self, graph):
        """接受 GraphData、GraphTensor 或 GraphBatch，并移动到模型所在设备。"""

        # next(self.parameters()) 给出当前模型参数所在 device/dtype。
        # ReplayBuffer 可以用 float16 省内存，但模型通常仍是 float32；
        # 因此进入网络前需要把图特征转回模型参数 dtype，避免 Linear dtype mismatch。
        first_parameter = next(self.parameters())
        device = first_parameter.device
        dtype = first_parameter.dtype

        # GraphTensor 已经是张量格式，只需要移动设备。
        if isinstance(graph, GraphTensor):
            return graph.to(device=device, dtype=dtype)

        # GraphBatch 已经是张量格式，只需要整体移动设备。
        if isinstance(graph, GraphBatch):
            return graph.to(device=device, dtype=dtype)

        # GraphData 是 GraphBuilder 的原始输出，先转成张量。
        if isinstance(graph, GraphData):
            return graph_to_tensor(graph, device=device)

        # 其它类型说明调用链有问题，直接报错比静默失败更容易定位。
        raise TypeError(f'unsupported graph type: {type(graph)!r}')

    def _validate_graph_tensor(self, graph):
        """检查输入图张量的基础形状。

        这里只检查模型必须依赖的基本条件。
        更细的语义检查，例如 action 节点类型是否真的正确，由 GraphBuilder 负责。
        """

        # 节点特征必须是二维矩阵：[节点数量, 节点特征维度]。
        if graph.node_features.dim() != 2:
            raise ValueError('node_features must have shape [num_nodes, node_feature_dim]')

        # 边特征必须是二维矩阵：[边数量, 边特征维度]。
        if graph.edge_features.dim() != 2:
            raise ValueError('edge_features must have shape [num_edges, edge_feature_dim]')

        # edge_index 必须保存 source/target 两行索引。
        if graph.edge_index.dim() != 2 or graph.edge_index.shape[0] != 2:
            raise ValueError('edge_index must have shape [2, num_edges]')

        # 如果 GraphBuilder 的 schema 改了，但模型仍按旧维度初始化，这里会立刻报错。
        if graph.node_feature_dim != self.node_feature_dim:
            raise ValueError(
                f'expected node_feature_dim={self.node_feature_dim}, got {graph.node_feature_dim}'
            )

        # 边特征维度同理，避免 Linear 层输入维度不匹配。
        if graph.edge_feature_dim != self.edge_feature_dim:
            raise ValueError(
                f'expected edge_feature_dim={self.edge_feature_dim}, got {graph.edge_feature_dim}'
            )

        # 没有 action 节点时，模型无法输出动作 Q 值。
        if graph.action_count == 0:
            raise ValueError('graph must contain at least one action node')
