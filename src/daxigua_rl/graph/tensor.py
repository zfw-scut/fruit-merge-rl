"""把框架无关 `GraphData` 转换成 PyTorch 张量。

本模块是训练侧的可选入口，依赖 `torch`。为了让只运行游戏或环境接口的
场景不强制安装 PyTorch，本模块不会被 `daxigua_rl.graph.__init__` 自动导入。
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GraphTensor:
    """GNN 模型直接使用的张量化图数据。

    `GraphData` 更适合调试和文档查看，因为它使用 Python list 保存数据；
    `GraphTensor` 更适合神经网络训练，因为 PyTorch 层只能高效处理 Tensor。
    """

    # 节点特征矩阵，shape: [num_nodes, node_feature_dim]。
    # 每一行对应图中的一个节点，例如真实水果、待投放队列水果、候选动作等。
    node_features: torch.Tensor

    # 有向边索引，shape: [2, num_edges]。
    # edge_index[0] 是每条边的 source 节点下标，edge_index[1] 是 target 节点下标。
    edge_index: torch.Tensor

    # 边特征矩阵，shape: [num_edges, edge_feature_dim]。
    # 第 i 行特征描述 edge_index[:, i] 这条边，例如距离、等级差、是否同级等。
    edge_features: torch.Tensor

    # 候选动作节点在 node_features 中的行号。
    # GNN 会先更新所有节点表示，最后只读取这些 action 节点并输出 Q 值。
    action_node_indices: torch.Tensor

    # 候选动作的原始动作编号。
    # 它和 action_node_indices 一一对应，用于把模型输出重新映射回环境动作。
    action_indices: torch.Tensor

    # 节点特征名快照，用于调试、检查维度、消融实验定位特征列。
    node_feature_names: tuple

    # 边特征名快照，用于调试、检查维度、消融实验定位特征列。
    edge_feature_names: tuple

    @property
    def num_nodes(self):
        """返回节点数量。"""

        return int(self.node_features.shape[0])

    @property
    def num_edges(self):
        """返回边数量。"""

        return int(self.edge_features.shape[0])

    @property
    def node_feature_dim(self):
        """返回节点特征维度。"""

        return int(self.node_features.shape[1])

    @property
    def edge_feature_dim(self):
        """返回边特征维度。"""

        return int(self.edge_features.shape[1])

    @property
    def action_count(self):
        """返回候选动作数量，也就是模型最终需要输出多少个 Q 值。"""

        return int(self.action_node_indices.shape[0])

    def to(self, device=None, dtype=None):
        """返回移动到指定设备和 dtype 的新 GraphTensor。

        这里不原地修改当前对象，而是返回一个新的 `GraphTensor`。
        这样同一份图数据可以同时保留 CPU 版本和 GPU 版本，调试时更安全。
        """

        return GraphTensor(
            # 特征张量参与神经网络计算，需要根据训练配置切换 float32/float16 等 dtype。
            node_features=self.node_features.to(device=device, dtype=dtype),

            # 索引张量必须保持整数类型，只移动设备，不改变 dtype。
            edge_index=self.edge_index.to(device=device),

            # 边特征同样参与网络计算，因此和节点特征使用相同的 dtype 策略。
            edge_features=self.edge_features.to(device=device, dtype=dtype),

            # action_node_indices/action_indices 都是索引，保持 long 类型。
            action_node_indices=self.action_node_indices.to(device=device),
            action_indices=self.action_indices.to(device=device),

            # 特征名只是元数据，不需要移动设备。
            node_feature_names=self.node_feature_names,
            edge_feature_names=self.edge_feature_names,
        )


@dataclass(frozen=True)
class GraphBatch:
    """多张 `GraphTensor` 拼成的一张不连通大图。

    GraphBatch 不改变任何单图内部结构，只把节点、边和 action 节点按顺序拼接。
    每张原始图之间没有新增边，因此 message passing 不会跨样本传播信息。
    """

    # 拼接后的节点特征矩阵，shape: [total_nodes, node_feature_dim]。
    node_features: torch.Tensor

    # 拼接并完成节点下标偏移后的有向边索引，shape: [2, total_edges]。
    edge_index: torch.Tensor

    # 拼接后的边特征矩阵，shape: [total_edges, edge_feature_dim]。
    edge_features: torch.Tensor

    # 拼接后所有 action 节点在 node_features 中的行号。
    action_node_indices: torch.Tensor

    # 拼接后所有 action 的原始动作编号。
    action_indices: torch.Tensor

    # 每张原始图在扁平 Q 值输出中的 action 区间，形如 `(start, end)`。
    action_slices: tuple

    # 每张原始图在 node_features 中的节点区间，主要用于测试和调试。
    node_slices: tuple

    # 每张原始图在 edge_features 中的边区间，主要用于测试和调试。
    edge_slices: tuple

    # 特征名快照，含义和 GraphTensor 一致。
    node_feature_names: tuple
    edge_feature_names: tuple

    @property
    def num_graphs(self):
        """返回 batch 中包含多少张原始图。"""

        return len(self.action_slices)

    @property
    def num_nodes(self):
        """返回拼接后的节点数量。"""

        return int(self.node_features.shape[0])

    @property
    def num_edges(self):
        """返回拼接后的边数量。"""

        return int(self.edge_features.shape[0])

    @property
    def node_feature_dim(self):
        """返回节点特征维度。"""

        return int(self.node_features.shape[1])

    @property
    def edge_feature_dim(self):
        """返回边特征维度。"""

        return int(self.edge_features.shape[1])

    @property
    def action_count(self):
        """返回 batch 中所有候选动作节点数量。"""

        return int(self.action_node_indices.shape[0])

    @property
    def action_counts(self):
        """返回每张原始图各自的候选动作数量。"""

        return tuple(end - start for start, end in self.action_slices)

    def to(self, device=None, dtype=None):
        """返回移动到指定设备和 dtype 的新 GraphBatch。"""

        return GraphBatch(
            node_features=self.node_features.to(device=device, dtype=dtype),
            edge_index=self.edge_index.to(device=device),
            edge_features=self.edge_features.to(device=device, dtype=dtype),
            action_node_indices=self.action_node_indices.to(device=device),
            action_indices=self.action_indices.to(device=device),
            action_slices=self.action_slices,
            node_slices=self.node_slices,
            edge_slices=self.edge_slices,
            node_feature_names=self.node_feature_names,
            edge_feature_names=self.edge_feature_names,
        )


def graph_to_tensor(graph, device=None, dtype=torch.float32):
    """把 `GraphData` 转换成 `GraphTensor`。

    `GraphData.edge_index` 使用 `(source, target)` 元组列表；
    PyTorch 模型中统一转换成 shape 为 `[2, num_edges]` 的 LongTensor。
    """

    # 节点和边特征都来自 GraphBuilder 输出的普通 Python list。
    # 进入模型前先转成连续的浮点矩阵，后续 Linear/MLP 才能直接处理。
    node_features = torch.tensor(graph.node_features, dtype=dtype, device=device)
    edge_features = torch.tensor(graph.edge_features, dtype=dtype, device=device)

    if graph.edge_index:
        # 原始格式: [(src0, dst0), (src1, dst1), ...]，shape 等价于 [num_edges, 2]。
        # GNN 聚合时更常用 [2, num_edges]，所以这里转置一次。
        edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device).t().contiguous()
    else:
        # 极端情况下图里没有边，也保持固定二维 shape，避免模型里出现特殊维度分支。
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    # `action_node_indices` 用于从全部节点表示中取出候选动作节点。
    # `action_indices` 用于保留环境动作编号，方便训练或推理阶段回传给环境。
    action_node_indices = torch.tensor(graph.action_node_indices, dtype=torch.long, device=device)
    action_indices = torch.tensor(graph.action_indices, dtype=torch.long, device=device)

    return GraphTensor(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        action_node_indices=action_node_indices,
        action_indices=action_indices,
        node_feature_names=graph.node_feature_names,
        edge_feature_names=graph.edge_feature_names,
    )


def collate_graph_tensors(graphs, device=None, dtype=None):
    """把多张 `GraphTensor` 拼成一张不连通 `GraphBatch`。

    这个函数是批量 GNN 训练的核心胶水：
    - 节点特征和边特征直接按图顺序拼接；
    - 每张图的 `edge_index` 和 `action_node_indices` 按累计节点数做偏移；
    - `action_slices` 记录每张图在扁平 Q 输出中的动作区间。
    """

    graphs = tuple(graphs)
    if not graphs:
        raise ValueError('graphs must contain at least one GraphTensor')

    first = graphs[0]
    if not isinstance(first, GraphTensor):
        raise TypeError(f'graphs must contain GraphTensor, got {type(first)!r}')

    node_feature_names = first.node_feature_names
    edge_feature_names = first.edge_feature_names
    node_feature_dim = first.node_feature_dim
    edge_feature_dim = first.edge_feature_dim

    node_chunks = []
    edge_chunks = []
    edge_index_chunks = []
    action_node_chunks = []
    action_index_chunks = []
    action_slices = []
    node_slices = []
    edge_slices = []

    node_offset = 0
    edge_offset = 0
    action_offset = 0

    for graph in graphs:
        if not isinstance(graph, GraphTensor):
            raise TypeError(f'graphs must contain GraphTensor, got {type(graph)!r}')
        if graph.node_feature_names != node_feature_names:
            raise ValueError('all graphs must use the same node feature schema')
        if graph.edge_feature_names != edge_feature_names:
            raise ValueError('all graphs must use the same edge feature schema')
        if graph.node_feature_dim != node_feature_dim:
            raise ValueError('all graphs must use the same node feature dimension')
        if graph.edge_feature_dim != edge_feature_dim:
            raise ValueError('all graphs must use the same edge feature dimension')
        if graph.action_count <= 0:
            raise ValueError('each graph must contain at least one action node')

        # 训练主链路中大多数 GraphTensor 已经在 CPU/float16 上，collate 时通常
        # 不需要额外转换；避免每张图都创建一个临时 GraphTensor 包装对象。
        if device is not None or dtype is not None:
            graph = graph.to(device=device, dtype=dtype)

        node_start = node_offset
        node_end = node_start + graph.num_nodes
        edge_start = edge_offset
        edge_end = edge_start + graph.num_edges
        action_start = action_offset
        action_end = action_start + graph.action_count

        node_chunks.append(graph.node_features)
        edge_chunks.append(graph.edge_features)
        if graph.num_edges:
            edge_index_chunks.append(graph.edge_index + node_offset)
        action_node_chunks.append(graph.action_node_indices + node_offset)
        action_index_chunks.append(graph.action_indices)

        node_slices.append((node_start, node_end))
        edge_slices.append((edge_start, edge_end))
        action_slices.append((action_start, action_end))

        node_offset = node_end
        edge_offset = edge_end
        action_offset = action_end

    node_features = torch.cat(node_chunks, dim=0)
    edge_features = torch.cat(edge_chunks, dim=0)
    action_node_indices = torch.cat(action_node_chunks, dim=0)
    action_indices = torch.cat(action_index_chunks, dim=0)

    if edge_index_chunks:
        edge_index = torch.cat(edge_index_chunks, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=node_features.device)

    return GraphBatch(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        action_node_indices=action_node_indices,
        action_indices=action_indices,
        action_slices=tuple(action_slices),
        node_slices=tuple(node_slices),
        edge_slices=tuple(edge_slices),
        node_feature_names=node_feature_names,
        edge_feature_names=edge_feature_names,
    )
