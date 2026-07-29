"""把训练期 GNN-Q 网络收窄成适合移动端部署的纯张量接口。

训练模型接收 :class:`~daxigua_rl.graph.tensor.GraphTensor`。这个 Python
dataclass 同时携带特征名和动作切片，便于训练期校验，但 ONNX/Android 运行时只应
看到普通张量。这里因此只支持“一张图、固定 21 个动作”的部署边界：

```
node_features[N, 62]
edge_index[2, E]
edge_features[E, 47]
action_node_indices[21]
global_node_index[1]
        -> q_values[21]
```

节点数 ``N`` 和边数 ``E`` 保持动态。动作数固定是当前游戏与状态归因 schema 的
共同契约；如果以后修改动作数，应重新导出模型和元数据，而不是让旧 APK 静默解释
一套不同的动作空间。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from daxigua_rl.attribution import ANALYSIS_ACTION_COUNT
from daxigua_rl.graph.schema import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from daxigua_rl.graph.tensor import GraphTensor
from daxigua_rl.models.gnn_q import GNNQNetwork


MOBILE_MODEL_SCHEMA_VERSION = 1
MOBILE_ACTION_COUNT = ANALYSIS_ACTION_COUNT
MOBILE_NODE_FEATURE_DIM = len(NODE_FEATURE_NAMES)
MOBILE_EDGE_FEATURE_DIM = len(EDGE_FEATURE_NAMES)

MOBILE_INPUT_NAMES = (
    'node_features',
    'edge_index',
    'edge_features',
    'action_node_indices',
    'global_node_index',
)
MOBILE_OUTPUT_NAMES = ('q_values',)


class MobileGNNQNetwork(nn.Module):
    """用五个普通张量执行单图 GNN-Q 推理。

    这里直接复用训练模型已经学到的 encoder、message layer 和 dueling heads，
    没有复制或转换权重。消息聚合刻意写成无 Python 分支的 ``scatter_add``：
    ONNX opset 18 会把它表示为带 ``reduction=add`` 的 ``ScatterElements``，
    ONNX Runtime Android 可以执行同一图。

    移动端的正式图始终包含 global/action 等关系边，因此 ``E`` 可以动态变化但
    必须大于零。输入合法性在 Python/Android 预处理边界检查，不能把异常处理分支
    烘进计算图。
    """

    def __init__(self, network: GNNQNetwork):
        super().__init__()
        if not isinstance(network, GNNQNetwork):
            raise TypeError('network must be GNNQNetwork')
        if network.node_feature_dim != MOBILE_NODE_FEATURE_DIM:
            raise ValueError(
                'mobile node schema does not match network: '
                f'{MOBILE_NODE_FEATURE_DIM} != '
                f'{network.node_feature_dim}'
            )
        if network.edge_feature_dim != MOBILE_EDGE_FEATURE_DIM:
            raise ValueError(
                'mobile edge schema does not match network: '
                f'{MOBILE_EDGE_FEATURE_DIM} != '
                f'{network.edge_feature_dim}'
            )
        self.network = network

    def forward(
            self,
            node_features,
            edge_index,
            edge_features,
            action_node_indices,
            global_node_index):
        """返回当前单图 21 个候选落点的 Q 值。"""

        node_hidden = self.network.node_encoder(node_features)
        edge_hidden = self.network.edge_encoder(edge_features)

        # 不调用训练层的 shape 校验和空边 Python 分支，避免 tracing 把一次样本的
        # N/E 固化。下面的计算与 MessagePassingLayer.forward 数值完全等价。
        source_index = edge_index[0]
        target_index = edge_index[1]
        for layer in self.network.layers:
            source_hidden = node_hidden[source_index]
            target_hidden = node_hidden[target_index]
            message_input = torch.cat(
                (source_hidden, target_hidden, edge_hidden),
                dim=-1,
            )
            messages = layer.message_mlp(message_input)
            relation_weights = layer.relation_gate(edge_hidden)
            attention_weights = layer.attention_gate(message_input)
            weighted_messages = (
                messages
                * relation_weights
                * attention_weights
            )

            aggregated = torch.zeros_like(node_hidden)
            message_target_index = target_index.unsqueeze(-1).expand_as(
                weighted_messages
            )
            # ``index_add`` 的旧 ONNX symbolic 在重复 target 时会退化成覆盖写。
            # 显式 ``scatter_add`` 才能生成 reduction=add，图里的多个入边因此
            # 会与 PyTorch 一样全部参与聚合。
            aggregated = torch.scatter_add(
                aggregated,
                0,
                message_target_index,
                weighted_messages,
            )
            attention_sum = torch.zeros_like(node_hidden[:, :1])
            attention_sum = torch.scatter_add(
                attention_sum,
                0,
                target_index.unsqueeze(-1),
                attention_weights,
            )
            epsilon = torch.finfo(node_hidden.dtype).eps
            aggregated = (
                aggregated
                / torch.clamp_min(attention_sum, epsilon)
            )

            update = layer.update_mlp(
                torch.cat((node_hidden, aggregated), dim=-1)
            )
            node_hidden = layer.norm(
                node_hidden + layer.dropout(update)
            )

        action_hidden = node_hidden[action_node_indices]
        global_hidden = node_hidden[global_node_index]
        state_value = (
            self.network.q_head(global_hidden)
            .reshape(-1)[0]
        )
        advantages = (
            self.network.advantage_head(action_hidden)
            .squeeze(-1)
        )
        return state_value + advantages - advantages.mean()


def mobile_inputs_from_graph(graph: GraphTensor):
    """从训练期 ``GraphTensor`` 提取并校验五个部署输入。

    返回值顺序与 :data:`MOBILE_INPUT_NAMES` 完全一致，可直接传给包装器或 ONNX
    exporter。索引统一为 int64，这是 ONNX Runtime 的 ``Gather`` /
    ``ScatterElements`` 所需类型。
    """

    if not isinstance(graph, GraphTensor):
        raise TypeError('graph must be GraphTensor')
    if graph.node_feature_names != NODE_FEATURE_NAMES:
        raise ValueError('graph uses an incompatible node feature schema')
    if graph.edge_feature_names != EDGE_FEATURE_NAMES:
        raise ValueError('graph uses an incompatible edge feature schema')
    if graph.node_features.ndim != 2:
        raise ValueError('node_features must have shape [N, 62]')
    if graph.node_feature_dim != MOBILE_NODE_FEATURE_DIM:
        raise ValueError(
            f'node feature dimension must be {MOBILE_NODE_FEATURE_DIM}'
        )
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError('edge_index must have shape [2, E]')
    if graph.edge_features.ndim != 2:
        raise ValueError('edge_features must have shape [E, 47]')
    if graph.edge_feature_dim != MOBILE_EDGE_FEATURE_DIM:
        raise ValueError(
            f'edge feature dimension must be {MOBILE_EDGE_FEATURE_DIM}'
        )
    if graph.edge_index.shape[1] != graph.edge_features.shape[0]:
        raise ValueError(
            'edge_index and edge_features must contain the same edges'
        )
    if graph.num_edges <= 0:
        raise ValueError('mobile graph must contain at least one edge')
    if graph.action_count != MOBILE_ACTION_COUNT:
        raise ValueError(
            f'mobile graph must contain {MOBILE_ACTION_COUNT} actions'
        )
    expected_action_indices = torch.arange(
        MOBILE_ACTION_COUNT,
        dtype=graph.action_indices.dtype,
        device=graph.action_indices.device,
    )
    if not torch.equal(graph.action_indices, expected_action_indices):
        raise ValueError(
            'mobile actions must be ordered from 0 through 20'
        )

    global_column = NODE_FEATURE_NAMES.index('is_global_node')
    global_node_indices = torch.nonzero(
        graph.node_features[:, global_column] > 0.5,
        as_tuple=False,
    ).flatten()
    if global_node_indices.numel() != 1:
        raise ValueError(
            'mobile graph must contain exactly one global node'
        )

    node_count = graph.node_features.shape[0]
    if (
            torch.any(graph.edge_index < 0)
            or torch.any(graph.edge_index >= node_count)):
        raise ValueError('edge_index contains an out-of-range node')
    if (
            torch.any(graph.action_node_indices < 0)
            or torch.any(graph.action_node_indices >= node_count)):
        raise ValueError(
            'action_node_indices contains an out-of-range node'
        )
    if torch.unique(graph.action_node_indices).numel() != MOBILE_ACTION_COUNT:
        raise ValueError('action_node_indices must be unique')
    action_column = NODE_FEATURE_NAMES.index('is_action_node')
    if not torch.all(
            graph.node_features[
                graph.action_node_indices,
                action_column,
            ] > 0.5):
        raise ValueError(
            'every action_node_indices entry must select an action node'
        )

    return (
        graph.node_features.to(dtype=torch.float32).contiguous(),
        graph.edge_index.to(dtype=torch.int64).contiguous(),
        graph.edge_features.to(dtype=torch.float32).contiguous(),
        graph.action_node_indices.to(dtype=torch.int64).contiguous(),
        global_node_indices.to(dtype=torch.int64).contiguous(),
    )


def export_mobile_onnx(
        model: MobileGNNQNetwork,
        sample_inputs,
        destination,
        *,
        opset_version=18):
    """把包装器导出为单文件 ONNX，并返回目标路径。

    旧 exporter 在当前模型上能稳定生成 ``ScatterElements(reduction=add)``，
    并允许显式声明动态 N/E。权重只有约 14 MB，不使用 external data，Android
    assets 因而只需携带一个 ``.onnx`` 文件。
    """

    if not isinstance(model, MobileGNNQNetwork):
        raise TypeError('model must be MobileGNNQNetwork')
    if len(tuple(sample_inputs)) != len(MOBILE_INPUT_NAMES):
        raise ValueError('sample_inputs must contain five tensors')
    opset_version = int(opset_version)
    if opset_version < 18:
        raise ValueError('mobile export requires ONNX opset 18 or newer')

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            tuple(sample_inputs),
            str(destination),
            input_names=list(MOBILE_INPUT_NAMES),
            output_names=list(MOBILE_OUTPUT_NAMES),
            dynamic_axes={
                'node_features': {0: 'num_nodes'},
                'edge_index': {1: 'num_edges'},
                'edge_features': {0: 'num_edges'},
            },
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
        )
    return destination


__all__ = [
    'MOBILE_ACTION_COUNT',
    'MOBILE_EDGE_FEATURE_DIM',
    'MOBILE_INPUT_NAMES',
    'MOBILE_MODEL_SCHEMA_VERSION',
    'MOBILE_NODE_FEATURE_DIM',
    'MOBILE_OUTPUT_NAMES',
    'MobileGNNQNetwork',
    'export_mobile_onnx',
    'mobile_inputs_from_graph',
]
