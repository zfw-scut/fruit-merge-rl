"""结构感知 GNN-Q 的模型专项测试。"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from daxigua_rl import DaxiguaEnv, DaxiguaEnvConfig, GraphBuilder
from daxigua_rl.graph.tensor import (
    GraphTensor,
    collate_graph_tensors,
    graph_to_tensor,
)
from daxigua_rl.models.gnn_q import (
    STRUCTURE_PREDICTION_NAMES,
    GNNQNetwork,
    MessagePassingLayer,
    StructureAwareQOutput,
)


def _manual_graph(action_signals, state_value):
    """构造一个把 global 标记放在非首列的小图。"""

    node_features = [
        (float(signal), 0.0, 0.0)
        for signal in action_signals
    ]
    node_features.append((0.0, 1.0, float(state_value)))
    action_count = len(action_signals)
    return GraphTensor(
        node_features=torch.tensor(
            node_features,
            dtype=torch.float32,
        ),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_features=torch.empty((0, 3), dtype=torch.float32),
        action_node_indices=torch.arange(
            action_count,
            dtype=torch.long,
        ),
        action_indices=torch.arange(
            action_count,
            dtype=torch.long,
        ),
        node_feature_names=(
            'action_signal',
            'is_global_node',
            'state_value',
        ),
        edge_feature_names=('edge_a', 'edge_b', 'edge_c'),
    )


class StructureAwareGNNTest(unittest.TestCase):
    """覆盖新消息层、dueling 读出和辅助输出契约。"""

    def setUp(self):
        torch.manual_seed(17)

    @staticmethod
    def _builder_graph(action_count=7):
        env = DaxiguaEnv(config=DaxiguaEnvConfig(
            action_count=action_count,
        ))
        state, info = env.reset(seed=23)
        return GraphBuilder().build(
            state,
            tuple(info['action_candidates']),
        )

    def test_forward_with_aux_supports_data_tensor_and_batch(self):
        graph_data = self._builder_graph()
        graph_tensor = graph_to_tensor(graph_data)
        graph_batch = collate_graph_tensors(
            (graph_tensor, graph_tensor)
        )
        model = GNNQNetwork(
            hidden_dim=32,
            message_layers=2,
        )
        model.eval()

        with torch.no_grad():
            data_output = model.forward_with_aux(graph_data)
            tensor_output = model.forward_with_aux(graph_tensor)
            batch_output = model.forward_with_aux(graph_batch)
            legacy_q_values = model(graph_tensor)

        self.assertIsInstance(data_output, StructureAwareQOutput)
        self.assertEqual(
            data_output.prediction_names,
            STRUCTURE_PREDICTION_NAMES,
        )
        self.assertEqual(
            tuple(data_output.q_values.shape),
            (graph_tensor.action_count,),
        )
        self.assertEqual(
            tuple(data_output.structure_predictions.shape),
            (
                graph_tensor.action_count,
                len(STRUCTURE_PREDICTION_NAMES),
            ),
        )
        self.assertEqual(data_output.q_values.dim(), 1)
        self.assertTrue(torch.allclose(
            data_output.q_values,
            tensor_output.q_values,
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            data_output.structure_predictions,
            tensor_output.structure_predictions,
            atol=1e-6,
        ))
        self.assertTrue(torch.allclose(
            legacy_q_values,
            tensor_output.q_values,
            atol=1e-6,
        ))

        for action_start, action_end in graph_batch.action_slices:
            self.assertTrue(torch.allclose(
                batch_output.q_values[action_start:action_end],
                tensor_output.q_values,
                atol=1e-6,
            ))
            self.assertTrue(torch.allclose(
                batch_output.structure_predictions[
                    action_start:action_end
                ],
                tensor_output.structure_predictions,
                atol=1e-6,
            ))

        predictions = tensor_output.structure_predictions
        self.assertTrue(torch.all(predictions >= -1.0))
        self.assertTrue(torch.all(predictions <= 1.0))

    def test_dueling_q_centers_advantage_inside_each_graph(self):
        first_graph = _manual_graph(
            action_signals=(1.0, 3.0),
            state_value=2.0,
        )
        second_graph = _manual_graph(
            action_signals=(4.0, 8.0),
            state_value=5.0,
        )
        graph_batch = collate_graph_tensors(
            (first_graph, second_graph)
        )
        model = GNNQNetwork(
            node_feature_dim=3,
            edge_feature_dim=3,
            hidden_dim=3,
            message_layers=0,
        )

        # 用恒等编码器和单层读出构造可手算的 dueling 结果。global 标记位于
        # node_feature_names 的第二列，验证实现不是硬编码默认列下标。
        model.node_encoder = nn.Identity()
        model.edge_encoder = nn.Identity()
        model.q_head = nn.Linear(3, 1, bias=False)
        model.advantage_head = nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            model.q_head.weight.copy_(torch.tensor([
                [0.0, 0.0, 1.0],
            ]))
            model.advantage_head.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0],
            ]))

        with torch.no_grad():
            q_values = model(graph_batch)

        # 图 1: V=2, A=[1,3], mean(A)=2 -> Q=[1,3]
        # 图 2: V=5, A=[4,8], mean(A)=6 -> Q=[3,7]
        self.assertTrue(torch.allclose(
            q_values,
            torch.tensor([1.0, 3.0, 3.0, 7.0]),
            atol=1e-6,
        ))
        self.assertAlmostEqual(
            float(q_values[0:2].mean()),
            2.0,
        )
        self.assertAlmostEqual(
            float(q_values[2:4].mean()),
            5.0,
        )

    def test_relation_gate_and_attention_depend_on_edge_context(self):
        layer = MessagePassingLayer(
            hidden_dim=4,
            dropout=0.0,
        )
        node_hidden = torch.tensor([
            [0.2, -0.3, 0.5, 0.7],
            [-0.4, 0.8, 0.1, -0.2],
            [0.6, 0.2, -0.7, 0.3],
        ])
        # 两条边共同指向节点 2，attention 权重不会在单入边归一化中抵消。
        edge_index = torch.tensor([
            [0, 1],
            [2, 2],
        ], dtype=torch.long)
        baseline_edges = torch.zeros((2, 4))
        changed_edges = baseline_edges.clone()
        changed_edges[1] = torch.tensor(
            [2.0, -1.0, 0.5, 3.0]
        )

        baseline = layer(
            node_hidden,
            edge_index,
            baseline_edges,
        )
        changed = layer(
            node_hidden,
            edge_index,
            changed_edges,
        )

        self.assertFalse(torch.allclose(
            baseline[2],
            changed[2],
        ))

        weights = torch.tensor([0.3, -0.7, 1.1, 0.2])
        loss = (changed[2] * weights).sum()
        loss.backward()
        relation_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in layer.relation_gate.parameters()
            if parameter.grad is not None
        )
        attention_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in layer.attention_gate.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(relation_grad, 0.0)
        self.assertGreater(attention_grad, 0.0)

    def test_global_node_must_be_uniquely_identified_per_graph(self):
        graph = _manual_graph(
            action_signals=(1.0, 2.0),
            state_value=3.0,
        )
        invalid_features = graph.node_features.clone()
        invalid_features[0, 1] = 1.0
        invalid_graph = GraphTensor(
            node_features=invalid_features,
            edge_index=graph.edge_index,
            edge_features=graph.edge_features,
            action_node_indices=graph.action_node_indices,
            action_indices=graph.action_indices,
            node_feature_names=graph.node_feature_names,
            edge_feature_names=graph.edge_feature_names,
        )
        model = GNNQNetwork(
            node_feature_dim=3,
            edge_feature_dim=3,
            hidden_dim=8,
            message_layers=1,
        )

        with self.assertRaisesRegex(
                ValueError,
                'exactly one global node'):
            model(invalid_graph)


if __name__ == '__main__':
    unittest.main()
