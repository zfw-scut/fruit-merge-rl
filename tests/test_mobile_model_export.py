"""移动端单图张量包装与 ONNX 动态图导出的测试。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from daxigua_rl.graph.schema import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from daxigua_rl.graph.tensor import GraphTensor
from daxigua_rl.models.gnn_q import GNNQNetwork
from daxigua_rl.models.mobile_export import (
    MOBILE_ACTION_COUNT,
    MobileGNNQNetwork,
    export_mobile_onnx,
    mobile_inputs_from_graph,
)


def _synthetic_graph(node_count, edge_count, *, seed):
    """构造包含重复 target 的合法单图，真实覆盖加法聚合。"""

    if node_count < MOBILE_ACTION_COUNT + 1:
        raise ValueError('node_count is too small')
    generator = torch.Generator().manual_seed(seed)
    node_features = torch.randn(
        node_count,
        len(NODE_FEATURE_NAMES),
        generator=generator,
    )
    global_index = node_count - MOBILE_ACTION_COUNT - 1
    global_column = NODE_FEATURE_NAMES.index('is_global_node')
    node_features[:, global_column] = 0.0
    node_features[global_index, global_column] = 1.0
    action_column = NODE_FEATURE_NAMES.index('is_action_node')
    node_features[:, action_column] = 0.0
    node_features[
        node_count - MOBILE_ACTION_COUNT:,
        action_column,
    ] = 1.0

    source = torch.randint(
        node_count,
        (edge_count,),
        generator=generator,
    )
    # 只使用较少 target，使 index_add 必然处理重复下标。
    target = torch.randint(
        max(2, node_count // 3),
        node_count,
        (edge_count,),
        generator=generator,
    )
    edge_index = torch.stack((source, target), dim=0)
    edge_features = torch.randn(
        edge_count,
        len(EDGE_FEATURE_NAMES),
        generator=generator,
    )
    action_node_indices = torch.arange(
        node_count - MOBILE_ACTION_COUNT,
        node_count,
        dtype=torch.long,
    )
    return GraphTensor(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        action_node_indices=action_node_indices,
        action_indices=torch.arange(
            MOBILE_ACTION_COUNT,
            dtype=torch.long,
        ),
        node_feature_names=NODE_FEATURE_NAMES,
        edge_feature_names=EDGE_FEATURE_NAMES,
    )


class MobileModelExportTests(unittest.TestCase):
    """验证 Python 原模型、纯张量包装器和 ONNX Runtime 一致。"""

    def setUp(self):
        torch.manual_seed(123)
        self.network = GNNQNetwork(
            hidden_dim=32,
            message_layers=2,
            dropout=0.0,
        ).eval()
        self.mobile = MobileGNNQNetwork(self.network).eval()

    def test_tensor_wrapper_matches_original_model(self):
        graph = _synthetic_graph(35, 83, seed=1)
        mobile_inputs = mobile_inputs_from_graph(graph)

        with torch.inference_mode():
            expected = self.network(graph)
            actual = self.mobile(*mobile_inputs)

        self.assertEqual(tuple(actual.shape), (MOBILE_ACTION_COUNT,))
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(
            int(torch.argmax(actual)),
            int(torch.argmax(expected)),
        )

    def test_input_adapter_rejects_non_mobile_action_count(self):
        graph = _synthetic_graph(35, 83, seed=2)
        graph = GraphTensor(
            node_features=graph.node_features,
            edge_index=graph.edge_index,
            edge_features=graph.edge_features,
            action_node_indices=graph.action_node_indices[:-1],
            action_indices=graph.action_indices[:-1],
            node_feature_names=graph.node_feature_names,
            edge_feature_names=graph.edge_feature_names,
        )
        with self.assertRaisesRegex(ValueError, '21 actions'):
            mobile_inputs_from_graph(graph)

    @unittest.skipUnless(
        importlib.util.find_spec('onnx') is not None
        and importlib.util.find_spec('onnxruntime') is not None,
        'onnx and onnxruntime are optional export dependencies',
    )
    def test_onnx_runtime_matches_two_dynamic_graph_shapes(self):
        import onnx
        import onnxruntime as ort

        first_graph = _synthetic_graph(35, 83, seed=3)
        second_graph = _synthetic_graph(47, 129, seed=4)
        first_inputs = mobile_inputs_from_graph(first_graph)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / 'model.onnx'
            export_mobile_onnx(
                self.mobile,
                first_inputs,
                output_path,
            )
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            scatter_nodes = [
                node
                for node in onnx_model.graph.node
                if node.op_type == 'ScatterElements'
            ]
            self.assertTrue(scatter_nodes)
            reductions = {
                attribute.s.decode('ascii')
                for node in scatter_nodes
                for attribute in node.attribute
                if attribute.name == 'reduction'
            }
            self.assertIn('add', reductions)

            session = ort.InferenceSession(
                str(output_path),
                providers=['CPUExecutionProvider'],
            )
            for graph in (first_graph, second_graph):
                inputs = mobile_inputs_from_graph(graph)
                feed = {
                    name: tensor.detach().cpu().numpy()
                    for name, tensor in zip(
                        (
                            'node_features',
                            'edge_index',
                            'edge_features',
                            'action_node_indices',
                            'global_node_index',
                        ),
                        inputs,
                    )
                }
                (ort_q_values,) = session.run(['q_values'], feed)
                with torch.inference_mode():
                    expected = self.mobile(*inputs).cpu().numpy()
                np.testing.assert_allclose(
                    ort_q_values,
                    expected,
                    rtol=2e-5,
                    atol=2e-5,
                )
                self.assertEqual(
                    int(np.argmax(ort_q_values)),
                    int(np.argmax(expected)),
                )


if __name__ == '__main__':
    unittest.main()
