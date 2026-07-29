#!/usr/bin/env python3
"""把可信 GNN-Q checkpoint 导出为 Android 可直接加载的 ONNX assets。

工具会用正式 ``DaxiguaEnv -> StateAnalyzer -> GraphBuilder`` 链路生成多个真实局面，
而不是只拿一张随机矩阵证明 exporter 能运行。每个局面依次比较：

1. 原始 ``GNNQNetwork(GraphTensor)``；
2. 五输入纯张量包装器；
3. ONNX Runtime CPU。

所有样本通过 Q 数值和 argmax 门禁后，才会写出 ``fruit_merge_ai.onnx`` 与
``fruit_merge_ai.metadata.json``。checkpoint 使用 pickle 反序列化，只能传入项目
自己产生并已确认可信的文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from daxigua_rl.env import DaxiguaEnv, DaxiguaEnvConfig  # noqa: E402
from daxigua_rl.graph import GraphBuilder  # noqa: E402
from daxigua_rl.graph.schema import (  # noqa: E402
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from daxigua_rl.graph.tensor import graph_to_tensor  # noqa: E402
from daxigua_rl.models.gnn_q import GNNQNetwork  # noqa: E402
from daxigua_rl.models.mobile_export import (  # noqa: E402
    MOBILE_ACTION_COUNT,
    MOBILE_INPUT_NAMES,
    MOBILE_MODEL_SCHEMA_VERSION,
    MobileGNNQNetwork,
    export_mobile_onnx,
    mobile_inputs_from_graph,
)
from daxigua_rl.training.checkpointing import (  # noqa: E402
    extract_inference_checkpoint,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / 'runs'
    / 'cloud_checkpoints'
    / 'size_transfer_560x1120_15k_best'
    / 'best_inference.pt'
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / 'runs'
    / 'mobile_export'
    / 'size_transfer_560x1120_15k_best'
)


def parse_args(argv=None):
    """解析导出参数。"""

    parser = argparse.ArgumentParser(
        description=(
            'Export the fruit-merge GNN-Q checkpoint to a dynamic '
            'single-graph ONNX model.'
        ),
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help='可信 inference/training checkpoint。',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='写入 ONNX 和 metadata 的目录。',
    )
    parser.add_argument(
        '--validation-samples',
        type=int,
        default=24,
        help='使用多少个真实稳定局面做三路一致性验证。',
    )
    parser.add_argument('--seed', type=int, default=20260730)
    parser.add_argument('--opset', type=int, default=18)
    parser.add_argument('--rtol', type=float, default=2e-4)
    parser.add_argument('--atol', type=float, default=2e-4)
    return parser.parse_args(argv)


def _sha256(path):
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_model(checkpoint_path):
    """加载可信 checkpoint 并严格恢复推理网络。"""

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(
        checkpoint_path,
        map_location='cpu',
        weights_only=False,
    )
    checkpoint_args, online_model_state = extract_inference_checkpoint(
        payload
    )
    model = GNNQNetwork(
        hidden_dim=int(checkpoint_args.get('hidden_dim', 128)),
        message_layers=int(
            checkpoint_args.get('message_layers', 3)
        ),
        activation=str(checkpoint_args.get('activation', 'silu')),
        dropout=float(checkpoint_args.get('dropout', 0.0)),
    )
    model.load_state_dict(online_model_state, strict=True)
    model.eval()
    return checkpoint_path, checkpoint_args, model


def _build_environment(checkpoint_args):
    """按训练 checkpoint 的物理与几何参数构造验证环境。"""

    action_count = int(
        checkpoint_args.get('action_count', MOBILE_ACTION_COUNT)
    )
    if action_count != MOBILE_ACTION_COUNT:
        raise ValueError(
            'checkpoint action_count is incompatible with Android '
            f'contract: {action_count} != {MOBILE_ACTION_COUNT}'
        )
    return DaxiguaEnv(
        DaxiguaEnvConfig(
            board_width=int(checkpoint_args.get('board_width', 560)),
            board_height=int(checkpoint_args.get('board_height', 1120)),
            spawn_y=int(checkpoint_args.get('spawn_y', 252)),
            action_count=action_count,
            physics_fps=int(
                checkpoint_args.get('physics_fps', 30)
            ),
            max_physics_frames=int(
                checkpoint_args.get('max_physics_frames', 240)
            ),
            stable_frames=int(
                checkpoint_args.get('stable_frames', 6)
            ),
            space_iterations=int(
                checkpoint_args.get('space_iterations', 8)
            ),
        )
    )


def _collect_real_graphs(
        model,
        checkpoint_args,
        *,
        sample_count,
        seed):
    """沿当前模型的 greedy 轨迹收集真实稳定边界图。"""

    if sample_count <= 0:
        raise ValueError('validation_samples must be positive')
    environment = _build_environment(checkpoint_args)
    graph_builder = GraphBuilder()
    episode_seed = int(seed)
    state, _info = environment.reset(seed=episode_seed)
    graphs = []

    while len(graphs) < sample_count:
        actions = environment.action_candidates()
        state_analysis = environment.prepare_state_analysis()
        graph = graph_to_tensor(
            graph_builder.build(
                state,
                actions,
                state_analysis=state_analysis,
            ),
            dtype=torch.float32,
        )
        mobile_inputs_from_graph(graph)
        graphs.append(graph)

        with torch.inference_mode():
            action_offset = int(torch.argmax(model(graph)).item())
        state, _reward, terminated, truncated, _step_info = (
            environment.step(action_offset)
        )
        if terminated or truncated:
            episode_seed += 1
            state, _info = environment.reset(seed=episode_seed)
    return tuple(graphs)


def _ort_feed(inputs):
    """把五个 CPU tensor 转为 ONNX Runtime 输入字典。"""

    return {
        name: tensor.detach().cpu().numpy()
        for name, tensor in zip(MOBILE_INPUT_NAMES, inputs)
    }


def _onnx_operator_manifest(model):
    """收集 Android ORT 需要支持的算子与 scatter reduction。"""

    operators = sorted(
        {
            f'{node.domain or "ai.onnx"}::{node.op_type}'
            for node in model.graph.node
        }
    )
    scatter_reductions = sorted(
        {
            attribute.s.decode('ascii')
            for node in model.graph.node
            if node.op_type == 'ScatterElements'
            for attribute in node.attribute
            if attribute.name == 'reduction'
        }
    )
    return operators, scatter_reductions


def _validate_all_paths(
        original_model,
        mobile_model,
        graphs,
        onnx_path,
        *,
        rtol,
        atol):
    """验证原模型、包装器和 ORT，并汇总误差与延迟。"""

    session = ort.InferenceSession(
        str(onnx_path),
        providers=['CPUExecutionProvider'],
    )
    wrapper_max_abs_error = 0.0
    ort_max_abs_error = 0.0
    wrapper_argmax_matches = 0
    ort_argmax_matches = 0
    ort_inference_ms = []
    graph_shapes = []

    for graph in graphs:
        inputs = mobile_inputs_from_graph(graph)
        with torch.inference_mode():
            original_q = original_model(graph).detach().cpu().numpy()
            wrapper_q = mobile_model(*inputs).detach().cpu().numpy()

        start = time.perf_counter()
        (ort_q,) = session.run(['q_values'], _ort_feed(inputs))
        ort_inference_ms.append(
            (time.perf_counter() - start) * 1000.0
        )

        wrapper_error = float(
            np.max(np.abs(wrapper_q - original_q))
        )
        ort_error = float(np.max(np.abs(ort_q - original_q)))
        wrapper_max_abs_error = max(
            wrapper_max_abs_error,
            wrapper_error,
        )
        ort_max_abs_error = max(ort_max_abs_error, ort_error)
        original_action = int(np.argmax(original_q))
        wrapper_argmax_matches += int(
            int(np.argmax(wrapper_q)) == original_action
        )
        ort_argmax_matches += int(
            int(np.argmax(ort_q)) == original_action
        )

        np.testing.assert_allclose(
            wrapper_q,
            original_q,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            ort_q,
            original_q,
            rtol=rtol,
            atol=atol,
        )
        graph_shapes.append(
            {
                'num_nodes': int(graph.num_nodes),
                'num_edges': int(graph.num_edges),
            }
        )

    sample_count = len(graphs)
    if wrapper_argmax_matches != sample_count:
        raise RuntimeError(
            'mobile tensor wrapper changed at least one greedy action'
        )
    if ort_argmax_matches != sample_count:
        raise RuntimeError(
            'ONNX Runtime changed at least one greedy action'
        )
    return {
        'sample_count': sample_count,
        'rtol': float(rtol),
        'atol': float(atol),
        'wrapper_max_abs_error': wrapper_max_abs_error,
        'wrapper_argmax_matches': wrapper_argmax_matches,
        'onnxruntime_max_abs_error': ort_max_abs_error,
        'onnxruntime_argmax_matches': ort_argmax_matches,
        'onnxruntime_latency_ms': {
            'mean': float(statistics.fmean(ort_inference_ms)),
            'median': float(statistics.median(ort_inference_ms)),
            'maximum': float(max(ort_inference_ms)),
        },
        'graph_shapes': graph_shapes,
    }


def main(argv=None):
    """执行导出并打印机器可读摘要。"""

    args = parse_args(argv)
    (
        checkpoint_path,
        checkpoint_args,
        original_model,
    ) = _load_model(args.checkpoint)
    mobile_model = MobileGNNQNetwork(original_model).eval()
    graphs = _collect_real_graphs(
        original_model,
        checkpoint_args,
        sample_count=int(args.validation_samples),
        seed=int(args.seed),
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / 'fruit_merge_ai.onnx'
    metadata_path = (
        output_dir / 'fruit_merge_ai.metadata.json'
    )
    export_mobile_onnx(
        mobile_model,
        mobile_inputs_from_graph(graphs[0]),
        onnx_path,
        opset_version=args.opset,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    operators, scatter_reductions = _onnx_operator_manifest(
        onnx_model
    )
    if 'add' not in scatter_reductions:
        raise RuntimeError(
            'exported ONNX does not preserve add reduction for '
            'duplicate graph targets'
        )
    validation = _validate_all_paths(
        original_model,
        mobile_model,
        graphs,
        onnx_path,
        rtol=float(args.rtol),
        atol=float(args.atol),
    )

    metadata = {
        'schema_version': MOBILE_MODEL_SCHEMA_VERSION,
        'format': 'onnx',
        'opset_version': int(args.opset),
        'model_file': onnx_path.name,
        'model_bytes': onnx_path.stat().st_size,
        'model_sha256': _sha256(onnx_path),
        'checkpoint': {
            'file': checkpoint_path.name,
            'bytes': checkpoint_path.stat().st_size,
            'sha256': _sha256(checkpoint_path),
        },
        'architecture': {
            'name': 'structure_aware_dueling_gnn_q',
            'hidden_dim': int(original_model.hidden_dim),
            'message_layers': int(
                original_model.message_layers
            ),
            'activation': str(
                checkpoint_args.get('activation', 'silu')
            ),
            'action_count': MOBILE_ACTION_COUNT,
        },
        'geometry': {
            'board_width': int(
                checkpoint_args.get('board_width', 560)
            ),
            'board_height': int(
                checkpoint_args.get('board_height', 1120)
            ),
            'spawn_y': int(
                checkpoint_args.get('spawn_y', 252)
            ),
        },
        'inputs': [
            {
                'name': 'node_features',
                'dtype': 'float32',
                'shape': ['N', len(NODE_FEATURE_NAMES)],
            },
            {
                'name': 'edge_index',
                'dtype': 'int64',
                'shape': [2, 'E'],
            },
            {
                'name': 'edge_features',
                'dtype': 'float32',
                'shape': ['E', len(EDGE_FEATURE_NAMES)],
            },
            {
                'name': 'action_node_indices',
                'dtype': 'int64',
                'shape': [MOBILE_ACTION_COUNT],
            },
            {
                'name': 'global_node_index',
                'dtype': 'int64',
                'shape': [1],
            },
        ],
        'outputs': [
            {
                'name': 'q_values',
                'dtype': 'float32',
                'shape': [MOBILE_ACTION_COUNT],
                'selection': 'argmax',
            }
        ],
        'node_feature_names': list(NODE_FEATURE_NAMES),
        'edge_feature_names': list(EDGE_FEATURE_NAMES),
        'required_onnx_operators': operators,
        'scatter_reductions': scatter_reductions,
        'validation': validation,
        'runtime': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'torch': str(torch.__version__),
            'onnx': str(onnx.__version__),
            'onnxruntime': str(ort.__version__),
        },
    }
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )

    summary = {
        'onnx': str(onnx_path),
        'metadata': str(metadata_path),
        'model_bytes': metadata['model_bytes'],
        'model_sha256': metadata['model_sha256'],
        'validation_samples': validation['sample_count'],
        'onnxruntime_max_abs_error': (
            validation['onnxruntime_max_abs_error']
        ),
        'onnxruntime_argmax_matches': (
            validation['onnxruntime_argmax_matches']
        ),
    }
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
