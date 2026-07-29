"""Android 游戏复用的纯 Python 状态分析入口。

本包只处理普通 ``dict`` / JSON 和项目已有的只读状态、StateAnalyzer、GraphBuilder。
它不导入 pygame、Pymunk 或 PyTorch，适合由 Chaquopy 直接调用。真正的 Android
渲染、触控、物理推进和 ONNX 推理由宿主工程负责。
"""

from .bridge import (
    ACTION_COUNT,
    BOARD_HEIGHT,
    BOARD_WIDTH,
    EDGE_FEATURE_DIM,
    MOBILE_GRAPH_SCHEMA_VERSION,
    NODE_FEATURE_DIM,
    SPAWN_Y,
    MobileGraphBridge,
    MobileSceneError,
    build_action_candidates,
    build_mobile_graph,
    build_mobile_graph_json,
    scene_to_game_state,
    scene_to_transition_key,
)


__all__ = [
    'ACTION_COUNT',
    'BOARD_HEIGHT',
    'BOARD_WIDTH',
    'EDGE_FEATURE_DIM',
    'MOBILE_GRAPH_SCHEMA_VERSION',
    'NODE_FEATURE_DIM',
    'SPAWN_Y',
    'MobileGraphBridge',
    'MobileSceneError',
    'build_action_candidates',
    'build_mobile_graph',
    'build_mobile_graph_json',
    'scene_to_game_state',
    'scene_to_transition_key',
]
