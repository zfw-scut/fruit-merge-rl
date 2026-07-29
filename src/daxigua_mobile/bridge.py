"""把 Android 局面 JSON 转换为结构感知 GNN 的纯 Python 输入。

Android 侧只需要提供当前稳定动作边界的几何、q0～q3 和水果刚体快照。本模块会：

1. 用游戏规则构造 ``GameState`` 与固定 21 个 ``ActionCandidate``；
2. 用 ``TransitionKey`` 锁定当前动作前状态的时间语义；
3. 运行训练时同一份 ``StateAnalyzer`` 和 ``GraphBuilder``；
4. 返回不含 dataclass、Tensor 或第三方对象的扁平 JSON 友好结构。

坐标沿用当前游戏约定：原点位于画布左上方，x 向右、y 向下。调用方最好显式提供
每颗水果的 ``physics_radius``，因为合成水果和直接投放水果的碰撞半径可能相差
一个像素；省略时仅为方便最小接入而使用直接投放半径。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from operator import index

from daxigua.core.rules import (
    FRUIT_QUEUE_LENGTH,
    MAX_FRUIT_LEVEL,
    MIN_FRUIT_LEVEL,
    SPAWN_FRUIT_MAX_LEVEL,
    SPAWN_FRUIT_MIN_LEVEL,
    dropped_fruit_physics_radius,
    fruit_radius,
)
from daxigua.core.state import (
    ActionCandidate,
    BoardGeometry,
    FruitState,
    GameState,
)
from daxigua_rl.attribution.state_analyzer import (
    StateAnalyzer,
    StateAnalyzerConfig,
    drop_x_positions_for_level,
)
from daxigua_rl.graph.builder import GraphBuilder, GraphBuilderConfig
from daxigua_rl.graph.schema import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
)
from daxigua_rl.training.identity import TransitionKey


MOBILE_GRAPH_SCHEMA_VERSION = 1

# 当前最终模型只维护扩大后的场景规格。这里显式拒绝旧地图，避免 Android 侧把
# 400x800 状态静默归一化后交给 560x1120 模型。
BOARD_WIDTH = 560
BOARD_HEIGHT = 1120
SPAWN_Y = 252
WALL_WIDTH = 20
FLOOR_Y = BOARD_HEIGHT - WALL_WIDTH

ACTION_COUNT = 21
NODE_FEATURE_DIM = 62
EDGE_FEATURE_DIM = 47

# 与 HeadlessGame.get_state() 保持一致；未显式传 stable 时据此计算。
STABLE_VELOCITY_EPSILON = 35.0
STABLE_ANGULAR_VELOCITY_EPSILON = 4.0


class MobileSceneError(ValueError):
    """表示 Android 局面缺字段、类型错误或与最终模型规格不兼容。"""


def _mapping(payload):
    """接受普通 Mapping 或 UTF-8 JSON，并返回最外层字典。"""

    if isinstance(payload, (str, bytes, bytearray)):
        try:
            payload = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MobileSceneError(f'invalid scene JSON: {exc}') from exc
    if not isinstance(payload, Mapping):
        raise MobileSceneError('scene must be a mapping or JSON object')
    return payload


def _sequence(name, value):
    """读取真正的数组，拒绝字符串被误拆成逐字符列表。"""

    if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes, bytearray))):
        raise MobileSceneError(f'{name} must be an array')
    return tuple(value)


def _integer(name, value, *, minimum=None, maximum=None):
    """读取无损整数；避免 ``int(1.8)`` 之类的静默截断。"""

    if isinstance(value, bool):
        raise MobileSceneError(f'{name} must be an integer')
    try:
        result = index(value)
    except TypeError as exc:
        raise MobileSceneError(f'{name} must be an integer') from exc
    if minimum is not None and result < minimum:
        raise MobileSceneError(f'{name} must be >= {minimum}')
    if maximum is not None and result > maximum:
        raise MobileSceneError(f'{name} must be <= {maximum}')
    return result


def _number(name, value, *, minimum=None):
    """读取 JSON 有限实数，拒绝 NaN/Inf 污染模型输入。"""

    if isinstance(value, bool):
        raise MobileSceneError(f'{name} must be a finite number')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise MobileSceneError(f'{name} must be a finite number') from exc
    if not math.isfinite(result):
        raise MobileSceneError(f'{name} must be a finite number')
    if minimum is not None and result < minimum:
        raise MobileSceneError(f'{name} must be >= {minimum}')
    return result


def _boolean(name, value):
    """读取严格布尔值，避免 JSON 中的 0/1 隐式改变稳定边界语义。"""

    if not isinstance(value, bool):
        raise MobileSceneError(f'{name} must be a boolean')
    return value


def _optional_number(mapping, key, default, *, minimum=None, prefix=''):
    """读取一个可选数值字段并统一生成带路径的错误名。"""

    if key not in mapping:
        return default
    return _number(f'{prefix}{key}', mapping[key], minimum=minimum)


def _geometry(scene):
    """解析并锁定最终模型的 560x1120 场景规格。"""

    raw = scene.get('geometry', {})
    if not isinstance(raw, Mapping):
        raise MobileSceneError('geometry must be an object')

    def value(name, default):
        return raw.get(name, scene.get(name, default))

    geometry = BoardGeometry(
        width=_integer('geometry.width', value('width', BOARD_WIDTH)),
        height=_integer('geometry.height', value('height', BOARD_HEIGHT)),
        spawn_y=_integer('geometry.spawn_y', value('spawn_y', SPAWN_Y)),
        wall_width=_integer(
            'geometry.wall_width',
            value('wall_width', WALL_WIDTH),
            minimum=1,
        ),
        floor_y=_integer('geometry.floor_y', value('floor_y', FLOOR_Y)),
    )
    expected = (
        BOARD_WIDTH,
        BOARD_HEIGHT,
        SPAWN_Y,
        WALL_WIDTH,
        FLOOR_Y,
    )
    actual = (
        geometry.width,
        geometry.height,
        geometry.spawn_y,
        geometry.wall_width,
        geometry.floor_y,
    )
    if actual != expected:
        raise MobileSceneError(
            'scene geometry must be exactly '
            f'{BOARD_WIDTH}x{BOARD_HEIGHT}, spawn_y={SPAWN_Y}, '
            f'wall_width={WALL_WIDTH}, floor_y={FLOOR_Y}; got {actual}'
        )
    return geometry


def _queue_level(name, value):
    """允许 q0～q3 直接写等级，或写成 ``{"level": n}``。"""

    if isinstance(value, Mapping):
        if 'level' not in value:
            raise MobileSceneError(f'{name}.level is required')
        value = value['level']
    return _integer(
        name,
        value,
        minimum=SPAWN_FRUIT_MIN_LEVEL,
        maximum=SPAWN_FRUIT_MAX_LEVEL,
    )


def _fruit_queue(scene):
    """兼容 ``fruit_queue`` / ``queue`` 数组与独立 q0～q3 字段。"""

    if 'fruit_queue' in scene:
        values = _sequence('fruit_queue', scene['fruit_queue'])
    elif 'queue' in scene:
        values = _sequence('queue', scene['queue'])
    else:
        missing = [
            f'q{offset}'
            for offset in range(FRUIT_QUEUE_LENGTH)
            if f'q{offset}' not in scene
        ]
        if missing:
            raise MobileSceneError(
                'scene must provide fruit_queue/queue or all q0-q3 fields'
            )
        values = tuple(
            scene[f'q{offset}']
            for offset in range(FRUIT_QUEUE_LENGTH)
        )
    if len(values) != FRUIT_QUEUE_LENGTH:
        raise MobileSceneError(
            f'fruit queue must contain exactly {FRUIT_QUEUE_LENGTH} levels'
        )
    return tuple(
        _queue_level(f'fruit_queue[{offset}]', value)
        for offset, value in enumerate(values)
    )


def _fruit_state(raw, offset, geometry):
    """把一颗 Android 刚体快照转换成训练时的 ``FruitState``。"""

    if not isinstance(raw, Mapping):
        raise MobileSceneError(f'fruits[{offset}] must be an object')
    prefix = f'fruits[{offset}].'

    fruit_id_value = raw.get('fruit_id', raw.get('id'))
    if fruit_id_value is None:
        raise MobileSceneError(f'{prefix}fruit_id is required')
    fruit_id = _integer(
        f'{prefix}fruit_id',
        fruit_id_value,
        minimum=1,
    )
    if 'level' not in raw:
        raise MobileSceneError(f'{prefix}level is required')
    level = _integer(
        f'{prefix}level',
        raw['level'],
        minimum=MIN_FRUIT_LEVEL,
        maximum=MAX_FRUIT_LEVEL,
    )
    if 'x' not in raw or 'y' not in raw:
        raise MobileSceneError(f'{prefix}x and {prefix}y are required')

    display_radius = float(fruit_radius(level))
    if 'radius' in raw:
        supplied_radius = _number(
            f'{prefix}radius',
            raw['radius'],
            minimum=1e-9,
        )
        if not math.isclose(
                supplied_radius,
                display_radius,
                rel_tol=0.0,
                abs_tol=1e-6):
            raise MobileSceneError(
                f'{prefix}radius must match level {level} display radius '
                f'{display_radius}'
            )

    physics_radius_value = raw.get(
        'physics_radius',
        raw.get(
            'collision_radius',
            dropped_fruit_physics_radius(level),
        ),
    )
    physics_radius = _number(
        f'{prefix}physics_radius',
        physics_radius_value,
        minimum=1e-9,
    )
    x = _number(f'{prefix}x', raw['x'])
    y = _number(f'{prefix}y', raw['y'])
    vx = _optional_number(raw, 'vx', 0.0, prefix=prefix)
    vy = _optional_number(raw, 'vy', 0.0, prefix=prefix)
    angular_velocity = _optional_number(
        raw,
        'angular_velocity',
        0.0,
        prefix=prefix,
    )
    if 'stable' in raw:
        stable = _boolean(f'{prefix}stable', raw['stable'])
    else:
        stable = (
            math.hypot(vx, vy) <= STABLE_VELOCITY_EPSILON
            and abs(angular_velocity)
            <= STABLE_ANGULAR_VELOCITY_EPSILON
        )

    return FruitState(
        fruit_id=fruit_id,
        level=level,
        radius=display_radius,
        physics_radius=physics_radius,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        angle=_optional_number(raw, 'angle', 0.0, prefix=prefix),
        angular_velocity=angular_velocity,
        age_frames=_integer(
            f'{prefix}age_frames',
            raw.get('age_frames', 0),
            minimum=0,
        ),
        stable=stable,
        distance_to_left_wall=float(
            x - (geometry.wall_width + physics_radius)
        ),
        distance_to_right_wall=float(
            geometry.width
            - geometry.wall_width
            - physics_radius
            - x
        ),
        distance_to_floor=float(
            geometry.floor_y - physics_radius - y
        ),
        distance_to_danger_line=float(
            y - physics_radius - geometry.spawn_y
        ),
    )


def scene_to_game_state(payload):
    """把 Android ``dict`` / JSON 转成与 HeadlessGame 一致的 ``GameState``。"""

    scene = _mapping(payload)
    geometry = _geometry(scene)
    queue = _fruit_queue(scene)

    raw_fruits = scene.get('fruits', scene.get('board_fruits', ()))
    fruits = tuple(
        _fruit_state(raw, offset, geometry)
        for offset, raw in enumerate(_sequence('fruits', raw_fruits))
    )
    fruit_ids = tuple(fruit.fruit_id for fruit in fruits)
    if len(set(fruit_ids)) != len(fruit_ids):
        raise MobileSceneError('fruit_id values must be unique')

    highest_top = min(
        (
            fruit.y - fruit.physics_radius
            for fruit in fruits
        ),
        default=float(geometry.height),
    )
    playable_area = max(
        1.0,
        geometry.width * (geometry.height - geometry.spawn_y),
    )
    occupied_area = sum(
        math.pi * fruit.physics_radius ** 2
        for fruit in fruits
    )

    score = _integer('score', scene.get('score', 0), minimum=0)
    return GameState(
        board_fruits=fruits,
        fruit_queue=queue,
        score=score,
        last_score=_integer(
            'last_score',
            scene.get('last_score', score),
            minimum=0,
        ),
        step_count=_integer(
            'step_count',
            scene.get('step_count', 0),
            minimum=0,
        ),
        physics_frame=_integer(
            'physics_frame',
            scene.get('physics_frame', 0),
            minimum=0,
        ),
        done=_boolean('done', scene.get('done', False)),
        geometry=geometry,
        max_height=(
            0.0
            if not fruits
            else float(geometry.height - highest_top)
        ),
        fruit_count=len(fruits),
        max_level=max((fruit.level for fruit in fruits), default=0),
        empty_space_ratio=max(
            0.0,
            min(1.0, 1.0 - occupied_area / playable_area),
        ),
    )


def build_action_candidates(state):
    """按 q0 显示半径生成和训练完全一致的 21 个动作节点。"""

    if not isinstance(state, GameState):
        raise TypeError('state must be GameState')
    if not state.fruit_queue:
        raise MobileSceneError('state fruit_queue must not be empty')

    current_level = int(state.fruit_queue[0])
    positions = drop_x_positions_for_level(
        state.geometry,
        current_level,
        action_count=ACTION_COUNT,
    )
    left = positions[0]
    right = positions[-1]
    return tuple(
        ActionCandidate(
            action_index=offset,
            drop_x=drop_x,
            normalized_drop_x=(
                0.0
                if right == left
                else (drop_x - left) / (right - left)
            ),
            current_level=current_level,
            current_radius=float(fruit_radius(current_level)),
            current_physics_radius=float(
                dropped_fruit_physics_radius(current_level)
            ),
        )
        for offset, drop_x in enumerate(positions)
    )


def scene_to_transition_key(payload, state):
    """构造推理局面的稳定身份，并强制 key 与 ``step_count`` 对齐。"""

    scene = _mapping(payload)
    if not isinstance(state, GameState):
        raise TypeError('state must be GameState')
    raw = scene.get('transition_key', scene.get('transition', {}))
    if not isinstance(raw, Mapping):
        raise MobileSceneError('transition_key must be an object')

    step_index = _integer(
        'transition_key.step_index',
        raw.get('step_index', state.step_count),
        minimum=0,
    )
    if step_index != state.step_count:
        raise MobileSceneError(
            'transition_key.step_index must equal scene step_count'
        )
    return TransitionKey(
        worker_id=_integer(
            'transition_key.worker_id',
            raw.get('worker_id', 0),
            minimum=0,
        ),
        episode_id=_integer(
            'transition_key.episode_id',
            raw.get('episode_id', scene.get('episode_id', 0)),
            minimum=0,
        ),
        step_index=step_index,
    )


def _flatten_rows(rows):
    """按 row-major 顺序展开二维特征，避免移动端再处理嵌套 Python tuple。"""

    return [
        float(value)
        for row in rows
        for value in row
    ]


def _flat_edge_index(edge_index):
    """把 ``[(src, dst), ...]`` 展开成模型需要的连续 ``[2, E]`` 布局。"""

    sources = [int(source) for source, _ in edge_index]
    targets = [int(target) for _, target in edge_index]
    return sources + targets


class MobileGraphBridge:
    """可在 Chaquopy 进程内复用 analyzer/builder 实例的无状态桥。"""

    def __init__(
            self,
            analyzer_config=None,
            graph_builder_config=None):
        self.analyzer = StateAnalyzer(
            analyzer_config or StateAnalyzerConfig()
        )
        self.graph_builder = GraphBuilder(
            graph_builder_config or GraphBuilderConfig()
        )

        # 62/47 是已训练 checkpoint 的 ABI，不允许源码特征表变化后静默继续推理。
        if len(NODE_FEATURE_NAMES) != NODE_FEATURE_DIM:
            raise RuntimeError(
                f'node feature ABI changed: expected {NODE_FEATURE_DIM}, '
                f'got {len(NODE_FEATURE_NAMES)}'
            )
        if len(EDGE_FEATURE_NAMES) != EDGE_FEATURE_DIM:
            raise RuntimeError(
                f'edge feature ABI changed: expected {EDGE_FEATURE_DIM}, '
                f'got {len(EDGE_FEATURE_NAMES)}'
            )

    def build(self, payload):
        """运行完整结构分析并返回扁平、可直接 JSON 序列化的模型输入。"""

        scene = _mapping(payload)
        state = scene_to_game_state(scene)
        actions = build_action_candidates(state)
        transition_key = scene_to_transition_key(scene, state)

        stable_boundary = scene.get('stable_boundary')
        if stable_boundary is not None:
            stable_boundary = _boolean(
                'stable_boundary',
                stable_boundary,
            )
        analysis = self.analyzer.analyze(
            state,
            actions,
            transition_key,
            stable_boundary=stable_boundary,
        )
        graph = self.graph_builder.build(
            state,
            actions,
            state_analysis=analysis,
        )

        if graph.node_feature_dim != NODE_FEATURE_DIM:
            raise RuntimeError('GraphBuilder returned wrong node feature dim')
        if graph.edge_feature_dim != EDGE_FEATURE_DIM:
            raise RuntimeError('GraphBuilder returned wrong edge feature dim')
        if graph.action_indices != tuple(range(ACTION_COUNT)):
            raise RuntimeError('GraphBuilder returned an incompatible action ABI')

        global_node_indices = tuple(
            node_index
            for node_index, node_ref in enumerate(graph.node_refs)
            if node_ref.node_type == 'global'
        )
        if len(global_node_indices) != 1:
            raise RuntimeError(
                'GraphBuilder must return exactly one global node'
            )

        result = {
            'schema_version': MOBILE_GRAPH_SCHEMA_VERSION,
            'node_features': _flatten_rows(graph.node_features),
            'node_features_shape': [
                graph.num_nodes,
                NODE_FEATURE_DIM,
            ],
            # 先放全部 source，再放全部 target；reshape(2, num_edges) 后与
            # GraphTensor.edge_index 的连续内存布局完全相同。
            'edge_index': _flat_edge_index(graph.edge_index),
            'edge_index_shape': [2, graph.num_edges],
            'edge_features': _flatten_rows(graph.edge_features),
            'edge_features_shape': [
                graph.num_edges,
                EDGE_FEATURE_DIM,
            ],
            'action_node_indices': [
                int(value)
                for value in graph.action_node_indices
            ],
            'action_indices': [
                int(value)
                for value in graph.action_indices
            ],
            # 名称与 ONNX 的第五个输入 ``global_node_index`` 完全一致；虽然只有
            # 一个全局节点，仍保留长度为 1 的数组以匹配模型张量 shape [1]。
            'global_node_index': [
                int(value)
                for value in global_node_indices
            ],
            'node_feature_names': list(graph.node_feature_names),
            'edge_feature_names': list(graph.edge_feature_names),
            'analysis': {
                'valid_for_attribution': bool(
                    analysis.diagnostics.valid_for_attribution
                ),
                'degraded': bool(analysis.diagnostics.degraded),
                'top_connected_capacity': float(
                    analysis.top_connected_capacity
                ),
                'recoverability': float(analysis.recoverability),
                'chain_readiness': float(analysis.chain_readiness),
                'warning_codes': list(
                    analysis.diagnostics.warning_codes
                ),
            },
        }

        # ``allow_nan=False`` 的轻量预检保证 Java/Kotlin 不会收到非标准 JSON 数值。
        try:
            json.dumps(result, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                'mobile graph contains a non-serializable value'
            ) from exc
        return result

    def build_json(self, payload):
        """返回紧凑 UTF-8 JSON 文本，适合 Chaquopy ``callAttr`` 直接取回。"""

        return json.dumps(
            self.build(payload),
            allow_nan=False,
            ensure_ascii=True,
            separators=(',', ':'),
        )


_DEFAULT_BRIDGE = MobileGraphBridge()


def build_mobile_graph(payload):
    """使用默认训练配置构建一个移动端模型输入字典。"""

    return _DEFAULT_BRIDGE.build(payload)


def build_mobile_graph_json(payload):
    """使用默认训练配置构建一个移动端模型输入 JSON 字符串。"""

    return _DEFAULT_BRIDGE.build_json(payload)


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
