"""把游戏状态转换成 GNN 图。

`GraphBuilder` 属于 RL 侧，它只读取 `GameState` 和 `ActionCandidate`
这些公开数据结构，不访问 pygame、pymunk，也不反向修改游戏状态。
"""

import math
from dataclasses import dataclass

from daxigua.core.rules import FRUIT_RADII, MAX_FRUIT_LEVEL, fruit_radius

from .schema import (
    BOUNDARY_TYPES,
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    GraphData,
    GraphEdgeRef,
    GraphNodeRef,
)


# 节点类型到 one-hot 特征名的映射。
# GraphBuilder 内部用 `node_type` 表示节点语义，最终写入统一的 `node_features` 时，
# 会把对应的 `is_xxx_node` 特征置为 1，帮助模型区分不同节点来源。
NODE_TYPE_FEATURES = {
    'board_fruit': 'is_board_fruit_node',       # 场地内已经真实存在、会参与物理碰撞和合成的水果节点。
    'queue_fruit': 'is_queue_fruit_node',       # 顶部待投放序列中的水果节点，例如 q0、q1、q2、q3。
    'action': 'is_action_node',                 # 一个候选投放落点节点；后续 Q 值会从这些节点上读出。
    'global': 'is_global_node',                 # 全局局面摘要节点，保存分数、高度、场上水果数量等整体信息。
    'boundary': 'is_boundary_node',             # 边界节点，包括左墙、右墙、地板和死亡线。
}

# 边类型到 one-hot 特征名的映射。
# 这里的 key 是构图时使用的关系类型，value 是写入 `edge_features` 的类型标记。
# 第一版先把所有边放在同一个特征矩阵里，后续如果切换成真正异构图，也可以
# 根据这些类型拆成不同 relation。
EDGE_TYPE_FEATURES = {
    'board_fruit_to_board_fruit': 'is_board_fruit_pair_edge',       # 场上水果与场上水果之间的关系，描述距离、等级差、同级判断和相对速度。
    'action_to_board_fruit': 'is_action_board_fruit_edge',           # 候选动作与场上水果之间的关系，描述投放路径、水平距离和潜在合成机会。
    'queue_fruit_to_queue_fruit': 'is_queue_fruit_order_edge',       # 待投放队列内部的顺序关系，描述 q0 到 q3 的先后位置。
    'queue_fruit_to_board_fruit': 'is_queue_board_fruit_edge',       # 待投放水果与场上水果之间的等级匹配关系，用于表达未来合成潜力。
    'action_to_queue_fruit': 'is_action_queue_fruit_edge',           # 候选动作与待投放队列之间的关系，让动作节点知道当前与未来水果。
    'board_fruit_to_boundary': 'is_board_boundary_edge',             # 场上水果与边界之间的关系，描述靠墙、贴地、接近死亡线等风险。
    'global_to_node': 'is_global_edge',                              # 全局节点与其他节点之间的广播关系。
}

# 具体边界类型到 one-hot 特征名的映射。
# 同一组特征会同时用于边界节点和水果-边界边，保持含义一致。
BOUNDARY_FEATURES = {
    'left_wall': 'is_left_wall',           # 左侧物理墙体。
    'right_wall': 'is_right_wall',         # 右侧物理墙体。
    'floor': 'is_floor',                   # 底部地板。
    'danger_line': 'is_danger_line',       # 顶部死亡/危险线，水果长时间越过这里会导致游戏结束。
}

# 用集合做特征名校验，避免新增特征时因为拼写错误被静默丢弃。
NODE_FEATURE_NAME_SET = set(NODE_FEATURE_NAMES)
EDGE_FEATURE_NAME_SET = set(EDGE_FEATURE_NAMES)


@dataclass(frozen=True)
class GraphBuilderConfig:
    """GraphBuilder 的归一化配置。

    这些值不是游戏规则，只是为了把不同量纲的特征压到相近范围，
    避免模型一开始被分数、速度这类大数值主导。
    """

    velocity_scale: float = 2000.0                 # 线速度归一化比例，用于 vx、vy、relative_vx、relative_vy 等特征。
    fruit_count_scale: float = 64.0                # 场上水果数量归一化比例，用于 fruit_count。
    connect_global_node: bool = True               # 是否让 global 节点和其他所有节点双向连接。


class GraphBuilder:
    """从 `GameState` 和候选动作构建一张有向关系图。"""

    def __init__(self, config=None):
        self.config = config or GraphBuilderConfig()
        self.max_radius = max(FRUIT_RADII.values())
        # 特征向量构造是训练采样中的高频路径。提前缓存“特征名 -> 列号”，
        # 后续每个节点/边只需要修改少量非零列，避免反复创建完整字段字典。
        self._node_feature_indices = {
            feature_name: index
            for index, feature_name in enumerate(NODE_FEATURE_NAMES)
        }
        self._edge_feature_indices = {
            feature_name: index
            for index, feature_name in enumerate(EDGE_FEATURE_NAMES)
        }
        self._node_type_feature_indices = {
            node_type: self._node_feature_indices[feature_name]
            for node_type, feature_name in NODE_TYPE_FEATURES.items()
        }
        self._edge_type_feature_indices = {
            edge_type: self._edge_feature_indices[feature_name]
            for edge_type, feature_name in EDGE_TYPE_FEATURES.items()
        }

    def build(self, state, action_candidates):
        """构建一张 GNN 输入图。

        参数：
        - `state`: `HeadlessGame.get_state()` 返回的 `GameState`
        - `action_candidates`: `HeadlessGame.get_action_candidates(...)` 返回的动作列表

        返回：
        - `GraphData`: 框架无关图数据，后续可以转换成 torch tensor 或 PyG Data
        """

        fruits = tuple(state.board_fruits)
        queue = tuple(state.fruit_queue)
        actions = tuple(action_candidates)
        geometry = state.geometry

        nodes = []
        node_features = []
        edge_index = []
        edge_features = []
        edge_refs = []

        board_node_indices = []
        queue_node_indices = []
        action_node_indices = []
        boundary_node_indices = {}

        # 1. 场地中的真实水果节点。
        for fruit_index, fruit in enumerate(fruits):
            node_index = self._add_node(
                nodes,
                node_features,
                GraphNodeRef(
                    node_type='board_fruit',
                    source_index=fruit_index,
                    source_id=fruit.fruit_id,
                    label=f'fruit:{fruit.fruit_id}',
                ),
                self._board_fruit_features(fruit, geometry),
            )
            board_node_indices.append(node_index)

        # 2. 待投放队列节点。它们没有物理位置，只表达未来水果的等级和顺序。
        for queue_index, level in enumerate(queue):
            node_index = self._add_node(
                nodes,
                node_features,
                GraphNodeRef(
                    node_type='queue_fruit',
                    source_index=queue_index,
                    label=f'q{queue_index}',
                ),
                self._queue_fruit_features(level, queue_index, len(queue)),
            )
            queue_node_indices.append(node_index)

        # 3. 候选动作节点。最终 Q 值会从这些节点读出。
        for action_offset, action in enumerate(actions):
            node_index = self._add_node(
                nodes,
                node_features,
                GraphNodeRef(
                    node_type='action',
                    source_index=action.action_index,
                    label=f'action:{action.action_index}',
                ),
                self._action_features(action, action_offset, len(actions), geometry),
            )
            action_node_indices.append(node_index)

        # 4. 全局节点。它像一个广播节点，向局部对象提供全局局面摘要。
        global_node_index = self._add_node(
            nodes,
            node_features,
            GraphNodeRef(node_type='global', label='global'),
            self._global_features(state),
        )

        # 5. 边界节点。边界也作为对象进入图，让模型显式看到死亡线和墙体。
        for boundary_type in BOUNDARY_TYPES:
            node_index = self._add_node(
                nodes,
                node_features,
                GraphNodeRef(node_type='boundary', label=boundary_type),
                self._boundary_features(boundary_type, geometry),
            )
            boundary_node_indices[boundary_type] = node_index

        # 6. 按设计文档建立不同类型的边。所有空间边都做成有向边，
        # 这样普通 message passing 层不需要额外处理无向图。

        # 场上水果之间的空间/合成关系。
        self._connect_board_fruits(
            fruits,
            board_node_indices,
            geometry,
            edge_index,
            edge_features,
            edge_refs,
        )

        # 候选动作和场上水果之间的关系，表达每个落点可能影响哪些水果。
        self._connect_actions_to_board(
            actions,
            action_node_indices,
            fruits,
            board_node_indices,
            geometry,
            edge_index,
            edge_features,
            edge_refs,
        )

        # 队列内水果的顺序关系。
        self._connect_queue_order(
            queue,
            queue_node_indices,
            edge_index,
            edge_features,
            edge_refs,
        )

        # 队列水果和场上水果之间的等级匹配关系，表达未来合成潜力。
        self._connect_queue_to_board(
            queue,
            queue_node_indices,
            fruits,
            board_node_indices,
            geometry,
            edge_index,
            edge_features,
            edge_refs,
        )

        # 候选动作和队列水果之间的关系，让动作节点知道当前与未来水果。
        self._connect_actions_to_queue(
            actions,
            action_node_indices,
            queue,
            queue_node_indices,
            edge_index,
            edge_features,
            edge_refs,
        )

        # 场上水果和边界之间的关系，显式暴露墙体、地板和死亡线风险。
        self._connect_board_to_boundaries(
            fruits,
            board_node_indices,
            boundary_node_indices,
            geometry,
            edge_index,
            edge_features,
            edge_refs,
        )

        if self.config.connect_global_node:
            self._connect_global(
                global_node_index,
                len(nodes),
                edge_index,
                edge_features,
                edge_refs,
            )

        return GraphData(
            node_features=tuple(node_features),
            edge_index=tuple(edge_index),
            edge_features=tuple(edge_features),
            node_refs=tuple(nodes),
            edge_refs=tuple(edge_refs),
            action_node_indices=tuple(action_node_indices),
            action_indices=tuple(action.action_index for action in actions),
        )

    def _add_node(self, nodes, node_features, node_ref, feature_values):
        """追加一个节点，并返回它在图中的整数编号。"""

        node_index = len(nodes)
        nodes.append(node_ref)
        node_features.append(self._node_vector(node_ref.node_type, feature_values))
        return node_index

    def _add_edge(self, edge_index, edge_features, edge_refs, source, target, edge_type, feature_values):
        """追加一条有向边。"""

        edge_index.append((source, target))
        edge_features.append(self._edge_vector(edge_type, feature_values))
        edge_refs.append(GraphEdgeRef(edge_type=edge_type, source_node=source, target_node=target))

    def _node_vector(self, node_type, feature_values):
        """把字典形式的节点特征转成固定顺序的向量。"""

        unknown_names = set(feature_values) - NODE_FEATURE_NAME_SET
        if unknown_names:
            raise KeyError(f'unknown node feature names: {sorted(unknown_names)}')

        values = [0.0] * len(NODE_FEATURE_NAMES)
        values[self._node_type_feature_indices[node_type]] = 1.0
        for feature_name, feature_value in feature_values.items():
            values[self._node_feature_indices[feature_name]] = float(feature_value)
        return tuple(values)

    def _edge_vector(self, edge_type, feature_values):
        """把字典形式的边特征转成固定顺序的向量。"""

        unknown_names = set(feature_values) - EDGE_FEATURE_NAME_SET
        if unknown_names:
            raise KeyError(f'unknown edge feature names: {sorted(unknown_names)}')

        values = [0.0] * len(EDGE_FEATURE_NAMES)
        values[self._edge_type_feature_indices[edge_type]] = 1.0
        for feature_name, feature_value in feature_values.items():
            values[self._edge_feature_indices[feature_name]] = float(feature_value)
        return tuple(values)

    def _board_fruit_features(self, fruit, geometry):
        """生成场上水果节点特征。"""

        return {
            'x': self._signed(fruit.x, geometry.width),
            'y': self._signed(fruit.y, geometry.height),
            'vx': self._signed(fruit.vx, self.config.velocity_scale),
            'vy': self._signed(fruit.vy, self.config.velocity_scale),
            'level': self._level(fruit.level),
            'radius': self._radius(fruit.radius),
            'stable': self._flag(fruit.stable),
            'distance_to_left_wall': self._signed(fruit.distance_to_left_wall, geometry.width),
            'distance_to_right_wall': self._signed(fruit.distance_to_right_wall, geometry.width),
            'distance_to_floor': self._signed(fruit.distance_to_floor, geometry.height),
            'distance_to_danger_line': self._signed(
                fruit.distance_to_danger_line,
                self._playable_height(geometry),
            ),
        }

    def _queue_fruit_features(self, level, queue_index, queue_length):
        """生成待投放队列节点特征。"""

        return {
            'level': self._level(level),
            'radius': self._radius(fruit_radius(level)),
            'queue_index': self._queue_index(queue_index, queue_length),
            'is_current_queue_fruit': self._flag(queue_index == 0),
        }

    def _action_features(self, action, action_offset, action_count, geometry):
        """生成候选动作节点特征。"""

        return {
            'x': self._signed(action.drop_x, geometry.width),
            'action_index': self._queue_index(action_offset, action_count),
            'level': self._level(action.current_level),
            'radius': self._radius(action.current_radius),
        }

    def _global_features(self, state):
        """生成全局节点特征。"""

        geometry = state.geometry
        return {
            'max_height': self._unsigned(state.max_height, self._playable_height(geometry)),
            'fruit_count': self._unsigned(state.fruit_count, self.config.fruit_count_scale),
            'max_level': self._level(state.max_level),
            'empty_space_ratio': self._unit(state.empty_space_ratio),
        }

    def _boundary_features(self, boundary_type, geometry):
        """生成边界节点特征。"""

        values = {
            BOUNDARY_FEATURES[boundary_type]: 1.0,
        }

        if boundary_type == 'left_wall':
            values.update({
                'x': 0.0,
                'y': 0.5,
                'boundary_position': 0.0,
            })
        elif boundary_type == 'right_wall':
            values.update({
                'x': 1.0,
                'y': 0.5,
                'boundary_position': 1.0,
            })
        elif boundary_type == 'floor':
            values.update({
                'x': 0.5,
                'y': self._signed(geometry.floor_y, geometry.height),
                'boundary_position': self._signed(geometry.floor_y, geometry.height),
            })
        elif boundary_type == 'danger_line':
            values.update({
                'x': 0.5,
                'y': self._signed(geometry.spawn_y, geometry.height),
                'boundary_position': self._signed(geometry.spawn_y, geometry.height),
            })

        return values

    def _connect_board_fruits(
            self,
            fruits,
            board_node_indices,
            geometry,
            edge_index,
            edge_features,
            edge_refs):
        """连接场上水果之间的空间/合成关系。"""

        for source_offset, source_fruit in enumerate(fruits):
            for target_offset, target_fruit in enumerate(fruits):
                if source_offset == target_offset:
                    continue

                features = self._fruit_pair_edge_features(source_fruit, target_fruit, geometry)
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    board_node_indices[source_offset],
                    board_node_indices[target_offset],
                    'board_fruit_to_board_fruit',
                    features,
                )

    def _connect_actions_to_board(
            self,
            actions,
            action_node_indices,
            fruits,
            board_node_indices,
            geometry,
            edge_index,
            edge_features,
            edge_refs):
        """连接候选动作和场上水果，表达每个落点可能影响哪些水果。"""

        for action_offset, action in enumerate(actions):
            for fruit_offset, fruit in enumerate(fruits):
                action_to_fruit = self._action_board_edge_features(action, fruit, geometry, reverse=False)
                fruit_to_action = self._action_board_edge_features(action, fruit, geometry, reverse=True)

                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    action_node_indices[action_offset],
                    board_node_indices[fruit_offset],
                    'action_to_board_fruit',
                    action_to_fruit,
                )
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    board_node_indices[fruit_offset],
                    action_node_indices[action_offset],
                    'action_to_board_fruit',
                    fruit_to_action,
                )

    def _connect_queue_order(self, queue, queue_node_indices, edge_index, edge_features, edge_refs):
        """连接队列水果之间的时间顺序关系。"""

        queue_length = len(queue)
        for source_index, source_level in enumerate(queue):
            for target_index, target_level in enumerate(queue):
                if source_index == target_index:
                    continue

                features = {
                    'order_gap': self._signed(target_index - source_index, max(1, queue_length - 1)),
                    'is_next_queue_fruit': self._flag(target_index == source_index + 1),
                    'level_diff': self._level_diff(target_level - source_level),
                    'abs_level_diff': self._abs_level_diff(target_level - source_level),
                    'same_level': self._flag(target_level == source_level),
                }
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    queue_node_indices[source_index],
                    queue_node_indices[target_index],
                    'queue_fruit_to_queue_fruit',
                    features,
                )

    def _connect_queue_to_board(
            self,
            queue,
            queue_node_indices,
            fruits,
            board_node_indices,
            geometry,
            edge_index,
            edge_features,
            edge_refs):
        """连接未来水果和场上水果，表达等级匹配与未来合成潜力。"""

        for queue_index, queue_level in enumerate(queue):
            for fruit_index, fruit in enumerate(fruits):
                level_diff = queue_level - fruit.level
                features = {
                    'queue_index': self._queue_index(queue_index, len(queue)),
                    'level_diff': self._level_diff(level_diff),
                    'abs_level_diff': self._abs_level_diff(level_diff),
                    'same_level': self._flag(level_diff == 0),
                }

                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    queue_node_indices[queue_index],
                    board_node_indices[fruit_index],
                    'queue_fruit_to_board_fruit',
                    features,
                )
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    board_node_indices[fruit_index],
                    queue_node_indices[queue_index],
                    'queue_fruit_to_board_fruit',
                    features,
                )

    def _connect_actions_to_queue(
            self,
            actions,
            action_node_indices,
            queue,
            queue_node_indices,
            edge_index,
            edge_features,
            edge_refs):
        """连接候选动作和队列水果，让动作节点知道当前与未来水果。"""

        for action_offset in range(len(actions)):
            for queue_index in range(len(queue)):
                features = {
                    'queue_index': self._queue_index(queue_index, len(queue)),
                }

                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    action_node_indices[action_offset],
                    queue_node_indices[queue_index],
                    'action_to_queue_fruit',
                    features,
                )
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    queue_node_indices[queue_index],
                    action_node_indices[action_offset],
                    'action_to_queue_fruit',
                    features,
                )

    def _connect_board_to_boundaries(
            self,
            fruits,
            board_node_indices,
            boundary_node_indices,
            geometry,
            edge_index,
            edge_features,
            edge_refs):
        """连接场上水果和边界，显式暴露墙体、地板和死亡线风险。"""

        for fruit_index, fruit in enumerate(fruits):
            for boundary_type in BOUNDARY_TYPES:
                features = self._boundary_edge_features(fruit, boundary_type, geometry)
                boundary_node_index = boundary_node_indices[boundary_type]
                fruit_node_index = board_node_indices[fruit_index]

                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    fruit_node_index,
                    boundary_node_index,
                    'board_fruit_to_boundary',
                    features,
                )
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    boundary_node_index,
                    fruit_node_index,
                    'board_fruit_to_boundary',
                    features,
                )

    def _connect_global(self, global_node_index, node_count, edge_index, edge_features, edge_refs):
        """把全局节点和其他所有节点双向连接。"""

        for node_index in range(node_count):
            if node_index == global_node_index:
                continue

            self._add_edge(
                edge_index,
                edge_features,
                edge_refs,
                global_node_index,
                node_index,
                'global_to_node',
                {},
            )
            self._add_edge(
                edge_index,
                edge_features,
                edge_refs,
                node_index,
                global_node_index,
                'global_to_node',
                {},
            )

    def _fruit_pair_edge_features(self, source, target, geometry):
        """计算两个场上水果之间的有向边特征。"""

        dx = target.x - source.x
        dy = target.y - source.y
        distance = math.hypot(dx, dy)
        radius_sum = source.radius + target.radius
        level_diff = target.level - source.level
        relative_vx = target.vx - source.vx
        relative_vy = target.vy - source.vy

        return {
            'dx': self._signed(dx, geometry.width),
            'dy': self._signed(dy, geometry.height),
            'distance': self._unsigned(distance, self._diagonal(geometry)),
            'horizontal_distance': self._unsigned(abs(dx), geometry.width),
            'vertical_distance': self._unsigned(abs(dy), geometry.height),
            'radius_sum': self._unsigned(radius_sum, self.max_radius * 2),
            'overlap_margin': self._signed(radius_sum - distance, self.max_radius * 2),
            'level_diff': self._level_diff(level_diff),
            'abs_level_diff': self._abs_level_diff(level_diff),
            'same_level': self._flag(level_diff == 0),
            'relative_vx': self._signed(relative_vx, self.config.velocity_scale),
            'relative_vy': self._signed(relative_vy, self.config.velocity_scale),
        }

    def _action_board_edge_features(self, action, fruit, geometry, reverse=False):
        """计算候选动作和场上水果之间的边特征。"""

        dx = fruit.x - action.drop_x        # 带方向的水平差：水果在落点右侧为正，左侧为负。
        dy = fruit.y - geometry.spawn_y     # 带方向的垂直差：水果在生成线下方通常为正。
        if reverse:                         # 如果是反向边，则交换方向。
            dx = -dx
            dy = -dy

        horizontal_distance = abs(action.drop_x - fruit.x)  # 不带方向的水平距离，判断水果是否接近投放路径。
        vertical_distance = abs(fruit.y - geometry.spawn_y)  # 不带方向的垂直距离，判断水果位于投放起点下方多远。
        radius_sum = action.current_radius + fruit.radius    # 两个水果横向接触所需的距离阈值。
        path_overlap_margin = radius_sum - horizontal_distance
        level_diff = action.current_level - fruit.level      # 当前投放水果和场上水果的等级差。

        return {
            'dx': self._signed(dx, geometry.width),
            'dy': self._signed(dy, geometry.height),
            'horizontal_distance': self._unsigned(horizontal_distance, geometry.width),
            'vertical_distance': self._unsigned(vertical_distance, geometry.height),
            'radius_sum': self._unsigned(radius_sum, self.max_radius * 2),
            'path_overlap_margin': self._signed(path_overlap_margin, self.max_radius * 2),
            'level_diff': self._level_diff(level_diff),
            'abs_level_diff': self._abs_level_diff(level_diff),
            'same_level': self._flag(level_diff == 0),
            'is_under_drop_path': self._flag(horizontal_distance <= radius_sum),
        }

    def _boundary_edge_features(self, fruit, boundary_type, geometry):
        """计算水果和边界之间的风险关系。"""

        if boundary_type == 'left_wall':
            distance = fruit.distance_to_left_wall
            scale = geometry.width
        elif boundary_type == 'right_wall':
            distance = fruit.distance_to_right_wall
            scale = geometry.width
        elif boundary_type == 'floor':
            distance = fruit.distance_to_floor
            scale = geometry.height
        elif boundary_type == 'danger_line':
            distance = fruit.distance_to_danger_line
            scale = self._playable_height(geometry)
        else:
            distance = 0.0
            scale = 1.0

        return {
            'distance_to_boundary': self._signed(distance, scale),
            # `distance` 已经是“水果外缘到边界”的距离，小于半径就说明比较贴近。
            'is_near_boundary': self._flag(distance <= fruit.radius),
        }

    def _level(self, level):
        """归一化水果等级。"""

        return self._unsigned(level, MAX_FRUIT_LEVEL)

    def _radius(self, radius):
        """归一化水果半径。"""

        return self._unsigned(radius, self.max_radius)

    def _level_diff(self, level_diff):
        """归一化有符号等级差。"""

        return self._signed(level_diff, MAX_FRUIT_LEVEL)

    def _abs_level_diff(self, level_diff):
        """归一化无符号等级差。"""

        return self._unsigned(abs(level_diff), MAX_FRUIT_LEVEL)

    def _queue_index(self, index, length):
        """把队列位置或动作位置压到 0 到 1。"""

        if length <= 1:
            return 0.0
        return self._unit(index / (length - 1))

    def _playable_height(self, geometry):
        """返回死亡线以下的可玩高度。"""

        return max(1.0, geometry.height - geometry.spawn_y)

    def _diagonal(self, geometry):
        """返回场地对角线长度，用于距离归一化。"""

        return max(1.0, math.hypot(geometry.width, geometry.height))

    def _flag(self, value):
        """布尔值转成 0/1 浮点数。"""

        return 1.0 if value else 0.0

    def _unsigned(self, value, scale):
        """把非负量压到 0 到 1。"""

        if scale == 0:
            return 0.0
        return self._unit(value / scale)

    def _signed(self, value, scale):
        """把有符号量压到 -1 到 1。"""

        if scale == 0:
            return 0.0
        return max(-1.0, min(1.0, value / scale))

    def _unit(self, value):
        """把数值截断到 0 到 1。"""

        return max(0.0, min(1.0, value))
