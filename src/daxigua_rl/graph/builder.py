"""把游戏状态转换成 GNN 图。

`GraphBuilder` 属于 RL 侧，它只读取 `GameState`、`ActionCandidate` 和可选的
`StateAnalysis` 这些公开只读数据结构，不访问 pygame、pymunk，也不反向修改游戏状态。
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from daxigua.core.rules import (
    FRUIT_RADII,
    MAX_FRUIT_LEVEL,
    dropped_fruit_physics_radius,
)
if TYPE_CHECKING:
    # 只供静态检查使用。运行时顶层导入 attribution.schema 会经过
    # training -> graph 的包初始化路径，形成不必要的循环导入。
    from daxigua_rl.attribution.schema import StateAnalysis

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
    'chain_motif': 'is_chain_motif_node',       # 当前稳定边界中可被当前或可见队列触发的局部连锁结构。
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
    'board_fruit_to_chain_motif': 'is_board_chain_motif_edge',       # motif 与组成它的场上水果之间的角色关系。
    'queue_fruit_to_chain_motif': 'is_queue_chain_motif_edge',       # motif 与当前可见兼容队列槽之间的时序关系。
    'action_to_chain_motif': 'is_action_chain_motif_edge',           # motif 与每个动作之间的立即触发、保护和破坏风险关系。
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
    relation_count_scale: float = 8.0              # 水果伙伴、支撑和 blocker 数量的归一化比例。
    cavity_count_scale: float = 8.0                # 封闭空腔数量的归一化比例。
    motif_member_count_scale: float = 4.0          # 一个局部 motif 成员数量的归一化比例。
    contact_gap_tolerance: float = 2.0             # 当前几何接触关系允许的像素级静态间隙。
    connect_global_node: bool = True               # 是否让 global 节点和其他所有节点双向连接。


@dataclass(frozen=True)
class _StructureContext:
    """一次构图内复用的 StateAnalysis 索引。

    这些字典和集合只用于把分析对象中的水果 ID 对齐回图节点，不会把 ID 的数值
    写入模型特征。显式索引也避免在 O(n²) 水果边循环里反复遍历 analysis tuple。
    """

    analysis: 'StateAnalysis'
    fruit_by_id: dict
    region_top_connected: dict
    support_relations: dict
    critical_edges: frozenset
    inversion_edges: frozenset
    analysis_offsets: tuple


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

    def build(self, state, action_candidates, state_analysis=None):
        """构建一张 GNN 输入图。

        参数：
        - `state`: `HeadlessGame.get_state()` 返回的 `GameState`
        - `action_candidates`: `HeadlessGame.get_action_candidates(...)` 返回的动作列表
        - `state_analysis`: 可选的同一动作前边界 `StateAnalysis`。传入后会添加
          显式结构特征和 chain motif 节点；省略时保持旧调用方式，所有新增列为 0

        返回：
        - `GraphData`: 框架无关图数据，后续可以转换成 torch tensor 或 PyG Data
        """

        fruits = tuple(state.board_fruits)
        queue = tuple(state.fruit_queue)
        actions = tuple(action_candidates)
        geometry = state.geometry
        structure = self._prepare_structure_context(
            state,
            actions,
            state_analysis,
        )

        nodes = []
        node_features = []
        edge_index = []
        edge_features = []
        edge_refs = []

        board_node_indices = []
        queue_node_indices = []
        action_node_indices = []
        motif_node_indices = []
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
                self._board_fruit_features(fruit, geometry, structure),
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
                self._action_features(
                    action,
                    action_offset,
                    len(actions),
                    geometry,
                    structure,
                ),
            )
            action_node_indices.append(node_index)

        # 4. 连锁 motif 虚拟节点。每个节点代表当前边界已由 StateAnalyzer
        # 识别出的一个 merge_pair 或 level_ladder，不代表预测出来的未来状态。
        if structure is not None:
            for motif_offset, motif in enumerate(
                    structure.analysis.chain_motifs):
                node_index = self._add_node(
                    nodes,
                    node_features,
                    GraphNodeRef(
                        node_type='chain_motif',
                        source_index=motif_offset,
                        label=f'chain_motif:{motif.motif_type}:{motif_offset}',
                    ),
                    self._chain_motif_features(
                        motif,
                        structure.analysis,
                    ),
                )
                motif_node_indices.append(node_index)

        # 5. 全局节点。它像一个广播节点，向局部对象提供全局局面摘要。
        global_node_index = self._add_node(
            nodes,
            node_features,
            GraphNodeRef(node_type='global', label='global'),
            self._global_features(state, structure),
        )

        # 6. 边界节点。边界也作为对象进入图，让模型显式看到死亡线和墙体。
        for boundary_type in BOUNDARY_TYPES:
            node_index = self._add_node(
                nodes,
                node_features,
                GraphNodeRef(node_type='boundary', label=boundary_type),
                self._boundary_features(boundary_type, geometry),
            )
            boundary_node_indices[boundary_type] = node_index

        # 7. 按设计文档建立不同类型的边。所有空间边都做成有向边，
        # 这样普通 message passing 层不需要额外处理无向图。

        # 场上水果之间的空间/合成关系。
        self._connect_board_fruits(
            fruits,
            board_node_indices,
            geometry,
            edge_index,
            edge_features,
            edge_refs,
            structure=structure,
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
            structure=structure,
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

        # motif 与成员水果、兼容队列槽和全部动作双向相连。尤其是 action 边会把
        # 21 位 trigger mask 拆成逐 action 的 0/1，避免把 mask 整数误当连续特征。
        if structure is not None:
            self._connect_chain_motifs(
                structure,
                fruits,
                board_node_indices,
                queue_node_indices,
                action_node_indices,
                motif_node_indices,
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

    def _prepare_structure_context(
            self,
            state,
            actions,
            state_analysis):
        """校验 StateAnalysis 与当前图输入严格对齐，并建立高频查询索引。

        图中的动作相关结构特征依赖 action offset，而不是外部 action_index。
        因此只要状态、动作顺序或投放横坐标有一项错位，就必须显式报错；静默接入
        另一时刻的 analysis 会构成比普通数值噪声更危险的监督污染。
        """

        if state_analysis is None:
            return None

        # 局部导入避开 attribution.schema -> training -> graph 的包初始化环。
        # build 真正被调用时各包已经完成初始化，因此这里仍能执行严格类型检查。
        from daxigua_rl.attribution.schema import StateAnalysis

        if not isinstance(state_analysis, StateAnalysis):
            raise TypeError('state_analysis must be StateAnalysis or None')
        if state_analysis.transition_key.step_index != state.step_count:
            raise ValueError(
                'state_analysis step_index must match state.step_count'
            )

        if not actions:
            raise ValueError(
                'action_candidates must not be empty when state_analysis '
                'is present'
            )
        if any(
                left.drop_x >= right.drop_x
                for left, right in zip(actions, actions[1:])):
            raise ValueError(
                'action_candidates must be ordered by increasing drop_x'
            )

        analyzed_positions = state_analysis.action_drop_x_by_offset
        exact_layout = (
            len(actions) == len(analyzed_positions)
            and tuple(
                action.action_index
                for action in actions
            ) == state_analysis.action_indices
            and all(
                math.isclose(
                    float(action.drop_x),
                    float(analyzed_x),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for action, analyzed_x in zip(
                    actions,
                    analyzed_positions,
                )
            )
        )
        if exact_layout:
            analysis_offsets = tuple(range(len(actions)))
        else:
            if len(actions) == len(analyzed_positions):
                raise ValueError(
                    'full action layout must match state_analysis '
                    'action_indices and drop_x exactly'
                )
            # StateAnalyzer 的规范协议固定 21 列，但小型测试环境仍允许 1/3/7
            # 个策略动作。此时按真实 drop_x 映射到最近的规范列，不按两边恰好都
            # 从 0 开始的 action_index 猜位置；正式 21 动作训练仍走上面的精确路径。
            analysis_offsets = tuple(
                min(
                    range(len(analyzed_positions)),
                    key=lambda analysis_offset: (
                        abs(
                            float(action.drop_x)
                            - float(
                                analyzed_positions[analysis_offset]
                            )
                        ),
                        analysis_offset,
                    ),
                )
                for action in actions
            )
            if any(
                    left >= right
                    for left, right in zip(
                        analysis_offsets,
                        analysis_offsets[1:],
                    )):
                raise ValueError(
                    'action_candidates cannot be aligned injectively to '
                    'state_analysis columns'
                )
            analyzed_step = min(
                right - left
                for left, right in zip(
                    analyzed_positions,
                    analyzed_positions[1:],
                )
            )
            for action, analysis_offset in zip(
                    actions,
                    analysis_offsets):
                distance = abs(
                    float(action.drop_x)
                    - float(analyzed_positions[analysis_offset])
                )
                if distance > analyzed_step / 2.0 + 1e-9:
                    raise ValueError(
                        'action candidate drop_x is outside the nearest '
                        'state_analysis column cell'
                    )

        queue = tuple(state.fruit_queue)
        analyzed_queue = tuple(
            lane.level
            for lane in state_analysis.queue_lane_analyses
        )
        if analyzed_queue != queue:
            raise ValueError(
                'state_analysis queue lanes must match state.fruit_queue'
            )
        q0_lane = state_analysis.queue_lane_analyses[0]
        for action_offset, action in enumerate(actions):
            if action.current_level != q0_lane.level:
                raise ValueError(
                    'action candidate level must match analyzed q0 at '
                    f'offset {action_offset}'
                )
            if not math.isclose(
                    self._action_physics_radius(action),
                    q0_lane.physics_radius,
                    rel_tol=0.0,
                    abs_tol=1e-9):
                raise ValueError(
                    'action candidate physics radius must match analyzed '
                    f'q0 at offset {action_offset}'
                )

        state_fruit_by_id = {
            fruit.fruit_id: fruit
            for fruit in state.board_fruits
        }
        if len(state_fruit_by_id) != len(tuple(state.board_fruits)):
            raise ValueError('state.board_fruits contains duplicate fruit_id')
        fruit_by_id = {
            fruit.fruit_id: fruit
            for fruit in state_analysis.fruit_analyses
        }
        if set(fruit_by_id) != set(state_fruit_by_id):
            raise ValueError(
                'state_analysis fruit IDs must match state.board_fruits'
            )
        for fruit_id, analyzed_fruit in fruit_by_id.items():
            state_fruit = state_fruit_by_id[fruit_id]
            if analyzed_fruit.level != state_fruit.level:
                raise ValueError(
                    f'state_analysis level mismatch for fruit {fruit_id}'
                )
            if not math.isclose(
                    analyzed_fruit.physics_radius,
                    self._fruit_physics_radius(state_fruit),
                    rel_tol=0.0,
                    abs_tol=1e-9):
                raise ValueError(
                    f'state_analysis physics radius mismatch for fruit '
                    f'{fruit_id}'
                )

        region_top_connected = {
            region.region_id: bool(region.top_connected)
            for region in state_analysis.free_space_regions
        }
        support_relations = {}
        for edge in state_analysis.support_edges:
            # 墙和地板已经由 boundary 节点显式表达；水果关系边这里只索引
            # fruit -> fruit 的 canonical 支撑、盖压和桥接方向。
            if edge.supporter_fruit_id is None:
                continue
            support_relations[
                (
                    edge.supporter_fruit_id,
                    edge.supported_fruit_id,
                    edge.relation,
                )
            ] = edge.confidence

        critical_edges = frozenset(
            (blocker_id, fruit.fruit_id)
            for fruit in state_analysis.fruit_analyses
            for blocker_id in fruit.critical_blocker_ids
        )
        inversion_edges = frozenset(
            (blocker_id, fruit.fruit_id)
            for fruit in state_analysis.fruit_analyses
            for blocker_id in fruit.inversion_blocker_ids
        )
        return _StructureContext(
            analysis=state_analysis,
            fruit_by_id=fruit_by_id,
            region_top_connected=region_top_connected,
            support_relations=support_relations,
            critical_edges=critical_edges,
            inversion_edges=inversion_edges,
            analysis_offsets=analysis_offsets,
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

    def _board_fruit_features(self, fruit, geometry, structure=None):
        """生成场上水果节点特征。"""

        values = {
            'x': self._signed(fruit.x, geometry.width),
            'y': self._signed(fruit.y, geometry.height),
            'vx': self._signed(fruit.vx, self.config.velocity_scale),
            'vy': self._signed(fruit.vy, self.config.velocity_scale),
            'level': self._level(fruit.level),
            'radius': self._radius(self._fruit_physics_radius(fruit)),
            'stable': self._flag(fruit.stable),
            'distance_to_left_wall': self._signed(fruit.distance_to_left_wall, geometry.width),
            'distance_to_right_wall': self._signed(fruit.distance_to_right_wall, geometry.width),
            'distance_to_floor': self._signed(fruit.distance_to_floor, geometry.height),
            'distance_to_danger_line': self._signed(
                fruit.distance_to_danger_line,
                self._playable_height(geometry),
            ),
        }
        if structure is None:
            return values

        analyzed = structure.fruit_by_id[fruit.fruit_id]
        connected_to_top = (
            analyzed.connected_region_id is not None
            and structure.region_top_connected.get(
                analyzed.connected_region_id,
                False,
            )
        )
        values.update({
            'reachable_action_fraction': analyzed.reachable_fraction,
            'top_visible_ratio': analyzed.top_visible_ratio,
            'partner_reachable': self._flag(analyzed.partner_reachable),
            'partner_count': self._unsigned(
                len(analyzed.partner_ids),
                self.config.relation_count_scale,
            ),
            'reachable_partner_count': self._unsigned(
                len(analyzed.reachable_partner_ids),
                self.config.relation_count_scale,
            ),
            'support_parent_count': self._unsigned(
                len(analyzed.support_parent_ids),
                self.config.relation_count_scale,
            ),
            'supported_child_count': self._unsigned(
                len(analyzed.supported_child_ids),
                self.config.relation_count_scale,
            ),
            'burial_depth': analyzed.burial_depth,
            'inversion_count': self._unsigned(
                analyzed.inversion_count,
                self.config.relation_count_scale,
            ),
            'critical_blocker_count': self._unsigned(
                len(analyzed.critical_blocker_ids),
                self.config.relation_count_scale,
            ),
            'connected_to_top_space': self._flag(connected_to_top),
        })
        return values

    def _queue_fruit_features(self, level, queue_index, queue_length):
        """生成待投放队列节点特征。"""

        return {
            'level': self._level(level),
            'radius': self._radius(dropped_fruit_physics_radius(level)),
            'queue_index': self._queue_index(queue_index, queue_length),
            'is_current_queue_fruit': self._flag(queue_index == 0),
        }

    def _action_features(
            self,
            action,
            action_offset,
            action_count,
            geometry,
            structure=None):
        """生成候选动作节点特征。"""

        values = {
            'x': self._signed(action.drop_x, geometry.width),
            'action_index': self._queue_index(action_offset, action_count),
            'level': self._level(action.current_level),
            'radius': self._radius(self._action_physics_radius(action)),
        }
        if structure is None:
            return values

        q0_lane = structure.analysis.queue_lane_analyses[0]
        analysis_offset = structure.analysis_offsets[action_offset]
        values.update({
            'q0_landing_depth': q0_lane.landing_depths_by_action[
                analysis_offset
            ],
            'q0_is_safe': self._flag(
                self._mask_contains(
                    q0_lane.safe_action_mask,
                    analysis_offset,
                )
            ),
            'q0_blocker_count': self._unsigned(
                len(q0_lane.blocker_ids_by_action[analysis_offset]),
                self.config.relation_count_scale,
            ),
        })
        return values

    def _global_features(self, state, structure=None):
        """生成全局节点特征。"""

        geometry = state.geometry
        values = {
            'max_height': self._unsigned(state.max_height, self._playable_height(geometry)),
            'fruit_count': self._unsigned(state.fruit_count, self.config.fruit_count_scale),
            'max_level': self._level(state.max_level),
            'empty_space_ratio': self._unit(state.empty_space_ratio),
        }
        if structure is None:
            return values

        analysis = structure.analysis
        values.update({
            'has_state_analysis': 1.0,
            'analysis_valid': self._flag(
                analysis.diagnostics.valid_for_attribution
            ),
            'analysis_degraded': self._flag(analysis.diagnostics.degraded),
            'top_connected_capacity': analysis.top_connected_capacity,
            'recoverability': analysis.recoverability,
            'chain_readiness': analysis.chain_readiness,
            'top_connected_free_space_ratio': (
                analysis.top_connected_free_space_ratio
            ),
            'sealed_cavity_ratio': analysis.sealed_cavity_ratio,
            'sealed_cavity_count': self._unsigned(
                analysis.sealed_cavity_count,
                self.config.cavity_count_scale,
            ),
        })
        return values

    def _chain_motif_features(self, motif, analysis):
        """把局部连锁结构压缩成不含身份编号的 motif 节点特征。"""

        future_queue_indices = tuple(
            queue_index
            for queue_index in motif.compatible_queue_indices
            if queue_index > 0
        )
        future_weight = max(
            (
                analysis.queue_decay ** queue_index
                for queue_index in future_queue_indices
            ),
            default=0.0,
        )
        return {
            'level': self._level(motif.levels[0]),
            'motif_is_merge_pair': self._flag(
                motif.motif_type == 'merge_pair'
            ),
            'motif_is_level_ladder': self._flag(
                motif.motif_type == 'level_ladder'
            ),
            'motif_base_level': self._level(motif.levels[0]),
            'motif_member_count': self._unsigned(
                len(motif.fruit_ids),
                self.config.motif_member_count_scale,
            ),
            'motif_depth': self._unsigned(
                motif.depth,
                MAX_FRUIT_LEVEL,
            ),
            'motif_readiness': motif.readiness,
            'motif_trigger_action_fraction': (
                motif.trigger_action_mask.bit_count()
                / max(1, analysis.action_count)
            ),
            'motif_current_queue_compatible': self._flag(
                0 in motif.compatible_queue_indices
            ),
            'motif_future_queue_compatible': self._flag(
                bool(future_queue_indices)
            ),
            'motif_future_queue_weight': future_weight,
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
            edge_refs,
            structure=None):
        """连接场上水果之间的空间/合成关系。"""

        for source_offset, source_fruit in enumerate(fruits):
            for target_offset, target_fruit in enumerate(fruits):
                if source_offset == target_offset:
                    continue

                features = self._fruit_pair_edge_features(
                    source_fruit,
                    target_fruit,
                    geometry,
                    structure=structure,
                )
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
            edge_refs,
            structure=None):
        """连接候选动作和场上水果，表达每个落点可能影响哪些水果。"""

        for action_offset, action in enumerate(actions):
            for fruit_offset, fruit in enumerate(fruits):
                action_to_fruit = self._action_board_edge_features(
                    action,
                    fruit,
                    geometry,
                    reverse=False,
                    action_offset=action_offset,
                    structure=structure,
                )
                fruit_to_action = self._action_board_edge_features(
                    action,
                    fruit,
                    geometry,
                    reverse=True,
                    action_offset=action_offset,
                    structure=structure,
                )

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

    def _connect_chain_motifs(
            self,
            structure,
            fruits,
            board_node_indices,
            queue_node_indices,
            action_node_indices,
            motif_node_indices,
            edge_index,
            edge_features,
            edge_refs):
        """连接 motif 与水果、队列和动作，并显式表达角色与触发时机。

        这里不执行一次额外物理推演。``preserve`` 和 ``break_risk`` 只是由当前
        q0 第一阻挡者得到的保守静态提示：避开 motif 成员且投放列安全可视为保护候选，
        非触发动作直接把 q0 落到 motif 成员上则记为破坏风险。真正动作后是否保住或
        破坏结构，仍应由下一状态的辅助标签或因果归因确认。
        """

        analysis = structure.analysis
        board_node_by_id = {
            fruit.fruit_id: board_node_indices[fruit_offset]
            for fruit_offset, fruit in enumerate(fruits)
        }
        q0_lane = analysis.queue_lane_analyses[0]
        queue_denominator = max(1, len(queue_node_indices) - 1)

        for motif_offset, motif in enumerate(analysis.chain_motifs):
            motif_node = motif_node_indices[motif_offset]
            base_level = motif.levels[0]
            stage_denominator = max(1, motif.depth - 1)

            # 一条 ladder 的前两个同级水果属于第 0 阶材料，后续更高等级水果
            # 属于 chain target。用等级差定义 stage，避免编码任意成员顺序。
            for fruit_id, level in zip(motif.fruit_ids, motif.levels):
                fruit_node = board_node_by_id[fruit_id]
                is_pair_member = level == base_level
                stage = self._unit(
                    (level - base_level) / stage_denominator
                )
                features = {
                    'motif_role_pair_member': self._flag(is_pair_member),
                    'motif_role_chain_target': self._flag(
                        not is_pair_member
                    ),
                    'motif_stage': stage,
                    'motif_preserve': 1.0,
                    'motif_relation_strength': motif.readiness,
                }
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    fruit_node,
                    motif_node,
                    'board_fruit_to_chain_motif',
                    features,
                )
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    motif_node,
                    fruit_node,
                    'board_fruit_to_chain_motif',
                    features,
                )

            # 只连接真正兼容的可见队列槽。q0 表示现在可触发，q1-q3 表示
            # “值得暂时保护到未来”的有限前瞻；不可见队列绝不参与构图。
            for queue_index in motif.compatible_queue_indices:
                queue_node = queue_node_indices[queue_index]
                queue_weight = analysis.queue_decay ** queue_index
                features = {
                    'queue_index': self._queue_index(
                        queue_index,
                        len(queue_node_indices),
                    ),
                    'motif_stage': self._unit(
                        queue_index / queue_denominator
                    ),
                    'motif_trigger_now': self._flag(queue_index == 0),
                    'motif_future_queue': self._flag(queue_index > 0),
                    'motif_preserve': self._flag(queue_index > 0),
                    'motif_relation_strength': (
                        motif.readiness * queue_weight
                    ),
                }
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    queue_node,
                    motif_node,
                    'queue_fruit_to_chain_motif',
                    features,
                )
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    motif_node,
                    queue_node,
                    'queue_fruit_to_chain_motif',
                    features,
                )

            current_queue_compatible = (
                0 in motif.compatible_queue_indices
            )
            future_queue_indices = tuple(
                queue_index
                for queue_index in motif.compatible_queue_indices
                if queue_index > 0
            )
            earliest_compatible = min(
                motif.compatible_queue_indices,
                default=len(queue_node_indices) - 1,
            )
            motif_member_ids = frozenset(motif.fruit_ids)

            # 每个动作都连到 motif，trigger mask 的第 a 位只写到第 a 个动作边。
            # 这种逐 action 展开保证网络不会把 21 位 mask 的二进制数值大小误当成
            # “更强触发”，也让未触发动作显式看到保护/破坏风险。
            for action_offset, action_node in enumerate(action_node_indices):
                analysis_offset = structure.analysis_offsets[
                    action_offset
                ]
                trigger_now = (
                    current_queue_compatible
                    and self._mask_contains(
                        motif.trigger_action_mask,
                        analysis_offset,
                    )
                )
                blocker_ids = frozenset(
                    q0_lane.blocker_ids_by_action[analysis_offset]
                )
                member_blocker_count = len(
                    blocker_ids & motif_member_ids
                )
                break_risk = (
                    0.0
                    if trigger_now
                    else self._unit(
                        member_blocker_count
                        / max(1, len(motif_member_ids))
                    )
                )
                is_safe = self._mask_contains(
                    q0_lane.safe_action_mask,
                    analysis_offset,
                )
                preserve = (
                    not trigger_now
                    and is_safe
                    and member_blocker_count == 0
                )
                features = {
                    'motif_stage': self._unit(
                        earliest_compatible / queue_denominator
                    ),
                    'motif_trigger_now': self._flag(trigger_now),
                    'motif_future_queue': self._flag(
                        bool(future_queue_indices)
                    ),
                    'motif_preserve': self._flag(preserve),
                    'motif_break_risk': break_risk,
                    'motif_relation_strength': motif.readiness,
                }
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    action_node,
                    motif_node,
                    'action_to_chain_motif',
                    features,
                )
                self._add_edge(
                    edge_index,
                    edge_features,
                    edge_refs,
                    motif_node,
                    action_node,
                    'action_to_chain_motif',
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

    def _fruit_pair_edge_features(
            self,
            source,
            target,
            geometry,
            structure=None):
        """计算两个场上水果之间的有向边特征。"""

        dx = target.x - source.x
        dy = target.y - source.y
        distance = math.hypot(dx, dy)
        radius_sum = (
            self._fruit_physics_radius(source)
            + self._fruit_physics_radius(target)
        )
        level_diff = target.level - source.level
        relative_vx = target.vx - source.vx
        relative_vy = target.vy - source.vy

        values = {
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
        if structure is None:
            return values

        source_id = source.fruit_id
        target_id = target.fruit_id
        contact = (
            distance
            <= radius_sum + self.config.contact_gap_tolerance
        )
        supports_confidence = structure.support_relations.get(
            (source_id, target_id, 'supports'),
            0.0,
        )
        caps_confidence = structure.support_relations.get(
            (source_id, target_id, 'caps'),
            0.0,
        )
        bridges_confidence = structure.support_relations.get(
            (source_id, target_id, 'bridges'),
            0.0,
        )
        reachable_partner = (
            target_id
            in structure.fruit_by_id[source_id].reachable_partner_ids
        )
        critical_blocker = (
            (source_id, target_id)
            in structure.critical_edges
        )
        inversion_blocker = (
            (source_id, target_id)
            in structure.inversion_edges
        )
        relation_confidences = (
            self._flag(contact),
            supports_confidence,
            caps_confidence,
            bridges_confidence,
            self._flag(reachable_partner),
            self._flag(critical_blocker),
            self._flag(inversion_blocker),
        )
        values.update({
            'is_contact_relation': self._flag(contact),
            'is_support_relation': self._flag(
                supports_confidence > 0.0
            ),
            'is_caps_relation': self._flag(caps_confidence > 0.0),
            'is_bridges_relation': self._flag(
                bridges_confidence > 0.0
            ),
            'is_reachable_partner_relation': self._flag(
                reachable_partner
            ),
            'is_critical_blocker_relation': self._flag(
                critical_blocker
            ),
            'is_inversion_blocker_relation': self._flag(
                inversion_blocker
            ),
            'structure_confidence': max(relation_confidences),
        })
        return values

    def _action_board_edge_features(
            self,
            action,
            fruit,
            geometry,
            reverse=False,
            action_offset=None,
            structure=None):
        """计算候选动作和场上水果之间的边特征。"""

        dx = fruit.x - action.drop_x        # 带方向的水平差：水果在落点右侧为正，左侧为负。
        dy = fruit.y - geometry.spawn_y     # 带方向的垂直差：水果在生成线下方通常为正。
        if reverse:                         # 如果是反向边，则交换方向。
            dx = -dx
            dy = -dy

        horizontal_distance = abs(action.drop_x - fruit.x)  # 不带方向的水平距离，判断水果是否接近投放路径。
        vertical_distance = abs(fruit.y - geometry.spawn_y)  # 不带方向的垂直距离，判断水果位于投放起点下方多远。
        # 两个水果横向接触所需的距离阈值必须使用真实碰撞半径。
        radius_sum = (
            self._action_physics_radius(action)
            + self._fruit_physics_radius(fruit)
        )
        path_overlap_margin = radius_sum - horizontal_distance
        level_diff = action.current_level - fruit.level      # 当前投放水果和场上水果的等级差。

        values = {
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
        if structure is None:
            return values
        if action_offset is None:
            raise ValueError(
                'action_offset is required when state_analysis is present'
            )

        analyzed_fruit = structure.fruit_by_id[fruit.fruit_id]
        q0_lane = structure.analysis.queue_lane_analyses[0]
        analysis_offset = structure.analysis_offsets[action_offset]
        values.update({
            'action_reaches_fruit': self._flag(
                self._mask_contains(
                    analyzed_fruit.reachable_action_mask,
                    analysis_offset,
                )
            ),
            'is_q0_first_blocker': self._flag(
                fruit.fruit_id
                in q0_lane.blocker_ids_by_action[analysis_offset]
            ),
        })
        return values

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
            'is_near_boundary': self._flag(
                distance <= self._fruit_physics_radius(fruit)
            ),
        }

    def _fruit_physics_radius(self, fruit):
        """读取场上水果的碰撞半径，并兼容旧状态对象。"""

        physics_radius = getattr(fruit, 'physics_radius', None)
        return float(fruit.radius if physics_radius is None else physics_radius)

    def _action_physics_radius(self, action):
        """读取待投放水果的碰撞半径，并兼容旧动作对象。"""

        physics_radius = getattr(action, 'current_physics_radius', None)
        return float(
            action.current_radius
            if physics_radius is None
            else physics_radius
        )

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

    @staticmethod
    def _mask_contains(mask, action_offset):
        """按 action offset 解码 21 位分析 mask 中的一位。"""

        return bool(mask & (1 << action_offset))

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
