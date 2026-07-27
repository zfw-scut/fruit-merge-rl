"""GNN 图数据结构。

本模块只定义“图长什么样”，不负责从游戏状态中计算图。
这样后续无论使用 PyTorch Geometric、DGL，还是手写 message passing，
都可以先复用同一份中间图数据。
"""

from dataclasses import dataclass


NODE_TYPES = (
    'board_fruit',
    'queue_fruit',
    'action',
    'chain_motif',
    'global',
    'boundary',
)

BOUNDARY_TYPES = (
    'left_wall',
    'right_wall',
    'floor',
    'danger_line',
)

EDGE_TYPES = (
    'board_fruit_to_board_fruit',
    'action_to_board_fruit',
    'queue_fruit_to_queue_fruit',
    'queue_fruit_to_board_fruit',
    'action_to_queue_fruit',
    'board_fruit_to_boundary',
    'board_fruit_to_chain_motif',
    'queue_fruit_to_chain_motif',
    'action_to_chain_motif',
    'global_to_node',
)

NODE_FEATURE_NAMES = (
    # 节点类型 one-hot。先用统一特征矩阵，后续再按需要适配真正的异构图库。
    'is_board_fruit_node',
    'is_queue_fruit_node',
    'is_action_node',
    'is_chain_motif_node',
    'is_global_node',
    'is_boundary_node',
    # 通用空间和运动状态。没有真实空间位置的节点会保持 0。
    'x',
    'y',
    'vx',
    'vy',
    # 水果语义。场上水果、队列水果、动作当前水果都会用到这些字段。
    'level',
    'radius',
    'stable',
    'distance_to_left_wall',
    'distance_to_right_wall',
    'distance_to_floor',
    'distance_to_danger_line',
    # 队列语义。
    'queue_index',
    'is_current_queue_fruit',
    # 动作语义。
    'action_index',
    # StateAnalysis 给水果补充的结构语义。这里不编码 fruit_id、region_id 等
    # 任意身份数字，只保留可跨 episode 泛化的比例、计数和布尔关系。
    'reachable_action_fraction',
    'top_visible_ratio',
    'partner_reachable',
    'partner_count',
    'reachable_partner_count',
    'support_parent_count',
    'supported_child_count',
    'burial_depth',
    'inversion_count',
    'critical_blocker_count',
    'connected_to_top_space',
    # 当前 q0 在每个动作列上的静态落点摘要。mask 必须在 GraphBuilder 中按
    # action offset 解码，不能把整个整数 mask 当成连续数值喂给模型。
    'q0_landing_depth',
    'q0_is_safe',
    'q0_blocker_count',
    # 全局语义。
    'max_height',
    'fruit_count',
    'max_level',
    'empty_space_ratio',
    'has_state_analysis',
    'analysis_valid',
    'analysis_degraded',
    'top_connected_capacity',
    'recoverability',
    'chain_readiness',
    'top_connected_free_space_ratio',
    'sealed_cavity_ratio',
    'sealed_cavity_count',
    # 连锁 motif 虚拟节点。motif 只概括当前稳定边界已经识别出的局部结构，
    # 不携带未来动作结果、历史奖励或任意 motif 身份编号。
    'motif_is_merge_pair',
    'motif_is_level_ladder',
    'motif_base_level',
    'motif_member_count',
    'motif_depth',
    'motif_readiness',
    'motif_trigger_action_fraction',
    'motif_current_queue_compatible',
    'motif_future_queue_compatible',
    'motif_future_queue_weight',
    # 边界语义。
    'is_left_wall',
    'is_right_wall',
    'is_floor',
    'is_danger_line',
    'boundary_position',
)

EDGE_FEATURE_NAMES = (
    # 边类型 one-hot。第一版保持单一 edge_features 矩阵，方便先跑通训练闭环。
    'is_board_fruit_pair_edge',
    'is_action_board_fruit_edge',
    'is_queue_fruit_order_edge',
    'is_queue_board_fruit_edge',
    'is_action_queue_fruit_edge',
    'is_board_boundary_edge',
    'is_board_chain_motif_edge',
    'is_queue_chain_motif_edge',
    'is_action_chain_motif_edge',
    'is_global_edge',
    # 空间关系。
    'dx',
    'dy',
    'distance',
    'horizontal_distance',
    'vertical_distance',
    'radius_sum',
    'overlap_margin',
    # 等级和合成关系。
    'level_diff',
    'abs_level_diff',
    'same_level',
    # 运动关系。
    'relative_vx',
    'relative_vy',
    # 动作和投放路径关系。
    'path_overlap_margin',
    'is_under_drop_path',
    # 队列关系。
    'order_gap',
    'is_next_queue_fruit',
    'queue_index',
    # 边界关系。
    'distance_to_boundary',
    'is_near_boundary',
    # 当前稳定边界中的显式结构关系。支撑、盖压和桥接方向遵循
    # supporter -> supported fruit；critical/inversion 则是 blocker -> victim。
    'is_contact_relation',
    'is_support_relation',
    'is_caps_relation',
    'is_bridges_relation',
    'is_reachable_partner_relation',
    'is_critical_blocker_relation',
    'is_inversion_blocker_relation',
    'structure_confidence',
    # action/queue/fruit 与 motif 的角色关系。role 和 stage 是结构语义，
    # trigger_now 是逐 action 解码后的标志；future_queue 只来自当前可见 q1-q3。
    'motif_role_pair_member',
    'motif_role_chain_target',
    'motif_stage',
    'motif_trigger_now',
    'motif_future_queue',
    'motif_preserve',
    'motif_break_risk',
    'motif_relation_strength',
    # 与当前 action offset 对齐的水果可达性和 q0 第一阻挡关系。
    'action_reaches_fruit',
    'is_q0_first_blocker',
)


@dataclass(frozen=True)
class GraphNodeRef:
    """图节点和原始游戏对象之间的对应关系。

    模型训练只需要特征矩阵，但调试时经常需要知道“第 17 个节点对应谁”，
    所以这里保留一份轻量元数据。
    """

    node_type: str
    source_index: int | None = None
    source_id: int | None = None
    label: str = ''


@dataclass(frozen=True)
class GraphEdgeRef:
    """图边的调试元数据。"""

    edge_type: str
    source_node: int
    target_node: int


@dataclass(frozen=True)
class GraphData:
    """训练前的框架无关图数据。

    字段说明：
    - `node_features`: shape 近似为 `[num_nodes, node_feature_dim]`
    - `edge_index`: 每个元素是 `(source_node, target_node)`
    - `edge_features`: shape 近似为 `[num_edges, edge_feature_dim]`
    - `action_node_indices`: 哪些节点是动作节点，用于最终读出 Q 值
    - `action_indices`: 和 `action_node_indices` 对齐的原始动作编号
    """

    node_features: tuple
    edge_index: tuple
    edge_features: tuple
    node_refs: tuple
    edge_refs: tuple
    action_node_indices: tuple
    action_indices: tuple
    node_feature_names: tuple = NODE_FEATURE_NAMES
    edge_feature_names: tuple = EDGE_FEATURE_NAMES

    @property
    def num_nodes(self):
        """返回图中的节点数量。"""

        return len(self.node_features)

    @property
    def num_edges(self):
        """返回图中的有向边数量。"""

        return len(self.edge_index)

    @property
    def node_feature_dim(self):
        """返回单个节点特征维度。"""

        if not self.node_features:
            return 0
        return len(self.node_features[0])

    @property
    def edge_feature_dim(self):
        """返回单条边特征维度。"""

        if not self.edge_features:
            return 0
        return len(self.edge_features[0])
