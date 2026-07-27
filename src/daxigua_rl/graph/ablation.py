"""GNN 图特征与虚拟结构消融工具。

普通消融只把已经构建好的 ``GraphData`` 中部分特征置零，保持拓扑不变。
若某类虚拟节点的存在本身就会泄漏被消融信息（例如 ``chain_motif``），配置还可
显式删除该类节点及其关联边；特征维度仍保持一致。
"""

from dataclasses import dataclass, field

from .schema import EDGE_TYPES, NODE_TYPES, GraphData, GraphEdgeRef


@dataclass(frozen=True)
class FeatureMask:
    """一条特征遮罩规则。

    `target_type` 为空时，表示遮罩所有节点/边上的这些特征；
    `target_type` 不为空时，只遮罩指定节点类型或边类型上的这些特征。
    """

    feature_names: tuple
    target_type: str | None = None


@dataclass(frozen=True)
class FeatureAblationConfig:
    """一次消融实验的特征遮罩配置。"""

    disabled_node_features: tuple = field(default_factory=tuple)          # 对所有节点置零的节点特征名。
    disabled_edge_features: tuple = field(default_factory=tuple)          # 对所有边置零的边特征名。
    disabled_node_feature_groups: tuple = field(default_factory=tuple)    # 预定义节点特征组名称。
    disabled_edge_feature_groups: tuple = field(default_factory=tuple)    # 预定义边特征组名称。
    disabled_node_masks: tuple = field(default_factory=tuple)             # 更精确的节点遮罩规则。
    disabled_edge_masks: tuple = field(default_factory=tuple)             # 更精确的边遮罩规则。
    dropped_node_types: tuple = field(default_factory=tuple)              # 删除会以拓扑本身泄漏信息的虚拟节点类型。


# 节点特征组。每个组可以只作用于某一种节点，避免统一特征矩阵里的同名字段被误伤。
NODE_FEATURE_GROUPS = {
    'board_motion': (
        FeatureMask(('vx', 'vy', 'stable'), target_type='board_fruit'),                 # 屏蔽场上水果的运动/稳定状态。
    ),
    'board_boundary_distance': (
        FeatureMask((
            'distance_to_left_wall',
            'distance_to_right_wall',
            'distance_to_floor',
            'distance_to_danger_line',
        ), target_type='board_fruit'),                                                   # 屏蔽场上水果到墙体、地板、死亡线的距离。
    ),
    'queue_order': (
        FeatureMask(('queue_index', 'is_current_queue_fruit'), target_type='queue_fruit'),  # 屏蔽待投放队列的顺序信息。
    ),
    'action_identity': (
        FeatureMask(('action_index',), target_type='action'),                           # 屏蔽候选动作在离散动作列表中的编号。
    ),
    'global_summary': (
        FeatureMask((
            'max_height',
            'fruit_count',
            'max_level',
            'empty_space_ratio',
        ), target_type='global'),                                                        # 屏蔽全局局面摘要。
    ),
    'fruit_structure': (
        FeatureMask((
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
        ), target_type='board_fruit'),                                                   # 屏蔽 StateAnalysis 提供的单果结构摘要。
    ),
    'action_lane_structure': (
        FeatureMask((
            'q0_landing_depth',
            'q0_is_safe',
            'q0_blocker_count',
        ), target_type='action'),                                                        # 屏蔽 q0 与动作列对齐的落点/安全摘要。
    ),
    'global_structure': (
        FeatureMask((
            'has_state_analysis',
            'analysis_valid',
            'analysis_degraded',
            'top_connected_capacity',
            'recoverability',
            'chain_readiness',
            'top_connected_free_space_ratio',
            'sealed_cavity_ratio',
            'sealed_cavity_count',
        ), target_type='global'),                                                        # 屏蔽 C/R/K 与自由空间拓扑摘要。
    ),
    'chain_motif': (
        FeatureMask((
            'is_chain_motif_node',
            'level',
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
        ), target_type='chain_motif'),                                                   # 保留 motif 节点但屏蔽其结构内容，便于固定图形状做消融。
    ),
}


# 边特征组。边类型约束用于避免把同名字段在不同语义关系中一起置零。
EDGE_FEATURE_GROUPS = {
    'fruit_pair_spatial': (
        FeatureMask((
            'dx',
            'dy',
            'distance',
            'horizontal_distance',
            'vertical_distance',
            'radius_sum',
            'overlap_margin',
        ), target_type='board_fruit_to_board_fruit'),                                    # 屏蔽场上水果之间的空间关系。
    ),
    'fruit_pair_motion': (
        FeatureMask((
            'relative_vx',
            'relative_vy',
        ), target_type='board_fruit_to_board_fruit'),                                    # 屏蔽场上水果之间的相对运动关系。
    ),
    'fruit_pair_level': (
        FeatureMask((
            'level_diff',
            'abs_level_diff',
            'same_level',
        ), target_type='board_fruit_to_board_fruit'),                                    # 屏蔽场上水果之间的等级/同级关系。
    ),
    'action_board_spatial': (
        FeatureMask((
            'dx',
            'dy',
            'horizontal_distance',
            'vertical_distance',
            'radius_sum',
            'path_overlap_margin',
            'is_under_drop_path',
        ), target_type='action_to_board_fruit'),                                         # 屏蔽候选动作与场上水果的空间/投放路径关系。
    ),
    'action_board_level': (
        FeatureMask((
            'level_diff',
            'abs_level_diff',
            'same_level',
        ), target_type='action_to_board_fruit'),                                         # 屏蔽候选动作与场上水果的等级匹配关系。
    ),
    'queue_order': (
        FeatureMask((
            'order_gap',
            'is_next_queue_fruit',
        ), target_type='queue_fruit_to_queue_fruit'),                                    # 屏蔽待投放队列内部的顺序边信息。
    ),
    'queue_board_match': (
        FeatureMask((
            'queue_index',
            'level_diff',
            'abs_level_diff',
            'same_level',
        ), target_type='queue_fruit_to_board_fruit'),                                    # 屏蔽未来水果与场上水果的匹配关系。
    ),
    'action_queue_match': (
        FeatureMask((
            'queue_index',
        ), target_type='action_to_queue_fruit'),                                         # 屏蔽候选动作与待投放队列的顺序连接信息。
    ),
    'boundary_distance': (
        FeatureMask((
            'distance_to_boundary',
            'is_near_boundary',
        ), target_type='board_fruit_to_boundary'),                                       # 屏蔽场上水果与边界之间的距离/风险关系。
    ),
    'fruit_structure': (
        FeatureMask((
            'is_contact_relation',
            'is_support_relation',
            'is_caps_relation',
            'is_bridges_relation',
            'is_reachable_partner_relation',
            'is_critical_blocker_relation',
            'is_inversion_blocker_relation',
            'structure_confidence',
        ), target_type='board_fruit_to_board_fruit'),                                    # 屏蔽水果之间的显式结构关系。
        FeatureMask((
            'action_reaches_fruit',
            'is_q0_first_blocker',
        ), target_type='action_to_board_fruit'),                                         # 屏蔽按 action offset 对齐的可达/第一阻挡关系。
    ),
    'chain_motif': (
        FeatureMask((
            'is_board_chain_motif_edge',
            'motif_role_pair_member',
            'motif_role_chain_target',
            'motif_stage',
            'motif_trigger_now',
            'motif_future_queue',
            'motif_preserve',
            'motif_break_risk',
            'motif_relation_strength',
        ), target_type='board_fruit_to_chain_motif'),
        FeatureMask((
            'is_queue_chain_motif_edge',
            'queue_index',
            'motif_role_pair_member',
            'motif_role_chain_target',
            'motif_stage',
            'motif_trigger_now',
            'motif_future_queue',
            'motif_preserve',
            'motif_break_risk',
            'motif_relation_strength',
        ), target_type='queue_fruit_to_chain_motif'),
        FeatureMask((
            'is_action_chain_motif_edge',
            'motif_role_pair_member',
            'motif_role_chain_target',
            'motif_stage',
            'motif_trigger_now',
            'motif_future_queue',
            'motif_preserve',
            'motif_break_risk',
            'motif_relation_strength',
        ), target_type='action_to_chain_motif'),                                         # 屏蔽 motif 的成员角色、时序和动作风险关系。
    ),
}


ABLATION_PRESETS = {
    'full': FeatureAblationConfig(),                                                     # 不做任何消融，使用完整图。
    'no_board_motion': FeatureAblationConfig(
        disabled_node_feature_groups=('board_motion',),
        disabled_edge_feature_groups=('fruit_pair_motion',),
    ),                                                                                    # 去掉速度、稳定状态和相对运动信息。
    'no_global_summary': FeatureAblationConfig(
        disabled_node_feature_groups=('global_summary',),
    ),                                                                                    # 去掉全局局面摘要，只依赖局部节点和边。
    'no_queue_planning': FeatureAblationConfig(
        disabled_node_feature_groups=('queue_order',),
        disabled_edge_feature_groups=('queue_order', 'queue_board_match', 'action_queue_match'),
    ),                                                                                    # 弱化待投放序列带来的未来规划信息。
    'no_action_board_relation': FeatureAblationConfig(
        disabled_edge_feature_groups=('action_board_spatial', 'action_board_level'),
    ),                                                                                    # 去掉动作和场上水果之间的直接关系，只保留动作自身特征。
    'no_structure_analysis': FeatureAblationConfig(
        disabled_node_feature_groups=(
            'fruit_structure',
            'action_lane_structure',
            'global_structure',
            'chain_motif',
        ),
        disabled_edge_feature_groups=(
            'fruit_structure',
            'chain_motif',
        ),
        # motif 数量、成员连接和动作连接的拓扑本身就是 StateAnalysis 输出，
        # 仅把特征置零仍会泄漏结构，因此完整消融必须删除这些虚拟节点及关联边。
        dropped_node_types=('chain_motif',),
    ),                                                                                    # 屏蔽普通结构特征，并移除会泄漏 motif 的虚拟拓扑。
}


class GraphAblator:
    """对 `GraphData` 应用特征消融配置。"""

    def __init__(self, config=None):
        self.config = config or FeatureAblationConfig()

    @classmethod
    def from_preset(cls, preset_name):
        """使用预定义消融方案创建 GraphAblator。"""

        return cls(get_ablation_preset(preset_name))

    def apply(self, graph):
        """返回消融后的新图。

        原始 ``graph`` 不会被修改。普通遮罩保持节点/边编号不变；若配置了
        ``dropped_node_types``，关联节点和边会被删除并规范重编号，但节点/边
        特征维度不变。
        """

        node_masks = self._collect_node_masks()
        edge_masks = self._collect_edge_masks()

        node_features = self._apply_masks(
            rows=graph.node_features,
            refs=graph.node_refs,
            feature_names=graph.node_feature_names,
            masks=node_masks,
            type_attr='node_type',
            known_types=NODE_TYPES,
            label='node',
        )
        edge_features = self._apply_masks(
            rows=graph.edge_features,
            refs=graph.edge_refs,
            feature_names=graph.edge_feature_names,
            masks=edge_masks,
            type_attr='edge_type',
            known_types=EDGE_TYPES,
            label='edge',
        )

        masked_graph = GraphData(
            node_features=node_features,
            edge_index=graph.edge_index,
            edge_features=edge_features,
            node_refs=graph.node_refs,
            edge_refs=graph.edge_refs,
            action_node_indices=graph.action_node_indices,
            action_indices=graph.action_indices,
            node_feature_names=graph.node_feature_names,
            edge_feature_names=graph.edge_feature_names,
        )
        return self._drop_configured_node_types(masked_graph)

    def _drop_configured_node_types(self, graph):
        """删除指定虚拟节点及关联边，避免拓扑成为消融侧信道。"""

        dropped_types = frozenset(self.config.dropped_node_types)
        if not dropped_types:
            return graph
        unknown_types = dropped_types.difference(NODE_TYPES)
        if unknown_types:
            raise KeyError(
                'unknown dropped node types: '
                + ', '.join(sorted(unknown_types))
            )
        protected_types = dropped_types.intersection(
            {'action', 'global'}
        )
        if protected_types:
            raise ValueError(
                'action/global nodes cannot be dropped from a policy graph: '
                + ', '.join(sorted(protected_types))
            )

        kept_old_indices = tuple(
            node_index
            for node_index, ref in enumerate(graph.node_refs)
            if ref.node_type not in dropped_types
        )
        old_to_new = {
            old_index: new_index
            for new_index, old_index in enumerate(kept_old_indices)
        }

        edge_index = []
        edge_features = []
        edge_refs = []
        for (source, target), features, ref in zip(
                graph.edge_index,
                graph.edge_features,
                graph.edge_refs):
            if source not in old_to_new or target not in old_to_new:
                continue
            new_source = old_to_new[source]
            new_target = old_to_new[target]
            edge_index.append((new_source, new_target))
            edge_features.append(features)
            edge_refs.append(GraphEdgeRef(
                edge_type=ref.edge_type,
                source_node=new_source,
                target_node=new_target,
            ))

        return GraphData(
            node_features=tuple(
                graph.node_features[index]
                for index in kept_old_indices
            ),
            edge_index=tuple(edge_index),
            edge_features=tuple(edge_features),
            node_refs=tuple(
                graph.node_refs[index]
                for index in kept_old_indices
            ),
            edge_refs=tuple(edge_refs),
            action_node_indices=tuple(
                old_to_new[index]
                for index in graph.action_node_indices
            ),
            action_indices=graph.action_indices,
            node_feature_names=graph.node_feature_names,
            edge_feature_names=graph.edge_feature_names,
        )

    def _collect_node_masks(self):
        """汇总本次消融需要应用的节点遮罩规则。"""

        masks = []
        if self.config.disabled_node_features:
            masks.append(FeatureMask(tuple(self.config.disabled_node_features)))
        masks.extend(self._expand_groups(self.config.disabled_node_feature_groups, NODE_FEATURE_GROUPS, 'node'))
        masks.extend(self.config.disabled_node_masks)
        return tuple(masks)

    def _collect_edge_masks(self):
        """汇总本次消融需要应用的边遮罩规则。"""

        masks = []
        if self.config.disabled_edge_features:
            masks.append(FeatureMask(tuple(self.config.disabled_edge_features)))
        masks.extend(self._expand_groups(self.config.disabled_edge_feature_groups, EDGE_FEATURE_GROUPS, 'edge'))
        masks.extend(self.config.disabled_edge_masks)
        return tuple(masks)

    def _expand_groups(self, group_names, group_map, label):
        """把特征组名称展开成具体遮罩规则。"""

        masks = []
        for group_name in group_names:
            if group_name not in group_map:
                valid_names = ', '.join(sorted(group_map))
                raise KeyError(f'unknown {label} feature group: {group_name}; valid groups: {valid_names}')
            masks.extend(group_map[group_name])
        return masks

    def _apply_masks(self, rows, refs, feature_names, masks, type_attr, known_types, label):
        """在特征矩阵上应用遮罩规则。"""

        feature_to_index = {name: index for index, name in enumerate(feature_names)}
        self._validate_masks(masks, feature_to_index, known_types, label)

        masked_rows = []
        for row, ref in zip(rows, refs):
            values = list(row)
            ref_type = getattr(ref, type_attr)
            for mask in masks:
                if mask.target_type is not None and mask.target_type != ref_type:
                    continue
                for feature_name in mask.feature_names:
                    values[feature_to_index[feature_name]] = 0.0
            masked_rows.append(tuple(values))
        return tuple(masked_rows)

    def _validate_masks(self, masks, feature_to_index, known_types, label):
        """提前校验遮罩规则，避免实验配置拼写错误后静默失效。"""

        for mask in masks:
            if mask.target_type is not None and mask.target_type not in known_types:
                valid_types = ', '.join(known_types)
                raise KeyError(f'unknown {label} type: {mask.target_type}; valid types: {valid_types}')

            for feature_name in mask.feature_names:
                if feature_name not in feature_to_index:
                    valid_features = ', '.join(feature_to_index)
                    raise KeyError(
                        f'unknown {label} feature: {feature_name}; valid features: {valid_features}'
                    )


def get_ablation_preset(preset_name):
    """返回一个预定义消融配置。"""

    if preset_name not in ABLATION_PRESETS:
        valid_names = ', '.join(sorted(ABLATION_PRESETS))
        raise KeyError(f'unknown ablation preset: {preset_name}; valid presets: {valid_names}')
    return ABLATION_PRESETS[preset_name]
