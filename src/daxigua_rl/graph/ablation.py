"""GNN 图特征消融工具。

消融层只负责把已经构建好的 `GraphData` 中部分特征置零。
它不修改游戏状态，也不改变图的节点数量、边数量或特征维度。
这样不同实验之间可以保持完全一致的模型输入形状，便于对比。
"""

from dataclasses import dataclass, field

from .schema import EDGE_TYPES, NODE_TYPES, GraphData


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
            'approaching_speed',
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
            'distance',
            'horizontal_distance',
            'vertical_distance',
            'radius_sum',
            'overlap_margin',
            'drop_x_minus_fruit_x',
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
            'queue_index',
        ), target_type='queue_fruit_to_queue_fruit'),                                    # 屏蔽待投放队列内部的顺序边信息。
    ),
    'queue_board_match': (
        FeatureMask((
            'queue_index',
            'level_diff',
            'abs_level_diff',
            'same_level',
            'board_fruit_x',
            'board_fruit_y',
        ), target_type='queue_fruit_to_board_fruit'),                                    # 屏蔽未来水果与场上水果的匹配关系。
    ),
    'action_queue_match': (
        FeatureMask((
            'action_index',
            'queue_index',
            'is_current_queue_fruit',
            'level_diff',
            'abs_level_diff',
            'same_level',
        ), target_type='action_to_queue_fruit'),                                         # 屏蔽候选动作与待投放队列的规划关系。
    ),
    'boundary_distance': (
        FeatureMask((
            'distance_to_boundary',
            'is_near_boundary',
        ), target_type='board_fruit_to_boundary'),                                       # 屏蔽场上水果与边界之间的距离/风险关系。
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

        原始 `graph` 不会被修改；新图的维度、节点编号、边编号保持不变，
        只有被遮罩的特征列会被置为 0。
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

        return GraphData(
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
