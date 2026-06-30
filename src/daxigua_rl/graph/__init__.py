"""GNN 图构建模块的对外入口。

外部训练代码优先从这里导入 GraphBuilder 和图数据结构，
避免直接依赖 graph 子模块里的具体文件布局。
"""

from .ablation import (
    ABLATION_PRESETS,
    EDGE_FEATURE_GROUPS,
    NODE_FEATURE_GROUPS,
    FeatureAblationConfig,
    FeatureMask,
    GraphAblator,
    get_ablation_preset,
)
from .builder import GraphBuilder, GraphBuilderConfig
from .schema import (
    BOUNDARY_TYPES,
    EDGE_FEATURE_NAMES,
    EDGE_TYPES,
    NODE_FEATURE_NAMES,
    NODE_TYPES,
    GraphData,
    GraphEdgeRef,
    GraphNodeRef,
)


# 明确声明对外暴露的图构建相关类型，方便后续训练代码稳定引用。
__all__ = [
    'BOUNDARY_TYPES',
    'ABLATION_PRESETS',
    'EDGE_FEATURE_NAMES',
    'EDGE_FEATURE_GROUPS',
    'EDGE_TYPES',
    'FeatureAblationConfig',
    'FeatureMask',
    'GraphAblator',
    'GraphBuilder',
    'GraphBuilderConfig',
    'GraphData',
    'GraphEdgeRef',
    'GraphNodeRef',
    'get_ablation_preset',
    'NODE_FEATURE_NAMES',
    'NODE_FEATURE_GROUPS',
    'NODE_TYPES',
]
