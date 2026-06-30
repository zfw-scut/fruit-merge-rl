"""GNN 图构建模块的对外入口。

外部训练代码优先从这里导入 GraphBuilder 和图数据结构，
避免直接依赖 graph 子模块里的具体文件布局。
"""

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
    'EDGE_FEATURE_NAMES',
    'EDGE_TYPES',
    'GraphBuilder',
    'GraphBuilderConfig',
    'GraphData',
    'GraphEdgeRef',
    'GraphNodeRef',
    'NODE_FEATURE_NAMES',
    'NODE_TYPES',
]
