"""场景实验室前后端共享的稳定水果规格。"""

from __future__ import annotations

from daxigua.core.rules import (
    FRUIT_NAMES,
    FRUIT_RADII,
    MAX_FRUIT_LEVEL,
    MERGE_SCORES,
    MIN_FRUIT_LEVEL,
    dropped_fruit_physics_radius,
    merged_fruit_physics_radius,
)


def fruit_specs():
    """返回前端绘制和几何提示所需的稳定水果规则。"""

    return [
        {
            'level': level,
            'name': FRUIT_NAMES[level],
            'radius': FRUIT_RADII[level],
            'dropped_physics_radius': dropped_fruit_physics_radius(level),
            'merged_physics_radius': merged_fruit_physics_radius(level),
            'merge_score': MERGE_SCORES[level],
        }
        for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1)
    ]


__all__ = ['fruit_specs']
