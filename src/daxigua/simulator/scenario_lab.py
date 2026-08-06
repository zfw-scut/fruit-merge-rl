"""生成可离线使用的自定义场景实验室前端。"""

from __future__ import annotations

import json
from pathlib import Path

from daxigua.core.rules import (
    FRUIT_NAMES,
    FRUIT_RADII,
    MAX_FRUIT_LEVEL,
    MERGE_SCORES,
    MIN_FRUIT_LEVEL,
)

from .replay import load_fruit_texture_data_urls
from .scenario_lab_web import render_scenario_lab_document


def _safe_json(value):
    return (
        json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        .replace('</', '<\\/')
        .replace('\u2028', '\\u2028')
        .replace('\u2029', '\\u2029')
    )


def fruit_specs():
    """返回前端绘制和几何提示所需的稳定水果规则。"""

    return [
        {
            'level': level,
            'name': FRUIT_NAMES[level],
            'radius': FRUIT_RADII[level],
            'merge_score': MERGE_SCORES[level],
        }
        for level in range(MIN_FRUIT_LEVEL, MAX_FRUIT_LEVEL + 1)
    ]


def write_scenario_lab_html(
        path, *, title='合成大西瓜 · 场景实验室', texture_dir=None):
    """写出自包含 HTML；当前文件不启动或依赖物理后端。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_scenario_lab_document(
        title=title,
        fruit_specs_json=_safe_json(fruit_specs()),
        textures_json=_safe_json(
            load_fruit_texture_data_urls(texture_dir)
        ),
    )
    output_path.write_text(html, encoding='utf-8')
    return output_path
