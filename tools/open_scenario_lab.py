"""生成并可选打开自定义场景实验室前端。"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from daxigua.simulator.scenario_lab import write_scenario_lab_html


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='生成鼠标交互式自定义场景实验室前端。',
    )
    parser.add_argument(
        '--output',
        default='recordings/scenario-lab/index.html',
        help='输出 HTML，默认 recordings/scenario-lab/index.html。',
    )
    parser.add_argument(
        '--open',
        action='store_true',
        help='生成后使用系统默认浏览器打开。',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_path = write_scenario_lab_html(Path(args.output).resolve())
    print(f'场景实验室前端：{output_path}', flush=True)
    if args.open:
        webbrowser.open(output_path.as_uri())


if __name__ == '__main__':
    main()
