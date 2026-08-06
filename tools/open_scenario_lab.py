"""生成离线场景页面，或启动真实物理与Reward V2场景实验室。"""

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
        help='生成或启动后使用系统默认浏览器打开。',
    )
    parser.add_argument(
        '--serve',
        action='store_true',
        help='启动真实物理与Reward V2后端；不指定时只生成离线前端。',
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='服务监听地址，默认 127.0.0.1。',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8769,
        help='服务端口，默认 8769；设为 0 可自动选择空闲端口。',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='评估设备，例如 cpu、cuda 或 cuda:0。',
    )
    parser.add_argument(
        '--reward-scale',
        type=float,
        default=1.0,
        help='Reward V2 最终缩放，默认 1.0。',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.serve:
        from daxigua.simulator.scenario_lab_server import ScenarioLabServer
        from daxigua.simulator.scenario_lab_service import ScenarioLabEvaluator

        evaluator = ScenarioLabEvaluator(
            device=args.device,
            reward_scale=args.reward_scale,
        )
        server = ScenarioLabServer(
            evaluator,
            host=args.host,
            port=args.port,
        )
        print(
            f'场景实验室服务：{server.url} '
            f'（{evaluator.device}，Reward V2 × {args.reward_scale:g}）',
            flush=True,
        )
        if args.open:
            webbrowser.open(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('正在关闭场景实验室服务……', flush=True)
        finally:
            server.httpd.server_close()
        return

    output_path = write_scenario_lab_html(Path(args.output).resolve())
    print(f'场景实验室离线前端：{output_path}', flush=True)
    if args.open:
        webbrowser.open(output_path.as_uri())


if __name__ == '__main__':
    main()
