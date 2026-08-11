"""启动供 Xigua Atlas 门户使用的场景实验室 API。"""

from __future__ import annotations

import argparse


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='启动场景实验室物理、评估与模型推理 API。',
    )
    parser.add_argument(
        '--serve',
        action='store_true',
        help=argparse.SUPPRESS,
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
        help='Reward V2.1 最终缩放，默认 1.0。',
    )
    parser.add_argument(
        '--checkpoint',
        help='可选：接入场景实验室的 GNN-DQN checkpoint。',
    )
    parser.add_argument(
        '--model-device',
        default='auto',
        help='模型推理设备，默认 auto。',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from daxigua.simulator.scenario_lab_server import ScenarioLabServer
    from daxigua.simulator.scenario_lab_live import ScenarioLabLiveSession
    from daxigua.simulator.scenario_lab_service import ScenarioLabEvaluator

    evaluator = ScenarioLabEvaluator(
        device=args.device,
        reward_scale=args.reward_scale,
    )
    model_evaluator = None
    model_controller = None
    live_session = ScenarioLabLiveSession()
    if args.checkpoint:
        from daxigua.rl.scenario_model_controller import (
            ScenarioModelController,
        )
        from daxigua.rl.scenario_model_evaluator import (
            ScenarioModelEvaluator,
        )
        from daxigua.rl.viewer import load_viewer_model

        loaded = load_viewer_model(
            args.checkpoint, device=args.model_device
        )
        model_evaluator = ScenarioModelEvaluator(loaded)
        model_controller = ScenarioModelController(
            live_session, model_evaluator
        )
    server = ScenarioLabServer(
        evaluator,
        model_evaluator=model_evaluator,
        model_controller=model_controller,
        live_session=live_session,
        host=args.host,
        port=args.port,
    )
    print(
        f'场景实验室 API：{server.url} '
        f'（{evaluator.device}，Reward V2.1 × {args.reward_scale:g}）',
        flush=True,
    )
    if model_evaluator is not None:
        identity = model_evaluator.identity
        print(
            f"模型评估：{identity['checkpoint']} "
            f"（{identity['checkpoint_sha256']}，{identity['device']}）",
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('正在关闭场景实验室服务……', flush=True)
    finally:
        server.close()


if __name__ == '__main__':
    main()
