#!/usr/bin/env python3
"""临时 GNN rollout smoke test。

【临时代码说明】
本脚本用于验证当前最小 RL 链路是否已经闭合：

    DaxiguaEnv -> GraphBuilder -> GNNQNetwork -> 选择动作 -> DaxiguaEnv.step()

它不是正式训练入口，也不保存模型、不更新参数、不写 replay buffer。
等后续正式训练循环实现并验证完成后，本文件可以删除或改造成正式测试。
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch


# 允许用户直接从项目根目录运行：
#
#     conda run -n python-torch python tools/temporary_rollout_smoke_test.py
#
# 由于当前项目还没有 packaging/install 配置，这里临时把 `src/` 加入 import 路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from daxigua_rl import DaxiguaEnv, DaxiguaEnvConfig, GraphBuilder  # noqa: E402
from daxigua_rl.models import GNNQNetwork  # noqa: E402


def parse_args():
    """解析命令行参数。

    参数默认值尽量贴近当前环境接口：
    - 15 个候选动作；
    - 每一步最多推进 720 帧物理；
    - 连续稳定 15 帧后认为本次投放结束。
    """

    parser = argparse.ArgumentParser(
        description='临时验证 GNN-Q 模型是否能驱动 DaxiguaEnv 进行无训练 rollout。'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=30,
        help='最多执行多少次投放动作。',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='游戏随机种子和模型随机初始化种子。',
    )
    parser.add_argument(
        '--action-count',
        type=int,
        default=15,
        help='每个状态生成多少个离散候选投放动作。',
    )
    parser.add_argument(
        '--max-physics-frames',
        type=int,
        default=720,
        help='每次投放后最多推进多少帧物理模拟。',
    )
    parser.add_argument(
        '--stable-frames',
        type=int,
        default=15,
        help='连续多少帧稳定后认为本次投放结束。',
    )
    parser.add_argument(
        '--policy',
        choices=('argmax', 'random'),
        default='argmax',
        help='动作选择方式：argmax 使用未训练模型的最大 Q 值；random 随机选择动作。',
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='GNN 隐藏层维度。',
    )
    parser.add_argument(
        '--message-layers',
        type=int,
        default=3,
        help='GNN message passing 层数。',
    )
    parser.add_argument(
        '--device',
        default='cpu',
        help='模型运行设备，例如 cpu、cuda 或 cuda:0。',
    )
    parser.add_argument(
        '--fruit-queue',
        default=None,
        help='可选的固定初始水果队列，例如 "1,2,3,4"。不传则使用游戏随机队列。',
    )
    parser.add_argument(
        '--print-q-values',
        action='store_true',
        help='打印每一步所有候选动作的 Q 值，调试动作排序时使用。',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='不打印每一步详情，只打印开头、结尾和 profile 汇总。',
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='统计每一步闭环的耗时，用于粗略估算训练吞吐。',
    )
    return parser.parse_args()


def parse_fruit_queue(value):
    """把命令行中的 `"1,2,3,4"` 解析成整数 tuple。"""

    if value is None:
        return None

    parts = [part.strip() for part in value.split(',')]
    if not parts or any(part == '' for part in parts):
        raise ValueError('--fruit-queue must look like "1,2,3,4"')

    return tuple(int(part) for part in parts)


def resolve_device(device_name):
    """解析模型设备，并对常见 CUDA 配置错误给出更直接的报错。"""

    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA device, but torch.cuda.is_available() is False')
    return device


def choose_action_offset(q_values, policy, rng):
    """根据策略选择动作在候选动作列表中的位置。

    注意这里返回的是 action offset，也就是 `DaxiguaEnv.step(action_index)` 当前需要的
    候选列表下标，而不是物理横坐标，也不是模型内部节点编号。
    """

    action_count = int(q_values.shape[0])
    if action_count <= 0:
        raise ValueError('q_values must contain at least one action')

    if policy == 'random':
        return rng.randrange(action_count)

    # 未训练模型的 argmax 没有游戏能力，但可以验证“模型输出 -> 选择动作”这段链路。
    return int(torch.argmax(q_values).item())


def q_summary(q_values):
    """返回当前动作 Q 值的简短统计，方便观察输出是否正常。"""

    return {
        'min': float(q_values.min().item()),
        'mean': float(q_values.mean().item()),
        'max': float(q_values.max().item()),
    }


def format_queue(queue):
    """把水果队列格式化成紧凑字符串。"""

    return '[' + ','.join(str(level) for level in queue) + ']'


def summarize_times(values):
    """返回耗时列表的常用统计量。

    这里避免引入 numpy，临时脚本只用标准库即可完成粗略评估。
    """

    if not values:
        return {
            'mean': 0.0,
            'min': 0.0,
            'max': 0.0,
            'p50': 0.0,
            'p90': 0.0,
        }

    ordered = sorted(values)

    def percentile(percent):
        # 使用简单的 nearest-rank 近似百分位，足够用于当前粗略性能观察。
        index = int(round((len(ordered) - 1) * percent))
        return ordered[index]

    return {
        'mean': sum(values) / len(values),
        'min': ordered[0],
        'max': ordered[-1],
        'p50': percentile(0.50),
        'p90': percentile(0.90),
    }


def print_profile_summary(profile_records):
    """打印 rollout 各阶段耗时统计。"""

    if not profile_records:
        return

    keys = (
        'graph_build',
        'model_forward',
        'action_select',
        'env_step',
        'total',
    )

    print()
    print('profile summary')
    print('-' * 96)
    print('stage              mean_ms   p50_ms   p90_ms   min_ms   max_ms   percent')

    total_mean = summarize_times([record['total'] for record in profile_records])['mean']
    for key in keys:
        values = [record[key] for record in profile_records]
        stats = summarize_times(values)
        percent = 0.0 if total_mean == 0 else stats['mean'] / total_mean * 100
        print(
            f'{key:<18}'
            f'{stats["mean"] * 1000:8.2f} '
            f'{stats["p50"] * 1000:8.2f} '
            f'{stats["p90"] * 1000:8.2f} '
            f'{stats["min"] * 1000:8.2f} '
            f'{stats["max"] * 1000:8.2f} '
            f'{percent:8.1f}%'
        )

    total_seconds = sum(record['total'] for record in profile_records)
    steps_per_second = 0.0 if total_seconds == 0 else len(profile_records) / total_seconds
    print('-' * 96)
    print(
        f'profiled_steps={len(profile_records)} '
        f'total_wall_time={total_seconds:.3f}s '
        f'throughput={steps_per_second:.2f} env_steps/s'
    )


def run_rollout(args):
    """执行一次无训练 rollout，并把每一步的关键信息打印到终端。"""

    if args.steps <= 0:
        raise ValueError('--steps must be positive')
    if args.action_count <= 0:
        raise ValueError('--action-count must be positive')

    device = resolve_device(args.device)
    fruit_queue = parse_fruit_queue(args.fruit_queue)

    # 同时固定 Python 随机源和 PyTorch 随机源，方便复现实验输出。
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # 这里仍然使用正式的 DaxiguaEnv，而不是直接操作 HeadlessGame。
    # 这样 smoke test 验证的是后续训练实际会使用的环境接口。
    env = DaxiguaEnv(
        DaxiguaEnvConfig(
            action_count=args.action_count,
            max_physics_frames=args.max_physics_frames,
            stable_frames=args.stable_frames,
        )
    )
    graph_builder = GraphBuilder()

    # 这是一个随机初始化、不会训练的 Q 网络。
    # 它的输出只用于验证 shape、索引映射和 step 调用是否正常。
    model = GNNQNetwork(
        hidden_dim=args.hidden_dim,
        message_layers=args.message_layers,
    ).to(device)
    model.eval()

    obs, info = env.reset(seed=args.seed, fruit_queue=fruit_queue)
    total_reward = 0.0
    profile_records = []

    print('temporary rollout smoke test')
    print(
        f'seed={args.seed} policy={args.policy} device={device} '
        f'action_count={args.action_count} max_steps={args.steps}'
    )
    print('-' * 96)

    for rollout_step in range(args.steps):
        candidates = tuple(info['action_candidates'])
        if not candidates:
            print(f'step={rollout_step:02d} stopped: no action candidates')
            break

        # 当前状态和候选动作先构造成图。
        # action 节点数量应该和 candidates 数量一致。
        step_started_at = time.perf_counter()

        graph_started_at = time.perf_counter()
        graph = graph_builder.build(obs, candidates)
        graph_finished_at = time.perf_counter()

        # smoke test 不训练模型，所以关闭梯度追踪，减少开销并避免误以为这里在更新参数。
        forward_started_at = time.perf_counter()
        with torch.no_grad():
            q_values = model(graph).detach().cpu()
        forward_finished_at = time.perf_counter()

        if int(q_values.shape[0]) != len(candidates):
            raise RuntimeError(
                f'q_values length mismatch: got {q_values.shape[0]}, '
                f'expected {len(candidates)}'
            )

        action_started_at = time.perf_counter()
        action_offset = choose_action_offset(q_values, args.policy, rng)
        action_id = graph.action_indices[action_offset]
        action = candidates[action_offset]
        selected_q = float(q_values[action_offset].item())
        summary = q_summary(q_values)
        action_finished_at = time.perf_counter()

        # env.step 当前接收的是候选动作列表下标。
        # 这和 `action.action_index` 目前相同，但这里仍然保留两者打印，方便发现未来错位。
        env_step_started_at = time.perf_counter()
        next_obs, reward, terminated, truncated, next_info = env.step(action_offset)
        env_step_finished_at = time.perf_counter()
        step_finished_at = time.perf_counter()
        total_reward += reward

        if args.profile:
            profile_records.append(
                {
                    'graph_build': graph_finished_at - graph_started_at,
                    'model_forward': forward_finished_at - forward_started_at,
                    'action_select': action_finished_at - action_started_at,
                    'env_step': env_step_finished_at - env_step_started_at,
                    'total': step_finished_at - step_started_at,
                }
            )

        if not args.quiet:
            print(
                f'step={rollout_step:02d} '
                f'action_offset={action_offset:02d} '
                f'action_id={action_id:02d} '
                f'drop_x={action.drop_x:7.2f} '
                f'level={action.current_level} '
                f'q={selected_q:+.5f} '
                f'q[min/mean/max]={summary["min"]:+.5f}/{summary["mean"]:+.5f}/{summary["max"]:+.5f} '
                f'reward={reward:+.1f} '
                f'score={next_obs.score} '
                f'fruits={next_obs.fruit_count} '
                f'merges={len(next_info["merge_events"])} '
                f'frames={next_info["frames_simulated"]} '
                f'stable={next_info["stable"]} '
                f'done={terminated} '
                f'truncated={truncated} '
                f'queue={format_queue(next_obs.fruit_queue)}'
            )

            if args.print_q_values:
                formatted_q_values = ', '.join(f'{value:+.5f}' for value in q_values.tolist())
                print(f'  q_values=[{formatted_q_values}]')

        obs = next_obs
        info = next_info

        if terminated or truncated:
            if not args.quiet:
                print(
                    f'stopped at step={rollout_step:02d}: '
                    f'terminated={terminated}, truncated={truncated}'
                )
            break

    print('-' * 96)
    print(
        f'rollout finished: steps_taken={obs.step_count} '
        f'total_reward={total_reward:+.1f} score={obs.score} done={obs.done}'
    )

    if args.profile:
        print_profile_summary(profile_records)


def main():
    """脚本入口。"""

    args = parse_args()
    run_rollout(args)


if __name__ == '__main__':
    main()
