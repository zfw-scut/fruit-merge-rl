"""对比 accurate 和 fast 物理训练模式。

本脚本用于回答一个很实际的问题：降低 headless 训练的物理精度后，速度提升
有多大，游戏分布又会偏移多少。

它只放在 `daxigua_rl.scripts` 中，不改 pygame 游戏表现层。对比逻辑通过
`DaxiguaEnv` 和公开 `GameState` / `ActionCandidate` 接口访问游戏，继续保持
RL 实验代码和游戏本体隔离。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import torch

from daxigua.config import FPS
from daxigua_rl import DaxiguaEnv, DaxiguaEnvConfig, GraphBuilder
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.reward import RewardConfig


EPISODE_FIELDS = (
    'mode',
    'mode_zh',
    'episode_index',
    'seed',
    'policy',
    'fps',
    'max_physics_frames',
    'stable_frames',
    'space_iterations',
    'score',
    'episode_reward',
    'episode_length',
    'terminated',
    'truncated',
    'max_steps_reached',
    'elapsed_seconds',
    'env_steps_per_second',
    'total_physics_frames',
    'mean_physics_frames',
    'max_step_physics_frames',
    'total_merges',
    'mean_merges_per_step',
    'score_delta_total',
    'mean_score_delta_per_step',
    'final_fruit_count',
    'max_fruit_count',
    'final_max_level',
    'final_height_ratio',
)


SUMMARY_FIELDS = (
    'mode',
    'mode_zh',
    'policy',
    'fps',
    'max_physics_frames',
    'stable_frames',
    'space_iterations',
    'episodes',
    'total_env_steps',
    'elapsed_seconds',
    'env_steps_per_second',
    'mean_score',
    'max_score',
    'min_score',
    'mean_episode_reward',
    'mean_episode_length',
    'mean_physics_frames',
    'mean_merges_per_step',
    'mean_score_delta_per_step',
    'terminal_rate',
    'truncated_rate',
    'max_steps_reached_rate',
    'mean_final_fruit_count',
    'mean_max_fruit_count',
    'mean_final_height_ratio',
    'score_vs_accurate_delta',
    'speed_vs_accurate_ratio',
)


@dataclass(frozen=True)
class PhysicsModeSpec:
    """一个待对比的物理模式配置。"""

    name: str
    name_zh: str
    fps: int
    max_physics_frames: int
    stable_frames: int
    space_iterations: int


@dataclass(frozen=True)
class EpisodeResult:
    """单局对比结果。

    单局结果保留得比较细，是为了后续判断 fast 模式到底改变了哪些行为：
    是局长变短、合成次数变化、物理截断变多，还是单纯变快。
    """

    mode: str
    mode_zh: str
    episode_index: int
    seed: int
    policy: str
    fps: int
    max_physics_frames: int
    stable_frames: int
    space_iterations: int
    score: float
    episode_reward: float
    episode_length: int
    terminated: bool
    truncated: bool
    max_steps_reached: bool
    elapsed_seconds: float
    env_steps_per_second: float
    total_physics_frames: int
    mean_physics_frames: float
    max_step_physics_frames: int
    total_merges: int
    mean_merges_per_step: float
    score_delta_total: float
    mean_score_delta_per_step: float
    final_fruit_count: int
    max_fruit_count: int
    final_max_level: int
    final_height_ratio: float


def parse_args():
    """解析物理模式对比命令行参数。"""

    parser = argparse.ArgumentParser(description='对比 accurate 和 fast 物理模式的速度与游戏分布差异。')

    parser.add_argument('--checkpoint', default=None, help='可选 DQN checkpoint；不提供时使用随机策略。')
    parser.add_argument('--device', default='cpu', help='模型运行设备，例如 cpu、cuda 或 cuda:0。')
    parser.add_argument('--episodes', type=int, default=20, help='每种物理模式评估多少局。')
    parser.add_argument('--max-steps', type=int, default=500, help='每局最多投放多少次，防止极端长局。')
    parser.add_argument('--seed', type=int, default=0, help='基准随机种子；各模式会复用同一批 episode seed。')
    parser.add_argument('--action-count', type=int, default=None, help='候选动作数量；默认读取 checkpoint args 或使用 15。')
    parser.add_argument('--run-dir', default=None, help='输出目录；默认 runs/physics_mode_compare_YYYYMMDD_HHMMSS。')
    parser.add_argument('--progress-interval', type=float, default=3.0, help='每多少秒打印一次实时进度；0 表示关闭。')

    # accurate 模式默认使用当前项目真实 headless 配置。
    parser.add_argument('--accurate-fps', type=int, default=FPS, help='accurate 模式物理 fps。')
    parser.add_argument('--accurate-max-physics-frames', type=int, default=720, help='accurate 模式每次投放最大物理帧。')
    parser.add_argument('--accurate-stable-frames', type=int, default=15, help='accurate 模式稳定判定连续帧数。')
    parser.add_argument('--accurate-space-iterations', type=int, default=32, help='accurate 模式 Pymunk 迭代次数。')

    # fast 模式集中测试用户关心的 15/30/45 fps，其他参数按当前讨论方案固定。
    parser.add_argument('--fast-fps-values', default='15,30,45', help='逗号分隔的 fast fps 列表，例如 15,30,45。')
    parser.add_argument('--fast-max-physics-frames', type=int, default=200, help='fast 模式每次投放最大物理帧。')
    parser.add_argument('--fast-stable-frames', type=int, default=6, help='fast 模式稳定判定连续帧数。')
    parser.add_argument('--fast-space-iterations', type=int, default=8, help='fast 模式 Pymunk 迭代次数。')

    # reward 参数保持和训练脚本一致，避免对比时 reward 定义漂移。
    parser.add_argument('--score-scale', type=float, default=1.0, help='合成分数奖励缩放。')
    parser.add_argument('--survival-bonus', type=float, default=0.05, help='未死亡 step 的存活小奖励。')
    parser.add_argument('--height-delta-weight', type=float, default=0.02, help='高度变化奖励权重。')
    parser.add_argument('--danger-height-weight', type=float, default=1.0, help='危险高度持续惩罚权重。')
    parser.add_argument('--terminal-penalty', type=float, default=-100.0, help='游戏失败终局惩罚。')

    return parser.parse_args()


def validate_args(args):
    """检查明显错误，让实验参数问题尽早暴露。"""

    positive_int_fields = (
        'episodes',
        'max_steps',
        'accurate_fps',
        'accurate_max_physics_frames',
        'accurate_stable_frames',
        'accurate_space_iterations',
        'fast_max_physics_frames',
        'fast_stable_frames',
        'fast_space_iterations',
    )
    for field_name in positive_int_fields:
        if int(getattr(args, field_name)) <= 0:
            raise ValueError(f'--{field_name.replace("_", "-")} must be positive')

    if args.action_count is not None and int(args.action_count) <= 0:
        raise ValueError('--action-count must be positive')
    if args.progress_interval < 0.0:
        raise ValueError('--progress-interval must be >= 0')

    parse_fast_fps_values(args.fast_fps_values)


def parse_fast_fps_values(raw_value):
    """把 `15,30,45` 这种字符串解析成整数 fps 列表。"""

    values = []
    for item in str(raw_value).split(','):
        item = item.strip()
        if not item:
            continue
        fps = int(item)
        if fps <= 0:
            raise ValueError('fast fps values must be positive')
        values.append(fps)
    if not values:
        raise ValueError('--fast-fps-values must contain at least one fps value')
    return tuple(values)


def build_mode_specs(args):
    """根据命令行参数生成 accurate 和 fast 模式列表。"""

    modes = [
        PhysicsModeSpec(
            name='accurate',
            name_zh='精确模式',
            fps=int(args.accurate_fps),
            max_physics_frames=int(args.accurate_max_physics_frames),
            stable_frames=int(args.accurate_stable_frames),
            space_iterations=int(args.accurate_space_iterations),
        )
    ]

    for fps in parse_fast_fps_values(args.fast_fps_values):
        modes.append(
            PhysicsModeSpec(
                name=f'fast_{fps}',
                name_zh=f'快速模式{fps}fps',
                fps=int(fps),
                max_physics_frames=int(args.fast_max_physics_frames),
                stable_frames=int(args.fast_stable_frames),
                space_iterations=int(args.fast_space_iterations),
            )
        )

    return tuple(modes)


def resolve_device(device_name):
    """解析 torch 设备。"""

    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA device, but torch.cuda.is_available() is False')
    return device


def create_run_dir(run_dir):
    """创建本次对比实验输出目录。"""

    if run_dir:
        path = Path(run_dir)
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = Path('runs') / f'physics_mode_compare_{stamp}'

    path.mkdir(parents=True, exist_ok=True)
    (path / 'plots').mkdir(exist_ok=True)
    (path / 'mplconfig').mkdir(exist_ok=True)
    return path


def load_checkpoint(path, device):
    """读取训练 checkpoint；没有路径时返回 None。"""

    if not path:
        return None

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {checkpoint_path}')
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def build_model_from_checkpoint(checkpoint, device):
    """根据 checkpoint 中保存的参数重建 GNN-Q 模型。"""

    if checkpoint is None:
        return None

    checkpoint_args = checkpoint.get('args', {})
    model = GNNQNetwork(
        hidden_dim=int(checkpoint_args.get('hidden_dim', 128)),
        message_layers=int(checkpoint_args.get('message_layers', 3)),
        activation=checkpoint_args.get('activation', 'silu'),
        dropout=float(checkpoint_args.get('dropout', 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint['online_model'])
    model.eval()
    return model


def resolve_action_count(args, checkpoint):
    """决定候选动作数量，优先使用命令行，其次使用 checkpoint，最后回到默认 15。"""

    if args.action_count is not None:
        return int(args.action_count)
    if checkpoint is not None:
        return int(checkpoint.get('args', {}).get('action_count', 15))
    return 15


def build_reward_config(args):
    """创建和训练脚本一致的 reward 配置。"""

    return RewardConfig(
        score_scale=args.score_scale,
        survival_bonus=args.survival_bonus,
        height_delta_weight=args.height_delta_weight,
        danger_height_weight=args.danger_height_weight,
        terminal_penalty=args.terminal_penalty,
    )


def run_episode(mode, args, action_count, reward_config, model, graph_builder, device, episode_index):
    """在指定物理模式下跑一局游戏，并返回单局统计。"""

    env = DaxiguaEnv(
        config=DaxiguaEnvConfig(
            action_count=action_count,
            physics_fps=mode.fps,
            max_physics_frames=mode.max_physics_frames,
            stable_frames=mode.stable_frames,
            space_iterations=mode.space_iterations,
            reward_config=reward_config,
        )
    )
    episode_seed = int(args.seed) + int(episode_index)
    action_rng = random.Random(episode_seed)
    obs, info = env.reset(seed=episode_seed)

    policy_name = 'checkpoint' if model is not None else 'random'
    episode_reward = 0.0
    episode_length = 0
    total_physics_frames = 0
    max_step_physics_frames = 0
    total_merges = 0
    score_delta_total = 0.0
    max_fruit_count = int(obs.fruit_count)
    terminated = False
    truncated = False
    max_steps_reached = False

    start_time = time.perf_counter()
    for _step_index in range(int(args.max_steps)):
        candidates = tuple(info['action_candidates'])
        if not candidates:
            break

        action_offset = choose_action(
            model=model,
            graph_builder=graph_builder,
            obs=obs,
            candidates=candidates,
            action_rng=action_rng,
            device=device,
        )
        obs, reward, terminated, truncated, info = env.step(action_offset)

        frames_simulated = int(info.get('frames_simulated', 0))
        merge_count = len(info.get('merge_events', ()))
        score_delta = float(info.get('score_delta', 0.0))

        episode_reward += float(reward)
        episode_length += 1
        total_physics_frames += frames_simulated
        max_step_physics_frames = max(max_step_physics_frames, frames_simulated)
        total_merges += merge_count
        score_delta_total += score_delta
        max_fruit_count = max(max_fruit_count, int(obs.fruit_count))

        if terminated or truncated:
            break
    else:
        max_steps_reached = True

    elapsed = time.perf_counter() - start_time
    speed = 0.0 if elapsed <= 0.0 else episode_length / elapsed
    mean_frames = 0.0 if episode_length <= 0 else total_physics_frames / episode_length
    mean_merges = 0.0 if episode_length <= 0 else total_merges / episode_length
    mean_score_delta = 0.0 if episode_length <= 0 else score_delta_total / episode_length

    return EpisodeResult(
        mode=mode.name,
        mode_zh=mode.name_zh,
        episode_index=int(episode_index),
        seed=episode_seed,
        policy=policy_name,
        fps=mode.fps,
        max_physics_frames=mode.max_physics_frames,
        stable_frames=mode.stable_frames,
        space_iterations=mode.space_iterations,
        score=float(obs.score),
        episode_reward=float(episode_reward),
        episode_length=int(episode_length),
        terminated=bool(terminated),
        truncated=bool(truncated),
        max_steps_reached=bool(max_steps_reached),
        elapsed_seconds=float(elapsed),
        env_steps_per_second=float(speed),
        total_physics_frames=int(total_physics_frames),
        mean_physics_frames=float(mean_frames),
        max_step_physics_frames=int(max_step_physics_frames),
        total_merges=int(total_merges),
        mean_merges_per_step=float(mean_merges),
        score_delta_total=float(score_delta_total),
        mean_score_delta_per_step=float(mean_score_delta),
        final_fruit_count=int(obs.fruit_count),
        max_fruit_count=int(max_fruit_count),
        final_max_level=int(obs.max_level),
        final_height_ratio=float(height_ratio(obs)),
    )


def choose_action(model, graph_builder, obs, candidates, action_rng, device):
    """根据 checkpoint 模型或随机策略选择动作。"""

    if model is None:
        return action_rng.randrange(len(candidates))

    graph = graph_builder.build(obs, candidates)
    with torch.no_grad():
        q_values = model(graph).detach().cpu()
    return int(torch.argmax(q_values).item())


def height_ratio(state):
    """计算和 reward 中一致的堆叠高度比例。"""

    playable_height = max(1.0, float(state.geometry.height - state.geometry.spawn_y))
    ratio = float(state.max_height) / playable_height
    return max(0.0, min(1.0, ratio))


def result_to_row(result):
    """把 EpisodeResult 转换成 CSV 友好的 dict。"""

    row = asdict(result)
    row['terminated'] = int(result.terminated)
    row['truncated'] = int(result.truncated)
    row['max_steps_reached'] = int(result.max_steps_reached)
    return row


def summarize_mode(mode, policy_name, results, accurate_summary=None):
    """把一个模式下的多局结果汇总成一行 summary。"""

    total_steps = sum(result.episode_length for result in results)
    elapsed = sum(result.elapsed_seconds for result in results)
    mean_score = mean(result.score for result in results)
    speed = 0.0 if elapsed <= 0.0 else total_steps / elapsed

    summary = {
        'mode': mode.name,
        'mode_zh': mode.name_zh,
        'policy': policy_name,
        'fps': mode.fps,
        'max_physics_frames': mode.max_physics_frames,
        'stable_frames': mode.stable_frames,
        'space_iterations': mode.space_iterations,
        'episodes': len(results),
        'total_env_steps': total_steps,
        'elapsed_seconds': elapsed,
        'env_steps_per_second': speed,
        'mean_score': mean_score,
        'max_score': max((result.score for result in results), default=0.0),
        'min_score': min((result.score for result in results), default=0.0),
        'mean_episode_reward': mean(result.episode_reward for result in results),
        'mean_episode_length': mean(result.episode_length for result in results),
        'mean_physics_frames': mean(result.mean_physics_frames for result in results),
        'mean_merges_per_step': mean(result.mean_merges_per_step for result in results),
        'mean_score_delta_per_step': mean(result.mean_score_delta_per_step for result in results),
        'terminal_rate': mean(1.0 if result.terminated else 0.0 for result in results),
        'truncated_rate': mean(1.0 if result.truncated else 0.0 for result in results),
        'max_steps_reached_rate': mean(1.0 if result.max_steps_reached else 0.0 for result in results),
        'mean_final_fruit_count': mean(result.final_fruit_count for result in results),
        'mean_max_fruit_count': mean(result.max_fruit_count for result in results),
        'mean_final_height_ratio': mean(result.final_height_ratio for result in results),
        'score_vs_accurate_delta': '',
        'speed_vs_accurate_ratio': '',
    }

    if accurate_summary is not None:
        summary['score_vs_accurate_delta'] = mean_score - float(accurate_summary['mean_score'])
        accurate_speed = float(accurate_summary['env_steps_per_second'])
        summary['speed_vs_accurate_ratio'] = '' if accurate_speed <= 0.0 else speed / accurate_speed

    return summary


def mean(values):
    """计算平均值；输入为空时返回 0。"""

    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def write_csv(path, fieldnames, rows):
    """把字典行写成 CSV。"""

    with Path(path).open('w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})


def maybe_print_progress(args, last_progress_at, mode, episode_index, total_episodes, total_elapsed):
    """按固定时间间隔打印双语进度。"""

    if args.progress_interval <= 0.0:
        return last_progress_at

    now = time.perf_counter()
    if now - last_progress_at < args.progress_interval:
        return last_progress_at

    done = max(1, episode_index)
    remaining = max(0, total_episodes - episode_index)
    seconds_per_episode = total_elapsed / done
    eta_seconds = remaining * seconds_per_episode
    print(
        '[progress 进度] '
        f'mode={mode.name} 模式={mode.name_zh} | '
        f'episode={episode_index}/{total_episodes} 局数={episode_index}/{total_episodes} | '
        f'elapsed={total_elapsed:.1f}s 已用={total_elapsed:.1f}秒 | '
        f'eta={eta_seconds:.1f}s 预计剩余={eta_seconds:.1f}秒',
        flush=True,
    )
    return now


def print_summary_table(summary_rows):
    """在终端打印紧凑汇总表。"""

    print('\n[summary 汇总]', flush=True)
    header = (
        'mode 模式',
        'fps',
        'speed 投放/秒',
        'score 平均分',
        'len 平均局长',
        'frames 平均物理帧',
        'trunc 截断率',
        'speed_ratio 速度倍率',
    )
    print(
        f'{header[0]:<18} {header[1]:>5} {header[2]:>14} {header[3]:>12} '
        f'{header[4]:>12} {header[5]:>16} {header[6]:>10} {header[7]:>16}',
        flush=True,
    )
    for row in summary_rows:
        speed_ratio = row['speed_vs_accurate_ratio']
        speed_ratio_text = '' if speed_ratio == '' else f'{float(speed_ratio):.2f}x'
        print(
            f"{row['mode']:<18} {int(row['fps']):>5} "
            f"{float(row['env_steps_per_second']):>14.2f} "
            f"{float(row['mean_score']):>12.1f} "
            f"{float(row['mean_episode_length']):>12.1f} "
            f"{float(row['mean_physics_frames']):>16.1f} "
            f"{float(row['truncated_rate']):>10.2f} "
            f"{speed_ratio_text:>16}",
            flush=True,
        )


def maybe_plot_summary(run_dir, summary_rows):
    """生成物理模式对比图；matplotlib 不可用时只跳过图片。"""

    if not summary_rows:
        return False

    try:
        import os

        os.environ.setdefault('MPLCONFIGDIR', str((run_dir / 'mplconfig').resolve()))

        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'[plot skipped 绘图跳过] reason={exc}', flush=True)
        return False

    labels = [row['mode'] for row in summary_rows]
    speeds = [float(row['env_steps_per_second']) for row in summary_rows]
    scores = [float(row['mean_score']) for row in summary_rows]
    lengths = [float(row['mean_episode_length']) for row in summary_rows]
    frames = [float(row['mean_physics_frames']) for row in summary_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    # 图片标题保持英文，避免没有 CJK 字体的环境在保存图表时刷屏 warning；
    # 终端输出仍保留中英双语，方便直接阅读。
    plot_bar(axes[0], labels, speeds, 'Speed', 'env steps per second')
    plot_bar(axes[1], labels, scores, 'Mean score', 'score')
    plot_bar(axes[2], labels, lengths, 'Mean episode length', 'steps')
    plot_bar(axes[3], labels, frames, 'Mean physics frames', 'frames per step')

    output_path = run_dir / 'plots' / 'physics_mode_comparison.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def plot_bar(axis, labels, values, title, ylabel):
    """绘制一个简单柱状图。"""

    axis.bar(labels, values)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis='y', alpha=0.25)
    axis.tick_params(axis='x', rotation=20)


def write_config(run_dir, args, modes, policy_name):
    """保存本次对比配置，方便之后复现实验。"""

    config = {
        'argv': sys.argv,
        'args': vars(args),
        'policy': policy_name,
        'modes': [asdict(mode) for mode in modes],
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    (run_dir / 'config.json').write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def run_comparison(args):
    """执行 accurate/fast 模式对比实验。"""

    validate_args(args)
    device = resolve_device(args.device)
    run_dir = create_run_dir(args.run_dir)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model_from_checkpoint(checkpoint, device)
    policy_name = 'checkpoint' if model is not None else 'random'
    action_count = resolve_action_count(args, checkpoint)
    reward_config = build_reward_config(args)
    modes = build_mode_specs(args)
    graph_builder = GraphBuilder()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    write_config(run_dir, args, modes, policy_name)

    print(f'run_dir={run_dir}', flush=True)
    print(f'policy={policy_name} 策略={policy_name}', flush=True)
    print(f'action_count={action_count} 候选动作数={action_count}', flush=True)
    print(f'episodes_per_mode={args.episodes} 每种模式局数={args.episodes}', flush=True)

    all_episode_rows = []
    summary_rows = []
    accurate_summary = None

    for mode in modes:
        print(
            '\n[mode start 模式开始] '
            f'mode={mode.name} 模式={mode.name_zh} | '
            f'fps={mode.fps} | max_frames={mode.max_physics_frames} 最大物理帧={mode.max_physics_frames} | '
            f'stable_frames={mode.stable_frames} 稳定帧={mode.stable_frames} | '
            f'iterations={mode.space_iterations} 迭代次数={mode.space_iterations}',
            flush=True,
        )

        mode_results = []
        mode_start = time.perf_counter()
        last_progress_at = mode_start
        for episode_index in range(int(args.episodes)):
            result = run_episode(
                mode=mode,
                args=args,
                action_count=action_count,
                reward_config=reward_config,
                model=model,
                graph_builder=graph_builder,
                device=device,
                episode_index=episode_index,
            )
            mode_results.append(result)
            all_episode_rows.append(result_to_row(result))
            last_progress_at = maybe_print_progress(
                args=args,
                last_progress_at=last_progress_at,
                mode=mode,
                episode_index=episode_index + 1,
                total_episodes=args.episodes,
                total_elapsed=time.perf_counter() - mode_start,
            )

        summary = summarize_mode(
            mode=mode,
            policy_name=policy_name,
            results=mode_results,
            accurate_summary=accurate_summary,
        )
        if mode.name == 'accurate':
            accurate_summary = summary
        summary_rows.append(summary)

        print(
            '[mode done 模式完成] '
            f'mode={mode.name} | speed={summary["env_steps_per_second"]:.2f} 投放/秒 | '
            f'mean_score={summary["mean_score"]:.1f} 平均分 | '
            f'mean_length={summary["mean_episode_length"]:.1f} 平均局长 | '
            f'truncated_rate={summary["truncated_rate"]:.2f} 截断率',
            flush=True,
        )

    write_csv(run_dir / 'episode_metrics.csv', EPISODE_FIELDS, all_episode_rows)
    write_csv(run_dir / 'summary.csv', SUMMARY_FIELDS, summary_rows)
    maybe_plot_summary(run_dir, summary_rows)
    print_summary_table(summary_rows)

    print(f'\nfinished 完成 | run_dir={run_dir}', flush=True)
    print(f'episode_csv 单局明细={run_dir / "episode_metrics.csv"}', flush=True)
    print(f'summary_csv 汇总={run_dir / "summary.csv"}', flush=True)
    print(f'plot 图表={run_dir / "plots" / "physics_mode_comparison.png"}', flush=True)
    return run_dir


def main():
    """命令行入口。"""

    run_comparison(parse_args())


if __name__ == '__main__':
    main()
