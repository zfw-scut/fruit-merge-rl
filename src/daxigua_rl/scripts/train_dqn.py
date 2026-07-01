"""第一版 DQN 训练入口。

运行方式：

    PYTHONPATH=src python -m daxigua_rl.scripts.train_dqn

当前脚本负责把已有训练组件串成完整闭环：

    RolloutCollector -> ReplayBuffer -> DQNTrainer -> checkpoint/metrics/plots

它仍然是第一版同步单进程训练脚本，不包含多进程采样、GraphBatch、Double DQN
或 TensorBoard。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from daxigua_rl import DaxiguaEnv, DaxiguaEnvConfig, GraphBuilder, ReplayBuffer
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.reward import RewardConfig
from daxigua_rl.training import (
    DQNTrainer,
    DQNTrainerConfig,
    RolloutCollector,
)


METRIC_FIELDS = (
    'update_step',
    'env_steps',
    'epsilon',
    'buffer_size',
    'loss',
    'mean_q',
    'mean_target',
    'mean_reward',
    'mean_abs_td_error',
    'bootstrap_count',
    'grad_norm',
    'target_synced',
    'collect_steps',
    'collect_total_reward',
    'collect_episodes',
    'collect_mean_episode_reward',
    'collect_mean_episode_length',
    'collect_mean_episode_score',
    'random_actions',
    'greedy_actions',
    'eval_score_mean',
    'eval_reward_mean',
    'eval_length_mean',
    'eval_episodes',
    'updates_per_second',
    'env_steps_per_second',
)


def parse_args():
    """解析训练命令行参数。"""

    parser = argparse.ArgumentParser(description='训练第一版 GNN-DQN 合成大西瓜智能体。')

    # 训练规模。
    parser.add_argument('--total-updates', type=int, default=10_000, help='总共执行多少次 DQN 参数更新。')
    parser.add_argument('--warmup-steps', type=int, default=1_000, help='正式训练前随机收集多少条经验。')
    parser.add_argument('--collect-per-update', type=int, default=1, help='每次参数更新前收集多少条新经验。')
    parser.add_argument('--batch-size', type=int, default=32, help='每次 train_step 从 ReplayBuffer 采样多少条经验。')
    parser.add_argument('--replay-capacity', type=int, default=100_000, help='ReplayBuffer 最大容量。')

    # epsilon-greedy。
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='初始随机探索概率。')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='最终保留的随机探索概率。')
    parser.add_argument('--epsilon-decay-steps', type=int, default=50_000, help='epsilon 线性衰减需要的环境步数。')

    # DQN 算法。
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Adam 学习率。')
    parser.add_argument('--gamma', type=float, default=0.99, help='未来奖励折扣因子。')
    parser.add_argument('--target-update-interval', type=int, default=1_000, help='target network 同步间隔，按 train_step 计。')
    parser.add_argument('--grad-clip-norm', type=float, default=10.0, help='梯度裁剪阈值；传 0 表示关闭。')

    # 模型规模。
    parser.add_argument('--hidden-dim', type=int, default=128, help='GNN 隐藏层维度。')
    parser.add_argument('--message-layers', type=int, default=3, help='GNN message passing 层数。')
    parser.add_argument('--dropout', type=float, default=0.0, help='GNN dropout。')
    parser.add_argument('--activation', choices=('relu', 'silu'), default='silu', help='GNN 激活函数。')

    # 环境参数。
    parser.add_argument('--seed', type=int, default=0, help='随机种子。')
    parser.add_argument('--action-count', type=int, default=15, help='离散候选投放动作数量。')
    parser.add_argument('--max-physics-frames', type=int, default=720, help='每次投放后最多推进多少物理帧。')
    parser.add_argument('--stable-frames', type=int, default=15, help='连续多少帧稳定后结束本次 step。')

    # reward 参数。
    parser.add_argument('--score-scale', type=float, default=1.0, help='合成分数奖励缩放。')
    parser.add_argument('--survival-bonus', type=float, default=0.05, help='未死亡 step 的存活小奖励。')
    parser.add_argument('--height-delta-weight', type=float, default=0.02, help='高度变化奖励权重。')
    parser.add_argument('--danger-height-weight', type=float, default=1.0, help='危险高度持续惩罚权重。')
    parser.add_argument('--terminal-penalty', type=float, default=-100.0, help='游戏失败终局惩罚。')

    # 日志、保存、评估和可视化。
    parser.add_argument('--run-dir', default=None, help='训练输出目录；默认 runs/dqn_YYYYMMDD_HHMMSS。')
    parser.add_argument('--log-interval', type=int, default=100, help='每多少次 update 记录并打印一次日志。')
    parser.add_argument('--save-interval', type=int, default=5_000, help='每多少次 update 保存一次 step checkpoint；0 表示关闭周期保存。')
    parser.add_argument('--eval-interval', type=int, default=5_000, help='每多少次 update 执行一次 greedy 评估；0 表示关闭。')
    parser.add_argument('--eval-episodes', type=int, default=5, help='每次评估跑多少局。')
    parser.add_argument('--eval-max-steps', type=int, default=500, help='每局评估最多投放多少次，防止极端长局。')
    parser.add_argument('--plot-interval', type=int, default=1_000, help='每多少次 update 生成一次曲线图；0 表示只在结束时尝试生成。')
    parser.add_argument('--progress-interval', type=float, default=3.0, help='每多少秒打印一次轻量训练进度；0 表示关闭。')

    # 运行设备。
    parser.add_argument('--device', default='cpu', help='模型设备，例如 cpu、cuda 或 cuda:0。')

    return parser.parse_args()


def validate_args(args):
    """检查训练参数中的明显错误。"""

    positive_int_fields = (
        'total_updates',
        'warmup_steps',
        'collect_per_update',
        'batch_size',
        'replay_capacity',
        'epsilon_decay_steps',
        'target_update_interval',
        'hidden_dim',
        'message_layers',
        'action_count',
        'max_physics_frames',
        'stable_frames',
        'log_interval',
        'eval_episodes',
        'eval_max_steps',
    )
    for field_name in positive_int_fields:
        if int(getattr(args, field_name)) <= 0:
            raise ValueError(f'--{field_name.replace("_", "-")} must be positive')

    non_negative_intervals = ('save_interval', 'eval_interval', 'plot_interval')
    for field_name in non_negative_intervals:
        if int(getattr(args, field_name)) < 0:
            raise ValueError(f'--{field_name.replace("_", "-")} must be >= 0')

    if args.epsilon_start < 0.0 or args.epsilon_start > 1.0:
        raise ValueError('--epsilon-start must be in [0, 1]')
    if args.epsilon_end < 0.0 or args.epsilon_end > 1.0:
        raise ValueError('--epsilon-end must be in [0, 1]')
    if args.learning_rate <= 0.0:
        raise ValueError('--learning-rate must be positive')
    if args.gamma < 0.0 or args.gamma > 1.0:
        raise ValueError('--gamma must be in [0, 1]')
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError('--dropout must be in [0, 1)')
    if args.progress_interval < 0.0:
        raise ValueError('--progress-interval must be >= 0')


def resolve_device(device_name):
    """解析 torch 设备。"""

    device = torch.device(device_name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA device, but torch.cuda.is_available() is False')
    return device


def create_run_dir(run_dir):
    """创建本次训练输出目录。"""

    if run_dir:
        path = Path(run_dir)
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = Path('runs') / f'dqn_{stamp}'

    path.mkdir(parents=True, exist_ok=True)
    (path / 'checkpoints').mkdir(exist_ok=True)
    (path / 'plots').mkdir(exist_ok=True)
    (path / 'mplconfig').mkdir(exist_ok=True)
    return path


def set_random_seeds(seed):
    """设置 Python 和 PyTorch 随机种子。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env_config(args):
    """根据命令行参数创建环境配置。"""

    reward_config = RewardConfig(
        score_scale=args.score_scale,
        survival_bonus=args.survival_bonus,
        height_delta_weight=args.height_delta_weight,
        danger_height_weight=args.danger_height_weight,
        terminal_penalty=args.terminal_penalty,
    )
    return DaxiguaEnvConfig(
        action_count=args.action_count,
        max_physics_frames=args.max_physics_frames,
        stable_frames=args.stable_frames,
        reward_config=reward_config,
    )


def build_model(args):
    """创建一份 GNN-Q 模型。"""

    return GNNQNetwork(
        hidden_dim=args.hidden_dim,
        message_layers=args.message_layers,
        activation=args.activation,
        dropout=args.dropout,
    )


def linear_epsilon(env_steps, args):
    """按环境步数线性衰减 epsilon。"""

    progress = min(1.0, max(0.0, float(env_steps) / float(args.epsilon_decay_steps)))
    return args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)


class MetricLogger:
    """把训练指标同时保存到内存和 CSV。"""

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.rows = []
        self._file = self.csv_path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=METRIC_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row):
        """写入一行指标。"""

        normalized = {field: row.get(field, '') for field in METRIC_FIELDS}
        self.rows.append(normalized)
        self._writer.writerow(normalized)
        self._file.flush()

    def close(self):
        """关闭 CSV 文件。"""

        self._file.close()


def write_config(run_dir, args):
    """保存本次训练配置。"""

    config = {
        'argv': sys.argv,
        'args': vars(args),
        'created_at': datetime.now().isoformat(timespec='seconds'),
    }
    path = run_dir / 'config.json'
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')


def save_checkpoint(
        run_dir,
        online_model,
        target_model,
        optimizer,
        args,
        update_step,
        env_steps,
        epsilon,
        latest_metrics=None,
        step_checkpoint=False):
    """保存模型 checkpoint。"""

    checkpoint = {
        'online_model': online_model.state_dict(),
        'target_model': target_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(args),
        'update_step': int(update_step),
        'env_steps': int(env_steps),
        'epsilon': float(epsilon),
        'latest_metrics': latest_metrics or {},
        'saved_at': datetime.now().isoformat(timespec='seconds'),
    }

    checkpoint_dir = run_dir / 'checkpoints'
    latest_path = checkpoint_dir / 'latest.pt'
    torch.save(checkpoint, latest_path)

    if step_checkpoint:
        step_path = checkpoint_dir / f'step_{update_step:08d}.pt'
        torch.save(checkpoint, step_path)


def evaluate_policy(model, args, device, seed_offset=10_000):
    """使用独立环境进行 greedy 评估，不写 replay buffer。"""

    env_config = build_env_config(args)
    env = DaxiguaEnv(config=env_config)
    graph_builder = GraphBuilder()

    was_training = model.training
    model.eval()

    episode_scores = []
    episode_rewards = []
    episode_lengths = []

    try:
        for episode_index in range(args.eval_episodes):
            obs, info = env.reset(seed=args.seed + seed_offset + episode_index)
            episode_reward = 0.0
            episode_length = 0

            for _ in range(args.eval_max_steps):
                candidates = tuple(info['action_candidates'])
                if not candidates:
                    break

                graph = graph_builder.build(obs, candidates)
                with torch.no_grad():
                    q_values = model(graph).detach().cpu()
                action_offset = int(torch.argmax(q_values).item())

                obs, reward, terminated, truncated, info = env.step(action_offset)
                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

            episode_scores.append(float(obs.score))
            episode_rewards.append(float(episode_reward))
            episode_lengths.append(int(episode_length))
    finally:
        if was_training:
            model.train()

    return {
        'eval_score_mean': _mean(episode_scores),
        'eval_reward_mean': _mean(episode_rewards),
        'eval_length_mean': _mean(episode_lengths),
        'eval_episodes': len(episode_scores),
    }


def maybe_plot_metrics(run_dir, rows):
    """根据已记录指标生成训练曲线图。"""

    if not rows:
        return False

    # Matplotlib 会尝试写用户目录缓存；当前环境中用户目录可能不可写，所以放到 run 目录。
    os.environ.setdefault('MPLCONFIGDIR', str((run_dir / 'mplconfig').resolve()))

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = _series(rows, 'update_step')
    if not x:
        return False

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.ravel()

    _plot_one(axes[0], x, _series(rows, 'loss'), 'loss', 'SmoothL1 loss')
    _plot_one(axes[1], x, _series(rows, 'collect_mean_episode_score'), 'train score', 'Episode score')
    _plot_one(axes[1], x, _series(rows, 'eval_score_mean'), 'eval score', 'Episode score')
    _plot_one(axes[2], x, _series(rows, 'epsilon'), 'epsilon', 'Epsilon')
    _plot_one(axes[3], x, _series(rows, 'mean_abs_td_error'), 'td error', 'Mean abs TD error')
    _plot_one(axes[4], x, _series(rows, 'grad_norm'), 'grad norm', 'Gradient norm')
    _plot_one(axes[5], x, _series(rows, 'mean_q'), 'mean q', 'Q / target')
    _plot_one(axes[5], x, _series(rows, 'mean_target'), 'mean target', 'Q / target')

    for axis in axes:
        axis.set_xlabel('update')
        axis.grid(True, alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc='best')

    output_path = run_dir / 'plots' / 'training_curves.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _plot_one(axis, x_values, y_values, label, title):
    """绘制单条曲线，自动跳过缺失值。"""

    points = [
        (x_value, y_value)
        for x_value, y_value in zip(x_values, y_values)
        if y_value is not None
    ]
    if not points:
        axis.set_title(title)
        return

    xs, ys = zip(*points)
    axis.plot(xs, ys, label=label, linewidth=1.5)
    axis.set_title(title)


def _series(rows, field):
    """从 metrics rows 中取一列浮点序列。"""

    values = []
    for row in rows:
        value = row.get(field, '')
        if value == '' or value is None:
            values.append(None)
        else:
            values.append(float(value))
    return values


def _mean(values):
    """计算平均值；空列表返回 0。"""

    if not values:
        return 0.0
    return sum(values) / len(values)


def print_log(row):
    """打印一行紧凑训练日志。"""

    parts = [
        f"update={int(row['update_step'])}",
        f"env_steps={int(row['env_steps'])}",
        f"eps={float(row['epsilon']):.3f}",
        f"buf={int(row['buffer_size'])}",
        f"loss={float(row['loss']):.4f}",
        f"q={float(row['mean_q']):+.3f}",
        f"target={float(row['mean_target']):+.3f}",
        f"reward={float(row['mean_reward']):+.3f}",
        f"td={float(row['mean_abs_td_error']):.3f}",
        f"grad={float(row['grad_norm']):.3f}",
        f"rand/greedy={int(row['random_actions'])}/{int(row['greedy_actions'])}",
    ]

    if row.get('collect_mean_episode_score') not in ('', None):
        parts.append(f"train_score={float(row['collect_mean_episode_score']):.1f}")
    if row.get('eval_score_mean') not in ('', None):
        parts.append(f"eval_score={float(row['eval_score_mean']):.1f}")

    print(' | '.join(parts), flush=True)


def maybe_print_progress(
        args,
        last_progress_at,
        phase,
        current,
        total,
        env_steps,
        buffer_size,
        epsilon,
        elapsed,
        latest_loss=None):
    """按固定时间间隔打印轻量进度心跳。"""

    if args.progress_interval <= 0.0:
        return last_progress_at

    now = time.perf_counter()
    if now - last_progress_at < args.progress_interval:
        return last_progress_at

    percent = 0.0 if total <= 0 else min(100.0, current / total * 100.0)
    speed = 0.0 if elapsed <= 0.0 else env_steps / elapsed
    parts = [
        '[progress]',
        f'phase={phase}',
        f'{current}/{total}',
        f'{percent:.1f}%',
        f'env_steps={env_steps}',
        f'buffer={buffer_size}',
        f'eps={epsilon:.3f}',
        f'speed={speed:.2f} env_steps/s',
    ]

    if latest_loss is not None:
        parts.append(f'loss={latest_loss:.4f}')

    print(' | '.join(parts), flush=True)
    return now


def build_metric_row(
        update_step,
        env_steps,
        epsilon,
        train_stats,
        collect_stats,
        eval_stats,
        timing):
    """把训练、采集、评估统计合成一行 CSV 指标。"""

    elapsed = max(1e-9, timing['elapsed'])
    collect_mean_episode_reward = (
        collect_stats.mean_episode_reward if collect_stats.episodes > 0 else ''
    )
    collect_mean_episode_length = (
        collect_stats.mean_episode_length if collect_stats.episodes > 0 else ''
    )
    collect_mean_episode_score = (
        collect_stats.mean_episode_score if collect_stats.episodes > 0 else ''
    )

    return {
        'update_step': update_step,
        'env_steps': env_steps,
        'epsilon': epsilon,
        'buffer_size': collect_stats.buffer_size,
        'loss': train_stats.loss,
        'mean_q': train_stats.mean_q,
        'mean_target': train_stats.mean_target,
        'mean_reward': train_stats.mean_reward,
        'mean_abs_td_error': train_stats.mean_abs_td_error,
        'bootstrap_count': train_stats.bootstrap_count,
        'grad_norm': train_stats.grad_norm,
        'target_synced': int(train_stats.target_synced),
        'collect_steps': collect_stats.steps,
        'collect_total_reward': collect_stats.total_reward,
        'collect_episodes': collect_stats.episodes,
        'collect_mean_episode_reward': collect_mean_episode_reward,
        'collect_mean_episode_length': collect_mean_episode_length,
        'collect_mean_episode_score': collect_mean_episode_score,
        'random_actions': collect_stats.random_actions,
        'greedy_actions': collect_stats.greedy_actions,
        'eval_score_mean': eval_stats.get('eval_score_mean', '') if eval_stats else '',
        'eval_reward_mean': eval_stats.get('eval_reward_mean', '') if eval_stats else '',
        'eval_length_mean': eval_stats.get('eval_length_mean', '') if eval_stats else '',
        'eval_episodes': eval_stats.get('eval_episodes', '') if eval_stats else '',
        'updates_per_second': update_step / elapsed,
        'env_steps_per_second': env_steps / elapsed,
    }


def train(args):
    """执行完整训练流程。"""

    validate_args(args)
    device = resolve_device(args.device)
    run_dir = create_run_dir(args.run_dir)

    # 设置 MPLCONFIGDIR 要在首次 import pyplot 前完成。
    os.environ.setdefault('MPLCONFIGDIR', str((run_dir / 'mplconfig').resolve()))

    set_random_seeds(args.seed)
    write_config(run_dir, args)

    env_config = build_env_config(args)
    env = DaxiguaEnv(config=env_config)
    graph_builder = GraphBuilder()
    replay_buffer = ReplayBuffer(capacity=args.replay_capacity, seed=args.seed + 1)

    online_model = build_model(args).to(device)
    target_model = build_model(args).to(device)
    optimizer = torch.optim.Adam(online_model.parameters(), lr=args.learning_rate)

    grad_clip_norm = None if args.grad_clip_norm == 0 else args.grad_clip_norm
    trainer_config = DQNTrainerConfig(
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        grad_clip_norm=grad_clip_norm,
    )
    trainer = DQNTrainer(
        online_model=online_model,
        target_model=target_model,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        config=trainer_config,
    )
    collector = RolloutCollector(
        env=env,
        graph_builder=graph_builder,
        replay_buffer=replay_buffer,
        model=online_model,
        seed=args.seed + 2,
    )

    metrics = MetricLogger(run_dir / 'metrics.csv')
    env_steps = 0
    latest_row = None

    print(f'run_dir={run_dir}', flush=True)
    print(f'device={device} matplotlib_output={run_dir / "plots" / "training_curves.png"}', flush=True)
    print(f'warmup_steps={args.warmup_steps}', flush=True)

    start_time = time.perf_counter()
    last_progress_at = start_time
    warmup_done = 0
    warmup_total_reward = 0.0
    warmup_chunk_size = max(1, min(100, args.warmup_steps))

    while warmup_done < args.warmup_steps:
        chunk_size = min(warmup_chunk_size, args.warmup_steps - warmup_done)
        warmup_stats = collector.collect_steps(chunk_size, epsilon=1.0)
        warmup_done += warmup_stats.steps
        env_steps += warmup_stats.steps
        warmup_total_reward += warmup_stats.total_reward
        last_progress_at = maybe_print_progress(
            args=args,
            last_progress_at=last_progress_at,
            phase='warmup',
            current=warmup_done,
            total=args.warmup_steps,
            env_steps=env_steps,
            buffer_size=len(replay_buffer),
            epsilon=1.0,
            elapsed=time.perf_counter() - start_time,
        )

    print(
        f'warmup done | env_steps={env_steps} | buffer={len(replay_buffer)} '
        f'| reward={warmup_total_reward:+.2f}',
        flush=True,
    )

    try:
        for update_step in range(1, args.total_updates + 1):
            epsilon = linear_epsilon(env_steps, args)

            collect_stats = collector.collect_steps(args.collect_per_update, epsilon=epsilon)
            env_steps += collect_stats.steps

            train_stats = trainer.train_step()
            last_progress_at = maybe_print_progress(
                args=args,
                last_progress_at=last_progress_at,
                phase='train',
                current=update_step,
                total=args.total_updates,
                env_steps=env_steps,
                buffer_size=len(replay_buffer),
                epsilon=epsilon,
                elapsed=time.perf_counter() - start_time,
                latest_loss=train_stats.loss,
            )

            should_log = update_step % args.log_interval == 0 or update_step == 1
            should_eval = args.eval_interval > 0 and update_step % args.eval_interval == 0
            should_save = args.save_interval > 0 and update_step % args.save_interval == 0
            should_plot = args.plot_interval > 0 and update_step % args.plot_interval == 0

            eval_stats = None
            if should_eval:
                eval_stats = evaluate_policy(online_model, args, device)

            if should_log or should_eval or should_save or should_plot or update_step == args.total_updates:
                latest_row = build_metric_row(
                    update_step=update_step,
                    env_steps=env_steps,
                    epsilon=epsilon,
                    train_stats=train_stats,
                    collect_stats=collect_stats,
                    eval_stats=eval_stats,
                    timing={'elapsed': time.perf_counter() - start_time},
                )
                metrics.log(latest_row)

                if should_log or should_eval:
                    print_log(latest_row)

            if should_save:
                save_checkpoint(
                    run_dir=run_dir,
                    online_model=online_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    args=args,
                    update_step=update_step,
                    env_steps=env_steps,
                    epsilon=epsilon,
                    latest_metrics=latest_row,
                    step_checkpoint=True,
                )

            if should_plot:
                maybe_plot_metrics(run_dir, metrics.rows)

        final_epsilon = linear_epsilon(env_steps, args)
        save_checkpoint(
            run_dir=run_dir,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
            args=args,
            update_step=args.total_updates,
            env_steps=env_steps,
            epsilon=final_epsilon,
            latest_metrics=latest_row,
            step_checkpoint=False,
        )
        maybe_plot_metrics(run_dir, metrics.rows)
    finally:
        metrics.close()

    print(f'training finished | run_dir={run_dir} | env_steps={env_steps}', flush=True)
    return run_dir


def main():
    """命令行入口。"""

    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
