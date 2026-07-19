"""第一版 DQN 训练入口。

运行方式：

    PYTHONPATH=src python -m daxigua_rl.scripts.train_dqn

当前脚本负责把已有训练组件串成完整闭环：

    RolloutCollector -> ReplayBuffer -> DQNTrainer -> checkpoint/metrics/plots

当前 DQN 更新器已经使用 GraphBatch 执行批量图前向；当 `--num-envs > 1`
时，rollout 采集可以切换为多进程 headless 环境并行。
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

from daxigua.config import FPS
from daxigua_rl import DaxiguaEnv, DaxiguaEnvConfig, GraphBuilder, ReplayBuffer
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.reward import REWARD_BREAKDOWN_FIELDS, RewardConfig
from daxigua_rl.training import (
    DQNTrainer,
    DQNTrainerConfig,
    ParallelRolloutCollector,
    RolloutCollector,
    RolloutStats,
)


REWARD_BREAKDOWN_METRIC_FIELDS = (
    ('total', 'collect_mean_reward_total'),
    ('score_reward', 'collect_mean_score_reward'),
    ('survival_bonus', 'collect_mean_survival_bonus'),
    ('height_delta_reward', 'collect_mean_height_delta_reward'),
    ('danger_penalty', 'collect_mean_danger_penalty'),
    ('terminal_penalty', 'collect_mean_terminal_penalty'),
    ('previous_height_ratio', 'collect_mean_previous_height_ratio'),
    ('next_height_ratio', 'collect_mean_next_height_ratio'),
    ('height_delta_ratio', 'collect_mean_height_delta_ratio'),
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
    'collect_mean_reward_total',
    'collect_mean_score_reward',
    'collect_mean_survival_bonus',
    'collect_mean_height_delta_reward',
    'collect_mean_danger_penalty',
    'collect_mean_terminal_penalty',
    'collect_mean_previous_height_ratio',
    'collect_mean_next_height_ratio',
    'collect_mean_height_delta_ratio',
    'collect_seconds',
    'collect_graph_build_seconds',
    'collect_tensor_convert_seconds',
    'collect_action_select_seconds',
    'collect_env_step_seconds',
    'collect_mean_physics_frames',
    'collect_mean_fruit_count',
    'collect_mean_graph_nodes',
    'collect_mean_graph_edges',
    'collect_graph_cache_hit_rate',
    'train_step_seconds',
    'replay_sample_seconds',
    'current_collate_seconds',
    'online_forward_seconds',
    'target_compute_seconds',
    'backward_seconds',
    'optimizer_seconds',
    'eval_seconds',
    'save_seconds',
    'plot_seconds',
    'replay_mode',
    'replay_hot_count',
    'replay_cold_count',
    'replay_pending_cold_count',
    'replay_cold_segments',
    'replay_cold_cache_count',
    'random_actions',
    'greedy_actions',
    'eval_score_mean',
    'eval_score_max',
    'eval_score_min',
    'eval_reward_mean',
    'eval_length_mean',
    'eval_episodes',
    'best_eval_score',
    'best_eval_update',
    'updates_per_second',
    'env_steps_per_second',
)

EPISODE_METRIC_FIELDS = (
    'episode_index',
    'phase',
    'update_step',
    'env_steps',
    'epsilon',
    'score',
    'episode_reward',
    'episode_length',
    'terminated',
    'truncated',
)


def parse_args(argv=None):
    """解析训练命令行参数。"""

    parser = build_arg_parser()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default=None)
    config_args, remaining_args = config_parser.parse_known_args(argv)

    if config_args.config:
        config_defaults = load_config_defaults(config_args.config, parser)
        parser.set_defaults(**config_defaults)
        parser.set_defaults(config=config_args.config)

    return parser.parse_args(remaining_args)


def build_arg_parser():
    """创建训练命令行参数解析器。"""

    parser = argparse.ArgumentParser(description='训练第一版 GNN-DQN 合成大西瓜智能体。')
    parser.add_argument('--config', default=None, help='从 TOML 文件读取训练参数，命令行显式参数会覆盖配置文件。')

    # 训练规模。
    parser.add_argument('--total-updates', type=int, default=10_000, help='总共执行多少次 DQN 参数更新。')
    parser.add_argument('--warmup-steps', type=int, default=1_000, help='正式训练前随机收集多少条经验。')
    parser.add_argument('--collect-per-update', type=int, default=1, help='每次参数更新前收集多少条新经验。')
    parser.add_argument('--batch-size', type=int, default=32, help='每次 train_step 从 ReplayBuffer 采样多少条经验。')
    parser.add_argument('--replay-capacity', type=int, default=100_000, help='ReplayBuffer 最大容量。')
    parser.add_argument('--hot-replay-capacity', type=int, default=None, help='常驻内存的最新 replay 数量；默认 min(10000, replay_capacity)。')
    parser.add_argument('--replay-cold-dir', default=None, help='冷 replay 磁盘目录；默认 run_dir/replay_cold。')
    parser.add_argument('--replay-segment-size', type=int, default=1024, help='冷 replay 每多少条 transition 写一个段文件。')
    parser.add_argument('--replay-cold-cache-size', type=int, default=4096, help='训练采样时最多缓存多少条冷 replay。')
    parser.add_argument('--replay-cold-sample-ratio', type=float, default=0.25, help='每个 batch 期望从冷 replay 采样的比例。')
    parser.add_argument('--replay-cold-cache-refresh-interval', type=int, default=500, help='每多少次 sample 刷新冷 replay 缓存。')

    # epsilon-greedy。
    parser.add_argument(
        '--epsilon-schedule',
        choices=('smooth', 'linear'),
        default='smooth',
        help='epsilon 衰减方式：smooth 按训练进度平滑下降，linear 按环境步数线性下降。',
    )
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='初始随机探索概率。')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='最终保留的随机探索概率。')
    parser.add_argument('--epsilon-decay-steps', type=int, default=50_000, help='linear schedule 下 epsilon 衰减需要的环境步数。')

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
    parser.add_argument(
        '--physics-mode',
        choices=('accurate', 'fast30'),
        default='accurate',
        help='物理模式：accurate 使用当前游戏精度，fast30 使用 30fps 快速训练候选参数。',
    )
    parser.add_argument('--physics-fps', type=int, default=None, help='headless 训练物理步频；不传则由 physics-mode 决定。')
    parser.add_argument('--max-physics-frames', type=int, default=None, help='每次投放后最多推进多少物理帧；不传则由 physics-mode 决定。')
    parser.add_argument('--stable-frames', type=int, default=None, help='连续多少帧稳定后结束本次 step；不传则由 physics-mode 决定。')
    parser.add_argument('--space-iterations', type=int, default=None, help='Pymunk 每个物理步的约束求解迭代次数；不传则由 physics-mode 决定。')

    # 并行采样。num_envs=1 时使用单进程 collector；大于 1 时启用 worker 采样。
    parser.add_argument('--num-envs', type=int, default=1, help='并行 headless 采样环境数量；1 表示关闭并行采样。')
    parser.add_argument('--worker-sync-interval', type=int, default=100, help='并行采样时每多少次 update 同步一次 worker 模型参数。')
    parser.add_argument(
        '--async-rollout',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='并行采样时提前提交下一批 rollout，让采样和训练尽量重叠。',
    )

    # reward 参数。
    parser.add_argument('--score-scale', type=float, default=1.0, help='合成分数奖励缩放。')
    parser.add_argument('--survival-bonus', type=float, default=0.05, help='未死亡 step 的存活小奖励。')
    parser.add_argument('--height-delta-weight', type=float, default=0.02, help='高度变化奖励权重。')
    parser.add_argument('--danger-height-weight', type=float, default=1.0, help='危险高度持续惩罚权重。')
    parser.add_argument('--terminal-penalty', type=float, default=-100.0, help='游戏失败终局惩罚。')

    # 日志、保存、评估和可视化。
    parser.add_argument('--run-dir', default=None, help='训练输出目录；默认 runs/dqn_YYYYMMDD_HHMMSS。')
    parser.add_argument('--log-interval', type=int, default=500, help='每多少次 update 记录并打印一次日志。')
    parser.add_argument('--save-interval', type=int, default=20_000, help='每多少次 update 保存一次 step checkpoint；0 表示关闭周期保存。')
    parser.add_argument('--eval-interval', type=int, default=20_000, help='每多少次 update 执行一次 greedy 评估；0 表示关闭。')
    parser.add_argument('--eval-episodes', type=int, default=10, help='每次评估跑多少局。')
    parser.add_argument('--eval-max-steps', type=int, default=500, help='每局评估最多投放多少次，防止极端长局。')
    parser.add_argument('--plot-interval', type=int, default=10_000, help='每多少次 update 生成一次曲线图；0 表示只在结束时尝试生成。')
    parser.add_argument('--progress-interval', type=float, default=3.0, help='每多少秒打印一次轻量训练进度；0 表示关闭。')

    # 运行设备。
    parser.add_argument('--device', default='cpu', help='模型设备，例如 cpu、cuda 或 cuda:0。')

    return parser


def load_config_defaults(config_path, parser=None):
    """读取 TOML 配置文件，并转换成 argparse 默认值字典。"""

    try:
        import tomllib
    except ModuleNotFoundError as exc:
        raise RuntimeError('loading TOML config requires Python 3.11+ tomllib') from exc

    path = Path(config_path)
    with path.open('rb') as file_obj:
        config = tomllib.load(file_obj)

    if not isinstance(config, dict):
        raise ValueError('training config must be a TOML table')

    if parser is None:
        parser = build_arg_parser()
    allowed_keys = _parser_destinations(parser)
    defaults = {}
    for section_name, section_values in config.items():
        if not isinstance(section_values, dict):
            raise ValueError(f'TOML section [{section_name}] must contain key/value pairs')

        for key, value in section_values.items():
            if allowed_keys is not None and key not in allowed_keys:
                raise ValueError(f'unknown training config key: [{section_name}].{key}')
            defaults[key] = value

    return defaults


def _parser_destinations(parser):
    """返回 argparse parser 当前支持的参数 dest 名称集合。"""

    destinations = set()
    for action in parser._actions:
        if action.dest != 'help':
            destinations.add(action.dest)
    return destinations


def validate_args(args):
    """检查训练参数中的明显错误。"""

    apply_physics_mode_defaults(args)

    positive_int_fields = (
        'total_updates',
        'warmup_steps',
        'collect_per_update',
        'batch_size',
        'replay_capacity',
        'replay_segment_size',
        'replay_cold_cache_refresh_interval',
        'epsilon_decay_steps',
        'target_update_interval',
        'hidden_dim',
        'message_layers',
        'action_count',
        'physics_fps',
        'max_physics_frames',
        'stable_frames',
        'space_iterations',
        'num_envs',
        'worker_sync_interval',
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
    if args.hot_replay_capacity is not None and int(args.hot_replay_capacity) <= 0:
        raise ValueError('--hot-replay-capacity must be positive')
    if args.replay_cold_cache_size < 0:
        raise ValueError('--replay-cold-cache-size must be >= 0')
    if args.replay_cold_sample_ratio < 0.0 or args.replay_cold_sample_ratio > 1.0:
        raise ValueError('--replay-cold-sample-ratio must be in [0, 1]')
    if args.async_rollout and args.num_envs <= 1:
        raise ValueError('--async-rollout requires --num-envs > 1')


def apply_physics_mode_defaults(args):
    """根据 `--physics-mode` 填充未显式指定的物理参数。"""

    mode = getattr(args, 'physics_mode', 'accurate')
    if mode == 'fast30':
        defaults = {
            'physics_fps': 30,
            'max_physics_frames': 240,
            'stable_frames': 6,
            'space_iterations': 8,
        }
    else:
        defaults = {
            'physics_fps': FPS,
            'max_physics_frames': 720,
            'stable_frames': 15,
            'space_iterations': 32,
        }

    for field_name, default_value in defaults.items():
        if getattr(args, field_name, None) is None:
            setattr(args, field_name, default_value)


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
        # 兼容旧测试对象或旧 checkpoint 参数：没有新字段时继续使用当前项目默认值。
        physics_fps=getattr(args, 'physics_fps', FPS),
        max_physics_frames=args.max_physics_frames,
        stable_frames=args.stable_frames,
        space_iterations=getattr(args, 'space_iterations', 32),
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


def build_model_config(args):
    """返回可传给 worker 进程创建同结构 GNN-Q 模型的配置。"""

    return {
        'hidden_dim': args.hidden_dim,
        'message_layers': args.message_layers,
        'activation': args.activation,
        'dropout': args.dropout,
    }


def build_collector(args, env_config, replay_buffer, online_model):
    """根据 `--num-envs` 创建单进程或多进程 rollout collector。"""

    if args.num_envs <= 1:
        return RolloutCollector(
            env=DaxiguaEnv(config=env_config),
            graph_builder=GraphBuilder(),
            replay_buffer=replay_buffer,
            model=online_model,
            seed=args.seed + 2,
        )

    return ParallelRolloutCollector(
        worker_count=args.num_envs,
        env_config=env_config,
        replay_buffer=replay_buffer,
        model_config=build_model_config(args),
        model=online_model,
        seed=args.seed + 2,
    )


def build_replay_buffer(args, run_dir):
    """根据训练参数创建 replay buffer。

    小容量训练会自动退化为纯内存模式；大容量训练默认把最近 10000 条作为热
    数据，其余旧数据写入 `run_dir/replay_cold`，降低长期内存占用。
    """

    hot_capacity = args.hot_replay_capacity
    if hot_capacity is None:
        hot_capacity = min(10_000, int(args.replay_capacity))
    hot_capacity = min(int(hot_capacity), int(args.replay_capacity))

    cold_dir = None
    if hot_capacity < int(args.replay_capacity):
        cold_dir = Path(args.replay_cold_dir) if args.replay_cold_dir else run_dir / 'replay_cold'

    return ReplayBuffer(
        capacity=args.replay_capacity,
        seed=args.seed + 1,
        hot_capacity=hot_capacity,
        cold_dir=cold_dir,
        segment_size=args.replay_segment_size,
        cold_cache_size=args.replay_cold_cache_size,
        cold_sample_ratio=args.replay_cold_sample_ratio,
        cold_cache_refresh_interval=args.replay_cold_cache_refresh_interval,
    )


SMOOTH_EPSILON_ANCHORS = (
    # (训练进度, 已完成衰减比例)。默认 start=1.0/end=0.05 时大致对应：
    # 0% -> 1.00, 30% -> 0.50, 50% -> 0.20, 70% -> 0.07, 80% -> 0.05。
    (0.0, 0.0),
    (0.30, 0.5263157894736842),
    (0.50, 0.8421052631578948),
    (0.70, 0.9789473684210527),
    (0.80, 1.0),
    (1.0, 1.0),
)


def scheduled_epsilon(update_step, env_steps, args):
    """根据当前配置计算 epsilon。"""

    if args.epsilon_schedule == 'linear':
        return linear_epsilon(env_steps, args)

    progress = _bounded_unit(float(update_step) / float(args.total_updates))
    return smooth_epsilon(progress, args)


def linear_epsilon(env_steps, args):
    """按环境步数线性衰减 epsilon。"""

    progress = _bounded_unit(float(env_steps) / float(args.epsilon_decay_steps))
    return args.epsilon_start + progress * (args.epsilon_end - args.epsilon_start)


def smooth_epsilon(progress, args):
    """按训练进度平滑衰减 epsilon。"""

    progress = _bounded_unit(progress)
    if args.epsilon_start == args.epsilon_end:
        return float(args.epsilon_start)

    for anchor_index in range(len(SMOOTH_EPSILON_ANCHORS) - 1):
        left_progress, left_fraction = SMOOTH_EPSILON_ANCHORS[anchor_index]
        right_progress, right_fraction = SMOOTH_EPSILON_ANCHORS[anchor_index + 1]
        if progress <= right_progress:
            local_progress = 0.0
            if right_progress > left_progress:
                local_progress = (progress - left_progress) / (right_progress - left_progress)
            smooth_progress = _smoothstep(_bounded_unit(local_progress))
            decay_fraction = left_fraction + smooth_progress * (right_fraction - left_fraction)
            return args.epsilon_start + decay_fraction * (args.epsilon_end - args.epsilon_start)

    return float(args.epsilon_end)


def _smoothstep(value):
    """返回三次 smoothstep 插值值，保证分段内部变化更平滑。"""

    value = _bounded_unit(value)
    return value * value * (3.0 - 2.0 * value)


def _bounded_unit(value):
    """把数值限制在 [0, 1]。"""

    return min(1.0, max(0.0, float(value)))


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


class EpisodeLogger:
    """按 episode 结束事件记录单局训练得分。"""

    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.rows = []
        self._episode_index = 0
        self._file = self.csv_path.open('w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_METRIC_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def log_collect_stats(self, collect_stats, phase, update_step, start_env_steps, epsilon):
        """把一次 collect 中结束的 episode 逐条写入 CSV。"""

        count = 0
        episode_data = zip(
            collect_stats.episode_scores,
            collect_stats.episode_rewards,
            collect_stats.episode_lengths,
            collect_stats.episode_end_offsets,
            collect_stats.episode_terminated_flags,
            collect_stats.episode_truncated_flags,
        )
        for score, reward, length, end_offset, terminated, truncated in episode_data:
            self._episode_index += 1
            row = {
                'episode_index': self._episode_index,
                'phase': phase,
                'update_step': int(update_step),
                'env_steps': int(start_env_steps + end_offset),
                'epsilon': float(epsilon),
                'score': float(score),
                'episode_reward': float(reward),
                'episode_length': int(length),
                'terminated': int(bool(terminated)),
                'truncated': int(bool(truncated)),
            }
            self.rows.append(row)
            self._writer.writerow(row)
            count += 1

        if count:
            self._file.flush()
        return count

    def close(self):
        """关闭 CSV 文件。"""

        self._file.close()


class CollectStatsWindow:
    """把多次 collect 统计合并成一个日志窗口。

    训练通常是每次 update 只采集 1 个环境 step，但 `metrics.csv` 可能每 100 次
    update 才写一行。如果直接记录最后 1 个 step，reward breakdown 曲线会非常
    抖动；窗口汇总能让每行日志代表最近一段训练过程的平均奖励组成。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """清空当前窗口，等待下一段 collect 统计写入。"""

        self.steps = 0
        self.total_reward = 0.0
        self.episodes = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.episode_end_offsets = []
        self.episode_terminated_flags = []
        self.episode_truncated_flags = []
        self.terminated_episodes = 0
        self.truncated_episodes = 0
        self.random_actions = 0
        self.greedy_actions = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.collect_seconds = 0.0
        self.graph_build_seconds = 0.0
        self.tensor_convert_seconds = 0.0
        self.action_select_seconds = 0.0
        self.env_step_seconds = 0.0
        self.physics_frames_total = 0
        self.fruit_count_total = 0
        self.graph_node_count_total = 0
        self.graph_edge_count_total = 0
        self.graph_cache_hits = 0
        self.graph_cache_misses = 0
        self.reward_breakdown_totals = {
            field_name: 0.0
            for field_name in REWARD_BREAKDOWN_FIELDS
        }

    def add(self, stats):
        """把一次 `RolloutCollector.collect_steps()` 的结果并入窗口。"""

        step_offset = self.steps
        self.steps += stats.steps
        self.total_reward += stats.total_reward
        self.episodes += stats.episodes
        self.episode_rewards.extend(stats.episode_rewards)
        self.episode_lengths.extend(stats.episode_lengths)
        self.episode_scores.extend(stats.episode_scores)
        self.episode_end_offsets.extend(
            step_offset + offset
            for offset in stats.episode_end_offsets
        )
        self.episode_terminated_flags.extend(stats.episode_terminated_flags)
        self.episode_truncated_flags.extend(stats.episode_truncated_flags)
        self.terminated_episodes += stats.terminated_episodes
        self.truncated_episodes += stats.truncated_episodes
        self.random_actions += stats.random_actions
        self.greedy_actions += stats.greedy_actions
        self.current_episode_reward = stats.current_episode_reward
        self.current_episode_length = stats.current_episode_length
        self.collect_seconds += getattr(stats, 'collect_seconds', 0.0)
        self.graph_build_seconds += getattr(stats, 'graph_build_seconds', 0.0)
        self.tensor_convert_seconds += getattr(stats, 'tensor_convert_seconds', 0.0)
        self.action_select_seconds += getattr(stats, 'action_select_seconds', 0.0)
        self.env_step_seconds += getattr(stats, 'env_step_seconds', 0.0)
        self.physics_frames_total += getattr(stats, 'physics_frames_total', 0)
        self.fruit_count_total += getattr(stats, 'fruit_count_total', 0)
        self.graph_node_count_total += getattr(stats, 'graph_node_count_total', 0)
        self.graph_edge_count_total += getattr(stats, 'graph_edge_count_total', 0)
        self.graph_cache_hits += getattr(stats, 'graph_cache_hits', 0)
        self.graph_cache_misses += getattr(stats, 'graph_cache_misses', 0)

        totals = stats.reward_breakdown_totals_dict
        for field_name in REWARD_BREAKDOWN_FIELDS:
            self.reward_breakdown_totals[field_name] += float(totals.get(field_name, 0.0))

    def to_rollout_stats(self, buffer_size):
        """转换成和 collector 输出兼容的 `RolloutStats`，供日志代码复用。"""

        return RolloutStats(
            steps=self.steps,
            episodes=self.episodes,
            total_reward=self.total_reward,
            reward_breakdown_totals=tuple(
                (field_name, self.reward_breakdown_totals[field_name])
                for field_name in REWARD_BREAKDOWN_FIELDS
            ),
            episode_rewards=tuple(self.episode_rewards),
            episode_lengths=tuple(self.episode_lengths),
            episode_scores=tuple(self.episode_scores),
            episode_end_offsets=tuple(self.episode_end_offsets),
            episode_terminated_flags=tuple(self.episode_terminated_flags),
            episode_truncated_flags=tuple(self.episode_truncated_flags),
            terminated_episodes=self.terminated_episodes,
            truncated_episodes=self.truncated_episodes,
            random_actions=self.random_actions,
            greedy_actions=self.greedy_actions,
            buffer_size=buffer_size,
            current_episode_reward=self.current_episode_reward,
            current_episode_length=self.current_episode_length,
            collect_seconds=self.collect_seconds,
            graph_build_seconds=self.graph_build_seconds,
            tensor_convert_seconds=self.tensor_convert_seconds,
            action_select_seconds=self.action_select_seconds,
            env_step_seconds=self.env_step_seconds,
            physics_frames_total=self.physics_frames_total,
            fruit_count_total=self.fruit_count_total,
            graph_node_count_total=self.graph_node_count_total,
            graph_edge_count_total=self.graph_edge_count_total,
            graph_cache_hits=self.graph_cache_hits,
            graph_cache_misses=self.graph_cache_misses,
        )


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
        step_checkpoint=False,
        extra_checkpoint_name=None):
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

    if extra_checkpoint_name:
        extra_path = checkpoint_dir / extra_checkpoint_name
        torch.save(checkpoint, extra_path)


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
        'eval_score_max': max(episode_scores) if episode_scores else 0.0,
        'eval_score_min': min(episode_scores) if episode_scores else 0.0,
        'eval_reward_mean': _mean(episode_rewards),
        'eval_length_mean': _mean(episode_lengths),
        'eval_episodes': len(episode_scores),
    }


def maybe_plot_metrics(run_dir, rows, episode_rows=None):
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
    episode_rows = episode_rows or []
    _plot_one(
        axes[1],
        _episode_series(episode_rows, 'update_step'),
        _episode_series(episode_rows, 'score'),
        'train episode',
        'Episode score',
    )
    _plot_one(axes[1], x, _series(rows, 'collect_mean_episode_score'), 'train mean', 'Episode score')
    _plot_one(axes[1], x, _series(rows, 'eval_score_mean'), 'eval mean', 'Episode score')
    _plot_one(axes[1], x, _series(rows, 'eval_score_max'), 'eval max', 'Episode score')
    _plot_one(axes[1], x, _series(rows, 'best_eval_score'), 'best eval', 'Episode score')
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
    _maybe_plot_reward_breakdown(run_dir, rows, x, plt)
    return True


def _maybe_plot_reward_breakdown(run_dir, rows, x, plt):
    """生成独立的 reward breakdown 曲线图。"""

    reward_fields = tuple(
        metric_field
        for _reward_field, metric_field in REWARD_BREAKDOWN_METRIC_FIELDS
    )
    if not _has_any_points(rows, reward_fields):
        return False

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    _plot_one(
        axes[0],
        x,
        _series(rows, 'collect_mean_reward_total'),
        'total',
        'Reward total and score component',
    )
    _plot_one(
        axes[0],
        x,
        _series(rows, 'collect_mean_score_reward'),
        'score',
        'Reward total and score component',
    )

    _plot_one(
        axes[1],
        x,
        _series(rows, 'collect_mean_survival_bonus'),
        'survival',
        'Reward shaping components',
    )
    _plot_one(
        axes[1],
        x,
        _series(rows, 'collect_mean_height_delta_reward'),
        'height delta',
        'Reward shaping components',
    )
    _plot_one(
        axes[1],
        x,
        _series(rows, 'collect_mean_danger_penalty'),
        'danger',
        'Reward shaping components',
    )
    _plot_one(
        axes[1],
        x,
        _series(rows, 'collect_mean_terminal_penalty'),
        'terminal',
        'Reward shaping components',
    )

    _plot_one(
        axes[2],
        x,
        _series(rows, 'collect_mean_previous_height_ratio'),
        'previous',
        'Height ratios used by reward',
    )
    _plot_one(
        axes[2],
        x,
        _series(rows, 'collect_mean_next_height_ratio'),
        'next',
        'Height ratios used by reward',
    )
    _plot_one(
        axes[2],
        x,
        _series(rows, 'collect_mean_height_delta_ratio'),
        'delta',
        'Height ratios used by reward',
    )

    for axis in axes:
        axis.set_xlabel('update')
        axis.grid(True, alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc='best')

    output_path = run_dir / 'plots' / 'reward_breakdown_curves.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def _has_any_points(rows, fields):
    """判断指定字段中是否至少存在一个可绘制的数值。"""

    for row in rows:
        for field in fields:
            if row.get(field, '') not in ('', None):
                return True
    return False


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


def _episode_series(rows, field):
    """从 episode metrics rows 中取一列浮点序列。"""

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
    if row.get('eval_score_max') not in ('', None):
        parts.append(f"eval_max={float(row['eval_score_max']):.1f}")
    if row.get('best_eval_score') not in ('', None):
        parts.append(f"best_eval={float(row['best_eval_score']):.1f}")
    if row.get('collect_mean_reward_total') not in ('', None):
        parts.append(f"r_total={float(row['collect_mean_reward_total']):+.3f}")
        parts.append(f"r_score={float(row['collect_mean_score_reward']):+.3f}")
        parts.append(f"r_danger={float(row['collect_mean_danger_penalty']):+.3f}")
    if row.get('collect_seconds') not in ('', None):
        parts.append(f"collect={_format_ms(row['collect_seconds'])}(采集)")
        parts.append(f"env={_format_ms(row['collect_env_step_seconds'])}(环境)")
        parts.append(f"graph={_format_ms(row['collect_graph_build_seconds'])}(构图)")
        parts.append(f"train={_format_ms(row['train_step_seconds'])}(训练)")
        parts.append(f"sample={_format_ms(row['replay_sample_seconds'])}(采样)")
        parts.append(f"frames={float(row['collect_mean_physics_frames']):.1f}(物理帧)")
        parts.append(f"nodes={float(row['collect_mean_graph_nodes']):.1f}(节点)")
        parts.append(f"edges={float(row['collect_mean_graph_edges']):.1f}(边)")

    print(' | '.join(parts), flush=True)


def _format_ms(value):
    """把秒格式化成毫秒字符串。"""

    if value in ('', None):
        return ''
    return f'{float(value) * 1000.0:.1f}ms'


def should_sync_parallel_workers(update_step, worker_sync_interval):
    """判断当前 update 采集前是否需要同步并行 worker 模型。"""

    return update_step == 1 or (update_step - 1) % int(worker_sync_interval) == 0


def should_sync_parallel_workers_after_train(update_step, worker_sync_interval):
    """判断当前 update 训练后是否需要先同步 worker，再提交下一批异步采集。"""

    return update_step % int(worker_sync_interval) == 0


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
    update_speed = 0.0 if elapsed <= 0.0 else current / elapsed
    remaining_updates = max(0.0, total - current)
    eta_seconds = 0.0 if update_speed <= 0.0 else remaining_updates / update_speed
    parts = [
        '[progress 进度]',
        f'phase={phase} 阶段={phase}',
        f'{current}/{total}',
        f'{percent:.1f}%',
        f'env_steps={env_steps} 投放={env_steps}',
        f'buffer={buffer_size} 经验池={buffer_size}',
        f'eps={epsilon:.3f}',
        f'speed={speed:.2f} env_steps/s 投放/秒={speed:.2f}',
        f'eta={eta_seconds / 60.0:.1f}min 预计剩余={eta_seconds / 60.0:.1f}分钟',
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
        best_eval_score,
        best_eval_update,
        timing,
        replay_stats=None):
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
    replay_stats = replay_stats or {}
    graph_cache_total = collect_stats.graph_cache_hits + collect_stats.graph_cache_misses
    graph_cache_hit_rate = (
        collect_stats.graph_cache_hits / graph_cache_total
        if graph_cache_total > 0
        else ''
    )

    row = {
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
        'collect_seconds': collect_stats.collect_seconds,
        'collect_graph_build_seconds': collect_stats.graph_build_seconds,
        'collect_tensor_convert_seconds': collect_stats.tensor_convert_seconds,
        'collect_action_select_seconds': collect_stats.action_select_seconds,
        'collect_env_step_seconds': collect_stats.env_step_seconds,
        'collect_mean_physics_frames': collect_stats.mean_physics_frames,
        'collect_mean_fruit_count': collect_stats.mean_fruit_count,
        'collect_mean_graph_nodes': collect_stats.mean_graph_nodes,
        'collect_mean_graph_edges': collect_stats.mean_graph_edges,
        'collect_graph_cache_hit_rate': graph_cache_hit_rate,
        'train_step_seconds': getattr(train_stats, 'train_step_seconds', ''),
        'replay_sample_seconds': getattr(train_stats, 'sample_seconds', ''),
        'current_collate_seconds': getattr(train_stats, 'current_collate_seconds', ''),
        'online_forward_seconds': getattr(train_stats, 'online_forward_seconds', ''),
        'target_compute_seconds': getattr(train_stats, 'target_compute_seconds', ''),
        'backward_seconds': getattr(train_stats, 'backward_seconds', ''),
        'optimizer_seconds': getattr(train_stats, 'optimizer_seconds', ''),
        'eval_seconds': timing.get('eval_seconds', ''),
        'save_seconds': timing.get('save_seconds', ''),
        'plot_seconds': timing.get('plot_seconds', ''),
        'replay_mode': replay_stats.get('mode', ''),
        'replay_hot_count': replay_stats.get('hot_count', ''),
        'replay_cold_count': replay_stats.get('cold_count', ''),
        'replay_pending_cold_count': replay_stats.get('pending_cold_count', ''),
        'replay_cold_segments': replay_stats.get('cold_segment_count', ''),
        'replay_cold_cache_count': replay_stats.get('cold_cache_count', ''),
        'random_actions': collect_stats.random_actions,
        'greedy_actions': collect_stats.greedy_actions,
        'eval_score_mean': eval_stats.get('eval_score_mean', '') if eval_stats else '',
        'eval_score_max': eval_stats.get('eval_score_max', '') if eval_stats else '',
        'eval_score_min': eval_stats.get('eval_score_min', '') if eval_stats else '',
        'eval_reward_mean': eval_stats.get('eval_reward_mean', '') if eval_stats else '',
        'eval_length_mean': eval_stats.get('eval_length_mean', '') if eval_stats else '',
        'eval_episodes': eval_stats.get('eval_episodes', '') if eval_stats else '',
        'best_eval_score': best_eval_score if best_eval_update else '',
        'best_eval_update': best_eval_update if best_eval_update else '',
        'updates_per_second': update_step / elapsed,
        'env_steps_per_second': env_steps / elapsed,
    }

    for reward_field, metric_field in REWARD_BREAKDOWN_METRIC_FIELDS:
        row[metric_field] = (
            collect_stats.mean_reward_breakdown(reward_field)
            if collect_stats.steps > 0
            else ''
        )

    return row


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
    replay_buffer = build_replay_buffer(args, run_dir)

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
    collector = build_collector(args, env_config, replay_buffer, online_model)

    metrics = MetricLogger(run_dir / 'metrics.csv')
    episode_metrics = EpisodeLogger(run_dir / 'episode_metrics.csv')
    env_steps = 0
    latest_row = None
    best_eval_score = float('-inf')
    best_eval_update = 0
    metric_window = CollectStatsWindow()

    print(f'run_dir={run_dir}', flush=True)
    print(f'device={device} matplotlib_output={run_dir / "plots" / "training_curves.png"}', flush=True)
    print(
        'physics_mode={} 物理模式={} | fps={} | max_frames={} 最大物理帧={} | '
        'stable_frames={} 稳定帧={} | iterations={} 迭代次数={}'.format(
            args.physics_mode,
            args.physics_mode,
            args.physics_fps,
            args.max_physics_frames,
            args.max_physics_frames,
            args.stable_frames,
            args.stable_frames,
            args.space_iterations,
            args.space_iterations,
        ),
        flush=True,
    )
    print(f'replay_storage={replay_buffer.storage_stats}', flush=True)
    print(
        f'collector={"parallel" if isinstance(collector, ParallelRolloutCollector) else "single"} '
        f'num_envs={args.num_envs} async_rollout={int(bool(args.async_rollout))}',
        flush=True,
    )
    if isinstance(collector, ParallelRolloutCollector) and args.collect_per_update < args.num_envs:
        print(
            'warning=collect_per_update 小于 num_envs，单次采样不会用满所有 worker；'
            '建议把 --collect-per-update 调到 num_envs 的整数倍。',
            flush=True,
        )
    print(f'warmup_steps={args.warmup_steps}', flush=True)

    start_time = time.perf_counter()
    last_progress_at = start_time
    warmup_done = 0
    warmup_total_reward = 0.0
    warmup_chunk_size = max(1, min(100, args.warmup_steps))

    while warmup_done < args.warmup_steps:
        chunk_size = min(warmup_chunk_size, args.warmup_steps - warmup_done)
        chunk_start_env_steps = env_steps
        warmup_stats = collector.collect_steps(chunk_size, epsilon=1.0)
        warmup_done += warmup_stats.steps
        env_steps += warmup_stats.steps
        warmup_total_reward += warmup_stats.total_reward
        episode_metrics.log_collect_stats(
            warmup_stats,
            phase='warmup',
            update_step=0,
            start_env_steps=chunk_start_env_steps,
            epsilon=1.0,
        )
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

    pending_collect = None
    try:
        for update_step in range(1, args.total_updates + 1):
            epsilon = scheduled_epsilon(update_step, env_steps, args)

            # 收集训练数据
            collect_start_env_steps = env_steps
            if isinstance(collector, ParallelRolloutCollector) and args.async_rollout:
                if pending_collect is None:
                    if should_sync_parallel_workers(update_step, args.worker_sync_interval):
                        collector.sync_model(online_model)
                    pending_collect = collector.start_collect_steps(
                        args.collect_per_update,
                        epsilon=epsilon,
                    )
                collect_stats = collector.finish_collect_steps(pending_collect)
                pending_collect = None
            else:
                if (
                        isinstance(collector, ParallelRolloutCollector)
                        and should_sync_parallel_workers(update_step, args.worker_sync_interval)):
                    collector.sync_model(online_model)
                collect_stats = collector.collect_steps(args.collect_per_update, epsilon=epsilon)
            env_steps += collect_stats.steps
            metric_window.add(collect_stats)
            episode_metrics.log_collect_stats(
                collect_stats,
                phase='train',
                update_step=update_step,
                start_env_steps=collect_start_env_steps,
                epsilon=epsilon,
            )

            # 异步采样路径会在当前 train_step 前提交下一轮 collect，让 CPU 物理模拟
            # 尽量和主进程的模型反向传播重叠。DQN 是 off-policy 算法，worker 使用
            # 间隔同步的稍旧 online model 做行为策略是可接受的。
            pending_next_collect = None
            next_epsilon = None
            if (
                    isinstance(collector, ParallelRolloutCollector)
                    and args.async_rollout
                    and update_step < args.total_updates):
                next_epsilon = scheduled_epsilon(update_step + 1, env_steps, args)
                if not should_sync_parallel_workers_after_train(update_step, args.worker_sync_interval):
                    pending_next_collect = collector.start_collect_steps(
                        args.collect_per_update,
                        epsilon=next_epsilon,
                    )

            # 执行一次 DQN 参数更新
            train_stats = trainer.train_step()

            if (
                    isinstance(collector, ParallelRolloutCollector)
                    and args.async_rollout
                    and update_step < args.total_updates):
                if pending_next_collect is None:
                    collector.sync_model(online_model)
                    pending_collect = collector.start_collect_steps(
                        args.collect_per_update,
                        epsilon=next_epsilon,
                    )
                else:
                    pending_collect = pending_next_collect

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

            # 记录指标、打印日志、评估、保存 checkpoint 和绘图
            should_log = update_step % args.log_interval == 0 or update_step == 1
            should_eval = args.eval_interval > 0 and update_step % args.eval_interval == 0
            should_save = args.save_interval > 0 and update_step % args.save_interval == 0
            should_plot = args.plot_interval > 0 and update_step % args.plot_interval == 0

            eval_stats = None
            eval_seconds = ''
            best_updated = False
            if should_eval:
                eval_start = time.perf_counter()
                eval_stats = evaluate_policy(online_model, args, device)
                eval_seconds = time.perf_counter() - eval_start
                if eval_stats['eval_score_max'] > best_eval_score:
                    best_eval_score = eval_stats['eval_score_max']
                    best_eval_update = update_step
                    best_updated = True

            if should_log or should_eval or should_save or should_plot or update_step == args.total_updates:
                # metrics.csv 中的 collect_* 字段代表“距离上一行日志以来”的窗口平均，
                # 比只记录最后一次投放更适合观察 reward breakdown 的趋势。
                logged_collect_stats = metric_window.to_rollout_stats(buffer_size=len(replay_buffer))
                latest_row = build_metric_row(
                    update_step=update_step,
                    env_steps=env_steps,
                    epsilon=epsilon,
                    train_stats=train_stats,
                    collect_stats=logged_collect_stats,
                    eval_stats=eval_stats,
                    best_eval_score=best_eval_score,
                    best_eval_update=best_eval_update,
                    timing={
                        'elapsed': time.perf_counter() - start_time,
                        'eval_seconds': eval_seconds,
                    },
                    replay_stats=replay_buffer.storage_stats,
                )
                save_seconds = 0.0
                if should_save:
                    save_start = time.perf_counter()
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
                    save_seconds += time.perf_counter() - save_start

                if best_updated:
                    save_start = time.perf_counter()
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
                        extra_checkpoint_name='best.pt',
                    )
                    save_seconds += time.perf_counter() - save_start
                latest_row['save_seconds'] = save_seconds if save_seconds else ''

                if should_plot:
                    plot_start = time.perf_counter()
                    row_for_plot = {field: latest_row.get(field, '') for field in METRIC_FIELDS}
                    maybe_plot_metrics(run_dir, metrics.rows + [row_for_plot], episode_metrics.rows)
                    latest_row['plot_seconds'] = time.perf_counter() - plot_start

                metrics.log(latest_row)
                metric_window.reset()

                if should_log or should_eval:
                    print_log(latest_row)

        final_epsilon = scheduled_epsilon(args.total_updates, env_steps, args)
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
        maybe_plot_metrics(run_dir, metrics.rows, episode_metrics.rows)
    finally:
        replay_buffer.flush()
        if isinstance(collector, ParallelRolloutCollector):
            collector.close()
        metrics.close()
        episode_metrics.close()

    print(f'training finished | run_dir={run_dir} | env_steps={env_steps}', flush=True)
    return run_dir


def main():
    """命令行入口。"""

    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
