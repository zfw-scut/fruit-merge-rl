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
import gc
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import pymunk

from daxigua.config import (
    DEFAULT_WINDOW_SIZE,
    FPS,
    LEGACY_SPAWN_LINE_Y,
    LEGACY_WINDOW_SIZE,
    SPAWN_LINE_Y,
)
from daxigua_rl import DaxiguaEnv, DaxiguaEnvConfig, GraphBuilder, ReplayBuffer
from daxigua_rl.attribution import ANALYSIS_ACTION_COUNT
from daxigua_rl.attribution.causal_replay import CausalReplayBuffer
from daxigua_rl.attribution.counterfactual import (
    CounterfactualConfig,
    FrozenGNNModelConfig,
    LocalShapleyConfig,
)
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.reward import (
    REWARD_BREAKDOWN_FIELDS,
    RewardConfig,
    merge_utility,
)
from daxigua_rl.training import (
    DQNTrainer,
    DQNTrainerConfig,
    CounterfactualCoordinator,
    effective_cpu_count,
    ParallelRolloutCollector,
    RolloutCollector,
    RolloutStats,
    TransitionKey,
    recommended_counterfactual_worker_count,
)
from daxigua_rl.training.checkpointing import (
    DEFAULT_RESUME_MUTABLE_FIELDS,
    RunManifest,
    atomic_clone_file,
    atomic_torch_save,
    build_training_checkpoint,
    config_fingerprint,
    create_run_manifest,
    load_training_checkpoint,
    restore_rng_state,
)
from daxigua_rl.training.local_shapley_coordinator import (
    LocalShapleyCoordinator,
)


REWARD_BREAKDOWN_METRIC_FIELDS = (
    ('total', 'collect_mean_reward_total'),
    ('task_reward', 'collect_mean_task_reward'),
    ('potential_shaping_reward', 'collect_mean_potential_shaping_reward'),
    ('terminal_penalty', 'collect_mean_terminal_penalty'),
    ('previous_potential', 'collect_mean_previous_potential'),
    ('next_potential', 'collect_mean_next_potential'),
    ('potential_delta', 'collect_mean_potential_delta'),
    (
        'previous_top_connected_capacity',
        'collect_mean_previous_top_connected_capacity',
    ),
    (
        'next_top_connected_capacity',
        'collect_mean_next_top_connected_capacity',
    ),
    ('previous_recoverability', 'collect_mean_previous_recoverability'),
    ('next_recoverability', 'collect_mean_next_recoverability'),
    ('previous_chain_readiness', 'collect_mean_previous_chain_readiness'),
    ('next_chain_readiness', 'collect_mean_next_chain_readiness'),
    ('merge_event_count', 'collect_mean_merge_event_count'),
)


METRIC_FIELDS = (
    'update_step',
    'env_steps',
    'epsilon',
    'buffer_size',
    'loss',
    'td_loss',
    'rule_rank_loss',
    'weighted_rule_rank_loss',
    'counterfactual_loss',
    'weighted_counterfactual_loss',
    'structural_loss',
    'weighted_structural_loss',
    'structural_valid_count',
    'structural_sample_count',
    'structural_mean_abs_error',
    'causal_update_applied',
    'causal_batch_size',
    'rule_batch_size',
    'counterfactual_batch_size',
    'shapley_batch_size',
    'rule_pair_accuracy',
    'rule_margin_satisfaction_rate',
    'counterfactual_sign_accuracy',
    'counterfactual_mean_abs_error',
    'mean_q',
    'mean_target',
    'mean_reward',
    'mean_abs_td_error',
    'bootstrap_count',
    'grad_norm',
    'target_synced',
    'collect_steps',
    'collect_replay_transitions_emitted',
    'collect_n_step_pending_count',
    'collect_n_step_forced_flush_emitted',
    'collect_causal_samples_emitted',
    'collect_rule_causal_input_events',
    'collect_rule_causal_eligible_events',
    'collect_rule_causal_budget_count',
    'collect_rule_causal_skip_reasons',
    'collect_total_reward',
    'collect_episodes',
    'collect_mean_episode_reward',
    'collect_mean_episode_length',
    'collect_mean_episode_score',
    'collect_mean_reward_total',
    'collect_mean_task_reward',
    'collect_mean_potential_shaping_reward',
    'collect_mean_terminal_penalty',
    'collect_mean_previous_potential',
    'collect_mean_next_potential',
    'collect_mean_potential_delta',
    'collect_mean_previous_top_connected_capacity',
    'collect_mean_next_top_connected_capacity',
    'collect_mean_previous_recoverability',
    'collect_mean_next_recoverability',
    'collect_mean_previous_chain_readiness',
    'collect_mean_next_chain_readiness',
    'collect_mean_merge_event_count',
    'collect_p95_abs_potential_shaping_reward',
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
    'collect_state_analysis_calls',
    'collect_state_analysis_seconds',
    'collect_mean_state_analysis_seconds',
    'collect_state_analysis_cache_hits',
    'collect_state_analysis_cache_hit_rate',
    'collect_state_analysis_degraded_count',
    'collect_state_analysis_degraded_rate',
    'actor_inference_requests',
    'actor_inference_batches',
    'actor_inference_mean_batch_size',
    'actor_inference_max_batch',
    'actor_inference_seconds',
    'collect_attribution_tracker_calls',
    'collect_attribution_tracker_seconds',
    'collect_mean_attribution_tracker_seconds',
    'collect_attribution_events_created',
    'collect_attribution_events_confirmed',
    'collect_attribution_events_cancelled',
    'collect_attribution_events_interrupted',
    'collect_attribution_pending_event_count',
    'collect_attribution_lineage_merge_count',
    'collect_attribution_chain_merge_count',
    'collect_attribution_max_chain_depth',
    'collect_mean_attribution_delay',
    'collect_p95_attribution_delay',
    'collect_attribution_event_status_counts',
    'collect_attribution_confidence_tier_counts',
    'collect_merge_level_counts',
    'collect_max_fruit_level',
    'collect_counterfactual_snapshot_calls',
    'collect_counterfactual_snapshot_seconds',
    'collect_counterfactual_snapshot_failures',
    'collect_counterfactual_history_evictions',
    'collect_counterfactual_history_size',
    'collect_counterfactual_proposal_build_calls',
    'collect_counterfactual_proposal_build_seconds',
    'collect_counterfactual_proposal_input_events',
    'collect_counterfactual_proposal_confirmed_events',
    'collect_counterfactual_proposal_budget_count',
    'collect_counterfactual_proposals_generated',
    'collect_counterfactual_proposals_transfer_selected',
    'collect_counterfactual_proposals_transfer_throttled',
    'collect_counterfactual_proposal_skip_reasons',
    'collect_counterfactual_proposals_serialized',
    'collect_counterfactual_proposal_serialized_bytes',
    'counterfactual_enabled',
    'counterfactual_worker_count',
    'counterfactual_proposals_received',
    'counterfactual_proposals_admitted',
    'counterfactual_proposals_rejected',
    'counterfactual_pending_tasks',
    'counterfactual_admission_slots_used',
    'counterfactual_admission_slots_available',
    'counterfactual_candidate_pool_capacity',
    'counterfactual_candidate_pool_count',
    'counterfactual_candidate_offers',
    'counterfactual_candidate_pool_evictions',
    'counterfactual_candidate_dispatch_attempts',
    'counterfactual_candidate_dispatch_admitted',
    'counterfactual_candidate_close_dropped',
    'counterfactual_results_completed',
    'counterfactual_results_partial',
    'counterfactual_results_failed',
    'counterfactual_reproduction_passed',
    'counterfactual_reproduction_failed',
    'counterfactual_numeric_jitter_dropped',
    'counterfactual_semantic_divergence_dropped',
    'counterfactual_numeric_jitter_max_merge_event_position_error',
    'counterfactual_numeric_jitter_max_fruit_position_error',
    'counterfactual_numeric_jitter_max_linear_velocity_error',
    'counterfactual_numeric_jitter_max_orientation_error',
    'counterfactual_numeric_jitter_max_angular_velocity_error',
    'counterfactual_label_ready_results',
    'counterfactual_samples_inserted',
    'counterfactual_tokens_reserved',
    'counterfactual_tokens_consumed',
    'counterfactual_tokens_refunded',
    'counterfactual_actual_token_ratio',
    'counterfactual_projected_token_ratio',
    'counterfactual_hard_budget_respected',
    'counterfactual_circuit_open',
    'counterfactual_drop_reasons',
    'counterfactual_failure_reasons',
    'counterfactual_failure_diagnostic_codes',
    'counterfactual_failure_trigger_reasons',
    'shapley_enabled',
    'shapley_events_observed',
    'shapley_events_selected',
    'shapley_tasks_submitted',
    'shapley_tasks_completed',
    'shapley_tasks_failed',
    'shapley_terminal_dropped',
    'shapley_reproduction_passed',
    'shapley_reproduction_failed',
    'shapley_numeric_jitter_dropped',
    'shapley_semantic_divergence_dropped',
    'shapley_numeric_jitter_max_merge_event_position_error',
    'shapley_numeric_jitter_max_fruit_position_error',
    'shapley_numeric_jitter_max_linear_velocity_error',
    'shapley_numeric_jitter_max_orientation_error',
    'shapley_numeric_jitter_max_angular_velocity_error',
    'shapley_samples_inserted',
    'shapley_tokens_consumed',
    'shapley_pending_count',
    'shapley_drop_reasons',
    'train_step_seconds',
    'replay_sample_seconds',
    'current_collate_seconds',
    'online_forward_seconds',
    'target_compute_seconds',
    'causal_sample_seconds',
    'causal_collate_seconds',
    'causal_forward_seconds',
    'backward_seconds',
    'optimizer_seconds',
    'eval_seconds',
    'save_seconds',
    'checkpoint_bytes',
    'checkpoint_pruned_count',
    'checkpoint_step_materialization',
    'checkpoint_extra_materialization',
    'plot_seconds',
    'replay_mode',
    'replay_hot_count',
    'replay_cold_count',
    'replay_pending_cold_count',
    'replay_cold_segments',
    'replay_cold_cache_count',
    'causal_replay_size',
    'causal_replay_positive_count',
    'causal_replay_negative_count',
    'causal_replay_counterfactual_count',
    'causal_replay_rule_count',
    'causal_replay_cf_count',
    'causal_replay_shapley_count',
    'causal_replay_cause_type_counts',
    'causal_rule_empirical_agreement_count',
    'causal_rule_empirical_disagreement_count',
    'causal_rule_empirical_agreement_rate',
    'causal_replay_shared_tensor_bytes',
    'causal_replay_saved_tensor_bytes',
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
    parser.add_argument('--n-step', type=int, default=3, help='rollout 顺序聚合的 n-step return 长度。')
    parser.add_argument('--target-update-interval', type=int, default=1_000, help='target network 同步间隔，按 train_step 计。')
    parser.add_argument('--grad-clip-norm', type=float, default=10.0, help='梯度裁剪阈值；传 0 表示关闭。')

    # 稀疏因果训练。规则与反事实样本使用独立纯内存 replay，不修改主 TD replay。
    parser.add_argument('--causal-replay-capacity', type=int, default=20_000, help='稀疏因果回放最大样本数。')
    parser.add_argument('--causal-batch-size', type=int, default=32, help='一次因果更新最多采样多少条动作对。')
    parser.add_argument('--causal-update-interval', type=int, default=2, help='每多少次 TD update 联合执行一次因果更新。')
    parser.add_argument('--lambda-rule', type=float, default=0.15, help='规则 Q 排序 loss 权重。')
    parser.add_argument(
        '--lambda-cf',
        '--lambda-counterfactual',
        dest='lambda_cf',
        type=float,
        default=0.10,
        help='反事实与局部 Shapley 差值 loss 权重。',
    )
    parser.add_argument(
        '--lambda-structural',
        type=float,
        default=0.15,
        help=(
            '动作后六维结构辅助预测 loss 权重；该监督不修改环境 reward。'
        ),
    )
    parser.add_argument(
        '--counterfactual-return-scale',
        type=float,
        default=merge_utility(7),
        help='反事实回报差值进入 Huber 前的归一化尺度。',
    )
    parser.add_argument('--counterfactual-target-clip', type=float, default=5.0, help='归一化反事实差值的绝对裁剪上限。')

    # 稀疏物理反事实。collector 只捕获稳定边界并生成 proposal；主进程把它们
    # 交给独立 CPU 进程，所有失败都退化为规则/TD 训练。
    parser.add_argument(
        '--counterfactual-enabled',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='启用稳定快照、预算调度和真实物理反事实回灌。',
    )
    parser.add_argument(
        '--counterfactual-workers',
        type=int,
        default=None,
        help='物理归因进程总预算（含 Shapley）；默认按 CPU/rollout 数计算。',
    )
    parser.add_argument('--counterfactual-horizon', type=int, default=10)
    parser.add_argument('--counterfactual-cost-ratio', type=float, default=0.08)
    parser.add_argument('--counterfactual-hard-limit', type=float, default=0.10)
    parser.add_argument(
        '--counterfactual-external-token-reserve-ratio',
        type=float,
        default=0.0,
        help=(
            '从共享硬预算中为局部 Shapley 等外部物理任务保留的比例；'
            '普通反事实不能借用该份额。'
        ),
    )
    parser.add_argument('--counterfactual-min-real-steps', type=int, default=256)
    parser.add_argument('--counterfactual-cpu-core-ratio', type=float, default=0.25)
    parser.add_argument('--counterfactual-queue-capacity', type=int, default=256)
    parser.add_argument('--counterfactual-snapshot-ring-size', type=int, default=32)
    parser.add_argument(
        '--counterfactual-proposal-sample-rate',
        type=float,
        default=1.0,
        help=(
            '常规物理 proposal 的确定性跨进程抽样率；'
            '高价值合成始终保留。'
        ),
    )
    parser.add_argument('--counterfactual-max-alternatives', type=int, default=3)
    parser.add_argument('--counterfactual-max-inflight-per-worker', type=int, default=2)
    parser.add_argument('--counterfactual-circuit-breaker-failures', type=int, default=5)
    parser.add_argument('--counterfactual-soft-borrow-priority', type=float, default=10.0)

    # 极稀疏局部 Shapley 与普通反事实共用同一个 10% 物理 token 硬账本。
    parser.add_argument(
        '--shapley-enabled',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='对极少量高价值协同事件执行局部 Shapley 物理归因。',
    )
    parser.add_argument('--shapley-event-ratio-max', type=float, default=0.0005)
    parser.add_argument('--shapley-candidate-limit', type=int, default=3)
    parser.add_argument('--shapley-paired-permutations', type=int, default=4)
    parser.add_argument('--shapley-minimum-candidates', type=int, default=2)
    parser.add_argument('--shapley-minimum-utility', type=float, default=merge_utility(7))
    parser.add_argument('--shapley-pending-capacity', type=int, default=4)

    # 模型规模。
    parser.add_argument('--hidden-dim', type=int, default=128, help='GNN 隐藏层维度。')
    parser.add_argument('--message-layers', type=int, default=3, help='GNN message passing 层数。')
    parser.add_argument('--dropout', type=float, default=0.0, help='GNN dropout。')
    parser.add_argument('--activation', choices=('relu', 'silu'), default='silu', help='GNN 激活函数。')

    # 环境参数。
    parser.add_argument('--seed', type=int, default=0, help='随机种子。')
    parser.add_argument(
        '--action-count',
        type=int,
        default=ANALYSIS_ACTION_COUNT,
        help='离散候选投放动作数量；当前完整归因规范固定为 21。',
    )
    parser.add_argument(
        '--board-width',
        type=int,
        default=DEFAULT_WINDOW_SIZE[0],
        help='headless 场地宽度；当前默认 560。',
    )
    parser.add_argument(
        '--board-height',
        type=int,
        default=DEFAULT_WINDOW_SIZE[1],
        help='headless 场地高度；当前默认 1120。',
    )
    parser.add_argument(
        '--spawn-y',
        type=int,
        default=SPAWN_LINE_Y,
        help='水果生成线/失败警戒线 y 坐标；当前默认 252。',
    )
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
    parser.add_argument(
        '--centralized-actor-inference',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            '把并行 worker 的 greedy Q 请求汇总到主进程模型设备做微批推理。'
        ),
    )
    parser.add_argument(
        '--actor-batch-size',
        type=int,
        default=16,
        help='集中式 actor 单次 GPU 推理最多聚合多少张状态图。',
    )
    parser.add_argument(
        '--actor-batch-wait-ms',
        type=float,
        default=2.0,
        help='集中式 actor 等待同批其他 worker 请求的最长毫秒数。',
    )
    parser.add_argument(
        '--actor-request-timeout-seconds',
        type=float,
        default=120.0,
        help='worker 等待集中式 actor 响应的超时时间。',
    )

    # Reward V2。gamma 直接复用上面的 DQN gamma，保证 potential shaping 和
    # TD / n-step return 使用同一个折扣因子。
    parser.add_argument('--lambda-phi', type=float, default=0.5, help='状态 potential shaping 的总权重。')
    parser.add_argument('--capacity-weight', type=float, default=0.6, help='顶部可达投放容量 C(s) 的 potential 权重。')
    parser.add_argument('--recoverability-weight', type=float, default=0.3, help='水果可恢复性 R(s) 的 potential 权重。')
    parser.add_argument('--chain-readiness-weight', type=float, default=0.1, help='连锁就绪度 K(s) 的 potential 权重。')
    parser.add_argument('--terminal-penalty', type=float, default=0.0, help='真实终局的可选额外惩罚；Reward V2 默认关闭。')

    # 日志、保存、评估和可视化。
    parser.add_argument('--run-dir', default=None, help='训练输出目录；默认 runs/dqn_YYYYMMDD_HHMMSS。')
    parser.add_argument(
        '--resume',
        default=None,
        help='从可信 checkpoint 恢复；默认继续写 checkpoint 所属 run 目录。',
    )
    parser.add_argument(
        '--init-checkpoint',
        default=None,
        help=(
            '从可信版本化 checkpoint 只加载 online 模型权重并开始一个新 run；'
            '不恢复 optimizer、replay、RNG 或训练步数，适合尺寸迁移。'
        ),
    )
    parser.add_argument(
        '--overwrite-run-dir',
        action='store_true',
        help='显式允许清空一个已有但非 resume 的训练输出目录。',
    )
    parser.add_argument('--log-interval', type=int, default=500, help='每多少次 update 记录并打印一次日志。')
    parser.add_argument('--save-interval', type=int, default=20_000, help='每多少次 update 保存一次 step checkpoint；0 表示关闭周期保存。')
    parser.add_argument(
        '--checkpoint-keep-last',
        type=int,
        default=3,
        help='只保留最近多少个 step checkpoint；0 表示全部保留。',
    )
    parser.add_argument('--eval-interval', type=int, default=20_000, help='每多少次 update 执行一次 greedy 评估；0 表示关闭。')
    parser.add_argument('--eval-episodes', type=int, default=10, help='每次评估跑多少局。')
    parser.add_argument('--eval-max-steps', type=int, default=500, help='每局评估最多投放多少次，防止极端长局。')
    parser.add_argument('--plot-interval', type=int, default=10_000, help='每多少次 update 生成一次曲线图；0 表示只在结束时尝试生成。')
    parser.add_argument('--progress-interval', type=float, default=3.0, help='每多少秒打印一次轻量训练进度；0 表示关闭。')

    # 运行设备。
    parser.add_argument('--device', default='cpu', help='模型设备，例如 cpu、cuda 或 cuda:0。')

    return parser


def load_config_defaults(config_path, parser=None, _seen=None):
    """读取 TOML 配置文件，并转换成 argparse 默认值字典。"""

    try:
        import tomllib
    except ModuleNotFoundError as exc:
        raise RuntimeError('loading TOML config requires Python 3.11+ tomllib') from exc

    path = Path(config_path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f'cyclic training config extends: {path}')
    seen.add(path)
    with path.open('rb') as file_obj:
        config = tomllib.load(file_obj)

    if not isinstance(config, dict):
        raise ValueError('training config must be a TOML table')

    if parser is None:
        parser = build_arg_parser()
    allowed_keys = _parser_destinations(parser)
    defaults = {}
    extends = config.pop('extends', None)
    if extends is not None:
        if not isinstance(extends, str) or not extends.strip():
            raise ValueError('training config extends must be a path string')
        base_path = (path.parent / extends).resolve()
        defaults.update(load_config_defaults(
            base_path,
            parser=parser,
            _seen=seen,
        ))
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
        'n_step',
        'target_update_interval',
        'causal_replay_capacity',
        'causal_batch_size',
        'causal_update_interval',
        'hidden_dim',
        'message_layers',
        'action_count',
        'board_width',
        'board_height',
        'physics_fps',
        'max_physics_frames',
        'stable_frames',
        'space_iterations',
        'num_envs',
        'worker_sync_interval',
        'actor_batch_size',
        'log_interval',
        'eval_episodes',
        'eval_max_steps',
    )
    for field_name in positive_int_fields:
        if int(getattr(args, field_name)) <= 0:
            raise ValueError(f'--{field_name.replace("_", "-")} must be positive')
    if int(args.action_count) != ANALYSIS_ACTION_COUNT:
        raise ValueError(
            'full state attribution requires '
            f'--action-count {ANALYSIS_ACTION_COUNT}'
        )
    if int(args.spawn_y) < 0 or int(args.spawn_y) >= int(args.board_height):
        raise ValueError(
            '--spawn-y must be >= 0 and smaller than --board-height'
        )
    if args.resume and args.init_checkpoint:
        raise ValueError(
            '--resume and --init-checkpoint are mutually exclusive'
        )

    non_negative_intervals = (
        'save_interval',
        'eval_interval',
        'plot_interval',
        'checkpoint_keep_last',
    )
    for field_name in non_negative_intervals:
        if int(getattr(args, field_name)) < 0:
            raise ValueError(f'--{field_name.replace("_", "-")} must be >= 0')

    if args.epsilon_start < 0.0 or args.epsilon_start > 1.0:
        raise ValueError('--epsilon-start must be in [0, 1]')
    if args.epsilon_end < 0.0 or args.epsilon_end > 1.0:
        raise ValueError('--epsilon-end must be in [0, 1]')
    if args.learning_rate <= 0.0:
        raise ValueError('--learning-rate must be positive')
    if not math.isfinite(args.gamma) or args.gamma < 0.0 or args.gamma > 1.0:
        raise ValueError('--gamma must be in [0, 1]')
    if (
            not math.isfinite(args.lambda_phi)
            or args.lambda_phi < 0.0
            or args.lambda_phi > 1.0):
        raise ValueError('--lambda-phi must be in [0, 1]')
    potential_weights = (
        args.capacity_weight,
        args.recoverability_weight,
        args.chain_readiness_weight,
    )
    if any(
            not math.isfinite(weight) or weight < 0.0 or weight > 1.0
            for weight in potential_weights):
        raise ValueError('Reward V2 potential weights must each be in [0, 1]')
    if not math.isclose(
            sum(potential_weights),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9):
        raise ValueError('Reward V2 potential weights must sum to 1')
    if not math.isfinite(args.terminal_penalty) or args.terminal_penalty > 0.0:
        raise ValueError('--terminal-penalty must be finite and <= 0')
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
    if args.centralized_actor_inference and args.num_envs <= 1:
        raise ValueError(
            '--centralized-actor-inference requires --num-envs > 1'
        )
    if (
            not math.isfinite(args.actor_batch_wait_ms)
            or args.actor_batch_wait_ms < 0.0):
        raise ValueError(
            '--actor-batch-wait-ms must be finite and >= 0'
        )
    if (
            not math.isfinite(args.actor_request_timeout_seconds)
            or args.actor_request_timeout_seconds <= 0.0):
        raise ValueError(
            '--actor-request-timeout-seconds must be finite and positive'
        )
    for field_name in (
            'lambda_rule',
            'lambda_cf',
            'lambda_structural'):
        value = float(getattr(args, field_name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f'--{field_name.replace("_", "-")} must be finite and >= 0'
            )
    for field_name in (
            'counterfactual_return_scale',
            'counterfactual_target_clip'):
        value = float(getattr(args, field_name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f'--{field_name.replace("_", "-")} must be finite and positive'
            )
    if (
            args.counterfactual_workers is not None
            and int(args.counterfactual_workers) < 0):
        raise ValueError('--counterfactual-workers must be >= 0')
    if (
            not math.isfinite(args.counterfactual_proposal_sample_rate)
            or not 0.0 <= args.counterfactual_proposal_sample_rate <= 1.0):
        raise ValueError(
            '--counterfactual-proposal-sample-rate must be finite '
            'and in [0, 1]'
        )
    if args.shapley_enabled and not args.counterfactual_enabled:
        raise ValueError(
            '--shapley-enabled requires --counterfactual-enabled so both '
            'paths share one hard token budget'
        )
    if (
            args.lambda_cf > 0.0
            and not args.counterfactual_enabled
            and not args.shapley_enabled):
        raise ValueError(
            'positive --lambda-cf requires counterfactual or Shapley labels'
        )
    # Dataclass 自身执行 horizon、预算比例、候选数等跨字段约束。
    build_counterfactual_config(args)
    build_shapley_config(args)


def build_counterfactual_config(args):
    """把扁平 CLI/TOML 参数冻结为反事实调度配置。"""

    return CounterfactualConfig(
        horizon=args.counterfactual_horizon,
        cost_ratio=args.counterfactual_cost_ratio,
        cost_hard_limit=args.counterfactual_hard_limit,
        external_token_reserve_ratio=(
            args.counterfactual_external_token_reserve_ratio
        ),
        min_real_steps=args.counterfactual_min_real_steps,
        cpu_core_ratio=args.counterfactual_cpu_core_ratio,
        queue_capacity=args.counterfactual_queue_capacity,
        snapshot_ring_size=args.counterfactual_snapshot_ring_size,
        max_alternatives=args.counterfactual_max_alternatives,
        max_inflight_per_worker=(
            args.counterfactual_max_inflight_per_worker
        ),
        soft_budget_borrow_priority=(
            args.counterfactual_soft_borrow_priority
        ),
        circuit_breaker_failures=(
            args.counterfactual_circuit_breaker_failures
        ),
    )


def build_shapley_config(args):
    """冻结局部 Shapley 的累计稀疏配额。"""

    return LocalShapleyConfig(
        event_ratio_max=args.shapley_event_ratio_max,
        candidate_limit=args.shapley_candidate_limit,
        paired_permutations=args.shapley_paired_permutations,
        minimum_candidates=args.shapley_minimum_candidates,
        minimum_utility=args.shapley_minimum_utility,
    )


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


def create_run_dir(run_dir, *, resume=False, overwrite=False):
    """创建本次训练输出目录，并拒绝静默覆盖历史实验。"""

    if run_dir:
        path = Path(run_dir)
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = Path('runs') / f'dqn_{stamp}'

    if path.exists() and any(path.iterdir()):
        if resume:
            pass
        elif not overwrite:
            raise FileExistsError(
                f'run directory is not empty: {path}; use a new --run-dir, '
                '--resume, or explicitly pass --overwrite-run-dir'
            )
        else:
            backup = path.with_name(
                f'{path.name}.backup_'
                f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
            )
            path.replace(backup)
            print(
                f'warning=existing run directory moved to {backup}',
                flush=True,
            )

    path.mkdir(parents=True, exist_ok=True)
    (path / 'checkpoints').mkdir(exist_ok=True)
    (path / 'plots').mkdir(exist_ok=True)
    (path / 'mplconfig').mkdir(exist_ok=True)
    return path


def resolve_resume_location(args):
    """校验 checkpoint 所属 run，并在未指定时推导 ``run_dir``。"""

    if not args.resume:
        return None
    if args.overwrite_run_dir:
        raise ValueError(
            '--overwrite-run-dir cannot be combined with --resume'
        )
    checkpoint_path = Path(args.resume).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'resume checkpoint not found: {checkpoint_path}'
        )
    inferred_run_dir = (
        checkpoint_path.parent.parent
        if checkpoint_path.parent.name == 'checkpoints'
        else checkpoint_path.parent
    )
    if args.run_dir is None:
        args.run_dir = str(inferred_run_dir)
    else:
        requested_run_dir = Path(args.run_dir).expanduser().resolve()
        if requested_run_dir != inferred_run_dir:
            raise ValueError(
                'resume checkpoint does not belong to --run-dir: '
                f'checkpoint_run={inferred_run_dir} '
                f'requested_run={requested_run_dir}'
            )
    return checkpoint_path


def resolve_initialization_location(args):
    """校验 weights-only 初始化来源，且不把它解释为原 run 的 resume。"""

    if not args.init_checkpoint:
        return None
    if args.resume:
        raise ValueError(
            '--init-checkpoint cannot be combined with --resume'
        )
    checkpoint_path = Path(
        args.init_checkpoint
    ).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'initialization checkpoint not found: {checkpoint_path}'
        )
    return checkpoint_path


def set_random_seeds(seed):
    """设置 Python 和 PyTorch 随机种子。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env_config(args):
    """根据命令行参数创建环境配置。"""

    reward_defaults = RewardConfig()
    reward_config = RewardConfig(
        gamma=getattr(args, 'gamma', reward_defaults.gamma),
        lambda_phi=getattr(args, 'lambda_phi', reward_defaults.lambda_phi),
        capacity_weight=getattr(
            args,
            'capacity_weight',
            reward_defaults.capacity_weight,
        ),
        recoverability_weight=getattr(
            args,
            'recoverability_weight',
            reward_defaults.recoverability_weight,
        ),
        chain_readiness_weight=getattr(
            args,
            'chain_readiness_weight',
            reward_defaults.chain_readiness_weight,
        ),
        terminal_penalty=getattr(
            args,
            'terminal_penalty',
            reward_defaults.terminal_penalty,
        ),
    )
    return DaxiguaEnvConfig(
        board_width=getattr(
            args,
            'board_width',
            DEFAULT_WINDOW_SIZE[0],
        ),
        board_height=getattr(
            args,
            'board_height',
            DEFAULT_WINDOW_SIZE[1],
        ),
        spawn_y=getattr(args, 'spawn_y', SPAWN_LINE_Y),
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


def build_collector(
        args,
        env_config,
        replay_buffer,
        online_model,
        causal_replay_buffer=None,
        episode_id_start=0,
        policy_version=None):
    """根据 `--num-envs` 创建单进程或多进程 rollout collector。"""

    if args.num_envs <= 1:
        return RolloutCollector(
            env=DaxiguaEnv(config=env_config),
            graph_builder=GraphBuilder(),
            replay_buffer=replay_buffer,
            model=online_model,
            seed=args.seed + 2,
            n_step=args.n_step,
            gamma=args.gamma,
            policy_version=policy_version,
            causal_replay_buffer=causal_replay_buffer,
            counterfactual_enabled=bool(getattr(
                args,
                'counterfactual_enabled',
                False,
            )),
            counterfactual_ring_size=int(getattr(
                args,
                'counterfactual_snapshot_ring_size',
                32,
            )),
            counterfactual_proposal_sample_rate=float(getattr(
                args,
                'counterfactual_proposal_sample_rate',
                1.0,
            )),
            episode_id_start=episode_id_start,
        )

    return ParallelRolloutCollector(
        worker_count=args.num_envs,
        env_config=env_config,
        replay_buffer=replay_buffer,
        model_config=build_model_config(args),
        model=online_model,
        seed=args.seed + 2,
        n_step=args.n_step,
        gamma=args.gamma,
        policy_version=policy_version,
        causal_replay_buffer=causal_replay_buffer,
        counterfactual_enabled=bool(getattr(
            args,
            'counterfactual_enabled',
            False,
        )),
        counterfactual_ring_size=int(getattr(
            args,
            'counterfactual_snapshot_ring_size',
            32,
        )),
        counterfactual_proposal_sample_rate=float(getattr(
            args,
            'counterfactual_proposal_sample_rate',
            1.0,
        )),
        episode_id_start=episode_id_start,
        centralized_actor_inference=bool(getattr(
            args,
            'centralized_actor_inference',
            False,
        )),
        actor_batch_size=int(getattr(
            args,
            'actor_batch_size',
            16,
        )),
        actor_batch_wait_ms=float(getattr(
            args,
            'actor_batch_wait_ms',
            2.0,
        )),
        actor_request_timeout_seconds=float(getattr(
            args,
            'actor_request_timeout_seconds',
            120.0,
        )),
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


def build_causal_replay_buffer(args):
    """创建与主 TD replay 解耦的纯内存因果回放。"""

    return CausalReplayBuffer(
        capacity=args.causal_replay_capacity,
        seed=args.seed + 3,
    )


def build_counterfactual_coordinator(args, causal_replay_buffer):
    """创建共享预算协调器，并为极稀疏 Shapley 预留一个 CPU worker。"""

    if not args.counterfactual_enabled:
        return None, 0
    effective_cpus = effective_cpu_count()
    recommended = recommended_counterfactual_worker_count(
        cpu_count=effective_cpus,
        rollout_worker_count=args.num_envs,
        cpu_core_ratio=args.counterfactual_cpu_core_ratio,
    )
    total_workers = (
        recommended
        if args.counterfactual_workers is None
        else int(args.counterfactual_workers)
    )
    shapley_workers = 1 if args.shapley_enabled else 0
    physical_worker_capacity = max(
        0,
        effective_cpus - int(args.num_envs) - 1,
    )
    if total_workers > physical_worker_capacity:
        raise RuntimeError(
            'physical attribution workers exceed effective CPU capacity: '
            f'configured={total_workers}, capacity={physical_worker_capacity}, '
            f'effective_cpus={effective_cpus}, rollout={args.num_envs}'
        )
    if total_workers <= shapley_workers:
        raise RuntimeError(
            'physical attribution has insufficient CPU workers: '
            f'total={total_workers}, shapley_reserved={shapley_workers}; '
            'increase --counterfactual-workers or disable Shapley'
        )
    coordinator = CounterfactualCoordinator(
        causal_replay_buffer=causal_replay_buffer,
        rollout_worker_count=args.num_envs,
        cpu_count=effective_cpus,
        worker_count=total_workers - shapley_workers,
        scheduler_config=build_counterfactual_config(args),
    )
    if not coordinator.enabled:
        raise RuntimeError(
            'counterfactual labels are enabled but no worker was created'
        )
    return coordinator, shapley_workers


def refresh_counterfactual_target(
        coordinator,
        target_model,
        args,
        env_config,
        *,
        update_step):
    """只在 target network 同步边界冻结一次 CPU 策略 payload。"""

    if coordinator is None:
        return None
    return coordinator.refresh_target_policy(
        model=target_model,
        model_config=FrozenGNNModelConfig(
            **build_model_config(args)
        ),
        policy_version=f'target-update-{int(update_step):08d}',
        gamma=args.gamma,
        max_physics_frames=args.max_physics_frames,
        stable_frames=args.stable_frames,
        reward_config=env_config.reward_config,
        state_analyzer_config=env_config.state_analyzer_config,
    )


def process_counterfactual_rollout(
        collector,
        coordinator,
        collect_stats,
        *,
        shapley_coordinator=None):
    """把本轮 proposal 非阻塞路由到 Shapley 或普通反事实。"""

    if coordinator is None:
        return ()
    coordinator.record_real_steps(
        collect_stats.steps,
        dispatch_candidates=False,
    )
    if shapley_coordinator is not None:
        shapley_coordinator.retry_pending()
    proposals = collector.drain_counterfactual_proposals()
    ordinary_proposals = []
    scheduler_stats = coordinator.stats.scheduler
    created_real_step = (
        scheduler_stats.real_steps
        if scheduler_stats is not None
        else 0
    )
    for proposal in proposals:
        routed_to_shapley = False
        if shapley_coordinator is not None:
            decision = shapley_coordinator.consider(
                proposal,
                coordinator.target_policy,
                created_real_step=created_real_step,
            )
            routed_to_shapley = bool(
                decision.skip_counterfactual
            )
        if not routed_to_shapley:
            ordinary_proposals.append(proposal)
    submissions = coordinator.offer_many(ordinary_proposals)
    coordinator.poll()
    if shapley_coordinator is not None:
        shapley_coordinator.poll()
    return tuple(submissions)


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


def scheduled_epsilon(
        update_step,
        env_steps,
        args,
        *,
        schedule_total_updates=None):
    """根据当前配置计算 epsilon。

    ``schedule_total_updates`` 在 checkpoint 首次创建时冻结。恢复时即使只延长
    ``total_updates``，探索率也不会从已经达到的低值突然跳高。
    """

    if args.epsilon_schedule == 'linear':
        return linear_epsilon(env_steps, args)

    if schedule_total_updates is None:
        schedule_total_updates = args.total_updates
    schedule_total_updates = int(schedule_total_updates)
    if schedule_total_updates <= 0:
        raise ValueError('schedule_total_updates must be positive')
    progress = _bounded_unit(
        float(update_step) / float(schedule_total_updates)
    )
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


def _prepare_resume_csv(csv_path, fieldnames, checkpoint_update_step):
    """保留 checkpoint 以前的行，并备份崩溃后产生的超前日志。"""

    csv_path = Path(csv_path)
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return [], 0, None
    with csv_path.open('r', newline='', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        if tuple(reader.fieldnames or ()) != tuple(fieldnames):
            raise ValueError(
                f'CSV schema mismatch while resuming: {csv_path}'
            )
        rows = list(reader)
    retained = []
    orphaned = []
    for row in rows:
        try:
            row_update = int(row['update_step'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f'invalid update_step in resume CSV: {csv_path}'
            ) from exc
        if row_update <= int(checkpoint_update_step):
            retained.append(row)
        else:
            orphaned.append(row)
    if not orphaned:
        return retained, 0, None

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup = csv_path.with_name(
        f'{csv_path.stem}.orphaned_after_{checkpoint_update_step:08d}_'
        f'{stamp}{csv_path.suffix}'
    )
    os.replace(csv_path, backup)
    with csv_path.open('w', newline='', encoding='utf-8') as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained)
        file_obj.flush()
        os.fsync(file_obj.fileno())
    return retained, len(orphaned), backup


class MetricLogger:
    """把训练指标同时保存到内存和 CSV。"""

    def __init__(self, csv_path, *, resume_update_step=None):
        self.csv_path = Path(csv_path)
        self.orphaned_row_count = 0
        self.orphaned_backup_path = None
        if resume_update_step is None:
            self.rows = []
            mode = 'w'
        else:
            (
                self.rows,
                self.orphaned_row_count,
                self.orphaned_backup_path,
            ) = _prepare_resume_csv(
                self.csv_path,
                METRIC_FIELDS,
                resume_update_step,
            )
            mode = 'a'
        needs_header = (
            mode == 'w'
            or not self.csv_path.exists()
            or self.csv_path.stat().st_size == 0
        )
        self._file = self.csv_path.open(
            mode,
            newline='',
            encoding='utf-8',
        )
        self._writer = csv.DictWriter(self._file, fieldnames=METRIC_FIELDS)
        if needs_header:
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

    def __init__(self, csv_path, *, resume_update_step=None):
        self.csv_path = Path(csv_path)
        self.orphaned_row_count = 0
        self.orphaned_backup_path = None
        if resume_update_step is None:
            self.rows = []
            mode = 'w'
        else:
            (
                self.rows,
                self.orphaned_row_count,
                self.orphaned_backup_path,
            ) = _prepare_resume_csv(
                self.csv_path,
                EPISODE_METRIC_FIELDS,
                resume_update_step,
            )
            mode = 'a'
        self._episode_index = max(
            (
                int(row['episode_index'])
                for row in self.rows
            ),
            default=0,
        )
        needs_header = (
            mode == 'w'
            or not self.csv_path.exists()
            or self.csv_path.stat().st_size == 0
        )
        self._file = self.csv_path.open(
            mode,
            newline='',
            encoding='utf-8',
        )
        self._writer = csv.DictWriter(self._file, fieldnames=EPISODE_METRIC_FIELDS)
        if needs_header:
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
        self.replay_transitions_emitted = 0
        self.n_step_pending_count = 0
        self.n_step_forced_flush_emitted = 0
        self.causal_rule_build_calls = 0
        self.causal_rule_build_seconds = 0.0
        self.causal_rule_input_event_count = 0
        self.causal_rule_eligible_event_count = 0
        self.causal_rule_budget_count = 0
        self.causal_rule_samples_generated = 0
        self.causal_samples_pushed = 0
        self.causal_samples_emitted = 0
        self.causal_rule_skip_reason_counts = {}
        self.causal_buffer_size = 0
        self.causal_context_count = 0
        self.counterfactual_snapshot_calls = 0
        self.counterfactual_snapshot_seconds = 0.0
        self.counterfactual_snapshot_failures = 0
        self.counterfactual_history_evictions = 0
        self.counterfactual_history_size = 0
        self.counterfactual_proposal_build_calls = 0
        self.counterfactual_proposal_build_seconds = 0.0
        self.counterfactual_proposal_input_event_count = 0
        self.counterfactual_proposal_confirmed_event_count = 0
        self.counterfactual_proposal_budget_count = 0
        self.counterfactual_proposals_generated = 0
        self.counterfactual_proposals_transfer_selected = 0
        self.counterfactual_proposals_transfer_throttled = 0
        self.counterfactual_proposal_skip_reason_counts = {}
        self.counterfactual_proposals_serialized = 0
        self.counterfactual_proposal_serialized_bytes = 0
        self.total_reward = 0.0
        self.episodes = 0
        self.transition_keys = []
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
        self.potential_shaping_abs_values = []
        self.state_analysis_calls = 0
        self.state_analysis_seconds = 0.0
        self.state_analysis_cache_hits = 0
        self.state_analysis_degraded_count = 0
        self.attribution_tracker_calls = 0
        self.attribution_tracker_seconds = 0.0
        self.attribution_events_created = 0
        self.attribution_events_confirmed = 0
        self.attribution_events_cancelled = 0
        self.attribution_events_interrupted = 0
        self.attribution_pending_event_count = 0
        self.attribution_lineage_merge_count = 0
        self.attribution_chain_merge_count = 0
        self.attribution_max_chain_depth = 0
        self.attribution_event_status_counts = {}
        self.attribution_confidence_tier_counts = {}
        self.attribution_delays = []
        self.merge_level_counts = {}
        self.max_fruit_level = 0
        self.reward_breakdown_totals = {
            field_name: 0.0
            for field_name in REWARD_BREAKDOWN_FIELDS
        }

    def add(self, stats):
        """把一次 `RolloutCollector.collect_steps()` 的结果并入窗口。"""

        step_offset = self.steps
        self.steps += stats.steps
        self.replay_transitions_emitted += getattr(
            stats,
            'replay_transitions_emitted',
            stats.steps,
        )
        self.n_step_pending_count = getattr(
            stats,
            'n_step_pending_count',
            self.n_step_pending_count,
        )
        self.n_step_forced_flush_emitted += getattr(
            stats,
            'n_step_forced_flush_emitted',
            0,
        )
        self.causal_rule_build_calls += getattr(
            stats,
            'causal_rule_build_calls',
            0,
        )
        self.causal_rule_build_seconds += getattr(
            stats,
            'causal_rule_build_seconds',
            0.0,
        )
        self.causal_rule_input_event_count += getattr(
            stats,
            'causal_rule_input_event_count',
            0,
        )
        self.causal_rule_eligible_event_count += getattr(
            stats,
            'causal_rule_eligible_event_count',
            0,
        )
        self.causal_rule_budget_count += getattr(
            stats,
            'causal_rule_budget_count',
            0,
        )
        self.causal_rule_samples_generated += getattr(
            stats,
            'causal_rule_samples_generated',
            0,
        )
        self.causal_samples_pushed += getattr(
            stats,
            'causal_samples_pushed',
            0,
        )
        self.causal_samples_emitted += getattr(
            stats,
            'causal_samples_emitted',
            0,
        )
        self.causal_buffer_size = getattr(
            stats,
            'causal_buffer_size',
            self.causal_buffer_size,
        )
        self.causal_context_count = getattr(
            stats,
            'causal_context_count',
            self.causal_context_count,
        )
        for reason, count in getattr(
                stats,
                'causal_rule_skip_reason_counts',
                ()):
            self.causal_rule_skip_reason_counts[reason] = (
                self.causal_rule_skip_reason_counts.get(reason, 0)
                + int(count)
            )
        for field_name in (
                'counterfactual_snapshot_calls',
                'counterfactual_snapshot_seconds',
                'counterfactual_snapshot_failures',
                'counterfactual_history_evictions',
                'counterfactual_proposal_build_calls',
                'counterfactual_proposal_build_seconds',
                'counterfactual_proposal_input_event_count',
                'counterfactual_proposal_confirmed_event_count',
                'counterfactual_proposal_budget_count',
                'counterfactual_proposals_generated',
                'counterfactual_proposals_transfer_selected',
                'counterfactual_proposals_transfer_throttled',
                'counterfactual_proposals_serialized',
                'counterfactual_proposal_serialized_bytes'):
            setattr(
                self,
                field_name,
                getattr(self, field_name)
                + getattr(stats, field_name, 0),
            )
        self.counterfactual_history_size = getattr(
            stats,
            'counterfactual_history_size',
            self.counterfactual_history_size,
        )
        for reason, count in getattr(
                stats,
                'counterfactual_proposal_skip_reason_counts',
                ()):
            self.counterfactual_proposal_skip_reason_counts[reason] = (
                self.counterfactual_proposal_skip_reason_counts.get(
                    reason,
                    0,
                )
                + int(count)
            )
        self.total_reward += stats.total_reward
        self.episodes += stats.episodes
        self.transition_keys.extend(stats.transition_keys)
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
        self.potential_shaping_abs_values.extend(
            float(value)
            for value in getattr(stats, 'potential_shaping_abs_values', ())
        )
        self.state_analysis_calls += getattr(stats, 'state_analysis_calls', 0)
        self.state_analysis_seconds += getattr(stats, 'state_analysis_seconds', 0.0)
        self.state_analysis_cache_hits += getattr(
            stats,
            'state_analysis_cache_hits',
            0,
        )
        self.state_analysis_degraded_count += getattr(
            stats,
            'state_analysis_degraded_count',
            0,
        )
        self.attribution_tracker_calls += getattr(
            stats,
            'attribution_tracker_calls',
            0,
        )
        self.attribution_tracker_seconds += getattr(
            stats,
            'attribution_tracker_seconds',
            0.0,
        )
        self.attribution_events_created += getattr(
            stats,
            'attribution_events_created',
            0,
        )
        self.attribution_events_confirmed += getattr(
            stats,
            'attribution_events_confirmed',
            0,
        )
        self.attribution_events_cancelled += getattr(
            stats,
            'attribution_events_cancelled',
            0,
        )
        self.attribution_events_interrupted += getattr(
            stats,
            'attribution_events_interrupted',
            0,
        )
        self.attribution_pending_event_count = getattr(
            stats,
            'attribution_pending_event_count',
            self.attribution_pending_event_count,
        )
        self.attribution_lineage_merge_count += getattr(
            stats,
            'attribution_lineage_merge_count',
            0,
        )
        self.attribution_chain_merge_count += getattr(
            stats,
            'attribution_chain_merge_count',
            0,
        )
        self.attribution_max_chain_depth = max(
            self.attribution_max_chain_depth,
            getattr(stats, 'attribution_max_chain_depth', 0),
        )
        for event_type, status, count in getattr(
                stats,
                'attribution_event_status_counts',
                ()):
            key = event_type, status
            self.attribution_event_status_counts[key] = (
                self.attribution_event_status_counts.get(key, 0)
                + int(count)
            )
        for tier, count in getattr(
                stats,
                'attribution_confidence_tier_counts',
                ()):
            self.attribution_confidence_tier_counts[tier] = (
                self.attribution_confidence_tier_counts.get(
                    tier,
                    0,
                )
                + int(count)
            )
        self.attribution_delays.extend(
            int(delay)
            for delay in getattr(stats, 'attribution_delays', ())
        )
        for level, count in getattr(stats, 'merge_level_counts', ()):
            level = int(level)
            self.merge_level_counts[level] = (
                self.merge_level_counts.get(level, 0)
                + int(count)
            )
        self.max_fruit_level = max(
            self.max_fruit_level,
            int(getattr(stats, 'max_fruit_level', 0)),
        )

        totals = stats.reward_breakdown_totals_dict
        for field_name in REWARD_BREAKDOWN_FIELDS:
            self.reward_breakdown_totals[field_name] += float(totals.get(field_name, 0.0))

    def to_rollout_stats(self, buffer_size):
        """转换成和 collector 输出兼容的 `RolloutStats`，供日志代码复用。"""

        return RolloutStats(
            steps=self.steps,
            replay_transitions_emitted=(
                self.replay_transitions_emitted
            ),
            n_step_pending_count=self.n_step_pending_count,
            n_step_forced_flush_emitted=(
                self.n_step_forced_flush_emitted
            ),
            causal_rule_build_calls=self.causal_rule_build_calls,
            causal_rule_build_seconds=self.causal_rule_build_seconds,
            causal_rule_input_event_count=(
                self.causal_rule_input_event_count
            ),
            causal_rule_eligible_event_count=(
                self.causal_rule_eligible_event_count
            ),
            causal_rule_budget_count=self.causal_rule_budget_count,
            causal_rule_samples_generated=(
                self.causal_rule_samples_generated
            ),
            causal_samples_pushed=self.causal_samples_pushed,
            causal_samples_emitted=self.causal_samples_emitted,
            causal_rule_skip_reason_counts=tuple(sorted(
                self.causal_rule_skip_reason_counts.items()
            )),
            causal_buffer_size=self.causal_buffer_size,
            causal_context_count=self.causal_context_count,
            counterfactual_snapshot_calls=(
                self.counterfactual_snapshot_calls
            ),
            counterfactual_snapshot_seconds=(
                self.counterfactual_snapshot_seconds
            ),
            counterfactual_snapshot_failures=(
                self.counterfactual_snapshot_failures
            ),
            counterfactual_history_evictions=(
                self.counterfactual_history_evictions
            ),
            counterfactual_history_size=(
                self.counterfactual_history_size
            ),
            counterfactual_proposal_build_calls=(
                self.counterfactual_proposal_build_calls
            ),
            counterfactual_proposal_build_seconds=(
                self.counterfactual_proposal_build_seconds
            ),
            counterfactual_proposal_input_event_count=(
                self.counterfactual_proposal_input_event_count
            ),
            counterfactual_proposal_confirmed_event_count=(
                self.counterfactual_proposal_confirmed_event_count
            ),
            counterfactual_proposal_budget_count=(
                self.counterfactual_proposal_budget_count
            ),
            counterfactual_proposals_generated=(
                self.counterfactual_proposals_generated
            ),
            counterfactual_proposals_transfer_selected=(
                self.counterfactual_proposals_transfer_selected
            ),
            counterfactual_proposals_transfer_throttled=(
                self.counterfactual_proposals_transfer_throttled
            ),
            counterfactual_proposal_skip_reason_counts=tuple(sorted(
                self.counterfactual_proposal_skip_reason_counts.items()
            )),
            counterfactual_proposals_serialized=(
                self.counterfactual_proposals_serialized
            ),
            counterfactual_proposal_serialized_bytes=(
                self.counterfactual_proposal_serialized_bytes
            ),
            episodes=self.episodes,
            total_reward=self.total_reward,
            reward_breakdown_totals=tuple(
                (field_name, self.reward_breakdown_totals[field_name])
                for field_name in REWARD_BREAKDOWN_FIELDS
            ),
            transition_keys=tuple(self.transition_keys),
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
            potential_shaping_abs_values=tuple(
                self.potential_shaping_abs_values
            ),
            state_analysis_calls=self.state_analysis_calls,
            state_analysis_seconds=self.state_analysis_seconds,
            state_analysis_cache_hits=self.state_analysis_cache_hits,
            state_analysis_degraded_count=self.state_analysis_degraded_count,
            attribution_tracker_calls=self.attribution_tracker_calls,
            attribution_tracker_seconds=self.attribution_tracker_seconds,
            attribution_events_created=self.attribution_events_created,
            attribution_events_confirmed=(
                self.attribution_events_confirmed
            ),
            attribution_events_cancelled=(
                self.attribution_events_cancelled
            ),
            attribution_events_interrupted=(
                self.attribution_events_interrupted
            ),
            attribution_pending_event_count=(
                self.attribution_pending_event_count
            ),
            attribution_lineage_merge_count=(
                self.attribution_lineage_merge_count
            ),
            attribution_chain_merge_count=(
                self.attribution_chain_merge_count
            ),
            attribution_max_chain_depth=(
                self.attribution_max_chain_depth
            ),
            attribution_event_status_counts=tuple(
                (
                    event_type,
                    status,
                    count,
                )
                for (event_type, status), count
                in sorted(
                    self.attribution_event_status_counts.items()
                )
            ),
            attribution_confidence_tier_counts=tuple(sorted(
                self.attribution_confidence_tier_counts.items()
            )),
            attribution_delays=tuple(self.attribution_delays),
            merge_level_counts=tuple(sorted(
                self.merge_level_counts.items()
            )),
            max_fruit_level=self.max_fruit_level,
        )


def _git_metadata():
    """尽力记录当前代码身份；Git 不可用时显式返回 unknown。"""

    def run_git(*arguments):
        try:
            completed = subprocess.run(
                ('git', *arguments),
                cwd=Path(__file__).resolve().parents[3],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    return {
        'commit': run_git('rev-parse', 'HEAD'),
        'branch': run_git('branch', '--show-current'),
        'dirty': bool(run_git('status', '--porcelain')),
    }


def _atomic_json_write(path, payload):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as file_obj:
            json.dump(
                payload,
                file_obj,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def write_config(
        run_dir,
        args,
        *,
        run_manifest,
        env_config,
        trainer_config,
        counterfactual_config,
        shapley_config,
        resume_checkpoint=None,
        init_checkpoint=None,
        initialization_state=None):
    """冻结完整解析配置、归因契约、依赖和代码身份。"""

    config = {
        'argv': sys.argv,
        'args': vars(args),
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'run_manifest': run_manifest.to_dict(),
        'resolved': {
            'environment': asdict(env_config),
            'trainer': asdict(trainer_config),
            'counterfactual': asdict(counterfactual_config),
            'shapley': asdict(shapley_config),
        },
        'fingerprints': {
            'training_config': run_manifest.config_fingerprint,
            'counterfactual': counterfactual_config.fingerprint,
            'shapley': shapley_config.fingerprint,
            'state_analyzer': (
                env_config.state_analyzer_config.fingerprint
            ),
        },
        'runtime': {
            'python': platform.python_version(),
            'python_executable': sys.executable,
            'torch': str(torch.__version__),
            'torch_cuda_runtime': torch.version.cuda,
            'pymunk': str(pymunk.version),
            'chipmunk': str(pymunk.chipmunk_version),
        },
        'git': _git_metadata(),
        'resume_checkpoint': (
            str(resume_checkpoint)
            if resume_checkpoint is not None
            else None
        ),
        'initialization': (
            {
                'checkpoint': str(init_checkpoint),
                **(initialization_state or {}),
            }
            if init_checkpoint is not None
            else None
        ),
    }
    if resume_checkpoint is None:
        output_path = run_dir / 'config.json'
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output_path = run_dir / f'resume_config_{stamp}.json'
    _atomic_json_write(output_path, config)
    return output_path


def write_failure_diagnostic(
        run_dir,
        exception,
        *,
        stage,
        update_step,
        trainer_update_step,
        env_steps,
        latest_metrics=None,
        replay_buffer=None,
        causal_replay_buffer=None,
        counterfactual_coordinator=None,
        shapley_coordinator=None):
    """原子记录异常现场；内容限定为轻量 JSON，不依赖绘图库。"""

    counterfactual_stats = (
        counterfactual_coordinator.stats.to_dict()
        if counterfactual_coordinator is not None
        else None
    )
    shapley_stats = (
        shapley_coordinator.stats.to_dict()
        if (
            shapley_coordinator is not None
            and hasattr(shapley_coordinator.stats, 'to_dict')
        )
        else (
            asdict(shapley_coordinator.stats)
            if shapley_coordinator is not None
            else None
        )
    )
    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'stage': str(stage),
        'exception_type': type(exception).__name__,
        'exception_message': str(exception),
        'traceback': ''.join(traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__,
        )),
        'update_step': int(update_step),
        'trainer_update_step': int(trainer_update_step),
        'env_steps': int(env_steps),
        'latest_metrics': latest_metrics or {},
        'replay_storage': (
            replay_buffer.storage_stats
            if replay_buffer is not None
            else None
        ),
        'causal_replay_storage': (
            causal_replay_buffer.storage_stats
            if causal_replay_buffer is not None
            else None
        ),
        'counterfactual': counterfactual_stats,
        'shapley': shapley_stats,
    }
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = run_dir / f'failure_{stamp}.json'
    _atomic_json_write(path, payload)
    # latest 指针同样原子替换，历史诊断仍保留。
    _atomic_json_write(run_dir / 'failure_latest.json', payload)
    return path


def clear_active_failure_diagnostic(run_dir):
    """成功完成恢复训练后清除活动失败指针，保留带时间戳的历史诊断。"""

    (Path(run_dir) / 'failure_latest.json').unlink(
        missing_ok=True,
    )


def write_attribution_shutdown_summary(
        run_dir,
        collector,
        finalized):
    """持久化训练退出时被取消的 pending，避免 worker 静默丢事件。"""

    event_type_counts = {}
    resolution_reason_counts = {}
    workers = []
    if isinstance(collector, ParallelRolloutCollector):
        for summary in finalized or ():
            workers.append({
                'worker_id': summary.worker_id,
                'cancelled_pending_count': (
                    summary.cancelled_pending_count
                ),
                'n_step_flush_emitted': int(getattr(
                    summary,
                    'n_step_flush_emitted',
                    0,
                )),
                'event_type_counts': dict(summary.event_type_counts),
                'resolution_reason_counts': dict(
                    summary.resolution_reason_counts
                ),
            })
            for event_type, count in summary.event_type_counts:
                event_type_counts[event_type] = (
                    event_type_counts.get(event_type, 0)
                    + int(count)
                )
            for reason, count in summary.resolution_reason_counts:
                resolution_reason_counts[reason] = (
                    resolution_reason_counts.get(reason, 0)
                    + int(count)
                )
    else:
        events = tuple(finalized or ())
        for event in events:
            event_type_counts[event.event_type] = (
                event_type_counts.get(event.event_type, 0) + 1
            )
            reason = event.resolution_reason or 'unknown'
            resolution_reason_counts[reason] = (
                resolution_reason_counts.get(reason, 0) + 1
            )
        workers.append({
            'worker_id': getattr(collector, 'worker_id', 0),
            'cancelled_pending_count': len(events),
            'n_step_flush_emitted': int(getattr(
                collector,
                'close_n_step_flush_emitted',
                0,
            )),
            'event_type_counts': dict(sorted(
                event_type_counts.items()
            )),
            'resolution_reason_counts': dict(sorted(
                resolution_reason_counts.items()
            )),
        })

    payload = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'cancelled_pending_count': sum(
            item['cancelled_pending_count']
            for item in workers
        ),
        'n_step_flush_emitted': sum(
            item['n_step_flush_emitted']
            for item in workers
        ),
        'event_type_counts': dict(sorted(event_type_counts.items())),
        'resolution_reason_counts': dict(sorted(
            resolution_reason_counts.items()
        )),
        'workers': workers,
    }
    path = run_dir / 'attribution_shutdown.json'
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def write_attribution_warmup_summary(
        run_dir,
        stats,
        *,
        phase='warmup',
        output_name='attribution_warmup.json'):
    """单独记录 warmup 归因，避免与首个训练指标窗口混在一起。"""

    event_status_counts = {}
    for event_type, status, count in getattr(
            stats,
            'attribution_event_status_counts',
            ()):
        event_status_counts.setdefault(event_type, {})[status] = int(
            count
        )
    delays = tuple(
        float(delay)
        for delay in getattr(stats, 'attribution_delays', ())
    )
    calls = int(getattr(stats, 'attribution_tracker_calls', 0))
    seconds = float(
        getattr(stats, 'attribution_tracker_seconds', 0.0)
    )
    state_analysis_calls = int(getattr(
        stats,
        'state_analysis_calls',
        0,
    ))
    state_analysis_degraded_count = int(getattr(
        stats,
        'state_analysis_degraded_count',
        0,
    ))
    payload = {
        'schema_version': 1,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'phase': str(phase),
        'steps': int(stats.steps),
        'counterfactual_snapshot_failures': int(getattr(
            stats,
            'counterfactual_snapshot_failures',
            0,
        )),
        'state_analysis_calls': state_analysis_calls,
        'state_analysis_degraded_count': (
            state_analysis_degraded_count
        ),
        'state_analysis_degraded_rate': (
            state_analysis_degraded_count / state_analysis_calls
            if state_analysis_calls > 0
            else 0.0
        ),
        'p95_abs_potential_shaping_reward': float(getattr(
            stats,
            'p95_abs_potential_shaping_reward',
            0.0,
        )),
        'attribution_tracker_calls': calls,
        'attribution_tracker_seconds': seconds,
        'mean_attribution_tracker_seconds': (
            seconds / calls if calls > 0 else None
        ),
        'events_created': int(getattr(
            stats,
            'attribution_events_created',
            0,
        )),
        'events_confirmed': int(getattr(
            stats,
            'attribution_events_confirmed',
            0,
        )),
        'events_cancelled': int(getattr(
            stats,
            'attribution_events_cancelled',
            0,
        )),
        'events_interrupted': int(getattr(
            stats,
            'attribution_events_interrupted',
            0,
        )),
        'pending_event_count_at_end': int(getattr(
            stats,
            'attribution_pending_event_count',
            0,
        )),
        'lineage_merge_count': int(getattr(
            stats,
            'attribution_lineage_merge_count',
            0,
        )),
        'chain_merge_count': int(getattr(
            stats,
            'attribution_chain_merge_count',
            0,
        )),
        'max_chain_depth': int(getattr(
            stats,
            'attribution_max_chain_depth',
            0,
        )),
        'replay_transitions_emitted': int(getattr(
            stats,
            'replay_transitions_emitted',
            stats.steps,
        )),
        'n_step_pending_count_at_end': int(getattr(
            stats,
            'n_step_pending_count',
            0,
        )),
        'n_step_forced_flush_emitted': int(getattr(
            stats,
            'n_step_forced_flush_emitted',
            0,
        )),
        'causal_rule_build_calls': int(getattr(
            stats,
            'causal_rule_build_calls',
            0,
        )),
        'causal_rule_build_seconds': float(getattr(
            stats,
            'causal_rule_build_seconds',
            0.0,
        )),
        'causal_rule_input_event_count': int(getattr(
            stats,
            'causal_rule_input_event_count',
            0,
        )),
        'causal_rule_eligible_event_count': int(getattr(
            stats,
            'causal_rule_eligible_event_count',
            0,
        )),
        'causal_rule_budget_count': int(getattr(
            stats,
            'causal_rule_budget_count',
            0,
        )),
        'causal_rule_samples_generated': int(getattr(
            stats,
            'causal_rule_samples_generated',
            0,
        )),
        'causal_samples_emitted': int(getattr(
            stats,
            'causal_samples_emitted',
            0,
        )),
        'causal_buffer_size_at_end': int(getattr(
            stats,
            'causal_buffer_size',
            0,
        )),
        'causal_context_count_at_end': int(getattr(
            stats,
            'causal_context_count',
            0,
        )),
        'causal_rule_skip_reason_counts': dict(getattr(
            stats,
            'causal_rule_skip_reason_counts',
            (),
        )),
        'mean_attribution_delay': (
            sum(delays) / len(delays) if delays else None
        ),
        'p95_attribution_delay': (
            _linear_percentile(delays, 0.95)
            if delays
            else None
        ),
        'event_status_counts': {
            event_type: dict(sorted(status_counts.items()))
            for event_type, status_counts
            in sorted(event_status_counts.items())
        },
        'confidence_tier_counts': dict(getattr(
            stats,
            'attribution_confidence_tier_counts',
            (),
        )),
        'merge_level_counts': dict(getattr(
            stats,
            'merge_level_counts',
            (),
        )),
        'max_fruit_level': int(getattr(
            stats,
            'max_fruit_level',
            0,
        )),
    }
    path = run_dir / output_name
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def close_training_resources(
        *,
        run_dir,
        replay_buffer,
        collector,
        metrics,
        episode_metrics,
        counterfactual_coordinator=None,
        shapley_coordinator=None,
        suppress_errors=False):
    """尽最大努力关闭所有资源，并保证 tracker finalize 不被 flush 阻断。"""

    errors = []
    finalized_attribution = ()

    try:
        if hasattr(collector, 'close'):
            finalized_attribution = collector.close()
    except BaseException as exc:
        errors.append(('collector.close', exc))

    try:
        write_attribution_shutdown_summary(
            run_dir,
            collector,
            finalized_attribution,
        )
    except BaseException as exc:
        errors.append(('attribution shutdown summary', exc))

    if shapley_coordinator is not None:
        try:
            shapley_coordinator.close(wait=True)
        except BaseException as exc:
            errors.append(('shapley_coordinator.close', exc))
    if counterfactual_coordinator is not None:
        try:
            counterfactual_coordinator.close(wait=True)
        except BaseException as exc:
            errors.append(('counterfactual_coordinator.close', exc))
        try:
            summary_path = run_dir / 'counterfactual_shutdown.json'
            summary_path.write_text(
                json.dumps(
                    counterfactual_coordinator.checkpoint_state(),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding='utf-8',
            )
        except BaseException as exc:
            errors.append(('counterfactual shutdown summary', exc))

    for label, operation in (
            ('replay_buffer.flush', replay_buffer.flush),
            ('metrics.close', metrics.close),
            ('episode_metrics.close', episode_metrics.close)):
        try:
            operation()
        except BaseException as exc:
            errors.append((label, exc))

    if errors:
        labels = ', '.join(label for label, _exc in errors)
        message = f'training resource cleanup failed: {labels}'
        if suppress_errors:
            print(f'warning={message}', file=sys.stderr, flush=True)
        else:
            raise RuntimeError(message) from errors[0][1]
    return finalized_attribution


def prune_step_checkpoints(checkpoint_dir, keep_last):
    """删除超出保留窗口的本 run 周期 checkpoint，返回删除数量。"""

    keep_last = int(keep_last)
    if keep_last < 0:
        raise ValueError('keep_last must be non-negative')
    if keep_last == 0:
        return 0
    checkpoint_dir = Path(checkpoint_dir).resolve()
    candidates = sorted(checkpoint_dir.glob('step_*.pt'))
    obsolete = candidates[:-keep_last]
    removed = 0
    for path in obsolete:
        resolved = path.resolve()
        if resolved.parent != checkpoint_dir:
            raise RuntimeError(
                f'refusing to prune checkpoint outside directory: {path}'
            )
        resolved.unlink()
        removed += 1
    return removed


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
        extra_checkpoint_name=None,
        run_manifest=None,
        trainer=None,
        replay_buffer=None,
        causal_replay_buffer=None,
        counterfactual_coordinator=None,
        shapley_coordinator=None,
        best_eval_score=None,
        best_eval_update=0):
    """原子保存版本化 checkpoint；任何一次写入都不破坏旧文件。"""

    epsilon_schedule_total_updates = int(
        (
            run_manifest.config.get(
                'total_updates',
                args.total_updates,
            )
            if run_manifest is not None
            else args.total_updates
        )
    )
    if epsilon_schedule_total_updates <= 0:
        raise ValueError(
            'epsilon schedule total updates must be positive'
        )
    training_state = {
        'online_model': online_model.state_dict(),
        'target_model': target_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'update_step': int(update_step),
        'env_steps': int(env_steps),
        'epsilon': float(epsilon),
        'epsilon_schedule_total_updates': (
            epsilon_schedule_total_updates
        ),
        'latest_metrics': latest_metrics or {},
        'best_eval_score': (
            float(best_eval_score)
            if best_eval_score is not None
            else float('-inf')
        ),
        'best_eval_update': int(best_eval_update),
        'saved_at': datetime.now().isoformat(timespec='seconds'),
        'trainer_update_step': int(
            trainer.update_step if trainer is not None else update_step
        ),
        'replay_storage': (
            replay_buffer.storage_stats
            if replay_buffer is not None
            else None
        ),
        'causal_replay_storage': (
            causal_replay_buffer.storage_stats
            if causal_replay_buffer is not None
            else None
        ),
        'counterfactual': (
            counterfactual_coordinator.checkpoint_state()
            if counterfactual_coordinator is not None
            else None
        ),
        'shapley': (
            shapley_coordinator.checkpoint_state()
            if shapley_coordinator is not None
            else None
        ),
    }
    if run_manifest is None:
        run_manifest = create_run_manifest(
            vars(args),
            metadata={
                'experiment': 'fruit-merge-dqn',
                'config_fingerprint': config_fingerprint(vars(args)),
            },
        )
    checkpoint = build_training_checkpoint(
        training_state=training_state,
        config=vars(args),
        manifest=run_manifest,
        components={
            'replay_buffer': replay_buffer,
            'causal_replay_buffer': causal_replay_buffer,
        },
    )

    checkpoint_dir = run_dir / 'checkpoints'
    latest_path = checkpoint_dir / 'latest.pt'
    atomic_torch_save(checkpoint, latest_path)

    step_path = None
    step_materialization = None
    pruned_count = 0
    if step_checkpoint:
        step_path = checkpoint_dir / f'step_{update_step:08d}.pt'
        _path, step_materialization = atomic_clone_file(
            latest_path,
            step_path,
        )
        pruned_count = prune_step_checkpoints(
            checkpoint_dir,
            getattr(args, 'checkpoint_keep_last', 0),
        )

    extra_path = None
    extra_materialization = None
    if extra_checkpoint_name:
        extra_path = checkpoint_dir / extra_checkpoint_name
        _path, extra_materialization = atomic_clone_file(
            latest_path,
            extra_path,
        )
    return {
        'latest_path': latest_path,
        'step_path': step_path,
        'extra_path': extra_path,
        'checkpoint_bytes': latest_path.stat().st_size,
        'pruned_count': pruned_count,
        'step_materialization': step_materialization,
        'extra_materialization': extra_materialization,
    }


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

                transition_key = TransitionKey(
                    worker_id=int(getattr(args, 'num_envs', 1)),
                    episode_id=episode_index,
                    step_index=episode_length,
                )
                state_analysis = env.prepare_state_analysis(
                    transition_key
                )
                graph = graph_builder.build(
                    obs,
                    candidates,
                    state_analysis=state_analysis,
                )
                with torch.no_grad():
                    q_values = model(graph).detach().cpu()
                action_offset = int(torch.argmax(q_values).item())

                obs, reward, terminated, truncated, info = env.step(
                    action_offset,
                    transition_key=transition_key,
                )
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
    _maybe_plot_structure_learning(run_dir, rows, x, plt)
    return True


def _maybe_plot_reward_breakdown(run_dir, rows, x, plt):
    """生成独立的 reward breakdown 曲线图。"""

    reward_fields = tuple(
        metric_field
        for _reward_field, metric_field in REWARD_BREAKDOWN_METRIC_FIELDS
    )
    if not _has_any_points(rows, reward_fields):
        return False

    fig, axes = plt.subplots(4, 1, figsize=(12, 13), constrained_layout=True)

    _plot_one(
        axes[0],
        x,
        _series(rows, 'collect_mean_reward_total'),
        'total',
        'Reward V2 total and task utility',
    )
    _plot_one(
        axes[0],
        x,
        _series(rows, 'collect_mean_task_reward'),
        'task',
        'Reward V2 total and task utility',
    )

    _plot_one(
        axes[1],
        x,
        _series(rows, 'collect_mean_potential_shaping_reward'),
        'potential shaping',
        'Potential shaping and terminal component',
    )
    _plot_one(
        axes[1],
        x,
        _series(rows, 'collect_mean_terminal_penalty'),
        'terminal',
        'Potential shaping and terminal component',
    )

    _plot_one(
        axes[2],
        x,
        _series(rows, 'collect_mean_previous_potential'),
        'previous',
        'State potential',
    )
    _plot_one(
        axes[2],
        x,
        _series(rows, 'collect_mean_next_potential'),
        'next',
        'State potential',
    )
    _plot_one(
        axes[2],
        x,
        _series(rows, 'collect_mean_potential_delta'),
        'delta',
        'State potential',
    )

    for metric_field, label in (
            ('collect_mean_next_top_connected_capacity', 'capacity C'),
            ('collect_mean_next_recoverability', 'recoverability R'),
            ('collect_mean_next_chain_readiness', 'chain readiness K')):
        _plot_one(
            axes[3],
            x,
            _series(rows, metric_field),
            label,
            'Next-state potential components',
        )
    _plot_one(
        axes[3],
        x,
        _series(rows, 'collect_p95_abs_potential_shaping_reward'),
        'abs shaping p95',
        'Next-state potential components',
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


def _maybe_plot_structure_learning(run_dir, rows, x, plt):
    """生成结构辅助监督与集中 actor 的独立诊断曲线。"""

    structure_fields = (
        'structural_loss',
        'weighted_structural_loss',
        'structural_valid_count',
        'structural_sample_count',
        'structural_mean_abs_error',
        'actor_inference_mean_batch_size',
        'actor_inference_max_batch',
    )
    if not _has_any_points(rows, structure_fields):
        return False

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True,
    )
    axes = axes.ravel()
    _plot_one(
        axes[0],
        x,
        _series(rows, 'structural_loss'),
        'raw',
        'Structural auxiliary loss',
    )
    _plot_one(
        axes[0],
        x,
        _series(rows, 'weighted_structural_loss'),
        'weighted',
        'Structural auxiliary loss',
    )
    _plot_one(
        axes[1],
        x,
        _series(rows, 'structural_mean_abs_error'),
        'valid dimensions',
        'Structural mean absolute error',
    )
    _plot_one(
        axes[2],
        x,
        _series(rows, 'structural_valid_count'),
        'valid dimensions',
        'Structural label coverage per update',
    )
    _plot_one(
        axes[2],
        x,
        _series(rows, 'structural_sample_count'),
        'labeled samples',
        'Structural label coverage per update',
    )
    _plot_one(
        axes[3],
        x,
        _series(rows, 'actor_inference_mean_batch_size'),
        'mean batch',
        'Central actor micro-batch size',
    )
    _plot_one(
        axes[3],
        x,
        _series(rows, 'actor_inference_max_batch'),
        'max batch',
        'Central actor micro-batch size',
    )
    for axis in axes:
        axis.set_xlabel('update')
        axis.grid(True, alpha=0.25)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc='best')

    output_path = (
        run_dir
        / 'plots'
        / 'structure_learning_curves.png'
    )
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


def _linear_percentile(values, quantile):
    """按线性插值计算分位数；空序列返回空值以便写入 CSV。"""

    values = tuple(float(value) for value in values)
    if not values:
        return ''
    ordered = sorted(values)
    position = float(quantile) * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index] * (1.0 - fraction)
        + ordered[upper_index] * fraction
    )


def print_log(row):
    """打印一行紧凑训练日志。"""

    parts = [
        f"update={int(row['update_step'])}",
        f"env_steps={int(row['env_steps'])}",
        f"eps={float(row['epsilon']):.3f}",
        f"buf={int(row['buffer_size'])}",
        f"loss={float(row['loss']):.4f}",
        f"td_loss={float(row.get('td_loss', row['loss'])):.4f}",
        f"q={float(row['mean_q']):+.3f}",
        f"target={float(row['mean_target']):+.3f}",
        f"reward={float(row['mean_reward']):+.3f}",
        f"td={float(row['mean_abs_td_error']):.3f}",
        f"grad={float(row['grad_norm']):.3f}",
        f"rand/greedy={int(row['random_actions'])}/{int(row['greedy_actions'])}",
    ]
    if int(row.get('causal_batch_size', 0) or 0) > 0:
        parts.append(
            'causal='
            f"{int(row['rule_batch_size'])}/"
            f"{int(row['counterfactual_batch_size'])}/"
            f"{int(row['shapley_batch_size'])}"
            '(规则/反事实/Shapley)'
        )
        parts.append(
            f"rule_loss={float(row['rule_rank_loss']):.4f}"
        )
        parts.append(
            f"cf_loss={float(row['counterfactual_loss']):.4f}"
        )
        parts.append(
            'rule_ok='
            f"{float(row['rule_pair_accuracy']):.1%}/"
            f"{float(row['rule_margin_satisfaction_rate']):.1%}"
            '(方向/margin)'
        )
    if int(row.get('causal_replay_size', 0) or 0) > 0:
        parts.append(
            'causal_buf='
            f"{int(row['causal_replay_size'])}"
            f"[{int(row['causal_replay_positive_count'])}/"
            f"{int(row['causal_replay_negative_count'])}/"
            f"{int(row['causal_replay_counterfactual_count'])}]"
        )
    if int(row.get('structural_valid_count', 0) or 0) > 0:
        parts.append(
            'structure='
            f"{float(row['structural_loss']):.4f}/"
            f"{float(row['structural_mean_abs_error']):.4f}"
            '(loss/mae)'
        )
        parts.append(
            'structure_labels='
            f"{int(row['structural_sample_count'])}/"
            f"{int(row['structural_valid_count'])}"
            '(samples/dims)'
        )
    if int(row.get('actor_inference_batches', 0) or 0) > 0:
        parts.append(
            'actor_batch='
            f"{float(row['actor_inference_mean_batch_size']):.2f}/"
            f"{int(row['actor_inference_max_batch'])}"
            '(mean/max)'
        )

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
        parts.append(f"r_task={float(row['collect_mean_task_reward']):+.3f}")
        parts.append(
            f"r_phi={float(row['collect_mean_potential_shaping_reward']):+.3f}"
        )
        parts.append(
            'phi_abs_p95='
            f"{float(row['collect_p95_abs_potential_shaping_reward']):.3f}"
        )
    if row.get('collect_seconds') not in ('', None):
        parts.append(f"collect={_format_ms(row['collect_seconds'])}(采集)")
        parts.append(f"env={_format_ms(row['collect_env_step_seconds'])}(环境)")
        parts.append(f"graph={_format_ms(row['collect_graph_build_seconds'])}(构图)")
        parts.append(f"train={_format_ms(row['train_step_seconds'])}(训练)")
        parts.append(f"sample={_format_ms(row['replay_sample_seconds'])}(采样)")
        parts.append(f"frames={float(row['collect_mean_physics_frames']):.1f}(物理帧)")
        parts.append(f"nodes={float(row['collect_mean_graph_nodes']):.1f}(节点)")
        parts.append(f"edges={float(row['collect_mean_graph_edges']):.1f}(边)")
        if row.get('collect_mean_state_analysis_seconds') not in ('', None):
            parts.append(
                'analyze='
                f"{_format_ms(row['collect_mean_state_analysis_seconds'])}"
                '(状态分析)'
            )
            parts.append(
                'analysis_cache='
                f"{float(row['collect_state_analysis_cache_hit_rate']):.1%}"
            )
            parts.append(
                'analysis_degraded='
                f"{int(row['collect_state_analysis_degraded_count'])}"
            )
        if row.get('collect_attribution_tracker_calls', 0):
            parts.append(
                'attribute='
                f"{_format_ms(row['collect_mean_attribution_tracker_seconds'])}"
                '(状态归因)'
            )
            parts.append(
                'attr_events='
                f"{int(row['collect_attribution_events_created'])}/"
                f"{int(row['collect_attribution_events_confirmed'])}/"
                f"{int(row['collect_attribution_events_cancelled'])}"
                '(新建/确认/取消)'
            )
            parts.append(
                'attr_pending='
                f"{int(row['collect_attribution_pending_event_count'])}"
            )

    print(' | '.join(parts), flush=True)


def _format_ms(value):
    """把秒格式化成毫秒字符串。"""

    if value in ('', None):
        return ''
    return f'{float(value) * 1000.0:.1f}ms'


def should_sync_parallel_workers(
        update_step,
        worker_sync_interval,
        *,
        model_synced=True):
    """判断当前 update 采集前是否需要同步并行 worker 模型。

    新建 collector（包括从任意 checkpoint 恢复后的 collector）必须先收到一次
    online model，不能只依赖 update 是否恰好落在周期同步边界。
    """

    return (
        not bool(model_synced)
        or update_step == 1
        or (update_step - 1) % int(worker_sync_interval) == 0
    )


def should_sync_parallel_workers_after_train(update_step, worker_sync_interval):
    """判断当前 update 训练后是否需要先同步 worker，再提交下一批异步采集。"""

    return update_step % int(worker_sync_interval) == 0


def online_policy_version(completed_update_step):
    """返回与 online 权重唯一对应、跨恢复过程稳定的策略版本。"""

    completed_update_step = int(completed_update_step)
    if completed_update_step < 0:
        raise ValueError('completed_update_step must be non-negative')
    return f'online-update-{completed_update_step:08d}'


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
        latest_loss=None,
        start_update_step=0,
        start_env_steps=0):
    """按固定时间间隔打印轻量进度心跳。"""

    if args.progress_interval <= 0.0:
        return last_progress_at

    now = time.perf_counter()
    if now - last_progress_at < args.progress_interval:
        return last_progress_at

    percent = 0.0 if total <= 0 else min(100.0, current / total * 100.0)
    completed_env_steps = max(0, env_steps - int(start_env_steps))
    completed_updates = max(0, current - int(start_update_step))
    speed = (
        0.0
        if elapsed <= 0.0
        else completed_env_steps / elapsed
    )
    update_speed = (
        0.0
        if elapsed <= 0.0
        else completed_updates / elapsed
    )
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
        replay_stats=None,
        causal_replay_stats=None,
        counterfactual_stats=None,
        shapley_stats=None,
        actor_stats=None):
    """把训练、采集、评估统计合成一行 CSV 指标。"""

    elapsed = max(1e-9, timing['elapsed'])
    completed_updates = max(
        0,
        int(timing.get('completed_updates', update_step)),
    )
    completed_env_steps = max(
        0,
        int(timing.get('completed_env_steps', env_steps)),
    )
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
    causal_replay_stats = causal_replay_stats or {}
    actor_stats = actor_stats or {}
    causal_strata = causal_replay_stats.get('stratum_counts', {})
    causal_supervision = causal_replay_stats.get(
        'supervision_kind_counts',
        {},
    )
    cf_cumulative = getattr(counterfactual_stats, 'cumulative', None)
    cf_scheduler = getattr(counterfactual_stats, 'scheduler', None)
    if shapley_stats is None:
        shapley_stats = {}
    elif not isinstance(shapley_stats, dict):
        shapley_cumulative = getattr(
            shapley_stats,
            'cumulative',
            None,
        )
        shapley_stats = {
            'enabled': getattr(shapley_stats, 'enabled', False),
            'events_observed': getattr(
                shapley_stats,
                'observed_event_count',
                0,
            ),
            'events_selected': getattr(
                shapley_stats,
                'selected_event_count',
                0,
            ),
            'tasks_submitted': getattr(
                shapley_cumulative,
                'tasks_submitted',
                0,
            ),
            'tasks_completed': getattr(
                shapley_cumulative,
                'results_completed',
                0,
            ),
            'tasks_failed': getattr(
                shapley_cumulative,
                'results_failed',
                0,
            ),
            'terminal_dropped': getattr(
                shapley_cumulative,
                'selected_terminal_dropped',
                0,
            ),
            'reproduction_passed': getattr(
                shapley_cumulative,
                'reproduction_passed',
                0,
            ),
            'reproduction_failed': getattr(
                shapley_cumulative,
                'reproduction_failed',
                0,
            ),
            'numeric_jitter_dropped': getattr(
                shapley_cumulative,
                'numeric_jitter_dropped',
                0,
            ),
            'semantic_divergence_dropped': getattr(
                shapley_cumulative,
                'semantic_divergence_dropped',
                0,
            ),
            **{
                f'numeric_jitter_max_{suffix}_error': getattr(
                    shapley_cumulative,
                    f'numeric_jitter_max_{suffix}_error',
                    0.0,
                )
                for suffix in (
                    'merge_event_position',
                    'fruit_position',
                    'linear_velocity',
                    'orientation',
                    'angular_velocity')
            },
            'samples_inserted': getattr(
                shapley_cumulative,
                'samples_inserted',
                0,
            ),
            'tokens_consumed': getattr(
                shapley_cumulative,
                'consumed_tokens_total',
                0,
            ),
            'pending_count': getattr(
                shapley_stats,
                'pending_task_count',
                0,
            ),
            'drop_reason_counts': dict(getattr(
                shapley_cumulative,
                'drop_reason_counts',
                (),
            )),
        }
    graph_cache_total = collect_stats.graph_cache_hits + collect_stats.graph_cache_misses
    graph_cache_hit_rate = (
        collect_stats.graph_cache_hits / graph_cache_total
        if graph_cache_total > 0
        else ''
    )
    state_analysis_calls = int(
        getattr(collect_stats, 'state_analysis_calls', 0)
    )
    state_analysis_seconds = float(
        getattr(collect_stats, 'state_analysis_seconds', 0.0)
    )
    state_analysis_cache_hits = int(
        getattr(collect_stats, 'state_analysis_cache_hits', 0)
    )
    state_analysis_degraded_count = int(
        getattr(collect_stats, 'state_analysis_degraded_count', 0)
    )
    state_analysis_cache_hit_rate = (
        state_analysis_cache_hits / collect_stats.steps
        if collect_stats.steps > 0
        else ''
    )
    state_analysis_degraded_rate = (
        state_analysis_degraded_count / state_analysis_calls
        if state_analysis_calls > 0
        else ''
    )
    mean_state_analysis_seconds = (
        state_analysis_seconds / state_analysis_calls
        if state_analysis_calls > 0
        else ''
    )
    attribution_tracker_calls = int(
        getattr(collect_stats, 'attribution_tracker_calls', 0)
    )
    attribution_tracker_seconds = float(
        getattr(collect_stats, 'attribution_tracker_seconds', 0.0)
    )
    mean_attribution_tracker_seconds = (
        attribution_tracker_seconds / attribution_tracker_calls
        if attribution_tracker_calls > 0
        else ''
    )
    attribution_delays = tuple(
        float(delay)
        for delay in getattr(collect_stats, 'attribution_delays', ())
    )
    mean_attribution_delay = (
        sum(attribution_delays) / len(attribution_delays)
        if attribution_delays
        else ''
    )
    p95_attribution_delay = _linear_percentile(
        attribution_delays,
        0.95,
    )
    attribution_event_status_counts = {}
    for event_type, status, count in getattr(
            collect_stats,
            'attribution_event_status_counts',
            ()):
        status_counts = attribution_event_status_counts.setdefault(
            event_type,
            {},
        )
        status_counts[status] = (
            status_counts.get(status, 0) + int(count)
        )
    attribution_confidence_tier_counts = dict(getattr(
        collect_stats,
        'attribution_confidence_tier_counts',
        (),
    ))
    merge_level_counts = dict(getattr(
        collect_stats,
        'merge_level_counts',
        (),
    ))
    rule_empirical_agreements = int(causal_replay_stats.get(
        'rule_empirical_agreement_count',
        0,
    ))
    rule_empirical_disagreements = int(causal_replay_stats.get(
        'rule_empirical_disagreement_count',
        0,
    ))
    rule_empirical_comparisons = (
        rule_empirical_agreements + rule_empirical_disagreements
    )

    row = {
        'update_step': update_step,
        'env_steps': env_steps,
        'epsilon': epsilon,
        'buffer_size': collect_stats.buffer_size,
        'loss': train_stats.loss,
        'td_loss': getattr(train_stats, 'td_loss', train_stats.loss),
        'rule_rank_loss': getattr(train_stats, 'rule_rank_loss', 0.0),
        'weighted_rule_rank_loss': getattr(
            train_stats,
            'weighted_rule_rank_loss',
            0.0,
        ),
        'counterfactual_loss': getattr(
            train_stats,
            'counterfactual_loss',
            0.0,
        ),
        'weighted_counterfactual_loss': getattr(
            train_stats,
            'weighted_counterfactual_loss',
            0.0,
        ),
        'structural_loss': getattr(
            train_stats,
            'structural_loss',
            0.0,
        ),
        'weighted_structural_loss': getattr(
            train_stats,
            'weighted_structural_loss',
            0.0,
        ),
        'structural_valid_count': getattr(
            train_stats,
            'structural_valid_count',
            0,
        ),
        'structural_sample_count': getattr(
            train_stats,
            'structural_sample_count',
            0,
        ),
        'structural_mean_abs_error': getattr(
            train_stats,
            'structural_mean_abs_error',
            0.0,
        ),
        'causal_update_applied': int(bool(getattr(
            train_stats,
            'causal_update_applied',
            False,
        ))),
        'causal_batch_size': getattr(
            train_stats,
            'causal_batch_size',
            0,
        ),
        'rule_batch_size': getattr(
            train_stats,
            'rule_batch_size',
            0,
        ),
        'counterfactual_batch_size': getattr(
            train_stats,
            'counterfactual_batch_size',
            0,
        ),
        'shapley_batch_size': getattr(
            train_stats,
            'shapley_batch_size',
            0,
        ),
        'rule_pair_accuracy': getattr(
            train_stats,
            'rule_pair_accuracy',
            0.0,
        ),
        'rule_margin_satisfaction_rate': getattr(
            train_stats,
            'rule_margin_satisfaction_rate',
            0.0,
        ),
        'counterfactual_sign_accuracy': getattr(
            train_stats,
            'counterfactual_sign_accuracy',
            0.0,
        ),
        'counterfactual_mean_abs_error': getattr(
            train_stats,
            'counterfactual_mean_abs_error',
            0.0,
        ),
        'mean_q': train_stats.mean_q,
        'mean_target': train_stats.mean_target,
        'mean_reward': train_stats.mean_reward,
        'mean_abs_td_error': train_stats.mean_abs_td_error,
        'bootstrap_count': train_stats.bootstrap_count,
        'grad_norm': train_stats.grad_norm,
        'target_synced': int(train_stats.target_synced),
        'collect_steps': collect_stats.steps,
        'collect_replay_transitions_emitted': getattr(
            collect_stats,
            'replay_transitions_emitted',
            collect_stats.steps,
        ),
        'collect_n_step_pending_count': getattr(
            collect_stats,
            'n_step_pending_count',
            0,
        ),
        'collect_n_step_forced_flush_emitted': getattr(
            collect_stats,
            'n_step_forced_flush_emitted',
            0,
        ),
        'collect_causal_samples_emitted': getattr(
            collect_stats,
            'causal_samples_emitted',
            0,
        ),
        'collect_rule_causal_input_events': getattr(
            collect_stats,
            'causal_rule_input_event_count',
            0,
        ),
        'collect_rule_causal_eligible_events': getattr(
            collect_stats,
            'causal_rule_eligible_event_count',
            0,
        ),
        'collect_rule_causal_budget_count': getattr(
            collect_stats,
            'causal_rule_budget_count',
            0,
        ),
        'collect_rule_causal_skip_reasons': json.dumps(
            dict(getattr(
                collect_stats,
                'causal_rule_skip_reason_counts',
                (),
            )),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
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
        'collect_p95_abs_potential_shaping_reward': getattr(
            collect_stats,
            'p95_abs_potential_shaping_reward',
            0.0,
        ),
        'collect_state_analysis_calls': state_analysis_calls,
        'collect_state_analysis_seconds': state_analysis_seconds,
        'collect_mean_state_analysis_seconds': mean_state_analysis_seconds,
        'collect_state_analysis_cache_hits': state_analysis_cache_hits,
        'collect_state_analysis_cache_hit_rate': state_analysis_cache_hit_rate,
        'collect_state_analysis_degraded_count': (
            state_analysis_degraded_count
        ),
        'collect_state_analysis_degraded_rate': (
            state_analysis_degraded_rate
        ),
        'actor_inference_requests': actor_stats.get(
            'requests',
            0,
        ),
        'actor_inference_batches': actor_stats.get(
            'batches',
            0,
        ),
        'actor_inference_mean_batch_size': actor_stats.get(
            'mean_batch_size',
            0.0,
        ),
        'actor_inference_max_batch': actor_stats.get(
            'max_batch',
            0,
        ),
        'actor_inference_seconds': actor_stats.get(
            'seconds',
            0.0,
        ),
        'collect_attribution_tracker_calls': attribution_tracker_calls,
        'collect_attribution_tracker_seconds': attribution_tracker_seconds,
        'collect_mean_attribution_tracker_seconds': (
            mean_attribution_tracker_seconds
        ),
        'collect_attribution_events_created': int(getattr(
            collect_stats,
            'attribution_events_created',
            0,
        )),
        'collect_attribution_events_confirmed': int(getattr(
            collect_stats,
            'attribution_events_confirmed',
            0,
        )),
        'collect_attribution_events_cancelled': int(getattr(
            collect_stats,
            'attribution_events_cancelled',
            0,
        )),
        'collect_attribution_events_interrupted': int(getattr(
            collect_stats,
            'attribution_events_interrupted',
            0,
        )),
        'collect_attribution_pending_event_count': int(getattr(
            collect_stats,
            'attribution_pending_event_count',
            0,
        )),
        'collect_attribution_lineage_merge_count': int(getattr(
            collect_stats,
            'attribution_lineage_merge_count',
            0,
        )),
        'collect_attribution_chain_merge_count': int(getattr(
            collect_stats,
            'attribution_chain_merge_count',
            0,
        )),
        'collect_attribution_max_chain_depth': int(getattr(
            collect_stats,
            'attribution_max_chain_depth',
            0,
        )),
        'collect_mean_attribution_delay': mean_attribution_delay,
        'collect_p95_attribution_delay': p95_attribution_delay,
        'collect_attribution_event_status_counts': json.dumps(
            {
                event_type: dict(sorted(status_counts.items()))
                for event_type, status_counts
                in sorted(attribution_event_status_counts.items())
            },
            ensure_ascii=False,
            separators=(',', ':'),
        ),
        'collect_attribution_confidence_tier_counts': json.dumps(
            attribution_confidence_tier_counts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
        'collect_merge_level_counts': json.dumps(
            merge_level_counts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
        'collect_max_fruit_level': int(getattr(
            collect_stats,
            'max_fruit_level',
            0,
        )),
        'collect_counterfactual_snapshot_calls': int(getattr(
            collect_stats,
            'counterfactual_snapshot_calls',
            0,
        )),
        'collect_counterfactual_snapshot_seconds': float(getattr(
            collect_stats,
            'counterfactual_snapshot_seconds',
            0.0,
        )),
        'collect_counterfactual_snapshot_failures': int(getattr(
            collect_stats,
            'counterfactual_snapshot_failures',
            0,
        )),
        'collect_counterfactual_history_evictions': int(getattr(
            collect_stats,
            'counterfactual_history_evictions',
            0,
        )),
        'collect_counterfactual_history_size': int(getattr(
            collect_stats,
            'counterfactual_history_size',
            0,
        )),
        'collect_counterfactual_proposal_build_calls': int(getattr(
            collect_stats,
            'counterfactual_proposal_build_calls',
            0,
        )),
        'collect_counterfactual_proposal_build_seconds': float(getattr(
            collect_stats,
            'counterfactual_proposal_build_seconds',
            0.0,
        )),
        'collect_counterfactual_proposal_input_events': int(getattr(
            collect_stats,
            'counterfactual_proposal_input_event_count',
            0,
        )),
        'collect_counterfactual_proposal_confirmed_events': int(getattr(
            collect_stats,
            'counterfactual_proposal_confirmed_event_count',
            0,
        )),
        'collect_counterfactual_proposal_budget_count': int(getattr(
            collect_stats,
            'counterfactual_proposal_budget_count',
            0,
        )),
        'collect_counterfactual_proposals_generated': int(getattr(
            collect_stats,
            'counterfactual_proposals_generated',
            0,
        )),
        'collect_counterfactual_proposals_transfer_selected': int(getattr(
            collect_stats,
            'counterfactual_proposals_transfer_selected',
            0,
        )),
        'collect_counterfactual_proposals_transfer_throttled': int(getattr(
            collect_stats,
            'counterfactual_proposals_transfer_throttled',
            0,
        )),
        'collect_counterfactual_proposal_skip_reasons': json.dumps(
            dict(getattr(
                collect_stats,
                'counterfactual_proposal_skip_reason_counts',
                (),
            )),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
        'collect_counterfactual_proposals_serialized': int(getattr(
            collect_stats,
            'counterfactual_proposals_serialized',
            0,
        )),
        'collect_counterfactual_proposal_serialized_bytes': int(getattr(
            collect_stats,
            'counterfactual_proposal_serialized_bytes',
            0,
        )),
        'counterfactual_enabled': int(bool(getattr(
            counterfactual_stats,
            'enabled',
            False,
        ))),
        'counterfactual_worker_count': int(getattr(
            counterfactual_stats,
            'worker_count',
            0,
        )),
        'counterfactual_proposals_received': int(getattr(
            cf_cumulative,
            'proposals_received',
            0,
        )),
        'counterfactual_proposals_admitted': int(getattr(
            cf_cumulative,
            'proposals_admitted',
            0,
        )),
        'counterfactual_proposals_rejected': int(getattr(
            cf_cumulative,
            'proposals_rejected',
            0,
        )),
        'counterfactual_pending_tasks': int(getattr(
            counterfactual_stats,
            'pending_task_count',
            0,
        )),
        'counterfactual_admission_slots_used': int(getattr(
            cf_scheduler,
            'admission_slots_used',
            0,
        )),
        'counterfactual_admission_slots_available': int(getattr(
            cf_scheduler,
            'admission_slots_available',
            0,
        )),
        'counterfactual_candidate_pool_capacity': int(getattr(
            counterfactual_stats,
            'candidate_pool_capacity',
            0,
        )),
        'counterfactual_candidate_pool_count': int(getattr(
            counterfactual_stats,
            'candidate_pool_count',
            0,
        )),
        'counterfactual_candidate_offers': int(getattr(
            cf_cumulative,
            'candidate_offers',
            0,
        )),
        'counterfactual_candidate_pool_evictions': int(getattr(
            cf_cumulative,
            'candidate_pool_evictions',
            0,
        )),
        'counterfactual_candidate_dispatch_attempts': int(getattr(
            cf_cumulative,
            'candidate_dispatch_attempts',
            0,
        )),
        'counterfactual_candidate_dispatch_admitted': int(getattr(
            cf_cumulative,
            'candidate_dispatch_admitted',
            0,
        )),
        'counterfactual_candidate_close_dropped': int(getattr(
            cf_cumulative,
            'candidate_close_dropped',
            0,
        )),
        'counterfactual_results_completed': int(getattr(
            cf_cumulative,
            'results_completed',
            0,
        )),
        'counterfactual_results_partial': int(getattr(
            cf_cumulative,
            'results_partial',
            0,
        )),
        'counterfactual_results_failed': int(getattr(
            cf_cumulative,
            'results_failed',
            0,
        )),
        'counterfactual_reproduction_passed': int(getattr(
            cf_cumulative,
            'reproduction_passed',
            0,
        )),
        'counterfactual_reproduction_failed': int(getattr(
            cf_cumulative,
            'reproduction_failed',
            0,
        )),
        'counterfactual_numeric_jitter_dropped': int(getattr(
            cf_cumulative,
            'numeric_jitter_dropped',
            0,
        )),
        'counterfactual_semantic_divergence_dropped': int(getattr(
            cf_cumulative,
            'semantic_divergence_dropped',
            0,
        )),
        **{
            f'counterfactual_numeric_jitter_max_{suffix}_error': float(
                getattr(
                    cf_cumulative,
                    f'numeric_jitter_max_{suffix}_error',
                    0.0,
                )
            )
            for suffix in (
                'merge_event_position',
                'fruit_position',
                'linear_velocity',
                'orientation',
                'angular_velocity')
        },
        'counterfactual_label_ready_results': int(getattr(
            cf_cumulative,
            'label_ready_results',
            0,
        )),
        'counterfactual_samples_inserted': int(getattr(
            cf_cumulative,
            'samples_inserted',
            0,
        )),
        'counterfactual_tokens_reserved': int(getattr(
            cf_scheduler,
            'tokens_reserved',
            0,
        )),
        'counterfactual_tokens_consumed': int(getattr(
            cf_scheduler,
            'tokens_consumed',
            0,
        )),
        'counterfactual_tokens_refunded': int(getattr(
            cf_scheduler,
            'tokens_refunded',
            0,
        )),
        'counterfactual_actual_token_ratio': float(getattr(
            counterfactual_stats,
            'actual_token_ratio',
            0.0,
        )),
        'counterfactual_projected_token_ratio': float(getattr(
            counterfactual_stats,
            'projected_token_ratio',
            0.0,
        )),
        'counterfactual_hard_budget_respected': int(bool(getattr(
            counterfactual_stats,
            'hard_budget_respected',
            True,
        ))),
        'counterfactual_circuit_open': int(bool(getattr(
            counterfactual_stats,
            'circuit_open',
            False,
        ))),
        'counterfactual_drop_reasons': json.dumps(
            dict(getattr(cf_cumulative, 'drop_reason_counts', ())),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
        'counterfactual_failure_reasons': json.dumps(
            dict(getattr(
                cf_cumulative,
                'failure_reason_counts',
                (),
            )),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
        'counterfactual_failure_diagnostic_codes': json.dumps(
            dict(getattr(
                cf_cumulative,
                'failure_diagnostic_code_counts',
                (),
            )),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
        'counterfactual_failure_trigger_reasons': json.dumps(
            dict(getattr(
                cf_cumulative,
                'failure_trigger_reason_counts',
                (),
            )),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
        'shapley_enabled': int(bool(shapley_stats.get('enabled', False))),
        'shapley_events_observed': int(shapley_stats.get(
            'events_observed',
            0,
        )),
        'shapley_events_selected': int(shapley_stats.get(
            'events_selected',
            0,
        )),
        'shapley_tasks_submitted': int(shapley_stats.get(
            'tasks_submitted',
            0,
        )),
        'shapley_tasks_completed': int(shapley_stats.get(
            'tasks_completed',
            0,
        )),
        'shapley_tasks_failed': int(shapley_stats.get(
            'tasks_failed',
            0,
        )),
        'shapley_terminal_dropped': int(shapley_stats.get(
            'terminal_dropped',
            0,
        )),
        'shapley_reproduction_passed': int(shapley_stats.get(
            'reproduction_passed',
            0,
        )),
        'shapley_reproduction_failed': int(shapley_stats.get(
            'reproduction_failed',
            0,
        )),
        'shapley_numeric_jitter_dropped': int(shapley_stats.get(
            'numeric_jitter_dropped',
            0,
        )),
        'shapley_semantic_divergence_dropped': int(shapley_stats.get(
            'semantic_divergence_dropped',
            0,
        )),
        **{
            f'shapley_numeric_jitter_max_{suffix}_error': float(
                shapley_stats.get(
                    f'numeric_jitter_max_{suffix}_error',
                    0.0,
                )
            )
            for suffix in (
                'merge_event_position',
                'fruit_position',
                'linear_velocity',
                'orientation',
                'angular_velocity')
        },
        'shapley_samples_inserted': int(shapley_stats.get(
            'samples_inserted',
            0,
        )),
        'shapley_tokens_consumed': int(shapley_stats.get(
            'tokens_consumed',
            0,
        )),
        'shapley_pending_count': int(shapley_stats.get(
            'pending_count',
            0,
        )),
        'shapley_drop_reasons': json.dumps(
            shapley_stats.get('drop_reason_counts', {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
        'train_step_seconds': getattr(train_stats, 'train_step_seconds', ''),
        'replay_sample_seconds': getattr(train_stats, 'sample_seconds', ''),
        'current_collate_seconds': getattr(train_stats, 'current_collate_seconds', ''),
        'online_forward_seconds': getattr(train_stats, 'online_forward_seconds', ''),
        'target_compute_seconds': getattr(train_stats, 'target_compute_seconds', ''),
        'causal_sample_seconds': getattr(
            train_stats,
            'causal_sample_seconds',
            '',
        ),
        'causal_collate_seconds': getattr(
            train_stats,
            'causal_collate_seconds',
            '',
        ),
        'causal_forward_seconds': getattr(
            train_stats,
            'causal_forward_seconds',
            '',
        ),
        'backward_seconds': getattr(train_stats, 'backward_seconds', ''),
        'optimizer_seconds': getattr(train_stats, 'optimizer_seconds', ''),
        'eval_seconds': timing.get('eval_seconds', ''),
        'save_seconds': timing.get('save_seconds', ''),
        'checkpoint_bytes': timing.get('checkpoint_bytes', ''),
        'checkpoint_pruned_count': timing.get(
            'checkpoint_pruned_count',
            '',
        ),
        'checkpoint_step_materialization': timing.get(
            'checkpoint_step_materialization',
            '',
        ),
        'checkpoint_extra_materialization': timing.get(
            'checkpoint_extra_materialization',
            '',
        ),
        'plot_seconds': timing.get('plot_seconds', ''),
        'replay_mode': replay_stats.get('mode', ''),
        'replay_hot_count': replay_stats.get('hot_count', ''),
        'replay_cold_count': replay_stats.get('cold_count', ''),
        'replay_pending_cold_count': replay_stats.get('pending_cold_count', ''),
        'replay_cold_segments': replay_stats.get('cold_segment_count', ''),
        'replay_cold_cache_count': replay_stats.get('cold_cache_count', ''),
        'causal_replay_size': causal_replay_stats.get('total_count', 0),
        'causal_replay_positive_count': causal_strata.get(
            'positive_setup',
            0,
        ),
        'causal_replay_negative_count': causal_strata.get(
            'negative_blocking',
            0,
        ),
        'causal_replay_counterfactual_count': causal_strata.get(
            'counterfactual',
            0,
        ),
        'causal_replay_rule_count': causal_supervision.get('rule', 0),
        'causal_replay_cf_count': causal_supervision.get(
            'counterfactual',
            0,
        ),
        'causal_replay_shapley_count': causal_supervision.get(
            'shapley',
            0,
        ),
        'causal_replay_cause_type_counts': json.dumps(
            causal_replay_stats.get('cause_type_counts', {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        ),
        'causal_rule_empirical_agreement_count': (
            rule_empirical_agreements
        ),
        'causal_rule_empirical_disagreement_count': (
            rule_empirical_disagreements
        ),
        'causal_rule_empirical_agreement_rate': (
            rule_empirical_agreements / rule_empirical_comparisons
            if rule_empirical_comparisons > 0
            else ''
        ),
        'causal_replay_shared_tensor_bytes': causal_replay_stats.get(
            'estimated_unique_graph_bytes',
            0,
        ),
        'causal_replay_saved_tensor_bytes': causal_replay_stats.get(
            'estimated_graph_sharing_saved_bytes',
            0,
        ),
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
        'updates_per_second': completed_updates / elapsed,
        'env_steps_per_second': completed_env_steps / elapsed,
    }

    for reward_field, metric_field in REWARD_BREAKDOWN_METRIC_FIELDS:
        row[metric_field] = (
            collect_stats.mean_reward_breakdown(reward_field)
            if collect_stats.steps > 0
            else ''
        )

    return row


def load_resume_training_state(
        checkpoint_path,
        *,
        args,
        device,
        online_model,
        target_model,
        optimizer,
        trainer,
        replay_buffer,
        causal_replay_buffer):
    """可信加载一次训练状态；所有校验在训练日志打开前完成。"""

    payload = load_training_checkpoint(
        checkpoint_path,
        current_config=vars(args),
        mutable_fields=DEFAULT_RESUME_MUTABLE_FIELDS,
        map_location='cpu',
        restore_rng=False,
        components={
            'replay_buffer': replay_buffer,
            'causal_replay_buffer': causal_replay_buffer,
        },
        strict_components=True,
    )
    state = payload['training_state']
    manifest = RunManifest.from_dict(payload['run_manifest'])
    required = {
        'online_model',
        'target_model',
        'optimizer',
        'update_step',
        'env_steps',
        'epsilon',
    }
    missing = required.difference(state)
    if missing:
        raise ValueError(
            f'resume training state is missing: {sorted(missing)!r}'
        )
    update_step = int(state['update_step'])
    trainer_update_step = int(
        state.get('trainer_update_step', update_step)
    )
    if update_step < 0 or trainer_update_step != update_step:
        raise ValueError(
            'checkpoint update_step/trainer_update_step mismatch'
        )
    if update_step >= int(args.total_updates):
        raise ValueError(
            'resume checkpoint has already reached requested '
            f'--total-updates ({update_step} >= {args.total_updates})'
        )
    env_steps = int(state['env_steps'])
    if env_steps < 0:
        raise ValueError('checkpoint env_steps must be non-negative')

    online_model.load_state_dict(state['online_model'], strict=True)
    target_model.load_state_dict(state['target_model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    # Optimizer state may have been deserialized on CPU. Make the destination
    # explicit so a CUDA resume cannot fail only at the first Adam update.
    for optimizer_state in optimizer.state.values():
        for name, value in tuple(optimizer_state.items()):
            if isinstance(value, torch.Tensor):
                optimizer_state[name] = value.to(device)
    trainer.restore_update_step(trainer_update_step)
    epsilon_schedule_total_updates = int(state.get(
        'epsilon_schedule_total_updates',
        manifest.config.get('total_updates', args.total_updates),
    ))
    if epsilon_schedule_total_updates <= 0:
        raise ValueError(
            'checkpoint epsilon_schedule_total_updates must be positive'
        )

    return {
        'payload': payload,
        'manifest': manifest,
        'update_step': update_step,
        'env_steps': env_steps,
        'epsilon': float(state['epsilon']),
        'epsilon_schedule_total_updates': (
            epsilon_schedule_total_updates
        ),
        'latest_metrics': dict(state.get('latest_metrics') or {}),
        'best_eval_score': float(
            state.get('best_eval_score', float('-inf'))
        ),
        'best_eval_update': int(state.get('best_eval_update', 0)),
        'rng_state': payload['rng_state'],
        'replay_resume': (
            payload.get('component_snapshots', {})
            .get('replay_buffer', {})
            .get('state', {})
        ),
    }


def load_initial_model_weights(
        checkpoint_path,
        *,
        args,
        online_model,
        target_model):
    """只继承可信 checkpoint 的策略权重，返回可写入 JSON 的迁移报告。

    这是一个新 run 的初始化操作，不是 resume。加载器会完整校验版本化
    checkpoint，但刻意不传 components、也不恢复 RNG；旧 optimizer、主/因果
    replay、update/env 计数和 epsilon 因而不会进入新尺寸训练。
    """

    payload = load_training_checkpoint(
        checkpoint_path,
        map_location='cpu',
        restore_rng=False,
        components=None,
    )
    training_state = None
    try:
        manifest = RunManifest.from_dict(payload['run_manifest'])
        source_config = manifest.config
        architecture_fields = (
            'hidden_dim',
            'message_layers',
            'activation',
            'dropout',
        )
        mismatches = []
        for field_name in architecture_fields:
            if field_name not in source_config:
                mismatches.append(
                    f'{field_name}: source=<missing> '
                    f'target={getattr(args, field_name)!r}'
                )
                continue
            source_value = source_config[field_name]
            target_value = getattr(args, field_name)
            if source_value != target_value:
                mismatches.append(
                    f'{field_name}: source={source_value!r} '
                    f'target={target_value!r}'
                )
        if mismatches:
            raise ValueError(
                'initialization checkpoint model architecture mismatch: '
                + '; '.join(mismatches)
            )

        training_state = payload['training_state']
        if 'online_model' not in training_state:
            raise ValueError(
                'initialization checkpoint has no online_model state'
            )
        source_update_step = int(training_state.get('update_step', -1))
        if source_update_step < 0:
            raise ValueError(
                'initialization checkpoint update_step must be non-negative'
            )

        # strict=True 同时验证所有 tensor 键和 shape。目标网络从同一组权重复制，
        # 而不是继承旧 run 中可能滞后若干 update 的 target_model。
        online_model.load_state_dict(
            training_state['online_model'],
            strict=True,
        )
        target_model.load_state_dict(
            online_model.state_dict(),
            strict=True,
        )

        # 首次 250K 的 checkpoint 生成于几何字段进入 manifest 之前；
        # 缺字段时按其真实旧场地记录，而不是误报为当前新默认值。
        default_width, default_height = LEGACY_WINDOW_SIZE
        report = {
            'mode': 'weights_only',
            'source_checkpoint': str(Path(checkpoint_path).resolve()),
            'source_checkpoint_bytes': int(
                Path(checkpoint_path).stat().st_size
            ),
            'source_manifest_created_at_utc': manifest.created_at_utc,
            'source_config_fingerprint': manifest.config_fingerprint,
            'source_update_step': source_update_step,
            'source_env_steps': int(
                training_state.get('env_steps', 0)
            ),
            'source_geometry': {
                'board_width': int(source_config.get(
                    'board_width',
                    default_width,
                )),
                'board_height': int(source_config.get(
                    'board_height',
                    default_height,
                )),
                'spawn_y': int(source_config.get(
                    'spawn_y',
                    LEGACY_SPAWN_LINE_Y,
                )),
            },
            'target_geometry': {
                'board_width': int(args.board_width),
                'board_height': int(args.board_height),
                'spawn_y': int(args.spawn_y),
            },
            'inherited_state': [
                'online_model',
                'target_model_copied_from_online',
            ],
            'reset_state': [
                'optimizer',
                'td_replay',
                'causal_replay',
                'rng',
                'update_step',
                'env_steps',
                'epsilon_schedule',
                'best_eval',
                'rollout_episodes',
            ],
        }
    finally:
        # 正式 checkpoint 还包含数 GB replay。权重复制完成后尽快释放反序列化
        # payload，避免随后启动 rollout/反事实 worker 时保留无用内存峰值。
        training_state = None
        payload = None
        gc.collect()
    return report


def train(args):
    """执行完整训练流程。"""

    validate_args(args)
    resume_checkpoint = resolve_resume_location(args)
    init_checkpoint = resolve_initialization_location(args)
    device = resolve_device(args.device)
    run_dir = create_run_dir(
        args.run_dir,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite_run_dir),
    )

    # 设置 MPLCONFIGDIR 要在首次 import pyplot 前完成。
    os.environ.setdefault('MPLCONFIGDIR', str((run_dir / 'mplconfig').resolve()))

    set_random_seeds(args.seed)

    env_config = build_env_config(args)
    replay_buffer = build_replay_buffer(args, run_dir)
    causal_replay_buffer = build_causal_replay_buffer(args)

    online_model = build_model(args).to(device)
    target_model = build_model(args).to(device)
    optimizer = torch.optim.Adam(online_model.parameters(), lr=args.learning_rate)

    grad_clip_norm = None if args.grad_clip_norm == 0 else args.grad_clip_norm
    trainer_config = DQNTrainerConfig(
        gamma=args.gamma,
        n_step=args.n_step,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
        grad_clip_norm=grad_clip_norm,
        causal_batch_size=args.causal_batch_size,
        causal_update_interval=args.causal_update_interval,
        lambda_rule=args.lambda_rule,
        lambda_counterfactual=args.lambda_cf,
        lambda_structural=args.lambda_structural,
        counterfactual_return_scale=args.counterfactual_return_scale,
        counterfactual_target_clip=args.counterfactual_target_clip,
    )
    if env_config.reward_config.gamma != trainer_config.gamma:
        raise RuntimeError(
            'Reward V2 gamma must exactly match DQN trainer gamma'
        )
    trainer = DQNTrainer(
        online_model=online_model,
        target_model=target_model,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        config=trainer_config,
        causal_replay_buffer=causal_replay_buffer,
    )
    resume_state = None
    initialization_state = None
    if resume_checkpoint is not None:
        resume_state = load_resume_training_state(
            resume_checkpoint,
            args=args,
            device=device,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
            trainer=trainer,
            replay_buffer=replay_buffer,
            causal_replay_buffer=causal_replay_buffer,
        )
    elif init_checkpoint is not None:
        initialization_state = load_initial_model_weights(
            init_checkpoint,
            args=args,
            online_model=online_model,
            target_model=target_model,
        )
    resume_update_step = (
        resume_state['update_step'] if resume_state is not None else 0
    )
    epsilon_schedule_total_updates = (
        resume_state['epsilon_schedule_total_updates']
        if resume_state is not None
        else int(args.total_updates)
    )
    env_steps = (
        resume_state['env_steps'] if resume_state is not None else 0
    )
    collector = build_collector(
        args,
        env_config,
        replay_buffer,
        online_model,
        causal_replay_buffer=causal_replay_buffer,
        # 恢复后的环境从新 episode 边界开始；使用历史 env_steps 之后的
        # episode id，避免与已恢复因果样本的稳定键发生碰撞。
        episode_id_start=(
            env_steps + 1 if resume_state is not None else 0
        ),
        # 版本按“已经完成的 optimizer update”命名。恢复同一 checkpoint
        # 会得到同一版本和同一权重，不会重新从 parallel-sync-1 开始碰撞。
        policy_version=online_policy_version(resume_update_step),
    )
    counterfactual_config = build_counterfactual_config(args)
    shapley_config = build_shapley_config(args)
    counterfactual_coordinator, shapley_worker_count = (
        build_counterfactual_coordinator(
            args,
            causal_replay_buffer,
        )
    )
    refresh_counterfactual_target(
        counterfactual_coordinator,
        target_model,
        args,
        env_config,
        update_step=resume_update_step,
    )
    shapley_coordinator = (
        LocalShapleyCoordinator(
            causal_replay_buffer=causal_replay_buffer,
            shared_budget=counterfactual_coordinator,
            config=shapley_config,
            pending_capacity=args.shapley_pending_capacity,
        )
        if args.shapley_enabled
        else None
    )
    run_manifest = (
        resume_state['manifest']
        if resume_state is not None
        else create_run_manifest(
            vars(args),
            metadata={
                'experiment': 'first-large-causal-attribution',
                'git': _git_metadata(),
                'counterfactual_fingerprint': (
                    counterfactual_config.fingerprint
                ),
                'shapley_fingerprint': shapley_config.fingerprint,
                'state_analyzer_fingerprint': (
                    env_config.state_analyzer_config.fingerprint
                ),
                'weights_initialization': initialization_state,
            },
        )
    )
    config_output_path = write_config(
        run_dir,
        args,
        run_manifest=run_manifest,
        env_config=env_config,
        trainer_config=trainer_config,
        counterfactual_config=counterfactual_config,
        shapley_config=shapley_config,
        resume_checkpoint=resume_checkpoint,
        init_checkpoint=init_checkpoint,
        initialization_state=initialization_state,
    )
    if initialization_state is not None:
        _atomic_json_write(
            run_dir / 'initialization.json',
            {
                'created_at': datetime.now().isoformat(
                    timespec='seconds'
                ),
                **initialization_state,
            },
        )

    metrics = MetricLogger(
        run_dir / 'metrics.csv',
        resume_update_step=(
            resume_update_step if resume_state is not None else None
        ),
    )
    episode_metrics = EpisodeLogger(
        run_dir / 'episode_metrics.csv',
        resume_update_step=(
            resume_update_step if resume_state is not None else None
        ),
    )
    latest_row = (
        resume_state['latest_metrics'] or None
        if resume_state is not None
        else None
    )
    best_eval_score = (
        resume_state['best_eval_score']
        if resume_state is not None
        else float('-inf')
    )
    best_eval_update = (
        resume_state['best_eval_update']
        if resume_state is not None
        else 0
    )
    if (
            resume_state is not None
            and latest_row
            and int(latest_row.get('update_step', -1))
            == resume_update_step
            and not any(
                int(row.get('update_step', -1)) == resume_update_step
                for row in metrics.rows
            )):
        # save_checkpoint 发生在 metrics.log 之前，崩溃窗口内这行可能只存在
        # checkpoint 中；补写一次即可保持 canonical CSV 完整。
        metrics.log(latest_row)
    if resume_state is not None:
        restore_rng_state(
            resume_state['rng_state'],
            strict_cuda=True,
        )
        resume_replay_state = resume_state['replay_resume']
        resume_report = {
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'checkpoint': str(resume_checkpoint),
            'config_record': str(config_output_path),
            'saved_update_step': resume_update_step,
            'saved_env_steps': env_steps,
            'next_update_step': resume_update_step + 1,
            'epsilon_schedule_total_updates': (
                epsilon_schedule_total_updates
            ),
            'requested_total_updates': int(args.total_updates),
            'epsilon_schedule_extended_without_reexpansion': (
                int(args.total_updates)
                > epsilon_schedule_total_updates
            ),
            'td_replay_policy': resume_replay_state.get(
                'resume_policy',
                'warm_refill',
            ),
            'td_replay_source_count': resume_replay_state.get(
                'source_total_count',
                0,
            ),
            'td_replay_restored_count': len(replay_buffer),
            'td_replay_omitted_cold_count': resume_replay_state.get(
                'omitted_cold_count',
                0,
            ),
            'causal_replay_restored_count': len(
                causal_replay_buffer
            ),
            'metrics_orphaned_rows': metrics.orphaned_row_count,
            'metrics_orphaned_backup': (
                str(metrics.orphaned_backup_path)
                if metrics.orphaned_backup_path is not None
                else None
            ),
            'episode_metrics_orphaned_rows': (
                episode_metrics.orphaned_row_count
            ),
            'episode_metrics_orphaned_backup': (
                str(episode_metrics.orphaned_backup_path)
                if episode_metrics.orphaned_backup_path is not None
                else None
            ),
            'trajectory_resume_contract': (
                'model/target/optimizer/RNG and replay hot layer are '
                'restored; rollout environments restart at a fresh '
                'episode boundary'
            ),
        }
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        _atomic_json_write(
            run_dir / f'resume_{stamp}.json',
            resume_report,
        )
    metric_window = CollectStatsWindow()

    print(f'run_dir={run_dir}', flush=True)
    print(
        f'board_geometry={args.board_width}x{args.board_height} '
        f'spawn_y={args.spawn_y}',
        flush=True,
    )
    if initialization_state is not None:
        print(
            'initialization=weights_only '
            f'source_update={initialization_state["source_update_step"]} '
            f'source={init_checkpoint}',
            flush=True,
        )
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
        f'causal_replay_capacity={causal_replay_buffer.capacity} '
        f'n_step={args.n_step} '
        f'lambda_rule={args.lambda_rule} '
        f'lambda_cf={args.lambda_cf} '
        f'lambda_structural={args.lambda_structural}',
        flush=True,
    )
    print(
        f'collector={"parallel" if isinstance(collector, ParallelRolloutCollector) else "single"} '
        f'num_envs={args.num_envs} '
        f'async_rollout={int(bool(args.async_rollout))} '
        f'centralized_actor={int(bool(args.centralized_actor_inference))} '
        f'actor_batch_size={args.actor_batch_size}',
        flush=True,
    )
    if counterfactual_coordinator is not None:
        print(
            'counterfactual_workers='
            f'{counterfactual_coordinator.worker_count} '
            f'shapley_workers={shapley_worker_count} '
            f'budget={args.counterfactual_cost_ratio:.3f}/'
            f'{args.counterfactual_hard_limit:.3f} '
            f'external_reserve='
            f'{args.counterfactual_external_token_reserve_ratio:.3f}',
            flush=True,
        )
    if isinstance(collector, ParallelRolloutCollector) and args.collect_per_update < args.num_envs:
        print(
            'warning=collect_per_update 小于 num_envs，单次采样不会用满所有 worker；'
            '建议把 --collect-per-update 调到 num_envs 的整数倍。',
            flush=True,
        )
    print(
        f'warmup_steps={args.warmup_steps} '
        f'effective_min={max(args.warmup_steps, args.batch_size + (args.n_step - 1) * args.num_envs)}',
        flush=True,
    )

    process_start_update_step = resume_update_step
    process_start_env_steps = env_steps
    start_time = time.perf_counter()
    last_progress_at = start_time
    warmup_done = 0
    warmup_total_reward = 0.0
    n_step_tail_reserve = (
        args.n_step - 1
    ) * (
        args.num_envs
        if isinstance(collector, ParallelRolloutCollector)
        else 1
    )
    warmup_target_steps = (
        0
        if resume_state is not None and trainer.is_ready()
        else max(
            args.warmup_steps,
            args.batch_size + n_step_tail_reserve,
        )
    )
    warmup_chunk_size = max(1, min(100, warmup_target_steps))
    warmup_metric_window = CollectStatsWindow()
    warmup_complete = False
    try:
        while warmup_done < warmup_target_steps or not trainer.is_ready():
            chunk_size = min(
                warmup_chunk_size,
                max(1, warmup_target_steps - warmup_done),
            )
            chunk_start_env_steps = env_steps
            warmup_stats = collector.collect_steps(
                chunk_size,
                epsilon=1.0,
            )
            warmup_metric_window.add(warmup_stats)
            process_counterfactual_rollout(
                collector,
                counterfactual_coordinator,
                warmup_stats,
                shapley_coordinator=shapley_coordinator,
            )
            warmup_done += warmup_stats.steps
            env_steps += warmup_stats.steps
            warmup_total_reward += warmup_stats.total_reward
            episode_metrics.log_collect_stats(
                warmup_stats,
                phase=(
                    'resume_warmup'
                    if resume_state is not None
                    else 'warmup'
                ),
                update_step=resume_update_step,
                start_env_steps=chunk_start_env_steps,
                epsilon=1.0,
            )
            last_progress_at = maybe_print_progress(
                args=args,
                last_progress_at=last_progress_at,
                phase='warmup',
                current=warmup_done,
                total=warmup_target_steps,
                env_steps=env_steps,
                buffer_size=len(replay_buffer),
                epsilon=1.0,
                elapsed=time.perf_counter() - start_time,
                start_update_step=0,
                start_env_steps=process_start_env_steps,
            )
        if warmup_done > 0 or resume_state is None:
            write_attribution_warmup_summary(
                run_dir,
                warmup_metric_window.to_rollout_stats(
                    buffer_size=len(replay_buffer),
                ),
                phase=(
                    'resume_warmup'
                    if resume_state is not None
                    else 'warmup'
                ),
                output_name=(
                    'attribution_resume_warmup_'
                    f'{resume_update_step:08d}.json'
                    if resume_state is not None
                    else 'attribution_warmup.json'
                ),
            )
        warmup_complete = True
    except BaseException as exc:
        try:
            write_failure_diagnostic(
                run_dir,
                exc,
                stage='warmup',
                update_step=resume_update_step,
                trainer_update_step=trainer.update_step,
                env_steps=env_steps,
                latest_metrics=latest_row,
                replay_buffer=replay_buffer,
                causal_replay_buffer=causal_replay_buffer,
                counterfactual_coordinator=(
                    counterfactual_coordinator
                ),
                shapley_coordinator=shapley_coordinator,
            )
        except BaseException as diagnostic_exc:
            print(
                'warning=failure diagnostic write failed: '
                f'{type(diagnostic_exc).__name__}: {diagnostic_exc}',
                file=sys.stderr,
                flush=True,
            )
        raise
    finally:
        if not warmup_complete:
            close_training_resources(
                run_dir=run_dir,
                replay_buffer=replay_buffer,
                collector=collector,
                metrics=metrics,
                episode_metrics=episode_metrics,
                counterfactual_coordinator=(
                    counterfactual_coordinator
                ),
                shapley_coordinator=shapley_coordinator,
                suppress_errors=sys.exc_info()[0] is not None,
            )

    print(
        f'warmup done | env_steps={env_steps} | buffer={len(replay_buffer)} '
        f'| reward={warmup_total_reward:+.2f}',
        flush=True,
    )

    pending_collect = None
    active_update_step = resume_update_step
    epsilon = (
        resume_state['epsilon']
        if resume_state is not None
        else 1.0
    )
    failure_stage = 'training'
    try:
        for update_step in range(
                resume_update_step + 1,
                args.total_updates + 1):
            active_update_step = update_step
            epsilon = scheduled_epsilon(
                update_step,
                env_steps,
                args,
                schedule_total_updates=(
                    epsilon_schedule_total_updates
                ),
            )

            # 收集训练数据
            failure_stage = 'collect'
            collect_start_env_steps = env_steps
            if isinstance(collector, ParallelRolloutCollector) and args.async_rollout:
                if pending_collect is None:
                    if should_sync_parallel_workers(
                            update_step,
                            args.worker_sync_interval,
                            model_synced=collector.model_synced):
                        collector.sync_model(
                            online_model,
                            policy_version=online_policy_version(
                                update_step - 1
                            ),
                        )
                    pending_collect = collector.start_collect_steps(
                        args.collect_per_update,
                        epsilon=epsilon,
                    )
                collect_stats = collector.finish_collect_steps(pending_collect)
                pending_collect = None
            else:
                if (
                        isinstance(collector, ParallelRolloutCollector)
                        and should_sync_parallel_workers(
                            update_step,
                            args.worker_sync_interval,
                            model_synced=collector.model_synced)):
                    collector.sync_model(
                        online_model,
                        policy_version=online_policy_version(
                            update_step - 1
                        ),
                    )
                collect_stats = collector.collect_steps(args.collect_per_update, epsilon=epsilon)
            env_steps += collect_stats.steps
            metric_window.add(collect_stats)
            process_counterfactual_rollout(
                collector,
                counterfactual_coordinator,
                collect_stats,
                shapley_coordinator=shapley_coordinator,
            )
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
                next_epsilon = scheduled_epsilon(
                    update_step + 1,
                    env_steps,
                    args,
                    schedule_total_updates=(
                        epsilon_schedule_total_updates
                    ),
                )
                if not should_sync_parallel_workers_after_train(update_step, args.worker_sync_interval):
                    pending_next_collect = collector.start_collect_steps(
                        args.collect_per_update,
                        epsilon=next_epsilon,
                    )

            # 执行一次 DQN 参数更新
            failure_stage = 'optimizer_step'
            train_stats = trainer.train_step()
            if not isinstance(collector, ParallelRolloutCollector):
                # 单进程 collector 直接引用 online_model；参数每次 optimizer
                # step 都会变化，因此 provenance 版本也必须同步前进。
                collector.set_policy_version(
                    online_policy_version(update_step)
                )
            if train_stats.target_synced:
                refresh_counterfactual_target(
                    counterfactual_coordinator,
                    target_model,
                    args,
                    env_config,
                    update_step=update_step,
                )

            if (
                    isinstance(collector, ParallelRolloutCollector)
                    and args.async_rollout
                    and update_step < args.total_updates):
                if pending_next_collect is None:
                    collector.sync_model(
                        online_model,
                        policy_version=online_policy_version(update_step),
                    )
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
                start_update_step=process_start_update_step,
                start_env_steps=process_start_env_steps,
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
                failure_stage = 'evaluation'
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
                        'completed_updates': (
                            update_step - process_start_update_step
                        ),
                        'completed_env_steps': (
                            env_steps - process_start_env_steps
                        ),
                        'eval_seconds': eval_seconds,
                    },
                    replay_stats=replay_buffer.storage_stats,
                    causal_replay_stats=(
                        causal_replay_buffer.storage_stats
                    ),
                    counterfactual_stats=(
                        counterfactual_coordinator.stats
                        if counterfactual_coordinator is not None
                        else None
                    ),
                    shapley_stats=(
                        shapley_coordinator.stats
                        if shapley_coordinator is not None
                        else None
                    ),
                    actor_stats=(
                        collector.actor_stats_snapshot()
                        if isinstance(
                            collector,
                            ParallelRolloutCollector,
                        )
                        and collector.centralized_actor_inference
                        else None
                    ),
                )
                save_seconds = 0.0
                checkpoint_bytes = ''
                checkpoint_pruned_count = ''
                checkpoint_step_materialization = ''
                checkpoint_extra_materialization = ''
                if should_save or best_updated:
                    failure_stage = 'checkpoint'
                    save_start = time.perf_counter()
                    checkpoint_result = save_checkpoint(
                        run_dir=run_dir,
                        online_model=online_model,
                        target_model=target_model,
                        optimizer=optimizer,
                        args=args,
                        update_step=update_step,
                        env_steps=env_steps,
                        epsilon=epsilon,
                        latest_metrics=latest_row,
                        step_checkpoint=should_save,
                        extra_checkpoint_name=(
                            'best.pt'
                            if best_updated
                            else None
                        ),
                        run_manifest=run_manifest,
                        trainer=trainer,
                        replay_buffer=replay_buffer,
                        causal_replay_buffer=causal_replay_buffer,
                        counterfactual_coordinator=(
                            counterfactual_coordinator
                        ),
                        shapley_coordinator=shapley_coordinator,
                        best_eval_score=best_eval_score,
                        best_eval_update=best_eval_update,
                    )
                    save_seconds += time.perf_counter() - save_start
                    checkpoint_bytes = checkpoint_result[
                        'checkpoint_bytes'
                    ]
                    checkpoint_pruned_count = checkpoint_result[
                        'pruned_count'
                    ]
                    checkpoint_step_materialization = (
                        checkpoint_result['step_materialization'] or ''
                    )
                    checkpoint_extra_materialization = (
                        checkpoint_result['extra_materialization'] or ''
                    )
                latest_row['save_seconds'] = save_seconds if save_seconds else ''
                latest_row['checkpoint_bytes'] = checkpoint_bytes
                latest_row['checkpoint_pruned_count'] = (
                    checkpoint_pruned_count
                )
                latest_row['checkpoint_step_materialization'] = (
                    checkpoint_step_materialization
                )
                latest_row['checkpoint_extra_materialization'] = (
                    checkpoint_extra_materialization
                )

                if should_plot:
                    failure_stage = 'plot'
                    plot_start = time.perf_counter()
                    row_for_plot = {field: latest_row.get(field, '') for field in METRIC_FIELDS}
                    maybe_plot_metrics(run_dir, metrics.rows + [row_for_plot], episode_metrics.rows)
                    latest_row['plot_seconds'] = time.perf_counter() - plot_start

                metrics.log(latest_row)
                metric_window.reset()

                if should_log or should_eval:
                    print_log(latest_row)
            failure_stage = 'training'

        # 正常结束时先等待已经获批的稀疏物理任务，确保最终 checkpoint 和
        # shutdown 摘要包含最后一批可信标签；异常路径仍由 finally 幂等关闭。
        if hasattr(collector, 'close'):
            collector.close()
        if shapley_coordinator is not None:
            shapley_coordinator.close(wait=True)
        if counterfactual_coordinator is not None:
            counterfactual_coordinator.close(wait=True)

        # 最后一批不足一个 segment 的冷经验必须先持久化，再构建最终
        # checkpoint。这样 next_segment_index/storage 统计与磁盘真实状态一致，
        # finally 中的第二次 flush 会成为无操作。
        failure_stage = 'replay_flush'
        replay_buffer.flush()
        failure_stage = 'checkpoint'
        final_epsilon = scheduled_epsilon(
            args.total_updates,
            env_steps,
            args,
            schedule_total_updates=(
                epsilon_schedule_total_updates
            ),
        )
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
            run_manifest=run_manifest,
            trainer=trainer,
            replay_buffer=replay_buffer,
            causal_replay_buffer=causal_replay_buffer,
            counterfactual_coordinator=(
                counterfactual_coordinator
            ),
            shapley_coordinator=shapley_coordinator,
            best_eval_score=best_eval_score,
            best_eval_update=best_eval_update,
        )
        failure_stage = 'plot'
        maybe_plot_metrics(run_dir, metrics.rows, episode_metrics.rows)
        failure_stage = 'training'
    except BaseException as exc:
        try:
            write_failure_diagnostic(
                run_dir,
                exc,
                stage=failure_stage,
                update_step=active_update_step,
                trainer_update_step=trainer.update_step,
                env_steps=env_steps,
                latest_metrics=latest_row,
                replay_buffer=replay_buffer,
                causal_replay_buffer=causal_replay_buffer,
                counterfactual_coordinator=(
                    counterfactual_coordinator
                ),
                shapley_coordinator=shapley_coordinator,
            )
        except BaseException as diagnostic_exc:
            print(
                'warning=failure diagnostic write failed: '
                f'{type(diagnostic_exc).__name__}: {diagnostic_exc}',
                file=sys.stderr,
                flush=True,
            )
        if isinstance(exc, FloatingPointError):
            try:
                save_checkpoint(
                    run_dir=run_dir,
                    online_model=online_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    args=args,
                    update_step=trainer.update_step,
                    env_steps=env_steps,
                    epsilon=epsilon,
                    latest_metrics=latest_row,
                    extra_checkpoint_name='failure_last_normal.pt',
                    run_manifest=run_manifest,
                    trainer=trainer,
                    replay_buffer=replay_buffer,
                    causal_replay_buffer=causal_replay_buffer,
                    counterfactual_coordinator=(
                        counterfactual_coordinator
                    ),
                    shapley_coordinator=shapley_coordinator,
                    best_eval_score=best_eval_score,
                    best_eval_update=best_eval_update,
                )
            except BaseException as checkpoint_exc:
                print(
                    'warning=failure checkpoint write failed: '
                    f'{type(checkpoint_exc).__name__}: {checkpoint_exc}',
                    file=sys.stderr,
                    flush=True,
                )
        raise
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            close_training_resources(
                run_dir=run_dir,
                replay_buffer=replay_buffer,
                collector=collector,
                metrics=metrics,
                episode_metrics=episode_metrics,
                counterfactual_coordinator=(
                    counterfactual_coordinator
                ),
                shapley_coordinator=shapley_coordinator,
                suppress_errors=active_exception,
            )
            if not active_exception:
                clear_active_failure_diagnostic(run_dir)
        except BaseException as cleanup_exc:
            if active_exception:
                print(
                    'warning=cleanup raised while another exception was '
                    f'active: {type(cleanup_exc).__name__}: '
                    f'{cleanup_exc}',
                    file=sys.stderr,
                    flush=True,
                )
            else:
                try:
                    write_failure_diagnostic(
                        run_dir,
                        cleanup_exc,
                        stage='cleanup',
                        update_step=active_update_step,
                        trainer_update_step=trainer.update_step,
                        env_steps=env_steps,
                        latest_metrics=latest_row,
                        replay_buffer=replay_buffer,
                        causal_replay_buffer=(
                            causal_replay_buffer
                        ),
                        counterfactual_coordinator=(
                            counterfactual_coordinator
                        ),
                        shapley_coordinator=shapley_coordinator,
                    )
                except BaseException as diagnostic_exc:
                    print(
                        'warning=cleanup failure diagnostic write failed: '
                        f'{type(diagnostic_exc).__name__}: '
                        f'{diagnostic_exc}',
                        file=sys.stderr,
                        flush=True,
                    )
                raise

    print(f'training finished | run_dir={run_dir} | env_steps={env_steps}', flush=True)
    return run_dir


def main():
    """命令行入口。"""

    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
