#!/usr/bin/env python3
"""正式大规模训练前的一次性环境、配置和确定性门禁。

本脚本只做短小的只读/临时计算，不启动训练，也不创建 run 目录。任何必需检查失败
都会返回非零退出码，并把完整结果写成 JSON，便于迁移到云服务器后重复执行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pymunk  # noqa: E402
import torch  # noqa: E402

from daxigua.core.engine import HeadlessGame  # noqa: E402
from daxigua_rl import DaxiguaEnv, GraphBuilder, ReplayBuffer  # noqa: E402
from daxigua_rl.attribution.causal_replay import (  # noqa: E402
    CausalReplayBuffer,
    CausalTransitionContext,
)
from daxigua_rl.attribution.counterfactual import (  # noqa: E402
    FrozenGNNModelConfig,
    LocalShapleyConfig,
    create_counterfactual_task,
)
from daxigua_rl.attribution.counterfactual_proposal import (  # noqa: E402
    CounterfactualHistoryEntry,
    CounterfactualProposal,
    stable_counterfactual_proposal_id,
)
from daxigua_rl.attribution.counterfactual_runner import (  # noqa: E402
    counterfactual_result_to_causal_samples,
    freeze_target_policy_payload,
    run_counterfactual_task,
)
from daxigua_rl.attribution.local_shapley_runner import (  # noqa: E402
    create_local_shapley_task,
    local_shapley_result_to_causal_samples,
    run_local_shapley_task,
)
from daxigua_rl.attribution.schema import (  # noqa: E402
    AttributionEvent,
    AttributionEventKey,
    AttributionEvidence,
    Contributor,
)
from daxigua_rl.attribution.state_analyzer import StateAnalyzer  # noqa: E402
from daxigua_rl.graph.tensor import graph_to_tensor  # noqa: E402
from daxigua_rl.scripts.train_dqn import (  # noqa: E402
    build_counterfactual_config,
    build_env_config,
    build_model,
    build_model_config,
    parse_args as parse_training_args,
    validate_args as validate_training_args,
)
from daxigua_rl.training.dqn import DQNTrainer, DQNTrainerConfig  # noqa: E402
from daxigua_rl.training.checkpointing import config_fingerprint  # noqa: E402
from daxigua_rl.training.counterfactual_coordinator import (  # noqa: E402
    effective_cpu_count,
    recommended_counterfactual_worker_count,
)
from daxigua_rl.training.identity import TransitionKey  # noqa: E402
from daxigua_rl.training.parallel_collector import (  # noqa: E402
    ParallelRolloutCollector,
)
from daxigua_rl.training.structural_targets import (  # noqa: E402
    STRUCTURAL_TARGET_DIMENSION_COUNT,
    STRUCTURAL_TARGET_DIMENSIONS,
    STRUCTURAL_TARGET_FULL_VALID_MASK,
    build_structural_target,
)
from daxigua_rl.training.tensor_transition import TensorTransition  # noqa: E402
from daxigua_rl.reward import merge_utility  # noqa: E402


PINNED_PYMUNK_VERSION = '7.3.0'
PINNED_CHIPMUNK_VERSION = (
    '2.0.1-ade7ed72849e60289eefb7a41e79ae6322fefaf3'
)
PINNED_TORCH_VERSION = '2.12.1+cu130'
PINNED_CUDA_RUNTIME = '13.0'

FORMAL_CONFIG_CONTRACT = {
    'run_dir': 'runs/dqn_causal_structure_h256_l4_n3_250k',
    'total_updates': 250_000,
    'epsilon_schedule': 'smooth',
    'warmup_steps': 5_000,
    'collect_per_update': 16,
    'batch_size': 128,
    'replay_capacity': 100_000,
    'hot_replay_capacity': 8_000,
    'n_step': 3,
    'causal_replay_capacity': 20_000,
    'causal_batch_size': 64,
    'causal_update_interval': 2,
    'counterfactual_workers': 6,
    'counterfactual_max_alternatives': 2,
    'shapley_candidate_limit': 3,
    'hidden_dim': 256,
    'message_layers': 4,
    'action_count': 15,
    'num_envs': 16,
    'worker_sync_interval': 100,
    'actor_batch_size': 16,
}
FORMAL_FLOAT_CONTRACT = {
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'lambda_rule': 0.15,
    'lambda_cf': 0.10,
    'lambda_structural': 0.15,
    'counterfactual_cost_ratio': 0.06,
    'counterfactual_hard_limit': 0.10,
    'counterfactual_external_token_reserve_ratio': 0.01,
    'counterfactual_cpu_core_ratio': 0.24,
    'counterfactual_proposal_sample_rate': 0.0625,
    'shapley_event_ratio_max': 0.0005,
    'actor_batch_wait_ms': 2.0,
    'actor_request_timeout_seconds': 120.0,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='验证正式训练配置、CUDA、依赖版本和 EngineSnapshot 确定性。',
    )
    parser.add_argument(
        '--config',
        default='configs/train_dqn_causal_250k.toml',
        help='待验证的正式训练 TOML。',
    )
    parser.add_argument(
        '--output',
        default='runs/preflight/latest.json',
        help='门禁 JSON 输出路径。',
    )
    parser.add_argument(
        '--snapshot-audits',
        type=int,
        default=32,
        help='连续执行多少次真实动作/快照重演比较。',
    )
    parser.add_argument(
        '--min-free-gb',
        type=float,
        default=80.0,
        help='训练输出盘最低可用空间；包含热 replay 与因果 replay checkpoint 余量。',
    )
    parser.add_argument(
        '--min-memory-gb',
        type=float,
        default=24.0,
        help='首轮 3+2 worker 配置最低物理内存；32 GiB 以上为推荐值。',
    )
    parser.add_argument(
        '--min-available-memory-gb',
        type=float,
        default=8.0,
        help='启动时最低可用物理内存，避免与其它大进程争抢并触发分页。',
    )
    return parser.parse_args(argv)


def _formal_config_audit(training_args):
    """核对首轮正式训练不可被 smoke 配置或关闭归因的配置替代。"""

    mismatches = {}
    for field_name, expected in FORMAL_CONFIG_CONTRACT.items():
        actual = getattr(training_args, field_name, None)
        if actual != expected:
            mismatches[field_name] = {
                'actual': actual,
                'expected': expected,
            }
    for field_name, expected in FORMAL_FLOAT_CONTRACT.items():
        actual = getattr(training_args, field_name, None)
        if (
                not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or not math.isclose(
                    float(actual),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )):
            mismatches[field_name] = {
                'actual': actual,
                'expected': expected,
            }
    required_flags = {
        'async_rollout': True,
        'centralized_actor_inference': True,
        'counterfactual_enabled': True,
        'shapley_enabled': True,
    }
    for field_name, expected in required_flags.items():
        actual = getattr(training_args, field_name, None)
        if actual is not expected:
            mismatches[field_name] = {
                'actual': actual,
                'expected': expected,
            }
    device = str(getattr(training_args, 'device', ''))
    if not device.startswith('cuda'):
        mismatches['device'] = {
            'actual': device,
            'expected': 'cuda or cuda:<index>',
        }
    return {
        'passed': not mismatches,
        'mismatches': mismatches,
        'value_contract': dict(FORMAL_CONFIG_CONTRACT),
        'float_contract': dict(FORMAL_FLOAT_CONTRACT),
        'required_flags': required_flags,
    }


def _check(name, passed, *, required=True, **details):
    return {
        'name': str(name),
        'passed': bool(passed),
        'required': bool(required),
        'details': details,
    }


def _physical_memory_status():
    """返回跨平台 ``(total_bytes, available_bytes)``，无法探测时为 None。"""

    if os.name == 'nt':
        try:
            import ctypes
            from ctypes import wintypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = (
                    ('dwLength', wintypes.DWORD),
                    ('dwMemoryLoad', wintypes.DWORD),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                )

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                    ctypes.byref(status)):
                return int(status.ullTotalPhys), int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    meminfo_path = Path('/proc/meminfo')
    try:
        meminfo = {}
        for line in meminfo_path.read_text(
                encoding='ascii').splitlines():
            name, value = line.split(':', 1)
            meminfo[name] = int(value.strip().split()[0]) * 1024
        total = meminfo.get('MemTotal')
        available = meminfo.get(
            'MemAvailable',
            meminfo.get('MemFree'),
        )
        if total and available is not None:
            return _apply_cgroup_memory_limit(
                int(total),
                int(available),
            )
    except (OSError, TypeError, ValueError):
        pass

    try:
        page_size = os.sysconf('SC_PAGE_SIZE')
        total_pages = os.sysconf('SC_PHYS_PAGES')
        available_pages = os.sysconf('SC_AVPHYS_PAGES')
        return _apply_cgroup_memory_limit(
            int(page_size * total_pages),
            int(page_size * available_pages),
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return None, None


def _apply_cgroup_memory_limit(total_bytes, available_bytes):
    """将宿主机内存收紧到当前 Linux cgroup 的可回收工作集余量。"""

    status = _cgroup_memory_status()
    if status is None:
        return int(total_bytes), int(available_bytes)
    return (
        min(int(total_bytes), status['limit_bytes']),
        min(
            int(available_bytes),
            status['effective_available_bytes'],
        ),
    )


def _cgroup_memory_paths():
    """返回当前进程及其祖先 cgroup 的 limit/usage/stat 路径。"""

    candidates = []

    def ancestor_dirs(root, relative_group):
        current = root / relative_group
        while True:
            yield current
            if current == root:
                break
            parent = current.parent
            if parent == current or root not in parent.parents and parent != root:
                break
            current = parent

    try:
        cgroup_lines = Path('/proc/self/cgroup').read_text(
            encoding='ascii',
        ).splitlines()
    except OSError:
        cgroup_lines = ()
    for line in cgroup_lines:
        fields = line.split(':', 2)
        if len(fields) != 3:
            continue
        _hierarchy, controllers, group_path = fields
        relative_group = group_path.strip().lstrip('/')
        if not controllers:
            for group_dir in ancestor_dirs(
                    Path('/sys/fs/cgroup'),
                    relative_group):
                candidates.append((
                    group_dir / 'memory.max',
                    group_dir / 'memory.current',
                    group_dir / 'memory.stat',
                ))
        elif 'memory' in controllers.split(','):
            root = Path('/sys/fs/cgroup/memory')
            for group_dir in ancestor_dirs(root, relative_group):
                candidates.append((
                    group_dir / 'memory.limit_in_bytes',
                    group_dir / 'memory.usage_in_bytes',
                    group_dir / 'memory.stat',
                ))
    candidates.extend((
        (
            Path('/sys/fs/cgroup/memory.max'),
            Path('/sys/fs/cgroup/memory.current'),
            Path('/sys/fs/cgroup/memory.stat'),
        ),
        (
            Path('/sys/fs/cgroup/memory/memory.limit_in_bytes'),
            Path('/sys/fs/cgroup/memory/memory.usage_in_bytes'),
            Path('/sys/fs/cgroup/memory/memory.stat'),
        ),
    ))
    # 同一个 root 路径可能同时来自 /proc/self/cgroup 与固定 fallback。
    return tuple(dict.fromkeys(candidates))


def _cgroup_memory_status(paths=None):
    """读取 cgroup 总量、原始余量和扣除 inactive_file 后的有效余量。"""

    records = []
    for limit_path, usage_path, stat_path in (
            _cgroup_memory_paths() if paths is None else tuple(paths)):
        try:
            limit_text = limit_path.read_text(
                encoding='ascii',
            ).strip()
            if limit_text == 'max':
                continue
            limit = int(limit_text)
            usage = int(usage_path.read_text(
                encoding='ascii',
            ).strip())
        except (OSError, TypeError, ValueError):
            continue
        # cgroup v1 常用一个接近 2**63 的数表示无限制。
        if limit <= 0 or limit >= (1 << 60):
            continue
        inactive_file = 0
        try:
            memory_stat = {}
            for line in stat_path.read_text(
                    encoding='ascii').splitlines():
                name, value = line.split()
                memory_stat[name] = int(value)
            inactive_file = max(
                0,
                memory_stat.get(
                    'total_inactive_file',
                    memory_stat.get('inactive_file', 0),
                ),
            )
        except (OSError, TypeError, ValueError):
            # 没有 stat 时保持旧的保守 raw headroom 语义。
            inactive_file = 0
        usage = max(0, usage)
        working_set = max(0, usage - inactive_file)
        records.append({
            'limit_bytes': limit,
            'usage_bytes': usage,
            'inactive_file_bytes': inactive_file,
            'working_set_bytes': working_set,
            'raw_available_bytes': max(0, limit - usage),
            'effective_available_bytes': max(
                0,
                limit - working_set,
            ),
            'limit_path': str(limit_path),
        })
    if not records:
        return None
    limiting_effective = min(
        records,
        key=lambda item: item['effective_available_bytes'],
    )
    limiting_raw = min(
        records,
        key=lambda item: item['raw_available_bytes'],
    )
    return {
        'limit_bytes': min(
            item['limit_bytes']
            for item in records
        ),
        'raw_available_bytes': limiting_raw[
            'raw_available_bytes'
        ],
        'effective_available_bytes': limiting_effective[
            'effective_available_bytes'
        ],
        'inactive_file_bytes': limiting_effective[
            'inactive_file_bytes'
        ],
        'working_set_bytes': limiting_effective[
            'working_set_bytes'
        ],
        'effective_limit_path': limiting_effective['limit_path'],
        'candidate_count': len(records),
    }


def _snapshot_audit(training_args, count):
    count = int(count)
    if count <= 0:
        raise ValueError('--snapshot-audits must be positive')
    game = HeadlessGame(
        fps=training_args.physics_fps,
        space_iterations=training_args.space_iterations,
        seed=training_args.seed,
    )
    game.reset(seed=training_args.seed)
    rng = random.Random(training_args.seed + 91_337)
    matches = 0
    mismatch_codes = {}
    maximum_position_error = 0.0
    reset_required = False

    for _audit_index in range(count):
        if reset_required or not game.alive:
            game.reset(seed=training_args.seed + _audit_index + 1)
            reset_required = False
        snapshot = game.capture_snapshot()
        candidates = game.get_action_candidates(
            training_args.action_count
        )
        action = candidates[rng.randrange(len(candidates))]
        expected = game.execute_action(
            action.drop_x,
            max_frames=training_args.max_physics_frames,
            stable_frames=training_args.stable_frames,
        )
        report = HeadlessGame.replay_and_compare_original_action(
            snapshot,
            expected,
            action.drop_x,
            max_frames=training_args.max_physics_frames,
            stable_frames=training_args.stable_frames,
        )
        maximum_position_error = max(
            maximum_position_error,
            report.max_position_error,
        )
        if report.matches:
            matches += 1
        else:
            for code in report.mismatch_codes:
                mismatch_codes[code] = mismatch_codes.get(code, 0) + 1
        # 物理帧上限属于 episode 截断；此时 HeadlessGame 仍可能保持 alive，但训练
        # 环境会在下一步 reset，预检必须遵循相同边界。
        reset_required = bool(expected.physics_result.truncated)

    return {
        'requested': count,
        'matches': matches,
        'match_rate': matches / count,
        'mismatch_codes': dict(sorted(mismatch_codes.items())),
        'max_position_error': maximum_position_error,
    }


def _model_device_audit(training_args):
    device = torch.device(training_args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'formal config requests CUDA but torch.cuda.is_available() is False'
        )

    env = DaxiguaEnv(config=build_env_config(training_args))
    obs, info = env.reset(seed=training_args.seed)
    analysis = env.prepare_state_analysis()
    graph = GraphBuilder().build(
        obs,
        tuple(info['action_candidates']),
        state_analysis=analysis,
    )
    tensor_graph = graph_to_tensor(graph, device=device)
    model = build_model(training_args).to(device)
    model.train()
    model_output = model.forward_with_aux(tensor_graph)
    q_values = model_output.q_values
    structure_predictions = model_output.structure_predictions
    if q_values.shape != (training_args.action_count,):
        raise RuntimeError(
            f'model output shape mismatch: {tuple(q_values.shape)!r}'
        )
    if structure_predictions.shape != (
            training_args.action_count,
            6):
        raise RuntimeError(
            'model structural output shape mismatch: '
            f'{tuple(structure_predictions.shape)!r}'
        )
    if (
            not bool(torch.isfinite(q_values).all().item())
            or not bool(torch.isfinite(
                structure_predictions
            ).all().item())):
        raise RuntimeError('model forward produced non-finite outputs')
    loss = (
        q_values.square().mean()
        + structure_predictions.square().mean()
    )
    loss.backward()
    gradients = tuple(
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not gradients or not all(
            bool(torch.isfinite(gradient).all().item())
            for gradient in gradients):
        raise RuntimeError('model backward produced missing/non-finite gradients')
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    result = {
        'device': str(device),
        'q_shape': list(q_values.shape),
        'structural_shape': list(structure_predictions.shape),
        'loss': float(loss.detach().cpu().item()),
        'torch_version': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
    }
    if device.type == 'cuda':
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update({
            'gpu_name': properties.name,
            'gpu_total_memory_bytes': properties.total_memory,
            'gpu_compute_capability': (
                f'{properties.major}.{properties.minor}'
            ),
        })
    return result


def _centralized_actor_rollout_audit(
        training_args,
        *,
        collector_factory=ParallelRolloutCollector,
        model_factory=build_model,
        worker_count_override=None):
    """以真实同步/异步采集顺序验证集中式 actor 的完整运行链。

    正式调用必须使用配置中的 16 个 worker、正式模型和正式 device。仅测试可以通过
    ``worker_count_override`` 和工厂注入缩小规模；生产 preflight 不传这些参数。
    replay 全程只存在于内存，且无论采集或校验是否失败都会关闭 worker、actor 线程和
    multiprocessing queue。
    """

    configured_worker_count = int(training_args.num_envs)
    formal_worker_count = FORMAL_CONFIG_CONTRACT['num_envs']
    if configured_worker_count != formal_worker_count:
        raise RuntimeError(
            'centralized actor preflight requires the formal '
            f'{formal_worker_count} rollout workers, got '
            f'{configured_worker_count}'
        )
    if not bool(training_args.centralized_actor_inference):
        raise RuntimeError(
            'centralized actor preflight requires '
            'centralized_actor_inference=true'
        )
    if not bool(training_args.async_rollout):
        raise RuntimeError(
            'centralized actor preflight requires async_rollout=true'
        )

    worker_count = (
        configured_worker_count
        if worker_count_override is None
        else int(worker_count_override)
    )
    if worker_count <= 1 or worker_count > configured_worker_count:
        raise ValueError(
            'worker_count_override must be in '
            f'[2, {configured_worker_count}]'
        )

    device = torch.device(training_args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'formal config requests CUDA but torch.cuda.is_available() is False'
        )
    model = model_factory(training_args).to(device)
    model_device = next(model.parameters()).device
    if (
            model_device.type != device.type
            or (
                device.index is not None
                and model_device.index != device.index
            )):
        raise RuntimeError(
            'centralized actor source model is on the wrong device: '
            f'{model_device} != {device}'
        )

    # 两轮各让每个 worker 执行一步。epsilon=0 保证每个动作都必须收到集中 actor
    # 的 Q 响应；正式探针因此会实际完成 32 个 GPU actor 请求。
    first_step_count = worker_count
    second_step_count = worker_count
    replay = ReplayBuffer(
        capacity=max(
            first_step_count + second_step_count,
            worker_count * max(2, int(training_args.n_step)),
        ),
        seed=training_args.seed + 84_211,
    )
    collector = None
    close_result = None
    close_error = None
    audit_error = None
    result = None
    try:
        collector = collector_factory(
            worker_count=worker_count,
            env_config=build_env_config(training_args),
            replay_buffer=replay,
            model_config=build_model_config(training_args),
            model=model,
            seed=training_args.seed + 84_223,
            n_step=training_args.n_step,
            gamma=training_args.gamma,
            policy_version=None,
            causal_replay_buffer=None,
            # 本门禁只验证 rollout/actor 数据通路；物理反事实和规则因果分别由
            # full_causal_optimizer_step 与 local_shapley_physical 覆盖。
            counterfactual_enabled=False,
            counterfactual_ring_size=(
                training_args.counterfactual_snapshot_ring_size
            ),
            counterfactual_proposal_sample_rate=0.0,
            centralized_actor_inference=True,
            actor_batch_size=training_args.actor_batch_size,
            actor_batch_wait_ms=training_args.actor_batch_wait_ms,
            actor_request_timeout_seconds=(
                training_args.actor_request_timeout_seconds
            ),
        )
        initial_actor = collector.actor_stats_snapshot()
        if (
                initial_actor['requests'] != 0
                or initial_actor['batches'] != 0):
            raise RuntimeError(
                'new centralized actor collector started with non-zero stats'
            )

        collector.sync_model(
            model,
            policy_version='preflight-actor-sync-v1',
        )
        if (
                not collector.model_synced
                or collector.policy_version
                != 'preflight-actor-sync-v1'):
            raise RuntimeError(
                'first centralized actor model synchronization was rejected'
            )

        # 显式使用 start/finish，而不是只调用 collect_steps，确保正式 async
        # rollout API 本身至少被执行一次。
        handle = collector.start_collect_steps(
            first_step_count,
            epsilon=0.0,
        )
        if (
                len(handle.futures) != worker_count
                or len(handle.worker_indices) != worker_count
                or sum(handle.counts) != first_step_count):
            raise RuntimeError(
                'async rollout did not dispatch exactly one request to '
                'each configured worker'
            )
        first_stats = collector.finish_collect_steps(handle)
        after_first = collector.actor_stats_snapshot()
        if (
                first_stats.steps != first_step_count
                or first_stats.greedy_actions != first_step_count
                or first_stats.random_actions != 0
                or len(first_stats.transition_keys) != first_step_count
                or after_first['requests'] != first_step_count
                or after_first['batches'] <= 0):
            raise RuntimeError(
                'async rollout did not receive one centralized actor '
                'response per greedy environment step'
            )

        # 第二次同步后走 collect_steps 包装路径，覆盖长训中的常规
        # sync -> collect 周期，并确认累计响应继续增长。
        collector.sync_model(
            model,
            policy_version='preflight-actor-sync-v2',
        )
        if (
                collector.policy_version
                != 'preflight-actor-sync-v2'
                or getattr(collector, '_policy_sync_count', 0) < 2):
            raise RuntimeError(
                'second centralized actor model synchronization was rejected'
            )
        second_stats = collector.collect_steps(
            second_step_count,
            epsilon=0.0,
        )
        actor = collector.actor_stats_snapshot()
        expected_responses = first_step_count + second_step_count
        if (
                second_stats.steps != second_step_count
                or second_stats.greedy_actions != second_step_count
                or second_stats.random_actions != 0
                or len(second_stats.transition_keys) != second_step_count
                or actor['requests'] != expected_responses):
            raise RuntimeError(
                'second rollout did not receive one centralized actor '
                'response per greedy environment step'
            )
        if (
                actor['batches'] <= 0
                or actor['batches'] > actor['requests']
                or actor['max_batch'] < 1
                or actor['max_batch'] > training_args.actor_batch_size
                or not math.isclose(
                    actor['mean_batch_size'],
                    actor['requests'] / actor['batches'],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isfinite(actor['seconds'])
                or actor['seconds'] <= 0.0):
            raise RuntimeError(
                'centralized actor request/batch timing statistics are '
                'not self-consistent'
            )

        actor_model = getattr(collector, '_actor_model', None)
        if actor_model is None:
            raise RuntimeError(
                'centralized actor model was not created'
            )
        actor_device = next(actor_model.parameters()).device
        if actor_device != model_device:
            raise RuntimeError(
                'centralized actor model is on the wrong device: '
                f'{actor_device} != {device}'
            )

        result = {
            'device': str(device),
            'configured_worker_count': configured_worker_count,
            'exercised_worker_count': worker_count,
            'formal_worker_target': formal_worker_count,
            'sync_count': int(collector._policy_sync_count),
            'async_start_finish_exercised': True,
            'first_collect_steps': first_stats.steps,
            'second_collect_steps': second_stats.steps,
            'greedy_responses_verified': expected_responses,
            'actor_requests': actor['requests'],
            'actor_batches': actor['batches'],
            'actor_mean_batch_size': actor['mean_batch_size'],
            'actor_max_batch': actor['max_batch'],
            'actor_inference_seconds': actor['seconds'],
            'replay_size_before_close': len(replay),
        }
    except Exception as exc:
        audit_error = exc
    finally:
        if collector is not None:
            try:
                close_result = collector.close()
            except Exception as exc:
                close_error = exc

    if audit_error is not None:
        if close_error is not None:
            raise RuntimeError(
                'centralized actor audit failed and cleanup also failed: '
                f'audit={type(audit_error).__name__}: {audit_error}; '
                f'close={type(close_error).__name__}: {close_error}'
            ) from audit_error
        raise audit_error
    if close_error is not None:
        raise RuntimeError(
            'centralized actor resources did not close cleanly: '
            f'{type(close_error).__name__}: {close_error}'
        ) from close_error
    if (
            collector is None
            or not collector._closed
            or collector._actor_thread is not None
            or close_result is None
            or len(close_result) != worker_count):
        raise RuntimeError(
            'centralized actor close did not finalize every worker and '
            'join the actor service'
        )
    if len(replay) != first_step_count + second_step_count:
        raise RuntimeError(
            'centralized actor close did not flush one replay transition '
            'per verified actor response'
        )
    result.update({
        'replay_size_after_close': len(replay),
        'worker_finalizations': len(close_result),
        'closed_cleanly': True,
    })
    return result


def _require_full_structural_target(structural_target):
    """拒绝缺失任一 StateAnalysis 或物理结果维度的正式结构标签。"""

    if structural_target.valid_mask != STRUCTURAL_TARGET_FULL_VALID_MASK:
        invalid_dimensions = tuple(
            dimension
            for offset, dimension in enumerate(
                STRUCTURAL_TARGET_DIMENSIONS
            )
            if not structural_target.valid_mask & (1 << offset)
        )
        raise RuntimeError(
            'full causal optimizer preflight requires all six structural '
            'target dimensions; invalid='
            + ','.join(invalid_dimensions)
        )


def _require_full_structural_optimizer_stats(stats, batch_size):
    """确认正式 batch 的每个样本都实际消费六维结构监督。"""

    expected_valid_count = (
        int(batch_size) * STRUCTURAL_TARGET_DIMENSION_COUNT
    )
    if (
            stats.structural_sample_count != int(batch_size)
            or stats.structural_valid_count != expected_valid_count):
        raise RuntimeError(
            'full causal optimizer step did not consume all six '
            'structural dimensions for every sample: '
            f'samples={stats.structural_sample_count}/{int(batch_size)} '
            f'valid={stats.structural_valid_count}/'
            f'{expected_valid_count}'
        )
    return expected_valid_count


def _full_causal_update_audit(training_args):
    """执行真实 CF、结构标签和正式 batch optimizer step 的完整门禁。"""

    device = torch.device(training_args.device)
    env_config = build_env_config(training_args)
    torch.manual_seed(training_args.seed + 71_119)

    # 先构造一个不对称真实局面，避免空盘左右镜像令 A/B 回报恰好相同。
    game = HeadlessGame(
        fps=training_args.physics_fps,
        space_iterations=training_args.space_iterations,
        seed=training_args.seed + 17,
    )
    game.reset(
        seed=training_args.seed + 17,
        fruit_queue=(1, 1, 2, 3, 1, 2, 4, 1),
    )
    for offset in (1, 12, 3, 10):
        candidates = game.get_action_candidates(
            training_args.action_count
        )
        outcome = game.execute_action(
            candidates[offset].drop_x,
            max_frames=training_args.max_physics_frames,
            stable_frames=training_args.stable_frames,
        )
        if (
                outcome.physics_result.done
                or outcome.physics_result.truncated):
            raise RuntimeError(
                'preflight setup trajectory terminated unexpectedly'
            )

    snapshot = game.capture_snapshot()
    state = game.get_state()
    candidates = tuple(game.get_action_candidates(
        training_args.action_count
    ))
    transition_key = TransitionKey(
        0,
        0,
        state.step_count,
    )
    analyzer = StateAnalyzer(
        config=env_config.state_analyzer_config
    )
    analysis = analyzer.analyze(
        state,
        candidates,
        transition_key,
        stable_boundary=True,
    )
    graph = graph_to_tensor(
        GraphBuilder().build(
            state,
            candidates,
            state_analysis=analysis,
        ),
        dtype=torch.float16,
    )
    actual_offset = training_args.action_count // 2
    context = CausalTransitionContext(
        graph=graph,
        state_analysis=analysis,
        actual_action_offset=actual_offset,
        actual_action_index=actual_offset,
        policy_version='preflight-online-v1',
    )
    factual_outcome = game.execute_action(
        candidates[actual_offset].drop_x,
        max_frames=training_args.max_physics_frames,
        stable_frames=training_args.stable_frames,
    )
    post_state = factual_outcome.final_state
    post_analysis = analyzer.analyze(
        post_state,
        tuple(game.get_action_candidates(
            training_args.action_count
        )),
        TransitionKey(
            transition_key.worker_id,
            transition_key.episode_id,
            transition_key.step_index + 1,
        ),
        stable_boundary=bool(
            factual_outcome.physics_result.stable
            or factual_outcome.physics_result.done
        ),
        incoming_transition_key=transition_key,
    )
    structural_target = build_structural_target(
        analysis,
        post_analysis,
        physics_result=factual_outcome.physics_result,
    )
    _require_full_structural_target(structural_target)

    model_config = FrozenGNNModelConfig(
        **build_model_config(training_args)
    )
    frozen_source = build_model(training_args)
    payload = freeze_target_policy_payload(
        model=frozen_source,
        model_config=model_config,
        policy_version='preflight-target-v1',
        gamma=training_args.gamma,
        max_physics_frames=training_args.max_physics_frames,
        stable_frames=training_args.stable_frames,
        reward_config=env_config.reward_config,
        state_analyzer_config=env_config.state_analyzer_config,
    )
    cf_config = build_counterfactual_config(training_args)
    alternatives = tuple(
        offset
        for offset in (0, 3, training_args.action_count - 1)
        if offset != actual_offset
    )
    task = create_counterfactual_task(
        budget_key=AttributionEventKey(0, 0, 0),
        transition_key=transition_key,
        snapshot=snapshot,
        factual_outcome=factual_outcome,
        target_policy=payload,
        actual_action_offset=actual_offset,
        alternative_action_offsets=alternatives,
        trigger_reasons=('random_rule_audit',),
        event_utility=1.0,
        placement_confidence=0.8,
        created_real_step=cf_config.min_real_steps,
        attribution_version='preflight-causal-v1',
        config=cf_config,
        label_confidence=0.8,
        attribution_delay=1,
    )
    result = run_counterfactual_task(task)
    if not result.original_reproduced or not result.label_ready:
        raise RuntimeError(
            'physical counterfactual did not pass reproduction/label gate: '
            f'{result.failure_reason or result.status}'
        )
    samples = counterfactual_result_to_causal_samples(
        task,
        result,
        context,
    )
    if not samples:
        raise RuntimeError(
            'physical counterfactual produced no non-zero causal labels'
        )

    replay = ReplayBuffer(
        capacity=training_args.batch_size,
        seed=training_args.seed + 1,
    )
    replay.extend(
        TensorTransition(
            graph=graph,
            action_offset=actual_offset,
            reward=float(index % 3) * 0.01,
            next_graph=None,
            terminated=True,
            truncated=False,
            bootstrap_steps=training_args.n_step,
            structural_target=structural_target,
        )
        for index in range(training_args.batch_size)
    )
    causal_replay = CausalReplayBuffer(
        capacity=max(training_args.causal_batch_size, len(samples)),
        seed=training_args.seed + 3,
    )
    inserted = causal_replay.extend(samples)
    if inserted <= 0:
        raise RuntimeError('counterfactual labels were rejected by causal replay')

    online_model = build_model(training_args).to(device)
    target_model = build_model(training_args).to(device)
    optimizer = torch.optim.Adam(
        online_model.parameters(),
        lr=training_args.learning_rate,
    )
    trainer = DQNTrainer(
        online_model=online_model,
        target_model=target_model,
        replay_buffer=replay,
        optimizer=optimizer,
        causal_replay_buffer=causal_replay,
        config=DQNTrainerConfig(
            gamma=training_args.gamma,
            n_step=training_args.n_step,
            batch_size=training_args.batch_size,
            target_update_interval=(
                training_args.target_update_interval
            ),
            grad_clip_norm=(
                None
                if training_args.grad_clip_norm == 0
                else training_args.grad_clip_norm
            ),
            causal_batch_size=training_args.causal_batch_size,
            causal_update_interval=1,
            lambda_rule=training_args.lambda_rule,
            lambda_counterfactual=training_args.lambda_cf,
            lambda_structural=training_args.lambda_structural,
            counterfactual_return_scale=(
                training_args.counterfactual_return_scale
            ),
            counterfactual_target_clip=(
                training_args.counterfactual_target_clip
            ),
        ),
    )
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    stats = trainer.train_step()
    expected_structural_valid_count = (
        _require_full_structural_optimizer_stats(
            stats,
            training_args.batch_size,
        )
    )
    if (
            not math.isfinite(stats.loss)
            or stats.counterfactual_batch_size <= 0
            or stats.counterfactual_loss <= 0.0
            or stats.structural_loss <= 0.0):
        raise RuntimeError(
            'full optimizer step did not consume finite non-zero CF and '
            'structural losses'
        )
    peak_memory = (
        torch.cuda.max_memory_allocated(device)
        if device.type == 'cuda'
        else 0
    )
    return {
        'original_reproduced': result.original_reproduced,
        'counterfactual_status': result.status,
        'physical_steps': result.simulated_steps,
        'return_delta_count': len(result.return_deltas),
        'causal_samples': len(samples),
        'batch_size': stats.batch_size,
        'causal_batch_size': stats.causal_batch_size,
        'counterfactual_batch_size': (
            stats.counterfactual_batch_size
        ),
        'loss': stats.loss,
        'counterfactual_loss': stats.counterfactual_loss,
        'structural_valid_count': stats.structural_valid_count,
        'structural_expected_valid_count': (
            expected_structural_valid_count
        ),
        'structural_sample_count': stats.structural_sample_count,
        'structural_target_valid_mask': structural_target.valid_mask,
        'structural_loss': stats.structural_loss,
        'grad_norm': stats.grad_norm,
        'train_step_seconds': stats.train_step_seconds,
        'cuda_peak_memory_bytes': int(peak_memory),
    }


def _local_shapley_physical_audit(training_args):
    """强制执行一次 2-candidate、4-subset 的真实局部 Shapley。"""

    env_config = build_env_config(training_args)
    model_config = FrozenGNNModelConfig(
        **build_model_config(training_args)
    )
    payload = freeze_target_policy_payload(
        model=build_model(training_args),
        model_config=model_config,
        policy_version='preflight-shapley-target-v1',
        gamma=training_args.gamma,
        max_physics_frames=training_args.max_physics_frames,
        stable_frames=training_args.stable_frames,
        reward_config=env_config.reward_config,
        state_analyzer_config=env_config.state_analyzer_config,
    )
    game = HeadlessGame(
        fps=training_args.physics_fps,
        space_iterations=training_args.space_iterations,
        seed=training_args.seed + 29,
    )
    game.reset(
        seed=training_args.seed + 29,
        fruit_queue=(1, 1, 2, 3),
    )
    entries = []
    for actual_offset, comparison_offset in ((1, 13), (1, 7)):
        snapshot = game.capture_snapshot()
        state = game.get_state()
        candidates = tuple(game.get_action_candidates(
            training_args.action_count
        ))
        key = TransitionKey(0, 0, state.step_count)
        analysis = StateAnalyzer(
            config=env_config.state_analyzer_config
        ).analyze(
            state,
            candidates,
            key,
            stable_boundary=True,
        )
        graph = graph_to_tensor(
            GraphBuilder().build(
                state,
                candidates,
                state_analysis=analysis,
            ),
            dtype=torch.float16,
        )
        context = CausalTransitionContext(
            graph=graph,
            state_analysis=analysis,
            actual_action_offset=actual_offset,
            actual_action_index=int(
                graph.action_indices[actual_offset].item()
            ),
            policy_version='preflight-shapley-online-v1',
        )
        factual = game.execute_action(
            candidates[actual_offset].drop_x,
            max_frames=training_args.max_physics_frames,
            stable_frames=training_args.stable_frames,
        )
        if factual.physics_result.done or factual.physics_result.truncated:
            raise RuntimeError(
                'Shapley factual setup ended before full trace'
            )
        entries.append(CounterfactualHistoryEntry(
            transition_key=key,
            context=context,
            snapshot=snapshot,
            factual_outcome=factual,
            alternative_action_offsets=(comparison_offset,),
        ))
    entries = tuple(entries)
    event_id = AttributionEventKey(0, 0, 1)
    contributors = tuple(
        Contributor(
            transition_key=entry.transition_key,
            action_offset=entry.actual_action_offset,
            action_index=entry.context.actual_action_index,
            fruit_id=entry.factual_outcome.drop_result.fruit_id,
            evidence_type='preflight_local_shapley',
            raw_evidence_weight=weight,
            contribution_weight=weight,
            role='material',
        )
        for entry, weight in zip(entries, (0.55, 0.45))
    )
    event = AttributionEvent(
        event_id=event_id,
        episode_key=(0, 0),
        attribution_version='preflight-shapley-v1',
        tracker_config_fingerprint='preflight-tracker-v1',
        detected_step=0,
        resolved_step=2,
        event_type='CHAIN_TRIGGER',
        status='confirmed',
        sign=1,
        target_fruit_ids=tuple(
            entry.factual_outcome.drop_result.fruit_id
            for entry in entries
        ),
        contributors=contributors,
        utility=merge_utility(7),
        link_confidence=0.95,
        placement_confidence=0.8,
        evidence=AttributionEvidence(
            reason_codes=('multi_stage_chain',),
        ),
        budget_key=event_id,
        resolution_reason='preflight_local_shapley',
    )
    representative = contributors[-1]
    representative_entry = entries[-1]
    proposal_id = stable_counterfactual_proposal_id(
        budget_key=event_id,
        representative_event=event,
        contributor=representative,
        context=representative_entry.context,
        snapshot=representative_entry.snapshot,
        factual_outcome=representative_entry.factual_outcome,
        alternative_action_offsets=(
            representative_entry.alternative_action_offsets
        ),
        trigger_reasons=('multi_stage_chain',),
        coalition_trace_entries=entries,
        coalition_candidate_keys=tuple(
            entry.transition_key for entry in entries
        ),
    )
    proposal = CounterfactualProposal(
        proposal_id=proposal_id,
        representative_event=event,
        budget_key=event_id,
        contributor=representative,
        context=representative_entry.context,
        snapshot=representative_entry.snapshot,
        factual_outcome=representative_entry.factual_outcome,
        actual_action_offset=representative_entry.actual_action_offset,
        alternative_action_offsets=(
            representative_entry.alternative_action_offsets
        ),
        trigger_reasons=('multi_stage_chain',),
        coalition_trace_entries=entries,
        coalition_candidate_keys=tuple(
            entry.transition_key for entry in entries
        ),
        utility=event.utility,
        confidence=min(
            event.link_confidence,
            event.placement_confidence,
        ),
        delay=event.delay,
    )
    config = LocalShapleyConfig(
        event_ratio_max=1.0,
        candidate_limit=max(
            2,
            min(4, training_args.shapley_candidate_limit),
        ),
        paired_permutations=(
            training_args.shapley_paired_permutations
        ),
        minimum_candidates=2,
        minimum_utility=training_args.shapley_minimum_utility,
    )
    task = create_local_shapley_task(
        proposal,
        payload,
        config=config,
        created_real_step=2_000,
    )
    result = run_local_shapley_task(task)
    samples = local_shapley_result_to_causal_samples(task, result)
    if (
            not result.grand_reproduced
            or not result.label_ready
            or result.evaluated_subset_count != 4
            or not samples):
        raise RuntimeError(
            'local Shapley physical gate did not produce trusted labels: '
            f'status={result.status} reason={result.failure_reason!r} '
            f'subsets={result.evaluated_subset_count} '
            f'samples={len(samples)}'
        )
    return {
        'grand_reproduced': result.grand_reproduced,
        'status': result.status,
        'candidate_count': len(task.candidate_keys),
        'evaluated_subset_count': result.evaluated_subset_count,
        'permutation_count': result.permutation_count,
        'simulated_steps': result.simulated_steps,
        'sample_count': len(samples),
        'efficiency_residual': result.efficiency_residual,
        'efficiency_tolerance': result.efficiency_tolerance,
    }


def run_preflight(args):
    config_path = (PROJECT_ROOT / args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'training config not found: {config_path}')
    for field_name in (
            'min_free_gb',
            'min_memory_gb',
            'min_available_memory_gb'):
        value = float(getattr(args, field_name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f'--{field_name.replace("_", "-")} must be finite '
                'and non-negative'
            )

    checks = []
    git_metadata = _git_metadata()
    checks.append(_check(
        'git_worktree_clean',
        (
            bool(git_metadata.get('commit'))
            and bool(git_metadata.get('branch'))
            and git_metadata.get('dirty') is False
        ),
        **git_metadata,
    ))
    try:
        training_args = parse_training_args(('--config', str(config_path)))
        validate_training_args(training_args)
    except Exception as exc:
        checks.append(_check(
            'training_config',
            False,
            error=f'{type(exc).__name__}: {exc}',
            path=str(config_path),
        ))
        training_args = None
    else:
        checks.append(_check(
            'training_config',
            True,
            path=str(config_path),
            total_updates=training_args.total_updates,
            warmup_steps=training_args.warmup_steps,
            action_count=training_args.action_count,
            n_step=training_args.n_step,
            num_envs=training_args.num_envs,
            device=training_args.device,
        ))
        formal_contract = _formal_config_audit(
            training_args
        )
        checks.append(_check(
            'formal_training_config_contract',
            formal_contract.pop('passed'),
            **formal_contract,
        ))

    checks.append(_check(
        'python_version',
        sys.version_info >= (3, 11),
        version=platform.python_version(),
        executable=sys.executable,
    ))
    checks.append(_check(
        'pymunk_version',
        str(pymunk.version) == PINNED_PYMUNK_VERSION,
        actual=str(pymunk.version),
        expected=PINNED_PYMUNK_VERSION,
    ))
    checks.append(_check(
        'chipmunk_version',
        str(pymunk.chipmunk_version) == PINNED_CHIPMUNK_VERSION,
        actual=str(pymunk.chipmunk_version),
        expected=PINNED_CHIPMUNK_VERSION,
    ))
    checks.append(_check(
        'torch_cuda_build',
        (
            str(torch.__version__) == PINNED_TORCH_VERSION
            and str(torch.version.cuda) == PINNED_CUDA_RUNTIME
        ),
        torch_actual=str(torch.__version__),
        torch_expected=PINNED_TORCH_VERSION,
        cuda_runtime_actual=str(torch.version.cuda),
        cuda_runtime_expected=PINNED_CUDA_RUNTIME,
    ))

    if training_args is not None:
        try:
            model_device = _model_device_audit(training_args)
        except Exception as exc:
            checks.append(_check(
                'model_forward_backward',
                False,
                error=f'{type(exc).__name__}: {exc}',
            ))
        else:
            checks.append(_check(
                'model_forward_backward',
                True,
                **model_device,
            ))

        try:
            actor_rollout = _centralized_actor_rollout_audit(
                training_args
            )
        except Exception as exc:
            checks.append(_check(
                'centralized_actor_async_rollout',
                False,
                error=f'{type(exc).__name__}: {exc}',
            ))
        else:
            checks.append(_check(
                'centralized_actor_async_rollout',
                True,
                **actor_rollout,
            ))

        try:
            causal_update = _full_causal_update_audit(
                training_args
            )
        except Exception as exc:
            checks.append(_check(
                'full_causal_optimizer_step',
                False,
                error=f'{type(exc).__name__}: {exc}',
            ))
        else:
            checks.append(_check(
                'full_causal_optimizer_step',
                True,
                **causal_update,
            ))

        try:
            shapley_audit = _local_shapley_physical_audit(
                training_args
            )
        except Exception as exc:
            checks.append(_check(
                'local_shapley_physical',
                False,
                error=f'{type(exc).__name__}: {exc}',
            ))
        else:
            checks.append(_check(
                'local_shapley_physical',
                True,
                **shapley_audit,
            ))

        try:
            snapshot = _snapshot_audit(
                training_args,
                args.snapshot_audits,
            )
        except Exception as exc:
            checks.append(_check(
                'engine_snapshot_reproduction',
                False,
                error=f'{type(exc).__name__}: {exc}',
            ))
        else:
            checks.append(_check(
                'engine_snapshot_reproduction',
                snapshot['matches'] == snapshot['requested'],
                **snapshot,
            ))

        output_root = (
            PROJECT_ROOT
            / Path(training_args.run_dir or 'runs')
        ).resolve()
        existing_parent = output_root
        while not existing_parent.exists():
            existing_parent = existing_parent.parent
        disk = shutil.disk_usage(existing_parent)
        free_gb = disk.free / (1024 ** 3)
        checks.append(_check(
            'output_disk_free',
            free_gb >= float(args.min_free_gb),
            path=str(existing_parent),
            free_gb=free_gb,
            recommended_min_gb=float(args.min_free_gb),
        ))

        total_memory, available_memory = _physical_memory_status()
        total_memory_gb = (
            total_memory / (1024 ** 3)
            if total_memory is not None
            else None
        )
        available_memory_gb = (
            available_memory / (1024 ** 3)
            if available_memory is not None
            else None
        )
        cgroup_memory = _cgroup_memory_status()
        checks.append(_check(
            'host_physical_memory',
            (
                total_memory_gb is not None
                and available_memory_gb is not None
                and total_memory_gb >= float(args.min_memory_gb)
                and available_memory_gb
                >= float(args.min_available_memory_gb)
            ),
            total_gb=total_memory_gb,
            available_gb=available_memory_gb,
            available_basis=(
                'min(host MemAvailable, cgroup reclaim-aware headroom)'
                if cgroup_memory is not None
                else 'host MemAvailable'
            ),
            cgroup=(
                {
                    'limit_gb': (
                        cgroup_memory['limit_bytes'] / (1024 ** 3)
                    ),
                    'raw_available_gb': (
                        cgroup_memory['raw_available_bytes']
                        / (1024 ** 3)
                    ),
                    'effective_available_gb': (
                        cgroup_memory['effective_available_bytes']
                        / (1024 ** 3)
                    ),
                    'inactive_file_gb': (
                        cgroup_memory['inactive_file_bytes']
                        / (1024 ** 3)
                    ),
                    'working_set_gb': (
                        cgroup_memory['working_set_bytes']
                        / (1024 ** 3)
                    ),
                    'effective_limit_path': (
                        cgroup_memory['effective_limit_path']
                    ),
                    'candidate_count': (
                        cgroup_memory['candidate_count']
                    ),
                }
                if cgroup_memory is not None
                else None
            ),
            minimum_total_gb=float(args.min_memory_gb),
            minimum_available_gb=float(
                args.min_available_memory_gb
            ),
            recommended_total_gb=64.0,
        ))

        host_cpu_count = os.cpu_count() or 1
        effective_cpus = effective_cpu_count()
        recommended_total_workers = (
            recommended_counterfactual_worker_count(
                cpu_count=effective_cpus,
                rollout_worker_count=training_args.num_envs,
                cpu_core_ratio=(
                    training_args.counterfactual_cpu_core_ratio
                ),
            )
        )
        configured_total_workers = (
            recommended_total_workers
            if training_args.counterfactual_workers is None
            else int(training_args.counterfactual_workers)
        )
        shapley_workers = 1 if training_args.shapley_enabled else 0
        ordinary_cf_workers = (
            configured_total_workers - shapley_workers
        )
        worker_capacity = max(
            0,
            effective_cpus - training_args.num_envs - 1,
        )
        checks.append(_check(
            'counterfactual_cpu_reserve',
            (
                ordinary_cf_workers >= 1
                and configured_total_workers <= worker_capacity
            ),
            host_cpu_count=host_cpu_count,
            effective_cpu_count=effective_cpus,
            rollout_workers=training_args.num_envs,
            configured_total_physical_workers=(
                configured_total_workers
            ),
            reserved_shapley_workers=shapley_workers,
            ordinary_counterfactual_workers=ordinary_cf_workers,
            physical_worker_capacity=worker_capacity,
            recommended_total_physical_workers=(
                recommended_total_workers
            ),
            cpu_core_ratio=(
                training_args.counterfactual_cpu_core_ratio
            ),
        ))

    required_failures = tuple(
        check['name']
        for check in checks
        if check['required'] and not check['passed']
    )
    warnings = tuple(
        check['name']
        for check in checks
        if not check['required'] and not check['passed']
    )
    return {
        'created_at': datetime.now().astimezone().isoformat(
            timespec='seconds'
        ),
        'project_root': str(PROJECT_ROOT),
        'config': str(config_path),
        'config_sha256': hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        'resolved_config_fingerprint': (
            config_fingerprint(vars(training_args))
            if training_args is not None
            else None
        ),
        'git': git_metadata,
        'ready': not required_failures,
        'required_failures': required_failures,
        'warnings': warnings,
        'checks': checks,
    }


def _git_metadata():
    def run(*arguments):
        try:
            result = subprocess.run(
                ('git', *arguments),
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    return {
        'commit': run('rev-parse', 'HEAD'),
        'branch': run('branch', '--show-current'),
        'dirty': bool(run('status', '--porcelain')),
    }


def main(argv=None):
    args = parse_args(argv)
    payload = run_preflight(args)
    output_path = (PROJECT_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f'preflight_output={output_path}')
    return 0 if payload['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
