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
from daxigua_rl.training.identity import TransitionKey  # noqa: E402
from daxigua_rl.training.tensor_transition import TensorTransition  # noqa: E402
from daxigua_rl.reward import merge_utility  # noqa: E402


PINNED_PYMUNK_VERSION = '7.3.0'
PINNED_CHIPMUNK_VERSION = (
    '2.0.1-ade7ed72849e60289eefb7a41e79ae6322fefaf3'
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='验证正式训练配置、CUDA、依赖版本和 EngineSnapshot 确定性。',
    )
    parser.add_argument(
        '--config',
        default='configs/train_dqn_fast30_parallel.toml',
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
        default=40.0,
        help='训练输出盘最低可用空间；包含热 replay 与因果 replay checkpoint 余量。',
    )
    return parser.parse_args(argv)


def _check(name, passed, *, required=True, **details):
    return {
        'name': str(name),
        'passed': bool(passed),
        'required': bool(required),
        'details': details,
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
    graph = GraphBuilder().build(
        obs,
        tuple(info['action_candidates']),
    )
    tensor_graph = graph_to_tensor(graph, device=device)
    model = build_model(training_args).to(device)
    model.train()
    q_values = model(tensor_graph)
    if q_values.shape != (training_args.action_count,):
        raise RuntimeError(
            f'model output shape mismatch: {tuple(q_values.shape)!r}'
        )
    if not bool(torch.isfinite(q_values).all().item()):
        raise RuntimeError('model forward produced non-finite Q values')
    loss = q_values.square().mean()
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


def _full_causal_update_audit(training_args):
    """执行一条真实 CF 标签到 batch64 optimizer step 的完整门禁。"""

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
        GraphBuilder().build(state, candidates),
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
    if (
            not math.isfinite(stats.loss)
            or stats.counterfactual_batch_size <= 0
            or stats.counterfactual_loss <= 0.0):
        raise RuntimeError(
            'full optimizer step did not consume a finite non-zero CF loss'
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
        'grad_norm': stats.grad_norm,
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
            GraphBuilder().build(state, candidates),
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

    checks = []
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

        cpu_count = os.cpu_count() or 1
        cf_workers = max(
            0,
            min(
                math.floor(cpu_count * 0.25),
                cpu_count - training_args.num_envs - 1,
            ),
        )
        checks.append(_check(
            'counterfactual_cpu_reserve',
            cf_workers >= 1,
            cpu_count=cpu_count,
            rollout_workers=training_args.num_envs,
            available_counterfactual_workers=cf_workers,
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
        'git': _git_metadata(),
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
