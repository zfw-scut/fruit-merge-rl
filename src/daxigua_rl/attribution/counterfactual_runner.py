"""可由 Windows ``spawn`` 直接执行的反事实物理 runner。

本模块只有模块顶层函数会进入 ``ProcessPoolExecutor``。每个任务先从
``EngineSnapshot`` 重演真实动作，并用 ``HeadlessGame.compare_action_outcomes``
执行保守门禁；门禁失败时立即返回不可造标签的 failed result。门禁通过后，实际分支
和所有替代分支仅首动作不同，之后都使用任务携带的同一份冻结 target GNN 贪心策略。

这里不修改 collector、trainer 或游戏核心，也不接触在线模型。冻结 state_dict 仅按
CPU 权重加载，Reward V2、状态分析与物理步参数全部来自带指纹 payload。
"""

from __future__ import annotations

import io
import math
from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType

import torch

from daxigua.core.engine import HeadlessGame
from daxigua.core.state import EngineActionOutcome
from daxigua_rl.env import DaxiguaEnv, DaxiguaEnvConfig
from daxigua_rl.graph import GraphBuilder, GraphBuilderConfig
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.reward import RewardConfig
from daxigua_rl.training.identity import TransitionKey

from .causal_replay import (
    CausalSample,
    CausalTransitionContext,
    graph_schema_fingerprint,
    stable_budget_key,
)
from .counterfactual import (
    CounterfactualBranchResult,
    CounterfactualResult,
    CounterfactualTask,
    FrozenGNNModelConfig,
    FrozenGraphBuilderConfig,
    FrozenTargetPolicyPayload,
)
from .schema import ANALYSIS_ACTION_COUNT
from .state_analyzer import StateAnalyzerConfig


# ProcessPool 的 worker 会处理多个任务。相邻任务通常使用同一 target sync，因此保留
# 两份只读 CPU 模型可以避免反复 torch.load；小容量也防止训练中每次 target 更新都
# 永久占用内存。该缓存只存在于单个 runner 进程，不跨进程共享。
_TARGET_MODEL_CACHE = OrderedDict()
_TARGET_MODEL_CACHE_CAPACITY = 2
_CPU_STATE_DICT_FORMAT = 'counterfactual_cpu_state_dict_v1'


@dataclass(slots=True)
class _FirstStep:
    """runner 内部的一步结果，保留环境以继续同一物理分支。"""

    env: DaxiguaEnv
    reward: float
    terminated: bool
    truncated: bool
    engine_action_outcome: EngineActionOutcome


def _exception_code(exc):
    name = type(exc).__name__
    return f'exception_{name}'


def _failed_branch(
        action_offset,
        *,
        simulated_steps,
        failure_reason,
        diagnostic_codes=()):
    return CounterfactualBranchResult(
        action_offset=action_offset,
        status='failed',
        objective_return=None,
        simulated_steps=simulated_steps,
        failure_reason=failure_reason,
        diagnostic_codes=tuple(diagnostic_codes),
    )


def _failed_result(
        task,
        *,
        original_reproduced,
        failure_reason,
        branches=(),
        diagnostic_codes=()):
    branches = tuple(branches)
    return CounterfactualResult(
        task_id=task.task_id,
        status='failed',
        actual_action_offset=task.actual_action_offset,
        original_reproduced=original_reproduced,
        branches=branches,
        simulated_steps=sum(
            branch.simulated_steps
            for branch in branches
        ),
        failure_reason=failure_reason,
        diagnostic_codes=tuple(diagnostic_codes),
    )


def _torch_load_state_dict(payload):
    """安全读取内部创建的 CPU 权重 envelope，拒绝任何 GPU tensor。

    新版 PyTorch 使用 ``weights_only=True``；兼容较旧版本时只在参数不存在的
    ``TypeError`` 上回退。这里刻意不使用 ``map_location='cpu'``：静默映射会掩盖
    生产者错误地序列化 GPU tensor，导致任务契约声称 CPU、实际却不是。payload
    仍只允许项目内部可信生产者创建，不能接收网络上传的不可信 pickle。
    """

    stream = io.BytesIO(payload.state_dict_bytes)
    try:
        envelope = torch.load(
            stream,
            weights_only=True,
        )
    except TypeError:
        stream.seek(0)
        envelope = torch.load(stream)
    if not isinstance(envelope, dict):
        raise TypeError(
            'target payload must contain a CPU state_dict envelope'
        )
    if envelope.get('format') != _CPU_STATE_DICT_FORMAT:
        raise ValueError('unsupported target state_dict format')
    if envelope.get('device') != 'cpu':
        raise ValueError('target state_dict envelope must declare CPU')
    if set(envelope) != {'format', 'device', 'state_dict'}:
        raise ValueError(
            'target state_dict envelope contains unexpected fields'
        )
    state_dict = envelope.get('state_dict')
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError('target state_dict payload must contain a mapping')
    if not state_dict:
        raise ValueError('target state_dict must not be empty')

    normalized = OrderedDict()
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not name:
            raise TypeError(
                'target state_dict keys must be non-empty strings'
            )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                'target state_dict values must be tensors'
            )
        if tensor.device.type != 'cpu':
            raise ValueError(
                'target state_dict tensors must be on CPU'
            )
        if tensor.requires_grad:
            tensor = tensor.detach()
        tensor = tensor.contiguous()
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(
                f'target state_dict tensor {name!r} is non-finite'
            )
        normalized[name] = tensor
    return normalized


def _load_target_model(payload):
    """按 payload 指纹复用或严格创建冻结 CPU target model。"""

    if not isinstance(payload, FrozenTargetPolicyPayload):
        raise TypeError(
            'payload must be FrozenTargetPolicyPayload'
        )
    if payload.expected_fingerprint() != payload.fingerprint:
        raise ValueError('target policy fingerprint changed after creation')

    cached = _TARGET_MODEL_CACHE.get(payload.fingerprint)
    if cached is not None:
        _TARGET_MODEL_CACHE.move_to_end(payload.fingerprint)
        return cached

    state_dict = _torch_load_state_dict(payload)
    model = GNNQNetwork(**payload.model_config.model_kwargs)
    model.load_state_dict(state_dict, strict=True)
    model.to(device='cpu')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    _TARGET_MODEL_CACHE[payload.fingerprint] = model
    _TARGET_MODEL_CACHE.move_to_end(payload.fingerprint)
    while len(_TARGET_MODEL_CACHE) > _TARGET_MODEL_CACHE_CAPACITY:
        _TARGET_MODEL_CACHE.popitem(last=False)
    return model


def _coerce_graph_builder_config(config):
    if config is None:
        return FrozenGraphBuilderConfig()
    if isinstance(config, FrozenGraphBuilderConfig):
        return config
    if isinstance(config, GraphBuilderConfig):
        return FrozenGraphBuilderConfig(
            velocity_scale=config.velocity_scale,
            fruit_count_scale=config.fruit_count_scale,
            connect_global_node=config.connect_global_node,
        )
    raise TypeError(
        'graph_builder_config must be FrozenGraphBuilderConfig, '
        'GraphBuilderConfig, or None'
    )


def freeze_target_policy_payload(
        *,
        model,
        model_config,
        policy_version,
        gamma,
        max_physics_frames,
        stable_frames,
        reward_config=None,
        state_analyzer_config=None,
        graph_builder_config=None):
    """把一份 target GNN 封装成严格、可 pickle 的 CPU payload。

    这是任务生产者应使用的唯一便利入口。它复制而不是引用原模型 tensor，因此主线程
    后续更新 target model 不会改变已经排队任务的策略。构造末尾会在当前进程严格
    加载一次，尽早发现模型配置与 state_dict 形状不一致。
    """

    if not isinstance(model, GNNQNetwork):
        raise TypeError('model must be GNNQNetwork')
    if not isinstance(model_config, FrozenGNNModelConfig):
        raise TypeError(
            'model_config must be FrozenGNNModelConfig'
        )
    if (
            model.node_feature_dim != model_config.node_feature_dim
            or model.edge_feature_dim != model_config.edge_feature_dim
            or model.hidden_dim != model_config.hidden_dim
            or model.message_layers != model_config.message_layers):
        raise ValueError(
            'model structure does not match FrozenGNNModelConfig'
        )

    reward_config = reward_config or RewardConfig(gamma=gamma)
    if not isinstance(reward_config, RewardConfig):
        raise TypeError('reward_config must be RewardConfig')
    state_analyzer_config = (
        state_analyzer_config
        or StateAnalyzerConfig()
    )
    if not isinstance(
            state_analyzer_config,
            StateAnalyzerConfig):
        raise TypeError(
            'state_analyzer_config must be StateAnalyzerConfig'
        )
    frozen_graph_config = _coerce_graph_builder_config(
        graph_builder_config
    )

    state_dict = OrderedDict(
        (
            name,
            tensor.detach().to(device='cpu').contiguous().clone(),
        )
        for name, tensor in model.state_dict().items()
    )
    stream = io.BytesIO()
    torch.save(
        {
            'format': _CPU_STATE_DICT_FORMAT,
            'device': 'cpu',
            'state_dict': state_dict,
        },
        stream,
    )
    payload = FrozenTargetPolicyPayload.create(
        policy_version=policy_version,
        model_config=model_config,
        graph_builder_config=frozen_graph_config,
        state_dict_bytes=stream.getvalue(),
        gamma=gamma,
        max_physics_frames=max_physics_frames,
        stable_frames=stable_frames,
        reward_config=reward_config,
        state_analyzer_config=state_analyzer_config,
    )
    # 严格 roundtrip 能同时验证 state_dict bytes、全部键和 tensor shape。
    _load_target_model(payload)
    return payload


def _env_config(task):
    payload = task.target_policy
    snapshot_config = task.snapshot.config
    return DaxiguaEnvConfig(
        action_count=ANALYSIS_ACTION_COUNT,
        physics_fps=snapshot_config.fps,
        max_physics_frames=payload.max_physics_frames,
        stable_frames=payload.stable_frames,
        space_iterations=snapshot_config.space_iterations,
        reward_config=payload.reward_config,
        state_analyzer_config=payload.state_analyzer_config,
    )


def _make_branch_env(task):
    """使用环境公开桥接恢复分支，不 reset，也不修改任务中的 snapshot。"""

    return DaxiguaEnv.from_snapshot(
        task.snapshot,
        config=_env_config(task),
    )


def _graph_builder(payload):
    config = GraphBuilderConfig(
        **payload.graph_builder_config.builder_kwargs
    )
    return GraphBuilder(config=config)


def _q_values(model, graph_builder, env):
    state = env.game.get_state()
    candidates = env.action_candidates()
    graph = graph_builder.build(state, candidates)
    with torch.inference_mode():
        q_values = model(graph)
    if (
            q_values.ndim != 1
            or q_values.shape[0] != ANALYSIS_ACTION_COUNT):
        raise ValueError(
            'frozen target model must return one Q value per action'
        )
    if not bool(torch.isfinite(q_values).all().item()):
        raise ValueError(
            'frozen target model returned non-finite Q values'
        )
    return q_values


def _greedy_action(model, graph_builder, env):
    """torch.argmax 的首个最大值规则给并列 Q 提供稳定 tie-break。"""

    q_values = _q_values(model, graph_builder, env)
    return int(torch.argmax(q_values).item())


def _bootstrap_value(model, graph_builder, env):
    q_values = _q_values(model, graph_builder, env)
    return float(torch.max(q_values).item())


def _complete_branch(
        task,
        *,
        action_offset,
        first_step,
        model,
        graph_builder):
    """从已经执行的首动作继续分支，返回折扣 Reward V2 + target bootstrap。"""

    payload = task.target_policy
    gamma = payload.gamma
    objective_return = float(first_step.reward)
    discount = gamma
    simulated_steps = 1
    terminated = first_step.terminated
    truncated = first_step.truncated
    env = first_step.env

    try:
        while (
                simulated_steps < task.horizon
                and not terminated
                and not truncated):
            action_offset_next = _greedy_action(
                model,
                graph_builder,
                env,
            )
            next_key = TransitionKey(
                worker_id=task.transition_key.worker_id,
                episode_id=task.transition_key.episode_id,
                step_index=(
                    task.transition_key.step_index
                    + simulated_steps
                ),
            )
            # 从调用物理 step 起就保守计一个预算 token；即使内部异常，也不能把已
            # 尝试的物理分支当成零成本。
            attempted_steps = simulated_steps + 1
            try:
                _obs, reward, terminated, truncated, _info = env.step(
                    action_offset_next,
                    transition_key=next_key,
                )
            except Exception as exc:
                return _failed_branch(
                    action_offset,
                    simulated_steps=attempted_steps,
                    failure_reason='branch_physics_failure',
                    diagnostic_codes=(_exception_code(exc),),
                )
            objective_return += discount * float(reward)
            discount *= gamma
            simulated_steps += 1

        diagnostics = []
        status = 'completed'
        early_stopped = bool(
            (terminated or truncated)
            and simulated_steps < task.horizon
        )
        if terminated:
            diagnostics.append('terminal_no_bootstrap')
        else:
            bootstrap = _bootstrap_value(
                model,
                graph_builder,
                env,
            )
            objective_return += discount * bootstrap
            if truncated:
                status = 'partial'
                diagnostics.append('physics_truncated_bootstrap')
            else:
                diagnostics.append('horizon_target_bootstrap')

        if not math.isfinite(objective_return):
            raise ValueError('branch objective_return is non-finite')
        return CounterfactualBranchResult(
            action_offset=action_offset,
            status=status,
            objective_return=objective_return,
            simulated_steps=simulated_steps,
            terminated=terminated,
            truncated=truncated,
            early_stopped=early_stopped,
            diagnostic_codes=tuple(diagnostics),
        )
    except Exception as exc:
        return _failed_branch(
            action_offset,
            simulated_steps=simulated_steps,
            failure_reason='branch_policy_or_bootstrap_failure',
            diagnostic_codes=(_exception_code(exc),),
        )


def _first_step_or_failed(task, action_offset):
    """执行首动作；恢复失败不计物理步，step 内异常则保守计一个。"""

    try:
        env = _make_branch_env(task)
    except Exception as exc:
        return None, _failed_branch(
            action_offset,
            simulated_steps=0,
            failure_reason='snapshot_restore_failure',
            diagnostic_codes=(_exception_code(exc),),
        )
    try:
        obs, reward, terminated, truncated, info = env.step(
            action_offset,
            transition_key=task.transition_key,
        )
        outcome = info.get('engine_action_outcome')
        if not isinstance(outcome, EngineActionOutcome):
            raise RuntimeError(
                'DaxiguaEnv did not expose EngineActionOutcome'
            )
        if outcome.final_state != obs:
            raise RuntimeError(
                'engine_action_outcome final_state differs from env obs'
            )
        return _FirstStep(
            env=env,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            engine_action_outcome=outcome,
        ), None
    except Exception as exc:
        return None, _failed_branch(
            action_offset,
            simulated_steps=1,
            failure_reason='first_action_failure',
            diagnostic_codes=(_exception_code(exc),),
        )


def run_counterfactual_task(task):
    """执行一个完整任务，所有可预期异常均降级为不可伪造标签的结果。

    此函数位于模块顶层且没有闭包，可直接作为
    ``BudgetedCounterfactualScheduler`` 的 spawn runner。
    """

    if not isinstance(task, CounterfactualTask):
        raise TypeError('task must be CounterfactualTask')

    # 先只运行 actual 首动作。任何替代分支或模型加载都必须位于复现门禁之后。
    actual_first, actual_failure = _first_step_or_failed(
        task,
        task.actual_action_offset,
    )
    if actual_failure is not None:
        return _failed_result(
            task,
            original_reproduced=False,
            failure_reason='original_action_execution_failure',
            branches=(actual_failure,),
            diagnostic_codes=actual_failure.diagnostic_codes,
        )

    try:
        replay_report = HeadlessGame.compare_action_outcomes(
            task.factual_outcome,
            actual_first.engine_action_outcome,
        )
    except Exception as exc:
        failed = _failed_branch(
            task.actual_action_offset,
            simulated_steps=1,
            failure_reason='original_comparison_failure',
            diagnostic_codes=(_exception_code(exc),),
        )
        return _failed_result(
            task,
            original_reproduced=False,
            failure_reason='original_comparison_failure',
            branches=(failed,),
            diagnostic_codes=failed.diagnostic_codes,
        )

    if not replay_report.matches:
        mismatch_codes = tuple(
            f'original_mismatch_{code}'
            for code in replay_report.mismatch_codes
        )
        failed = _failed_branch(
            task.actual_action_offset,
            simulated_steps=1,
            failure_reason='original_reproduction_mismatch',
            diagnostic_codes=mismatch_codes,
        )
        return _failed_result(
            task,
            original_reproduced=False,
            failure_reason='original_reproduction_mismatch',
            branches=(failed,),
            diagnostic_codes=mismatch_codes,
        )

    # 模型错误发生在复现成功之后，但仍不得把不完整实际分支用作 label。
    try:
        model = _load_target_model(task.target_policy)
        graph_builder = _graph_builder(task.target_policy)
    except Exception as exc:
        failed = _failed_branch(
            task.actual_action_offset,
            simulated_steps=1,
            failure_reason='target_policy_initialization_failure',
            diagnostic_codes=(_exception_code(exc),),
        )
        return _failed_result(
            task,
            original_reproduced=True,
            failure_reason='target_policy_initialization_failure',
            branches=(failed,),
            diagnostic_codes=failed.diagnostic_codes,
        )

    branches = [
        _complete_branch(
            task,
            action_offset=task.actual_action_offset,
            first_step=actual_first,
            model=model,
            graph_builder=graph_builder,
        ),
    ]
    if not branches[0].usable:
        return _failed_result(
            task,
            original_reproduced=True,
            failure_reason='actual_branch_failure',
            branches=branches,
            diagnostic_codes=branches[0].diagnostic_codes,
        )

    for action_offset in task.alternative_action_offsets:
        first_step, failure = _first_step_or_failed(
            task,
            action_offset,
        )
        if failure is not None:
            branches.append(failure)
            continue
        branches.append(_complete_branch(
            task,
            action_offset=action_offset,
            first_step=first_step,
            model=model,
            graph_builder=graph_builder,
        ))

    usable_alternatives = tuple(
        branch
        for branch in branches
        if (
            branch.action_offset != task.actual_action_offset
            and branch.usable
        )
    )
    if not usable_alternatives:
        diagnostics = tuple(dict.fromkeys(
            code
            for branch in branches
            for code in branch.diagnostic_codes
        ))
        return _failed_result(
            task,
            original_reproduced=True,
            failure_reason='no_usable_alternative_branch',
            branches=branches,
            diagnostic_codes=diagnostics,
        )

    all_completed = all(
        branch.status == 'completed'
        for branch in branches
    )
    status = 'completed' if all_completed else 'partial'
    failure_reason = (
        None
        if all_completed
        else 'one_or_more_branches_partial_or_failed'
    )
    diagnostics = tuple(dict.fromkeys(
        code
        for branch in branches
        for code in branch.diagnostic_codes
    ))
    return CounterfactualResult(
        task_id=task.task_id,
        status=status,
        actual_action_offset=task.actual_action_offset,
        original_reproduced=True,
        branches=tuple(branches),
        simulated_steps=sum(
            branch.simulated_steps
            for branch in branches
        ),
        failure_reason=failure_reason,
        diagnostic_codes=diagnostics,
    )


def counterfactual_result_to_causal_samples(
        task,
        result,
        context,
        *,
        zero_delta_epsilon=0.0):
    """把可信 A/B return 转成独立 ``CausalSample(counterfactual)``。

    ``task`` 不能省略：result 为减少进程回传体积只保存 task_id，而预算身份、冻结
    策略指纹和标签置信度都必须与原任务核对。函数不修改任何输入；未通过物理门禁、
    没有可用替代分支或差值落在零区间时返回空 tuple，绝不制造占位标签。
    """

    if not isinstance(task, CounterfactualTask):
        raise TypeError('task must be CounterfactualTask')
    if not isinstance(result, CounterfactualResult):
        raise TypeError('result must be CounterfactualResult')
    if not isinstance(context, CausalTransitionContext):
        raise TypeError('context must be CausalTransitionContext')
    epsilon = float(zero_delta_epsilon)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError(
            'zero_delta_epsilon must be finite and non-negative'
        )
    if result.task_id != task.task_id:
        raise ValueError('result task_id does not match task')
    if result.actual_action_offset != task.actual_action_offset:
        raise ValueError(
            'result actual_action_offset does not match task'
        )
    if context.transition_key != task.transition_key:
        raise ValueError(
            'context transition_key does not match task'
        )
    if context.actual_action_offset != task.actual_action_offset:
        raise ValueError(
            'context actual_action_offset does not match task'
        )
    analyzer_fingerprint = (
        context.state_analysis.analyzer_config_fingerprint
    )
    expected_analyzer_fingerprint = (
        task.target_policy.state_analyzer_config.fingerprint
    )
    if analyzer_fingerprint != expected_analyzer_fingerprint:
        raise ValueError(
            'context analyzer config does not match frozen target payload'
        )
    expected_drop_x = (
        context.state_analysis.action_drop_x_by_offset[
            task.actual_action_offset
        ]
    )
    factual_drop_x = task.factual_outcome.drop_result.drop_x
    if not math.isclose(
            float(expected_drop_x),
            float(factual_drop_x),
            rel_tol=0.0,
            abs_tol=1e-9):
        raise ValueError(
            'context actual action geometry does not match factual outcome'
        )

    if not result.label_ready:
        return ()
    graph_fingerprint = graph_schema_fingerprint(context.graph)
    if graph_fingerprint != context.graph_schema_fingerprint:
        raise ValueError(
            'context graph schema fingerprint changed'
        )

    samples = []
    event_key = f'counterfactual-task-v1:{task.task_id}'
    budget_key = stable_budget_key(task.budget_key)
    requested_alternatives = set(
        task.alternative_action_offsets
    )
    for comparison_offset, raw_delta in result.return_deltas:
        if comparison_offset not in requested_alternatives:
            raise ValueError(
                'result contains an unrequested comparison action'
            )
        delta = float(raw_delta)
        if not math.isfinite(delta):
            raise ValueError('counterfactual return delta must be finite')
        if abs(delta) <= epsilon:
            continue
        direction = 1 if delta > 0.0 else -1
        samples.append(CausalSample(
            graph=context.graph,
            actual_action_offset=task.actual_action_offset,
            comparison_action_offset=comparison_offset,
            direction=direction,
            target_margin=0.0,
            confidence=task.label_confidence,
            cause_type='COUNTERFACTUAL_RETURN',
            delay=task.attribution_delay,
            transition_key=task.transition_key,
            attribution_version=task.attribution_version,
            supervision_kind='counterfactual',
            stratum='counterfactual',
            event_key=event_key,
            budget_key=budget_key,
            target_delta=delta,
            policy_version=context.policy_version,
            analyzer_config_fingerprint=analyzer_fingerprint,
            graph_schema_fingerprint=graph_fingerprint,
        ))
    return tuple(samples)


def target_model_cache_info():
    """返回只读调试快照，不暴露模型对象。"""

    return MappingProxyType({
        'capacity': _TARGET_MODEL_CACHE_CAPACITY,
        'size': len(_TARGET_MODEL_CACHE),
        'fingerprints': tuple(_TARGET_MODEL_CACHE.keys()),
    })


__all__ = [
    'counterfactual_result_to_causal_samples',
    'freeze_target_policy_payload',
    'run_counterfactual_task',
    'target_model_cache_info',
]
