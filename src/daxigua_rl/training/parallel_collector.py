"""多进程 rollout 采集器。

本模块只属于训练侧优化层，用来把多个 `DaxiguaEnv` 放到独立 Python 进程中并行
采集 transition。游戏本体仍然只暴露 headless 环境接口，不需要知道训练进程如何
调度 worker。
"""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from io import BytesIO

from daxigua_rl.attribution.causal_replay import CausalReplayBuffer
from daxigua_rl.env import DaxiguaEnv
from daxigua_rl.graph import GraphBuilder
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.reward import REWARD_BREAKDOWN_FIELDS

from .collector import RolloutCollector, RolloutStats
from .replay_buffer import ReplayBuffer


_WORKER_COLLECTOR = None
_WORKER_MODEL = None
_WORKER_SEED = 0


def _configure_worker_torch_threads():
    """限制每个 rollout worker 的 Torch CPU 线程，避免 num_envs² 过额调度。"""

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # inter-op 线程池在进程内开始并行工作后不能再次修改；intra-op=1
        # 仍是防止 8 workers × 8 threads 过额调度的关键门禁。
        pass
    return torch.get_num_threads(), torch.get_num_interop_threads()


def _worker_torch_thread_counts():
    """测试/诊断当前 worker Torch 线程配置。"""

    import torch

    return torch.get_num_threads(), torch.get_num_interop_threads()


@dataclass(frozen=True)
class ParallelCollectHandle:
    """一次异步并行采集请求的句柄。"""

    futures: tuple
    counts: tuple
    started_at: float
    worker_indices: tuple = ()


@dataclass(frozen=True)
class WorkerAttributionFinalization:
    """一个 worker 关闭时被审查并取消的未确认归因摘要。"""

    worker_id: int
    cancelled_pending_count: int
    n_step_flush_emitted: int = 0
    event_type_counts: tuple = ()
    resolution_reason_counts: tuple = ()


def _worker_init(
        worker_index,
        env_config,
        model_config,
        seed,
        n_step,
        gamma,
        causal_enabled,
        policy_version,
        counterfactual_enabled,
        counterfactual_ring_size,
        counterfactual_proposal_sample_rate,
        episode_id_start):
    """初始化 worker 进程内长期复用的环境、模型和 collector。"""

    global _WORKER_COLLECTOR, _WORKER_MODEL, _WORKER_SEED

    _configure_worker_torch_threads()
    _WORKER_SEED = int(seed) + int(worker_index) * 100_003
    env = DaxiguaEnv(config=env_config)

    _WORKER_MODEL = None
    if model_config is not None:
        _WORKER_MODEL = GNNQNetwork(**model_config)
        _WORKER_MODEL.eval()

    # worker 内部的 replay buffer 只是临时收集容器。真正长期保存经验的 replay
    # buffer 位于主进程，worker 每次 collect 后会把 transition 打包返回。
    local_buffer = ReplayBuffer(
        capacity=max(1, int(n_step)),
        seed=_WORKER_SEED,
    )
    local_causal_buffer = (
        CausalReplayBuffer(capacity=256, seed=_WORKER_SEED)
        if causal_enabled
        else None
    )
    _WORKER_COLLECTOR = RolloutCollector(
        env=env,
        graph_builder=GraphBuilder(),
        replay_buffer=local_buffer,
        model=_WORKER_MODEL,
        seed=_WORKER_SEED,
        worker_id=worker_index,
        causal_replay_buffer=local_causal_buffer,
        n_step=n_step,
        gamma=gamma,
        policy_version=policy_version,
        counterfactual_enabled=counterfactual_enabled,
        counterfactual_ring_size=counterfactual_ring_size,
        counterfactual_proposal_sample_rate=(
            counterfactual_proposal_sample_rate
        ),
        episode_id_start=episode_id_start,
    )


def _worker_sync_model(state_dict, policy_version=None):
    """把主进程 online model 参数同步到当前 worker。"""

    if _WORKER_MODEL is None:
        return False

    state_dict = _load_from_bytes(state_dict)
    _WORKER_MODEL.load_state_dict(state_dict)
    _WORKER_MODEL.eval()
    _WORKER_COLLECTOR.set_policy_version(policy_version)
    return True


def _worker_collect(step_count, epsilon):
    """在 worker 进程内采集若干 transition 并返回。"""

    if _WORKER_COLLECTOR is None:
        raise RuntimeError('parallel rollout worker is not initialized')

    step_count = int(step_count)
    if step_count <= 0:
        return (_save_to_bytes(((), ())), b''), RolloutStats(
            steps=0,
            episodes=0,
            total_reward=0.0,
        )

    # 每次调用使用一个刚好足够大的临时 buffer，避免 worker 内保留历史 replay，
    # 也避免小容量 buffer 在 collect 中途覆盖刚采集到的 transition。
    local_buffer = ReplayBuffer(
        capacity=(
            step_count
            + _WORKER_COLLECTOR.n_step
            - 1
        ),
        seed=_WORKER_SEED + step_count,
    )
    _WORKER_COLLECTOR.replay_buffer = local_buffer
    local_causal_buffer = None
    if _WORKER_COLLECTOR.rule_causal_builder is not None:
        local_causal_buffer = CausalReplayBuffer(
            capacity=max(256, step_count),
            seed=_WORKER_SEED + step_count,
        )
        _WORKER_COLLECTOR.causal_replay_buffer = local_causal_buffer
    stats = _WORKER_COLLECTOR.collect_steps(step_count, epsilon=epsilon)
    causal_samples = (
        local_causal_buffer.to_tuple()
        if local_causal_buffer is not None
        else ()
    )
    proposals = _WORKER_COLLECTOR.drain_counterfactual_proposals()
    # proposal 独立序列化后直接作为传输 payload 返回：其长度就是实际跨进程
    # 字节数，同时避免为了测量体积把完整快照对象再序列化第二遍。
    proposal_payload = (
        _save_to_bytes(proposals)
        if proposals
        else b''
    )
    transition_payload = _save_to_bytes((
        local_buffer.to_tuple(),
        causal_samples,
    ))
    stats = replace(
        stats,
        causal_samples_emitted=len(causal_samples),
        causal_buffer_size=len(causal_samples),
        counterfactual_proposals_serialized=len(proposals),
        counterfactual_proposal_serialized_bytes=(
            len(proposal_payload)
        ),
        counterfactual_proposal_outbox_size=len(proposals),
    )
    return (transition_payload, proposal_payload), stats


def _worker_finalize_attribution():
    """在进程退出前取消归因 pending，并返回尚未上送的 n-step 尾巴。"""

    if _WORKER_COLLECTOR is None:
        raise RuntimeError('parallel rollout worker is not initialized')
    events = _WORKER_COLLECTOR.close()
    counts = {}
    reason_counts = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
        reason = event.resolution_reason or 'unknown'
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary = WorkerAttributionFinalization(
        worker_id=_WORKER_COLLECTOR.worker_id,
        cancelled_pending_count=len(events),
        n_step_flush_emitted=(
            _WORKER_COLLECTOR.close_n_step_flush_emitted
        ),
        event_type_counts=tuple(sorted(counts.items())),
        resolution_reason_counts=tuple(sorted(reason_counts.items())),
    )
    return (
        _save_to_bytes(
            _WORKER_COLLECTOR.close_n_step_transitions
        ),
        summary,
    )


class ParallelRolloutCollector:
    """主进程侧的多 worker rollout 调度器。

    `collect_steps()` 的外部语义和 `RolloutCollector` 保持一致：调用者只关心它写入
    主进程 replay buffer 多少条 transition，并拿到一份合并后的 `RolloutStats`。
    """

    def __init__(
            self,
            worker_count,
            env_config,
            replay_buffer,
            model_config=None,
            model=None,
            seed=0,
            causal_replay_buffer=None,
            n_step=1,
            gamma=0.99,
            policy_version=None,
            counterfactual_enabled=False,
            counterfactual_ring_size=32,
            counterfactual_proposal_sample_rate=1.0,
            episode_id_start=0):
        """创建多进程 collector。

        参数：
        - `worker_count`: worker 进程数量。
        - `env_config`: 传给每个 `DaxiguaEnv` 的环境配置。
        - `replay_buffer`: 主进程长期经验池。
        - `model_config`: 创建 worker 侧 GNNQNetwork 所需参数；epsilon < 1 时需要。
        - `model`: 主进程 online model；用于周期性同步参数到 worker。
        - `seed`: worker 随机种子基准。
        - `causal_replay_buffer`: 主进程长期因果经验池；None 表示关闭规则归因输出。
        - `n_step` / `gamma`: 每个 worker 独立持有的有序 n-step 聚合配置。
        - `policy_version`: 首次模型同步前使用的采样策略版本。
        - `counterfactual_enabled`: 是否让 worker 捕获快照并输出稀疏 proposal。
        - `counterfactual_ring_size`: 每个 worker 的稳定边界历史环容量。
        - `counterfactual_proposal_sample_rate`: 常规 proposal 的稳定传输抽样率；
          高价值合成候选始终保留。
        - `episode_id_start`: 每个 worker 恢复后首局使用的 episode id。
        """

        worker_count = int(worker_count)
        if worker_count <= 1:
            raise ValueError('worker_count must be greater than 1')
        if not isinstance(replay_buffer, ReplayBuffer):
            raise TypeError(f'replay_buffer must be ReplayBuffer, got {type(replay_buffer)!r}')
        if (
                causal_replay_buffer is not None
                and not isinstance(
                    causal_replay_buffer,
                    CausalReplayBuffer)):
            raise TypeError(
                'causal_replay_buffer must be CausalReplayBuffer or None'
            )
        # 复用单进程 accumulator 的严格参数校验，且不在主进程保留多余状态。
        from .n_step import NStepTransitionAccumulator
        n_step_config = NStepTransitionAccumulator(
            n_step=n_step,
            gamma=gamma,
        )
        if policy_version is not None:
            policy_version = str(policy_version).strip()
            if not policy_version:
                raise ValueError('policy_version must not be empty')
        if not isinstance(counterfactual_enabled, bool):
            raise TypeError('counterfactual_enabled must be bool')
        if isinstance(counterfactual_proposal_sample_rate, bool):
            raise TypeError(
                'counterfactual_proposal_sample_rate must be a real number'
            )
        counterfactual_proposal_sample_rate = float(
            counterfactual_proposal_sample_rate
        )
        if not 0.0 <= counterfactual_proposal_sample_rate <= 1.0:
            raise ValueError(
                'counterfactual_proposal_sample_rate must be in [0, 1]'
            )
        counterfactual_ring_size = int(counterfactual_ring_size)
        if counterfactual_ring_size <= 0:
            raise ValueError(
                'counterfactual_ring_size must be positive'
            )
        if isinstance(episode_id_start, bool):
            raise TypeError('episode_id_start must be an integer')
        episode_id_start = int(episode_id_start)
        if episode_id_start < 0:
            raise ValueError('episode_id_start must be non-negative')

        self.worker_count = worker_count
        self.env_config = env_config
        self.replay_buffer = replay_buffer
        self.model_config = model_config
        self.model = model
        self.seed = int(seed)
        self.causal_replay_buffer = causal_replay_buffer
        self.n_step = n_step_config.n_step
        self.gamma = n_step_config.gamma
        self.policy_version = policy_version
        self.counterfactual_enabled = counterfactual_enabled
        self.counterfactual_ring_size = counterfactual_ring_size
        self.counterfactual_proposal_sample_rate = (
            counterfactual_proposal_sample_rate
        )
        self.episode_id_start = episode_id_start
        self._counterfactual_proposal_outbox = []
        self._closed = False
        self._model_synced = False
        self._policy_sync_count = 0
        self.attribution_finalization_summaries = ()
        self._worker_pending_event_counts = [0] * worker_count

        # 使用 spawn 而不是 Linux 默认 fork，避免主进程已经初始化 CUDA 后 fork 出
        # worker 导致 CUDA/驱动状态异常。worker 只在 CPU 上做采样推理。
        context = multiprocessing.get_context('spawn')
        self._executors = tuple(
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=_worker_init,
                initargs=(
                    worker_index,
                    self.env_config,
                    self.model_config,
                    self.seed,
                    self.n_step,
                    self.gamma,
                    self.causal_replay_buffer is not None,
                    self.policy_version,
                    self.counterfactual_enabled,
                    self.counterfactual_ring_size,
                    self.counterfactual_proposal_sample_rate,
                    self.episode_id_start,
                ),
            )
            for worker_index in range(self.worker_count)
        )

    @property
    def model_synced(self):
        """主进程模型是否至少成功同步到全部 rollout worker 一次。"""

        return self._model_synced

    def close(self):
        """关闭 worker 进程池。"""

        if self._closed:
            return self.attribution_finalization_summaries
        try:
            futures = tuple(
                executor.submit(_worker_finalize_attribution)
                for executor in self._executors
            )
            results = tuple(
                future.result()
                for future in futures
            )
            tail_transitions = []
            summaries = []
            for transition_bytes, summary in results:
                transitions = _load_from_bytes(transition_bytes)
                if len(transitions) != summary.n_step_flush_emitted:
                    raise RuntimeError(
                        'parallel worker close returned an n-step tail '
                        'that does not match finalization stats'
                    )
                tail_transitions.extend(transitions)
                summaries.append(summary)
            self.replay_buffer.extend(tail_transitions)
            self.attribution_finalization_summaries = tuple(summaries)
        finally:
            for executor in self._executors:
                executor.shutdown(wait=True, cancel_futures=True)
            self._closed = True
        return self.attribution_finalization_summaries

    def sync_model(self, model=None, policy_version=None):
        """把主进程模型参数同步到所有 worker。"""

        self._ensure_open()
        model = model or self.model
        if model is None:
            raise ValueError('model is required to sync parallel rollout workers')

        state_dict = {
            name: parameter.detach().cpu()
            for name, parameter in model.state_dict().items()
        }
        self._policy_sync_count += 1
        if policy_version is None:
            policy_version = f'parallel-sync-{self._policy_sync_count}'
        else:
            policy_version = str(policy_version).strip()
            if not policy_version:
                raise ValueError('policy_version must not be empty')
        self.policy_version = policy_version
        state_bytes = _save_to_bytes(state_dict)
        futures = tuple(
            executor.submit(
                _worker_sync_model,
                state_bytes,
                policy_version,
            )
            for executor in self._executors
        )
        for future in futures:
            future.result()
        self._model_synced = True

    def collect_steps(self, step_count, epsilon=1.0):
        """同步并行采集指定数量 transition。"""

        handle = self.start_collect_steps(step_count, epsilon=epsilon)
        return self.finish_collect_steps(handle)

    def start_collect_steps(self, step_count, epsilon=1.0):
        """提交一次异步并行采集任务，返回可等待的 handle。"""

        self._ensure_open()
        step_count = int(step_count)
        if step_count <= 0:
            raise ValueError('step_count must be positive')
        epsilon = float(epsilon)
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError('epsilon must be in [0, 1]')
        if epsilon < 1.0 and self.model_config is not None and not self._model_synced:
            raise RuntimeError('parallel worker model must be synced before greedy collection')

        counts = self._split_step_count(step_count)
        worker_indices = tuple(
            worker_index
            for worker_index, count in enumerate(counts)
            if count > 0
        )
        futures = tuple(
            self._executors[worker_index].submit(
                _worker_collect,
                counts[worker_index],
                epsilon,
            )
            for worker_index in worker_indices
        )
        return ParallelCollectHandle(
            futures=futures,
            counts=tuple(count for count in counts if count > 0),
            started_at=time.perf_counter(),
            worker_indices=worker_indices,
        )

    def finish_collect_steps(self, handle):
        """等待并行采集结束，把 transition 写入主进程 replay buffer。"""

        self._ensure_open()
        results = tuple(future.result() for future in handle.futures)
        wall_seconds = time.perf_counter() - handle.started_at

        all_transitions = []
        worker_stats = []
        all_causal_samples = []
        all_proposals = []
        for worker_index, (payloads, stats) in zip(
                handle.worker_indices,
                results):
            transition_payload, proposal_payload = payloads
            transitions, causal_samples = _load_from_bytes(
                transition_payload
            )
            proposals = (
                _load_from_bytes(proposal_payload)
                if proposal_payload
                else ()
            )
            if (
                    len(transitions)
                    != stats.replay_transitions_emitted):
                raise RuntimeError(
                    'parallel worker returned replay transition count '
                    'that does not match emitted stats'
                )
            if len(stats.transition_keys) != stats.steps:
                raise RuntimeError(
                    'parallel worker returned raw transition keys that '
                    'do not match environment step stats'
                )
            if len(causal_samples) != stats.causal_samples_emitted:
                raise RuntimeError(
                    'parallel worker returned causal sample count that '
                    'does not match emitted stats'
                )
            if (
                    len(proposals)
                    != stats.counterfactual_proposals_serialized):
                raise RuntimeError(
                    'parallel worker returned proposal count that does '
                    'not match serialization stats'
                )
            all_transitions.extend(transitions)
            all_causal_samples.extend(causal_samples)
            all_proposals.extend(proposals)
            worker_stats.append(stats)
            self._worker_pending_event_counts[worker_index] = int(
                stats.attribution_pending_event_count
            )

        self.replay_buffer.extend(all_transitions)
        if all_causal_samples:
            if self.causal_replay_buffer is None:
                raise RuntimeError(
                    'parallel worker emitted causal samples without a '
                    'main-process causal replay buffer'
                )
            self.causal_replay_buffer.extend(all_causal_samples)
        self._counterfactual_proposal_outbox.extend(all_proposals)
        merged = _merge_rollout_stats(
            worker_stats=worker_stats,
            buffer_size=len(self.replay_buffer),
            collect_seconds=wall_seconds,
            causal_buffer_size=(
                len(self.causal_replay_buffer)
                if self.causal_replay_buffer is not None
                else 0
            ),
        )
        return replace(
            merged,
            attribution_pending_event_count=sum(
                self._worker_pending_event_counts
            ),
            counterfactual_proposal_outbox_size=len(
                self._counterfactual_proposal_outbox
            ),
        )

    def _split_step_count(self, step_count):
        """把总采集步数尽量平均分给多个 worker。"""

        active_workers = min(self.worker_count, int(step_count))
        base_count = int(step_count) // active_workers
        remainder = int(step_count) % active_workers
        return tuple(
            base_count + (1 if worker_index < remainder else 0)
            for worker_index in range(active_workers)
        )

    def _ensure_open(self):
        """确保进程池仍处于可用状态。"""

        if self._closed:
            raise RuntimeError('parallel rollout collector is closed')

    def drain_counterfactual_proposals(self):
        """取出并清空所有 worker 已回传、尚未交给调度器的 proposal。"""

        proposals = tuple(self._counterfactual_proposal_outbox)
        self._counterfactual_proposal_outbox.clear()
        return proposals


def _merge_rollout_stats(
        worker_stats,
        buffer_size,
        collect_seconds,
        causal_buffer_size=0):
    """把多个 worker 返回的 `RolloutStats` 合并成一份统计。"""

    steps = sum(stats.steps for stats in worker_stats)
    total_reward = sum(stats.total_reward for stats in worker_stats)
    reward_breakdown_totals = {
        field_name: 0.0
        for field_name in REWARD_BREAKDOWN_FIELDS
    }

    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    potential_shaping_abs_values = []
    transition_keys = []
    episode_end_offsets = []
    episode_terminated_flags = []
    episode_truncated_flags = []
    random_actions = 0
    greedy_actions = 0
    terminated_episodes = 0
    truncated_episodes = 0
    graph_build_seconds = 0.0
    tensor_convert_seconds = 0.0
    action_select_seconds = 0.0
    env_step_seconds = 0.0
    state_analysis_calls = 0
    state_analysis_seconds = 0.0
    state_analysis_cache_hits = 0
    state_analysis_degraded_count = 0
    attribution_tracker_calls = 0
    attribution_tracker_seconds = 0.0
    attribution_events_created = 0
    attribution_events_confirmed = 0
    attribution_events_cancelled = 0
    attribution_events_interrupted = 0
    attribution_pending_event_count = 0
    attribution_lineage_merge_count = 0
    attribution_chain_merge_count = 0
    attribution_max_chain_depth = 0
    attribution_event_status_counts = {}
    attribution_confidence_tier_counts = {}
    attribution_delays = []
    merge_level_counts = {}
    max_fruit_level = 0
    physics_frames_total = 0
    fruit_count_total = 0
    graph_node_count_total = 0
    graph_edge_count_total = 0
    graph_cache_hits = 0
    graph_cache_misses = 0
    replay_transitions_emitted = 0
    n_step_pending_count = 0
    n_step_forced_flush_emitted = 0
    causal_rule_build_calls = 0
    causal_rule_build_seconds = 0.0
    causal_rule_input_event_count = 0
    causal_rule_eligible_event_count = 0
    causal_rule_budget_count = 0
    causal_rule_samples_generated = 0
    causal_samples_pushed = 0
    causal_samples_emitted = 0
    causal_context_count = 0
    causal_rule_skip_reason_counts = {}
    counterfactual_snapshot_calls = 0
    counterfactual_snapshot_seconds = 0.0
    counterfactual_snapshot_failures = 0
    counterfactual_history_evictions = 0
    counterfactual_history_size = 0
    counterfactual_proposal_build_calls = 0
    counterfactual_proposal_build_seconds = 0.0
    counterfactual_proposal_input_event_count = 0
    counterfactual_proposal_confirmed_event_count = 0
    counterfactual_proposal_budget_count = 0
    counterfactual_proposals_generated = 0
    counterfactual_proposals_transfer_selected = 0
    counterfactual_proposals_transfer_throttled = 0
    counterfactual_proposals_serialized = 0
    counterfactual_proposal_serialized_bytes = 0
    counterfactual_proposal_outbox_size = 0
    counterfactual_proposal_skip_reason_counts = {}

    step_offset = 0
    for stats in worker_stats:
        for field_name, value in stats.reward_breakdown_totals_dict.items():
            reward_breakdown_totals[field_name] += float(value)

        episode_rewards.extend(stats.episode_rewards)
        episode_lengths.extend(stats.episode_lengths)
        episode_scores.extend(stats.episode_scores)
        potential_shaping_abs_values.extend(
            stats.potential_shaping_abs_values
        )
        transition_keys.extend(stats.transition_keys)
        episode_end_offsets.extend(
            step_offset + int(offset)
            for offset in stats.episode_end_offsets
        )
        episode_terminated_flags.extend(stats.episode_terminated_flags)
        episode_truncated_flags.extend(stats.episode_truncated_flags)
        step_offset += stats.steps

        random_actions += stats.random_actions
        greedy_actions += stats.greedy_actions
        terminated_episodes += stats.terminated_episodes
        truncated_episodes += stats.truncated_episodes
        graph_build_seconds += stats.graph_build_seconds
        tensor_convert_seconds += stats.tensor_convert_seconds
        action_select_seconds += stats.action_select_seconds
        env_step_seconds += stats.env_step_seconds
        state_analysis_calls += stats.state_analysis_calls
        state_analysis_seconds += stats.state_analysis_seconds
        state_analysis_cache_hits += stats.state_analysis_cache_hits
        state_analysis_degraded_count += stats.state_analysis_degraded_count
        attribution_tracker_calls += stats.attribution_tracker_calls
        attribution_tracker_seconds += stats.attribution_tracker_seconds
        attribution_events_created += stats.attribution_events_created
        attribution_events_confirmed += stats.attribution_events_confirmed
        attribution_events_cancelled += stats.attribution_events_cancelled
        attribution_events_interrupted += (
            stats.attribution_events_interrupted
        )
        attribution_pending_event_count += (
            stats.attribution_pending_event_count
        )
        attribution_lineage_merge_count += (
            stats.attribution_lineage_merge_count
        )
        attribution_chain_merge_count += (
            stats.attribution_chain_merge_count
        )
        attribution_max_chain_depth = max(
            attribution_max_chain_depth,
            stats.attribution_max_chain_depth,
        )
        for event_type, status, count in (
                stats.attribution_event_status_counts):
            key = event_type, status
            attribution_event_status_counts[key] = (
                attribution_event_status_counts.get(key, 0)
                + int(count)
            )
        for tier, count in stats.attribution_confidence_tier_counts:
            attribution_confidence_tier_counts[tier] = (
                attribution_confidence_tier_counts.get(tier, 0)
                + int(count)
            )
        attribution_delays.extend(stats.attribution_delays)
        for level, count in stats.merge_level_counts:
            merge_level_counts[int(level)] = (
                merge_level_counts.get(int(level), 0)
                + int(count)
            )
        max_fruit_level = max(
            max_fruit_level,
            int(stats.max_fruit_level),
        )
        physics_frames_total += stats.physics_frames_total
        fruit_count_total += stats.fruit_count_total
        graph_node_count_total += stats.graph_node_count_total
        graph_edge_count_total += stats.graph_edge_count_total
        graph_cache_hits += stats.graph_cache_hits
        graph_cache_misses += stats.graph_cache_misses
        replay_transitions_emitted += (
            stats.replay_transitions_emitted
        )
        n_step_pending_count += stats.n_step_pending_count
        n_step_forced_flush_emitted += (
            stats.n_step_forced_flush_emitted
        )
        causal_rule_build_calls += stats.causal_rule_build_calls
        causal_rule_build_seconds += (
            stats.causal_rule_build_seconds
        )
        causal_rule_input_event_count += (
            stats.causal_rule_input_event_count
        )
        causal_rule_eligible_event_count += (
            stats.causal_rule_eligible_event_count
        )
        causal_rule_budget_count += stats.causal_rule_budget_count
        causal_rule_samples_generated += (
            stats.causal_rule_samples_generated
        )
        causal_samples_pushed += stats.causal_samples_pushed
        causal_samples_emitted += stats.causal_samples_emitted
        causal_context_count += stats.causal_context_count
        for reason, count in stats.causal_rule_skip_reason_counts:
            causal_rule_skip_reason_counts[reason] = (
                causal_rule_skip_reason_counts.get(reason, 0)
                + int(count)
            )
        counterfactual_snapshot_calls += (
            stats.counterfactual_snapshot_calls
        )
        counterfactual_snapshot_seconds += (
            stats.counterfactual_snapshot_seconds
        )
        counterfactual_snapshot_failures += (
            stats.counterfactual_snapshot_failures
        )
        counterfactual_history_evictions += (
            stats.counterfactual_history_evictions
        )
        counterfactual_history_size += (
            stats.counterfactual_history_size
        )
        counterfactual_proposal_build_calls += (
            stats.counterfactual_proposal_build_calls
        )
        counterfactual_proposal_build_seconds += (
            stats.counterfactual_proposal_build_seconds
        )
        counterfactual_proposal_input_event_count += (
            stats.counterfactual_proposal_input_event_count
        )
        counterfactual_proposal_confirmed_event_count += (
            stats.counterfactual_proposal_confirmed_event_count
        )
        counterfactual_proposal_budget_count += (
            stats.counterfactual_proposal_budget_count
        )
        counterfactual_proposals_generated += (
            stats.counterfactual_proposals_generated
        )
        counterfactual_proposals_transfer_selected += (
            stats.counterfactual_proposals_transfer_selected
        )
        counterfactual_proposals_transfer_throttled += (
            stats.counterfactual_proposals_transfer_throttled
        )
        counterfactual_proposals_serialized += (
            stats.counterfactual_proposals_serialized
        )
        counterfactual_proposal_serialized_bytes += (
            stats.counterfactual_proposal_serialized_bytes
        )
        counterfactual_proposal_outbox_size += (
            stats.counterfactual_proposal_outbox_size
        )
        for reason, count in (
                stats.counterfactual_proposal_skip_reason_counts):
            counterfactual_proposal_skip_reason_counts[reason] = (
                counterfactual_proposal_skip_reason_counts.get(
                    reason,
                    0,
                )
                + int(count)
            )

    return RolloutStats(
        steps=steps,
        episodes=len(episode_rewards),
        total_reward=total_reward,
        reward_breakdown_totals=tuple(
            (field_name, reward_breakdown_totals[field_name])
            for field_name in REWARD_BREAKDOWN_FIELDS
        ),
        potential_shaping_abs_values=tuple(
            potential_shaping_abs_values
        ),
        transition_keys=tuple(transition_keys),
        episode_rewards=tuple(episode_rewards),
        episode_lengths=tuple(episode_lengths),
        episode_scores=tuple(episode_scores),
        episode_end_offsets=tuple(episode_end_offsets),
        episode_terminated_flags=tuple(episode_terminated_flags),
        episode_truncated_flags=tuple(episode_truncated_flags),
        terminated_episodes=terminated_episodes,
        truncated_episodes=truncated_episodes,
        random_actions=random_actions,
        greedy_actions=greedy_actions,
        buffer_size=buffer_size,
        replay_transitions_emitted=replay_transitions_emitted,
        n_step_pending_count=n_step_pending_count,
        n_step_forced_flush_emitted=(
            n_step_forced_flush_emitted
        ),
        causal_rule_build_calls=causal_rule_build_calls,
        causal_rule_build_seconds=causal_rule_build_seconds,
        causal_rule_input_event_count=(
            causal_rule_input_event_count
        ),
        causal_rule_eligible_event_count=(
            causal_rule_eligible_event_count
        ),
        causal_rule_budget_count=causal_rule_budget_count,
        causal_rule_samples_generated=(
            causal_rule_samples_generated
        ),
        causal_samples_pushed=causal_samples_pushed,
        causal_samples_emitted=causal_samples_emitted,
        causal_rule_skip_reason_counts=tuple(sorted(
            causal_rule_skip_reason_counts.items()
        )),
        causal_buffer_size=int(causal_buffer_size),
        causal_context_count=causal_context_count,
        counterfactual_snapshot_calls=(
            counterfactual_snapshot_calls
        ),
        counterfactual_snapshot_seconds=(
            counterfactual_snapshot_seconds
        ),
        counterfactual_snapshot_failures=(
            counterfactual_snapshot_failures
        ),
        counterfactual_history_evictions=(
            counterfactual_history_evictions
        ),
        counterfactual_history_size=counterfactual_history_size,
        counterfactual_proposal_build_calls=(
            counterfactual_proposal_build_calls
        ),
        counterfactual_proposal_build_seconds=(
            counterfactual_proposal_build_seconds
        ),
        counterfactual_proposal_input_event_count=(
            counterfactual_proposal_input_event_count
        ),
        counterfactual_proposal_confirmed_event_count=(
            counterfactual_proposal_confirmed_event_count
        ),
        counterfactual_proposal_budget_count=(
            counterfactual_proposal_budget_count
        ),
        counterfactual_proposals_generated=(
            counterfactual_proposals_generated
        ),
        counterfactual_proposals_transfer_selected=(
            counterfactual_proposals_transfer_selected
        ),
        counterfactual_proposals_transfer_throttled=(
            counterfactual_proposals_transfer_throttled
        ),
        counterfactual_proposals_serialized=(
            counterfactual_proposals_serialized
        ),
        counterfactual_proposal_serialized_bytes=(
            counterfactual_proposal_serialized_bytes
        ),
        counterfactual_proposal_outbox_size=(
            counterfactual_proposal_outbox_size
        ),
        counterfactual_proposal_skip_reason_counts=tuple(sorted(
            counterfactual_proposal_skip_reason_counts.items()
        )),
        collect_seconds=collect_seconds,
        graph_build_seconds=graph_build_seconds,
        tensor_convert_seconds=tensor_convert_seconds,
        action_select_seconds=action_select_seconds,
        env_step_seconds=env_step_seconds,
        state_analysis_calls=state_analysis_calls,
        state_analysis_seconds=state_analysis_seconds,
        state_analysis_cache_hits=state_analysis_cache_hits,
        state_analysis_degraded_count=state_analysis_degraded_count,
        attribution_tracker_calls=attribution_tracker_calls,
        attribution_tracker_seconds=attribution_tracker_seconds,
        attribution_events_created=attribution_events_created,
        attribution_events_confirmed=attribution_events_confirmed,
        attribution_events_cancelled=attribution_events_cancelled,
        attribution_events_interrupted=(
            attribution_events_interrupted
        ),
        attribution_pending_event_count=(
            attribution_pending_event_count
        ),
        attribution_lineage_merge_count=(
            attribution_lineage_merge_count
        ),
        attribution_chain_merge_count=(
            attribution_chain_merge_count
        ),
        attribution_max_chain_depth=(
            attribution_max_chain_depth
        ),
        attribution_event_status_counts=tuple(
            (
                event_type,
                status,
                count,
            )
            for (event_type, status), count
            in sorted(attribution_event_status_counts.items())
        ),
        attribution_confidence_tier_counts=tuple(sorted(
            attribution_confidence_tier_counts.items()
        )),
        attribution_delays=tuple(attribution_delays),
        merge_level_counts=tuple(sorted(merge_level_counts.items())),
        max_fruit_level=max_fruit_level,
        physics_frames_total=physics_frames_total,
        fruit_count_total=fruit_count_total,
        graph_node_count_total=graph_node_count_total,
        graph_edge_count_total=graph_edge_count_total,
        graph_cache_hits=graph_cache_hits,
        graph_cache_misses=graph_cache_misses,
    )


def _save_to_bytes(value):
    """把包含 torch Tensor 的对象序列化成普通 bytes，避免跨进程共享 fd。"""

    import torch

    buffer = BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _load_from_bytes(payload):
    """从 `_save_to_bytes()` 的结果还原对象。"""

    import torch

    return torch.load(BytesIO(payload), map_location='cpu', weights_only=False)
