"""多进程 rollout 采集器。

本模块只属于训练侧优化层，用来把多个 `DaxiguaEnv` 放到独立 Python 进程中并行
采集 transition。游戏本体仍然只暴露 headless 环境接口，不需要知道训练进程如何
调度 worker。
"""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from io import BytesIO

from daxigua_rl.env import DaxiguaEnv
from daxigua_rl.graph import GraphBuilder
from daxigua_rl.models import GNNQNetwork
from daxigua_rl.reward import REWARD_BREAKDOWN_FIELDS

from .collector import RolloutCollector, RolloutStats
from .replay_buffer import ReplayBuffer


_WORKER_COLLECTOR = None
_WORKER_MODEL = None
_WORKER_SEED = 0


@dataclass(frozen=True)
class ParallelCollectHandle:
    """一次异步并行采集请求的句柄。"""

    futures: tuple
    counts: tuple
    started_at: float


def _worker_init(worker_index, env_config, model_config, seed):
    """初始化 worker 进程内长期复用的环境、模型和 collector。"""

    global _WORKER_COLLECTOR, _WORKER_MODEL, _WORKER_SEED

    _WORKER_SEED = int(seed) + int(worker_index) * 100_003
    env = DaxiguaEnv(config=env_config)

    _WORKER_MODEL = None
    if model_config is not None:
        _WORKER_MODEL = GNNQNetwork(**model_config)
        _WORKER_MODEL.eval()

    # worker 内部的 replay buffer 只是临时收集容器。真正长期保存经验的 replay
    # buffer 位于主进程，worker 每次 collect 后会把 transition 打包返回。
    local_buffer = ReplayBuffer(capacity=1, seed=_WORKER_SEED)
    _WORKER_COLLECTOR = RolloutCollector(
        env=env,
        graph_builder=GraphBuilder(),
        replay_buffer=local_buffer,
        model=_WORKER_MODEL,
        seed=_WORKER_SEED,
    )


def _worker_sync_model(state_dict):
    """把主进程 online model 参数同步到当前 worker。"""

    if _WORKER_MODEL is None:
        return False

    state_dict = _load_from_bytes(state_dict)
    _WORKER_MODEL.load_state_dict(state_dict)
    _WORKER_MODEL.eval()
    return True


def _worker_collect(step_count, epsilon):
    """在 worker 进程内采集若干 transition 并返回。"""

    if _WORKER_COLLECTOR is None:
        raise RuntimeError('parallel rollout worker is not initialized')

    step_count = int(step_count)
    if step_count <= 0:
        return (), RolloutStats(steps=0, episodes=0, total_reward=0.0)

    # 每次调用使用一个刚好足够大的临时 buffer，避免 worker 内保留历史 replay，
    # 也避免小容量 buffer 在 collect 中途覆盖刚采集到的 transition。
    local_buffer = ReplayBuffer(capacity=step_count, seed=_WORKER_SEED + step_count)
    _WORKER_COLLECTOR.replay_buffer = local_buffer
    stats = _WORKER_COLLECTOR.collect_steps(step_count, epsilon=epsilon)
    return _save_to_bytes(local_buffer.to_tuple()), stats


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
            seed=0):
        """创建多进程 collector。

        参数：
        - `worker_count`: worker 进程数量。
        - `env_config`: 传给每个 `DaxiguaEnv` 的环境配置。
        - `replay_buffer`: 主进程长期经验池。
        - `model_config`: 创建 worker 侧 GNNQNetwork 所需参数；epsilon < 1 时需要。
        - `model`: 主进程 online model；用于周期性同步参数到 worker。
        - `seed`: worker 随机种子基准。
        """

        worker_count = int(worker_count)
        if worker_count <= 1:
            raise ValueError('worker_count must be greater than 1')
        if not isinstance(replay_buffer, ReplayBuffer):
            raise TypeError(f'replay_buffer must be ReplayBuffer, got {type(replay_buffer)!r}')

        self.worker_count = worker_count
        self.env_config = env_config
        self.replay_buffer = replay_buffer
        self.model_config = model_config
        self.model = model
        self.seed = int(seed)
        self._closed = False
        self._model_synced = False

        # 使用 spawn 而不是 Linux 默认 fork，避免主进程已经初始化 CUDA 后 fork 出
        # worker 导致 CUDA/驱动状态异常。worker 只在 CPU 上做采样推理。
        context = multiprocessing.get_context('spawn')
        self._executors = tuple(
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=_worker_init,
                initargs=(worker_index, self.env_config, self.model_config, self.seed),
            )
            for worker_index in range(self.worker_count)
        )

    def close(self):
        """关闭 worker 进程池。"""

        if self._closed:
            return
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True

    def sync_model(self, model=None):
        """把主进程模型参数同步到所有 worker。"""

        self._ensure_open()
        model = model or self.model
        if model is None:
            raise ValueError('model is required to sync parallel rollout workers')

        state_dict = {
            name: parameter.detach().cpu()
            for name, parameter in model.state_dict().items()
        }
        state_bytes = _save_to_bytes(state_dict)
        futures = tuple(
            executor.submit(_worker_sync_model, state_bytes)
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
        futures = tuple(
            self._executors[worker_index].submit(_worker_collect, count, epsilon)
            for worker_index, count in enumerate(counts)
            if count > 0
        )
        return ParallelCollectHandle(
            futures=futures,
            counts=tuple(count for count in counts if count > 0),
            started_at=time.perf_counter(),
        )

    def finish_collect_steps(self, handle):
        """等待并行采集结束，把 transition 写入主进程 replay buffer。"""

        self._ensure_open()
        results = tuple(future.result() for future in handle.futures)
        wall_seconds = time.perf_counter() - handle.started_at

        all_transitions = []
        worker_stats = []
        for transition_bytes, stats in results:
            transitions = _load_from_bytes(transition_bytes)
            all_transitions.extend(transitions)
            worker_stats.append(stats)

        self.replay_buffer.extend(all_transitions)
        return _merge_rollout_stats(
            worker_stats=worker_stats,
            buffer_size=len(self.replay_buffer),
            collect_seconds=wall_seconds,
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


def _merge_rollout_stats(worker_stats, buffer_size, collect_seconds):
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
    physics_frames_total = 0
    fruit_count_total = 0
    graph_node_count_total = 0
    graph_edge_count_total = 0
    graph_cache_hits = 0
    graph_cache_misses = 0

    step_offset = 0
    for stats in worker_stats:
        for field_name, value in stats.reward_breakdown_totals_dict.items():
            reward_breakdown_totals[field_name] += float(value)

        episode_rewards.extend(stats.episode_rewards)
        episode_lengths.extend(stats.episode_lengths)
        episode_scores.extend(stats.episode_scores)
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
        physics_frames_total += stats.physics_frames_total
        fruit_count_total += stats.fruit_count_total
        graph_node_count_total += stats.graph_node_count_total
        graph_edge_count_total += stats.graph_edge_count_total
        graph_cache_hits += stats.graph_cache_hits
        graph_cache_misses += stats.graph_cache_misses

    return RolloutStats(
        steps=steps,
        episodes=len(episode_rewards),
        total_reward=total_reward,
        reward_breakdown_totals=tuple(
            (field_name, reward_breakdown_totals[field_name])
            for field_name in REWARD_BREAKDOWN_FIELDS
        ),
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
        collect_seconds=collect_seconds,
        graph_build_seconds=graph_build_seconds,
        tensor_convert_seconds=tensor_convert_seconds,
        action_select_seconds=action_select_seconds,
        env_step_seconds=env_step_seconds,
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
