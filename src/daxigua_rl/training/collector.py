"""单进程 rollout 采集器。

RolloutCollector 负责把当前已经完成的几个训练零件串起来：

    DaxiguaEnv -> GraphBuilder -> GNNQNetwork/EpsilonGreedyPolicy
    -> TensorTransition -> ReplayBuffer

它只负责“玩游戏并收集经验”，不负责从 replay buffer 采样训练，也不负责
计算 DQN loss、更新模型参数或同步 target network。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import torch

from daxigua_rl.attribution import (
    ANALYSIS_ACTION_COUNT,
    AttributionTracker,
    TrackerTransitionInput,
)
from daxigua_rl.attribution.causal_replay import (
    CausalTransitionContext,
    CausalReplayBuffer,
    RuleCausalSampleBuilder,
)
from daxigua_rl.attribution.counterfactual_proposal import (
    CounterfactualProposalBuilder,
)
from daxigua_rl.reward import REWARD_BREAKDOWN_FIELDS
from daxigua_rl.graph.tensor import graph_to_tensor

from .identity import TransitionKey
from .n_step import NStepTransitionAccumulator
from .replay_buffer import ReplayBuffer
from .tensor_transition import TensorTransition


REPLAY_GRAPH_DTYPE = torch.float16


@dataclass(frozen=True)
class RolloutStats:
    """一次 `collect_steps()` 的采集统计。

    `episode_rewards`、`episode_lengths` 和 `episode_scores` 只记录本次采集中
    已经结束的 episode；如果本次采集结束时仍处于一局游戏中，则未完成部分通过
    `current_episode_reward` 和 `current_episode_length` 暴露。
    """

    # 本次执行的真实环境动作数。n-step 模式下不等于 replay 写入数。
    steps: int

    # 本次采集中结束了多少局游戏。
    episodes: int

    # 本次采集得到的总 reward，包含未完成 episode 的部分 reward。
    total_reward: float

    # 本次采集中各 reward breakdown 字段的累计值。
    # 使用 `(字段名, 累计值)` 元组而不是裸 dict，避免 frozen dataclass 持有可变对象。
    reward_breakdown_totals: tuple = field(default_factory=tuple)

    # 每一步 potential shaping 绝对值；日志窗口保留轻量标量以计算真实 p95。
    potential_shaping_abs_values: tuple = field(default_factory=tuple)

    # 本次每个 transition 对应的稳定轨迹键；与采集顺序一一对齐。
    transition_keys: tuple = field(default_factory=tuple)

    # 本次采集中每个已结束 episode 的累计 reward。
    episode_rewards: tuple = field(default_factory=tuple)

    # 本次采集中每个已结束 episode 的投放次数。
    episode_lengths: tuple = field(default_factory=tuple)

    # 本次采集中每个已结束 episode 的最终游戏分数。
    episode_scores: tuple = field(default_factory=tuple)

    # 本次采集中每个已结束 episode 在当前 collect 调用内的结束 step offset。
    episode_end_offsets: tuple = field(default_factory=tuple)

    # 本次采集中每个已结束 episode 是否由游戏规则终止。
    episode_terminated_flags: tuple = field(default_factory=tuple)

    # 本次采集中每个已结束 episode 是否由环境流程截断。
    episode_truncated_flags: tuple = field(default_factory=tuple)

    # 由游戏规则结束的 episode 数量。
    terminated_episodes: int = 0

    # 由环境流程截断的 episode 数量。
    truncated_episodes: int = 0

    # 本次通过随机探索选择的动作数量。
    random_actions: int = 0

    # 本次通过模型 argmax 选择的动作数量。
    greedy_actions: int = 0

    # 采集结束后 replay buffer 中的经验数量。
    buffer_size: int = 0

    # 自上次 collect 返回后写入主 replay 的 n-step transition 数量；显式 reset
    # 发生在两次 collect 之间时，其短尾巴会在下一份统计中上报。
    replay_transitions_emitted: int = 0

    # 采集返回时仍在 worker-local n-step deque 中等待后续动作的单步尾巴。
    n_step_pending_count: int = 0

    # 自上次 collect 返回后，因调用者显式 reset 而提前按短 horizon 收口的数量。
    # 它已包含在 replay_transitions_emitted 中，单独暴露用于解释 raw/emitted 差异。
    n_step_forced_flush_emitted: int = 0

    # 本次规则归因构建器消费、判定可归因及聚合出的价值预算数量。
    causal_rule_build_calls: int = 0
    causal_rule_build_seconds: float = 0.0
    causal_rule_input_event_count: int = 0
    causal_rule_eligible_event_count: int = 0
    causal_rule_budget_count: int = 0

    # 本次构建出的规则样本数，以及因去重/覆盖后真正写入 causal replay 的数量。
    causal_rule_samples_generated: int = 0
    causal_samples_pushed: int = 0

    # 跨进程输出的不可变 CausalSample 数；单进程等于本次成功 push 次数。
    causal_samples_emitted: int = 0

    # `(reason, count)` 形式的规则样本跳过原因。
    causal_rule_skip_reason_counts: tuple = field(default_factory=tuple)

    # 采集结束后 causal replay 中的样本数；未启用时为 0。
    causal_buffer_size: int = 0

    # worker-local 延迟归因上下文缓存中仍保留的动作数量。
    causal_context_count: int = 0

    # worker-local 稀疏反事实 proposal 管线。默认关闭时这些字段必须全部为零。
    counterfactual_snapshot_calls: int = 0
    counterfactual_snapshot_seconds: float = 0.0
    counterfactual_snapshot_failures: int = 0
    counterfactual_history_evictions: int = 0
    counterfactual_history_size: int = 0
    counterfactual_proposal_build_calls: int = 0
    counterfactual_proposal_build_seconds: float = 0.0
    counterfactual_proposal_input_event_count: int = 0
    counterfactual_proposal_confirmed_event_count: int = 0
    counterfactual_proposal_budget_count: int = 0
    counterfactual_proposals_generated: int = 0
    counterfactual_proposal_skip_reason_counts: tuple = field(
        default_factory=tuple
    )
    counterfactual_proposal_outbox_size: int = 0

    # 并行 worker 单 payload 序列化统计；单进程 collector 保持为零。
    counterfactual_proposals_serialized: int = 0
    counterfactual_proposal_serialized_bytes: int = 0

    # 当前未完成 episode 已累计 reward。
    current_episode_reward: float = 0.0

    # 当前未完成 episode 已累计投放次数。
    current_episode_length: int = 0

    # 构建 GraphData 的累计耗时，单位秒。
    graph_build_seconds: float = 0.0

    # collect_steps 调用整体耗时，单位秒。
    collect_seconds: float = 0.0

    # GraphData 转 GraphTensor 的累计耗时，单位秒。
    tensor_convert_seconds: float = 0.0

    # epsilon-greedy 动作选择累计耗时，包含 greedy 分支的模型前向。
    action_select_seconds: float = 0.0

    # 环境 step 累计耗时，包含投放、物理推进、状态快照和 reward 计算。
    env_step_seconds: float = 0.0

    # StateAnalyzer 在当前采集窗口内的真实调用次数和累计耗时。
    state_analysis_calls: int = 0
    state_analysis_seconds: float = 0.0

    # 命中上一动作 next_analysis 缓存的 step 数。
    state_analysis_cache_hits: int = 0

    # 新生成分析中被标记为 degraded 的数量；truncated 边界通常会计入。
    state_analysis_degraded_count: int = 0

    # AttributionTracker 每个真实 transition 调用一次；完整事件仍只留在 worker。
    attribution_tracker_calls: int = 0
    attribution_tracker_seconds: float = 0.0

    # 事件生命周期轻量汇总。created 包含 pending 和即时 confirmed；
    # confirmed/cancelled 只在事件真正解决时各计一次。
    attribution_events_created: int = 0
    attribution_events_confirmed: int = 0
    attribution_events_cancelled: int = 0
    attribution_events_interrupted: int = 0
    attribution_pending_event_count: int = 0
    attribution_lineage_merge_count: int = 0
    attribution_chain_merge_count: int = 0
    attribution_max_chain_depth: int = 0
    attribution_event_status_counts: tuple = field(default_factory=tuple)
    attribution_confidence_tier_counts: tuple = field(
        default_factory=tuple
    )
    attribution_delays: tuple = field(default_factory=tuple)

    # 当前窗口真实物理合成的目标等级分布，以及场上曾出现的最大水果等级。
    merge_level_counts: tuple = field(default_factory=tuple)
    max_fruit_level: int = 0

    # 环境实际推进的物理帧总数，用于判断 fast physics 是否生效。
    physics_frames_total: int = 0

    # 每个采集 step 后场上水果数量的累计值。
    fruit_count_total: int = 0

    # 每个采集 step 对应当前图节点数量的累计值。
    graph_node_count_total: int = 0

    # 每个采集 step 对应当前图边数量的累计值。
    graph_edge_count_total: int = 0

    # 当前图直接复用上一轮 next_graph 的次数。
    graph_cache_hits: int = 0

    # 当前图需要重新构建的次数。
    graph_cache_misses: int = 0

    @property
    def mean_episode_reward(self):
        """本次已结束 episode 的平均 reward；没有结束 episode 时返回 0。"""

        if not self.episode_rewards:
            return 0.0
        return sum(self.episode_rewards) / len(self.episode_rewards)

    @property
    def mean_episode_length(self):
        """本次已结束 episode 的平均长度；没有结束 episode 时返回 0。"""

        if not self.episode_lengths:
            return 0.0
        return sum(self.episode_lengths) / len(self.episode_lengths)

    @property
    def mean_episode_score(self):
        """本次已结束 episode 的平均最终分数；没有结束 episode 时返回 0。"""

        if not self.episode_scores:
            return 0.0
        return sum(self.episode_scores) / len(self.episode_scores)

    @property
    def reward_breakdown_totals_dict(self):
        """返回 reward breakdown 累计值字典，供训练日志按字段读取。"""

        return dict(self.reward_breakdown_totals)

    def mean_reward_breakdown(self, field_name):
        """返回某个 reward breakdown 字段在本次采集窗口内的平均值。"""

        if self.steps <= 0:
            return 0.0
        return self.reward_breakdown_totals_dict.get(field_name, 0.0) / self.steps

    @property
    def mean_physics_frames(self):
        """平均每次投放推进多少物理帧。"""

        if self.steps <= 0:
            return 0.0
        return self.physics_frames_total / self.steps

    @property
    def p95_abs_potential_shaping_reward(self):
        """返回单步 potential shaping 绝对值的线性插值 p95。"""

        values = tuple(
            float(value)
            for value in self.potential_shaping_abs_values
        )
        if not values:
            return 0.0
        ordered = sorted(values)
        position = 0.95 * (len(ordered) - 1)
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        fraction = position - lower_index
        return (
            ordered[lower_index] * (1.0 - fraction)
            + ordered[upper_index] * fraction
        )

    @property
    def mean_state_analysis_ms(self):
        """每次实际 StateAnalyzer 调用的平均毫秒数。"""

        if self.state_analysis_calls <= 0:
            return 0.0
        return self.state_analysis_seconds * 1000.0 / self.state_analysis_calls

    @property
    def state_analysis_cache_hit_rate(self):
        """按环境 step 统计的分析缓存命中率。"""

        if self.steps <= 0:
            return 0.0
        return self.state_analysis_cache_hits / self.steps

    @property
    def mean_fruit_count(self):
        """平均每次投放后场上有多少水果。"""

        if self.steps <= 0:
            return 0.0
        return self.fruit_count_total / self.steps

    @property
    def mean_graph_nodes(self):
        """平均当前状态图节点数。"""

        if self.steps <= 0:
            return 0.0
        return self.graph_node_count_total / self.steps

    @property
    def mean_graph_edges(self):
        """平均当前状态图边数。"""

        if self.steps <= 0:
            return 0.0
        return self.graph_edge_count_total / self.steps


class EpsilonGreedyPolicy:
    """epsilon-greedy 动作选择策略。

    - 以 `epsilon` 的概率随机选动作，用于探索。
    - 以 `1 - epsilon` 的概率选择 Q 值最大的动作，用于利用当前模型。
    """

    def __init__(self, seed=None):
        # 策略使用独立随机源，避免影响环境随机队列或 replay buffer 采样。
        self._rng = random.Random(seed)

    def should_explore(self, epsilon):
        """判断当前 step 是否走随机探索分支。"""

        epsilon = self.normalize_epsilon(epsilon)
        if epsilon >= 1.0:
            return True
        if epsilon <= 0.0:
            return False
        return self._rng.random() < epsilon

    def random_action_offset(self, action_count):
        """在当前候选动作范围内随机返回一个 action_offset。"""

        action_count = int(action_count)
        if action_count <= 0:
            raise ValueError('action_count must be positive')
        return self._rng.randrange(action_count)

    def greedy_action_offset(self, q_values):
        """返回 Q 值最大的动作下标。"""

        if q_values.dim() != 1:
            raise ValueError('q_values must have shape [action_count]')
        if q_values.numel() <= 0:
            raise ValueError('q_values must contain at least one action')
        return int(torch.argmax(q_values).item())

    def normalize_epsilon(self, epsilon):
        """把 epsilon 归一化为 float，并检查范围。"""

        epsilon = float(epsilon)
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError('epsilon must be in [0, 1]')
        return epsilon


class RolloutCollector:
    """单环境、单进程的 rollout 采集器。

    第一版只处理最直接的同步流程：

    1. 从当前环境状态构图。
    2. 用 epsilon-greedy 选择动作。
    3. 调用 `env.step(action_offset)`。
    4. 构建 CPU `TensorTransition`。
    5. 写入 `ReplayBuffer`。

    后续如果要做多进程采样，可以把多个 collector 放到 worker 进程里运行，
    主进程负责接收 transition 并训练模型。
    """

    def __init__(
            self,
            env,
            graph_builder,
            replay_buffer,
            model=None,
            policy=None,
            seed=None,
            worker_id=0,
            attribution_tracker=None,
            causal_replay_buffer=None,
            n_step=1,
            gamma=0.99,
            policy_version=None,
            counterfactual_enabled=False,
            counterfactual_ring_size=32,
            episode_id_start=0):
        """创建 rollout collector。

        参数：
        - `env`: 类 Gym 的环境，当前预期为 `DaxiguaEnv`。
        - `graph_builder`: 当前预期为 `GraphBuilder`。
        - `replay_buffer`: `ReplayBuffer` 实例。
        - `model`: 可选 Q 网络；当 `epsilon < 1.0` 时必须提供。
        - `policy`: 可选动作策略，默认使用 `EpsilonGreedyPolicy`。
        - `seed`: 默认策略随机种子。
        - `worker_id`: 当前采集器在本次训练 run 内的稳定 worker 编号。
        - `causal_replay_buffer`: 可选规则/反事实因果经验池；启用后会在 worker
          内保留完整延迟归因上下文，只把不可变 ``CausalSample`` 写入该池。
        - `n_step`: replay return 的最大步数；默认 1 保持旧 API 行为，正式训练
          可显式传 3。
        - `gamma`: n-step reward 折扣因子。
        - `policy_version`: 当前采样策略的稳定版本标识，写入因果样本 provenance。
        - `counterfactual_enabled`: 是否捕获动作前物理快照并生成稀疏 proposal。
          默认关闭，不执行任何快照调用。
        - `counterfactual_ring_size`: 每个 worker 保存的最近稳定边界数量。
        - `episode_id_start`: 首局使用的 episode id；resume 时用于避开旧因果键。
        """

        if not isinstance(replay_buffer, ReplayBuffer):
            raise TypeError(f'replay_buffer must be ReplayBuffer, got {type(replay_buffer)!r}')

        self.env = env
        self.graph_builder = graph_builder
        self.replay_buffer = replay_buffer
        self.model = model
        self.policy = policy or EpsilonGreedyPolicy(seed=seed)
        self.worker_id = int(worker_id)
        if self.worker_id < 0:
            raise ValueError('worker_id must be non-negative')
        if attribution_tracker is False:
            attribution_tracker = None
        elif attribution_tracker is None:
            attribution_tracker = AttributionTracker()
        if (
                attribution_tracker is not None
                and not isinstance(
                    attribution_tracker,
                    AttributionTracker)):
            raise TypeError(
                'attribution_tracker must be AttributionTracker, '
                'False, or None'
            )
        env_action_count = getattr(
            getattr(env, 'config', None),
            'action_count',
            None,
        )
        if (
                attribution_tracker is not None
                and env_action_count is not None
                and int(env_action_count) != ANALYSIS_ACTION_COUNT):
            raise ValueError(
                'AttributionTracker requires exactly '
                f'{ANALYSIS_ACTION_COUNT} environment actions; pass '
                'attribution_tracker=False only for non-attribution tests'
            )
        self.attribution_tracker = attribution_tracker
        if (
                causal_replay_buffer is not None
                and not isinstance(
                    causal_replay_buffer,
                    CausalReplayBuffer)):
            raise TypeError(
                'causal_replay_buffer must be CausalReplayBuffer or None'
            )
        if (
                causal_replay_buffer is not None
                and attribution_tracker is None):
            raise ValueError(
                'causal_replay_buffer requires attribution tracking'
            )
        if policy_version is not None:
            policy_version = str(policy_version).strip()
            if not policy_version:
                raise ValueError('policy_version must not be empty')
        self.causal_replay_buffer = causal_replay_buffer
        self.policy_version = policy_version
        self.n_step_accumulator = NStepTransitionAccumulator(
            n_step=n_step,
            gamma=gamma,
        )
        self.rule_causal_builder = (
            RuleCausalSampleBuilder()
            if causal_replay_buffer is not None
            else None
        )
        if not isinstance(counterfactual_enabled, bool):
            raise TypeError('counterfactual_enabled must be bool')
        if counterfactual_enabled and attribution_tracker is None:
            raise ValueError(
                'counterfactual proposal generation requires '
                'attribution tracking'
            )
        self.counterfactual_enabled = counterfactual_enabled
        self.counterfactual_proposal_builder = (
            CounterfactualProposalBuilder(
                ring_size=counterfactual_ring_size,
            )
            if counterfactual_enabled
            else None
        )
        self._counterfactual_proposal_outbox = []

        # `_obs` 和 `_info` 保存当前 episode 的最新状态。
        # collect_steps 第一次调用时如果发现它们为空，会自动 reset。
        self._obs = None
        self._info = None
        self._current_graph = None
        self._episode_reward = 0.0
        self._episode_length = 0
        if isinstance(episode_id_start, bool):
            raise TypeError('episode_id_start must be an integer')
        episode_id_start = int(episode_id_start)
        if episode_id_start < 0:
            raise ValueError('episode_id_start must be non-negative')
        self._episode_id = episode_id_start - 1
        self._carried_attribution_resolutions = []
        self._carried_replay_transitions_emitted = 0
        self._carried_n_step_forced_flush_emitted = 0
        self.close_n_step_transitions = ()
        self.close_n_step_flush_emitted = 0
        self.attribution_finalization_events = ()
        self._closed = False

    def close(self):
        """显式收口 worker-local pending，绝不在退出时静默确认。"""

        if self._closed:
            return self.attribution_finalization_events
        self.close_n_step_transitions = (
            self.n_step_accumulator.flush()
        )
        self.close_n_step_flush_emitted = (
            self._emit_n_step_transitions(
                self.close_n_step_transitions
            )
        )
        events = list(self._carried_attribution_resolutions)
        self._carried_attribution_resolutions.clear()
        if (
                self.attribution_tracker is not None
                and self.attribution_tracker.episode_active):
            events.extend(self.attribution_tracker.finalize_episode(
                reason='worker_shutdown'
            ))
        if self.rule_causal_builder is not None and self._episode_id >= 0:
            self.rule_causal_builder.context_cache.discard_episode(
                self.worker_id,
                self._episode_id,
            )
        if (
                self.counterfactual_proposal_builder is not None
                and self._episode_id >= 0):
            self.counterfactual_proposal_builder.discard_episode(
                self.worker_id,
                self._episode_id,
            )
        self.attribution_finalization_events = tuple(events)
        self._closed = True
        return self.attribution_finalization_events

    def reset(self, seed=None, fruit_queue=None):
        """重置环境并开始一个新的 episode。

        这个方法主要给训练脚本显式控制初始种子或固定水果队列时使用。
        普通情况下可以直接调用 `collect_steps()`，collector 会自动 reset。
        """

        if self._closed:
            raise RuntimeError('rollout collector is closed')
        previous_episode_id = self._episode_id
        if self.has_state and self.n_step_accumulator.pending_count:
            forced_flush_count = self._emit_n_step_transitions(
                self.n_step_accumulator.flush()
            )
            self._carried_replay_transitions_emitted += (
                forced_flush_count
            )
            self._carried_n_step_forced_flush_emitted += (
                forced_flush_count
            )
        if (
                self.attribution_tracker is not None
                and self.attribution_tracker.episode_active):
            self._carried_attribution_resolutions.extend(
                self.attribution_tracker.finalize_episode(
                    reason='manual_reset'
                )
            )
        if (
                self.rule_causal_builder is not None
                and previous_episode_id >= 0):
            self.rule_causal_builder.context_cache.discard_episode(
                self.worker_id,
                previous_episode_id,
            )
        if (
                self.counterfactual_proposal_builder is not None
                and previous_episode_id >= 0):
            self.counterfactual_proposal_builder.discard_episode(
                self.worker_id,
                previous_episode_id,
            )

        obs, info = self.env.reset(seed=seed, fruit_queue=fruit_queue)
        self._episode_id += 1
        if self.attribution_tracker is not None:
            self.attribution_tracker.begin_episode(
                self.worker_id,
                self._episode_id,
            )
        self._obs, self._info = obs, info
        self._current_graph = None
        self._episode_reward = 0.0
        self._episode_length = 0
        return self._obs, self._info

    @property
    def has_state(self):
        """collector 当前是否已经持有一个可继续采集的环境状态。"""

        return self._obs is not None and self._info is not None

    @property
    def current_episode_id(self):
        """当前 worker-local episode 编号；尚未 reset 时返回 None。"""

        if not self.has_state:
            return None
        return self._episode_id

    @property
    def n_step(self):
        """当前 worker 的 n-step horizon。"""

        return self.n_step_accumulator.n_step

    @property
    def gamma(self):
        """当前 worker 聚合 n-step return 时使用的折扣因子。"""

        return self.n_step_accumulator.gamma

    @property
    def n_step_pending_count(self):
        """尚未形成 replay 起点样本的有序单步尾巴数量。"""

        return self.n_step_accumulator.pending_count

    def set_policy_version(self, policy_version):
        """更新后续因果上下文记录的采样策略版本。"""

        if policy_version is not None:
            policy_version = str(policy_version).strip()
            if not policy_version:
                raise ValueError('policy_version must not be empty')
        self.policy_version = policy_version

    def drain_counterfactual_proposals(self):
        """取出并清空当前 worker 尚未交给主进程调度器的 proposal。"""

        proposals = tuple(self._counterfactual_proposal_outbox)
        self._counterfactual_proposal_outbox.clear()
        return proposals

    @property
    def next_transition_key(self):
        """返回下一次动作将使用的稳定轨迹键。"""

        if not self.has_state:
            raise RuntimeError('collector must be reset before requesting a transition key')

        step_index = int(self._obs.step_count)
        if step_index != self._episode_length:
            raise RuntimeError(
                'collector episode length is out of sync with GameState.step_count: '
                f'{self._episode_length} != {step_index}'
            )
        return TransitionKey(
            worker_id=self.worker_id,
            episode_id=self._episode_id,
            step_index=step_index,
        )

    def collect_steps(self, step_count, epsilon=1.0):
        """收集指定数量的 transition，并写入 replay buffer。

        `step_count` 表示要收集多少个环境 step，也就是多少次水果投放。
        如果中途 episode 结束，collector 会自动 reset 并继续采集，直到达到
        指定 transition 数量。
        """

        if self._closed:
            raise RuntimeError('rollout collector is closed')
        step_count = int(step_count)
        if step_count <= 0:
            raise ValueError('step_count must be positive')

        epsilon = self.policy.normalize_epsilon(epsilon)
        if self.model is None and epsilon < 1.0:
            raise ValueError('model is required when epsilon < 1.0')

        if not self.has_state:
            self.reset()

        was_training = None
        if self.model is not None:
            was_training = self.model.training
            self.model.eval()

        try:
            return self._collect_steps_impl(step_count=step_count, epsilon=epsilon)
        finally:
            # collector 采样时会临时切到 eval 模式；采集结束后恢复调用者原本的模式。
            if self.model is not None and was_training:
                self.model.train()

    def _collect_steps_impl(self, step_count, epsilon):
        """`collect_steps()` 的主体实现。"""

        collect_start = time.perf_counter()
        steps = 0
        total_reward = 0.0
        transition_keys = []
        episode_rewards = []
        episode_lengths = []
        episode_scores = []
        episode_end_offsets = []
        episode_terminated_flags = []
        episode_truncated_flags = []
        terminated_episodes = 0
        truncated_episodes = 0
        random_actions = 0
        greedy_actions = 0
        reward_breakdown_totals = {
            field_name: 0.0
            for field_name in REWARD_BREAKDOWN_FIELDS
        }
        potential_shaping_abs_values = []
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
        attribution_pending_event_count = (
            self.attribution_tracker.pending_event_count
            if self.attribution_tracker is not None
            else 0
        )
        attribution_lineage_merge_count = 0
        attribution_chain_merge_count = 0
        attribution_max_chain_depth = 0
        attribution_event_status_counts = {}
        attribution_confidence_tier_counts = {}
        attribution_delays = []
        merge_level_counts = {}
        max_fruit_level = int(getattr(self._obs, 'max_level', 0))
        physics_frames_total = 0
        fruit_count_total = 0
        graph_node_count_total = 0
        graph_edge_count_total = 0
        graph_cache_hits = 0
        graph_cache_misses = 0
        replay_transitions_emitted = (
            self._carried_replay_transitions_emitted
        )
        self._carried_replay_transitions_emitted = 0
        n_step_forced_flush_emitted = (
            self._carried_n_step_forced_flush_emitted
        )
        self._carried_n_step_forced_flush_emitted = 0
        causal_rule_build_calls = 0
        causal_rule_build_seconds = 0.0
        causal_rule_input_event_count = 0
        causal_rule_eligible_event_count = 0
        causal_rule_budget_count = 0
        causal_rule_samples_generated = 0
        causal_samples_pushed = 0
        causal_rule_skip_reason_counts = {}
        counterfactual_snapshot_calls = 0
        counterfactual_snapshot_seconds = 0.0
        counterfactual_snapshot_failures = 0
        counterfactual_history_evictions = 0
        counterfactual_proposal_build_calls = 0
        counterfactual_proposal_build_seconds = 0.0
        counterfactual_proposal_input_event_count = 0
        counterfactual_proposal_confirmed_event_count = 0
        counterfactual_proposal_budget_count = 0
        counterfactual_proposals_generated = 0
        counterfactual_proposal_skip_reason_counts = {}

        for event in self._carried_attribution_resolutions:
            attribution_events_cancelled += int(
                event.status == 'cancelled'
            )
            attribution_events_interrupted += int(
                event.resolution_reason in {
                    'manual_reset',
                    'worker_shutdown',
                }
            )
            key = event.event_type, event.status
            attribution_event_status_counts[key] = (
                attribution_event_status_counts.get(key, 0) + 1
            )
            if event.delay is not None:
                attribution_delays.append(event.delay)
            tier = event.confidence_tier
            attribution_confidence_tier_counts[tier] = (
                attribution_confidence_tier_counts.get(tier, 0) + 1
            )
        self._carried_attribution_resolutions.clear()

        while steps < step_count:
            candidates = tuple(self._info['action_candidates'])
            if not candidates:
                # 正常情况下，terminal step 之后会立即 reset；这里是额外容错。
                self.reset()
                continue

            if self._current_graph is None:
                graph, build_seconds, convert_seconds = self._build_graph_tensor(self._obs, candidates)
                graph_build_seconds += build_seconds
                tensor_convert_seconds += convert_seconds
                graph_cache_misses += 1
            else:
                # 上一轮已经为了 DQN bootstrap 构建了 next_graph；
                # 当前轮的状态正是上一轮 next_state，因此可以直接复用同一张图。
                graph = self._current_graph
                graph_cache_hits += 1

            action_count = len(candidates)
            self._validate_action_count(graph, action_count)
            graph_node_count_total += graph.num_nodes
            graph_edge_count_total += graph.num_edges

            action_select_start = time.perf_counter()
            action_offset, used_random = self._select_action(
                graph=graph,
                action_count=action_count,
                epsilon=epsilon,
            )
            action_select_seconds += time.perf_counter() - action_select_start

            # 身份必须在动作执行前生成，后续状态分析、谱系和事件都绑定到同一键。
            transition_key = self.next_transition_key

            counterfactual_snapshot = None
            if self.counterfactual_proposal_builder is not None:
                snapshot_start = time.perf_counter()
                counterfactual_snapshot_calls += 1
                try:
                    # 默认 canonicalize=True：真实分支和恢复分支必须从同一
                    # broadphase 内部表示出发。
                    counterfactual_snapshot = (
                        self.env.game.capture_snapshot()
                    )
                except Exception:
                    # 反事实是可降级旁路；快照失败绝不能中断真实 rollout。
                    counterfactual_snapshot_failures += 1
                    counterfactual_proposal_skip_reason_counts[
                        'snapshot_capture_failure'
                    ] = (
                        counterfactual_proposal_skip_reason_counts.get(
                            'snapshot_capture_failure',
                            0,
                        )
                        + 1
                    )
                finally:
                    counterfactual_snapshot_seconds += (
                        time.perf_counter() - snapshot_start
                    )

            env_step_start = time.perf_counter()
            next_obs, reward, terminated, truncated, next_info = self.env.step(
                action_offset,
                transition_key=transition_key,
            )
            env_step_seconds += time.perf_counter() - env_step_start
            for merge_event in next_info.get('merge_events', ()):
                level = int(merge_event.new_level)
                merge_level_counts[level] = (
                    merge_level_counts.get(level, 0) + 1
                )
            max_fruit_level = max(
                max_fruit_level,
                int(next_obs.max_level),
            )
            episode_done = terminated or truncated
            breakdown_values = self._accumulate_reward_breakdown(
                reward_breakdown_totals,
                next_info.get('reward_breakdown'),
            )
            if breakdown_values:
                potential_shaping_abs_values.append(abs(float(
                    breakdown_values.get('potential_shaping_reward', 0.0)
                )))
            self._validate_state_analysis_info(
                transition_key=transition_key,
                terminated=terminated,
                next_info=next_info,
            )
            if self.attribution_tracker is not None:
                causal_context = None
                if (
                        self.rule_causal_builder is not None
                        or self.counterfactual_proposal_builder
                        is not None):
                    try:
                        causal_context = CausalTransitionContext(
                            graph=graph,
                            state_analysis=next_info[
                                'previous_state_analysis'
                            ],
                            actual_action_offset=action_offset,
                            actual_action_index=next_info[
                                'action'
                            ].action_index,
                            policy_version=self.policy_version,
                        )
                    except Exception:
                        # 规则归因启用时，context 失配属于主训练契约错误；
                        # 只有反事实启用时则按可选旁路降级。
                        if self.rule_causal_builder is not None:
                            raise
                        counterfactual_proposal_skip_reason_counts[
                            'context_build_failure'
                        ] = (
                            counterfactual_proposal_skip_reason_counts.get(
                                'context_build_failure',
                                0,
                            )
                            + 1
                        )
                if (
                        self.rule_causal_builder is not None
                        and causal_context is not None):
                    self.rule_causal_builder.remember_context(
                        causal_context
                    )
                if (
                        self.counterfactual_proposal_builder is not None
                        and causal_context is not None
                        and counterfactual_snapshot is not None):
                    try:
                        evicted = (
                            self.counterfactual_proposal_builder.remember(
                                context=causal_context,
                                snapshot=counterfactual_snapshot,
                                factual_outcome=next_info[
                                    'engine_action_outcome'
                                ],
                            )
                        )
                        counterfactual_history_evictions += int(
                            evicted is not None
                        )
                    except Exception:
                        counterfactual_proposal_skip_reason_counts[
                            'history_record_failure'
                        ] = (
                            counterfactual_proposal_skip_reason_counts.get(
                                'history_record_failure',
                                0,
                            )
                            + 1
                        )
                tracker_result = (
                    self.attribution_tracker.observe_transition(
                        TrackerTransitionInput(
                            transition_key=transition_key,
                            action_offset=action_offset,
                            action_index=next_info[
                                'action'
                            ].action_index,
                            previous_state=self._obs,
                            next_state=next_obs,
                            previous_analysis=next_info[
                                'previous_state_analysis'
                            ],
                            next_analysis=next_info.get(
                                'post_action_state_analysis',
                                next_info['next_state_analysis'],
                            ),
                            drop_result=next_info['drop_result'],
                            physics_result=next_info[
                                'physics_result'
                            ],
                        )
                    )
                )
                attribution_tracker_calls += 1
                attribution_tracker_seconds += (
                    tracker_result.tracker_seconds
                )
                attribution_events_created += len(
                    tracker_result.created_events
                )
                attribution_events_confirmed += len(
                    tracker_result.confirmed_events
                )
                attribution_events_cancelled += len(
                    tracker_result.cancelled_events
                )
                attribution_events_interrupted += (
                    tracker_result.interrupted_pending_count
                )
                attribution_pending_event_count = (
                    tracker_result.pending_event_count
                )
                attribution_lineage_merge_count += len(
                    tracker_result.merge_records
                )
                attribution_chain_merge_count += (
                    tracker_result.chain_merge_count
                )
                attribution_max_chain_depth = max(
                    attribution_max_chain_depth,
                    tracker_result.max_transition_chain_depth,
                )
                for event in (
                        tracker_result.created_events
                        + tracker_result.resolved_events):
                    status_key = event.event_type, event.status
                    attribution_event_status_counts[status_key] = (
                        attribution_event_status_counts.get(
                            status_key,
                            0,
                        ) + 1
                    )
                    if event.delay is not None:
                        attribution_delays.append(event.delay)
                for event in tracker_result.resolved_events:
                    tier = event.confidence_tier
                    attribution_confidence_tier_counts[tier] = (
                        attribution_confidence_tier_counts.get(
                            tier,
                            0,
                        )
                        + 1
                    )
                if self.rule_causal_builder is not None:
                    causal_build_start = time.perf_counter()
                    causal_build = (
                        self.rule_causal_builder.build_with_stats(
                            tracker_result.created_events
                            + tracker_result.resolved_events
                        )
                    )
                    causal_rule_build_seconds += (
                        time.perf_counter() - causal_build_start
                    )
                    causal_rule_build_calls += 1
                    causal_rule_input_event_count += (
                        causal_build.stats.input_event_count
                    )
                    causal_rule_eligible_event_count += (
                        causal_build.stats.eligible_event_count
                    )
                    causal_rule_budget_count += (
                        causal_build.stats.budget_count
                    )
                    causal_rule_samples_generated += (
                        causal_build.stats.generated_sample_count
                    )
                    for reason, count in (
                            causal_build.stats.reason_counts):
                        causal_rule_skip_reason_counts[reason] = (
                            causal_rule_skip_reason_counts.get(
                                reason,
                                0,
                            )
                            + int(count)
                        )
                    causal_samples_pushed += (
                        self.causal_replay_buffer.extend(
                            causal_build.samples
                        )
                    )
                if self.counterfactual_proposal_builder is not None:
                    proposal_build_start = time.perf_counter()
                    counterfactual_proposal_build_calls += 1
                    try:
                        proposal_build = (
                            self.counterfactual_proposal_builder
                            .build_with_stats(
                                tracker_result.created_events
                                + tracker_result.resolved_events,
                                merge_records=(
                                    tracker_result.merge_records
                                ),
                            )
                        )
                    except Exception:
                        counterfactual_proposal_skip_reason_counts[
                            'proposal_build_failure'
                        ] = (
                            counterfactual_proposal_skip_reason_counts.get(
                                'proposal_build_failure',
                                0,
                            )
                            + 1
                        )
                    else:
                        counterfactual_proposal_input_event_count += (
                            proposal_build.stats.input_event_count
                        )
                        counterfactual_proposal_confirmed_event_count += (
                            proposal_build.stats.confirmed_event_count
                        )
                        counterfactual_proposal_budget_count += (
                            proposal_build.stats.budget_count
                        )
                        counterfactual_proposals_generated += (
                            proposal_build.stats
                            .generated_proposal_count
                        )
                        for reason, count in (
                                proposal_build.stats.reason_counts):
                            (
                                counterfactual_proposal_skip_reason_counts[
                                    reason
                                ]
                            ) = (
                                counterfactual_proposal_skip_reason_counts
                                .get(reason, 0)
                                + int(count)
                            )
                        self._counterfactual_proposal_outbox.extend(
                            proposal_build.proposals
                        )
                    finally:
                        counterfactual_proposal_build_seconds += (
                            time.perf_counter()
                            - proposal_build_start
                        )
            state_analysis_calls += int(
                next_info.get('state_analysis_calls', 0)
            )
            state_analysis_seconds += float(
                next_info.get('state_analysis_seconds', 0.0)
            )
            state_analysis_cache_hits += int(bool(
                next_info.get('state_analysis_cache_hit', False)
            ))
            state_analysis_degraded_count += int(
                next_info.get('state_analysis_degraded_count', 0)
            )
            physics_frames_total += int(next_info.get('frames_simulated', 0))
            fruit_count_total += int(next_obs.fruit_count)

            # 只有真实终止关闭 bootstrap；物理截断仍保留可信 final observation。
            next_graph = None
            if not terminated:
                next_graph, build_seconds, convert_seconds = self._build_graph_tensor(
                    next_obs,
                    next_info['action_candidates'],
                )
                graph_build_seconds += build_seconds
                tensor_convert_seconds += convert_seconds

            transition = TensorTransition(
                graph=graph,
                action_offset=action_offset,
                reward=reward,
                next_graph=next_graph,
                terminated=terminated,
                truncated=truncated,
            )
            replay_transitions_emitted += (
                self._emit_n_step_transitions(
                    self.n_step_accumulator.append(transition)
                )
            )
            transition_keys.append(transition_key)

            steps += 1
            total_reward += reward
            self._episode_reward += reward
            self._episode_length += 1

            if used_random:
                random_actions += 1
            else:
                greedy_actions += 1

            if episode_done:
                episode_rewards.append(self._episode_reward)
                episode_lengths.append(self._episode_length)
                episode_scores.append(next_obs.score)
                episode_end_offsets.append(steps)
                episode_terminated_flags.append(terminated)
                episode_truncated_flags.append(truncated)

                if terminated:
                    terminated_episodes += 1
                if truncated:
                    truncated_episodes += 1

                self.reset()
            else:
                self._obs = next_obs
                self._info = next_info
                self._current_graph = next_graph

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
            buffer_size=len(self.replay_buffer),
            replay_transitions_emitted=(
                replay_transitions_emitted
            ),
            n_step_pending_count=(
                self.n_step_accumulator.pending_count
            ),
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
            causal_samples_emitted=causal_samples_pushed,
            causal_rule_skip_reason_counts=tuple(sorted(
                causal_rule_skip_reason_counts.items()
            )),
            causal_buffer_size=(
                len(self.causal_replay_buffer)
                if self.causal_replay_buffer is not None
                else 0
            ),
            causal_context_count=(
                len(self.rule_causal_builder.context_cache)
                if self.rule_causal_builder is not None
                else 0
            ),
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
            counterfactual_history_size=(
                len(
                    self.counterfactual_proposal_builder.history
                )
                if self.counterfactual_proposal_builder is not None
                else 0
            ),
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
            counterfactual_proposal_skip_reason_counts=tuple(
                sorted(
                    counterfactual_proposal_skip_reason_counts.items()
                )
            ),
            counterfactual_proposal_outbox_size=len(
                self._counterfactual_proposal_outbox
            ),
            current_episode_reward=self._episode_reward,
            current_episode_length=self._episode_length,
            collect_seconds=time.perf_counter() - collect_start,
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
            attribution_events_interrupted=attribution_events_interrupted,
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
            merge_level_counts=tuple(sorted(
                merge_level_counts.items()
            )),
            max_fruit_level=max_fruit_level,
            physics_frames_total=physics_frames_total,
            fruit_count_total=fruit_count_total,
            graph_node_count_total=graph_node_count_total,
            graph_edge_count_total=graph_edge_count_total,
            graph_cache_hits=graph_cache_hits,
            graph_cache_misses=graph_cache_misses,
        )

    def _emit_n_step_transitions(self, transitions):
        """把 accumulator 新形成的经验顺序写入主 replay。"""

        transitions = tuple(transitions)
        if transitions:
            self.replay_buffer.extend(transitions)
        return len(transitions)

    def _build_graph_tensor(self, obs, candidates):
        """构建当前状态图并转成 replay 长期保存用的 CPU tensor。"""

        build_start = time.perf_counter()
        graph_data = self.graph_builder.build(obs, candidates)
        build_seconds = time.perf_counter() - build_start

        convert_start = time.perf_counter()
        graph = graph_to_tensor(graph_data, dtype=REPLAY_GRAPH_DTYPE)
        convert_seconds = time.perf_counter() - convert_start
        return graph, build_seconds, convert_seconds

    def _accumulate_reward_breakdown(self, totals, reward_breakdown):
        """把环境返回的单步 reward 明细累加到当前采集统计里。"""

        if reward_breakdown is None:
            return {}

        # 正式环境返回 RewardBreakdown 对象；测试或后续适配器也可以返回普通 dict。
        if hasattr(reward_breakdown, 'to_dict'):
            values = reward_breakdown.to_dict()
        elif isinstance(reward_breakdown, dict):
            values = reward_breakdown
        else:
            values = {
                field_name: getattr(reward_breakdown, field_name, 0.0)
                for field_name in REWARD_BREAKDOWN_FIELDS
            }

        for field_name in REWARD_BREAKDOWN_FIELDS:
            totals[field_name] += float(values.get(field_name, 0.0))
        return values

    def _validate_state_analysis_info(
            self,
            *,
            transition_key,
            terminated,
            next_info):
        """确保环境分析身份与 collector 的唯一轨迹身份完全一致。"""

        previous_analysis = next_info.get('previous_state_analysis')
        next_analysis = next_info.get('next_state_analysis')
        post_action_analysis = next_info.get(
            'post_action_state_analysis',
            next_analysis,
        )
        if previous_analysis is not None:
            if previous_analysis.transition_key != transition_key:
                raise RuntimeError(
                    'environment previous StateAnalysis key does not match '
                    'collector transition key'
                )

        if terminated:
            if next_analysis is not None:
                raise RuntimeError(
                    'terminal transition must not return next StateAnalysis'
                )

        if post_action_analysis is not None:
            expected_next_key = TransitionKey(
                worker_id=transition_key.worker_id,
                episode_id=transition_key.episode_id,
                step_index=transition_key.step_index + 1,
            )
            if post_action_analysis.transition_key != expected_next_key:
                raise RuntimeError(
                    'environment post-action StateAnalysis key is not '
                    'adjacent to '
                    'collector transition key'
                )
            if (
                    post_action_analysis.incoming_transition_key
                    != transition_key):
                raise RuntimeError(
                    'environment post-action StateAnalysis incoming key '
                    'does not '
                    'match collector transition key'
                )

    def _select_action(self, graph, action_count, epsilon):
        """根据 epsilon-greedy 策略返回 `(action_offset, used_random)`。"""

        if self.policy.should_explore(epsilon):
            return self.policy.random_action_offset(action_count), True

        q_values = self._evaluate_q_values(graph)
        if int(q_values.shape[0]) != action_count:
            raise RuntimeError(
                f'q_values length mismatch: got {q_values.shape[0]}, expected {action_count}'
            )

        return self.policy.greedy_action_offset(q_values), False

    def _evaluate_q_values(self, graph):
        """使用当前模型计算一张图的动作 Q 值。"""

        if self.model is None:
            raise ValueError('model is required for greedy action selection')

        with torch.no_grad():
            # GNNQNetwork 内部会把 GraphTensor 转到模型所在设备。
            # collector 这里只拿 CPU 上的 1D q_values 做 argmax 和长度检查。
            return self.model(graph).detach().cpu()

    def _validate_action_count(self, graph, action_count):
        """检查候选动作数量和图中 action 节点数量是否一致。"""

        graph_action_count = len(graph.action_node_indices)
        if graph_action_count != action_count:
            raise RuntimeError(
                f'graph action count mismatch: graph={graph_action_count}, '
                f'candidates={action_count}'
            )
