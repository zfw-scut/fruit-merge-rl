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

from daxigua_rl.reward import REWARD_BREAKDOWN_FIELDS
from daxigua_rl.graph.tensor import graph_to_tensor

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

    # 本次实际写入 replay buffer 的 transition 数量。
    steps: int

    # 本次采集中结束了多少局游戏。
    episodes: int

    # 本次采集得到的总 reward，包含未完成 episode 的部分 reward。
    total_reward: float

    # 本次采集中各 reward breakdown 字段的累计值。
    # 使用 `(字段名, 累计值)` 元组而不是裸 dict，避免 frozen dataclass 持有可变对象。
    reward_breakdown_totals: tuple = field(default_factory=tuple)

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
            seed=None):
        """创建 rollout collector。

        参数：
        - `env`: 类 Gym 的环境，当前预期为 `DaxiguaEnv`。
        - `graph_builder`: 当前预期为 `GraphBuilder`。
        - `replay_buffer`: `ReplayBuffer` 实例。
        - `model`: 可选 Q 网络；当 `epsilon < 1.0` 时必须提供。
        - `policy`: 可选动作策略，默认使用 `EpsilonGreedyPolicy`。
        - `seed`: 默认策略随机种子。
        """

        if not isinstance(replay_buffer, ReplayBuffer):
            raise TypeError(f'replay_buffer must be ReplayBuffer, got {type(replay_buffer)!r}')

        self.env = env
        self.graph_builder = graph_builder
        self.replay_buffer = replay_buffer
        self.model = model
        self.policy = policy or EpsilonGreedyPolicy(seed=seed)

        # `_obs` 和 `_info` 保存当前 episode 的最新状态。
        # collect_steps 第一次调用时如果发现它们为空，会自动 reset。
        self._obs = None
        self._info = None
        self._current_graph = None
        self._episode_reward = 0.0
        self._episode_length = 0

    def reset(self, seed=None, fruit_queue=None):
        """重置环境并开始一个新的 episode。

        这个方法主要给训练脚本显式控制初始种子或固定水果队列时使用。
        普通情况下可以直接调用 `collect_steps()`，collector 会自动 reset。
        """

        self._obs, self._info = self.env.reset(seed=seed, fruit_queue=fruit_queue)
        self._current_graph = None
        self._episode_reward = 0.0
        self._episode_length = 0
        return self._obs, self._info

    @property
    def has_state(self):
        """collector 当前是否已经持有一个可继续采集的环境状态。"""

        return self._obs is not None and self._info is not None

    def collect_steps(self, step_count, epsilon=1.0):
        """收集指定数量的 transition，并写入 replay buffer。

        `step_count` 表示要收集多少个环境 step，也就是多少次水果投放。
        如果中途 episode 结束，collector 会自动 reset 并继续采集，直到达到
        指定 transition 数量。
        """

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

            env_step_start = time.perf_counter()
            next_obs, reward, terminated, truncated, next_info = self.env.step(action_offset)
            env_step_seconds += time.perf_counter() - env_step_start
            done = terminated or truncated
            self._accumulate_reward_breakdown(
                reward_breakdown_totals,
                next_info.get('reward_breakdown'),
            )
            physics_frames_total += int(next_info.get('frames_simulated', 0))
            fruit_count_total += int(next_obs.fruit_count)

            # terminal/truncated transition 不 bootstrap，因此不强制构建 next_graph。
            # 非终止 transition 需要下一状态图，后续 DQN target 要读取 max_next_q。
            next_graph = None
            if not done:
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
            self.replay_buffer.push(transition)

            steps += 1
            total_reward += reward
            self._episode_reward += reward
            self._episode_length += 1

            if used_random:
                random_actions += 1
            else:
                greedy_actions += 1

            if done:
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
            current_episode_reward=self._episode_reward,
            current_episode_length=self._episode_length,
            collect_seconds=time.perf_counter() - collect_start,
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
            return

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
