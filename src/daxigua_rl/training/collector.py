"""单进程 rollout 采集器。

RolloutCollector 负责把当前已经完成的几个训练零件串起来：

    DaxiguaEnv -> GraphBuilder -> GNNQNetwork/EpsilonGreedyPolicy
    -> TensorTransition -> ReplayBuffer

它只负责“玩游戏并收集经验”，不负责从 replay buffer 采样训练，也不负责
计算 DQN loss、更新模型参数或同步 target network。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from daxigua_rl.graph.tensor import graph_to_tensor

from .replay_buffer import ReplayBuffer
from .tensor_transition import TensorTransition


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

    # 本次采集中每个已结束 episode 的累计 reward。
    episode_rewards: tuple = field(default_factory=tuple)

    # 本次采集中每个已结束 episode 的投放次数。
    episode_lengths: tuple = field(default_factory=tuple)

    # 本次采集中每个已结束 episode 的最终游戏分数。
    episode_scores: tuple = field(default_factory=tuple)

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
        self._episode_reward = 0.0
        self._episode_length = 0

    def reset(self, seed=None, fruit_queue=None):
        """重置环境并开始一个新的 episode。

        这个方法主要给训练脚本显式控制初始种子或固定水果队列时使用。
        普通情况下可以直接调用 `collect_steps()`，collector 会自动 reset。
        """

        self._obs, self._info = self.env.reset(seed=seed, fruit_queue=fruit_queue)
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

        steps = 0
        total_reward = 0.0
        episode_rewards = []
        episode_lengths = []
        episode_scores = []
        terminated_episodes = 0
        truncated_episodes = 0
        random_actions = 0
        greedy_actions = 0

        while steps < step_count:
            candidates = tuple(self._info['action_candidates'])
            if not candidates:
                # 正常情况下，terminal step 之后会立即 reset；这里是额外容错。
                self.reset()
                continue

            graph_data = self.graph_builder.build(self._obs, candidates)
            graph = graph_to_tensor(graph_data)
            action_count = len(candidates)
            self._validate_action_count(graph, action_count)

            action_offset, used_random = self._select_action(
                graph=graph,
                action_count=action_count,
                epsilon=epsilon,
            )

            next_obs, reward, terminated, truncated, next_info = self.env.step(action_offset)
            done = terminated or truncated

            # terminal/truncated transition 不 bootstrap，因此不强制构建 next_graph。
            # 非终止 transition 需要下一状态图，后续 DQN target 要读取 max_next_q。
            next_graph = None
            if not done:
                next_graph_data = self.graph_builder.build(next_obs, next_info['action_candidates'])
                next_graph = graph_to_tensor(next_graph_data)

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

                if terminated:
                    terminated_episodes += 1
                if truncated:
                    truncated_episodes += 1

                self.reset()
            else:
                self._obs = next_obs
                self._info = next_info

        return RolloutStats(
            steps=steps,
            episodes=len(episode_rewards),
            total_reward=total_reward,
            episode_rewards=tuple(episode_rewards),
            episode_lengths=tuple(episode_lengths),
            episode_scores=tuple(episode_scores),
            terminated_episodes=terminated_episodes,
            truncated_episodes=truncated_episodes,
            random_actions=random_actions,
            greedy_actions=greedy_actions,
            buffer_size=len(self.replay_buffer),
            current_episode_reward=self._episode_reward,
            current_episode_length=self._episode_length,
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
