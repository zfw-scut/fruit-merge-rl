"""最小标准 DQN 更新器。

本模块负责从 `ReplayBuffer` 中采样经验，计算 TD target，
并更新 online Q 网络参数。

当前第一版刻意保持朴素：

- 使用标准 DQN target，不做 Double DQN。
- 使用 target network 稳定 bootstrap 目标。
- 使用 SmoothL1Loss/Huber loss，降低大 TD 误差带来的震荡。
- 使用 GraphBatch 把多张不连通图合成一次批量 forward。
- 可选梯度裁剪，默认 `grad_clip_norm=10.0`。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from daxigua_rl.graph.tensor import collate_graph_tensors

from .replay_buffer import ReplayBuffer
from .tensor_transition import TensorTransition


@dataclass(frozen=True)
class DQNTrainerConfig:
    """DQN 更新器配置。

    这些参数属于训练算法，不属于游戏规则。
    第一版先集中放在一个 dataclass 中，方便后续训练脚本直接打印和保存配置。
    """

    # 折扣因子 gamma，用于衡量未来奖励的重要性。
    # target = reward + gamma * max_next_q。
    gamma: float = 0.99

    # 每次 train_step 从 replay buffer 采样多少条 transition。
    batch_size: int = 32

    # 每隔多少次参数更新，把 online_model 同步到 target_model。
    target_update_interval: int = 1000

    # 梯度裁剪阈值。None 表示不裁剪。
    grad_clip_norm: float | None = 10.0

    # 初始化 trainer 时是否立刻把 online_model 参数复制给 target_model。
    sync_target_on_init: bool = True


@dataclass(frozen=True)
class DQNTrainStats:
    """一次 `train_step()` 的训练统计。"""

    # 当前已经完成的参数更新次数。
    update_step: int

    # 本次 batch 的 SmoothL1Loss/Huber loss。
    loss: float

    # 当前状态下被选动作 Q(s, a) 的平均值。
    mean_q: float

    # TD target 的平均值。
    mean_target: float

    # batch 中即时 reward 的平均值。
    mean_reward: float

    # 平均绝对 TD 误差，便于观察预测和目标差距。
    mean_abs_td_error: float

    # 本次 batch 中使用 next_graph bootstrap 的 transition 数量。
    bootstrap_count: int

    # 本次 batch 大小。
    batch_size: int

    # 本次梯度范数；未启用梯度裁剪时仍会记录裁剪前范数。
    grad_norm: float

    # 本次更新后是否同步了 target network。
    target_synced: bool


class DQNTrainer:
    """标准 DQN 单步更新器。

    `DQNTrainer` 不负责采集经验；它只从 `ReplayBuffer` 抽样并更新模型。
    经验采集由 `RolloutCollector` 负责，游戏状态图构建由 `GraphBuilder` 负责。
    """

    def __init__(
            self,
            online_model,
            target_model,
            replay_buffer,
            optimizer,
            config=None,
            loss_fn=None):
        """创建 DQN 更新器。

        参数：
        - `online_model`: 正在训练的 Q 网络。
        - `target_model`: 用于计算 next_state bootstrap 目标的冻结 Q 网络。
        - `replay_buffer`: 保存经验对象的回放池；正式训练路径通常保存 `TensorTransition`。
        - `optimizer`: 只应该包含 online_model 参数。
        - `config`: DQNTrainerConfig。
        - `loss_fn`: 可选自定义 loss；默认使用 `nn.SmoothL1Loss()`。
        """

        if not isinstance(replay_buffer, ReplayBuffer):
            raise TypeError(f'replay_buffer must be ReplayBuffer, got {type(replay_buffer)!r}')

        self.online_model = online_model
        self.target_model = target_model
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.config = config or DQNTrainerConfig()
        self.loss_fn = loss_fn or nn.SmoothL1Loss()

        self._validate_config()
        self._update_step = 0

        # target_model 只负责生成训练目标，不应该被 optimizer 更新。
        # 即便调用者错误地把 target_model 参数也传给 optimizer，冻结参数也能多一层保护。
        self._freeze_target_model()

        if self.config.sync_target_on_init:
            self.sync_target_model()

    @property
    def update_step(self):
        """当前已经完成的 online_model 参数更新次数。"""

        return self._update_step

    def is_ready(self):
        """ReplayBuffer 当前是否足够执行一次训练。"""

        return self.replay_buffer.is_ready(self.config.batch_size)

    def sync_target_model(self):
        """把 online_model 参数完整复制给 target_model。"""

        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

    def train_step(self):
        """执行一次 DQN 参数更新，并返回训练统计。"""

        if not self.is_ready():
            raise ValueError(
                f'replay buffer has {len(self.replay_buffer)} items, '
                f'but batch_size={self.config.batch_size}'
            )

        batch = tuple(
            TensorTransition.from_transition(transition)
            for transition in self.replay_buffer.sample(self.config.batch_size)
        )

        # online_model 需要梯度，target_model 只做无梯度推理。
        self.online_model.train()
        self.target_model.eval()

        # 把 batch 内所有当前状态图拼成一张不连通大图，只做一次 online forward。
        current_graph_batch = collate_graph_tensors(transition.graph for transition in batch)
        current_q_flat = self.online_model(current_graph_batch)
        current_q_tensor = self._select_current_q(current_q_flat, current_graph_batch, batch)

        # target 同样批量计算：所有可 bootstrap 的 next_graph 拼成一张不连通大图。
        target_tensor, bootstrap_count = self._compute_target_values(batch, current_q_tensor)
        rewards = [float(transition.reward) for transition in batch]

        # TD error = 当前 Q 预测 - 训练目标。
        # loss_fn 默认是 SmoothL1Loss，也就是 Huber 风格损失。
        loss = self.loss_fn(current_q_tensor, target_tensor)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # 先计算/裁剪梯度，再更新 online_model。
        grad_norm = self._clip_or_measure_grad_norm()
        self.optimizer.step()

        self._update_step += 1
        target_synced = self._maybe_sync_target_model()

        with torch.no_grad():
            td_error = current_q_tensor.detach() - target_tensor.detach()
            stats = DQNTrainStats(
                update_step=self._update_step,
                loss=float(loss.detach().cpu().item()),
                mean_q=float(current_q_tensor.detach().mean().cpu().item()),
                mean_target=float(target_tensor.detach().mean().cpu().item()),
                mean_reward=sum(rewards) / len(rewards),
                mean_abs_td_error=float(td_error.abs().mean().cpu().item()),
                bootstrap_count=bootstrap_count,
                batch_size=len(batch),
                grad_norm=float(grad_norm.detach().cpu().item()),
                target_synced=target_synced,
            )

        return stats

    def _select_current_q(self, q_values, graph_batch, transitions):
        """从扁平 Q 输出中取出每条 transition 实际执行动作的 Q 值。"""

        if q_values.dim() != 1:
            raise ValueError('q_values must have shape [total_action_count]')
        if int(q_values.shape[0]) != graph_batch.action_count:
            raise RuntimeError(
                f'q_values length mismatch: got {q_values.shape[0]}, '
                f'expected {graph_batch.action_count}'
            )

        selected_indices = [
            action_start + transition.action_offset
            for transition, (action_start, _action_end) in zip(transitions, graph_batch.action_slices)
        ]
        selected_indices = torch.tensor(
            selected_indices,
            dtype=torch.long,
            device=q_values.device,
        )
        return q_values.index_select(0, selected_indices)

    def _compute_target_values(self, transitions, selected_q):
        """批量计算 DQN TD target。

        标准 DQN target：

            reward + gamma * max_a' target_model(next_graph)[a']

        terminal/truncated transition 不使用 bootstrap，target 直接等于 reward。
        """

        rewards = torch.tensor(
            [float(transition.reward) for transition in transitions],
            dtype=selected_q.dtype,
            device=selected_q.device,
        )
        target_values = rewards.clone()

        bootstrap_items = [
            (transition_index, transition)
            for transition_index, transition in enumerate(transitions)
            if transition.can_bootstrap
        ]
        if not bootstrap_items:
            return target_values, 0

        next_graph_batch = collate_graph_tensors(
            transition.next_graph
            for _transition_index, transition in bootstrap_items
        )
        with torch.no_grad():
            # 标准 DQN 使用 target_model 同时完成动作选择和动作估值。
            # Double DQN 会改成 online_model 选动作、target_model 估值；第一版暂不做。
            next_q_flat = self.target_model(next_graph_batch)
            max_next_q = self._max_q_by_graph(next_q_flat, next_graph_batch)
            max_next_q = max_next_q.to(device=selected_q.device, dtype=selected_q.dtype)

        bootstrap_indices = torch.tensor(
            [transition_index for transition_index, _transition in bootstrap_items],
            dtype=torch.long,
            device=selected_q.device,
        )
        target_values[bootstrap_indices] = (
            rewards.index_select(0, bootstrap_indices)
            + self.config.gamma * max_next_q
        )
        return target_values, len(bootstrap_items)

    def _max_q_by_graph(self, q_values, graph_batch):
        """按 GraphBatch 中每张原始图分别求动作 Q 最大值。"""

        if q_values.dim() != 1:
            raise ValueError('q_values must have shape [total_action_count]')
        if int(q_values.shape[0]) != graph_batch.action_count:
            raise RuntimeError(
                f'q_values length mismatch: got {q_values.shape[0]}, '
                f'expected {graph_batch.action_count}'
            )

        max_values = []
        for action_start, action_end in graph_batch.action_slices:
            if action_end <= action_start:
                raise ValueError('each graph in GraphBatch must contain at least one action')
            max_values.append(q_values[action_start:action_end].max())
        return torch.stack(max_values)

    def _clip_or_measure_grad_norm(self):
        """裁剪或测量 online_model 的梯度范数。"""

        parameters = [
            parameter
            for parameter in self.online_model.parameters()
            if parameter.grad is not None
        ]

        if not parameters:
            return torch.tensor(0.0)

        if self.config.grad_clip_norm is not None:
            # clip_grad_norm_ 返回裁剪前的总范数，便于观察是否频繁触发裁剪。
            return torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip_norm)

        # 不裁剪时也记录当前总梯度范数。这里用 torch.norm 保持返回类型一致。
        per_parameter_norms = [parameter.grad.detach().norm(2) for parameter in parameters]
        return torch.norm(torch.stack(per_parameter_norms), 2)

    def _maybe_sync_target_model(self):
        """达到同步间隔时更新 target_model，返回本次是否同步。"""

        if self._update_step % self.config.target_update_interval != 0:
            return False

        self.sync_target_model()
        return True

    def _freeze_target_model(self):
        """冻结 target_model 参数，防止它参与反向传播或 optimizer 更新。"""

        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.target_model.eval()

    def _validate_config(self):
        """检查 DQN 配置中的明显错误。"""

        if self.config.gamma < 0.0 or self.config.gamma > 1.0:
            raise ValueError('gamma must be in [0, 1]')
        if int(self.config.batch_size) <= 0:
            raise ValueError('batch_size must be positive')
        if int(self.config.target_update_interval) <= 0:
            raise ValueError('target_update_interval must be positive')
        if self.config.grad_clip_norm is not None and self.config.grad_clip_norm <= 0.0:
            raise ValueError('grad_clip_norm must be positive or None')
