"""最小标准 DQN 更新器。

本模块负责从 `ReplayBuffer` 中采样 `Transition`，计算 TD target，
并更新 online Q 网络参数。

当前第一版刻意保持朴素：

- 使用标准 DQN target，不做 Double DQN。
- 使用 target network 稳定 bootstrap 目标。
- 使用 SmoothL1Loss/Huber loss，降低大 TD 误差带来的震荡。
- 逐条图 forward，不做 GraphBatch。
- 可选梯度裁剪，默认 `grad_clip_norm=10.0`。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .replay_buffer import ReplayBuffer


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
        - `replay_buffer`: 保存 `Transition` 的经验回放池。
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

        batch = self.replay_buffer.sample(self.config.batch_size)

        # online_model 需要梯度，target_model 只做无梯度推理。
        self.online_model.train()
        self.target_model.eval()

        current_q_values = []
        target_values = []
        rewards = []
        bootstrap_count = 0

        for transition in batch:
            # online_model(graph) 输出当前状态所有候选动作的 Q 值。
            # action_offset 指向这条 transition 当时实际选择的动作。
            q_values = self.online_model(transition.graph)
            selected_q = q_values[transition.action_offset]
            current_q_values.append(selected_q)

            # target value 是标量张量。terminal/truncated transition 不 bootstrap。
            target_value = self._compute_target_value(transition, selected_q)
            target_values.append(target_value)
            rewards.append(float(transition.reward))

            if transition.can_bootstrap:
                bootstrap_count += 1

        # shape: [batch_size]
        current_q_tensor = torch.stack(current_q_values)
        target_tensor = torch.stack(target_values).to(device=current_q_tensor.device)

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

    def _compute_target_value(self, transition, selected_q):
        """计算一条 transition 的 TD target 标量张量。

        标准 DQN target：

            reward + gamma * max_a' target_model(next_graph)[a']

        terminal/truncated transition 不使用 bootstrap，target 直接等于 reward。
        """

        # reward_tensor 使用 selected_q 的 device/dtype，避免 CPU/GPU 或 dtype 混用。
        reward_tensor = torch.tensor(
            float(transition.reward),
            dtype=selected_q.dtype,
            device=selected_q.device,
        )

        if not transition.can_bootstrap:
            return reward_tensor

        with torch.no_grad():
            # 标准 DQN 使用 target_model 同时完成动作选择和动作估值。
            # Double DQN 会改成 online_model 选动作、target_model 估值；第一版暂不做。
            next_q_values = self.target_model(transition.next_graph)
            max_next_q = next_q_values.max().to(device=selected_q.device, dtype=selected_q.dtype)

        return reward_tensor + self.config.gamma * max_next_q

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
