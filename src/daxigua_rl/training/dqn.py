"""Double DQN 更新器。

本模块负责从 `ReplayBuffer` 中采样经验，计算 TD target，
并更新 online Q 网络参数。

- 使用 online network 选择下一动作、target network 估值的 Double DQN target。
- 支持由 rollout 层预先聚合好的 n-step return。
- 使用 target network 稳定 bootstrap 目标。
- 使用 SmoothL1Loss/Huber loss，降低大 TD 误差带来的震荡。
- 使用 GraphBatch 把多张不连通图合成一次批量 forward。
- 可选梯度裁剪，默认 `grad_clip_norm=10.0`。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from daxigua_rl.attribution.causal_replay import (
    CausalReplayBuffer,
    CausalSample,
)
from daxigua_rl.graph.tensor import collate_graph_tensors
from daxigua_rl.reward import merge_utility

from .replay_buffer import ReplayBuffer
from .tensor_transition import TensorTransition


@dataclass(frozen=True)
class DQNTrainerConfig:
    """DQN 更新器配置。

    这些参数属于训练算法，不属于游戏规则。
    第一版先集中放在一个 dataclass 中，方便后续训练脚本直接打印和保存配置。
    """

    # 折扣因子 gamma，用于衡量未来奖励的重要性。
    # target = n_step_reward + gamma ** bootstrap_steps * next_q。
    gamma: float = 0.99

    # rollout 侧允许生成的最大 return horizon。正式训练使用 3-step；
    # terminal/truncated episode 尾部可以自然缩短为 1 或 2。
    n_step: int = 3

    # 每次 train_step 从 replay buffer 采样多少条 transition。
    batch_size: int = 32

    # 每隔多少次参数更新，把 online_model 同步到 target_model。
    target_update_interval: int = 1000

    # 梯度裁剪阈值。None 表示不裁剪。
    grad_clip_norm: float | None = 10.0

    # 初始化 trainer 时是否立刻把 online_model 参数复制给 target_model。
    sync_target_on_init: bool = True

    # 稀疏因果回放的最大 batch；样本不足时使用当前全部样本，不阻塞 TD 更新。
    causal_batch_size: int = 32

    # 每多少次参数更新读取一次因果回放。默认每两次更新一次。
    causal_update_interval: int = 2

    # 高置信规则动作排序监督权重。
    lambda_rule: float = 0.15

    # 反事实回报差值和局部 Shapley 监督的共同权重。
    lambda_counterfactual: float = 0.10

    # 把反事实回报差值归一化到与 Q 排序相近的量级；默认采用 7 级合成效用。
    counterfactual_return_scale: float = merge_utility(7)

    # 归一化后差值 target 的绝对裁剪上限，抑制少量异常分支主导梯度。
    counterfactual_target_clip: float = 5.0


@dataclass(frozen=True)
class DQNTrainStats:
    """一次 `train_step()` 的训练统计。"""

    # 当前已经完成的参数更新次数。
    update_step: int

    # 本次 batch 的 SmoothL1Loss/Huber loss。
    loss: float

    # TD 子损失；总 loss 还可能包含规则排序和反事实差值监督。
    td_loss: float

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

    # 从 ReplayBuffer 随机采样 batch 的耗时，单位秒。
    sample_seconds: float = 0.0

    # 当前状态图 batch 拼接耗时，单位秒。
    current_collate_seconds: float = 0.0

    # online_model 当前 Q 前向耗时，单位秒。
    online_forward_seconds: float = 0.0

    # target_model 下一状态 Q 和 TD target 计算耗时，单位秒。
    target_compute_seconds: float = 0.0

    # loss.backward() 反向传播耗时，单位秒。
    backward_seconds: float = 0.0

    # 梯度裁剪、optimizer.step() 和 target 同步耗时，单位秒。
    optimizer_seconds: float = 0.0

    # train_step 整体耗时，单位秒。
    train_step_seconds: float = 0.0

    # 当前 update 是否实际读取了因果回放，以及读取的总样本数。
    causal_update_applied: bool = False
    causal_batch_size: int = 0
    rule_batch_size: int = 0
    counterfactual_batch_size: int = 0
    shapley_batch_size: int = 0

    # 三路损失拆分；weighted 字段是乘以 lambda 后对总 loss 的真实贡献。
    rule_rank_loss: float = 0.0
    weighted_rule_rank_loss: float = 0.0
    counterfactual_loss: float = 0.0
    weighted_counterfactual_loss: float = 0.0

    # 规则方向正确率、达到完整 margin 的比例，以及经验差值符号正确率。
    rule_pair_accuracy: float = 0.0
    rule_margin_satisfaction_rate: float = 0.0
    counterfactual_sign_accuracy: float = 0.0
    counterfactual_mean_abs_error: float = 0.0

    # 因果 replay 采样、图拼接与前向的额外耗时。
    causal_sample_seconds: float = 0.0
    causal_collate_seconds: float = 0.0
    causal_forward_seconds: float = 0.0


class DQNTrainer:
    """Double DQN n-step 更新器。

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
            loss_fn=None,
            causal_replay_buffer=None):
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
        if (
                causal_replay_buffer is not None
                and not isinstance(
                    causal_replay_buffer,
                    CausalReplayBuffer)):
            raise TypeError(
                'causal_replay_buffer must be CausalReplayBuffer or None'
            )
        self.causal_replay_buffer = causal_replay_buffer

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

    def restore_update_step(self, update_step):
        """恢复已完成更新计数；模型和优化器状态由训练入口单独加载。

        公开这个窄接口可以避免 resume 路径直接改写私有字段，同时保留 target
        同步周期的相位。恢复值必须是非负整数。
        """

        if isinstance(update_step, bool):
            raise TypeError('update_step must be an integer')
        try:
            normalized = int(update_step)
        except (TypeError, ValueError) as exc:
            raise TypeError('update_step must be an integer') from exc
        if normalized != update_step or normalized < 0:
            raise ValueError('update_step must be a non-negative integer')
        self._update_step = normalized

    def is_ready(self):
        """ReplayBuffer 当前是否足够执行一次训练。"""

        return self.replay_buffer.is_ready(self.config.batch_size)

    def sync_target_model(self):
        """把 online_model 参数完整复制给 target_model。"""

        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

    def train_step(self):
        """执行一次 DQN 参数更新，并返回训练统计。"""

        train_step_start = time.perf_counter()
        if not self.is_ready():
            raise ValueError(
                f'replay buffer has {len(self.replay_buffer)} items, '
                f'but batch_size={self.config.batch_size}'
            )

        sample_start = time.perf_counter()
        batch = self.replay_buffer.sample(self.config.batch_size)
        sample_seconds = time.perf_counter() - sample_start
        for transition in batch:
            if not isinstance(transition, TensorTransition):
                raise TypeError(
                    'DQNTrainer expects ReplayBuffer to contain TensorTransition; '
                    f'got {type(transition)!r}'
                )
        self._validate_transition_horizons(batch)

        # online_model 需要梯度，target_model 只做无梯度推理。
        self.online_model.train()
        self.target_model.eval()

        # 把 batch 内所有当前状态图拼成一张不连通大图，只做一次 online forward。
        current_collate_start = time.perf_counter()
        current_graph_batch = collate_graph_tensors(transition.graph for transition in batch)
        current_collate_seconds = time.perf_counter() - current_collate_start

        online_forward_start = time.perf_counter()
        current_q_flat = self.online_model(current_graph_batch)
        self._synchronize_model_device()
        online_forward_seconds = time.perf_counter() - online_forward_start
        current_q_tensor = self._select_current_q(current_q_flat, current_graph_batch, batch)

        # target 同样批量计算：所有可 bootstrap 的 next_graph 拼成一张不连通大图。
        target_start = time.perf_counter()
        target_tensor, bootstrap_count = self._compute_target_values(batch, current_q_tensor)
        self._synchronize_model_device()
        target_compute_seconds = time.perf_counter() - target_start
        rewards = [float(transition.reward) for transition in batch]

        # TD error = 当前 Q 预测 - 训练目标。
        # loss_fn 默认是 SmoothL1Loss，也就是 Huber 风格损失。
        td_loss = self.loss_fn(current_q_tensor, target_tensor)

        # 因果样本直接约束同一个 Q 输出，不新增 causal head。规则排序与经验
        # 反事实差值在同一次 backward 中和 TD 联合优化；没有样本时返回可微零值。
        causal_result = self._compute_causal_losses(current_q_tensor)
        rule_rank_loss = causal_result['rule_rank_loss']
        counterfactual_loss = causal_result['counterfactual_loss']
        weighted_rule_rank_loss = (
            self.config.lambda_rule * rule_rank_loss
        )
        weighted_counterfactual_loss = (
            self.config.lambda_counterfactual
            * counterfactual_loss
        )
        loss = (
            td_loss
            + weighted_rule_rank_loss
            + weighted_counterfactual_loss
        )
        self._require_finite('current_q', current_q_tensor)
        self._require_finite('target', target_tensor)
        self._require_finite('td_loss', td_loss)
        self._require_finite('rule_rank_loss', rule_rank_loss)
        self._require_finite('counterfactual_loss', counterfactual_loss)
        self._require_finite('total_loss', loss)

        self.optimizer.zero_grad(set_to_none=True)
        backward_start = time.perf_counter()
        loss.backward()
        self._synchronize_model_device()
        backward_seconds = time.perf_counter() - backward_start

        # 先计算/裁剪梯度，再更新 online_model。
        optimizer_start = time.perf_counter()
        grad_norm = self._clip_or_measure_grad_norm()
        self.optimizer.step()

        self._update_step += 1
        target_synced = self._maybe_sync_target_model()
        self._synchronize_model_device()
        optimizer_seconds = time.perf_counter() - optimizer_start

        with torch.no_grad():
            td_error = current_q_tensor.detach() - target_tensor.detach()
            stats = DQNTrainStats(
                update_step=self._update_step,
                loss=float(loss.detach().cpu().item()),
                td_loss=float(td_loss.detach().cpu().item()),
                mean_q=float(current_q_tensor.detach().mean().cpu().item()),
                mean_target=float(target_tensor.detach().mean().cpu().item()),
                mean_reward=sum(rewards) / len(rewards),
                mean_abs_td_error=float(td_error.abs().mean().cpu().item()),
                bootstrap_count=bootstrap_count,
                batch_size=len(batch),
                grad_norm=float(grad_norm.detach().cpu().item()),
                target_synced=target_synced,
                sample_seconds=sample_seconds,
                current_collate_seconds=current_collate_seconds,
                online_forward_seconds=online_forward_seconds,
                target_compute_seconds=target_compute_seconds,
                backward_seconds=backward_seconds,
                optimizer_seconds=optimizer_seconds,
                train_step_seconds=time.perf_counter() - train_step_start,
                causal_update_applied=causal_result[
                    'causal_update_applied'
                ],
                causal_batch_size=causal_result['causal_batch_size'],
                rule_batch_size=causal_result['rule_batch_size'],
                counterfactual_batch_size=causal_result[
                    'counterfactual_batch_size'
                ],
                shapley_batch_size=causal_result[
                    'shapley_batch_size'
                ],
                rule_rank_loss=float(
                    rule_rank_loss.detach().cpu().item()
                ),
                weighted_rule_rank_loss=float(
                    weighted_rule_rank_loss.detach().cpu().item()
                ),
                counterfactual_loss=float(
                    counterfactual_loss.detach().cpu().item()
                ),
                weighted_counterfactual_loss=float(
                    weighted_counterfactual_loss.detach().cpu().item()
                ),
                rule_pair_accuracy=causal_result[
                    'rule_pair_accuracy'
                ],
                rule_margin_satisfaction_rate=causal_result[
                    'rule_margin_satisfaction_rate'
                ],
                counterfactual_sign_accuracy=causal_result[
                    'counterfactual_sign_accuracy'
                ],
                counterfactual_mean_abs_error=causal_result[
                    'counterfactual_mean_abs_error'
                ],
                causal_sample_seconds=causal_result[
                    'causal_sample_seconds'
                ],
                causal_collate_seconds=causal_result[
                    'causal_collate_seconds'
                ],
                causal_forward_seconds=causal_result[
                    'causal_forward_seconds'
                ],
            )

        return stats

    def _compute_causal_losses(self, reference_tensor):
        """读取一次稀疏因果 batch，并计算规则排序与经验差值损失。

        ``reference_tensor`` 只用于继承当前模型 device/dtype，并保证无因果样本时
        返回的零标量能安全参与同一个总 loss。因果 replay 与主 TD replay 完全
        独立，任何空池或调度间隔都只会令本次 update 退化为纯 TD。
        """

        zero = reference_tensor.new_zeros(())
        result = {
            'rule_rank_loss': zero,
            'counterfactual_loss': zero,
            'causal_update_applied': False,
            'causal_batch_size': 0,
            'rule_batch_size': 0,
            'counterfactual_batch_size': 0,
            'shapley_batch_size': 0,
            'rule_pair_accuracy': 0.0,
            'rule_margin_satisfaction_rate': 0.0,
            'counterfactual_sign_accuracy': 0.0,
            'counterfactual_mean_abs_error': 0.0,
            'causal_sample_seconds': 0.0,
            'causal_collate_seconds': 0.0,
            'causal_forward_seconds': 0.0,
        }
        replay = self.causal_replay_buffer
        next_update_step = self._update_step + 1
        if (
                replay is None
                or len(replay) == 0
                or next_update_step
                % self.config.causal_update_interval != 0
                or (
                    self.config.lambda_rule == 0.0
                    and self.config.lambda_counterfactual == 0.0
                )):
            return result

        sample_start = time.perf_counter()
        samples = replay.sample(min(
            self.config.causal_batch_size,
            len(replay),
        ))
        result['causal_sample_seconds'] = (
            time.perf_counter() - sample_start
        )
        if any(not isinstance(sample, CausalSample) for sample in samples):
            raise TypeError(
                'CausalReplayBuffer must contain CausalSample values'
            )

        collate_start = time.perf_counter()
        graph_batch = collate_graph_tensors(
            sample.graph
            for sample in samples
        )
        result['causal_collate_seconds'] = (
            time.perf_counter() - collate_start
        )

        forward_start = time.perf_counter()
        q_values = self.online_model(graph_batch)
        self._synchronize_model_device()
        result['causal_forward_seconds'] = (
            time.perf_counter() - forward_start
        )
        actual_indices = torch.tensor(
            [
                action_start + sample.actual_action_offset
                for sample, (action_start, _action_end)
                in zip(samples, graph_batch.action_slices)
            ],
            dtype=torch.long,
            device=q_values.device,
        )
        comparison_indices = torch.tensor(
            [
                action_start + sample.comparison_action_offset
                for sample, (action_start, _action_end)
                in zip(samples, graph_batch.action_slices)
            ],
            dtype=torch.long,
            device=q_values.device,
        )
        q_delta = (
            q_values.index_select(0, actual_indices)
            - q_values.index_select(0, comparison_indices)
        )

        rule_offsets = [
            offset
            for offset, sample in enumerate(samples)
            if sample.supervision_kind == 'rule'
        ]
        empirical_offsets = [
            offset
            for offset, sample in enumerate(samples)
            if (
                sample.supervision_kind
                in {'counterfactual', 'shapley'}
                and sample.target_delta is not None
            )
        ]

        if rule_offsets:
            indices = torch.tensor(
                rule_offsets,
                dtype=torch.long,
                device=q_delta.device,
            )
            deltas = q_delta.index_select(0, indices)
            directions = q_delta.new_tensor([
                samples[offset].direction
                for offset in rule_offsets
            ])
            margins = q_delta.new_tensor([
                samples[offset].target_margin
                for offset in rule_offsets
            ])
            confidences = q_delta.new_tensor([
                samples[offset].confidence
                for offset in rule_offsets
            ])
            signed_delta = directions * deltas
            violations = torch.relu(margins - signed_delta)
            result['rule_rank_loss'] = (
                confidences * violations
            ).mean()
            with torch.no_grad():
                result['rule_pair_accuracy'] = float(
                    (signed_delta > 0.0).float().mean().cpu().item()
                )
                result['rule_margin_satisfaction_rate'] = float(
                    (signed_delta >= margins).float().mean().cpu().item()
                )

        if empirical_offsets:
            indices = torch.tensor(
                empirical_offsets,
                dtype=torch.long,
                device=q_delta.device,
            )
            deltas = q_delta.index_select(0, indices)
            targets = q_delta.new_tensor([
                samples[offset].target_delta
                for offset in empirical_offsets
            ])
            confidences = q_delta.new_tensor([
                samples[offset].confidence
                for offset in empirical_offsets
            ])
            scale = self.config.counterfactual_return_scale
            # 只裁剪监督目标。若同时裁剪预测，预测超出边界后 clamp 的导数会
            # 变成零，造成“loss 非零但无法把 Q 差值拉回”的训练死区。
            normalized_deltas = deltas / scale
            normalized_targets = torch.clamp(
                targets / scale,
                min=-self.config.counterfactual_target_clip,
                max=self.config.counterfactual_target_clip,
            )
            element_losses = F.smooth_l1_loss(
                normalized_deltas,
                normalized_targets,
                reduction='none',
            )
            result['counterfactual_loss'] = (
                confidences * element_losses
            ).mean()
            with torch.no_grad():
                target_signs = torch.sign(targets)
                sign_matches = (
                    torch.sign(deltas) == target_signs
                ).float()
                result['counterfactual_sign_accuracy'] = float(
                    sign_matches.mean().cpu().item()
                )
                result['counterfactual_mean_abs_error'] = float(
                    (deltas - targets).abs().mean().cpu().item()
                )

        result['causal_update_applied'] = True
        result['causal_batch_size'] = len(samples)
        result['rule_batch_size'] = len(rule_offsets)
        result['counterfactual_batch_size'] = sum(
            samples[offset].supervision_kind == 'counterfactual'
            for offset in empirical_offsets
        )
        result['shapley_batch_size'] = sum(
            samples[offset].supervision_kind == 'shapley'
            for offset in empirical_offsets
        )
        return result

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
        """批量计算 Double DQN n-step TD target。

        Double DQN target：

            a* = argmax_a online_model(next_graph)[a]
            target = n_step_reward
                     + gamma ** bootstrap_steps
                     * target_model(next_graph)[a*]

        只有真实 terminated transition 不使用 bootstrap，target 直接等于 reward。
        truncated transition 保留可信 final observation，仍计算下一状态 Q 值。
        """

        self._validate_transition_horizons(transitions)
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

        # Double DQN 把动作选择和动作估值拆给两个网络。online network 在
        # train_step 主体中通常处于 train mode；若模型配置了 dropout，直接用
        # train mode 选择下一动作会让 TD target 随机抖动。因此这里只在无梯度选择
        # 期间临时切到 eval，并在 finally 中精确恢复调用前模式。
        online_was_training = self.online_model.training
        self.online_model.eval()
        with torch.no_grad():
            try:
                online_next_q = self.online_model(next_graph_batch)
                selected_next_indices = (
                    self._argmax_indices_by_graph(
                        online_next_q,
                        next_graph_batch,
                    )
                )
            finally:
                self.online_model.train(online_was_training)

            target_next_q = self.target_model(next_graph_batch)
            selected_next_indices = selected_next_indices.to(
                device=target_next_q.device,
            )
            selected_next_q = target_next_q.index_select(
                0,
                selected_next_indices,
            )
            selected_next_q = selected_next_q.to(
                device=selected_q.device,
                dtype=selected_q.dtype,
            )

        bootstrap_indices = torch.tensor(
            [transition_index for transition_index, _transition in bootstrap_items],
            dtype=torch.long,
            device=selected_q.device,
        )
        bootstrap_discounts = torch.tensor(
            [
                self.config.gamma ** transition.bootstrap_steps
                for _transition_index, transition in bootstrap_items
            ],
            dtype=selected_q.dtype,
            device=selected_q.device,
        )
        target_values[bootstrap_indices] = (
            rewards.index_select(0, bootstrap_indices)
            + bootstrap_discounts * selected_next_q
        )
        return target_values, len(bootstrap_items)

    def _argmax_indices_by_graph(self, q_values, graph_batch):
        """返回每张图 online argmax 在扁平 Q 数组中的绝对下标。"""

        if q_values.dim() != 1:
            raise ValueError(
                'q_values must have shape [total_action_count]'
            )
        if int(q_values.shape[0]) != graph_batch.action_count:
            raise RuntimeError(
                f'q_values length mismatch: got {q_values.shape[0]}, '
                f'expected {graph_batch.action_count}'
            )

        selected_indices = []
        for action_start, action_end in graph_batch.action_slices:
            if action_end <= action_start:
                raise ValueError(
                    'each graph in GraphBatch must contain at least one '
                    'action'
                )
            local_offset = int(torch.argmax(
                q_values[action_start:action_end]
            ).item())
            selected_indices.append(action_start + local_offset)
        return torch.tensor(
            selected_indices,
            dtype=torch.long,
            device=q_values.device,
        )

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
            return torch.nn.utils.clip_grad_norm_(
                parameters,
                self.config.grad_clip_norm,
                error_if_nonfinite=True,
            )

        # 不裁剪时也记录当前总梯度范数。这里用 torch.norm 保持返回类型一致。
        per_parameter_norms = [parameter.grad.detach().norm(2) for parameter in parameters]
        total_norm = torch.norm(torch.stack(per_parameter_norms), 2)
        self._require_finite('gradient_norm', total_norm)
        return total_norm

    @staticmethod
    def _require_finite(name, tensor):
        """在污染参数和 checkpoint 之前拒绝 NaN/Inf。"""

        if not bool(torch.isfinite(tensor).all().detach().cpu().item()):
            raise FloatingPointError(
                f'non-finite {name} detected before optimizer step'
            )

    def _maybe_sync_target_model(self):
        """达到同步间隔时更新 target_model，返回本次是否同步。"""

        if self._update_step % self.config.target_update_interval != 0:
            return False

        self.sync_target_model()
        return True

    def _synchronize_model_device(self):
        """如果模型在 CUDA 上，等待当前设备计算完成以获得可信耗时。"""

        try:
            device = next(self.online_model.parameters()).device
        except StopIteration:
            return
        if device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _freeze_target_model(self):
        """冻结 target_model 参数，防止它参与反向传播或 optimizer 更新。"""

        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self.target_model.eval()

    def _validate_config(self):
        """检查 DQN 配置中的明显错误。"""

        if self.config.gamma < 0.0 or self.config.gamma > 1.0:
            raise ValueError('gamma must be in [0, 1]')
        if (
                isinstance(self.config.n_step, bool)
                or not isinstance(self.config.n_step, int)
                or self.config.n_step <= 0):
            raise ValueError('n_step must be a positive integer')
        if int(self.config.batch_size) <= 0:
            raise ValueError('batch_size must be positive')
        if int(self.config.target_update_interval) <= 0:
            raise ValueError('target_update_interval must be positive')
        if self.config.grad_clip_norm is not None and self.config.grad_clip_norm <= 0.0:
            raise ValueError('grad_clip_norm must be positive or None')
        if int(self.config.causal_batch_size) <= 0:
            raise ValueError('causal_batch_size must be positive')
        if int(self.config.causal_update_interval) <= 0:
            raise ValueError('causal_update_interval must be positive')
        for field_name in (
                'lambda_rule',
                'lambda_counterfactual'):
            value = getattr(self.config, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f'{field_name} must be finite and non-negative')
        if (
                not math.isfinite(
                    self.config.counterfactual_return_scale)
                or self.config.counterfactual_return_scale <= 0.0):
            raise ValueError(
                'counterfactual_return_scale must be finite and positive'
            )
        if (
                not math.isfinite(
                    self.config.counterfactual_target_clip)
                or self.config.counterfactual_target_clip <= 0.0):
            raise ValueError(
                'counterfactual_target_clip must be finite and positive'
            )

    def _validate_transition_horizons(self, transitions):
        """拒绝超过 trainer 配置 horizon 的 replay 样本。

        rollout accumulator 可以在 episode 尾部生成更短样本，但不能把例如 5-step
        return 悄悄交给配置为 3-step 的 trainer；否则 gamma 指数和实验配置记录会
        失去一致性。
        """

        for transition in transitions:
            if transition.bootstrap_steps > self.config.n_step:
                raise ValueError(
                    'transition bootstrap_steps exceeds trainer n_step: '
                    f'{transition.bootstrap_steps} > '
                    f'{self.config.n_step}'
                )
