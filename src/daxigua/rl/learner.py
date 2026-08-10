"""1-step Dueling Double DQN learner。"""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy

import torch
from torch.nn import functional as F

from .config import DqnConfig
from .observations import TensorState


def _build_adam(parameters, config, device):
    options = {
        'lr': config.learning_rate,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.0,
    }
    if config.fused_adam and torch.device(device).type == 'cuda':
        try:
            return torch.optim.Adam(parameters, fused=True, **options), True
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(parameters, **options), False


def _masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _chosen_action(values, actions):
    if values is None:
        return None
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, actions]


def _chosen_candidate(values, candidates):
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, candidates.to(torch.long)]


def _binary_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(
        logits.float(), targets.to(torch.float32)
    )


def _categorical_loss(logits, targets, mask=None):
    values = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.to(torch.long).reshape(-1),
        reduction='none',
    ).reshape(targets.shape)
    return values.mean() if mask is None else _masked_mean(values, mask)


def _regression_loss(predictions, targets, mask=None):
    values = F.smooth_l1_loss(
        predictions.float(), targets.float(), reduction='none'
    )
    if values.ndim > targets.ndim:
        values = values.mean(dim=tuple(range(targets.ndim, values.ndim)))
    if mask is not None:
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        return _masked_mean(values, mask.expand_as(values))
    return values.mean()


class DqnLearner:
    def __init__(self, online_model, config=None):
        self.config = config or DqnConfig()
        if not isinstance(self.config, DqnConfig):
            raise TypeError('config must be DqnConfig')
        target_model = deepcopy(online_model).eval()
        self.online_module = online_model
        self.target_module = target_model
        self.uses_torch_compile = False
        if self.config.compile_model:
            if not hasattr(torch, 'compile'):
                raise RuntimeError('torch.compile is unavailable')
            online_model = torch.compile(
                online_model,
                mode=self.config.compile_mode,
                dynamic=True,
            )
            target_model = torch.compile(
                target_model,
                mode=self.config.compile_mode,
                dynamic=True,
            )
            self.uses_torch_compile = True
        self.online_model = online_model
        self.target_model = target_model
        self.target_module.requires_grad_(False)
        self.device = next(online_model.parameters()).device
        self.optimizer, self.uses_fused_adam = _build_adam(
            self.online_model.parameters(), self.config, self.device
        )
        self.update_count = 0

    def _autocast(self):
        if self.device.type == 'cuda' and self.config.use_bfloat16:
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        return nullcontext()

    def _action_effect_loss(self, predictions, targets, actions):
        if predictions is None or targets is None:
            zero = next(self.online_module.parameters()).sum() * 0.0
            return zero, {}
        prediction = type(predictions)(*(
            _chosen_action(value, actions) for value in predictions
        ))
        contact_valid = targets.contact_primary_type > 0
        fruit_contact = targets.contact_primary_type == 4
        generated = targets.generation_exists

        merge = (
            _binary_loss(prediction.merge_logit, targets.merge_happened)
            + _categorical_loss(
                prediction.merge_count_logits, targets.merge_count
            )
        ) / 2.0
        q0 = (
            _binary_loss(
                prediction.q0_participated_logit, targets.q0_participated
            )
            + _categorical_loss(
                prediction.q0_lineage_depth_logits,
                targets.q0_lineage_depth,
            )
            + _categorical_loss(
                prediction.q0_final_level_logits, targets.q0_final_level
            )
        ) / 3.0
        if prediction.contact_target_logits is None:
            contact_identity = _categorical_loss(
                prediction.contact_primary_type_logits,
                targets.contact_primary_type,
            )
            contact_location = _regression_loss(
                prediction.contact_position,
                targets.contact_position,
                contact_valid,
            )
        else:
            contact_identity = _categorical_loss(
                prediction.contact_target_logits,
                targets.contact_target,
            )
            contact_location = _regression_loss(
                _chosen_candidate(
                    prediction.contact_position_residual,
                    targets.contact_target,
                ),
                targets.contact_position_residual,
                contact_valid,
            )
        contact = (
            _binary_loss(
                prediction.contact_type_logits, targets.contact_type_bits
            )
            + contact_identity
            + contact_location
            + _categorical_loss(
                prediction.contact_level_delta_logits,
                targets.contact_level_delta,
                fruit_contact,
            )
            + _regression_loss(
                prediction.contact_normal,
                targets.contact_normal,
                contact_valid,
            )
            + _regression_loss(
                prediction.contact_age, targets.contact_age, contact_valid
            )
            + _regression_loss(
                prediction.contact_normal_speed,
                targets.contact_normal_speed,
                contact_valid,
            )
        ) / 7.0
        generation = (
            _binary_loss(
                prediction.generation_exists_logits,
                targets.generation_exists,
            )
            + _regression_loss(
                prediction.generation_position,
                targets.generation_position,
                generated,
            )
            + _categorical_loss(
                prediction.generation_level_logits,
                targets.generation_level,
                generated,
            )
        ) / 3.0
        final_valid = targets.final_exists
        outcome = (
            _regression_loss(prediction.score_delta, targets.score_delta)
            + _regression_loss(
                prediction.fruit_count_delta, targets.fruit_count_delta
            )
            + _binary_loss(
                prediction.final_exists_logit, targets.final_exists
            )
            + _regression_loss(
                prediction.final_state, targets.final_state, final_valid
            )
            + _binary_loss(prediction.stable_logit, targets.stable)
            + _binary_loss(
                prediction.settle_timeout_logit, targets.settle_timeout
            )
            + _binary_loss(prediction.terminal_logit, targets.terminal)
            + _regression_loss(
                prediction.settle_duration, targets.settle_duration
            )
            + _regression_loss(
                prediction.danger_delta, targets.danger_delta
            )
            + _binary_loss(
                prediction.over_danger_line_logit,
                targets.over_danger_line,
            )
        ) / 10.0
        total = (merge + q0 + contact + generation + outcome) / 5.0
        return total, {
            'aux_loss_merge': merge.detach(),
            'aux_loss_q0_lineage': q0.detach(),
            'aux_loss_first_contact': contact.detach(),
            'aux_loss_generation': generation.detach(),
            'aux_loss_outcome': outcome.detach(),
            'aux_loss_total': total.detach(),
        }

    def update(
            self,
            replay,
            batch_size,
            *,
            branch_replay=None,
            branch_batch_size=0,
            branch_loss_weight=0.0):
        batch = replay.sample(batch_size)
        branch_batch_size = int(branch_batch_size)
        use_branch = branch_replay is not None and branch_batch_size > 0
        branch = (
            branch_replay.sample(branch_batch_size) if use_branch else None
        )
        current_states = [batch.current]
        next_states = [batch.next_state]
        if branch is not None:
            current_states.append(branch.current)
            next_states.append(branch.next_state)
        current = TensorState.cat(current_states)
        next_state = TensorState.cat(next_states)
        combined = TensorState.cat((current, next_state))
        total_batch_size = current.batch_size
        self.online_model.train()
        with self._autocast():
            combined_output = self.online_model(
                combined, True, True, total_batch_size
            )
            current_heads, next_online_heads = (
                combined_output.head_q_values.split(total_batch_size, dim=0)
            )
            actions = (
                batch.action
                if branch is None
                else torch.cat((batch.action, branch.action), dim=0)
            )
            chosen_heads = current_heads.gather(
                2,
                actions[:, None, None].expand(
                    -1, current_heads.shape[1], 1
                ),
            ).squeeze(2)
            with torch.no_grad():
                next_actions = next_online_heads.argmax(dim=2)
                next_target_heads = self.target_model(
                    next_state, True, False
                ).head_q_values
                next_target_q = next_target_heads.gather(
                    2, next_actions.unsqueeze(2)
                ).squeeze(2)
                rewards = (
                    batch.reward
                    if branch is None
                    else torch.cat((batch.reward, branch.reward), dim=0)
                )
                terminals = (
                    batch.terminal
                    if branch is None
                    else torch.cat((batch.terminal, branch.terminal), dim=0)
                )
                target = rewards.unsqueeze(1) + (
                    ~terminals
                ).to(rewards.dtype).unsqueeze(1) * (
                    self.config.gamma * next_target_q.float()
                )
            td_values = F.huber_loss(
                chosen_heads.float(),
                target,
                delta=self.config.huber_delta,
                reduction='none',
            )
            bootstrap_mask = batch.bootstrap_mask
            if bootstrap_mask is None:
                bootstrap_mask = torch.ones_like(
                    chosen_heads[:batch_size], dtype=torch.bool
                )
            branch_bootstrap_mask = None
            if branch is not None:
                branch_bootstrap_mask = branch.bootstrap_mask
                if branch_bootstrap_mask is None:
                    branch_bootstrap_mask = torch.ones_like(
                        chosen_heads[batch_size:], dtype=torch.bool
                    )
            dqn_loss = _masked_mean(
                td_values[:batch_size], bootstrap_mask
            )
            action_effect_predictions = combined_output.action_effects
            parent_predictions = (
                None
                if action_effect_predictions is None
                else type(action_effect_predictions)(*(
                    None if value is None else value[:batch_size]
                    for value in action_effect_predictions
                ))
            )
            auxiliary_loss, auxiliary_metrics = self._action_effect_loss(
                parent_predictions,
                batch.action_effects,
                batch.action,
            )
            parent_loss = dqn_loss + (
                self.config.auxiliary_loss_weight * auxiliary_loss
            )
            branch_dqn_loss = parent_loss.new_zeros(())
            branch_auxiliary_loss = parent_loss.new_zeros(())
            branch_auxiliary_metrics = {}
            if branch is not None:
                branch_dqn_loss = _masked_mean(
                    td_values[batch_size:], branch_bootstrap_mask
                )
                branch_predictions = (
                    None
                    if action_effect_predictions is None
                    else type(action_effect_predictions)(*(
                        None if value is None else
                        value[batch_size:total_batch_size]
                        for value in action_effect_predictions
                    ))
                )
                (
                    branch_auxiliary_loss,
                    branch_auxiliary_metrics,
                ) = self._action_effect_loss(
                    branch_predictions,
                    branch.action_effects,
                    branch.action,
                )
            branch_loss = branch_dqn_loss + (
                self.config.auxiliary_loss_weight * branch_auxiliary_loss
            )
            loss = parent_loss + float(branch_loss_weight) * branch_loss

        if self.device.type == 'cuda':
            torch._assert_async(torch.isfinite(loss), 'non-finite DQN loss')
        elif not bool(torch.isfinite(loss).item()):
            raise FloatingPointError('non-finite DQN loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.online_model.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()
        self.update_count += 1
        target_synced = False
        if self.update_count % self.config.target_update_interval == 0:
            self.target_module.load_state_dict(self.online_module.state_dict())
            target_synced = True

        td_error = (
            target[:batch_size].detach()
            - chosen_heads[:batch_size].detach().float()
        )
        centered_heads = current_heads[:batch_size] - current_heads[
            :batch_size
        ].mean(
            dim=2, keepdim=True
        )
        result = {
            'loss': loss.detach(),
            'dqn_loss': dqn_loss.detach(),
            'mean_q': chosen_heads[:batch_size].detach().float().mean(),
            'mean_target': target[:batch_size].detach().mean(),
            'mean_reward': batch.reward.detach().mean(),
            'mean_abs_td_error': td_error.abs().mean(),
            'max_abs_td_error': td_error.abs().amax(),
            'policy_disagreement': centered_heads.float().std(
                dim=1, correction=0
            ).mean().detach(),
            'bootstrap_active_fraction': bootstrap_mask.float().mean().detach(),
            'grad_norm': torch.as_tensor(grad_norm).detach().float(),
            'target_synced': target_synced,
            'update_count': self.update_count,
            'branch_sample_fraction': dqn_loss.new_tensor(
                0.0
                if branch is None
                else branch_batch_size / total_batch_size
            ),
        }
        result.update(auxiliary_metrics)
        if branch is not None:
            branch_td_error = (
                target[batch_size:].detach()
                - chosen_heads[batch_size:].detach().float()
            )
            branch_centered = current_heads[batch_size:] - current_heads[
                batch_size:
            ].mean(dim=2, keepdim=True)
            result.update({
                'branch_loss': branch_loss.detach(),
                'branch_dqn_loss': branch_dqn_loss.detach(),
                'branch_mean_reward': branch.reward.detach().mean(),
                'branch_mean_abs_td_error': branch_td_error.abs().mean(),
                'branch_policy_disagreement': branch_centered.float().std(
                    dim=1, correction=0
                ).mean().detach(),
            })
            result.update({
                f'branch_{name}': value
                for name, value in branch_auxiliary_metrics.items()
            })
        return result

    def state_dict(self):
        return {
            # 始终保存未包装模块，保证 eager/torch.compile checkpoint 互通。
            'online_model': self.online_module.state_dict(),
            'target_model': self.target_module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'uses_fused_adam': self.uses_fused_adam,
            'uses_torch_compile': self.uses_torch_compile,
        }

    def load_state_dict(self, state):
        self.online_module.load_state_dict(state['online_model'])
        self.target_module.load_state_dict(state['target_model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.update_count = int(state['update_count'])
