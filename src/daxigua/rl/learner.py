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

    def update(self, replay, batch_size):
        batch = replay.sample(batch_size)
        combined = TensorState.cat((batch.current, batch.next_state))
        self.online_model.train()
        with self._autocast():
            combined_q = self.online_model(combined)
            current_q, next_online_q = combined_q.split(batch_size, dim=0)
            chosen_q = current_q.gather(
                1, batch.action.unsqueeze(1)
            ).squeeze(1)
            with torch.no_grad():
                next_actions = next_online_q.argmax(dim=1)
                next_target_q = self.target_model(batch.next_state).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
                target = batch.reward + (
                    ~batch.terminal
                ).to(batch.reward.dtype) * self.config.gamma * next_target_q.float()
            loss = F.huber_loss(
                chosen_q.float(),
                target,
                delta=self.config.huber_delta,
            )

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

        td_error = target.detach() - chosen_q.detach().float()
        return {
            'loss': loss.detach(),
            'mean_q': chosen_q.detach().float().mean(),
            'mean_target': target.detach().mean(),
            'mean_reward': batch.reward.detach().mean(),
            'mean_abs_td_error': td_error.abs().mean(),
            'max_abs_td_error': td_error.abs().amax(),
            'grad_norm': torch.as_tensor(grad_norm).detach().float(),
            'target_synced': target_synced,
            'update_count': self.update_count,
        }

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
