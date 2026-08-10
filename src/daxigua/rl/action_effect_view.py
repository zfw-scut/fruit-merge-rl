"""把辅助动作效果预测转换为场景实验室可读的数据。"""

from __future__ import annotations

import math

import torch

from .model import ActionEffectPredictions


CONTACT_NAMES = ('floor', 'left_wall', 'right_wall', 'fruit')
PRIMARY_CONTACT_NAMES = ('none', *CONTACT_NAMES)


def _round(value, digits=4):
    return round(float(value), digits)


def _probability(logit):
    return _round(torch.sigmoid(logit).item(), 5)


def _categorical(logits, labels=None):
    probabilities = torch.softmax(logits.float(), dim=-1)
    index = int(probabilities.argmax().item())
    return {
        'index': index,
        'label': labels[index] if labels is not None else index,
        'confidence': _round(probabilities[index].item(), 5),
        'probabilities': [_round(value, 5) for value in probabilities.tolist()],
    }


def _position(values, board_width, board_height):
    return {
        'x': _round((float(values[0]) + 1.0) * board_width / 2.0, 3),
        'y': _round((float(values[1]) + 1.0) * board_height / 2.0, 3),
    }


def serialize_action_effect_predictions(
        predictions,
        *,
        board_width=560.0,
        board_height=1120.0,
        gravity_y=1800.0,
        max_physics_frames=720,
        physics_fps=120.0):
    """解码一个 batch（通常为1）中全部动作的预测。

    返回值保留概率与类别分布，便于设置页按需显示；坐标和速度恢复到
    模拟器单位，画布无需理解训练标签的归一化方式。
    """

    if predictions is None:
        return None
    if not isinstance(predictions, ActionEffectPredictions):
        raise TypeError('predictions must be ActionEffectPredictions')
    if predictions.merge_logit.ndim != 2:
        raise ValueError('predictions must include batch and action dimensions')
    if predictions.merge_logit.shape[0] != 1:
        raise ValueError('scenario view expects a single scene batch')

    values = type(predictions)(*(
        value[0].detach().cpu() for value in predictions
    ))
    velocity_scale = math.sqrt(2.0 * float(gravity_y) * float(board_height))
    actions = []
    for action in range(values.merge_logit.shape[0]):
        primary = _categorical(
            values.contact_primary_type_logits[action],
            PRIMARY_CONTACT_NAMES,
        )
        level_delta = _categorical(
            values.contact_level_delta_logits[action]
        )
        level_delta['value'] = level_delta['index'] - 10
        contact_types = {
            name: _probability(values.contact_type_logits[action, index])
            for index, name in enumerate(CONTACT_NAMES)
        }
        generations = []
        for rank in range(3):
            generations.append({
                'rank': rank + 1,
                'exists_probability': _probability(
                    values.generation_exists_logits[action, rank]
                ),
                'position': _position(
                    values.generation_position[action, rank],
                    board_width,
                    board_height,
                ),
                'level': _categorical(
                    values.generation_level_logits[action, rank]
                ),
            })
        final_state = values.final_state[action]
        actions.append({
            'action': action,
            'merge': {
                'probability': _probability(values.merge_logit[action]),
                'count': _categorical(values.merge_count_logits[action]),
            },
            'q0': {
                'participated_probability': _probability(
                    values.q0_participated_logit[action]
                ),
                'lineage_depth': _categorical(
                    values.q0_lineage_depth_logits[action]
                ),
                'final_level': _categorical(
                    values.q0_final_level_logits[action]
                ),
                'final_exists_probability': _probability(
                    values.final_exists_logit[action]
                ),
                'final': {
                    **_position(final_state[0:2], board_width, board_height),
                    'vx': _round(
                        torch.atanh(final_state[2].clamp(-0.999, 0.999)).item()
                        * velocity_scale,
                        3,
                    ),
                    'vy': _round(
                        torch.atanh(final_state[3].clamp(-0.999, 0.999)).item()
                        * velocity_scale,
                        3,
                    ),
                    'angular_velocity': _round(
                        torch.atanh(final_state[4].clamp(-0.999, 0.999)).item()
                        * 10.0,
                        3,
                    ),
                },
            },
            'first_contact': {
                'types': contact_types,
                'primary': primary,
                'position': _position(
                    values.contact_position[action], board_width, board_height
                ),
                'level_delta': level_delta,
                'normal': {
                    'x': _round(values.contact_normal[action, 0]),
                    'y': _round(values.contact_normal[action, 1]),
                },
                'age_seconds': _round(
                    math.expm1(
                        float(values.contact_age[action].clamp(0.0, 2.0))
                        * math.log(21.0)
                    ),
                    4,
                ),
                'normal_speed': _round(
                    torch.atanh(values.contact_normal_speed[action].clamp(
                        -0.999, 0.999
                    )).item()
                    * velocity_scale,
                    3,
                ),
            },
            'generations': generations,
            'outcome': {
                'score_delta': _round(
                    math.expm1(
                        float(values.score_delta[action].clamp(0.0, 2.0))
                        * math.log(67.0)
                    ),
                    3,
                ),
                'fruit_count_delta': _round(
                    float(values.fruit_count_delta[action]) * 8.0, 3
                ),
                'stable_probability': _probability(
                    values.stable_logit[action]
                ),
                'settle_timeout_probability': _probability(
                    values.settle_timeout_logit[action]
                ),
                'terminal_probability': _probability(
                    values.terminal_logit[action]
                ),
                'settle_duration_seconds': _round(
                    float(values.settle_duration[action])
                    * float(max_physics_frames) / max(float(physics_fps), 1.0),
                    4,
                ),
                'danger_delta': _round(values.danger_delta[action]),
                'over_danger_line_probability': _probability(
                    values.over_danger_line_logit[action]
                ),
            },
        })
    return actions


__all__ = ['serialize_action_effect_predictions']
