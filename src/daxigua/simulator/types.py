"""批量模拟器使用的 Tensor 数据契约。"""

from dataclasses import dataclass

import torch

from daxigua.core import MergeEvent


@dataclass(frozen=True, slots=True)
class BatchObservation:
    """一批环境的定长状态。

    为避免热路径中的无条件拷贝，默认返回模拟器内部 Tensor 的视图。
    需要跨越下一次 ``step`` 保留时应调用 ``clone()``。
    """

    positions: torch.Tensor
    velocities: torch.Tensor
    angles: torch.Tensor
    angular_velocities: torch.Tensor
    levels: torch.Tensor
    physics_radii: torch.Tensor
    fruit_ids: torch.Tensor
    age_frames: torch.Tensor
    active: torch.Tensor
    fruit_queue: torch.Tensor
    score: torch.Tensor
    last_score: torch.Tensor
    step_count: torch.Tensor
    physics_frame: torch.Tensor
    done: torch.Tensor
    fruit_count: torch.Tensor
    max_level: torch.Tensor
    max_height: torch.Tensor
    empty_space_ratio: torch.Tensor
    danger_progress: torch.Tensor
    over_danger_line: torch.Tensor

    def clone(self):
        """返回与后续环境步进解耦的状态快照。"""

        values = {
            field_name: getattr(self, field_name).clone()
            for field_name in self.__dataclass_fields__
        }
        return type(self)(**values)


@dataclass(frozen=True, slots=True)
class BatchDecisionSidecar:
    """决策边界当前已识别的重放候选附加状态。

    模型状态和主 Replay 已经保存位置、速度、等级、碰撞半径、年龄、活动掩码、
    队列和危险进度。本契约只补充当前已知可能用于未来状态恢复、但不应进入
    模型输入的引擎字段。恢复器实现后仍需通过轨迹一致性测试确定字段集。
    所有 Tensor 的第一维都对应同一批环境。
    """

    angles: torch.Tensor
    fruit_ids: torch.Tensor
    score: torch.Tensor
    last_score: torch.Tensor
    step_count: torch.Tensor
    physics_frame: torch.Tensor
    fail_frames: torch.Tensor
    next_fruit_id: torch.Tensor
    rng_state: torch.Tensor
    episode_count: torch.Tensor
    terminated: torch.Tensor

    @property
    def batch_size(self):
        return int(self.score.shape[0])

    def index_select(self, rows):
        return type(self)(**{
            field_name: getattr(self, field_name).index_select(0, rows)
            for field_name in self.__dataclass_fields__
        })

    def cpu(self):
        return type(self)(**{
            field_name: getattr(self, field_name).detach().cpu()
            for field_name in self.__dataclass_fields__
        })


@dataclass(frozen=True, slots=True)
class BatchMergeEvents:
    """定长的批量合成事件缓冲。"""

    count: torch.Tensor
    source_levels: torch.Tensor
    new_levels: torch.Tensor
    positions: torch.Tensor
    score_deltas: torch.Tensor
    source_ids: torch.Tensor
    new_fruit_ids: torch.Tensor

    def to_python(self, env_index):
        """把一个环境的有效事件转为领域数据类。"""

        event_count = int(self.count[env_index].item())
        events = []
        for event_index in range(event_count):
            new_level = int(self.new_levels[env_index, event_index].item())
            new_fruit_id = int(
                self.new_fruit_ids[env_index, event_index].item()
            )
            events.append(
                MergeEvent(
                    new_level=None if new_level == 0 else new_level,
                    x=float(self.positions[env_index, event_index, 0].item()),
                    y=float(self.positions[env_index, event_index, 1].item()),
                    score_delta=int(
                        self.score_deltas[env_index, event_index].item()
                    ),
                    source_ids=tuple(
                        int(value)
                        for value in self.source_ids[
                            env_index, event_index
                        ].tolist()
                    ),
                    new_fruit_id=(
                        None if new_fruit_id == 0 else new_fruit_id
                    ),
                )
            )
        return tuple(events)


@dataclass(frozen=True, slots=True)
class BatchActionEffectEvents:
    """一次投放中为动作效果监督累计的定长物理事件。"""

    first_contact_type_mask: torch.Tensor
    first_contact_primary_type: torch.Tensor
    first_contact_position: torch.Tensor
    first_contact_level_delta: torch.Tensor
    first_contact_normal: torch.Tensor
    first_contact_age_frames: torch.Tensor
    first_contact_normal_speed: torch.Tensor
    q0_participated: torch.Tensor
    q0_lineage_depth: torch.Tensor
    q0_final_fruit_id: torch.Tensor
    q0_final_level: torch.Tensor
    q0_final_slot: torch.Tensor

    def index_select(self, rows):
        return type(self)(**{
            field_name: getattr(self, field_name).index_select(0, rows)
            for field_name in self.__dataclass_fields__
        })


@dataclass(frozen=True, slots=True)
class BatchDropResult:
    """一批环境的即时投放结果。"""

    dropped_levels: torch.Tensor
    drop_x: torch.Tensor
    fruit_ids: torch.Tensor
    queue_before: torch.Tensor
    queue_after: torch.Tensor


@dataclass(frozen=True, slots=True)
class BatchPhysicsResult:
    """一次批量投放的物理结果。"""

    frames_simulated: torch.Tensor
    stable: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor
    score_delta: torch.Tensor
    merge_events: BatchMergeEvents
    settle_timeout: torch.Tensor | None = None
    fast_forwarded_frames: torch.Tensor | None = None
    collision_substeps: torch.Tensor | None = None
    action_effects: BatchActionEffectEvents | None = None


@dataclass(frozen=True, slots=True)
class BatchStepResult:
    """不包含训练奖励的批量模拟步进结果。"""

    observation: BatchObservation
    drop: BatchDropResult
    physics: BatchPhysicsResult


@dataclass(frozen=True, slots=True)
class BatchSimulationTrace:
    """指定 CUDA 环境的一次投放逐帧记录。

    记录 0 是水果刚投放、尚未推进物理的状态，之后按 ``frame_stride``
    采样，并始终保留最终稳定、终止、等待超时或截断帧。
    """

    env_indices: torch.Tensor
    actions: torch.Tensor
    record_counts: torch.Tensor
    frame_numbers: torch.Tensor
    positions: torch.Tensor
    velocities: torch.Tensor
    angles: torch.Tensor
    angular_velocities: torch.Tensor
    levels: torch.Tensor
    physics_radii: torch.Tensor
    fruit_ids: torch.Tensor
    active: torch.Tensor
    scores: torch.Tensor
    merge_counts: torch.Tensor
    stable: torch.Tensor
    done: torch.Tensor
    truncated: torch.Tensor
    score_deltas: torch.Tensor
    physics_fps: int
    frame_stride: int
    settle_timeout: torch.Tensor | None = None

    def cpu(self):
        """返回可长期保存且与模拟器设备解耦的 CPU 副本。"""

        values = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            values[field_name] = (
                value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
            )
        return type(self)(**values)
