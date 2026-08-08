"""CUDA 批量物理模拟器的显式配置。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    """会影响批量游戏演化的全部配置。

    默认几何和时间参数对齐历史 Pymunk 环境。求解器本身是并行
    Jacobi 冲量法，因此 ``solver_iterations`` 不与 Pymunk 的顺序迭代次数
    直接等价，需要通过行为和分布对照校准。
    """

    board_width: int = 560
    board_height: int = 1120
    spawn_y: int = 252
    wall_width: int = 20
    action_count: int = 21
    max_fruits: int = 64
    queue_length: int = 4

    physics_fps: int = 120
    max_physics_frames: int = 720
    stable_frames: int = 15
    solver_iterations: int = 4
    drop_fast_forward: bool = False
    adaptive_collision_substeps: bool = False
    max_collision_substeps: int = 4

    gravity_y: float = 1800.0
    damping: float = 0.995
    fruit_elasticity: float = 0.18
    restitution_velocity_threshold: float = 35.0
    fruit_friction: float = 0.88
    wall_friction: float = 0.60

    stable_velocity_epsilon: float = 35.0
    stable_angular_velocity_epsilon: float = 4.0
    kinematic_rest_frames: int = 4
    # 兼容旧配置名：该数值定义在 120 FPS 参考时间步上，运行时会先
    # 换算为速度，再乘当前 dt，避免低帧率反而更难进入静止修正。
    kinematic_rest_displacement_epsilon: float = 0.10
    collision_substep_motion_fraction: float = 0.25
    collision_substep_penetration_threshold: float = 1.0
    danger_seconds: float = 2.0
    contact_slop: float = 0.05
    position_correction: float = 0.75
    merge_tolerance: float = 0.25
    sync_interval_frames: int = 8
    track_action_effects: bool = False
    use_cuda_extension: bool = True
    cuda_threads_per_block: int = 128

    def __post_init__(self):
        integer_fields = (
            'board_width',
            'board_height',
            'spawn_y',
            'wall_width',
            'action_count',
            'max_fruits',
            'queue_length',
            'physics_fps',
            'max_physics_frames',
            'stable_frames',
            'solver_iterations',
            'max_collision_substeps',
            'kinematic_rest_frames',
            'sync_interval_frames',
            'cuda_threads_per_block',
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'{name} must be an integer')

        positive_fields = (
            'board_width',
            'board_height',
            'wall_width',
            'action_count',
            'max_fruits',
            'queue_length',
            'physics_fps',
            'max_physics_frames',
            'stable_frames',
            'solver_iterations',
            'max_collision_substeps',
            'sync_interval_frames',
            'cuda_threads_per_block',
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')

        if self.max_fruits < 2:
            raise ValueError('max_fruits must be at least 2')
        if self.queue_length != 4:
            raise ValueError('queue_length must match FRUIT_QUEUE_LENGTH (4)')
        if not 0 <= self.spawn_y < self.board_height:
            raise ValueError('spawn_y must be inside the board')
        if self.wall_width * 2 >= self.board_width:
            raise ValueError('wall_width leaves no playable board width')
        if self.action_count < 2:
            raise ValueError('action_count must be at least 2')
        if self.kinematic_rest_frames < 0:
            raise ValueError('kinematic_rest_frames must be non-negative')
        if self.kinematic_rest_frames > 255:
            raise ValueError('kinematic_rest_frames must be <= 255')
        if not 1 <= self.max_collision_substeps <= 4:
            raise ValueError('max_collision_substeps must be in [1, 4]')
        for name in (
            'drop_fast_forward',
            'adaptive_collision_substeps',
            'track_action_effects',
            'use_cuda_extension',
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'{name} must be bool')
        if self.cuda_threads_per_block > 1024:
            raise ValueError('cuda_threads_per_block must be <= 1024')

        bounded_fields = (
            ('damping', self.damping, 0.0, 1.0),
            ('fruit_elasticity', self.fruit_elasticity, 0.0, 1.0),
            ('fruit_friction', self.fruit_friction, 0.0, None),
            ('wall_friction', self.wall_friction, 0.0, None),
            ('position_correction', self.position_correction, 0.0, 1.0),
        )
        for name, value, minimum, maximum in bounded_fields:
            value = float(value)
            if value < minimum or (maximum is not None and value > maximum):
                raise ValueError(f'{name} is outside its valid range')

        for name in (
            'gravity_y',
            'stable_velocity_epsilon',
            'stable_angular_velocity_epsilon',
            'kinematic_rest_displacement_epsilon',
            'restitution_velocity_threshold',
            'collision_substep_motion_fraction',
            'collision_substep_penetration_threshold',
            'danger_seconds',
            'contact_slop',
            'merge_tolerance',
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f'{name} must be non-negative')
        for name in (
            'collision_substep_motion_fraction',
            'collision_substep_penetration_threshold',
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f'{name} must be positive')

    @property
    def dt(self):
        return 1.0 / self.physics_fps

    @property
    def danger_frame_limit(self):
        return int(self.physics_fps * self.danger_seconds)

    @property
    def kinematic_rest_speed_epsilon(self):
        """返回与物理帧率无关的静止修正速度阈值。"""

        return self.kinematic_rest_displacement_epsilon * 120.0

    @classmethod
    def training_fast(cls, **overrides):
        """返回面向最大吞吐的大批量训练 30 FPS 配置。

        物理等待和稳定时间按秒近似保持默认语义；精确的 120 FPS 配置
        仍是默认值。30 FPS 会改变长期局长和得分分布，调用方应显式选择。
        """

        values = {
            'physics_fps': 30,
            'max_physics_frames': 180,
            'stable_frames': 4,
            'drop_fast_forward': True,
            'adaptive_collision_substeps': True,
            'max_collision_substeps': 2,
            'kinematic_rest_frames': 1,
            'position_correction': 0.90,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def high_fidelity_fast(cls, **overrides):
        """返回保留 120 FPS 离散语义、仅跳过自由下落的加速配置。"""

        values = {'drop_fast_forward': True}
        values.update(overrides)
        return cls(**values)
