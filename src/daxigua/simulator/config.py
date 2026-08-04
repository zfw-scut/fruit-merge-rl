"""CUDA 批量物理模拟器的显式配置。"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SimulatorConfig:
    """会影响批量游戏演化的全部配置。

    默认几何和时间参数对齐历史 Pymunk 环境。求解器本身是并行
    Jacobi 冲量法，因此 ``solver_iterations`` 不与 Pymunk 的顺序迭代次数
   76f4接等价，需要通过行为和分布对照校准。
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

    gravity_y: float = 1800.0
    damping: float = 0.995
    fruit_elasticity: float = 0.18
    fruit_friction: float = 0.88
    wall_friction: float = 0.60

    stable_velocity_epsilon: float = 35.0
    stable_angular_velocity_epsilon: float = 4.0
    kinematic_rest_frames: int = 4
    kinematic_rest_displacement_epsilon: float = 0.10
    danger_seconds: float = 2.0
    contact_slop: float = 0.05
    position_correction: float = 0.75
    merge_tolerance: float = 0.25
    sync_interval_frames: int = 8
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
        if not isinstance(self.use_cuda_extension, bool):
            raise TypeError('use_cuda_extension must be bool')
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
            'danger_seconds',
            'contact_slop',
            'merge_tolerance',
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f'{name} must be non-negative')

    @property
    def dt(self):
        return 1.0 / self.physics_fps

    @property
    def danger_frame_limit(self):
        return int(self.physics_fps * self.danger_seconds)
