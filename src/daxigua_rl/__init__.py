"""RL package for automation and training.

The game package must not import this package. Automation code should depend on
stable interfaces exposed by daxigua, never the other way around.

顶层公开 API 继续保持不变，但改为按需导入。这样 Android 只使用
``StateAnalyzer`` / ``GraphBuilder`` 时，不会因为包初始化而加载 Pymunk 或
PyTorch；桌面训练代码仍可继续使用 ``from daxigua_rl import DaxiguaEnv``。
"""


__all__ = [
    'DaxiguaEnv',
    'DaxiguaEnvConfig',
    'FeatureAblationConfig',
    'FeatureMask',
    'GraphAblator',
    'GraphBuilder',
    'GraphBuilderConfig',
    'GraphData',
    'ReplayBuffer',
    'RewardBreakdown',
    'RewardConfig',
    'compute_reward',
    'compute_state_potential',
    'merge_utility',
    'get_ablation_preset',
]


def __getattr__(name):
    """解析顶层兼容导出，并把重依赖留到调用方确实需要时再加载。"""

    if name in {'DaxiguaEnv', 'DaxiguaEnvConfig'}:
        from .env import DaxiguaEnv, DaxiguaEnvConfig

        exports = {
            'DaxiguaEnv': DaxiguaEnv,
            'DaxiguaEnvConfig': DaxiguaEnvConfig,
        }
    elif name in {
            'RewardBreakdown',
            'RewardConfig',
            'compute_reward',
            'compute_state_potential',
            'merge_utility'}:
        from .reward import (
            RewardBreakdown,
            RewardConfig,
            compute_reward,
            compute_state_potential,
            merge_utility,
        )

        exports = {
            'RewardBreakdown': RewardBreakdown,
            'RewardConfig': RewardConfig,
            'compute_reward': compute_reward,
            'compute_state_potential': compute_state_potential,
            'merge_utility': merge_utility,
        }
    elif name in {
            'FeatureAblationConfig',
            'FeatureMask',
            'GraphAblator',
            'GraphBuilder',
            'GraphBuilderConfig',
            'GraphData',
            'get_ablation_preset'}:
        from .graph import (
            FeatureAblationConfig,
            FeatureMask,
            GraphAblator,
            GraphBuilder,
            GraphBuilderConfig,
            GraphData,
            get_ablation_preset,
        )

        exports = {
            'FeatureAblationConfig': FeatureAblationConfig,
            'FeatureMask': FeatureMask,
            'GraphAblator': GraphAblator,
            'GraphBuilder': GraphBuilder,
            'GraphBuilderConfig': GraphBuilderConfig,
            'GraphData': GraphData,
            'get_ablation_preset': get_ablation_preset,
        }
    elif name == 'ReplayBuffer':
        from .training import ReplayBuffer

        exports = {'ReplayBuffer': ReplayBuffer}
    else:
        raise AttributeError(
            f'module {__name__!r} has no attribute {name!r}'
        )

    # 缓存同组对象，使后续顶层导入没有额外分派成本。
    globals().update(exports)
    return exports[name]
