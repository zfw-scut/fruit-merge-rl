"""游戏项目的基础配置。

本文件只放路径、窗口和帧率这类“全局但稳定”的配置。这样做的好处是：
游戏逻辑、渲染逻辑和资源加载逻辑不需要各自计算项目根目录，也方便以后
让 RL 接口复用同一套路径配置。
"""

from pathlib import Path


# `config.py` 位于 `src/daxigua/config.py`：
# parents[0] -> src/daxigua
# parents[1] -> src
# parents[2] -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 静态资源根目录。当前只有水果贴图，后续音频、字体也可以继续放在这里。
ASSETS_DIR = PROJECT_ROOT / 'assets'

# 水果贴图目录，要求包含 01.png 到 11.png。
FRUIT_ASSET_DIR = ASSETS_DIR / 'fruits'

# 旧版训练场地只用于解释迁移 checkpoint 的来源，不再作为运行默认值。
LEGACY_WINDOW_SIZE = (400, 800)
LEGACY_SPAWN_LINE_Y = 180

# 当前项目场地长宽均在旧版基础上扩大 40%。水果尺寸与物理参数保持不变，
# 因此这是实际可用空间扩大，而不是把整张画面等比拉伸。
DEFAULT_WINDOW_SIZE = (560, 1120)

# 水果生成线 / 死亡警戒线的 y 坐标。
# 当前顶部有独立信息层展示分数和待投放队列，生成线下移到信息层下方，
# 避免当前悬浮水果与队列预览互相遮挡。
SPAWN_LINE_Y = 252

# 目标帧率。物理世界每帧也按这个频率推进固定步长。
FPS = 120
