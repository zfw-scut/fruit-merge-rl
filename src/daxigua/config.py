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

# 初始窗口尺寸。游戏支持拖拽窗口边缘动态改变场地大小。
DEFAULT_WINDOW_SIZE = (400, 800)

# 可缩放窗口的最小尺寸，避免把场地压到无法正常操作。
MIN_WINDOW_SIZE = (360, 560)

# 目标帧率。物理世界每帧也按这个频率推进固定步长。
FPS = 120
