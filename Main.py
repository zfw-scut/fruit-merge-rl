"""项目旧入口。

这个文件保留在仓库根目录，是为了让已有使用习惯继续生效：
用户仍然可以通过 `python Main.py` 启动游戏。真正的游戏实现已经移动到
`src/daxigua/` 包内，这里只负责找到源码包并转交控制权。
"""

from pathlib import Path
import sys


# 根目录就是当前 `Main.py` 所在目录。
ROOT_DIR = Path(__file__).resolve().parent

# `src/` 采用 src-layout。直接运行 `python Main.py` 时，Python 默认不会把
# `src/` 放进模块搜索路径，所以这里手动插入一次。
SRC_DIR = ROOT_DIR / 'src'
if str(SRC_DIR) not in sys.path:
    # 插到最前面，优先使用当前工作区里的源码，而不是环境里可能安装过的同名包。
    sys.path.insert(0, str(SRC_DIR))

# 所有实际启动逻辑都放在包内，根入口只做兼容转发。
from daxigua.app import main


if __name__ == '__main__':
    # 只有作为脚本直接运行时才启动游戏；被测试或其他模块 import 时不会自动开窗口。
    main()
