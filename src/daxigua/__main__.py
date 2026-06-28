"""支持 `python -m daxigua` 的包入口。

注意：如果没有安装项目包，直接从仓库根目录执行 `python -m daxigua` 时通常需要
先把 `src/` 加入 `PYTHONPATH`。普通用户推荐继续使用根目录的 `python Main.py`。
"""

from .app import main


if __name__ == '__main__':
    # 和根目录入口保持一致，最终都进入 `daxigua.app.main()`。
    main()
