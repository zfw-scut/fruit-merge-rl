# 合成大西瓜 pygame 版

这是一个基于 `pygame` 和 `pymunk` 的《合成大西瓜》桌面小游戏。

当前项目只保留游戏本体。旧实验代码和旧环境封装已经移除，后续如果需要自动游玩能力，会重新设计相关模块。

## 运行

先安装依赖：

```bash
python -m pip install -r requirements.txt
```

启动游戏：

```bash
python Main.py
```

## 操作

- 鼠标移动：调整当前水果的投放位置。
- 鼠标左键：投放水果。
- `A` / `Left`：向左调整投放位置。
- `D` / `Right`：向右调整投放位置。
- `Space` / `Enter`：投放水果。
- `R`：重新开始。
- `Esc`：退出。
- 拖动窗口边框：改变游戏场地大小。

## 项目说明

- `Main.py`: 兼容旧启动方式的薄入口。
- `src/daxigua/`: 游戏本体包。
- `src/daxigua/app.py`: 游戏应用入口和当前表现层实现。
- `src/daxigua/core/`: 游戏核心逻辑，负责物理世界、边界、碰撞合成、计分和水果定义。
- `src/daxigua_rl/`: 后续自动游玩/RL 相关代码预留包，不会被游戏本体反向依赖。
- `assets/fruits/`: 水果图片资源。
- `assets/fruits.zip`: 原始水果图片压缩包归档。
- `docs/`: 项目文档和 Codex 修改记录。
