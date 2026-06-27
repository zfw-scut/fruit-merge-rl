# 项目文件索引

最后更新：2026-06-27

## 项目定位

本项目是一个用 pygame 和 pymunk 重写的《合成大西瓜》游戏，并将游戏封装为适合强化学习训练的环境。训练部分提供 Keras、PyTorch、Paddle/PARL 三套 DQN 脚本。

## 核心源码

| 路径 | 作用 | 主要入口或可复用点 |
| --- | --- | --- |
| `Main.py` | 手动游戏入口。负责正式游戏渲染、鼠标跟随投放、预览线、下一个水果、HUD、粒子、飘字、震动和音效反馈。 | `Board.next_frame()`、`Board.run()`、`on_fruit_merged()` |
| `Game.py` | 游戏公共基类。负责窗口、物理世界、墙体、碰撞合成、计分、失败检测，并向表现层暴露合成事件钩子。 | `GameBoard`、`create_ball()`、`setup_collision_handler()`、`check_fail()` |
| `Fruit.py` | 水果类型和贴图定义。维护 1 到 11 级水果的半径、类型编号和图片加载缓存。 | `create_fruit(type, x, y)`、`load_fruit_image()`、各水果类 |
| `State.py` | 强化学习环境封装。把游戏动作离散成 16 个落点，并返回画面、分数、奖励、存活状态。 | `AI_Board.next(action)`、`decode_action()` |

## 训练脚本

| 路径 | 作用 | 主要入口或可复用点 |
| --- | --- | --- |
| `train_keras.py` | Keras 版本 DQN 训练脚本。直接使用 RGB 截图作为观测输入。 | `build_network()`、`train_network()` |
| `train_torch.py` | PyTorch 版本 DQN 训练脚本。使用灰度图和 4 帧堆叠作为状态。 | `DeepNetWork`、`BrainDQNMain` |
| `train_paddle.py` | Paddle/PARL 版本 DQN 训练脚本。包含 agent、replay memory、训练循环。 | `Model`、`Agent`、`ReplayMemory`、`run_episode()` |
| `resnet.py` | Paddle 版本使用的 ResNet 风格网络结构。 | `DistResNet.infer()` |

## 资源和说明

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `res.zip` | 水果图片资源压缩包。 | 应保留在仓库中。 |
| `res/` | 解压后的水果图片目录，包含 `01.png` 到 `11.png`。 | 由 `unzip res.zip` 生成，已在 `.gitignore` 中忽略。 |
| `README.md` | 原项目说明，包含运行游戏和训练模型的基础命令。 | 依赖描述不完整。 |
| `requirements.txt` | 原项目依赖文件。 | 当前只写了 `paddlepaddle`，没有覆盖游戏运行依赖。 |
| `LICENSE` | 开源许可证。 | Apache 2.0。 |

## 本地和生成目录

| 路径 | 作用 | 处理建议 |
| --- | --- | --- |
| `.git/` | Git 仓库元数据。 | 不手动修改。 |
| `.vscode/` | VS Code 本地配置和缓存。 | 已忽略。 |
| `__pycache__/` | Python 字节码缓存。 | 已忽略。 |
| `.agents/`、`.codex/` | 当前工作环境辅助目录。 | 不属于原项目核心源码。 |

## 可复用组件

- `GameBoard`：后续优化游戏或重新设计 AI 环境时可复用的物理和合成基类。
- `create_fruit(type, x, y)`：统一创建水果对象，避免外部直接依赖具体水果类。
- `load_fruit_image(path, size)`：缓存水果贴图加载和缩放结果，避免重复磁盘读取。
- `create_ball(space, x, y, m, r, i)`：统一创建 pymunk 圆形刚体。
- `setup_collision_handler()`：水果合成逻辑所在位置，已兼容新版 `pymunk.Space.on_collision`，并在合成后调用可选的 `on_fruit_merged()`。
- `AI_Board.next(action)`：训练脚本与游戏环境交互的主要接口。

## 已知注意事项

- 游戏运行依赖至少包括 `pygame` 和 `pymunk`，但 `requirements.txt` 尚未完整声明。
- 当前 `Game.py` 已为 `pymunk 7.3.0` 做兼容处理。
- 当前手动游戏入口已经偏向正式表现层；RL 训练环境后续计划重写，不再要求与 `Main.py` 渲染保持一致。
- `train_torch.py` 中网络输出维度和环境动作数存在不一致风险，需要后续单独检查。
- `train_paddle.py` 的 `evaluate()` 仍引用了当前 `AI_Board` 不存在的接口，需要后续单独检查。
