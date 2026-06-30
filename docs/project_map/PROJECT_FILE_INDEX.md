# 项目文件索引

最后更新：2026-06-30

## 项目定位

本项目当前是一个基于 `pygame` 和 `pymunk` 的《合成大西瓜》桌面小游戏。当前只保留游戏本体，旧实验代码和旧环境封装已经删除。后续自动游玩/RL 能力会通过 `daxigua_rl` 包和游戏本体暴露的稳定接口接入，不让游戏本体反向依赖自动化代码。

## 核心源码

| 路径 | 作用 | 主要入口或可复用点 |
| --- | --- | --- |
| `Main.py` | 兼容旧启动方式的薄入口，将 `src/` 加入 import 路径后调用 `daxigua.app.main()`。 | `main()` |
| `src/daxigua/app.py` | 游戏应用入口和当前表现层实现。负责固定窗口、输入、正式渲染、鼠标跟随投放、预览线、顶部独立信息层、待投放水果队列、HUD、粒子、飘字、震动和音效反馈。 | `Board.next_frame()`、`Board.run()`、`main()` |
| `src/daxigua/config.py` | 项目路径和基础配置。 | `PROJECT_ROOT`、`FRUIT_ASSET_DIR`、`DEFAULT_WINDOW_SIZE`、`SPAWN_LINE_Y`、`FPS` |
| `src/daxigua/core/board.py` | 游戏公共逻辑。负责 pygame 画布、pymunk 物理世界、动态墙体、碰撞合成、计分、失败检测，并向表现层暴露合成事件钩子。 | `GameBoard`、`resize_world()`、`create_ball()`、`setup_collision_handler()`、`check_fail()` |
| `src/daxigua/core/engine.py` | 无渲染游戏引擎。负责 headless 物理世界、投放、队列推进、动作候选、状态快照和稳定推进，供训练环境调用。 | `HeadlessGame` |
| `src/daxigua/core/fruit.py` | 水果显示精灵和贴图加载。根据等级创建单一 `Fruit` 显示对象，并复用 `rules.py` 中的半径规则。 | `create_fruit(level, x, y)`、`Fruit`、`fruit_image_path()`、`load_fruit_image()` |
| `src/daxigua/core/rules.py` | 纯规则常量和辅助函数。集中维护水果半径、队列长度、随机生成范围、合成分数和物理半径。 | `FRUIT_RADII`、`FRUIT_QUEUE_LENGTH`、`fruit_radius()`、`merge_score()` |
| `src/daxigua/core/state.py` | 训练友好的纯数据状态结构。 | `GameState`、`FruitState`、`ActionCandidate`、`DropResult`、`PhysicsResult` |
| `src/daxigua_rl/` | 自动游玩/RL 相关代码。游戏本体不得 import 它。当前通过 `HeadlessGame` 访问游戏。 | `DaxiguaEnv`、`DaxiguaEnvConfig`、`README.md` 中记录边界规则 |
| `src/daxigua_rl/env.py` | 类 Gymnasium 的 RL 环境壳层。一次 `step(action_index)` 表示一次投放和无渲染物理稳定。 | `DaxiguaEnv.reset()`、`DaxiguaEnv.step()`、`action_candidates()` |
| `src/daxigua_rl/graph/` | GNN 图构建相关代码。负责把游戏状态和动作候选转换成模型输入图，并提供训练实验用的特征消融层。 | `GraphBuilder`、`GraphAblator` |
| `src/daxigua_rl/graph/schema.py` | 框架无关的图数据结构和节点/边特征名。 | `GraphData`、`GraphNodeRef`、`GraphEdgeRef`、`NODE_FEATURE_NAMES`、`EDGE_FEATURE_NAMES` |
| `src/daxigua_rl/graph/builder.py` | 从 `GameState` 和 `ActionCandidate` 构建 GNN 输入图。 | `GraphBuilder.build()` |
| `src/daxigua_rl/graph/ablation.py` | 图特征消融工具。在不改变图维度的前提下按配置置零部分节点或边特征。 | `GraphAblator`、`FeatureAblationConfig`、`FeatureMask`、`ABLATION_PRESETS` |
| `src/daxigua_rl/graph/tensor.py` | PyTorch 张量转换层。把框架无关 `GraphData` 转成模型可直接使用的 `GraphTensor`。 | `graph_to_tensor()`、`GraphTensor` |
| `src/daxigua_rl/models/` | 强化学习模型代码。当前只包含最小 GNN-Q 前向模型，不包含训练循环。 | `GNNQNetwork` |
| `src/daxigua_rl/models/gnn_q.py` | 统一图 message passing Q 网络。输入 `GraphData` 或 `GraphTensor`，输出每个候选动作的 Q 值。 | `GNNQNetwork.forward()`、`MessagePassingLayer` |

## 资源和说明

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `assets/fruits/` | 水果图片资源目录，包含 `01.png` 到 `11.png`。 | 游戏运行时直接读取。 |
| `assets/fruits.zip` | 原始水果图片压缩包归档。 | 不参与运行，只作资源备份。 |
| `README.md` | 当前项目说明，包含游戏运行方式和操作说明。 | 已更新为游戏本体说明。 |
| `requirements.txt` | 当前游戏依赖文件。 | 只保留 `pygame` 和 `pymunk`。 |
| `LICENSE` | 开源许可证。 | Apache 2.0。 |

## 文档目录

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `docs/README.md` | 文档目录入口。 | 说明文档阅读顺序。 |
| `docs/CODING_STYLE.md` | 项目编码风格说明。 | 当前记录游戏源码采用教学型详细注释，后续改代码时应同步维护注释。 |
| `docs/codex/` | Codex 较大修改记录。 | 每次较大修改按编号追加记录。 |
| `docs/project_map/` | 项目文件职责索引。 | 结构变化后需要同步更新。 |
| `docs/learning/` | 强化学习项目化学习文档。 | 放学习路线、阶段规划、练习说明和学习笔记。 |
| `docs/rl/` | 强化学习算法和环境接口设计文档。 | 当前包含 GNN 状态图设计参考，后续模型搭建前优先阅读。 |
| `docs/rl/INTERFACE_V0.md` | RL v0 接口说明。 | 记录 `HeadlessGame`、`DaxiguaEnv`、状态数据和边界规则。 |

## 学习练习目录

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `practice/` | 学习者练习代码、实验脚本和草稿的空白工作区。 | 初始保持干净，仅用 `.gitkeep` 保留目录。 |

## 本地和生成目录

| 路径 | 作用 | 处理建议 |
| --- | --- | --- |
| `.git/` | Git 仓库元数据。 | 不手动修改。 |
| `.vscode/` | VS Code 本地配置和缓存。 | 已忽略。 |
| `__pycache__/` | Python 字节码缓存。 | 已忽略。 |
| `src/**/__pycache__/` | 包内 Python 字节码缓存。 | 已忽略。 |
| `.agents/`、`.codex/` | 当前工作环境辅助目录。 | 不属于原项目核心源码。 |

## 可复用组件

- `GameBoard`：后续优化游戏时可复用的物理和合成基类。
- `create_fruit(level, x, y)`：统一创建 pygame 水果显示对象，避免外部关心贴图路径和 rect 同步细节。
- `load_fruit_image(path, size)`：缓存水果贴图加载和缩放结果，避免重复磁盘读取。
- `create_ball(space, x, y, m, r, i)`：统一创建 pymunk 圆形刚体。
- `Board.fruit_queue`：手动游戏的待投放水果队列，q0 是当前水果，q1 到 q3 是后续水果。
- `HeadlessGame`：后续训练环境优先使用的无渲染游戏接口，不依赖 pygame 窗口。
- `DaxiguaEnv`：隔离在 `daxigua_rl` 中的 RL 环境壳层，只通过 `HeadlessGame` 访问游戏。
- `GraphBuilder`：把无渲染游戏状态和候选动作转换成框架无关 `GraphData`，供后续 GNN/Q 网络使用。
- `GraphAblator`：训练实验用的图特征消融层，通过置零特征对比不同信息组对模型的影响。
- `graph_to_tensor()`：把 `GraphData` 转成 PyTorch 张量，形成 `node_features`、`edge_index`、`edge_features` 和 `action_node_indices`。
- `GNNQNetwork`：当前最小 GNN-Q 前向模型，输入一张状态图，输出 `[action_count]` 个动作 Q 值。
- `resize_world(width, height)`：按窗口尺寸重设 pygame 画布和 pymunk 边界。当前手动游戏窗口固定，此函数主要作为内部调试或未来实验工具保留。
- `setup_collision_handler()`：水果合成逻辑所在位置，已兼容新版 `pymunk.Space.on_collision`，并在合成后调用可选的 `on_fruit_merged()`。

## 已知注意事项

- 游戏运行时直接读取 `assets/fruits/`，不再需要手动解压资源。
- 当前手动游戏窗口固定为 `400x800`，不再通过拖动窗口边框改变场地大小。
- 顶部信息层和当前悬浮水果层已经分开；生成线固定为 `180px`，用于避免待投放队列与当前水果视野冲突。
- `daxigua` 游戏本体不得 import `daxigua_rl`；RL 代码只通过稳定游戏接口访问游戏。
- `daxigua_rl.graph.tensor` 和 `daxigua_rl.models` 依赖 PyTorch；它们不会在 `daxigua_rl` 顶层自动导入，避免非训练环境被强制要求安装 torch。
- 当前 `src/daxigua/core/board.py` 已为 `pymunk 7.3.0` 做兼容处理。
- 当前 `src/daxigua/app.py` 仍集中承载表现层细节；后续如确实需要拆分，再创建对应表现层模块。
