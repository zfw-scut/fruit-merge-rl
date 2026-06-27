# 项目文件索引

最后更新：2026-06-27

## 项目定位

本项目当前是一个基于 `pygame` 和 `pymunk` 的《合成大西瓜》桌面小游戏。当前只保留游戏本体，旧实验代码和旧环境封装已经删除，后续如需自动游玩能力，会重新设计相关模块。

## 核心源码

| 路径 | 作用 | 主要入口或可复用点 |
| --- | --- | --- |
| `Main.py` | 游戏入口和表现层。负责窗口、输入、正式渲染、鼠标跟随投放、预览线、可缩放窗口、下一个水果、HUD、粒子、飘字、震动和音效反馈。 | `Board.next_frame()`、`Board.run()`、`_resize_window()`、`on_fruit_merged()` |
| `Game.py` | 游戏公共逻辑。负责 pygame 画布、pymunk 物理世界、动态墙体、碰撞合成、计分、失败检测，并向表现层暴露合成事件钩子。 | `GameBoard`、`resize_world()`、`create_ball()`、`setup_collision_handler()`、`check_fail()` |
| `Fruit.py` | 水果类型和贴图定义。维护 1 到 11 级水果的半径、类型编号和图片加载缓存。 | `create_fruit(type, x, y)`、`load_fruit_image()`、各水果类 |

## 资源和说明

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `res.zip` | 水果图片资源压缩包。 | 保留在仓库中，首次运行前解压。 |
| `res/` | 解压后的水果图片目录，包含 `01.png` 到 `11.png`。 | 由 `unzip res.zip` 生成，已在 `.gitignore` 中忽略。 |
| `README.md` | 当前项目说明，包含游戏运行方式和操作说明。 | 已更新为游戏本体说明。 |
| `requirements.txt` | 当前游戏依赖文件。 | 只保留 `pygame` 和 `pymunk`。 |
| `LICENSE` | 开源许可证。 | Apache 2.0。 |

## 文档目录

| 路径 | 作用 | 备注 |
| --- | --- | --- |
| `docs/README.md` | 文档目录入口。 | 说明文档阅读顺序。 |
| `docs/codex/` | Codex 较大修改记录。 | 每次较大修改按编号追加记录。 |
| `docs/project_map/` | 项目文件职责索引。 | 结构变化后需要同步更新。 |

## 本地和生成目录

| 路径 | 作用 | 处理建议 |
| --- | --- | --- |
| `.git/` | Git 仓库元数据。 | 不手动修改。 |
| `.vscode/` | VS Code 本地配置和缓存。 | 已忽略。 |
| `__pycache__/` | Python 字节码缓存。 | 已忽略。 |
| `.agents/`、`.codex/` | 当前工作环境辅助目录。 | 不属于原项目核心源码。 |

## 可复用组件

- `GameBoard`：后续优化游戏时可复用的物理和合成基类。
- `create_fruit(type, x, y)`：统一创建水果对象，避免外部直接依赖具体水果类。
- `load_fruit_image(path, size)`：缓存水果贴图加载和缩放结果，避免重复磁盘读取。
- `create_ball(space, x, y, m, r, i)`：统一创建 pymunk 圆形刚体。
- `resize_world(width, height)`：按窗口尺寸重设 pygame 画布和 pymunk 边界，手动游戏 resize 功能依赖它。
- `setup_collision_handler()`：水果合成逻辑所在位置，已兼容新版 `pymunk.Space.on_collision`，并在合成后调用可选的 `on_fruit_merged()`。

## 已知注意事项

- 游戏运行前需要先解压 `res.zip` 生成 `res/` 图片目录。
- 当前 `Game.py` 已为 `pymunk 7.3.0` 做兼容处理。
- 当前根目录仍保留旧式单文件入口命名，后续可以继续整理为 `src/daxigua/` 包结构。
