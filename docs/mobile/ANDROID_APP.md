# Android AI 陪玩版

## 定位

`android/` 是当前 `560x1120` 场景的可安装移动端实现。它把本地训练得到的
结构感知 GNN-Q 模型直接封装进 APK，在手机上离线完成状态分析和 ONNX 推理：

- 不需要云服务器、网络接口或在线账号；
- 不依赖 Qt、Ubuntu、WSL 或 Android Studio；
- 画面、触控和物理由 libGDX + Box2D 提供；
- 训练时的 `StateAnalyzer -> GraphBuilder` 由 Chaquopy 中的纯 Python 桥复用；
- H256/L4 模型由 ONNX Runtime Android 在 CPU 后台线程运行；
- AI 不可用时游戏自动使用本地安全启发式，手动模式始终可以游玩。

首版只维护扩大后的 `560x1120 / spawn_y=252 / 21 actions` 规格，不兼容旧地图。

## 大厅与三种模式

应用启动后先进入大厅。设置、历史记录和所有模式入口都集中在这里；游戏内不再
提供 AI 控制权切换、设置或历史按钮。

- **单人模式**：完全由玩家控制，没有 AI 分析、移动或接管功能；
- **挑战 AI**：玩家和内置 AI 使用两个独立场景及相同水果序列进行对战；
- **AI 演示**：AI 从开局到结束全程独立游玩，人工不能接管投放；
- **继续上次**：存在有效的未完成进度时才显示。

已有进度时选择新模式会先询问是否覆盖，点“保留上次进度”即可返回大厅。大厅的
模式卡、继续、设置和历史按钮都有按压、随手指有界移动及松手回弹反馈。

## 局内行为

- 单人模式拖动顶部水果并抬手投放。上一颗水果投下后会立即显示下一颗并开放
  拖动；约 `0.14s` 的防误触冷却只限制再次投放。人工入口不等待 Box2D 稳定，
  也不等待上一组合成分值动画结束。
- AI 演示以及挑战 AI 的 AI 一方，只在自己的水果堆达到可分析边界后读取局面；
  模型尚未准备好或接近截止时自动使用本地安全启发式。
- 模型输出仍是 21 个离散动作。界面上的短暂停顿、备选位置试探、回拉和轻微颤动
  只负责模拟人的手势；最终投放会严格回到模型选中的规范动作坐标。
- 拟人轨迹使用围绕固定锚点的低频相关偏移，不逐帧累加漂移；视觉试探不会改变
  模型的规范投放列。
- 每次合成会在结果位置按水果等级显示不同颜色的爆浆、冲击环、立体 `+分值`，
  并播放两层短促合成音效。一次投放触发的多个事件会按约 `0.105s` 错峰弹出。
- 连锁层级由“源水果 ID -> 结果水果 ID”真实追踪：只有前一次结果继续参与下一次
  合成才算连锁，同次投放中互不相关的合成不会虚增连锁强度。
- 当 Box2D 确认本次连锁稳定后，同组分值会一起加速吸入分数卡。规则分数始终
  即时结算，表现动画不会延迟模型快照。
- 默认只有水果合成触发震动；水果投放和分值收集震动默认关闭，三种触发点都可在
  设置中独立修改。
- 单人和 AI 演示使用独占一行的大分数卡：玩家为淡红底/深红字，AI 为淡蓝底/
  深蓝字；对战在同一行并列红蓝双方分数与 `VS`。

## 设置与历史记录

大厅底部的“游戏设置”和“历史记录”打开对应页面。两页沿用暖色果园卡片和
站酷快乐体；它们不再占用局内 HUD，单人局内也不会出现 AI 控制权开关。

设置页提供：

- 总音效音量；
- 合成、投放、分值收集三个独立震动开关；
- 游戏速度；
- AI 对战每轮投放时限；
- 对战淘汰后进入结算页前的观察时间。

默认值为 `72%` 音量、仅合成震动、`1.0x` 游戏速度、每轮 `8s`、淘汰观察
`4s`。所有数值都会限制在安全范围内，点“恢复默认”只恢复设置，不清除历史。
配置由 libGDX `Preferences` 保存在应用私有的
`fruit-merge-ai-profile-v1` 中，修改后立即写入；重新打开应用仍会保留。

历史页顶部保留全模式、单人、挑战 AI 和 AI 演示的分类最高分、单局最多大西瓜及
对战胜/负/平；下方按新到旧分页显示最近 200 局明细。单人、挑战 AI、AI 演示
都会在正式结束时记录，明细包含模式、时间、双方或单方分数、大西瓜数和投放数。
每局使用稳定 session ID 做幂等门禁，即使从结束瞬间的草稿恢复，也不会重复
累计。旧版本的 overall 聚合会保留，但不会凭空伪造不存在的旧逐局明细。重置
历史需要再次确认，并同时清除明细、最优数据和防重账本，不改变设置。

## 自动保存、退出与恢复

未完成对局保存在应用私有的 `fruit-merge-ai-session-v1` 进度槽中。游戏约每
`1.5s` 保存一次；投放或合成后会把下一次写入提前到约 `0.18s` 内，同帧连锁只
合并写一次。进入后台、显式返回大厅和结算等关键点也会强制保存。单人/
AI 演示会保存队列随机状态、分数、计时、预览位置和每颗水果的完整 Box2D 状态；
挑战 AI 还会保存双方场景、共享回合、当前前景和 AI 已就位等待的目标列。

进度槽内部使用两份交替 bank。新快照先连同 generation 和 SHA-256 写入非活动
bank，再切换活动指针；若 Android 在写入途中结束进程，重新打开时会回退上一份
完整存档。两个 bank 都无法通过 schema、摘要和字段校验时才放弃恢复，不会用损坏
数据启动物理世界。

局内点击返回按钮后有三项选择：

- “保存并返回大厅”：保留当前局，大厅显示“继续上次”；
- “放弃本局”：删除进度，不把未完成局写入历史；
- “继续游玩”：关闭提示。

已有进度时从大厅选择其他模式还需要确认覆盖。自然游戏结束后必须点击
“确认结算”才会删除进度并返回大厅；点击结算背景不会误结束。合成粒子、飞行分值
和旧异步 AI 请求属于瞬态表现，恢复时不会重放；HUD 会直接对齐已保存的真实分数。

## AI 对战模式与结算

从大厅选择“挑战 AI”进入对战。它不是把同一个物理世界轮流交给两方，而是同时
维护玩家与 AI 两个完全独立的 Box2D 场景、分数、危险计时和大西瓜计数：

- 两方每轮共享同一颗待投水果和同一条后续队列，不能因随机序列获得优势；
- 每轮倒计时按真实时间推进，不受游戏速度设置影响；物理和危险线时间仍跟随
  游戏速度；
- 玩家可以在本轮内自由拖动并提交。若 AI 先决定，它会先把水果移动到目标列并
  悬浮等待；玩家抬手时双方原子同步落下；
- 若玩家先投，AI 不等待同步，仍按自己的思考和移动节奏投放；
- 双方都提交后短暂停顿，再共同进入下一颗；
- 倒计时结束仍未提交时，玩家按当前预览位置自动投放；AI 会在时间充足时等待
  自己的场景稳定并调用模型，临近截止时改用安全启发式。双方都在等待且 AI 已
  就位时会使用原子双投兜底；
- 同一帧总是先推进双方物理和计分，再统一判定胜负，避免更新顺序偏袒任意一方。

画面同时绘制两套场景。当前前景场景使用完整水果贴图，另一方以对应红/蓝色低透明
虚影留在背景；点击右上角“玩家前景/AI 前景”开关即可切换。顶部分数卡分别用
淡红和淡蓝显示玩家与 AI 分数，前景方的危险线、投放预览、合成爆浆和浮分保持
清晰可见。

任意一方水果持续越过警戒线后即被淘汰；若双方在同一帧淘汰，则按最终分数决定
胜负，同分记为平局。淘汰后先冻结规则推进并显示失败原因，在设置的观察时间内
仍允许切换前景查看两个场景；倒计时结束后才出现胜利、失败或平局结算页。正式
结算页仍需点击“确认结算”才返回大厅。

单场结算和对战结算都会显示“已超越一定比例玩家”的提示。这个百分比由本地内部
平滑估计产生，只用于提供成绩量级反馈，不联网、不收集玩家数据，也不代表真实
在线排行榜统计；界面只展示最终整数结果，不展示估计细节。

## AI 实时互动

AI 演示与挑战 AI 会显示带高对比颜文字的动态对话气泡。`WELCOME`、`READY`、
`THINKING`、`HESITATING`、`HAPPY`、`SURPRISED`、`WORRIED` 分别表达中性陪伴、
决策就位、思考、犹豫、AI 合成得分、对玩家高等级连锁的惊讶和临近危险线。不同
事件使用不同语料池，开局/读档不会抽到合成台词，就位也不会误说大厅或设置内容。

七类语料分别保存在 `assets/dialogue/*.txt`，每类 1024 条、合计 7168 条由
大模型逐条完整构思的独立句子。运行时不会组合前缀或词块，只从完整句子中用无
放回洗牌袋抽取。普通发言结束后随机静默约 3.5～7 秒，任何两次开口还有 2.2 秒
硬间隔；思考等低价值状态只按概率发言，高连锁和危险可以突破软冷却但不能刷屏。
同一 `ScoreSequence` 或同一对战回合/阵营的连续合成最多请求一次反馈，危险提示
按进入危险状态的边沿触发。

颜文字和文字使用两条表现通道：事件发生时颜文字可立即短促弹出或改变，文字仍
必须通过统一限频器。因此 AI 不会因为少说话而显得没有反应，也不会在连续事件中
不断刷句子。所谓“完整语料”是 APK 内预生成静态句库，不需要网络，也不会在手机
端调用在线大模型。

气泡改用独立暖色实体底、情绪色深描边、约 20px 正文和大号颜文字。气泡矩形与
当前可见水果相交时，背景会平滑过渡到约 55%～65% 不透明度；文字、边框和颜文字
仍保持高可见度，重叠解除后再平滑恢复。

这些互动只属于表现层：不会更改 Q 值、动作列或倒计时。单人模式完全不显示 AI
头像/气泡，结算页也会隐藏气泡，避免遮挡最终分数。

## 界面主题与修改入口

当前采用“暖色果园”外围主题：奶油渐变背景、暖杏信息卡、弹性模式卡、红蓝阵营
分数、珊瑚危险线和点状投放引导。选型稿保存在
`docs/mobile/ui-concepts/variant-a-warm-orchard.png`。

界面继续和游戏使用同一个 libGDX `560x1120` 逻辑画布，而不是额外叠加 Android
XML。这样计分、页面状态、触摸命中区和 `FitViewport` 始终使用同一套坐标，不需要
跨 GL/UI 线程同步动态状态，也不会在长宽比不同的手机上与水果画面错位。

外围 UI 的主要修改入口在
`android/core/src/main/java/com/fruitmerge/ai/game/FruitMergeApplication.java`：

- 文件顶部的 `Color` 常量负责主题调色板；
- `drawBackground()` 负责渐变和果园装饰；
- `drawHome()` 负责大厅、三张模式卡和继续入口；
- `drawPanels()` 负责独占分数行、下一颗提示、棋盘底色和提示线；
- `drawEffects()`、`drawScoreTokens()` 负责爆浆、连锁分值与吸附表现；
- `drawText()`、`drawGameOverOverlay()`、`drawDuelResultLayer()` 负责中文文字和
  需要显式确认的单场/对战结算；
- `drawAiReaction()` 负责 AI 表情头像与对话气泡；
- `drawOverlayPage()` 及其设置、历史子方法负责两个持久化浮层。

双场景规则集中在
`android/core/src/main/java/com/fruitmerge/ai/game/DuelMatch.java`。设置/历史由
同目录 `GameProfileStore.java` 保存，未完成局由 `GameSessionStore.java` 的双
bank 槽保存，按钮的按压/跟随/回弹状态由 `UiMotionController.java` 管理。规则、
持久化与绘制因此可以分别做纯 Java 测试。

按钮动画只改变绘制时的缩放和最大约 `8px` 跟手偏移，真实命中矩形保持不动；
拖出后取消、回到容错范围后可提交，松手产生短回弹。第二根手指不能抢占已有控件。

水果表现与主题明确隔离：`loadFruitTextures()`、`drawFruitBodies()`、
`drawPreviewAndQueue()`、`drawFruit()` 继续使用项目原有 `assets/fruits/01.png`
至 `11.png`。场内和待投水果的尺寸、位置、透明度、Box2D 物理和模型输入均未
改变；顶部三颗队列预览只重新排入新的提示槽，贴图、尺寸和透明度保持不变。

文字不再使用英文 Nunito 或 libGDX 内置的 15px 字体。Android 与 Windows 预览
共同加载 `assets/fonts/ui-cute.fnt` 和 `ui-cute.png`：它们由 OFL 1.1 授权的
站酷快乐体生成 64px 中文 UI 字表图集，再在运行时向下采样。标题、计分卡、
下一颗、AI 状态、操作提示与结束弹层均改为中文；浮动分值也使用同一 Q 版数字，
以一层紧凑暗色阴影和亮色正文绘成立体字。四个运行字号共享同一张生成图集纹理。
字体生成器会扫描 Android `core/app` 主源码中的真实 Java 字符串字面量以及
`assets/dialogue/*.txt` 完整语料并跳过注释，新增中文设置、结算文案或 AI 台词
不会再依赖手工猜测字表。

所有卡片文字继续通过 `GlyphLayout` 按实际字形宽高居中或截断，避免不同字体度量
与硬编码 baseline 重新产生穿线、重叠和模糊。字体 OFL 和 Kenney Impact Sounds
的 CC0 原始许可证会随 APK 一同放入 `assets/licenses/`；音效来源记录位于
`assets/audio/README.md`。

## Windows 本地 UI 预览

`android/desktop/` 是只用于表现层验收的 LWJGL3 入口。它直接运行 Android 使用的
同一个 `FruitMergeApplication`、Box2D 与 canonical 水果资源，不复制 UI 实现；
仅用 stub 替换 Android ONNX/震动服务，因此不代表移动模型推理性能。

打开可交互窗口：

```powershell
powershell -ExecutionPolicy Bypass -File `
  android\scripts\run-ui-preview.ps1
```

自动渲染、保存 PNG 并退出：

```powershell
powershell -ExecutionPolicy Bypass -File `
  android\scripts\run-ui-preview.ps1 -Capture
```

默认输出为 `runs/mobile_ui_preview/current.png`。可以用真实手机截图尺寸复查
`FitViewport` 留边和高倍率字体采样：

```powershell
powershell -ExecutionPolicy Bypass -File `
  android\scripts\run-ui-preview.ps1 -Capture `
  -Width 1156 -Height 2990 `
  -Output runs\mobile_ui_preview\phone-1156x2990.png
```

截图模式使用目标宽高的离屏 FrameBuffer，不受 Windows 桌面工作区对隐藏窗口尺寸
的限制；保存前会核对 FBO 与像素数，保存后还会读取 PNG IHDR 复核实际宽高。

检查真实共享渲染器中的爆浆、连锁浮分与吸附阶段：

```powershell
powershell -ExecutionPolicy Bypass -File `
  android\scripts\run-ui-preview.ps1 -Capture -Showcase `
  -CaptureFrames 18 `
  -Output runs\mobile_ui_preview\merge-burst.png
```

`-CaptureFrames` 可设为 `1..600`；在 60 FPS 左右时，约 18 帧可见错峰爆浆，
约 48 帧可见三组分值共同加速飞向计分卡。游戏按真实帧间隔推进，因此这些帧数
只是便捷观察点，不是严格的时间测试门禁。`-Showcase` 只向桌面预览排入视觉事件，
不创建 Box2D 水果，Android 正常入口不会调用它。

大厅、三种模式、设置/历史、退出弹窗和确认结算都能直接进入，不需要在自动截图
前模拟点击。`-Screen` 支持
`home/solo/duel/demo/reaction/reaction-overlap/settings/history/exit/new/result`：

```powershell
powershell -ExecutionPolicy Bypass -File `
  android\scripts\run-ui-preview.ps1 -Capture -Screen home

powershell -ExecutionPolicy Bypass -File `
  android\scripts\run-ui-preview.ps1 -Capture -Screen settings

powershell -ExecutionPolicy Bypass -File `
  android\scripts\run-ui-preview.ps1 -Capture -Screen duel

powershell -ExecutionPolicy Bypass -File `
  android\scripts\run-ui-preview.ps1 -Capture -Screen result
```

桌面入口能提前发现字体、卡片、水果预览和绘制顺序问题，但不能代替 Android
状态栏、刘海、安全区、触觉和真机持续帧率测试。

## 模型与移动契约

当前 APK 使用尺寸迁移训练中的本地最优推理 checkpoint：

- 架构：structure-aware dueling GNN-Q，hidden `256`，message layers `4`；
- 输入：动态 `N/E`，节点特征 `62`，边特征 `47`，固定 `21` 个动作；
- ONNX opset：`18`；
- 模型大小：`13,828,110` bytes；
- 模型 SHA-256：
  `7e1c95c958799714579f75d8dbd8dc7b1e3ad182ab23fa3dac51306d4865e5f4`；
- 24 个真实稳定局面中，PyTorch 与 ONNX Runtime 的 argmax 一致率为 `24/24`，
  最大绝对误差为 `0.0001220703125`。

`src/daxigua_mobile/` 是 Android 可调用的纯 Python 桥。它不导入 pygame、pymunk
或 torch，并将 Java 局面 JSON 转成 ONNX 五输入所需的连续数组。

## 在 Windows 构建 APK

要求：

- 64 位 Windows；
- JDK 17 或更高版本；
- 可联网下载首次构建所需的 Android/Gradle/Maven 依赖；
- 项目本地已有经过验证的 ONNX 导出产物。

首次构建直接运行：

```powershell
powershell -ExecutionPolicy Bypass -File `
  android\scripts\build-debug-apk.ps1 -BootstrapSdk
```

脚本会把 Android SDK 35、Build Tools 35 和 Gradle 缓存放在
`android/.toolchains/` 与 `android/.gradle-user-home/`，不会要求全局安装 Android
Studio。后续构建可省略 `-BootstrapSdk`：

```powershell
powershell -ExecutionPolicy Bypass -File `
  android\scripts\build-debug-apk.ps1
```

可直接安装的成品输出到：

```text
android/release/FruitMergeAI-v0.1.0-debug.apk
```

脚本在结束时打印 APK 的 SHA-256。当前只打包 `arm64-v8a`，适用于主流 64 位
Android 手机；最低系统版本为 Android 7.0（API 24）。

## 安装到手机

开启手机的 USB 调试并连接电脑后：

```powershell
android\.toolchains\android-sdk\platform-tools\adb.exe devices
android\.toolchains\android-sdk\platform-tools\adb.exe install -r `
  android\release\FruitMergeAI-v0.1.0-debug.apk
```

也可以把 APK 复制到手机后直接打开安装。它是本地 debug 签名包，Android 可能要求
允许文件管理器安装未知来源应用。

## 重新导出模型

只对项目自己生成并确认可信的 checkpoint 执行导出：

```powershell
conda run -n python-torch python -m pip install `
  -r requirements-mobile-export.txt
conda run -n python-torch python tools\export_android_model.py `
  --checkpoint runs\cloud_checkpoints\size_transfer_560x1120_15k_best\best_inference.pt `
  --output-dir runs\mobile_export\size_transfer_560x1120_15k_best
```

导出工具只有在真实局面的 PyTorch 原模型、移动包装器和 ONNX Runtime 三路数值与
argmax 都通过门禁后才写模型和元数据。Android 构建还会检查模型字节数，运行时再次
检查 SHA-256；更换模型后需要同步更新 Android 运行时中的固定哈希。

## 当前边界

- 移动端首版使用 Box2D，训练环境使用 Pymunk/Chipmunk。几何、碰撞半径、重力、
  摩擦、弹性和稳定门限已按现有规则映射，但两套求解器不会逐帧完全一致。
- 当前 ONNX Runtime 使用 CPU。单局面的图很小，GPU/NPU 委托的初始化和拷贝成本
  通常高于收益；推理在后台线程执行，不阻塞渲染。
- APK 内没有训练逻辑、反事实模拟或模型 checkpoint 恢复逻辑；这里的双 bank
  只用于恢复正在玩的游戏进度。因果归因与结构学习已体现在训练后的网络权重及
  移动端结构图输入中。
- AI 对战是同一设备上的玩家对内置 AI，不是联网匹配或远程 PvP；双方共享水果
  序列，但物理求解仍是两个本地独立世界。
- 当前只有一个未完成进度槽；确认用新模式覆盖后无法恢复旧局。
- 进度、设置与历史均保存在 Android 应用私有 Preferences 中；卸载应用或在系统
  设置中清除应用数据会删除这些本地记录。
- 结算百分比是离线体验反馈，不应被解释为真实用户分布、账号排名或统计承诺。
