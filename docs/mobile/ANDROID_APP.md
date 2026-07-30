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

## 游戏行为

- 默认开启 AI 陪玩；点击右上角 `AI ON / AI OFF` 可随时切换。
- 手动模式下拖动顶部水果，抬手投放。
- AI 只在水果堆连续稳定 `0.2s` 后读取一次局面，不会在上一颗水果仍运动时抢投。
- 模型输出仍是 21 个离散动作。界面上的短暂停顿、备选位置试探、回拉和轻微颤动
  只负责模拟人的手势；最终投放会严格回到模型选中的规范动作坐标。
- 合成、投放、危险线和游戏结束有轻量动效与触觉反馈。

## 界面主题与修改入口

当前采用“暖色果园”外围主题：奶油渐变背景、暖杏信息卡、薄荷色 AI 开关、
珊瑚危险线和点状投放引导。选型稿保存在
`docs/mobile/ui-concepts/variant-a-warm-orchard.png`。

界面继续和游戏使用同一个 libGDX `560x1120` 逻辑画布，而不是额外叠加 Android
XML。这样计分、AI 状态、触摸命中区和 `FitViewport` 始终使用同一套坐标，不需要
跨 GL/UI 线程同步动态状态，也不会在长宽比不同的手机上与水果画面错位。

外围 UI 的主要修改入口在
`android/core/src/main/java/com/fruitmerge/ai/game/FruitMergeApplication.java`：

- 文件顶部的 `Color` 常量负责主题调色板；
- `drawBackground()` 负责渐变和果园装饰；
- `drawPanels()` 负责计分卡、下一颗提示、AI 状态板、棋盘底色和提示线；
- `drawText()`、`drawOutlines()`、`drawGameOverOverlay()` 负责文字、描边和结束弹层。

水果表现与主题明确隔离：`loadFruitTextures()`、`drawFruitBodies()`、
`drawPreviewAndQueue()`、`drawFruit()` 继续使用项目原有 `assets/fruits/01.png`
至 `11.png`。场内和待投水果的尺寸、位置、透明度、Box2D 物理和模型输入均未
改变；顶部三颗队列预览只重新排入新的提示槽，贴图、尺寸和透明度保持不变。

文字不再使用 libGDX 内置的 15px 字体。Android 与 Windows 预览共同加载
`assets/fonts/ui-nunito.fnt` 和 `ui-nunito.png`：它们由 OFL 1.1 授权的 Nunito
源字体生成 64px 高分辨率图集，再在运行时向下采样。所有卡片文字通过
`GlyphLayout` 按实际字形宽高居中或截断，避免不同字体度量与硬编码 baseline
重新产生穿线、重叠和模糊。

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
- APK 内没有训练逻辑、反事实模拟或 checkpoint 恢复逻辑；因果归因与结构学习已
  体现在训练后的网络权重及移动端结构图输入中。
