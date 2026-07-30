# Fruit Merge AI for Android

这是当前 560x1120 AI 陪玩版的 Gradle 工程。完整架构、模型契约、Windows 构建和
安装说明见 [`../docs/mobile/ANDROID_APP.md`](../docs/mobile/ANDROID_APP.md)。

首次在 Windows 构建：

```powershell
powershell -ExecutionPolicy Bypass -File `
  scripts\build-debug-apk.ps1 -BootstrapSdk
```

后续构建：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-debug-apk.ps1
```

直接运行与 Android 共用渲染器的 Windows UI 预览（默认进入模式大厅）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run-ui-preview.ps1
```

加 `-Capture` 会自动截图到 `../runs/mobile_ui_preview/current.png` 后退出；也可以用
`-Width`、`-Height` 模拟具体手机截图尺寸，或用
`-Screen home/solo/score-low/score-high/duel/demo/settings/history/exit/new/`
`result/reaction/reaction-overlap` 直达页面；`score-low` / `score-high` 用于验收
分数卡底部/顶部停靠及棋盘联动，最后两项分别用于验收 AI 气泡和水果遮挡时的
半透明效果。

成品输出到 `release/FruitMergeAI-v0.1.0-debug.apk`。工具链、生成的 Python/模型
assets、native libraries、Gradle 缓存和 release APK 均为本地生成目录，不提交 Git。
