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

成品输出到 `release/FruitMergeAI-v0.1.0-debug.apk`。工具链、生成的 Python/模型
assets、native libraries、Gradle 缓存和 release APK 均为本地生成目录，不提交 Git。
