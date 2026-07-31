# 合成大西瓜 accelerated-v1

这是模型与训练系统重构使用的最小基线分支。当前工作树刻意不提供可运行游戏、
游戏模拟器、模型或训练入口。

## 当前仅保留

- 水果等级、半径、合成计分等稳定领域规则；
- 不依赖 pygame、pymunk 或 PyTorch 的状态数据契约；
- 仓库协作规则和最小契约测试。

保留源码全部基于 Python 标准库，因此当前没有 `requirements.txt`。

## 当前明确不包含

- 桌面或 Android 游戏；
- HeadlessGame、物理模拟和环境封装；
- GNN、DQN 或其他模型结构；
- replay、rollout、trainer 和训练脚本；
- Reward V1/V2、状态分析、因果归因、反事实或 Shapley；
- CUDA 训练管线及未来方案的占位实现。

这些能力只有在新设计被明确确认后才会逐项加入。

## 最小验证

PowerShell：

```powershell
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -v
```

## 分支来源

`codex/accelerated-v1` 从提交
`1a95a3d9a47bc6774144061f05ef067098b43544` 分叉。完整旧训练体系和 Android 成品
继续保留在原项目分支中，不复制到本分支。
