# 建立 accelerated-v1 最小基线

## 修改原因

旧训练分支已经同时包含游戏、模拟器、模型、奖励、因果归因、反事实和 Android 成品，
不适合继续通过删除局部功能开展新一轮训练架构重构。本分支需要从可以直接确认正确的
最小领域契约开始，尚未重新设计的能力一律暂缓加入。

## 分支来源

- 分支：`codex/accelerated-v1`；
- 起点：`1a95a3d9a47bc6774144061f05ef067098b43544`；
- worktree：`E:\Work\RL-learning\fruit-merge-rl-accelerated-v1`。

## 直接保留

- 水果等级、半径、质量、出生范围和游戏合成分值规则；
- 只依赖标准库的状态数据类；
- 显示半径与真实碰撞半径分离及正值校验；
- Apache 2.0 许可证和基础忽略规则。

`merge_score()` 表示游戏规则分值，不代表未来强化学习奖励设计。

## 明确移除

- pygame/pymunk 游戏、资源和物理模拟；
- HeadlessGame、DaxiguaEnv 和动作执行逻辑；
- GNN/DQN、GraphBuilder、replay、rollout、trainer 与训练配置；
- Reward V1/V2、StateAnalyzer、归因、反事实和 Shapley；
- 训练监控、CUDA 工具、旧实验记录及尚未确认的未来实现。

当前没有第三方运行依赖，因此不保留 `requirements.txt`。分支此时不能运行游戏或训练
模型，这是预期状态。

## 验证

执行 `$env:PYTHONPATH='src'; python -B -m unittest discover -s tests -v`：
7 项领域规则与状态契约测试全部通过。
