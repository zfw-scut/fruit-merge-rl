# 项目协作规则

## 最少阅读原则

- 先读目标目录最近的 `AGENTS.md`，再只读它指向的文件；不要预先通读 `README.md`、`docs/` 或整个模块。
- 先用文件名、符号名和 `rg` 定位。除非任务明确涉及产物诊断，不搜索 `runs/`、`recordings/`、`.torch_extensions/`、`portal/node_modules/`、`portal/.next/`、`portal/dist/`。
- 用户指定了文件和参数时，从该文件开始；仅在接口、约束或影响范围不清楚时向外扩展。

## 快速路由

| 任务 | 先读 |
| --- | --- |
| TOML 参数 | `configs/AGENTS.md` |
| 水果规则、领域状态 | `src/daxigua/core/AGENTS.md` |
| 物理、CUDA、奖励、回放、场景实验室后端 | `src/daxigua/simulator/AGENTS.md` |
| 模型、Replay、loss、训练、评估 | `src/daxigua/rl/AGENTS.md` |
| 云端正式训练部署、排队、自动接力 | `docs/CLOUD_TRAINING_RUNBOOK.md`、本机 `docs/CLOUD_SERVER_LOCAL.md` |
| 模型命名、短称登记 | `docs/model_naming/NAMING_CONVENTION.md`、`docs/model_naming/MODEL_REGISTRY.md` |
| 门户、场景实验室前端 | `portal/AGENTS.md` |
| 文档、正式实验结论 | `docs/AGENTS.md` |

入口脚本在 `tools/`，测试按同名功能在 `tests/` 查找；通常不需要为此浏览整个目录。

## 必须遵守

- Python 默认使用 conda 的 `python-torch` 环境。
- 云服务器、SSH 隧道或训练面板转发前，读取本机 `docs/CLOUD_SERVER_LOCAL.md`；不得使用未登记实例。普通状态检查、小型任务不需要读取训练部署文档。
- 只有部署正式训练、加入训练队列或设置自动接力时才读取 `docs/CLOUD_TRAINING_RUNBOOK.md`，并先检查现有训练与接力状态。
- 旧分支只可用 `git show` 等只读命令审计，不得作为运行依赖或整批迁移。
- 只有新模型、新奖励、新训练方法、正式训练或有决策价值的评估，才按 `docs/AGENTS.md` 阅读历史证据并更新报告。已有参数的小幅调整、修复、UI 和工具修改不触发这套阅读。
- 不得把单次训练、loss 下降或吞吐达标写成策略提升结论。
- 提交、合并和历史改写遵循 `docs/GIT_WORKFLOW.md`。Git 提交标题和正文使用简体中文。
- 创建提交后默认普通推送当前分支；首次推送设置 upstream。除非用户明确授权，不强推、不改写历史。
