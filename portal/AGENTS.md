# Xigua Atlas 前端导航

门户是本地应用，不公开部署。前端只调用回环 API，不直接读取文件或启动任意命令。

| 修改目标 | 文件 |
| --- | --- |
| 导航、总览、文档、工具页 | `app/PortalShell.tsx` |
| 训练页 | `app/TrainingWorkspace.tsx` |
| 场景实验室前端 | `app/ScenarioWorkspace.tsx` |
| 模型展示数据 | `app/model-data.ts` |
| 图表封装 | `app/EChart.tsx`、`app/chart-runtime.ts` |
| 全局样式 | `app/globals.css` |
| 本地 API、工具白名单 | `../src/daxigua/portal/service.py` |
| 统一启动器 | `../tools/open_project_portal.py` |

- 先在目标组件内修改，不为小 UI 改动阅读 RL 或模拟器实现。
- 场景请求/响应契约变化时才检查 `src/daxigua/simulator/scenario_lab_*` 和 `tests/test_scenario_lab.py`。
- 工具参数必须继续由 Python 后端白名单校验；不要让前端提交任意命令或任意路径。
- `model-data.ts` 只放经过选择的正式模型；新增结论时同步对应评估报告。
- 不搜索或修改 `node_modules/`、`.next/`、`.vinext/`、`dist/`。

最小验证：

```powershell
cd portal
npm run lint
npm run build
```

只改 Python 门户服务时运行 `tests.test_portal_service`；只改场景前端契约时运行 `tests.test_scenario_lab`。
