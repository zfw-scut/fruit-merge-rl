# Xigua Atlas 项目知识与工具门户

## 1. 定位

Xigua Atlas把项目里频繁变化的Markdown、正式模型评估、训练遥测和常用工具入口放在
同一个本地页面中。它是阅读与操作入口，不替代原始文档、模型报告或训练产物：图表中的
代表数据仍必须回到对应评估报告解释，不能把柱形高度直接当成因果结论。

门户当前包括八个页面：

- **总览**：代表模型的120 FPS均分、参数量、transition和估算训练时长；
- **模型图谱**：训练规模—120 FPS得分散点图、代表模型排序和报告跳转；
- **数据分析**：发现本地统计数据集，联动查看时间尺度、单因素关系、双因素交互和原始表格；
- **文档知识库**：全部`docs/**/*.md`的全文搜索、分类过滤、Markdown渲染和内部链接；
- **工具中心**：场景实验室后端、历史训练数据源、模型观看器和CUDA训练门禁；
- **实时训练**：原生展示云端/本地训练进度、训练队列、GPU资源、损失、评估曲线和
  Merge Potential大规模统计进度；
- **场景查看**：不启动物理或模型服务，直接读取单个或成组JSON场景快照；
- **场景实验室**：原生编辑实时物理场景，对照21动作模型预测与真实执行结果，并按需
  显示完整圆障碍加权 Voronoi / Free-Space Graph。

训练和场景页面不再使用iframe、弹出窗口或各自维护的旧HTML。`8765`和`8769`仅保留
数据/物理API，直接访问其根路径会返回`410`和门户地址，防止旧界面继续产生双重维护。

视觉语言采用浅色Soft UI新拟态：蓝紫暖粉外部渐变承托冷灰应用框架，左上高光与右下柔影
表达浮起层，内阴影表达输入和筛选层，皇家蓝作为主强调色。动画保留模型轨道、状态脉冲、
卡片悬浮和页面过渡；系统要求减少动态效果时会自动降级。

## 2. 启动

第一次使用先安装前端依赖：

```powershell
cd portal
npm install
cd ..
```

之后从仓库根目录一条命令启动：

```powershell
conda run -n python-torch python tools/open_project_portal.py
```

启动器默认读取本机Git忽略的`docs/CLOUD_SERVER_LOCAL.md`当前实例登记。表中的`门户遥测`
列决定连接范围：标为`on`的每个实例会获得独立本地回环端口和SSH隧道，并写入
`runs/portal_processes/telemetry_sources.json`；面板并行读取全部来源，实例增删不再要求
修改前端。只查看本地文档、不连接云端时可增加`--no-cloud-telemetry`。

门户运行期间会监测前端源码与生产构建身份。保存前端修改后，启动器会先构建
新版本，再替换过时的 Node 进程；构建失败时保留当前可用页面。启动器直接管理
实际监听3000端口的 Node PID，异常遗留的已登记进程会在下次启动时清理。
如4312上已有由本工作区登记的门户API，重新运行启动命令会先关闭整棵旧门户进程树，再用
当前Python代码、来源登记和前端构建重启；未登记的占用进程仍不会被误杀。前端被自动替换后，
已打开的页面需普通刷新一次以载入新资源。

默认地址：

| 服务 | 地址 | 说明 |
| --- | --- | --- |
| 门户页面 | `http://127.0.0.1:3000` | 自动打开浏览器 |
| 本地控制API | `http://127.0.0.1:4312` | 仅允许回环地址 |
| 训练数据API | 云端仍为`127.0.0.1:8765/api/status` | 启动器映射到本地`18765`起的独立端口 |
| 场景实验室API | `http://127.0.0.1:8769/api/health` | 从工具中心按需启动；实时物理使用Tensor/CUDA |

常用选项：

```powershell
# 只启动后端，供前端开发使用
conda run -n python-torch python tools/open_project_portal.py --backend-only --no-open

# 调整本地端口
conda run -n python-torch python tools/open_project_portal.py `
    --api-port 4312 --web-port 3000
```

关闭启动命令所在的终端或按`Ctrl+C`即可结束门户以及由该启动器创建的前端子进程。
工具中心另行启动的进程应在各自卡片上点击“停止”。

数据分析、实时训练、场景查看和场景实验室可用`#analysis`、`#live`、`#scenes`、`#lab`直接定位，例如
`http://127.0.0.1:3000/#analysis`。工具中心启动对应服务后会留在门户内并自动进入原生
页面，不会再打开旧页面。

前端开发需要隔离已有4312进程时，可用`?api=<回环端口>`指定控制API，例如
`http://127.0.0.1:3300/?api=4412#lab`；只接受1024～65535的整数端口，主机始终固定为
`127.0.0.1`。

## 3. 统计数据分析

数据分析页通过本地控制API读取`runs/analysis/*/analysis/analysis_manifest.json`及同目录的
小型CSV分析表，不读取原始张量分片，也不修改统计产物。数据集响应统一包含身份、采集条件、
表名、列定义和结构化行；Merge Potential在该通用表格契约上增加专用分析视图，未来其它
统计任务即使尚无专用图表，也可以先进入可搜索、排序和分页的数据表。

当前 Merge Potential 页面包含：

- 等级×未来投放窗口热图和等级生命周期摘要；
- 按等级、状态因素和时间指标联动的条件概率曲线；
- 单因素加权分箱关联强度，用于比较简单状态因素的区分能力；
- 高度×同级数量、占用率×最近同级距离的联合条件热图和精确概率矩阵；
- 五张基础分析表的等级筛选、表内搜索、列排序、核心列/全部列切换和分页。

“关联强度”是同一等级内加权分箱条件概率的分离程度，不是Pearson相关系数，也不表示因果。
状态条件表继续使用每颗水果总权重为1的归一化口径，避免长寿水果因重复快照获得更大权重。
页面只改变浏览方式，正式解释仍应保留策略、物理身份、样本条件和数据口径。

## 4. 训练遥测与队列

实时训练页先从门户聚合接口选择数据源，再消费该来源的`/api/status`，不进入actor、
learner或Replay热路径。来源条目含实例名称、用途、在线状态和响应延迟；某台离线不会让
其它实例或本机任务产物消失。
首屏只保留进度、吞吐、ETA、得分和GPU概览；学习细项、动作分布和事件使用按语义折叠的
`details`区块，避免一次铺开大量训练字段。

曲线使用ECharts原生渲染。纵轴先按数据范围选择`1 / 2 / 2.5 / 5 / 10 × 10^n`步长，
再按步长确定小数位，因此刻度优先是整数或`0.02`、`0.05`这类整小数。底部滑块、
Shift+滚轮和工具栏框选都可局部缩放；复位只改变浏览器视图，不修改训练文件。

统一队列文件位于`runs/training_queue.json`，格式版本为1。接力或调度脚本负责原子写入，
面板只读；训练服务会把当前run的实时进度合并进同名计划。单项支持：

```json
{
  "id": "aux-128m-to-120fps",
  "name": "120 FPS物理域适应",
  "status": "queued",
  "position": 2,
  "run_dir": "runs/cloud_example",
  "config": "configs/example.toml",
  "training_physics_fps": 120,
  "planned_transitions": 16000000,
  "depends_on": "aux-128m"
}
```

状态枚举为`queued / waiting / preflight / running / evaluating / completed / failed /
cancelled`。文件损坏或不存在时会退化为只显示当前run，不中断训练或面板。

Merge Potential采集使用同一个训练数据API同步到实时训练页。云端遥测服务只扫描
`runs/analysis/*/manifest.json`中用途为`merge_potential_t_merge_collection`的清单，不读取
原始张量分片；本地门户经SSH转发连接云端8765端口后，可直接看到云端完成局数、进度、
吞吐、样本行数、峰值显存、物理条件和离线汇总状态。清单超过默认120秒（或自定义进度
写入间隔的3倍）未更新且仍标记为`running`时，面板显示“更新已停滞”，但不会干预采集
进程。

除历史RL训练和Merge Potential专用兼容视图外，新增任务统一发布小型
`runs/task_telemetry/<task-id>.json`快照。格式版本1包含任务类型、阶段、状态、进度、
指标描述、有限长度历史和身份；门户只扫描该固定目录，不遍历数据集或checkpoint目录。
只要生产者使用`TaskTelemetryPublisher`，数据生成、监督训练或后续分析就可直接显示为
通用任务卡片，不需要在Python代理和React页面分别增加一个专用分支。堵塞风险数据生成与
轻量预测器训练已接入该契约。

## 5. 轻量场景查看与画布边界

`#scenes`是无后端依赖的快照查看入口，接受一个场景对象、场景数组，或包含`scenes`、
`snapshots`、`frames`字段的JSON。场景至少包含`fruits`数组；可选分数、投放数、物理帧、
队列和几何。它适合快速检查终局、检测前后帧和反事实分支结果，不承担实时推进、编辑、
模型推理或CUDA调用。

基础棋盘、水果、危险线、网格和墙体已经抽成只读`SceneCanvas`；场景实验室的双环境静态
对照也复用同一画布。画布预留水果前后两层React叠加入口，未来检测器图层可独立实现，
不必继续把所有诊断绘制逻辑塞进`ScenarioWorkspace`。实时编辑画布仍留在实验室内，避免
本轮结构调整同时改写成熟的指针手势和权威物理同步逻辑。

## 6. 频繁修改文档时的行为

后端每次请求都重新扫描当前工作树中的Markdown，不维护需要手动重建的静态索引。前端
每3秒只读取一个轻量revision；检测到文档数量或最新修改时间变化时，才重新获取正文。
因此保存Markdown后无需重新编译前端，通常在数秒内即可搜索和阅读新内容；也可以点击
顶栏刷新按钮立即更新。

当前选中文档会在刷新时保留，避免写作过程中阅读位置被无意义重置。模型主页的代表模型
是一组有意精选的正式对照，而不是自动把任意run当成有效模型；新增正式模型后应在更新
评估报告与`COMPARISON_MATRIX.md`的同时更新`portal/app/model-data.ts`。

## 7. 工具中心的安全边界

网页不能提交任意命令。后端只接受代码中登记的工具ID和结构化参数，并进行以下校验：

- 服务只绑定`127.0.0.1`；
- 设备、FPS等参数只能从白名单枚举中选择；
- 端口、评估局数等数值有上下界；
- checkpoint、run和config必须位于项目允许目录；
- 子进程使用参数数组与`shell=False`，不拼接shell字符串；
- 云服务器密码、SSH口令和其它凭据不会进入门户配置或日志。

CUDA训练门禁会占用GPU并写入`runs/preflight`，启动前有显式确认。门户当前不提供“直接
开始正式长训”按钮，避免将一次普通页面点击扩大为高成本训练授权。

## 8. 数据与物理身份

首页代表模型数据来自`docs/model_evaluations/COMPARISON_MATRIX.md`。五层Fast 128M及更早
模型使用历史“合成水果继承动量”物理；结构化辅助128M及其120 FPS迁移模型使用“新水果
线速度、角速度归零”物理。门户会分开显示两种身份，跨物理排序只描述已保存结果，不视为
严格同分布比较。

训练时长图属于近似投入视图：短训由稳定吞吐估算，128M基线使用冻结墙钟预算。它适合
观察数量级，不是云服务精确计费记录。

场景实验室的正常A0~A20投放与训练共享`TensorVectorSimulator`和CUDA碰撞Kernel。实时
会话使用同一Kernel的单帧增量入口，关闭自由下落快进以保留可见下落动画；页面身份区会
显示`Tensor / CUDA`和“训练物理同源”。旧Pymunk运行后端及其依赖已经从当前代码删除。

## 9. 开发与验证

```powershell
cd portal
npm run lint
npm run build
cd ..

$env:PYTHONPATH = 'src'
conda run -n python-torch python -m unittest `
    tests.test_portal_service tests.test_task_telemetry `
    tests.test_telemetry_sources tests.test_scenario_lab tests.test_rl_baseline -v
```

门户前端位于`portal/app`；新增工作区样式位于`workspace-modules.css`，不再继续扩大旧的
`globals.css`。大型分析、训练、场景查看和场景实验室组件按页面动态加载。本地API和工具
白名单位于`src/daxigua/portal/service.py`，多来源聚合位于
`src/daxigua/portal/telemetry_sources.py`，统一启动器位于`tools/open_project_portal.py`。
