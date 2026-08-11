# Xigua Atlas 项目知识与工具门户

## 1. 定位

Xigua Atlas把项目里频繁变化的Markdown、正式模型评估、训练遥测和常用工具入口放在
同一个本地页面中。它是阅读与操作入口，不替代原始文档、模型报告或训练产物：图表中的
代表数据仍必须回到对应评估报告解释，不能把柱形高度直接当成因果结论。

门户当前包括五个页面：

- **总览**：代表模型的120 FPS均分、参数量、transition和估算训练时长；
- **模型图谱**：训练规模—120 FPS得分散点图、代表模型排序和报告跳转；
- **文档知识库**：全部`docs/**/*.md`的全文搜索、分类过滤、Markdown渲染和内部链接；
- **工具中心**：场景实验室、历史训练面板、模型观看器和CUDA训练门禁；
- **实时训练**：轻量读取并嵌入现有`127.0.0.1:8765`训练面板。

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

默认地址：

| 服务 | 地址 | 说明 |
| --- | --- | --- |
| 门户页面 | `http://127.0.0.1:3000` | 自动打开浏览器 |
| 本地控制API | `http://127.0.0.1:4312` | 仅允许回环地址 |
| 训练面板 | `http://127.0.0.1:8765` | 已有本地面板或SSH转发 |

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

## 3. 频繁修改文档时的行为

后端每次请求都重新扫描当前工作树中的Markdown，不维护需要手动重建的静态索引。前端
每3秒只读取一个轻量revision；检测到文档数量或最新修改时间变化时，才重新获取正文。
因此保存Markdown后无需重新编译前端，通常在数秒内即可搜索和阅读新内容；也可以点击
顶栏刷新按钮立即更新。

当前选中文档会在刷新时保留，避免写作过程中阅读位置被无意义重置。模型主页的代表模型
是一组有意精选的正式对照，而不是自动把任意run当成有效模型；新增正式模型后应在更新
评估报告与`COMPARISON_MATRIX.md`的同时更新`portal/app/model-data.ts`。

## 4. 工具中心的安全边界

网页不能提交任意命令。后端只接受代码中登记的工具ID和结构化参数，并进行以下校验：

- 服务只绑定`127.0.0.1`；
- 设备、FPS等参数只能从白名单枚举中选择；
- 端口、评估局数等数值有上下界；
- checkpoint、run和config必须位于项目允许目录；
- 子进程使用参数数组与`shell=False`，不拼接shell字符串；
- 云服务器密码、SSH口令和其它凭据不会进入门户配置或日志。

CUDA训练门禁会占用GPU并写入`runs/preflight`，启动前有显式确认。门户当前不提供“直接
开始正式长训”按钮，避免将一次普通页面点击扩大为高成本训练授权。

## 5. 数据与物理身份

首页代表模型数据来自`docs/model_evaluations/COMPARISON_MATRIX.md`。现有归档模型全部使用
历史“合成水果继承动量”物理；当前代码已经采用“新水果线速度、角速度归零”。门户会把
这两个身份分开显示，当前零初速度长训完成前不会与旧模型画成无条件可比的单一结论。

训练时长图属于近似投入视图：短训由稳定吞吐估算，128M基线使用冻结墙钟预算。它适合
观察数量级，不是云服务精确计费记录。

## 6. 开发与验证

```powershell
cd portal
npm run lint
npm run build
cd ..

$env:PYTHONPATH = 'src'
conda run -n python-torch python -m unittest tests.test_portal_service -v
```

门户前端位于`portal/app`，本地API和工具白名单位于`src/daxigua/portal/service.py`，统一
启动器位于`tools/open_project_portal.py`。
