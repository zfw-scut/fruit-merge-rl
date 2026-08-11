# Xigua Atlas 前端

这是合成大西瓜项目本地知识与工具门户的前端。它使用 React、Vinext、ECharts 和
Framer Motion，但不独立承担本地文件读取或进程控制；这些能力只由回环地址上的
`daxigua.portal` 白名单服务提供。

通常不需要在这里手动启动。请从仓库根目录运行：

```powershell
conda run -n python-torch python tools/open_project_portal.py
```

单独开发前端时可使用：

```powershell
npm install
npm run dev
npm run lint
npm run build
```

前端默认访问 `http://127.0.0.1:4312`，页面自身默认位于
`http://127.0.0.1:3000`。门户是本地应用，不应部署为公开站点；否则启动本机工具、
访问训练产物和读取文档的信任边界都会改变。
