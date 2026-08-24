"use client";

import type { EChartsOption } from "echarts";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  ArrowUpRight,
  BarChart3,
  BookOpenText,
  CheckCircle2,
  ChevronRight,
  Command,
  Cpu,
  Database,
  ExternalLink,
  FileText,
  FlaskConical,
  Gauge,
  Images,
  Layers3,
  LayoutDashboard,
  Menu,
  Play,
  RefreshCw,
  Search,
  Server,
  Settings2,
  Sparkles,
  Square,
  Wrench,
  X,
  type LucideIcon,
} from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";
import { EChart } from "./EChart";
import { CURRENT_TRAINING, FEATURED_MODEL_IDS, MODELS, type ModelRecord } from "./model-data";
import type { DashboardStatus } from "./TrainingWorkspace";

const AnalysisWorkspace = lazy(async () => ({ default: (await import("./AnalysisWorkspace")).AnalysisWorkspace }));
const ScenarioWorkspace = lazy(async () => ({ default: (await import("./ScenarioWorkspace")).ScenarioWorkspace }));
const SceneViewerWorkspace = lazy(async () => ({ default: (await import("./SceneViewerWorkspace")).SceneViewerWorkspace }));
const TrainingWorkspace = lazy(async () => ({ default: (await import("./TrainingWorkspace")).TrainingWorkspace }));

const API = (() => {
  if (typeof window === "undefined") return "http://127.0.0.1:4312";
  const requestedPort = Number(new URLSearchParams(window.location.search).get("api"));
  const port = Number.isInteger(requestedPort) && requestedPort >= 1024 && requestedPort <= 65535 ? requestedPort : 4312;
  return `http://127.0.0.1:${port}`;
})();

type ViewId = "overview" | "models" | "analysis" | "documents" | "tools" | "live" | "scenes" | "lab";
type MetricId = "score30" | "score120" | "parameters" | "transitions" | "trainingHours";

type DocumentRecord = {
  id: string;
  path: string;
  title: string;
  category: string;
  category_label: string;
  content: string;
  excerpt: string;
  search_text: string;
  modified_at: number;
  word_count: number;
  is_evidence: boolean;
  is_history: boolean;
};

type ToolParameter = {
  id: string;
  label: string;
  type: "select" | "number" | "range" | "checkpoint" | "run" | "config" | "segmented";
  default: string | number | boolean;
  options?: Array<string | number>;
  min?: number;
  max?: number;
  step?: number;
  optional?: boolean;
};

type ProcessSnapshot = {
  pid: number;
  running: boolean;
  exit_code: number | null;
  started_at: number;
  url: string | null;
  command_preview: string;
  log_tail: string[];
  error_summary?: string | null;
  stopped_by_user?: boolean;
};

type ToolDefinition = {
  id: string;
  name: string;
  eyebrow: string;
  description: string;
  accent: string;
  kind: "service" | "task";
  primary_action: string;
  confirmation?: string;
  parameters: ToolParameter[];
  process: ProcessSnapshot | null;
};

type ToolChoices = {
  runs: Array<{ label: string; value: string }>;
  checkpoints: Array<{ label: string; value: string }>;
  configs: Array<{ label: string; value: string }>;
};

const NAVIGATION: Array<{ id: ViewId; label: string; hint: string; icon: LucideIcon }> = [
  { id: "overview", label: "总览", hint: "Overview", icon: LayoutDashboard },
  { id: "models", label: "模型图谱", hint: "Evidence", icon: BarChart3 },
  { id: "analysis", label: "数据分析", hint: "Analytics", icon: Database },
  { id: "documents", label: "文档知识库", hint: "Knowledge", icon: BookOpenText },
  { id: "tools", label: "工具中心", hint: "Launchpad", icon: Wrench },
  { id: "live", label: "实时训练", hint: "Telemetry", icon: Activity },
  { id: "scenes", label: "场景查看", hint: "Snapshots", icon: Images },
  { id: "lab", label: "场景实验室", hint: "Physics Lab", icon: FlaskConical },
];

const VIEW_IDS = new Set<ViewId>(NAVIGATION.map((item) => item.id));

const METRICS: Record<MetricId, { label: string; unit: string; title: string; subtitle: string }> = {
  score30: {
    label: "30 FPS 得分",
    unit: "分",
    title: "30 FPS物理下的最终策略表现",
    subtitle: "固定评估种子的平均原始分数 · 新旧物理身份必须分开解释",
  },
  score120: {
    label: "120 FPS 得分",
    unit: "分",
    title: "精确物理下的策略上限",
    subtitle: "固定评估种子的平均原始分数 · 点击柱体打开证据报告",
  },
  parameters: {
    label: "参数规模",
    unit: "M",
    title: "模型容量并不自动等于规划能力",
    subtitle: "在线模型可训练参数量 · 对照表达能力与实际得分",
  },
  transitions: {
    label: "训练规模",
    unit: "M",
    title: "被模型看过的决策边界",
    subtitle: "父轨迹与明确登记的额外旁路transition",
  },
  trainingHours: {
    label: "训练时长",
    unit: "h",
    title: "训练计算投入的近似比较",
    subtitle: "由稳定吞吐估算；128M使用冻结墙钟预算，非精确计费记录",
  },
};

const pageEnter = {
  opacity: [0.82, 1],
  y: [8, 0],
};

const pageTransition = { duration: 0.32, ease: [0.22, 1, 0.36, 1] };

function number(value: number | null | undefined, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

function modelValue(model: ModelRecord, metric: MetricId) {
  if (metric === "parameters") return model.parameters / 1_000_000;
  return model[metric];
}

function physicsLabel(model: ModelRecord) {
  return model.physics === "zero-velocity" ? "零初速度新物理" : "继承动量旧物理";
}

function gain(current: number, baseline: number) {
  return ((current / baseline) - 1) * 100;
}

const FEATURED_MODELS = FEATURED_MODEL_IDS
  .map((id) => MODELS.find((model) => model.id === id))
  .filter((model): model is ModelRecord => Boolean(model));

const BASELINE_128M = MODELS.find((model) => model.id === "baseline-128m") as ModelRecord;
const STRUCTURED_128M = MODELS.find((model) => model.id === "auxiliary-structured-128m") as ModelRecord;
const LATEST_MODEL = MODELS.find((model) => model.id === CURRENT_TRAINING.modelId) as ModelRecord;

function normalizeRelative(basePath: string, target: string) {
  const parts = basePath.split("/");
  parts.pop();
  for (const part of target.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") parts.pop();
    else parts.push(part);
  }
  return parts.join("/");
}

export function PortalShell() {
  const [activeView, setActiveView] = useState<ViewId>("overview");
  const [metric, setMetric] = useState<MetricId>("score120");
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<DocumentRecord | null>(null);
  const [documentCategory, setDocumentCategory] = useState("all");
  const [query, setQuery] = useState("");
  const [searchFocused, setSearchFocused] = useState(false);
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [choices, setChoices] = useState<ToolChoices>({ runs: [], checkpoints: [], configs: [] });
  const [selectedTool, setSelectedTool] = useState<ToolDefinition | null>(null);
  const [toolParams, setToolParams] = useState<Record<string, string | number | boolean>>({});
  const [toolError, setToolError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState(false);
  const [dashboard, setDashboard] = useState<DashboardStatus>({ available: false });
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);
  const documentRevisionRef = useRef("");

  const loadDocuments = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/documents`, { cache: "no-store" });
      if (!response.ok) throw new Error("文档索引不可用");
      const payload = (await response.json()) as { documents: DocumentRecord[]; revision: string };
      setDocuments(payload.documents);
      setSelectedDocument((current) => (
        (current ? payload.documents.find((item) => item.id === current.id) : null)
        ?? payload.documents.find((item) => item.path === "docs/README.md")
        ?? payload.documents[0]
      ));
      documentRevisionRef.current = payload.revision;
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }
  }, []);

  const loadTools = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/tools`, { cache: "no-store" });
      if (!response.ok) throw new Error("工具服务不可用");
      const payload = (await response.json()) as { tools: ToolDefinition[]; choices: ToolChoices };
      setTools(payload.tools);
      setChoices(payload.choices);
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }
  }, []);

  const loadDashboard = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/dashboard/status`, { cache: "no-store" });
      if (!response.ok) throw new Error("面板代理不可用");
      setDashboard((await response.json()) as DashboardStatus);
    } catch {
      setDashboard({ available: false });
    }
  }, []);

  const refreshDocumentsWhenChanged = useCallback(async () => {
    try {
      const response = await fetch(`${API}/api/documents/revision`, { cache: "no-store" });
      if (!response.ok) return;
      const payload = (await response.json()) as { revision: string };
      if (documentRevisionRef.current && documentRevisionRef.current !== payload.revision) {
        await loadDocuments();
      }
    } catch {
      // 网络抖动由既有在线状态和手动刷新入口表达，不清空当前阅读内容。
    }
  }, [loadDocuments]);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => {
      void loadDocuments();
      void loadTools();
      void loadDashboard();
    }, 0);
    const dashboardTimer = window.setInterval(loadDashboard, 5000);
    const toolTimer = window.setInterval(loadTools, 3500);
    const documentTimer = window.setInterval(refreshDocumentsWhenChanged, 3000);
    return () => {
      window.clearTimeout(initialLoad);
      window.clearInterval(dashboardTimer);
      window.clearInterval(toolTimer);
      window.clearInterval(documentTimer);
    };
  }, [loadDashboard, loadDocuments, loadTools, refreshDocumentsWhenChanged]);

  useEffect(() => {
    const syncRoute = () => {
      const route = window.location.hash.replace(/^#\/?/, "") as ViewId;
      if (VIEW_IDS.has(route)) setActiveView(route);
    };
    syncRoute();
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchRef.current?.focus();
      }
      if (event.key === "Escape") {
        setSearchFocused(false);
        setSelectedTool(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const searchResults = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase("zh-CN");
    if (!normalized) return [];
    const terms = normalized.split(/\s+/).filter(Boolean);
    return documents
      .map((doc) => {
        const title = doc.title.toLocaleLowerCase("zh-CN");
        const text = `${doc.path} ${doc.search_text}`.toLocaleLowerCase("zh-CN");
        const matched = terms.every((term) => title.includes(term) || text.includes(term));
        const score = terms.reduce((total, term) => total + (title.includes(term) ? 8 : 0) + (text.includes(term) ? 1 : 0), 0);
        return { doc, matched, score };
      })
      .filter((item) => item.matched)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
      .map((item) => item.doc);
  }, [documents, query]);

  const filteredDocuments = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase("zh-CN");
    return documents.filter((doc) => {
      const categoryMatch = documentCategory === "all" || doc.category === documentCategory;
      const queryMatch = !normalized || `${doc.title} ${doc.path} ${doc.search_text}`.toLocaleLowerCase("zh-CN").includes(normalized);
      return categoryMatch && queryMatch;
    });
  }, [documents, documentCategory, query]);

  const barOption = useMemo<EChartsOption>(() => {
    const definition = METRICS[metric];
    return {
      backgroundColor: "transparent",
      animationDuration: 900,
      animationDurationUpdate: 650,
      animationEasing: "cubicOut",
      grid: { left: 28, right: 22, top: 28, bottom: 90, containLabel: true },
      tooltip: {
        trigger: "item",
        backgroundColor: "rgba(242, 245, 249, .98)",
        borderColor: "rgba(75, 91, 116, .12)",
        padding: 14,
        extraCssText: "box-shadow: 10px 12px 28px rgba(110,122,142,.24); border-radius: 12px;",
        textStyle: { color: "#26354c" },
        formatter: (raw: unknown) => {
          const item = raw as { dataIndex: number; value: number };
          const model = MODELS[item.dataIndex];
          return `<div class="chart-tooltip"><strong>${model.name}</strong><span>${model.role} · ${model.commit}</span><b>${number(item.value, metric === "score30" || metric === "score120" ? 0 : 2)} ${definition.unit}</b><small>${physicsLabel(model)} · ${model.evidence}</small></div>`;
        },
      },
      xAxis: {
        type: "category",
        data: MODELS.map((model) => model.shortName),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: "rgba(79,95,119,.14)" } },
        axisLabel: { color: "#7d899b", fontSize: 10, interval: 0, rotate: 18, margin: 18 },
      },
      yAxis: {
        type: "value",
        name: definition.unit,
        nameTextStyle: { color: "#8b96a7", align: "right" },
        splitLine: { lineStyle: { color: "rgba(76,92,118,.09)" } },
        axisLabel: { color: "#8792a4" },
      },
      series: [
        {
          type: "bar",
          barMaxWidth: 56,
          data: MODELS.map((model) => ({
            value: modelValue(model, metric),
            itemStyle: {
              color: model.accent,
              borderRadius: [10, 10, 3, 3],
              shadowBlur: 22,
              shadowColor: `${model.accent}42`,
            },
          })),
          label: {
            show: true,
            position: "top",
            color: "#33425a",
            fontWeight: 700,
            formatter: (raw: unknown) => {
              const item = raw as { value: number };
              return number(item.value, metric === "score30" || metric === "score120" ? 0 : 2);
            },
          },
          emphasis: { scale: true, focus: "self" },
        },
      ],
    };
  }, [metric]);

  const scatterOption = useMemo<EChartsOption>(() => ({
    animationDuration: 1000,
    grid: { left: 58, right: 28, top: 34, bottom: 52 },
    tooltip: {
      trigger: "item",
      backgroundColor: "rgba(242,245,249,.98)",
      borderColor: "rgba(75,91,116,.12)",
      extraCssText: "box-shadow: 10px 12px 28px rgba(110,122,142,.24); border-radius: 12px;",
      textStyle: { color: "#26354c" },
      formatter: (raw: unknown) => {
        const item = raw as { data: [number, number, number, string, string, string] };
        const model = MODELS.find((candidate) => candidate.id === item.data[5]);
        return `<div class="chart-tooltip"><strong>${item.data[3]}</strong><span>${model ? physicsLabel(model) : "物理身份未知"} · ${item.data[0]}M transition</span><b>${number(item.data[1])} 分</b><small>${model?.budget ?? `${item.data[2].toFixed(2)}M 参数`}</small></div>`;
      },
    },
    legend: {
      top: 0,
      right: 8,
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { color: "#758298", fontSize: 10 },
    },
    xAxis: {
      type: "log", name: "训练 transition（M）", nameLocation: "middle", nameGap: 34,
      axisLabel: { color: "#8591a3" }, axisLine: { lineStyle: { color: "rgba(75,91,116,.14)" } },
      splitLine: { lineStyle: { color: "rgba(75,91,116,.08)" } },
    },
    yAxis: {
      type: "value", name: "120 FPS 平均分", min: 2800,
      axisLabel: { color: "#8591a3" }, splitLine: { lineStyle: { color: "rgba(75,91,116,.08)" } },
    },
    series: [
      {
        name: "继承动量旧物理",
        type: "scatter",
        symbol: "circle",
        symbolSize: (raw: unknown) => 18 + ((raw as [number, number, number])[2] * 16),
        data: MODELS.filter((model) => model.physics === "inherited-momentum")
          .map((model) => [model.transitions, model.score120, model.parameters / 1_000_000, model.shortName, model.accent, model.id]),
        itemStyle: {
          color: (raw: unknown) => (raw as { data: [number, number, number, string, string] }).data[4],
          shadowBlur: 24,
          shadowColor: "rgba(61,113,190,.2)",
        },
        label: { show: true, formatter: (raw: unknown) => (raw as { data: [number, number, number, string] }).data[3], position: "top", color: "#58677e", fontSize: 10 },
      },
      {
        name: "零初速度新物理",
        type: "scatter",
        symbol: "diamond",
        symbolSize: (raw: unknown) => 20 + ((raw as [number, number, number])[2] * 16),
        data: MODELS.filter((model) => model.physics === "zero-velocity")
          .map((model) => [model.transitions, model.score120, model.parameters / 1_000_000, model.shortName, model.accent, model.id]),
        itemStyle: {
          color: (raw: unknown) => (raw as { data: [number, number, number, string, string] }).data[4],
          shadowBlur: 28,
          shadowColor: "rgba(135,87,217,.26)",
        },
        label: { show: true, formatter: (raw: unknown) => (raw as { data: [number, number, number, string] }).data[3], position: "top", color: "#7357d9", fontSize: 10, fontWeight: 700 },
      },
    ],
  }), []);

  const openDocument = useCallback((doc: DocumentRecord) => {
    setSelectedDocument(doc);
    setActiveView("documents");
    setSearchFocused(false);
    setSidebarOpen(false);
  }, []);

  const openReport = useCallback((model: ModelRecord) => {
    const doc = documents.find((item) => item.path === model.report);
    if (doc) openDocument(doc);
  }, [documents, openDocument]);

  const navigate = (id: ViewId) => {
    setActiveView(id);
    setSidebarOpen(false);
    window.history.replaceState(null, "", `#${id}`);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const openToolSettings = (tool: ToolDefinition) => {
    setSelectedTool(tool);
    setToolError(null);
    setToolParams(Object.fromEntries(tool.parameters.map((parameter) => [parameter.id, parameter.default])));
  };

  const updateToolProcess = (toolId: string, process: ProcessSnapshot) => {
    setTools((current) => current.map((tool) => tool.id === toolId ? { ...tool, process } : tool));
  };

  const startTool = async () => {
    if (!selectedTool) return;
    if (selectedTool.confirmation && !window.confirm(selectedTool.confirmation)) return;
    setToolError(null);
    try {
      const response = await fetch(`${API}/api/tools/${selectedTool.id}/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params: toolParams }),
      });
      const payload = (await response.json()) as { process?: ProcessSnapshot; error?: string };
      if (!response.ok || !payload.process) throw new Error(payload.error ?? "启动失败");
      updateToolProcess(selectedTool.id, payload.process);
      const startedToolId = selectedTool.id;
      setSelectedTool(null);
      if (startedToolId === "scenario_lab") navigate("lab");
      if (startedToolId === "training_dashboard") navigate("live");
    } catch (error) {
      setToolError(error instanceof Error ? error.message : "启动失败");
    }
  };

  const stopTool = async (tool: ToolDefinition) => {
    try {
      const response = await fetch(`${API}/api/tools/${tool.id}/stop`, { method: "POST" });
      const payload = (await response.json()) as { process?: ProcessSnapshot; error?: string };
      if (!response.ok || !payload.process) throw new Error(payload.error ?? "停止失败");
      updateToolProcess(tool.id, payload.process);
    } catch (error) {
      setToolError(error instanceof Error ? error.message : "停止失败");
    }
  };

  return (
    <div className="portal-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="noise" />

      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <button className="brand" onClick={() => navigate("overview")} aria-label="返回总览">
          <span className="brand-mark"><span /></span>
          <span className="brand-copy"><b>XIGUA</b><small>ATLAS / 01</small></span>
        </button>

        <nav className="primary-nav" aria-label="主导航">
          {NAVIGATION.map(({ id, label, hint, icon: Icon }) => (
            <button key={id} className={activeView === id ? "active" : ""} onClick={() => navigate(id)}>
              <Icon size={18} strokeWidth={1.8} />
              <span><b>{label}</b><small>{hint}</small></span>
              {id === "live" && <i className={dashboard.available ? "live-dot on" : "live-dot"} />}
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">
          <div className="physics-identity">
            <Layers3 size={16} />
            <span><small>当前物理身份</small><b>新水果速度归零</b></span>
          </div>
          <div className={`backend-state ${backendOnline ? "online" : ""}`}>
            <span /> {backendOnline ? "控制服务在线" : "仅预览模式"}
          </div>
        </div>
      </aside>

      <header className="topbar">
        <button className="menu-button" onClick={() => setSidebarOpen((value) => !value)} aria-label="打开导航"><Menu size={20} /></button>
        <div className="global-search">
          <Search size={17} />
          <input
            ref={searchRef}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onFocus={() => setSearchFocused(true)}
            placeholder="搜索模型、结论、文档或提交…"
            aria-label="搜索全部文档"
          />
          {query && <button onClick={() => setQuery("")} aria-label="清除搜索"><X size={15} /></button>}
          <kbd><Command size={12} /> K</kbd>
        </div>
        <div className="topbar-actions">
          <button className="icon-button" onClick={() => { void loadDocuments(); void loadTools(); }} aria-label="刷新数据"><RefreshCw size={17} /></button>
          <button className="training-pill" onClick={() => navigate(dashboard.available ? "live" : "models")}>
            <span className={dashboard.available ? "pulse" : ""} />
            <span><small>{dashboard.available ? "CURRENT RUN" : "LATEST MODEL"}</small><b>{dashboard.available ? "训练遥测在线" : "120FPS迁移 · 7,568.18"}</b></span>
            <ChevronRight size={16} />
          </button>
        </div>

        <AnimatePresence>
          {searchFocused && query && (
            <motion.div className="search-popover" initial={{ opacity: 0, y: -8, scale: .98 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -8, scale: .98 }}>
              <div className="search-popover-head"><span>全文检索</span><small>{searchResults.length} 条最佳匹配</small></div>
              {searchResults.length ? searchResults.map((doc) => (
                <button key={doc.id} onClick={() => openDocument(doc)}>
                  <FileText size={16} />
                  <span><b>{doc.title}</b><small>{doc.category_label} · {doc.path}</small></span>
                  <ChevronRight size={15} />
                </button>
              )) : <div className="empty-search">没有找到包含“{query}”的文档</div>}
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      <main className="main-stage">
        <AnimatePresence mode="wait">
          {activeView === "overview" && (
            <motion.section key="overview" className="page overview-page" animate={pageEnter} exit={{ opacity: 0, y: -10 }} transition={pageTransition}>
              <div className="hero">
                <div className="hero-copy">
                  <div className="eyebrow"><Sparkles size={14} /> MODEL INTELLIGENCE / LIVE KNOWLEDGE</div>
                  <h1>让每一次训练<br /><span>成为可读的证据。</span></h1>
                  <p>统一浏览模型设计、正式评估、训练遥测与项目工具。所有图表保留代码身份、物理规则和证据边界。</p>
                  <div className="hero-actions">
                    <button className="primary-button" onClick={() => navigate("models")}>探索模型图谱 <ArrowUpRight size={17} /></button>
                    <button className="ghost-button" onClick={() => navigate("documents")}>浏览 {documents.length || 75}+ 篇文档</button>
                  </div>
                </div>
                <div className="hero-orbit" aria-hidden="true">
                  <div className="orbit-ring ring-one" />
                  <div className="orbit-ring ring-two" />
                  <div className="orbit-core"><span>7,568</span><small>120 FPS BEST</small></div>
                  <div className="orbit-node node-a">156M</div>
                  <div className="orbit-node node-b">1.22M</div>
                  <div className="orbit-node node-c">4096×2</div>
                </div>
              </div>

              <div className="stat-grid">
                <StatCard icon={Gauge} label="新物理120FPS最佳" value="7,568.18" note="结构化辅助迁移 · 相对来源 +15.51%" tone="cyan" />
                <StatCard icon={Sparkles} label="新物理30FPS最佳" value="8,006.41" note="同一迁移模型 · 4096局" tone="green" />
                <StatCard icon={Database} label="完整训练谱系" value="156M" note="128M父 + 12M旁路 + 16M适应" tone="violet" />
                <StatCard icon={FileText} label="可检索知识" value={String(documents.length || "—")} note="设计、评估与工程记录" tone="amber" />
              </div>

              <div className="dashboard-grid">
                <section className="panel chart-stage">
                  <div className="panel-head chart-head">
                    <div>
                      <span className="panel-kicker">MODEL COMPARISON</span>
                      <h2>{METRICS[metric].title}</h2>
                      <p>{METRICS[metric].subtitle}</p>
                    </div>
                    <div className="metric-switcher">
                      {(Object.keys(METRICS) as MetricId[]).map((id) => (
                        <button key={id} className={metric === id ? "active" : ""} onClick={() => setMetric(id)}>{METRICS[id].label}</button>
                      ))}
                    </div>
                  </div>
                  <EChart option={barOption} className="hero-chart" onClick={(payload) => {
                    if (payload.dataIndex !== undefined) openReport(MODELS[payload.dataIndex]);
                  }} />
                  <div className="chart-footnote"><AlertCircle size={13} /> 图表同时包含继承动量旧物理和零初速度新物理；点击柱体查看身份，跨物理分数只表示已保存模型的原始效果排序。</div>
                </section>

                <aside className="insight-stack">
                  <section className="panel run-card">
                    <div className="run-card-top"><span className="live-badge completed"><i /> LATEST COMPLETED</span><CheckCircle2 size={18} /></div>
                    <h3>{CURRENT_TRAINING.name}</h3>
                    <p>{CURRENT_TRAINING.budget}</p>
                    <div className="run-progress"><span style={{ width: "100%" }} /></div>
                    <div className="run-metrics">
                      <span><small>ADAPTATION</small><b>{CURRENT_TRAINING.stageTransitions}M</b></span>
                      <span><small>120 FPS SCORE</small><b>{number(CURRENT_TRAINING.score120, 2)}</b></span>
                    </div>
                    <button onClick={() => openReport(LATEST_MODEL)}>打开最终评估报告 <ChevronRight size={16} /></button>
                  </section>
                  <section className="panel evidence-card">
                    <span className="panel-kicker">EVIDENCE RULE</span>
                    <h3>图表不是结论的替代品</h3>
                    <p>同一张图同时展示评估局数、代码提交、训练预算和物理身份。点击数据点即可回到原始模型报告。</p>
                    <div className="evidence-tags"><span>事实</span><span>观察</span><span>假设</span><span>决策</span></div>
                  </section>
                </aside>
              </div>
            </motion.section>
          )}

          {activeView === "models" && (
            <motion.section key="models" className="page" animate={pageEnter} exit={{ opacity: 0 }} transition={pageTransition}>
              <PageHeader eyebrow="MODEL EVIDENCE MAP" title="模型性能，不只看一个最高分。" description="同时审视训练规模、模型容量、双帧率得分、L11事件与物理身份。所有点均可追溯到正式评估报告。" />
              <section className="panel latest-model-evidence">
                <div className="panel-head latest-evidence-head">
                  <div>
                    <span className="panel-kicker">LATEST TRAINING EVIDENCE</span>
                    <h2>旧基线 → 结构化旁路 → 120 FPS迁移</h2>
                    <p>每个数值来自最终4096局/FPS评估；旧物理与新物理并排展示，但不伪装成严格算法消融。</p>
                  </div>
                  <div className="evidence-seed"><CheckCircle2 size={14} /> seed base 42,000,000</div>
                </div>
                <div className="featured-model-grid">
                  {FEATURED_MODELS.map((model) => (
                    <button key={model.id} className={`featured-model ${model.id === LATEST_MODEL.id ? "latest" : ""}`} onClick={() => openReport(model)}>
                      <div className="featured-model-top">
                        <span className={`physics-chip ${model.physics === "zero-velocity" ? "new" : "legacy"}`}>{physicsLabel(model)}</span>
                        <ArrowUpRight size={16} />
                      </div>
                      <h3>{model.name}</h3>
                      <p>{model.budget}</p>
                      <div className="featured-score-pair">
                        <span><small>30 FPS</small><b>{number(model.score30, 2)}</b></span>
                        <span><small>120 FPS</small><b>{number(model.score120, 2)}</b></span>
                      </div>
                      <div className="featured-detail-grid">
                        <span><small>L11消除 30 / 120</small><b>{number(model.l11Removed30, 2)}% / {number(model.l11Removed120, 2)}%</b></span>
                        <span><small>危险投放 30 / 120</small><b>{number(model.danger30, 2)}% / {number(model.danger120, 2)}%</b></span>
                        <span><small>中位分 30 / 120</small><b>{number(model.median30, 1)} / {number(model.median120, 1)}</b></span>
                        <span><small>训练谱系</small><b>{number(model.transitions, 3)}M</b></span>
                      </div>
                      <footer>{model.role}<span>{model.commit}</span></footer>
                    </button>
                  ))}
                </div>
                <div className="comparison-summary">
                  <div><span>相对旧基线的原始得分</span><b>+{number(gain(LATEST_MODEL.score30, BASELINE_128M.score30), 2)}% / +{number(gain(LATEST_MODEL.score120, BASELINE_128M.score120), 2)}%</b><small>30 / 120 FPS · 跨物理，仅作最终效果排序</small></div>
                  <div><span>120 FPS适应的同物理净增益</span><b>+{number(gain(LATEST_MODEL.score30, STRUCTURED_128M.score30), 2)}% / +{number(gain(LATEST_MODEL.score120, STRUCTURED_128M.score120), 2)}%</b><small>30 / 120 FPS · 相同模型谱系与评估seed</small></div>
                  <div><span>适应后L11消除提升</span><b>+{number((LATEST_MODEL.l11Removed30 ?? 0) - (STRUCTURED_128M.l11Removed30 ?? 0), 2)} / +{number((LATEST_MODEL.l11Removed120 ?? 0) - (STRUCTURED_128M.l11Removed120 ?? 0), 2)} pp</b><small>30 / 120 FPS · 生成率与消除率分开统计</small></div>
                </div>
              </section>
              <div className="model-layout">
                <section className="panel scatter-panel">
                  <div className="panel-head"><div><span className="panel-kicker">SCALE × PERFORMANCE</span><h2>训练投入与120 FPS得分</h2><p>气泡大小代表参数量；圆形为旧物理，菱形为零初速度新物理，迁移模型横轴使用完整训练谱系。</p></div></div>
                  <EChart option={scatterOption} className="scatter-chart" />
                </section>
                <section className="panel leaderboard">
                  <div className="panel-head"><div><span className="panel-kicker">REPRESENTATIVE MODELS</span><h2>代表模型</h2></div></div>
                  {[...MODELS].sort((a, b) => b.score120 - a.score120).map((model, index) => (
                    <button key={model.id} onClick={() => openReport(model)}>
                      <span className="rank">{String(index + 1).padStart(2, "0")}</span>
                      <i style={{ background: model.accent }} />
                      <span className="leader-copy"><b>{model.shortName}</b><small>{model.role} · {model.transitions}M · {model.physics === "zero-velocity" ? "零速" : "动量"}</small></span>
                      <strong>{number(model.score120)}<small>分</small></strong>
                      <ChevronRight size={15} />
                    </button>
                  ))}
                </section>
              </div>
              <section className="model-cards">
                {MODELS.map((model) => (
                  <motion.button key={model.id} whileHover={{ y: -6 }} onClick={() => openReport(model)} className="model-card">
                    <div className="model-card-head"><span style={{ color: model.accent }}>{model.role}</span><span className={`physics-chip ${model.physics === "zero-velocity" ? "new" : "legacy"}`}>{model.physics === "zero-velocity" ? "零速" : "动量"}</span><ArrowUpRight size={17} /></div>
                    <h3>{model.name}</h3>
                    <strong>{number(model.score120)}<small>120 FPS</small></strong>
                    <div className="model-card-meta"><span>{(model.parameters / 1_000_000).toFixed(2)}M 参数</span><span>{model.transitions}M 谱系</span></div>
                    {model.l11Removed120 !== undefined && (
                      <div className="model-card-evidence"><span>L11消除 <b>{number(model.l11Removed120, 2)}%</b></span><span>危险投放 <b>{number(model.danger120, 2)}%</b></span></div>
                    )}
                  </motion.button>
                ))}
              </section>
            </motion.section>
          )}

          {activeView === "documents" && (
            <motion.section key="documents" className="documents-page" animate={{ opacity: [0.82, 1] }} exit={{ opacity: 0 }} transition={pageTransition}>
              <aside className="document-browser">
                <div className="document-browser-head">
                  <span className="panel-kicker">KNOWLEDGE INDEX</span>
                  <h2>文档知识库</h2>
                  <p>{filteredDocuments.length} / {documents.length} 篇</p>
                </div>
                <div className="category-tabs">
                  {[['all', '全部'], ['model', '设计'], ['evaluations', '评估'], ['codex', '记录'], ['guide', '指南']].map(([id, label]) => (
                    <button key={id} className={documentCategory === id ? "active" : ""} onClick={() => setDocumentCategory(id)}>{label}</button>
                  ))}
                </div>
                <div className="document-list">
                  {filteredDocuments.map((doc) => (
                    <button key={doc.id} className={selectedDocument?.id === doc.id ? "active" : ""} onClick={() => setSelectedDocument(doc)}>
                      <span className={`doc-type ${doc.category}`}><FileText size={14} /></span>
                      <span><b>{doc.title}</b><small>{doc.category_label} · {Math.max(1, Math.round(doc.word_count / 1000))}k 字符</small></span>
                    </button>
                  ))}
                </div>
              </aside>
              <article className="document-reader">
                {selectedDocument ? (
                  <>
                    <div className="reader-meta">
                      <span className={`status-chip ${selectedDocument.category}`}>{selectedDocument.category_label}</span>
                      <span>{selectedDocument.path}</span>
                      <span>更新于 {new Date(selectedDocument.modified_at * 1000).toLocaleDateString("zh-CN")}</span>
                    </div>
                    <div className="markdown-body">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeHighlight]}
                        components={{
                          a: ({ href = "", children, ...props }) => {
                            const normalized = href.endsWith('.md') ? normalizeRelative(selectedDocument.path, href.split('#')[0]) : '';
                            const target = documents.find((doc) => doc.path === normalized);
                            return target ? <button className="markdown-link" onClick={() => setSelectedDocument(target)}>{children}</button> : <a href={href} target={href.startsWith('http') ? '_blank' : undefined} rel="noreferrer" {...props}>{children}</a>;
                          },
                          img: ({ src = "", alt = "" }) => {
                            const asset = src.startsWith('http') || src.startsWith('data:') ? src : `${API}/api/file?path=${encodeURIComponent(normalizeRelative(selectedDocument.path, src))}`;
                            // Markdown图片来自频繁变化的本地文档，尺寸未知，不能使用构建期图像优化器。
                            // eslint-disable-next-line @next/next/no-img-element
                            return <img src={asset} alt={alt} loading="lazy" />;
                          },
                          table: ({ children }) => <div className="markdown-table-wrap"><table>{children}</table></div>,
                        }}
                      >{selectedDocument.content}</ReactMarkdown>
                    </div>
                  </>
                ) : (
                  <div className="reader-empty"><BookOpenText size={34} /><h2>等待文档索引</h2><p>请确认本地门户控制服务已经启动。</p></div>
                )}
              </article>
            </motion.section>
          )}

          {activeView === "analysis" && (
            <motion.section key="analysis" className="page analysis-page" animate={pageEnter} exit={{ opacity: 0, y: -8 }} transition={pageTransition}>
              <PageHeader eyebrow="STATISTICAL ANALYSIS" title="让统计关系可以被直接探索。" description="从表格追溯到条件概率、时间尺度和双因素交互；当前默认读取 SAB-128 Merge Potential 正式数据，后续统计可沿用同一数据集契约。" />
              <Suspense fallback={<WorkspaceLoading />}><AnalysisWorkspace apiBase={API} /></Suspense>
            </motion.section>
          )}

          {activeView === "tools" && (
            <motion.section key="tools" className="page" animate={pageEnter} exit={{ opacity: 0 }} transition={pageTransition}>
              <PageHeader eyebrow="LOCAL TOOL LAUNCHPAD" title="把复杂命令，收进一个按钮。" description="白名单工具、参数校验、进程状态与实时日志都在同一处完成。门户不会执行任意命令，也不会保存服务器密码。" />
              {!backendOnline && <div className="offline-banner"><AlertCircle size={18} /><span><b>控制服务未连接</b>当前可以预览界面，但需要通过项目门户启动器打开后才能执行工具。</span></div>}
              <div className="tool-grid">
                {tools.map((tool, index) => {
                  const running = Boolean(tool.process?.running);
                  return (
                    <motion.article key={tool.id} className={`tool-card accent-${tool.accent}`} initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * .08 }} whileHover={{ y: -5 }}>
                      <div className="tool-card-top"><span>{tool.eyebrow}</span><i className={running ? "running" : ""}>{running ? "RUNNING" : tool.kind.toUpperCase()}</i></div>
                      <div className="tool-icon"><ToolIcon id={tool.id} /></div>
                      <h3>{tool.name}</h3>
                      <p>{tool.description}</p>
                      {tool.process && (
                        <div className="process-strip">
                          <span className={running ? "on" : ""} />
                          <b>{running ? `PID ${tool.process.pid}` : tool.process.stopped_by_user ? "已手动停止" : tool.process.exit_code === 0 ? "最近执行完成" : `已退出 ${tool.process.exit_code ?? ''}`}</b>
                          <small>{tool.process.started_at ? new Date(tool.process.started_at * 1000).toLocaleTimeString('zh-CN', { hour12: false }) : ''}</small>
                        </div>
                      )}
                      <div className="tool-actions">
                        {running ? (
                          <>
                            {tool.process?.url && (
                              <button className="primary-button compact" onClick={() => {
                                if (tool.id === "scenario_lab") navigate("lab");
                                else if (tool.id === "training_dashboard") navigate("live");
                                else window.open(tool.process?.url ?? "", "_blank");
                              }}>打开 {tool.id === "scenario_lab" || tool.id === "training_dashboard" ? <ChevronRight size={15} /> : <ExternalLink size={15} />}</button>
                            )}
                            <button className="danger-button" onClick={() => void stopTool(tool)}><Square size={14} /> 停止</button>
                          </>
                        ) : <button className="primary-button compact" onClick={() => openToolSettings(tool)}><Play size={15} /> {tool.primary_action}</button>}
                        <button className="icon-button" onClick={() => openToolSettings(tool)} aria-label={`${tool.name}参数`}><Settings2 size={16} /></button>
                      </div>
                      {tool.process && !running && !tool.process.stopped_by_user && tool.process.exit_code !== 0 && tool.process.error_summary ? (
                        <div className="tool-process-error"><AlertCircle size={14} /><span><b>启动失败</b>{tool.process.error_summary}</span></div>
                      ) : null}
                      {tool.process?.log_tail?.length ? <details className="tool-log"><summary>最近日志</summary><pre>{tool.process.log_tail.join('\n')}</pre></details> : null}
                    </motion.article>
                  );
                })}
              </div>
              {!tools.length && <div className="tool-empty"><Server size={34} /><h3>工具清单等待控制服务</h3><p>使用项目门户启动器后，这里会自动出现已登记工具。</p></div>}
            </motion.section>
          )}

          {activeView === "live" && (
            <motion.section key="live" className="page live-page" animate={pageEnter} exit={{ opacity: 0 }} transition={pageTransition}>
              <PageHeader eyebrow="LIVE TRAINING TELEMETRY" title="训练发生时，证据也在生长。" description="云端遥测、训练队列、资源曲线和细分损失已经原生进入项目门户；所有高密度信息按诊断语义折叠。" />
              <Suspense fallback={<WorkspaceLoading />}><TrainingWorkspace dashboard={dashboard} onRefresh={() => void loadDashboard()} onOpenTools={() => navigate("tools")} /></Suspense>
            </motion.section>
          )}

          {activeView === "lab" && (
            <motion.section key="lab" className="page lab-page" animate={pageEnter} exit={{ opacity: 0 }} transition={pageTransition}>
              <PageHeader eyebrow="INTERACTIVE PHYSICS LAB" title="让预测直接落在物理场景里。" description="实时场景编辑、21动作预测、辅助效果对照与可视化图层共用一套门户布局；低频设置集中在右侧抽屉。" />
              <Suspense fallback={<WorkspaceLoading />}><ScenarioWorkspace tool={tools.find((tool) => tool.id === "scenario_lab")} onConfigure={() => {
                const tool = tools.find((item) => item.id === "scenario_lab");
                if (tool) openToolSettings(tool);
                else navigate("tools");
              }} /></Suspense>
            </motion.section>
          )}
          {activeView === "scenes" && (
            <motion.section key="scenes" className="page scene-viewer-page" animate={pageEnter} exit={{ opacity: 0 }} transition={pageTransition}>
              <Suspense fallback={<WorkspaceLoading />}><SceneViewerWorkspace /></Suspense>
            </motion.section>
          )}
        </AnimatePresence>
      </main>

      <AnimatePresence>
        {selectedTool && (
          <motion.div className="modal-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onMouseDown={(event) => { if (event.target === event.currentTarget) setSelectedTool(null); }}>
            <motion.section className="tool-modal" initial={{ opacity: 0, y: 24, scale: .97 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: 18, scale: .97 }} transition={{ type: 'spring', damping: 28, stiffness: 320 }}>
              <div className="modal-head"><div><span>{selectedTool.eyebrow}</span><h2>{selectedTool.name}</h2><p>{selectedTool.description}</p></div><button className="icon-button" onClick={() => setSelectedTool(null)}><X size={18} /></button></div>
              <div className="parameter-list">
                {selectedTool.parameters.map((parameter) => (
                  <ParameterControl key={parameter.id} parameter={parameter} value={toolParams[parameter.id] ?? parameter.default} choices={choices} onChange={(value) => setToolParams((current) => ({ ...current, [parameter.id]: value }))} />
                ))}
              </div>
              <div className="command-safety"><CheckCircle2 size={17} /><span><b>白名单命令模板</b>路径、枚举和数值范围将在后端再次校验；页面不能提交任意shell内容。</span></div>
              {toolError && <div className="modal-error"><AlertCircle size={16} />{toolError}</div>}
              <div className="modal-actions"><button className="ghost-button" onClick={() => setSelectedTool(null)}>取消</button><button className="primary-button" onClick={() => void startTool()}><Play size={16} />{selectedTool.primary_action}</button></div>
            </motion.section>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function WorkspaceLoading() {
  return <div className="workspace-loading"><RefreshCw size={18} /><span>正在加载工作区…</span></div>;
}

function StatCard({ icon: Icon, label, value, note, tone }: { icon: LucideIcon; label: string; value: string; note: string; tone: string }) {
  return <motion.article className={`stat-card tone-${tone}`} whileHover={{ y: -4 }}><span className="stat-icon"><Icon size={18} /></span><div><small>{label}</small><strong>{value}</strong><p>{note}</p></div><ArrowUpRight size={16} className="stat-arrow" /></motion.article>;
}

function PageHeader({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return <header className="page-header"><span className="panel-kicker">{eyebrow}</span><h1>{title}</h1><p>{description}</p></header>;
}

function ToolIcon({ id }: { id: string }) {
  if (id === 'scenario_lab') return <Sparkles size={25} />;
  if (id === 'training_dashboard') return <Activity size={25} />;
  if (id === 'model_viewer') return <Gauge size={25} />;
  return <Cpu size={25} />;
}

function ParameterControl({ parameter, value, choices, onChange }: { parameter: ToolParameter; value: string | number | boolean; choices: ToolChoices; onChange: (value: string | number | boolean) => void }) {
  const dynamicOptions = parameter.type === 'checkpoint' ? choices.checkpoints : parameter.type === 'run' ? choices.runs : parameter.type === 'config' ? choices.configs : null;
  const options = dynamicOptions ?? parameter.options?.map((option) => ({ label: String(option), value: String(option) })) ?? [];
  const numeric = parameter.type === 'number' || parameter.type === 'range';
  return (
    <label className={`parameter-control type-${parameter.type}`}>
      <span><b>{parameter.label}</b>{parameter.optional && <small>可选</small>}</span>
      {parameter.type === 'segmented' ? (
        <div className="segmented-control">{options.map((option) => <button type="button" key={option.value} className={String(value) === option.value ? 'active' : ''} onClick={() => onChange(Number(option.value))}>{option.label}</button>)}</div>
      ) : options.length || ['checkpoint', 'run', 'config', 'select'].includes(parameter.type) ? (
        <select value={String(value)} onChange={(event) => onChange(event.target.value)}>
          {(parameter.optional || !options.length) && <option value="">{options.length ? '不加载模型' : '暂无可用项目'}</option>}
          {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
      ) : numeric ? (
        <div className="numeric-control">
          {parameter.type === 'range' && <input type="range" min={parameter.min} max={parameter.max} step={parameter.step} value={Number(value)} onChange={(event) => onChange(Number(event.target.value))} />}
          <input type="number" min={parameter.min} max={parameter.max} step={parameter.step} value={Number(value)} onChange={(event) => onChange(Number(event.target.value))} />
        </div>
      ) : null}
    </label>
  );
}
