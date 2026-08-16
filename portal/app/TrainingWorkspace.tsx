"use client";

import type { EChartsOption } from "echarts";
import { motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Database,
  Gauge,
  ListOrdered,
  RefreshCw,
  Server,
  Timer,
  type LucideIcon,
} from "lucide-react";
import { useMemo, useState } from "react";
import { EChart } from "./EChart";

type Scalar = number | string | boolean | null | undefined;
type MetricRow = Record<string, unknown>;

type MergePotentialRun = {
  id: string;
  name: string;
  run_dir: string;
  status: string;
  stale?: boolean;
  target_episodes?: number;
  completed_episodes?: number;
  progress_fraction?: number;
  transitions?: number;
  elapsed_seconds?: number;
  env_steps_per_second?: number;
  parallel_envs?: number;
  max_drops?: number;
  physics_fps?: number;
  drop_fast_forward?: boolean;
  snapshot_rows?: number;
  merge_source_rows?: number;
  episode_rows?: number;
  peak_cuda_allocated_bytes?: number;
  checkpoint_name?: string | null;
  checkpoint_sha256?: string | null;
  cuda_device_name?: string | null;
  analysis_ready?: boolean;
  failure?: string | null;
};

type MergePotentialStatus = {
  available: boolean;
  current?: MergePotentialRun | null;
  runs?: MergePotentialRun[];
};

export type QueueItem = {
  id: string;
  name: string;
  status: string;
  position?: number | null;
  run_dir?: string | null;
  config?: string | null;
  training_physics_fps?: number | null;
  planned_transitions?: number | null;
  source_run?: string | null;
  depends_on?: string | null;
  message?: string | null;
  enqueued_at?: number | null;
  started_at?: number | null;
  completed_at?: number | null;
  transitions?: number | null;
  progress_fraction?: number | null;
};

export type DashboardPayload = {
  timestamp?: number;
  training?: MetricRow;
  resources?: MetricRow;
  history?: MetricRow[];
  resource_history?: MetricRow[];
  events?: Array<Record<string, Scalar>>;
  plots?: Record<string, Record<string, Scalar>>;
  queue?: {
    updated_at?: number;
    items?: QueueItem[];
    counts?: Record<string, number>;
  };
  merge_potential?: MergePotentialStatus;
};

export type DashboardStatus = {
  available: boolean;
  payload?: DashboardPayload | null;
};

type Props = {
  dashboard: DashboardStatus;
  onRefresh: () => void;
  onOpenTools: () => void;
};

type ChartId = "score" | "loss" | "aux" | "throughput" | "events" | "resources";

const SERIES_COLORS = ["#3478f6", "#7c5ce5", "#18a67e", "#e68a2e", "#db4f7d", "#65758b"];
const PHASE_LABELS: Record<string, string> = {
  initializing: "初始化",
  warmup: "Replay预热",
  training: "训练中",
  evaluation: "评估中",
  completed: "正常完成",
  stopped: "安全停止",
  failed: "异常结束",
};
const QUEUE_LABELS: Record<string, string> = {
  queued: "排队等待",
  waiting: "等待依赖",
  preflight: "执行预检",
  running: "正在训练",
  evaluating: "最终评估",
  completed: "已经完成",
  failed: "执行失败",
  cancelled: "已取消",
};

const CHARTS: Array<{ id: ChartId; label: string; description: string }> = [
  { id: "score", label: "策略效果", description: "训练窗口与双帧率评估" },
  { id: "loss", label: "价值学习", description: "总损失、DQN与TD误差" },
  { id: "aux", label: "辅助监督", description: "五类辅助效果和旁路损失" },
  { id: "throughput", label: "训练吞吐", description: "环境、学习和旁路速度" },
  { id: "events", label: "高等级合成", description: "L7～L11评估生成密度" },
  { id: "resources", label: "GPU资源", description: "利用率和显存时间曲线" },
];

const DETAIL_GROUPS = [
  {
    id: "progress",
    title: "训练进度与采样",
    hint: "投放、更新、Replay和探索分支",
    metrics: [
      ["transitions", "累计投放", "count"], ["total_transitions", "计划投放", "count"],
      ["branch_transitions", "旁路主动投放", "count"], ["updates", "模型更新", "count"],
      ["episodes", "完成局数", "count"], ["active_envs", "活跃环境", "count"],
      ["epsilon", "探索率", "ratio"], ["replay_size", "Replay占用", "count"],
      ["branch_replay_size", "旁路Replay", "count"], ["eta_seconds", "预计剩余", "duration"],
    ],
  },
  {
    id: "learning",
    title: "学习状态",
    hint: "主价值、辅助监督和策略分歧",
    metrics: [
      ["loss", "总损失", "decimal"], ["dqn_loss", "DQN损失", "decimal"],
      ["aux_loss_total", "辅助效果损失", "decimal"], ["mean_reward", "平均奖励", "decimal"],
      ["mean_q", "平均Q值", "decimal"], ["mean_target", "平均TD目标", "decimal"],
      ["mean_abs_td_error", "绝对TD误差", "decimal"], ["policy_disagreement", "策略头分歧", "decimal"],
      ["active_selected_rank_correlation", "主动动作排名相关性", "decimal"], ["grad_norm", "梯度范数", "decimal"],
      ["last_fast_eval_score", "最近30 FPS评估", "score"], ["last_accurate_eval_score", "最近120 FPS评估", "score"],
    ],
  },
  {
    id: "performance",
    title: "性能与资源",
    hint: "吞吐、GPU、显存和阶段耗时",
    metrics: [
      ["env_steps_per_second", "投放速度", "rate"], ["updates_per_second", "更新速度", "rate"],
      ["learner_samples_per_second", "学习样本速度", "rate"], ["branch_steps_per_second", "旁路速度", "rate"],
      ["gpu_utilization", "GPU使用率", "percent"], ["gpu_memory_used_mb", "GPU显存", "memory"],
      ["gpu_temperature", "GPU温度", "temperature"], ["gpu_power_watts", "GPU功耗", "power"],
      ["cpu_utilization", "CPU使用率", "percent"], ["process_rss_mb", "训练进程内存", "memory"],
      ["physics_seconds", "物理耗时窗口", "seconds"], ["learner_seconds", "学习耗时窗口", "seconds"],
    ],
  },
] as const;

function numeric(value: unknown) {
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function count(value: unknown) {
  const number = numeric(value);
  return number === null ? "—" : number.toLocaleString("zh-CN", { maximumFractionDigits: 0 });
}

function decimal(value: unknown, digits = 4) {
  const number = numeric(value);
  if (number === null) return "—";
  if (number !== 0 && Math.abs(number) < 0.0001) return number.toExponential(2);
  return number.toLocaleString("zh-CN", { maximumFractionDigits: digits });
}

function percent(value: unknown, isRatio = false) {
  const number = numeric(value);
  return number === null ? "—" : `${(number * (isRatio ? 100 : 1)).toFixed(2)}%`;
}

function duration(value: unknown) {
  const seconds = numeric(value);
  if (seconds === null || seconds < 0) return "—";
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours) return `${hours}小时 ${minutes}分`;
  return `${minutes}分 ${Math.round(seconds % 60)}秒`;
}

function formatMetric(value: unknown, kind: string, all: MetricRow) {
  if (kind === "count") return count(value);
  if (kind === "ratio") return percent(value, true);
  if (kind === "percent") return percent(value);
  if (kind === "duration") return duration(value);
  if (kind === "rate") return `${decimal(value, 1)} /秒`;
  if (kind === "score") return decimal(value, 1);
  if (kind === "memory") {
    const used = numeric(value);
    const total = numeric(all.gpu_memory_total_mb);
    return used === null ? "—" : `${(used / 1024).toFixed(2)}${total ? ` / ${(total / 1024).toFixed(1)}` : ""} GiB`;
  }
  if (kind === "temperature") return `${decimal(value, 1)} °C`;
  if (kind === "power") return `${decimal(value, 1)} W`;
  if (kind === "seconds") return `${decimal(value, 2)} 秒`;
  return decimal(value);
}

function downsample<T>(values: T[], limit = 1200) {
  if (values.length <= limit) return values;
  const last = values.length - 1;
  return Array.from({ length: limit }, (_, index) => values[Math.round(index * last / (limit - 1))]);
}

function niceAxis(values: number[], mode: "integer" | "decimal" | "percent") {
  if (mode === "percent") return { min: 0, max: 100, interval: 20, digits: 0 };
  if (!values.length) return { min: 0, max: mode === "integer" ? 1 : 0.1, interval: mode === "integer" ? 1 : 0.02, digits: mode === "integer" ? 0 : 2 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    const margin = Math.max(Math.abs(min) * 0.1, mode === "integer" ? 1 : 0.02);
    min -= margin;
    max += margin;
  }
  const raw = Math.max((max - min) / 5, Number.EPSILON);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const scaled = raw / magnitude;
  const stepFactor = [1, 2, 2.5, 5, 10].find((candidate) => scaled <= candidate) ?? 10;
  let interval = stepFactor * magnitude;
  if (mode === "integer") interval = Math.max(1, Math.ceil(interval));
  const axisMin = Math.floor(min / interval) * interval;
  const axisMax = Math.ceil(max / interval) * interval;
  const digits = mode === "integer" ? 0 : Math.max(0, Math.min(6, Math.ceil(-Math.log10(interval))));
  return { min: axisMin, max: axisMax, interval, digits };
}

function queueItems(payload: DashboardPayload | null | undefined): QueueItem[] {
  if (payload?.queue?.items?.length) return payload.queue.items;
  const training = payload?.training ?? {};
  if (!Object.keys(training).length) return [];
  return [{
    id: String(training.run_name ?? "current-run"),
    name: String(training.run_name ?? "当前训练"),
    status: String(training.phase ?? "running"),
    training_physics_fps: numeric(training.training_physics_fps),
    planned_transitions: numeric(training.total_transitions),
    transitions: numeric(training.transitions),
    progress_fraction: numeric(training.progress_fraction),
    message: String(training.completion_message ?? ""),
  }];
}

function chartDefinition(id: ChartId) {
  if (id === "score") return { source: "history", mode: "integer" as const, unit: "分", series: [
    ["training_window_mean_score", "窗口局均分"], ["training_rolling_mean_score", "近4096局均分"],
    ["training_window_max_score", "窗口最高分"], ["last_fast_eval_score", "30 FPS评估"],
    ["last_accurate_eval_score", "120 FPS评估"],
  ] };
  if (id === "loss") return { source: "history", mode: "decimal" as const, unit: "Loss", series: [
    ["loss", "总损失"], ["dqn_loss", "DQN损失"], ["mean_abs_td_error", "绝对TD误差"],
  ] };
  if (id === "aux") return { source: "history", mode: "decimal" as const, unit: "Loss", series: [
    ["aux_loss_merge", "合成"], ["aux_loss_q0_lineage", "q0谱系"], ["aux_loss_first_contact", "首次接触"],
    ["aux_loss_generation", "新水果"], ["aux_loss_outcome", "结局"], ["branch_aux_loss_total", "旁路辅助"],
  ] };
  if (id === "throughput") return { source: "history", mode: "integer" as const, unit: "/秒", series: [
    ["env_steps_per_second", "投放"], ["learner_samples_per_second", "学习样本"], ["branch_steps_per_second", "旁路"],
  ] };
  if (id === "events") return { source: "history", mode: "decimal" as const, unit: "/千投放", series: [
    ["eval_created_l7_per_1000", "L7"], ["eval_created_l8_per_1000", "L8"], ["eval_created_l9_per_1000", "L9"],
    ["eval_created_l10_per_1000", "L10"], ["eval_created_l11_per_1000", "L11"],
  ] };
  return { source: "resource_history", mode: "percent" as const, unit: "%", series: [
    ["gpu_utilization", "GPU利用率"], ["gpu_memory_utilization", "显存占用率"],
  ] };
}

function buildChartOption(payload: DashboardPayload | null | undefined, chartId: ChartId): EChartsOption {
  const definition = chartDefinition(chartId);
  const rows = downsample((definition.source === "history" ? payload?.history : payload?.resource_history) ?? []);
  const firstTimestamp = numeric(rows[0]?.timestamp) ?? 0;
  const x = (row: MetricRow, index: number) => definition.source === "history"
    ? (numeric(row.transitions) ?? index) / 1_000_000
    : ((numeric(row.timestamp) ?? firstTimestamp) - firstTimestamp) / 3600;
  const allValues = definition.series.flatMap(([key]) => rows.map((row) => numeric(row[key])).filter((value): value is number => value !== null));
  const axis = niceAxis(allValues, definition.mode);
  const visibleSeries = definition.series.filter(([key]) => rows.some((row) => numeric(row[key]) !== null));
  return {
    animationDurationUpdate: 350,
    color: SERIES_COLORS,
    grid: { left: 68, right: 32, top: 54, bottom: 72 },
    legend: { top: 4, type: "scroll", textStyle: { color: "#627187", fontSize: 11 } },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "cross", lineStyle: { color: "#91a0b4" } },
      backgroundColor: "rgba(248,250,253,.98)",
      borderColor: "rgba(91,109,136,.16)",
      textStyle: { color: "#26354c" },
      valueFormatter: (value) => `${Number(value).toFixed(axis.digits)} ${definition.unit}`,
    },
    toolbox: {
      right: 12,
      top: 0,
      feature: { dataZoom: { yAxisIndex: "none" }, restore: {} },
      iconStyle: { borderColor: "#6c7a90" },
    },
    xAxis: {
      type: "value",
      name: definition.source === "history" ? "训练投放（M）" : "运行时间（小时）",
      nameLocation: "middle",
      nameGap: 30,
      axisLabel: { color: "#7c899c", formatter: (value: number) => value.toFixed(value < 10 ? 1 : 0) },
      axisLine: { lineStyle: { color: "rgba(77,94,120,.18)" } },
      splitLine: { lineStyle: { color: "rgba(77,94,120,.08)" } },
    },
    yAxis: {
      type: "value",
      name: definition.unit,
      min: axis.min,
      max: axis.max,
      interval: axis.interval,
      axisLabel: { color: "#7c899c", formatter: (value: number) => value.toFixed(axis.digits) },
      splitLine: { lineStyle: { color: "rgba(77,94,120,.09)" } },
    },
    dataZoom: [
      { type: "inside", xAxisIndex: 0, filterMode: "none", zoomOnMouseWheel: "shift", moveOnMouseMove: true },
      { type: "slider", xAxisIndex: 0, height: 18, bottom: 10, borderColor: "transparent", backgroundColor: "rgba(83,102,130,.06)", fillerColor: "rgba(52,120,246,.16)" },
    ],
    series: visibleSeries.map(([key, name], index) => ({
      name,
      type: "line",
      showSymbol: false,
      smooth: false,
      connectNulls: true,
      lineStyle: { width: index === 0 ? 2.4 : 1.7 },
      emphasis: { focus: "series" },
      data: rows.map((row, rowIndex) => [x(row, rowIndex), numeric(row[key])]).filter((point) => point[1] !== null),
    })),
  };
}

function SummaryMetric({ icon: Icon, label, value, note }: { icon: LucideIcon; label: string; value: string; note: string }) {
  return <article className="training-summary-card"><span><Icon size={18} /></span><small>{label}</small><strong>{value}</strong><p>{note}</p></article>;
}

const MERGE_STATUS_LABELS: Record<string, string> = {
  running: "采集中",
  complete: "采集完成",
  interrupted: "已中断",
  wall_time_reached: "达到时间上限",
  failed: "采集失败",
};

function MergePotentialPanel({ status }: { status?: MergePotentialStatus }) {
  const run = status?.current;
  if (!status?.available || !run) return null;
  const progress = Math.max(0, Math.min(1, numeric(run.progress_fraction) ?? 0));
  const peakBytes = numeric(run.peak_cuda_allocated_bytes);
  const peakVram = peakBytes === null ? "—" : `${(peakBytes / 1024 ** 3).toFixed(2)} GiB`;
  const stateLabel = run.stale ? "更新已停滞" : (MERGE_STATUS_LABELS[run.status] ?? run.status);
  const checkpointSha = run.checkpoint_sha256 ? run.checkpoint_sha256.slice(0, 12) : "哈希待写入";
  const runCount = status.runs?.length ?? 1;
  return (
    <section className="merge-potential-section">
      <div className="section-heading-inline">
        <div><span>MERGE POTENTIAL DATA</span><h2>未来合成时间统计</h2><p>SAB-128 在当前无下落加速环境中的大规模对局采集；面板只读取采集清单。</p></div>
        <span className={`merge-status-chip merge-status-${run.stale ? "stale" : run.status}`}><Activity size={14} /> {stateLabel}</span>
      </div>
      <div className="merge-potential-current">
        <div><b>{run.name}</b><span>{run.run_dir}</span></div>
        <strong>{(progress * 100).toFixed(2)}%</strong>
        <div className="merge-progress-track"><motion.span initial={false} animate={{ width: `${progress * 100}%` }} /></div>
      </div>
      <div className="training-summary-grid merge-summary-grid">
        <SummaryMetric icon={Database} label="已完成对局" value={count(run.completed_episodes)} note={`目标 ${count(run.target_episodes)}`} />
        <SummaryMetric icon={Activity} label="采集吞吐" value={`${decimal(run.env_steps_per_second, 0)} /秒`} note={`${count(run.parallel_envs)} 个并行环境`} />
        <SummaryMetric icon={ListOrdered} label="快照样本" value={count(run.snapshot_rows)} note={`${count(run.merge_source_rows)} 条合成源事件`} />
        <SummaryMetric icon={Cpu} label="峰值显存" value={peakVram} note={run.cuda_device_name || "CUDA设备待记录"} />
        <SummaryMetric icon={Timer} label="已运行" value={duration(run.elapsed_seconds)} note={`${count(run.transitions)} 次策略投放`} />
      </div>
      <div className="merge-potential-meta">
        <span><b>模型</b>{run.checkpoint_name || "checkpoint待记录"} · {checkpointSha}</span>
        <span><b>物理</b>{count(run.physics_fps)} FPS · {run.drop_fast_forward ? "启用下落加速" : "无下落加速"} · {run.max_drops ? `${count(run.max_drops)}投放上限` : "单局无投放上限"}</span>
        <span><b>离线汇总</b>{run.analysis_ready ? "分析表已生成" : "等待采集后汇总"}</span>
        <span><b>已发现任务</b>{runCount}</span>
        {run.failure && <span className="merge-failure"><b>异常</b>{run.failure}</span>}
      </div>
    </section>
  );
}

export function TrainingWorkspace({ dashboard, onRefresh, onOpenTools }: Props) {
  const [chartId, setChartId] = useState<ChartId>("score");
  const payload = dashboard.payload;
  const training = payload?.training ?? {};
  const resources = payload?.resources ?? {};
  const all = { ...training, ...resources };
  const phase = String(training.phase ?? "offline");
  const progress = Math.max(0, Math.min(1, numeric(training.progress_fraction) ?? 0));
  const queue = queueItems(payload);
  const chartOption = useMemo(() => buildChartOption(payload, chartId), [payload, chartId]);
  const actionDistribution = Array.isArray(training.action_distribution)
    ? (training.action_distribution as unknown as number[])
    : [];
  const actionMax = Math.max(...actionDistribution, 1e-9);
  const hasTraining = Object.keys(training).length > 0;

  return (
    <section className="training-workspace">
      <div className="training-commandbar">
        <div>
          <span className={`training-live-indicator ${dashboard.available ? "is-online" : ""}`} />
          <span>{dashboard.available ? "运行遥测已连接" : "等待运行数据源"}</span>
          {numeric(training.training_physics_fps) !== null && <b>{count(training.training_physics_fps)} FPS训练</b>}
        </div>
        <button onClick={onRefresh}><RefreshCw size={15} /> 刷新</button>
      </div>

      {!dashboard.available ? (
        <div className="training-offline-state">
          <Server size={38} />
          <h2>训练数据源尚未连接</h2>
          <p>建立SSH转发，或从工具中心启动一个本地历史run数据服务。</p>
          <button className="primary-button compact" onClick={onOpenTools}>打开工具中心</button>
        </div>
      ) : (
        <>
          {hasTraining && <>
          <section className="training-run-hero">
            <div className="training-run-copy">
              <span className={`phase-chip phase-${phase}`}>{PHASE_LABELS[phase] ?? phase}</span>
              <h2>{String(training.run_name ?? queue.find((item) => ["running", "evaluating"].includes(item.status))?.name ?? "当前训练")}</h2>
              <p>{String(training.completion_message ?? "训练状态正在持续写入只读遥测。")}</p>
            </div>
            <div className="training-progress-orb"><strong>{(progress * 100).toFixed(2)}%</strong><span>{count(training.transitions)} / {count(training.total_transitions)}</span></div>
            <div className="training-progress-track"><motion.span initial={false} animate={{ width: `${progress * 100}%` }} /></div>
          </section>

          <div className="training-summary-grid">
            <SummaryMetric icon={Database} label="累计投放" value={count(training.transitions)} note={`目标 ${count(training.total_transitions)}`} />
            <SummaryMetric icon={Activity} label="实时吞吐" value={`${decimal(training.env_steps_per_second, 0)} /秒`} note={`${count(training.active_envs)} 个并行环境`} />
            <SummaryMetric icon={Timer} label="预计剩余" value={duration(training.eta_seconds)} note={`${count(training.updates)} 次模型更新`} />
            <SummaryMetric icon={Gauge} label="最近120 FPS" value={decimal(training.last_accurate_eval_score, 1)} note={`30 FPS ${decimal(training.last_fast_eval_score, 1)}`} />
            <SummaryMetric icon={Cpu} label="GPU利用率" value={percent(resources.gpu_utilization)} note={`${formatMetric(resources.gpu_memory_used_mb, "memory", all)} 显存`} />
          </div>

          <section className="training-queue-section">
            <div className="section-heading-inline">
              <div><span>TRAINING QUEUE</span><h2>训练队列</h2><p>当前训练、预检、评估和后续计划使用同一状态序列。</p></div>
              <span className="queue-count"><ListOrdered size={15} /> {queue.length} 项</span>
            </div>
            <div className="training-queue-list">
              {queue.map((item, index) => {
                const itemProgress = Math.max(0, Math.min(1, numeric(item.progress_fraction) ?? (item.status === "completed" ? 1 : 0)));
                return (
                  <motion.article key={item.id} className={`queue-item queue-${item.status}`} initial={{ opacity: 0, x: 18 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * .06 }}>
                    <span className="queue-order">{item.status === "running" ? <Activity size={16} /> : item.status === "completed" ? <CheckCircle2 size={16} /> : String(index + 1).padStart(2, "0")}</span>
                    <div className="queue-copy"><span>{QUEUE_LABELS[item.status] ?? item.status}</span><h3>{item.name}</h3><p>{item.message || item.source_run || item.config || "等待调度器写入更多说明"}</p></div>
                    <div className="queue-meta"><b>{item.training_physics_fps ? `${item.training_physics_fps} FPS` : "物理待定"}</b><span>{item.planned_transitions ? `${(item.planned_transitions / 1_000_000).toFixed(0)}M` : "预算待定"}</span></div>
                    <div className="queue-progress"><span style={{ width: `${itemProgress * 100}%` }} /></div>
                  </motion.article>
                );
              })}
            </div>
          </section>

          <section className="training-chart-explorer">
            <div className="section-heading-inline">
              <div><span>INTERACTIVE CURVES</span><h2>训练曲线</h2><p>滚轮移动，Shift+滚轮缩放；底部滑块选择局部区间，工具栏可框选和复位。</p></div>
            </div>
            <div className="training-chart-tabs">
              {CHARTS.map((chart) => <button key={chart.id} className={chartId === chart.id ? "active" : ""} onClick={() => setChartId(chart.id)}><b>{chart.label}</b><small>{chart.description}</small></button>)}
            </div>
            <EChart option={chartOption} className="training-native-chart" />
          </section>

          <section className="training-details-stack">
            {DETAIL_GROUPS.map((group, index) => (
              <details key={group.id} className="training-disclosure" open={index === 0}>
                <summary><span><ChevronDown size={16} /></span><div><b>{group.title}</b><small>{group.hint}</small></div></summary>
                <div className="training-metric-grid">
                  {group.metrics.map(([key, label, kind]) => <div key={key}><span>{label}</span><strong>{formatMetric(all[key], kind, all)}</strong></div>)}
                </div>
              </details>
            ))}

            <details className="training-disclosure">
              <summary><span><ChevronDown size={16} /></span><div><b>动作分布</b><small>最近统计窗口中的21个离散投放动作</small></div></summary>
              <div className="native-action-distribution">
                {actionDistribution.length ? actionDistribution.map((value, index) => <div key={index} title={`A${index} · ${percent(value, value <= 1)}`}><span style={{ height: `${Math.max(3, Number(value) / actionMax * 100)}%` }} /><small>A{index}</small></div>) : <p>等待动作分布数据。</p>}
              </div>
            </details>

            <details className="training-disclosure">
              <summary><span><ChevronDown size={16} /></span><div><b>训练事件</b><small>checkpoint、评估、扩容、完成和异常边界</small></div></summary>
              <div className="native-event-list">
                {(payload?.events ?? []).slice(-16).reverse().map((event, index) => <article key={`${event.kind}-${index}`}><span>{String(event.kind ?? "event")}</span><p>{String(event.message ?? "")}</p><time>{event.monitor_timestamp ? new Date(Number(event.monitor_timestamp) * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : "—"}</time></article>)}
                {!payload?.events?.length && <p>等待训练事件。</p>}
              </div>
            </details>
          </section>

          </>}

          <MergePotentialPanel status={payload?.merge_potential} />

          <div className="training-data-note"><AlertCircle size={15} /><span>界面只读展示聚合数据，不向训练进程发送控制命令。图表缩放只改变本地视图，不修改训练产物。</span></div>
        </>
      )}
    </section>
  );
}
