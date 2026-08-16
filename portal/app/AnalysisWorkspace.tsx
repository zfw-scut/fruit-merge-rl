"use client";

import type { EChartsOption } from "echarts";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  ArrowDownUp,
  Braces,
  Database,
  RefreshCw,
  Search,
  TableProperties,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { EChart } from "./EChart";

type Scalar = string | number | boolean | null;
type DataRow = Record<string, Scalar>;

type AnalysisTable = {
  id: string;
  label: string;
  source: string;
  columns: string[];
  rows: DataRow[];
  row_count: number;
  truncated: boolean;
};

type AnalysisDataset = {
  id: string;
  kind: string;
  name: string;
  status: string;
  created_at: string | null;
  updated_at: string | null;
  run_dir: string;
  metadata: {
    episodes: number;
    transitions: number;
    unique_observed_fruits: number;
    snapshot_rows: number;
    merge_sources: number;
    physics_fps: number;
    drop_fast_forward: boolean;
    max_drops: number;
    checkpoint: string;
    checkpoint_sha256: string;
    horizons: number[];
    factor_bins: number;
    interaction_bins: number;
  };
  tables: AnalysisTable[];
};

type AnalysisResponse = {
  available: boolean;
  datasets: AnalysisDataset[];
};

type ViewId = "time" | "factor" | "interaction" | "table";
type MetricId = "eventual" | "h8" | "h32" | "h128" | "terminal";

const VIEW_OPTIONS: Array<{ id: ViewId; label: string; note: string }> = [
  { id: "time", label: "时间尺度", note: "等级 × 未来窗口" },
  { id: "factor", label: "单因素关系", note: "条件概率与关联强度" },
  { id: "interaction", label: "双因素交互", note: "联合分箱热图" },
  { id: "table", label: "数据表", note: "搜索、排序与追溯" },
];

const METRICS: Record<MetricId, { label: string; field: string; weight: string }> = {
  eventual: {
    label: "最终合成",
    field: "eventual_merge_probability_resolved",
    weight: "resolved_weight",
  },
  h8: { label: "8次内", field: "merge_probability_h8", weight: "eligible_weight_h8" },
  h32: { label: "32次内", field: "merge_probability_h32", weight: "eligible_weight_h32" },
  h128: { label: "128次内", field: "merge_probability_h128", weight: "eligible_weight_h128" },
  terminal: {
    label: "终局未合成",
    field: "terminal_unmerged_probability_resolved",
    weight: "resolved_weight",
  },
};

const FACTOR_LABELS: Record<string, string> = {
  scene_occupancy_ratio: "场景占用率",
  center_height_normalized: "水果中心高度",
  same_level_peer_count: "同级水果数量",
  nearest_same_level_center_distance_normalized: "最近同级中心距离",
};

const INTERACTION_LABELS: Record<string, string> = {
  occupancy_x_nearest_same_level_distance: "占用率 × 最近同级距离",
  height_x_same_level_peer_count: "高度 × 同级数量",
};

const CORE_COLUMNS: Record<string, string[]> = {
  lifecycle_by_level: [
    "level", "fruits", "merged_fruits", "terminal_unmerged_fruits",
    "eventual_merge_probability_resolved", "terminal_unmerged_probability_resolved",
    "merged_t_median", "merged_t_p90", "merged_t_p95",
  ],
  lifecycle_t_merge_histogram: ["level", "t_merge", "count"],
  horizon_probabilities_by_level: [
    "level", "horizon_drops", "lifecycle_eligible_fruits",
    "lifecycle_merged_within", "lifecycle_probability", "snapshot_probability",
  ],
  factor_relationships_by_level: [
    "level", "factor", "bin_index", "bin_lower", "bin_upper",
    "missing_same_level_peer", "fruit_normalized_weight",
    "eventual_merge_probability_resolved", "merge_probability_h8",
    "merge_probability_h32", "merge_probability_h128",
    "terminal_unmerged_probability_resolved",
  ],
  factor_interactions_by_level: [
    "level", "interaction", "first_bin", "second_bin", "first_bin_lower",
    "first_bin_upper", "second_bin_lower", "second_bin_upper",
    "fruit_normalized_weight", "eventual_merge_probability_resolved",
    "merge_probability_h8", "merge_probability_h32", "merge_probability_h128",
    "terminal_unmerged_probability_resolved",
  ],
};

const FIELD_LABELS: Record<string, string> = {
  level: "等级",
  fruits: "水果数",
  merged_fruits: "最终合成数",
  terminal_unmerged_fruits: "终局未合成数",
  eventual_merge_probability_resolved: "最终合成率",
  terminal_unmerged_probability_resolved: "终局未合成率",
  merged_t_median: "T_merge 中位数",
  merged_t_p90: "T_merge P90",
  merged_t_p95: "T_merge P95",
  horizon_drops: "未来投放",
  lifecycle_probability: "生命周期概率",
  snapshot_probability: "状态快照概率",
  factor: "因素",
  interaction: "交互",
  bin_index: "分箱",
  raw_snapshot_rows: "快照行",
  fruit_normalized_weight: "水果归一化权重",
  merge_probability_h8: "8次内合成率",
  merge_probability_h32: "32次内合成率",
  merge_probability_h128: "128次内合成率",
};

const chartTooltip = {
  backgroundColor: "rgba(242,245,249,.98)",
  borderColor: "rgba(75,91,116,.12)",
  padding: 12,
  extraCssText: "box-shadow:10px 12px 28px rgba(110,122,142,.24);border-radius:12px;",
  textStyle: { color: "#26354c" },
};

function asNumber(value: Scalar | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatNumber(value: number | null | undefined, digits = 0) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString("zh-CN", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatPercent(value: number | null | undefined, digits = 1) {
  return value === null || value === undefined ? "—" : `${formatNumber(value * 100, digits)}%`;
}

function probabilityColumn(column: string) {
  return column.includes("probability") || column.startsWith("merge_probability_");
}

function tableOf(dataset: AnalysisDataset | undefined, id: string) {
  return dataset?.tables.find((table) => table.id === id);
}

function rangeLabel(lower: Scalar | undefined, upper: Scalar | undefined, percent = false) {
  const low = asNumber(lower);
  const high = asNumber(upper);
  if (low === null || high === null) return "缺失";
  if (low === high) return formatNumber(low);
  return percent
    ? `${formatNumber(low * 100)}–${formatNumber(high * 100)}%`
    : `${formatNumber(low, 2)}–${formatNumber(high, 2)}`;
}

function factorBinLabel(row: DataRow, factor: string) {
  if (row.missing_same_level_peer === true) return "无同级水果";
  if (factor === "same_level_peer_count") {
    const value = asNumber(row.bin_lower);
    return value === null ? "—" : value >= 8 ? "≥8" : formatNumber(value);
  }
  const percent = factor === "scene_occupancy_ratio" || factor === "center_height_normalized";
  return rangeLabel(row.bin_lower, row.bin_upper, percent);
}

function effectStrength(rows: DataRow[], metric: MetricId) {
  const definition = METRICS[metric];
  const valid = rows
    .map((row) => ({ p: asNumber(row[definition.field]), w: asNumber(row[definition.weight]) }))
    .filter((item): item is { p: number; w: number } => item.p !== null && item.w !== null && item.w > 0);
  const totalWeight = valid.reduce((total, item) => total + item.w, 0);
  if (!totalWeight) return 0;
  const mean = valid.reduce((total, item) => total + item.p * item.w, 0) / totalWeight;
  const denominator = totalWeight * mean * (1 - mean);
  if (denominator <= 0) return 0;
  const between = valid.reduce((total, item) => total + item.w * ((item.p - mean) ** 2), 0);
  return between / denominator;
}

function metricValue(row: DataRow, metric: MetricId) {
  return asNumber(row[METRICS[metric].field]);
}

function rawCell(value: Scalar, column: string) {
  if (value === null) return "—";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (typeof value === "number") {
    if (probabilityColumn(column)) return formatPercent(value, 2);
    return formatNumber(value, Number.isInteger(value) ? 0 : 4);
  }
  return String(value);
}

function datasetTitle(dataset: AnalysisDataset) {
  if (dataset.kind === "merge_potential") {
    return `SAB-128 · ${formatNumber(dataset.metadata.episodes)}局 Merge Potential`;
  }
  return dataset.name;
}

export function AnalysisWorkspace({ apiBase }: { apiBase: string }) {
  const reduceMotion = useReducedMotion();
  const [datasets, setDatasets] = useState<AnalysisDataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState("");
  const [activeView, setActiveView] = useState<ViewId>("time");
  const [selectedLevel, setSelectedLevel] = useState(9);
  const [selectedMetric, setSelectedMetric] = useState<MetricId>("h32");
  const [selectedFactor, setSelectedFactor] = useState("center_height_normalized");
  const [selectedInteraction, setSelectedInteraction] = useState("height_x_same_level_peer_count");
  const [selectedTableId, setSelectedTableId] = useState("lifecycle_by_level");
  const [rawQuery, setRawQuery] = useState("");
  const [rawLevel, setRawLevel] = useState("all");
  const [showAllColumns, setShowAllColumns] = useState(false);
  const [sortColumn, setSortColumn] = useState("level");
  const [sortAscending, setSortAscending] = useState(true);
  const [rawPage, setRawPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBase}/api/analysis/datasets`, { cache: "no-store" });
      if (!response.ok) throw new Error("统计数据接口不可用");
      const payload = (await response.json()) as AnalysisResponse;
      setDatasets(payload.datasets ?? []);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法读取统计数据");
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    const initialLoad = window.setTimeout(() => void loadDatasets(), 0);
    return () => window.clearTimeout(initialLoad);
  }, [loadDatasets]);

  const dataset = datasets.find((item) => item.id === selectedDatasetId) ?? datasets[0];
  const lifecycle = tableOf(dataset, "lifecycle_by_level");
  const horizons = tableOf(dataset, "horizon_probabilities_by_level");
  const factors = tableOf(dataset, "factor_relationships_by_level");
  const interactions = tableOf(dataset, "factor_interactions_by_level");

  const totals = useMemo(() => {
    const rows = lifecycle?.rows ?? [];
    const fruits = rows.reduce((sum, row) => sum + (asNumber(row.fruits) ?? 0), 0);
    const merged = rows.reduce((sum, row) => sum + (asNumber(row.merged_fruits) ?? 0), 0);
    const terminal = rows.reduce((sum, row) => sum + (asNumber(row.terminal_unmerged_fruits) ?? 0), 0);
    return { fruits, merged, terminal };
  }, [lifecycle]);

  const levelValues = useMemo(() => (
    [...new Set((lifecycle?.rows ?? []).map((row) => asNumber(row.level)).filter((value): value is number => value !== null))]
      .sort((a, b) => a - b)
  ), [lifecycle]);

  const horizonOption = useMemo<EChartsOption>(() => {
    const rows = horizons?.rows ?? [];
    const horizonValues = [...new Set(rows.map((row) => asNumber(row.horizon_drops)).filter((value): value is number => value !== null))].sort((a, b) => a - b);
    const levels = [...new Set(rows.map((row) => asNumber(row.level)).filter((value): value is number => value !== null))].sort((a, b) => a - b);
    const data = rows.flatMap((row) => {
      const horizon = asNumber(row.horizon_drops);
      const level = asNumber(row.level);
      const probability = asNumber(row.lifecycle_probability);
      if (horizon === null || level === null || probability === null) return [];
      return [[horizonValues.indexOf(horizon), levels.indexOf(level), probability * 100]];
    });
    return {
      animation: !reduceMotion,
      animationDurationUpdate: 420,
      animationEasingUpdate: "cubicOut",
      grid: { left: 55, right: 35, top: 24, bottom: 82 },
      tooltip: {
        ...chartTooltip,
        formatter: (raw: unknown) => {
          const item = raw as { value: [number, number, number] };
          return `<div class="analysis-tooltip"><b>L${levels[item.value[1]]}</b><span>未来 ${horizonValues[item.value[0]]} 次投放</span><strong>${formatNumber(item.value[2], 2)}%</strong></div>`;
        },
      },
      xAxis: {
        type: "category",
        name: "未来投放次数",
        nameLocation: "middle",
        nameGap: 38,
        data: horizonValues.map(String),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: "rgba(75,91,116,.14)" } },
        axisLabel: { color: "#7e8a9e", fontSize: 10 },
      },
      yAxis: {
        type: "category",
        name: "水果等级",
        data: levels.map((level) => `L${level}`),
        axisTick: { show: false },
        axisLine: { lineStyle: { color: "rgba(75,91,116,.14)" } },
        axisLabel: { color: "#7e8a9e", fontSize: 10 },
      },
      visualMap: {
        min: 0,
        max: 100,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 10,
        itemWidth: 14,
        itemHeight: 120,
        text: ["100%", "0%"],
        textStyle: { color: "#8793a6", fontSize: 9 },
        inRange: { color: ["#e2e8f0", "#c7d8f1", "#78a4e5", "#3478e5"] },
      },
      series: [{
        type: "heatmap",
        data,
        label: {
          show: true,
          color: "#34445c",
          fontSize: 8,
          formatter: (raw: unknown) => `${formatNumber((raw as { value: [number, number, number] }).value[2], 0)}%`,
        },
        itemStyle: { borderColor: "#edf1f7", borderWidth: 3, borderRadius: 5 },
        emphasis: { itemStyle: { shadowBlur: 12, shadowColor: "rgba(52,120,229,.25)" } },
      }],
    };
  }, [horizons, reduceMotion]);

  const selectedFactorRows = useMemo(() => (
    (factors?.rows ?? [])
      .filter((row) => asNumber(row.level) === selectedLevel && row.factor === selectedFactor)
      .sort((a, b) => (asNumber(a.bin_index) ?? 0) - (asNumber(b.bin_index) ?? 0))
  ), [factors, selectedFactor, selectedLevel]);

  const factorLineOption = useMemo<EChartsOption>(() => {
    const labels = selectedFactorRows.map((row) => factorBinLabel(row, selectedFactor));
    const seriesDefinitions: Array<{ id: MetricId; color: string }> = [
      { id: "h8", color: "#37bda1" },
      { id: "h32", color: "#3478e5" },
      { id: "h128", color: "#7d6ee7" },
      { id: "eventual", color: "#e7a43b" },
    ];
    return {
      animation: !reduceMotion,
      animationDurationUpdate: 420,
      animationEasingUpdate: "cubicOut",
      color: seriesDefinitions.map((item) => item.color),
      grid: { left: 54, right: 24, top: 70, bottom: 66 },
      legend: { top: 8, left: 48, right: 8, itemGap: 8, textStyle: { color: "#738196", fontSize: 8 }, itemWidth: 11, itemHeight: 6 },
      tooltip: { ...chartTooltip, trigger: "axis", valueFormatter: (value: unknown) => `${formatNumber(Number(value), 2)}%` },
      xAxis: {
        type: "category",
        data: labels,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: "rgba(75,91,116,.14)" } },
        axisLabel: { color: "#7e8a9e", fontSize: 9, rotate: labels.length > 7 ? 20 : 0 },
      },
      yAxis: {
        type: "value",
        name: "合成概率（%）",
        min: 0,
        max: 100,
        splitLine: { lineStyle: { color: "rgba(75,91,116,.08)" } },
        axisLabel: { color: "#8793a6", formatter: "{value}%" },
      },
      series: seriesDefinitions.map(({ id, color }) => ({
        name: METRICS[id].label,
        type: "line",
        smooth: 0.24,
        symbol: id === selectedMetric ? "circle" : "emptyCircle",
        symbolSize: id === selectedMetric ? 8 : 6,
        lineStyle: { width: id === selectedMetric ? 3 : 1.5, color },
        itemStyle: { color },
        data: selectedFactorRows.map((row) => {
          const value = metricValue(row, id);
          return value === null ? null : value * 100;
        }),
      })),
    };
  }, [reduceMotion, selectedFactor, selectedFactorRows, selectedMetric]);

  const effectRows = useMemo(() => {
    const rows = (factors?.rows ?? []).filter((row) => asNumber(row.level) === selectedLevel);
    return Object.keys(FACTOR_LABELS).map((factor) => ({
      factor,
      value: effectStrength(rows.filter((row) => row.factor === factor), selectedMetric),
    })).sort((a, b) => b.value - a.value);
  }, [factors, selectedLevel, selectedMetric]);

  const effectOption = useMemo<EChartsOption>(() => {
    const largest = Math.max(0, ...effectRows.map((item) => item.value * 100));
    const axisMaximum = Math.max(1, Math.ceil(largest * 1.35));
    const axisInterval = axisMaximum <= 5 ? 1 : axisMaximum <= 12 ? 2 : Math.ceil(axisMaximum / 4 / 5) * 5;
    return {
      animation: !reduceMotion,
      animationDurationUpdate: 420,
      grid: { left: 105, right: 34, top: 16, bottom: 42 },
      tooltip: {
        ...chartTooltip,
        trigger: "item",
        formatter: (raw: unknown) => {
          const item = raw as { name: string; value: number };
          return `<div class="analysis-tooltip"><b>${item.name}</b><span>加权分箱关联强度</span><strong>${formatNumber(item.value, 2)}%</strong></div>`;
        },
      },
      xAxis: {
        type: "value",
        name: "解释份额（%）",
        max: axisMaximum,
        interval: axisInterval,
        splitNumber: 4,
        splitLine: { lineStyle: { color: "rgba(75,91,116,.08)" } },
        axisLabel: { color: "#8793a6", formatter: "{value}%", hideOverlap: true },
      },
      yAxis: {
        type: "category",
        data: effectRows.map((item) => FACTOR_LABELS[item.factor]),
        axisTick: { show: false },
        axisLine: { show: false },
        axisLabel: { color: "#66758b", fontSize: 9 },
      },
      series: [{
        type: "bar",
        data: effectRows.map((item, index) => ({
          name: FACTOR_LABELS[item.factor],
          value: item.value * 100,
          itemStyle: { color: index === 0 ? "#3478e5" : "#9db9e2", borderRadius: [0, 7, 7, 0] },
        })),
        barMaxWidth: 18,
        label: { show: true, position: "right", color: "#65758b", fontSize: 9, formatter: (raw: unknown) => `${formatNumber((raw as { value: number }).value, 2)}%` },
      }],
    };
  }, [effectRows, reduceMotion]);

  const selectedInteractionRows = useMemo(() => (
    (interactions?.rows ?? [])
      .filter((row) => asNumber(row.level) === selectedLevel && row.interaction === selectedInteraction)
  ), [interactions, selectedInteraction, selectedLevel]);

  const interactionAxes = useMemo(() => {
    const firstBins = [...new Set(selectedInteractionRows.map((row) => asNumber(row.first_bin)).filter((value): value is number => value !== null))].sort((a, b) => a - b);
    const secondBins = [...new Set(selectedInteractionRows.map((row) => asNumber(row.second_bin)).filter((value): value is number => value !== null))].sort((a, b) => a - b);
    const firstLabels = firstBins.map((bin) => {
      const row = selectedInteractionRows.find((candidate) => asNumber(candidate.first_bin) === bin);
      return row ? rangeLabel(row.first_bin_lower, row.first_bin_upper, true) : String(bin);
    });
    const secondLabels = secondBins.map((bin) => {
      const row = selectedInteractionRows.find((candidate) => asNumber(candidate.second_bin) === bin);
      if (!row) return String(bin);
      if (row.second_is_missing_or_capped === true) {
        return selectedInteraction === "height_x_same_level_peer_count" ? "封顶档" : "无同级";
      }
      if (selectedInteraction === "height_x_same_level_peer_count") {
        return formatNumber(asNumber(row.second_bin_lower));
      }
      return rangeLabel(row.second_bin_lower, row.second_bin_upper, true);
    });
    return { firstBins, secondBins, firstLabels, secondLabels };
  }, [selectedInteraction, selectedInteractionRows]);

  const interactionOption = useMemo<EChartsOption>(() => {
    const values = selectedInteractionRows.flatMap((row) => {
      const first = asNumber(row.first_bin);
      const second = asNumber(row.second_bin);
      const value = metricValue(row, selectedMetric);
      if (first === null || second === null || value === null) return [];
      return [[interactionAxes.secondBins.indexOf(second), interactionAxes.firstBins.indexOf(first), value * 100]];
    });
    return {
      animation: !reduceMotion,
      animationDurationUpdate: 420,
      animationEasingUpdate: "cubicOut",
      grid: { left: 76, right: 34, top: 24, bottom: 96 },
      tooltip: {
        ...chartTooltip,
        formatter: (raw: unknown) => {
          const item = raw as { value: [number, number, number] };
          return `<div class="analysis-tooltip"><b>${interactionAxes.firstLabels[item.value[1]]}</b><span>${interactionAxes.secondLabels[item.value[0]]}</span><strong>${formatNumber(item.value[2], 2)}%</strong></div>`;
        },
      },
      xAxis: {
        type: "category",
        name: selectedInteraction === "height_x_same_level_peer_count" ? "同级水果数量" : "最近同级中心距离 / 棋盘宽度",
        nameLocation: "middle",
        nameGap: 48,
        data: interactionAxes.secondLabels,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: "rgba(75,91,116,.14)" } },
        axisLabel: { color: "#7e8a9e", fontSize: 9, rotate: 18 },
      },
      yAxis: {
        type: "category",
        name: selectedInteraction === "height_x_same_level_peer_count" ? "水果中心高度" : "场景占用率",
        data: interactionAxes.firstLabels,
        axisTick: { show: false },
        axisLine: { lineStyle: { color: "rgba(75,91,116,.14)" } },
        axisLabel: { color: "#7e8a9e", fontSize: 9 },
      },
      visualMap: {
        min: 0,
        max: 100,
        orient: "horizontal",
        left: "center",
        bottom: 12,
        itemWidth: 14,
        itemHeight: 120,
        text: ["100%", "0%"],
        textStyle: { color: "#8793a6", fontSize: 9 },
        inRange: { color: ["#e3e8ef", "#c9d8f0", "#8caee2", "#4b83d6", "#765fc7"] },
      },
      series: [{
        type: "heatmap",
        data: values,
        label: { show: true, color: "#34445c", fontSize: 9, formatter: (raw: unknown) => `${formatNumber((raw as { value: [number, number, number] }).value[2], 1)}%` },
        itemStyle: { borderColor: "#edf1f7", borderWidth: 4, borderRadius: 6 },
        emphasis: { itemStyle: { shadowBlur: 12, shadowColor: "rgba(52,120,229,.25)" } },
      }],
    };
  }, [interactionAxes, reduceMotion, selectedInteraction, selectedInteractionRows, selectedMetric]);

  const rawTable = tableOf(dataset, selectedTableId) ?? dataset?.tables[0];
  const rawColumns = useMemo(() => {
    if (!rawTable) return [];
    if (showAllColumns) return rawTable.columns;
    const preferred = CORE_COLUMNS[rawTable.id] ?? rawTable.columns.slice(0, 14);
    return preferred.filter((column) => rawTable.columns.includes(column));
  }, [rawTable, showAllColumns]);
  const filteredRawRows = useMemo(() => {
    if (!rawTable) return [];
    const query = rawQuery.trim().toLocaleLowerCase("zh-CN");
    return rawTable.rows.filter((row) => {
      if (rawLevel !== "all" && asNumber(row.level) !== Number(rawLevel)) return false;
      if (!query) return true;
      return rawColumns.some((column) => String(row[column] ?? "").toLocaleLowerCase("zh-CN").includes(query));
    }).sort((a, b) => {
      const left = a[sortColumn];
      const right = b[sortColumn];
      if (left === right) return 0;
      if (left === null) return 1;
      if (right === null) return -1;
      const result = typeof left === "number" && typeof right === "number"
        ? left - right
        : String(left).localeCompare(String(right), "zh-CN");
      return sortAscending ? result : -result;
    });
  }, [rawColumns, rawLevel, rawQuery, rawTable, sortAscending, sortColumn]);
  const rawPageCount = Math.max(1, Math.ceil(filteredRawRows.length / 50));
  const safeRawPage = Math.min(rawPage, rawPageCount - 1);
  const pagedRawRows = filteredRawRows.slice(safeRawPage * 50, safeRawPage * 50 + 50);

  if (loading) {
    return <div className="analysis-empty panel"><RefreshCw className="analysis-spin" size={28} /><h2>正在整理统计表</h2><p>只读取分析CSV，不加载原始张量分片。</p></div>;
  }
  if (error || !dataset) {
    return <div className="analysis-empty panel"><Database size={30} /><h2>{error ?? "暂无可分析的数据集"}</h2><p>完成统计汇总并将 analysis 目录迁回本地后，这里会自动发现数据。</p><button className="ghost-button" onClick={() => void loadDatasets()}><RefreshCw size={15} />重新读取</button></div>;
  }

  const transition = reduceMotion ? { duration: 0 } : { duration: 0.28, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] };
  return (
    <div className="analysis-workspace">
      <section className="analysis-dataset-bar panel">
        <div>
          <span className="panel-kicker">ACTIVE DATASET</span>
          <h2>{datasetTitle(dataset)}</h2>
          <p>{dataset.run_dir} · {dataset.metadata.physics_fps || "—"} FPS · {dataset.metadata.drop_fast_forward ? "自由下落加速" : "完整逐帧下落"} · {dataset.metadata.max_drops === 0 ? "无投放上限" : `上限 ${dataset.metadata.max_drops}`}</p>
        </div>
        <label className="analysis-select-field">
          <span>数据集</span>
          <select value={dataset.id} onChange={(event) => setSelectedDatasetId(event.target.value)}>
            {datasets.map((item) => <option key={item.id} value={item.id}>{datasetTitle(item)}</option>)}
          </select>
        </label>
        <button className="icon-button" onClick={() => void loadDatasets()} aria-label="刷新统计数据"><RefreshCw size={16} /></button>
      </section>

      <div className="analysis-summary-grid">
        <article className="analysis-summary-card panel"><span>自然终局对局</span><strong>{formatNumber(dataset.metadata.episodes)}</strong><small>{formatNumber(dataset.metadata.transitions)} 次投放</small></article>
        <article className="analysis-summary-card panel"><span>独立水果生命周期</span><strong>{formatNumber(totals.fruits || dataset.metadata.unique_observed_fruits)}</strong><small>{formatNumber(dataset.metadata.snapshot_rows)} 条快照</small></article>
        <article className="analysis-summary-card panel"><span>最终参与合成</span><strong>{formatPercent(totals.fruits ? totals.merged / totals.fruits : null, 2)}</strong><small>{formatNumber(totals.merged)} 颗水果</small></article>
        <article className="analysis-summary-card panel"><span>终局仍未合成</span><strong>{formatPercent(totals.fruits ? totals.terminal / totals.fruits : null, 2)}</strong><small>当前数据无删失样本</small></article>
      </div>

      <div className="analysis-view-tabs" role="tablist" aria-label="统计分析视图">
        {VIEW_OPTIONS.map((view) => (
          <button key={view.id} role="tab" aria-selected={activeView === view.id} className={activeView === view.id ? "active" : ""} onClick={() => setActiveView(view.id)}>
            <b>{view.label}</b><small>{view.note}</small>
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        {activeView === "time" && (
          <motion.div key="time" className="analysis-view-stack" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={transition}>
            <section className="analysis-chart-panel panel">
              <div className="section-heading-inline"><div><span>LEVEL × HORIZON</span><h2>不同等级在各时间窗口内的合成概率</h2><p>每颗水果生命周期只使用首次稳定观察，色块表示在未来指定投放次数内参与合成的比例。</p></div></div>
              <EChart option={horizonOption} className="analysis-heatmap-chart" ariaLabel="各水果等级在不同未来投放窗口内的合成概率热图" />
            </section>
            <section className="analysis-table-panel panel">
              <div className="section-heading-inline"><div><span>LIFECYCLE TABLE</span><h2>等级生命周期摘要</h2><p>概率、典型等待时间与长尾放在同一行比较。</p></div></div>
              <LifecycleTable rows={lifecycle?.rows ?? []} />
            </section>
          </motion.div>
        )}

        {activeView === "factor" && (
          <motion.div key="factor" className="analysis-view-stack" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={transition}>
            <AnalysisControls levels={levelValues} level={selectedLevel} onLevel={setSelectedLevel} metric={selectedMetric} onMetric={setSelectedMetric}>
              <label><span>状态因素</span><select value={selectedFactor} onChange={(event) => setSelectedFactor(event.target.value)}>{Object.entries(FACTOR_LABELS).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
            </AnalysisControls>
            <div className="analysis-factor-layout">
              <section className="analysis-chart-panel panel">
                <div className="section-heading-inline"><div><span>CONDITIONAL CURVES</span><h2>L{selectedLevel} · {FACTOR_LABELS[selectedFactor]}</h2><p>同一分箱同时显示短、中、长期与最终合成概率；粗线为当前关注指标。</p></div></div>
                <EChart option={factorLineOption} className="analysis-factor-chart" ariaLabel={`L${selectedLevel}${FACTOR_LABELS[selectedFactor]}与合成概率关系折线图`} />
              </section>
              <section className="analysis-chart-panel panel">
                <div className="section-heading-inline"><div><span>ASSOCIATION PROFILE</span><h2>{METRICS[selectedMetric].label}的单因素关联强度</h2><p>加权分箱条件概率的分离程度，不是 Pearson 相关系数，也不代表因果。</p></div></div>
                <EChart option={effectOption} className="analysis-effect-chart" ariaLabel={`L${selectedLevel}${METRICS[selectedMetric].label}的单因素加权关联强度条形图`} />
              </section>
            </div>
            <section className="analysis-table-panel panel">
              <div className="section-heading-inline"><div><span>BIN DETAILS</span><h2>当前因素的分箱明细</h2><p>重复观察已按每颗水果总权重为1归一化。</p></div></div>
              <FactorTable rows={selectedFactorRows} factor={selectedFactor} metric={selectedMetric} />
            </section>
          </motion.div>
        )}

        {activeView === "interaction" && (
          <motion.div key="interaction" className="analysis-view-stack" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={transition}>
            <AnalysisControls levels={levelValues} level={selectedLevel} onLevel={setSelectedLevel} metric={selectedMetric} onMetric={setSelectedMetric}>
              <label><span>因素组合</span><select value={selectedInteraction} onChange={(event) => setSelectedInteraction(event.target.value)}>{Object.entries(INTERACTION_LABELS).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
            </AnalysisControls>
            <section className="analysis-chart-panel panel">
              <div className="section-heading-inline"><div><span>JOINT CONDITIONS</span><h2>L{selectedLevel} · {INTERACTION_LABELS[selectedInteraction]} · {METRICS[selectedMetric].label}</h2><p>每个色块是一组联合条件下的水果归一化合成概率，可直接发现非线性区域和条件反转。</p></div></div>
              <EChart option={interactionOption} className="analysis-interaction-chart" ariaLabel={`L${selectedLevel}${INTERACTION_LABELS[selectedInteraction]}与${METRICS[selectedMetric].label}联合关系热图`} />
            </section>
            <section className="analysis-table-panel panel">
              <div className="section-heading-inline"><div><span>INTERACTION MATRIX</span><h2>联合条件概率表</h2><p>热图与表格使用同一组单元格，便于读取精确数值。</p></div></div>
              <InteractionTable rows={selectedInteractionRows} axes={interactionAxes} metric={selectedMetric} />
            </section>
          </motion.div>
        )}

        {activeView === "table" && (
          <motion.div key="table" className="analysis-view-stack" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={transition}>
            <section className="analysis-raw-panel panel">
              <div className="analysis-raw-toolbar">
                <label><span>数据表</span><select value={rawTable?.id ?? ""} onChange={(event) => { setSelectedTableId(event.target.value); setRawPage(0); setSortColumn("level"); }}>{dataset.tables.map((table) => <option key={table.id} value={table.id}>{table.label} · {formatNumber(table.row_count)}行</option>)}</select></label>
                <label><span>等级</span><select value={rawLevel} onChange={(event) => { setRawLevel(event.target.value); setRawPage(0); }}><option value="all">全部等级</option>{levelValues.map((level) => <option key={level} value={level}>L{level}</option>)}</select></label>
                <label className="analysis-search-field"><span>表内搜索</span><div><Search size={14} /><input value={rawQuery} onChange={(event) => { setRawQuery(event.target.value); setRawPage(0); }} placeholder="因素、分箱或数值" /></div></label>
                <button className={`ghost-button compact ${showAllColumns ? "active" : ""}`} onClick={() => setShowAllColumns((value) => !value)}><Braces size={14} />{showAllColumns ? "核心列" : "全部列"}</button>
              </div>
              <div className="analysis-table-meta"><span><TableProperties size={14} />{formatNumber(filteredRawRows.length)} 行 · {rawColumns.length} 列</span><span>{rawTable?.source}</span></div>
              <div className="analysis-table-scroll">
                <table className="analysis-table analysis-raw-table">
                  <thead><tr>{rawColumns.map((column) => <th key={column}><button onClick={() => { if (sortColumn === column) setSortAscending((value) => !value); else { setSortColumn(column); setSortAscending(true); } }}><span>{FIELD_LABELS[column] ?? column}</span><ArrowDownUp size={11} className={sortColumn === column ? "active" : ""} /></button></th>)}</tr></thead>
                  <tbody>{pagedRawRows.map((row, index) => <tr key={`${safeRawPage}-${index}`}>{rawColumns.map((column) => <td key={column} className={typeof row[column] === "number" ? "numeric" : ""}>{rawCell(row[column], column)}</td>)}</tr>)}</tbody>
                </table>
              </div>
              <div className="analysis-pagination"><span>第 {safeRawPage + 1} / {rawPageCount} 页</span><div><button className="ghost-button compact" disabled={safeRawPage === 0} onClick={() => setRawPage(Math.max(0, safeRawPage - 1))}>上一页</button><button className="ghost-button compact" disabled={safeRawPage + 1 >= rawPageCount} onClick={() => setRawPage(Math.min(rawPageCount - 1, safeRawPage + 1))}>下一页</button></div></div>
            </section>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function AnalysisControls({ levels, level, onLevel, metric, onMetric, children }: {
  levels: number[];
  level: number;
  onLevel: (value: number) => void;
  metric: MetricId;
  onMetric: (value: MetricId) => void;
  children: React.ReactNode;
}) {
  return <section className="analysis-controlbar panel">
    <label><span>水果等级</span><select value={level} onChange={(event) => onLevel(Number(event.target.value))}>{levels.map((item) => <option key={item} value={item}>L{item}</option>)}</select></label>
    {children}
    <label><span>关注指标</span><select value={metric} onChange={(event) => onMetric(event.target.value as MetricId)}>{Object.entries(METRICS).map(([id, definition]) => <option key={id} value={id}>{definition.label}</option>)}</select></label>
  </section>;
}

function LifecycleTable({ rows }: { rows: DataRow[] }) {
  return <div className="analysis-table-scroll"><table className="analysis-table"><thead><tr><th>等级</th><th>生命周期</th><th>最终合成</th><th>终局未合成</th><th>T_merge 中位数</th><th>P90</th><th>P95</th></tr></thead><tbody>{rows.map((row) => <tr key={String(row.level)}><td><b>L{rawCell(row.level, "level")}</b></td><td className="numeric">{rawCell(row.fruits, "fruits")}</td><td className="numeric probability-cell"><i style={{ width: `${(asNumber(row.eventual_merge_probability_resolved) ?? 0) * 100}%` }} />{formatPercent(asNumber(row.eventual_merge_probability_resolved), 2)}</td><td className="numeric">{formatPercent(asNumber(row.terminal_unmerged_probability_resolved), 2)}</td><td className="numeric">{rawCell(row.merged_t_median, "merged_t_median")}</td><td className="numeric">{rawCell(row.merged_t_p90, "merged_t_p90")}</td><td className="numeric">{rawCell(row.merged_t_p95, "merged_t_p95")}</td></tr>)}</tbody></table></div>;
}

function FactorTable({ rows, factor, metric }: { rows: DataRow[]; factor: string; metric: MetricId }) {
  return <div className="analysis-table-scroll"><table className="analysis-table"><thead><tr><th>条件分箱</th><th>归一化水果权重</th><th>8次内</th><th>32次内</th><th>128次内</th><th>最终合成</th><th>终局未合成</th></tr></thead><tbody>{rows.map((row) => <tr key={String(row.bin_index)}><td><b>{factorBinLabel(row, factor)}</b></td><td className="numeric">{formatNumber(asNumber(row.fruit_normalized_weight), 1)}</td>{(["h8", "h32", "h128", "eventual", "terminal"] as MetricId[]).map((id) => <td key={id} className={`numeric ${metric === id ? "selected-metric" : ""}`}>{formatPercent(metricValue(row, id), 2)}</td>)}</tr>)}</tbody></table></div>;
}

function InteractionTable({ rows, axes, metric }: {
  rows: DataRow[];
  axes: { firstBins: number[]; secondBins: number[]; firstLabels: string[]; secondLabels: string[] };
  metric: MetricId;
}) {
  return <div className="analysis-table-scroll"><table className="analysis-table analysis-matrix-table"><thead><tr><th>第一因素</th>{axes.secondLabels.map((label, index) => <th key={`${label}-${index}`}>{label}</th>)}</tr></thead><tbody>{axes.firstBins.map((first, firstIndex) => <tr key={first}><td><b>{axes.firstLabels[firstIndex]}</b></td>{axes.secondBins.map((second) => { const row = rows.find((candidate) => asNumber(candidate.first_bin) === first && asNumber(candidate.second_bin) === second); const value = row ? metricValue(row, metric) : null; return <td key={second} className="numeric matrix-value" style={{ "--cell-alpha": `${(value ?? 0) * 28}%` } as React.CSSProperties}>{formatPercent(value, 1)}</td>; })}</tr>)}</tbody></table></div>;
}
