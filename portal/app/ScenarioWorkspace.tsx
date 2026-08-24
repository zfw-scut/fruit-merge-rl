"use client";

/* eslint-disable @next/next/no-img-element -- 水果纹理由本地物理服务动态返回 data URL。 */

import type { EChartsOption } from "echarts";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertCircle,
  Boxes,
  ChevronDown,
  CirclePause,
  CirclePlay,
  Crosshair,
  Download,
  Eraser,
  FlaskConical,
  Import,
  MousePointer2,
  Pause,
  Play,
  Redo2,
  RefreshCw,
  RotateCcw,
  Settings2,
  Sparkles,
  Trash2,
  Undo2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { EChart } from "./EChart";
import { SceneCanvas } from "./scene/SceneCanvas";

type ProcessSnapshot = {
  running: boolean;
  url: string | null;
};

export type ScenarioTool = {
  id: string;
  process: ProcessSnapshot | null;
};

type Props = {
  tool?: ScenarioTool;
  onConfigure: () => void;
};

type FruitSpec = {
  level: number;
  name: string;
  radius: number;
  dropped_physics_radius: number;
  merged_physics_radius: number;
  merge_score: number;
};

type Geometry = {
  board_width: number;
  board_height: number;
  wall_width: number;
  spawn_y: number;
  action_count: number;
  queue_length: number;
  max_fruits: number;
};

type LabConfig = {
  fruit_specs: FruitSpec[];
  textures: string[];
  geometry: Geometry;
  voronoi?: { algorithm: string; sample_spacing: number; top_boundary: string; obstacle_boundaries: string[] };
};

type Fruit = {
  id: number;
  level: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  angle: number;
  angular_velocity: number;
  age_frames: number;
  physics_radius: number;
};

type LiveState = {
  sequence: number;
  physics_backend?: string;
  physics_device?: string;
  training_physics_equivalent?: boolean;
  physics_fps: number;
  paused: boolean;
  stable: boolean;
  done: boolean;
  danger_progress: number;
  over_danger_line: boolean;
  score: number;
  step_count: number;
  queue: number[];
  fruits: Fruit[];
  physics_frame?: number;
  profile?: ComparisonProfile;
  trace?: { record_index: number; record_count: number; semantic_frame: number; playback_complete: boolean; contains_fast_forward_gap: boolean };
  model_continuous?: { running?: boolean; decision_count?: number; message?: string; error?: string };
};

type Health = {
  ready: boolean;
  reward_version: string;
  device: string;
  live_physics_backend?: string;
  live_physics_device?: string;
  training_physics_equivalent?: boolean;
  model_available: boolean;
  model_continuous_available: boolean;
  comparison_model_continuous_available?: boolean;
  model?: Record<string, unknown> | null;
  comparison_available?: boolean;
  voronoi_available?: boolean;
  voronoi_device?: string;
  pair_risk_available?: boolean;
  pair_risk?: Record<string, unknown> | null;
};

type ComparisonProfile = {
  role: string;
  backend: string;
  device: string;
  physics_fps: number;
  drop_fast_forward: boolean;
  adaptive_collision_substeps: boolean;
  max_collision_substeps: number;
  position_correction: number;
  execution: string;
};

type ComparisonDifference = {
  shared_fruit_count: number;
  left_only_fruit_ids: number[];
  right_only_fruit_ids: number[];
  level_mismatch_fruit_ids: number[];
  queue_equal: boolean;
  score_delta_right_minus_left: number;
  fruit_count_delta_right_minus_left: number;
  max_position_delta: number;
  max_velocity_delta: number;
  max_angle_delta: number;
  max_angular_velocity_delta: number;
  worst_position_fruit_id: number | null;
  discrete_diverged: boolean;
  continuous_diverged: boolean;
  diverged: boolean;
  first_divergence?: { comparison_tick: number; left_physics_frame: number; right_physics_frame: number; discrete: boolean; max_position_delta: number; max_velocity_delta: number } | null;
};

type ComparisonState = {
  sequence: number;
  preset: "backend_parity" | "play_vs_training";
  paused: boolean;
  comparison_tick: number;
  profiles: { left: ComparisonProfile; right: ComparisonProfile };
  left: LiveState;
  right: LiveState;
  difference: ComparisonDifference;
  difference_comparable: boolean;
  action_in_progress: boolean;
  model_continuous?: { running?: boolean; decision_count?: number; message?: string; error?: string };
};

type Point = { x: number; y: number };
type EditTool = "select" | "place" | "erase";
type Prediction = {
  merge?: { probability?: number; count?: { index?: number } };
  q0?: { participated_probability?: number; lineage_depth?: { index?: number }; final_level?: { index?: number }; final?: Point & { vx?: number; vy?: number } };
  first_contact?: { primary?: { label?: string; confidence?: number }; position?: Point; normal?: Point; level_delta?: { value?: number } };
  generations?: Array<{ rank?: number; exists_probability?: number; position?: Point; level?: { index?: number } }>;
  outcome?: { score_delta?: number; fruit_count_delta?: number; stable_probability?: number; terminal_probability?: number };
};

type ModelEvaluation = {
  action: number;
  drop_x: number;
  selected_q: number;
  q_values: number[];
  inference_ms: number;
  action_effect_predictions?: Prediction[] | null;
  model?: Record<string, unknown>;
};

type ActualEffect = {
  merge?: { happened?: boolean; count?: number };
  q0?: { participated?: boolean; lineage_depth?: number; final_level?: number; final?: Point | null };
  first_contact?: { valid?: boolean; primary?: string; position?: Point | null; normal?: Point | null; level_delta?: number | null };
  generations?: Array<{ position?: Point; level?: number }>;
  outcome?: { score_delta?: number; fruit_count_delta?: number; stable?: boolean; terminal?: boolean };
};

type EvaluatedAction = {
  action?: number;
  reward?: number;
  done?: boolean;
  score_delta?: number;
  result_fruits?: Fruit[];
  action_effect?: ActualEffect;
};

type Evaluation = {
  best_action: number;
  selected_action: number;
  actions: EvaluatedAction[];
  message?: string;
};

type VoronoiEdge = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  clearance: number;
  owners: number[];
};

type VoronoiVertex = Point & { clearance: number; owners: number[] };

type VoronoiEvaluation = {
  algorithm: string;
  sampled: boolean;
  sample_spacing: number;
  raster_shape: number[];
  device: string;
  compute_ms: number;
  cache_hit: boolean;
  sites: Array<{ index: number; kind: string; fruit_id?: number; level?: number; visible_samples: number }>;
  edges: VoronoiEdge[];
  vertices: VoronoiVertex[];
  stats: {
    fruit_site_count: number;
    visible_fruit_site_count: number;
    edge_count: number;
    vertex_count: number;
    free_sample_ratio: number;
    min_edge_clearance: number;
    max_edge_clearance: number;
  };
};

type PairRiskPair = {
  fruit_id_i: number;
  fruit_id_j: number;
  level: number;
  probability: number;
};

type PairRiskEvaluation = {
  format_version: number;
  semantics: "onset_within_forecast_horizon";
  forecast_horizon: number;
  inference_ms: number;
  eligible_levels: number[];
  pair_count: number;
  pairs: PairRiskPair[];
  model: Record<string, unknown>;
};

type ViewSettings = {
  showGrid: boolean;
  showDanger: boolean;
  showAnchors: boolean;
  showVelocity: boolean;
  showPrediction: boolean;
  showActual: boolean;
  showNormal: boolean;
  realtimePrediction: boolean;
  showVoronoi: boolean;
  showVoronoiVertices: boolean;
  showPairRisk: boolean;
};

const DEFAULT_GEOMETRY: Geometry = { board_width: 560, board_height: 1120, wall_width: 8, spawn_y: 156, action_count: 21, queue_length: 4, max_fruits: 64 };
const DEFAULT_SETTINGS: ViewSettings = { showGrid: true, showDanger: true, showAnchors: true, showVelocity: false, showPrediction: true, showActual: true, showNormal: true, realtimePrediction: true, showVoronoi: false, showVoronoiVertices: true, showPairRisk: false };
const CONTACT_LABELS: Record<string, string> = { none: "未接触", floor: "地面", left_wall: "左墙", right_wall: "右墙", fruit: "水果", dynamic_fruit: "水果" };

function number(value: unknown, digits = 1) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "—";
}

function percentage(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : "—";
}

function voronoiColor(clearance: number) {
  const scale = Math.max(0, Math.min(1, clearance / 120));
  return `hsl(${12 + scale * 178} 72% ${49 - scale * 8}%)`;
}

function pairRiskColor(probability: number) {
  const risk = Math.max(0, Math.min(1, probability));
  return `hsl(${132 - risk * 127} 72% ${44 + risk * 5}%)`;
}

function pairRiskSceneKey(state: LiveState) {
  const ageQuantum = Math.max(1, Math.round(state.physics_fps * .5));
  return JSON.stringify({
    queue: state.queue,
    danger: Math.round(state.danger_progress * 50),
    over: state.over_danger_line,
    step: state.step_count,
    fruits: state.fruits.map((fruit) => [
      fruit.id,
      fruit.level,
      Math.round(fruit.x * 2),
      Math.round(fruit.y * 2),
      Math.round(fruit.physics_radius * 10),
      Math.floor(fruit.age_frames / ageQuantum),
    ]),
  });
}

function sceneKey(state: LiveState) {
  return JSON.stringify({
    queue: state.queue,
    fruits: state.fruits.map((fruit) => [fruit.id, fruit.level, Math.round(fruit.x * 2), Math.round(fruit.y * 2), Math.round(fruit.vx), Math.round(fruit.vy)]),
    score: state.score,
    step: state.step_count,
  });
}

function voronoiSceneKey(state: LiveState, sampleSpacing = 4) {
  const positionQuantum = Math.max(1, sampleSpacing);
  return JSON.stringify(state.fruits.map((fruit) => [
    fruit.id,
    fruit.level,
    Math.round(fruit.physics_radius * 100),
    Math.round(fruit.x / positionQuantum),
    Math.round(fruit.y / positionQuantum),
  ]));
}

function buildScene(state: LiveState, action: number, fruits = state.fruits, queue = state.queue) {
  return {
    name: "Xigua Atlas 场景",
    fps: state.physics_fps,
    queue,
    probe_action: action,
    score: state.score,
    step_count: state.step_count,
    danger_progress: state.danger_progress,
    over_danger_line: state.over_danger_line,
    fruits: fruits.map((fruit) => ({ ...fruit })),
  };
}

type SceneSnapshot = ReturnType<typeof buildScene>;
type BoardGesture =
  | { kind: "palette" | "place"; pointerId: number; level: number; x: number; y: number; inside: boolean }
  | { kind: "fruit-pending"; pointerId: number; fruitId: number; initial: Point; latest: Point; released: boolean; cancelled: boolean }
  | { kind: "fruit"; pointerId: number; fruitId: number; x: number; y: number; dx: number; dy: number; base: LiveState; autoResume: boolean };

function modelChart(model: ModelEvaluation | null, selectedAction: number): EChartsOption {
  const values = model?.q_values ?? [];
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const span = Math.max(max - min, 1e-6);
  return {
    animationDurationUpdate: 260,
    grid: { left: 48, right: 18, top: 24, bottom: 34 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (value) => Number(value).toFixed(5) },
    xAxis: { type: "category", data: values.map((_, index) => `A${index}`), axisLabel: { fontSize: 9, color: "#7d899b", interval: 1 }, axisTick: { show: false } },
    yAxis: { type: "value", min: Math.floor((min - span * .08) * 100) / 100, max: Math.ceil((max + span * .08) * 100) / 100, splitNumber: 4, axisLabel: { color: "#7d899b", formatter: (value: number) => value.toFixed(2) }, splitLine: { lineStyle: { color: "rgba(77,94,120,.08)" } } },
    series: [{
      type: "bar",
      data: values.map((value, index) => ({ value, itemStyle: { color: index === selectedAction ? "#3478f6" : index === model?.action ? "#18a67e" : "#b9c5d5", borderRadius: [4, 4, 0, 0] } })),
      barMaxWidth: 18,
    }],
  };
}

function settingLabel(label: string, checked: boolean, onChange: (value: boolean) => void) {
  return <label className="lab-switch"><span>{label}</span><input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} /><i /></label>;
}

function ComparisonBoard({ state, geometry, textures, specs, label }: { state: LiveState; geometry: Geometry; textures: string[]; specs: FruitSpec[]; label: string }) {
  const profile = state.profile;
  return <article className="lab-comparison-lane">
    <header><div><span>{label}</span><h3>{profile?.role ?? "物理环境"}</h3></div><div className="lab-comparison-badges"><b>{profile?.backend === "tensor_cuda" ? "CUDA" : "Tensor / CPU"}</b><b>{profile?.physics_fps ?? state.physics_fps} FPS</b><b>完整逐帧</b></div></header>
    <div className="lab-comparison-board-wrap"><SceneCanvas fruits={state.fruits} geometry={geometry} specs={specs} textures={textures} className="lab-comparison-board" ariaLabel={`${label}物理场景`} /></div>
    <footer><span>分数 <b>{state.score}</b></span><span>投放 <b>{state.step_count}</b></span><span>物理帧 <b>{state.physics_frame ?? 0}</b></span><span>水果 <b>{state.fruits.length}</b></span>{state.trace?.contains_fast_forward_gap && <span className="is-warning">快进缺口 · 语义帧 {state.trace.semantic_frame}</span>}</footer>
  </article>;
}

export function ScenarioWorkspace({ tool, onConfigure }: Props) {
  const running = Boolean(tool?.process?.running && tool.process.url);
  const baseUrl = (tool?.process?.url ?? "").replace(/\/$/, "");
  const [config, setConfig] = useState<LabConfig | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const [live, setLive] = useState<LiveState | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedLevel, setSelectedLevel] = useState(1);
  const [selectedFruit, setSelectedFruit] = useState<number | null>(null);
  const [editTool, setEditTool] = useState<EditTool>("place");
  const [selectedAction, setSelectedAction] = useState(10);
  const [model, setModel] = useState<ModelEvaluation | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [voronoi, setVoronoi] = useState<VoronoiEvaluation | null>(null);
  const [voronoiSourceKey, setVoronoiSourceKey] = useState("");
  const [voronoiBusy, setVoronoiBusy] = useState(false);
  const [voronoiError, setVoronoiError] = useState<string | null>(null);
  const [pairRisk, setPairRisk] = useState<PairRiskEvaluation | null>(null);
  const [pairRiskSourceStep, setPairRiskSourceStep] = useState(-1);
  const [pairRiskBusy, setPairRiskBusy] = useState(false);
  const [pairRiskError, setPairRiskError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [canvasMode, setCanvasMode] = useState<"live" | "after">("live");
  const [settings, setSettings] = useState<ViewSettings>(() => {
    if (typeof window === "undefined") return DEFAULT_SETTINGS;
    try { return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem("xigua-atlas-lab-settings") ?? "{}") }; } catch { return DEFAULT_SETTINGS; }
  });
  const predictionKey = useRef("");
  const voronoiKey = useRef("");
  const pairRiskKey = useRef("");
  const boardRef = useRef<SVGSVGElement>(null);
  const liveRef = useRef<LiveState | null>(null);
  const actionRef = useRef(selectedAction);
  const gestureRef = useRef<BoardGesture | null>(null);
  const touchHoldRef = useRef<{ pointerId: number; x: number; y: number; timer: number } | null>(null);
  const [gesture, setGesture] = useState<BoardGesture | null>(null);
  const [hoverPoint, setHoverPoint] = useState<Point | null>(null);
  const [history, setHistory] = useState<SceneSnapshot[]>([]);
  const [future, setFuture] = useState<SceneSnapshot[]>([]);
  const [comparison, setComparison] = useState<ComparisonState | null>(null);
  const [comparisonMode, setComparisonMode] = useState(false);

  const geometry = config?.geometry ?? DEFAULT_GEOMETRY;
  const specs = useMemo(() => config?.fruit_specs ?? [], [config?.fruit_specs]);
  const textures = config?.textures ?? [];
  const voronoiSpacing = config?.voronoi?.sample_spacing ?? 4;
  const liveVoronoiKey = live ? voronoiSceneKey(live, voronoiSpacing) : "";
  const livePairRiskKey = live?.stable ? pairRiskSceneKey(live) : "";
  const voronoiSettleDelay = live?.paused ? 0 : 120;

  useEffect(() => { liveRef.current = live; }, [live]);
  useEffect(() => { actionRef.current = selectedAction; }, [selectedAction]);

  useEffect(() => {
    localStorage.setItem("xigua-atlas-lab-settings", JSON.stringify(settings));
  }, [settings]);

  const api = useCallback(async <T,>(path: string, options?: RequestInit): Promise<T> => {
    const response = await fetch(`${baseUrl}${path}`, { cache: "no-store", ...options });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || payload.message || `HTTP ${response.status}`);
    return payload as T;
  }, [baseUrl]);

  useEffect(() => {
    if (!running || !baseUrl) {
      return;
    }
    let closed = false;
    let source: EventSource | null = null;
    let retryTimer: number | null = null;
    const retry = () => {
      if (!closed) retryTimer = window.setTimeout(connect, 900);
    };
    const connect = async () => {
      try {
        const [nextHealth, nextConfig, nextLive] = await Promise.all([
          api<Health>("/api/health"),
          api<LabConfig>("/api/config"),
          api<LiveState>("/api/live/state"),
        ]);
        if (closed) return;
        liveRef.current = nextLive;
        setHealth(nextHealth); setConfig(nextConfig); setLive(nextLive); setConnected(true); setError(null);
        source?.close();
        source = new EventSource(`${baseUrl}/api/live/events`);
        source.onmessage = (event) => { try { const next = JSON.parse(event.data) as LiveState; liveRef.current = next; setLive(next); setConnected(true); } catch { /* 忽略不完整事件 */ } };
        source.onerror = () => {
          setConnected(false);
          source?.close();
          source = null;
          retry();
        };
        if (nextHealth.comparison_available) {
          try { setComparison(await api<ComparisonState>("/api/comparison/state")); } catch { setComparison(null); }
        }
      } catch (reason) {
        if (!closed) { setError(String(reason)); setConnected(false); retry(); }
      }
    };
    void connect();
    return () => {
      closed = true;
      if (retryTimer !== null) window.clearTimeout(retryTimer);
      source?.close();
    };
  }, [api, baseUrl, running]);

  useEffect(() => {
    if (!running || !baseUrl || !health?.comparison_available) return;
    const source = new EventSource(`${baseUrl}/api/comparison/events`);
    source.onmessage = (event) => { try { setComparison(JSON.parse(event.data) as ComparisonState); } catch { /* ignore */ } };
    return () => source.close();
  }, [baseUrl, health?.comparison_available, running]);

  const sendComparisonCommand = useCallback(async (command: Record<string, unknown>) => {
    const result = await api<Record<string, unknown>>("/api/comparison/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command }) });
    setComparison(await api<ComparisonState>("/api/comparison/state"));
    return result;
  }, [api]);

  const sendCommand = useCallback(async (command: Record<string, unknown>) => {
    const result = await api<Record<string, unknown>>("/api/live/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command }) });
    const next = await api<LiveState>("/api/live/state");
    liveRef.current = next;
    setLive(next);
    return { result, state: next };
  }, [api]);

  const evaluateModel = useCallback(async (quiet = false) => {
    if (!live || !health?.model_available) return;
    if (!quiet) setBusy("model");
    try {
      const result = await api<ModelEvaluation>("/api/model/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scene: buildScene(live, selectedAction) }) });
      setModel(result); predictionKey.current = sceneKey(live);
      if (!quiet) setSelectedAction(result.action);
      setError(null);
    } catch (reason) { setError(String(reason)); } finally { if (!quiet) setBusy(null); }
  }, [api, health?.model_available, live, selectedAction]);

  useEffect(() => {
    if (!settings.realtimePrediction || !live?.stable || !health?.model_available || busy === "model") return;
    const key = sceneKey(live);
    if (key === predictionKey.current) return;
    const timer = window.setTimeout(() => void evaluateModel(true), 260);
    return () => window.clearTimeout(timer);
  }, [busy, evaluateModel, health?.model_available, live, settings.realtimePrediction]);

  const evaluateVoronoi = useCallback(async () => {
    const state = liveRef.current;
    if (!state || !health?.voronoi_available || voronoiBusy) return;
    const key = voronoiSceneKey(state, voronoiSpacing);
    setVoronoiBusy(true);
    try {
      const result = await api<VoronoiEvaluation>("/api/voronoi/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scene: buildScene(state, selectedAction) }) });
      const current = liveRef.current;
      if (!current || voronoiSceneKey(current, voronoiSpacing) !== key) return;
      voronoiKey.current = key;
      setVoronoiSourceKey(key);
      setVoronoi(result);
      setVoronoiError(null);
    } catch (reason) {
      setVoronoiError(String(reason));
    } finally {
      setVoronoiBusy(false);
    }
  }, [api, health?.voronoi_available, selectedAction, voronoiBusy, voronoiSpacing]);

  useEffect(() => {
    if (!settings.showVoronoi || !liveVoronoiKey || !health?.voronoi_available || voronoiBusy) return;
    if (liveVoronoiKey === voronoiKey.current) return;
    const timer = window.setTimeout(() => void evaluateVoronoi(), voronoiSettleDelay);
    return () => window.clearTimeout(timer);
  }, [evaluateVoronoi, health?.voronoi_available, liveVoronoiKey, settings.showVoronoi, voronoiBusy, voronoiSettleDelay]);

  const evaluatePairRisk = useCallback(async () => {
    const state = liveRef.current;
    if (!state?.stable || !health?.pair_risk_available || pairRiskBusy) return;
    const key = pairRiskSceneKey(state);
    setPairRiskBusy(true);
    try {
      const result = await api<PairRiskEvaluation>("/api/pair-risk/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scene: buildScene(state, selectedAction) }) });
      const current = liveRef.current;
      if (!current?.stable || pairRiskSceneKey(current) !== key) return;
      pairRiskKey.current = key;
      setPairRisk(result);
      setPairRiskSourceStep(state.step_count);
      setPairRiskError(null);
    } catch (reason) {
      setPairRiskError(String(reason));
    } finally {
      setPairRiskBusy(false);
    }
  }, [api, health?.pair_risk_available, pairRiskBusy, selectedAction]);

  useEffect(() => {
    if (!settings.showPairRisk || !livePairRiskKey || !health?.pair_risk_available || pairRiskBusy) return;
    if (livePairRiskKey === pairRiskKey.current) return;
    const timer = window.setTimeout(() => void evaluatePairRisk(), 260);
    return () => window.clearTimeout(timer);
  }, [evaluatePairRisk, health?.pair_risk_available, livePairRiskKey, pairRiskBusy, settings.showPairRisk]);

  const evaluatePhysics = async () => {
    if (!live) return;
    setBusy("physics");
    try {
      const result = await api<Evaluation>("/api/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode: "all", scene: buildScene(live, selectedAction) }) });
      setEvaluation(result); setCanvasMode("after"); setError(null);
    } catch (reason) { setError(String(reason)); } finally { setBusy(null); }
  };

  const modelControl = async () => {
    if (!health?.model_continuous_available) return;
    setBusy("control");
    try {
      await api("/api/model/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command: live?.model_continuous?.running ? "stop" : "start" }) });
    } catch (reason) { setError(String(reason)); } finally { setBusy(null); }
  };

  const comparisonModelControl = async () => {
    if (!health?.comparison_model_continuous_available) return;
    setBusy("comparison-control");
    try {
      const running = Boolean(comparison?.model_continuous?.running);
      await api("/api/comparison/model/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command: running ? "stop" : "start" }) });
      setComparison(await api<ComparisonState>("/api/comparison/state"));
      setError(null);
    } catch (reason) { setError(String(reason)); } finally { setBusy(null); }
  };

  const setActiveGesture = useCallback((next: BoardGesture | null) => {
    gestureRef.current = next;
    setGesture(next);
  }, []);

  const clearTouchHold = useCallback(() => {
    const hold = touchHoldRef.current;
    if (hold) window.clearTimeout(hold.timer);
    touchHoldRef.current = null;
  }, []);

  const invalidateDerivedResults = useCallback(() => {
    setEvaluation(null);
    setModel(null);
    setVoronoi(null);
    setVoronoiSourceKey("");
    setVoronoiError(null);
    setCanvasMode("live");
    predictionKey.current = "";
    voronoiKey.current = "";
  }, []);

  const pointFromClient = useCallback((clientX: number, clientY: number) => {
    const rect = boardRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return { point: { x: 0, y: 0 }, inside: false };
    const point = {
      x: (clientX - rect.left) / rect.width * geometry.board_width,
      y: (clientY - rect.top) / rect.height * geometry.board_height,
    };
    return {
      point,
      inside: clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom,
    };
  }, [geometry.board_height, geometry.board_width]);

  const clampPoint = useCallback((point: Point, level: number) => {
    const radius = specs.find((item) => item.level === level)?.radius ?? 20;
    return {
      x: Math.max(geometry.wall_width + radius, Math.min(geometry.board_width - geometry.wall_width - radius, point.x)),
      y: Math.max(radius, Math.min(geometry.board_height - geometry.wall_width - radius, point.y)),
    };
  }, [geometry.board_height, geometry.board_width, geometry.wall_width, specs]);

  const hitFruit = useCallback((point: Point, state = liveRef.current) => {
    if (!state) return null;
    for (let index = state.fruits.length - 1; index >= 0; index -= 1) {
      const fruit = state.fruits[index];
      const radius = specs.find((item) => item.level === fruit.level)?.radius ?? fruit.physics_radius;
      if (Math.hypot(point.x - fruit.x, point.y - fruit.y) <= radius) return fruit;
    }
    return null;
  }, [specs]);

  const rememberScene = useCallback((state: LiveState) => {
    const snapshot = buildScene(state, actionRef.current);
    const encoded = JSON.stringify(snapshot);
    setHistory((current) => {
      if (current.length && JSON.stringify(current[current.length - 1]) === encoded) return current;
      return [...current.slice(-79), snapshot];
    });
    setFuture([]);
  }, []);

  const loadEditedScene = useCallback(async (scene: SceneSnapshot, remember = true) => {
    const current = liveRef.current;
    if (remember && current) rememberScene(current);
    const { state } = await sendCommand({ type: "load_scene", scene, paused: true });
    invalidateDerivedResults();
    return state;
  }, [invalidateDerivedResults, rememberScene, sendCommand]);

  const placeFruit = useCallback(async (level: number, point: Point) => {
    const state = liveRef.current;
    if (!state || state.model_continuous?.running) return;
    const target = clampPoint(point, level);
    try {
      if (!state.paused) {
        await sendCommand({ type: "drop", level, x: target.x });
        invalidateDerivedResults();
        return;
      }
      if (state.fruits.length >= geometry.max_fruits) throw new Error(`场景最多包含 ${geometry.max_fruits} 个水果`);
      const spec = specs.find((item) => item.level === level);
      const fruit: Fruit = {
        id: Math.max(0, ...state.fruits.map((item) => item.id)) + 1,
        level,
        x: target.x,
        y: target.y,
        vx: 0,
        vy: 0,
        angle: 0,
        angular_velocity: 0,
        age_frames: 0,
        physics_radius: spec?.merged_physics_radius ?? spec?.radius ?? 20,
      };
      await loadEditedScene(buildScene(state, actionRef.current, [...state.fruits, fruit]));
      setSelectedFruit(fruit.id);
    } catch (reason) {
      setError(String(reason));
    }
  }, [clampPoint, geometry.max_fruits, invalidateDerivedResults, loadEditedScene, sendCommand, specs]);

  const removeFruitById = useCallback(async (fruitId: number) => {
    const state = liveRef.current;
    if (!state || state.model_continuous?.running) return;
    try {
      if (state.paused) rememberScene(state);
      await sendCommand({ type: "remove", fruit_id: fruitId });
      if (selectedFruit === fruitId) setSelectedFruit(null);
      invalidateDerivedResults();
    } catch (reason) {
      setError(String(reason));
    }
  }, [invalidateDerivedResults, rememberScene, selectedFruit, sendCommand]);

  const cycleFruitLevel = useCallback(async (direction: number, fruit: Fruit | null) => {
    const state = liveRef.current;
    if (!state) return;
    if (!fruit || !state.paused) {
      setSelectedLevel((current) => 1 + ((current - 1 + direction + 11) % 11));
      return;
    }
    const level = 1 + ((fruit.level - 1 + direction + 11) % 11);
    const spec = specs.find((item) => item.level === level);
    const point = clampPoint({ x: fruit.x, y: fruit.y }, level);
    const fruits = state.fruits.map((item) => item.id === fruit.id ? {
      ...item,
      level,
      x: point.x,
      y: point.y,
      vx: 0,
      vy: 0,
      angular_velocity: 0,
      physics_radius: spec?.merged_physics_radius ?? spec?.radius ?? item.physics_radius,
    } : item);
    try {
      await loadEditedScene(buildScene(state, actionRef.current, fruits));
      setSelectedLevel(level);
      setSelectedFruit(fruit.id);
    } catch (reason) {
      setError(String(reason));
    }
  }, [clampPoint, loadEditedScene, specs]);

  const finishFruitGesture = useCallback(async (active: Extract<BoardGesture, { kind: "fruit" }>, cancelled = false) => {
    clearTouchHold();
    setActiveGesture(null);
    try {
      if (!cancelled) {
        const fruits = active.base.fruits.map((fruit) => fruit.id === active.fruitId ? {
          ...fruit,
          x: active.x,
          y: active.y,
          vx: 0,
          vy: 0,
          angular_velocity: 0,
        } : fruit);
        rememberScene(active.base);
        await sendCommand({ type: "load_scene", scene: buildScene(active.base, actionRef.current, fruits), paused: true });
        invalidateDerivedResults();
      }
    } catch (reason) {
      setError(String(reason));
    } finally {
      if (active.autoResume) {
        try { await sendCommand({ type: "resume" }); } catch (reason) { setError(String(reason)); }
      }
    }
  }, [clearTouchHold, invalidateDerivedResults, rememberScene, sendCommand, setActiveGesture]);

  const beginTransientFruitGesture = useCallback(async (pending: Extract<BoardGesture, { kind: "fruit-pending" }>) => {
    try {
      const { state } = await sendCommand({ type: "pause" });
      const current = gestureRef.current;
      if (!current || current.kind !== "fruit-pending" || current.pointerId !== pending.pointerId) {
        await sendCommand({ type: "resume" });
        return;
      }
      const fruit = state.fruits.find((item) => item.id === pending.fruitId);
      if (!fruit) throw new Error("水果已在临时暂停前合成或消失");
      const target = clampPoint({
        x: current.latest.x + fruit.x - current.initial.x,
        y: current.latest.y + fruit.y - current.initial.y,
      }, fruit.level);
      const active: Extract<BoardGesture, { kind: "fruit" }> = {
        kind: "fruit",
        pointerId: current.pointerId,
        fruitId: fruit.id,
        x: target.x,
        y: target.y,
        dx: fruit.x - current.initial.x,
        dy: fruit.y - current.initial.y,
        base: state,
        autoResume: true,
      };
      setActiveGesture(active);
      if (current.released) await finishFruitGesture(active, current.cancelled);
    } catch (reason) {
      setActiveGesture(null);
      setError(String(reason));
      try { await sendCommand({ type: "resume" }); } catch { /* 保留原始错误。 */ }
    }
  }, [clampPoint, finishFruitGesture, sendCommand, setActiveGesture]);

  const finishDropGesture = useCallback(async (cancelled = false) => {
    const current = gestureRef.current;
    if (!current || (current.kind !== "palette" && current.kind !== "place")) return;
    setActiveGesture(null);
    if (!cancelled && current.inside) await placeFruit(current.level, { x: current.x, y: current.y });
  }, [placeFruit, setActiveGesture]);

  const onPalettePointerDown = useCallback((event: React.PointerEvent<HTMLButtonElement>, level: number) => {
    if (event.button !== 0 && event.pointerType === "mouse") return;
    if (gestureRef.current || !liveRef.current || liveRef.current.model_continuous?.running || canvasMode !== "live") return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setSelectedLevel(level);
    setEditTool("place");
    const { point, inside } = pointFromClient(event.clientX, event.clientY);
    setActiveGesture({ kind: "palette", pointerId: event.pointerId, level, x: point.x, y: point.y, inside });
  }, [canvasMode, pointFromClient, setActiveGesture]);

  const onPalettePointerMove = useCallback((event: React.PointerEvent<HTMLButtonElement>) => {
    const current = gestureRef.current;
    if (!current || current.kind !== "palette" || current.pointerId !== event.pointerId) return;
    const { point, inside } = pointFromClient(event.clientX, event.clientY);
    setActiveGesture({ ...current, x: point.x, y: point.y, inside });
  }, [pointFromClient, setActiveGesture]);

  const onFruitPointerDown = useCallback((event: React.PointerEvent<SVGGElement>, fruit: Fruit) => {
    if (canvasMode !== "live" || liveRef.current?.model_continuous?.running) return;
    if (gestureRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    setSelectedFruit(fruit.id);
    if ((event.button === 2 && event.pointerType === "mouse") || editTool === "erase") {
      void removeFruitById(fruit.id);
      return;
    }
    if (event.button !== 0 && event.pointerType === "mouse") return;
    const state = liveRef.current;
    if (!state) return;
    boardRef.current?.setPointerCapture(event.pointerId);
    const { point } = pointFromClient(event.clientX, event.clientY);
    if (!state.paused) {
      const pending: Extract<BoardGesture, { kind: "fruit-pending" }> = { kind: "fruit-pending", pointerId: event.pointerId, fruitId: fruit.id, initial: point, latest: point, released: false, cancelled: false };
      setActiveGesture(pending);
      void beginTransientFruitGesture(pending);
      return;
    }
    const active: Extract<BoardGesture, { kind: "fruit" }> = { kind: "fruit", pointerId: event.pointerId, fruitId: fruit.id, x: fruit.x, y: fruit.y, dx: fruit.x - point.x, dy: fruit.y - point.y, base: state, autoResume: false };
    setActiveGesture(active);
    if (event.pointerType === "touch") {
      clearTouchHold();
      const timer = window.setTimeout(() => {
        const current = gestureRef.current;
        if (current?.kind === "fruit" && current.pointerId === event.pointerId) {
          setActiveGesture(null);
          void removeFruitById(fruit.id);
        }
      }, 650);
      touchHoldRef.current = { pointerId: event.pointerId, x: point.x, y: point.y, timer };
    }
  }, [beginTransientFruitGesture, canvasMode, clearTouchHold, editTool, pointFromClient, removeFruitById, setActiveGesture]);

  const onBoardPointerDown = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
    if (canvasMode !== "live" || liveRef.current?.model_continuous?.running) return;
    if (gestureRef.current) return;
    if (event.button !== 0 && event.pointerType === "mouse") return;
    event.preventDefault();
    const { point, inside } = pointFromClient(event.clientX, event.clientY);
    if (editTool === "select") {
      setSelectedFruit(null);
      return;
    }
    if (editTool === "erase") return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setActiveGesture({ kind: "place", pointerId: event.pointerId, level: selectedLevel, x: point.x, y: point.y, inside });
  }, [canvasMode, editTool, pointFromClient, selectedLevel, setActiveGesture]);

  const onBoardPointerMove = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
    const { point, inside } = pointFromClient(event.clientX, event.clientY);
    setHoverPoint(inside ? point : null);
    const hold = touchHoldRef.current;
    if (hold?.pointerId === event.pointerId && Math.hypot(point.x - hold.x, point.y - hold.y) > 8) clearTouchHold();
    const current = gestureRef.current;
    if (!current || current.pointerId !== event.pointerId) return;
    if (current.kind === "place" || current.kind === "palette") {
      setActiveGesture({ ...current, x: point.x, y: point.y, inside });
    } else if (current.kind === "fruit-pending") {
      setActiveGesture({ ...current, latest: point });
    } else if (current.kind === "fruit") {
      const fruit = current.base.fruits.find((item) => item.id === current.fruitId);
      if (!fruit) return;
      const target = clampPoint({ x: point.x + current.dx, y: point.y + current.dy }, fruit.level);
      setActiveGesture({ ...current, x: target.x, y: target.y });
    }
  }, [clampPoint, clearTouchHold, pointFromClient, setActiveGesture]);

  const onBoardPointerUp = useCallback(async (event: React.PointerEvent<SVGSVGElement>) => {
    clearTouchHold();
    const current = gestureRef.current;
    if (!current || current.pointerId !== event.pointerId) return;
    if (current.kind === "place" || current.kind === "palette") await finishDropGesture(false);
    else if (current.kind === "fruit") await finishFruitGesture(current, false);
    else if (current.kind === "fruit-pending") setActiveGesture({ ...current, released: true });
  }, [clearTouchHold, finishDropGesture, finishFruitGesture, setActiveGesture]);

  const onBoardPointerCancel = useCallback(async (event: React.PointerEvent<SVGSVGElement>) => {
    clearTouchHold();
    const current = gestureRef.current;
    if (!current || current.pointerId !== event.pointerId) return;
    if (current.kind === "place" || current.kind === "palette") await finishDropGesture(true);
    else if (current.kind === "fruit") await finishFruitGesture(current, true);
    else if (current.kind === "fruit-pending") setActiveGesture({ ...current, released: true, cancelled: true });
  }, [clearTouchHold, finishDropGesture, finishFruitGesture, setActiveGesture]);

  const onBoardWheel = useCallback((event: React.WheelEvent<SVGSVGElement>) => {
    event.preventDefault();
    if (canvasMode !== "live" || liveRef.current?.model_continuous?.running) return;
    const { point } = pointFromClient(event.clientX, event.clientY);
    void cycleFruitLevel(event.deltaY > 0 ? 1 : -1, hitFruit(point));
  }, [canvasMode, cycleFruitLevel, hitFruit, pointFromClient]);

  const removeSelected = async () => {
    if (selectedFruit !== null) await removeFruitById(selectedFruit);
  };

  const undo = useCallback(async () => {
    const state = liveRef.current;
    if (!state?.paused || !history.length) return;
    const previous = history[history.length - 1];
    const current = buildScene(state, actionRef.current);
    setHistory((items) => items.slice(0, -1));
    setFuture((items) => [current, ...items].slice(0, 80));
    try { await loadEditedScene(previous, false); } catch (reason) { setError(String(reason)); }
  }, [history, loadEditedScene]);

  const redo = useCallback(async () => {
    const state = liveRef.current;
    if (!state?.paused || !future.length) return;
    const [next, ...rest] = future;
    setHistory((items) => [...items.slice(-79), buildScene(state, actionRef.current)]);
    setFuture(rest);
    try { await loadEditedScene(next, false); } catch (reason) { setError(String(reason)); }
  }, [future, loadEditedScene]);

  const togglePause = async () => {
    if (!live) return;
    const current = gestureRef.current;
    if (current?.kind === "fruit") await finishFruitGesture(current, true);
    else if (current?.kind === "place" || current?.kind === "palette") await finishDropGesture(true);
    if (live.paused && editTool === "erase") setEditTool("place");
    await sendCommand({ type: live.paused ? "resume" : "pause" });
  };

  const updateQueue = async (index: number, delta: number) => {
    if (!live?.paused) return;
    const queue = [...live.queue]; queue[index] = 1 + ((queue[index] - 1 + delta + 5) % 5);
    await loadEditedScene(buildScene(live, selectedAction, live.fruits, queue));
  };

  const loadPreset = async (kind: "empty" | "stack" | "gap") => {
    if (!live) return;
    const fruits: Fruit[] = kind === "empty" ? [] : kind === "stack"
      ? [1, 2, 3, 4, 5, 6].map((level, index) => ({ id: index + 1, level, x: 150 + (index % 3) * 130, y: geometry.board_height - 50 - Math.floor(index / 3) * 115, vx: 0, vy: 0, angle: 0, angular_velocity: 0, age_frames: 20, physics_radius: specs.find((item) => item.level === level)?.merged_physics_radius ?? 20 }))
      : [1, 2, 2, 3].map((level, index) => ({ id: index + 1, level, x: [70, 170, 390, 490][index], y: geometry.board_height - 50 - (index % 2) * 95, vx: 0, vy: 0, angle: 0, angular_velocity: 0, age_frames: 20, physics_radius: specs.find((item) => item.level === level)?.merged_physics_radius ?? 20 }));
    await loadEditedScene(buildScene(live, selectedAction, fruits));
  };

  const exportScene = () => {
    if (!live) return;
    const blob = new Blob([JSON.stringify(buildScene(live, selectedAction), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = "xigua-scenario.json"; anchor.click(); URL.revokeObjectURL(url);
  };

  const importScene = async (file: File) => {
    if (!live) return;
    const scene = JSON.parse(await file.text());
    await loadEditedScene(scene as SceneSnapshot);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && /INPUT|SELECT|TEXTAREA/.test(target.tagName)) return;
      if (event.key === "Escape") {
        if (settingsOpen) { setSettingsOpen(false); return; }
        const current = gestureRef.current;
        if (current?.kind === "fruit") void finishFruitGesture(current, true);
        else if (current?.kind === "place" || current?.kind === "palette") void finishDropGesture(true);
        else if (current?.kind === "fruit-pending") setActiveGesture({ ...current, released: true, cancelled: true });
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) void redo(); else void undo();
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        void redo();
        return;
      }
      if ((event.key === "Delete" || event.key === "Backspace") && selectedFruit !== null) {
        event.preventDefault();
        void removeFruitById(selectedFruit);
        return;
      }
      if (event.key.toLowerCase() === "v") setEditTool("select");
      if (event.key.toLowerCase() === "b") setEditTool("place");
      if (event.key.toLowerCase() === "e" && liveRef.current?.paused) setEditTool("erase");
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [finishDropGesture, finishFruitGesture, redo, removeFruitById, selectedFruit, setActiveGesture, settingsOpen, undo]);

  const selectedEvaluation = evaluation?.actions?.[selectedAction];
  const prediction = model?.action_effect_predictions?.[selectedAction] ?? null;
  const actual = selectedEvaluation?.action_effect ?? null;
  const afterFruits = selectedEvaluation?.result_fruits;
  const displayFruits = canvasMode === "after" && afterFruits?.length ? afterFruits : live?.fruits ?? [];
  const displayedVoronoi = settings.showVoronoi && canvasMode === "live" && voronoiSourceKey === liveVoronoiKey ? voronoi : null;
  const displayedPairRisk = settings.showPairRisk && canvasMode === "live" && live?.stable && pairRiskSourceStep === live.step_count ? pairRisk : null;
  const pairRiskByFruit = new Map<number, PairRiskPair[]>();
  for (const pair of displayedPairRisk?.pairs ?? []) {
    pairRiskByFruit.set(pair.fruit_id_i, [...(pairRiskByFruit.get(pair.fruit_id_i) ?? []), pair]);
    pairRiskByFruit.set(pair.fruit_id_j, [...(pairRiskByFruit.get(pair.fruit_id_j) ?? []), pair]);
  }
  const qChart = useMemo(() => modelChart(model, selectedAction), [model, selectedAction]);
  const selectedSpec = specs.find((item) => item.level === selectedLevel);
  const left = geometry.wall_width + (selectedSpec?.radius ?? 20) + 2;
  const right = geometry.board_width - geometry.wall_width - (selectedSpec?.radius ?? 20) - 2;
  const actionX = left + (right - left) * selectedAction / Math.max(1, geometry.action_count - 1);
  const dropGesture = gesture?.kind === "palette" || gesture?.kind === "place" ? gesture : null;
  const ghostPoint = dropGesture?.inside ? clampPoint({ x: dropGesture.x, y: live?.paused ? dropGesture.y : geometry.spawn_y }, dropGesture.level) : null;
  const ghostSpec = dropGesture ? specs.find((item) => item.level === dropGesture.level) : null;
  const interactionHint = gesture?.kind === "fruit-pending"
    ? "正在临时暂停物理，松开后会自动同步并恢复"
    : gesture?.kind === "fruit"
      ? gesture.autoResume ? "拖动调整位置；松开后同步并恢复实时物理" : "拖动调整位置；松开保存，Esc取消"
      : dropGesture
        ? dropGesture.inside ? `松开即${live?.paused ? "放置" : "投放"} ${ghostSpec?.name ?? `L${dropGesture.level}`}` : "拖入场景后松开放置"
        : live?.paused
          ? editTool === "place" ? "空白处按住拖动放置 · 水果可拖动 · 右键删除 · 滚轮切级" : editTool === "erase" ? "点击水果删除 · 触摸长按也可删除" : "选择水果后拖动 · 右键删除 · Delete删除选中"
          : "从左侧拖入或在空白处拖动投放 · 按住水果可临时编辑 · 右键删除";

  if (!running) {
    return <section className="lab-empty-state"><FlaskConical size={44} /><span>SCENARIO LAB</span><h2>场景实验室后端尚未启动</h2><p>在统一设置中选择物理设备、模型checkpoint和奖励缩放；启动后将在本页直接进入实验，不再打开旧页面。</p><button className="primary-button" onClick={onConfigure}><Settings2 size={16} /> 配置并启动</button></section>;
  }

  return (
    <section className="scenario-workspace">
      <div className="lab-commandbar">
        <div><span className={`training-live-indicator ${connected ? "is-online" : ""}`} /><b>{connected ? "实时物理已连接" : "正在重连"}</b><span>{health?.reward_version ?? "Reward V2.1"} · {health?.device ?? "—"}</span></div>
        <div>{health?.comparison_available && <button className={comparisonMode ? "active" : ""} onClick={() => setComparisonMode((value) => !value)}><Boxes size={15} />{comparisonMode ? "返回普通场景" : "双环境对照"}</button>}<button disabled={Boolean(gesture)} onClick={() => comparisonMode ? void sendComparisonCommand({ type: comparison?.paused ? "resume" : "pause" }) : void togglePause()}>{(comparisonMode ? comparison?.paused : live?.paused) ? <CirclePlay size={15} /> : <CirclePause size={15} />}{(comparisonMode ? comparison?.paused : live?.paused) ? "恢复" : "暂停"}</button><button onClick={() => setSettingsOpen(true)}><Settings2 size={15} /> 显示设置</button></div>
      </div>

      {error && <div className="lab-error"><AlertCircle size={16} />{error}<button onClick={() => setError(null)}><X size={14} /></button></div>}

      {comparisonMode && comparison ? <div className="lab-comparison-mode">
        <div className="lab-comparison-toolbar"><div><button className={comparison.preset === "play_vs_training" ? "active" : ""} disabled={comparison.model_continuous?.running} onClick={() => void sendComparisonCommand({ type: "set_preset", preset: "play_vs_training" })}>游玩 vs 训练</button><button className={comparison.preset === "backend_parity" ? "active" : ""} disabled={comparison.model_continuous?.running} onClick={() => void sendComparisonCommand({ type: "set_preset", preset: "backend_parity" })}>Tensor vs CUDA</button></div><span>同一命令同步驱动 · 按水果 ID 对齐 · {comparison.model_continuous?.running ? `模型持续决策 ${comparison.model_continuous.decision_count ?? 0} 次` : comparison.action_in_progress ? "训练轨迹回放中" : "等待操作"}</span></div>
        <div className="lab-comparison-actions"><label>A{selectedAction}<input type="range" min="0" max={geometry.action_count - 1} value={selectedAction} disabled={comparison.model_continuous?.running} onChange={(event) => setSelectedAction(Number(event.target.value))} /></label><button className="primary-button compact" disabled={comparison.action_in_progress || comparison.model_continuous?.running} onClick={() => void sendComparisonCommand({ type: "drop_action", action: selectedAction })}><Play size={15} />同步执行 A{selectedAction}</button><button disabled={!health?.comparison_model_continuous_available || Boolean(busy)} onClick={() => void comparisonModelControl()}>{comparison.model_continuous?.running ? <Pause size={15} /> : <Activity size={15} />}{comparison.model_continuous?.running ? "停止模型持续决策" : "启动模型持续决策"}</button><button disabled={comparison.model_continuous?.running} onClick={() => void sendComparisonCommand({ type: "clear", queue: [1, 2, 3, 4] })}><RotateCcw size={14} />清空并重置差异</button></div>
        <div className="lab-comparison-grid"><ComparisonBoard state={comparison.left} geometry={geometry} textures={textures} specs={specs} label="LEFT" /><ComparisonBoard state={comparison.right} geometry={geometry} textures={textures} specs={specs} label="RIGHT" /></div>
        <section className={`lab-difference-panel${comparison.difference.diverged ? " is-diverged" : ""}`}><header><div><span>LIVE PHYSICS DIFF</span><h3>{!comparison.difference_comparable ? "等待同一物理时刻" : comparison.difference.diverged ? "轨迹已经分歧" : "当前状态一致"}</h3></div><b>{comparison.preset === "play_vs_training" ? "实际配置差异" : "后端实现校验"}</b></header><div><span>最大位置差<b>{comparison.difference_comparable ? `${number(comparison.difference.max_position_delta, 3)} px` : "—"}</b></span><span>最大速度差<b>{comparison.difference_comparable ? `${number(comparison.difference.max_velocity_delta, 3)} px/s` : "—"}</b></span><span>最大角度差<b>{comparison.difference_comparable ? `${number(comparison.difference.max_angle_delta, 6)} rad` : "—"}</b></span><span>最大角速度差<b>{comparison.difference_comparable ? `${number(comparison.difference.max_angular_velocity_delta, 6)} rad/s` : "—"}</b></span><span>分数差（右−左）<b>{comparison.difference_comparable ? comparison.difference.score_delta_right_minus_left : "—"}</b></span><span>离散状态<b>{comparison.difference_comparable ? comparison.difference.discrete_diverged ? "已分歧" : "一致" : "未对齐"}</b></span></div>{comparison.difference.first_divergence && <p>首次分歧：对照 tick {comparison.difference.first_divergence.comparison_tick} · 左帧 {comparison.difference.first_divergence.left_physics_frame} / 右帧 {comparison.difference.first_divergence.right_physics_frame} · 当时位置差 {number(comparison.difference.first_divergence.max_position_delta, 3)} px、速度差 {number(comparison.difference.first_divergence.max_velocity_delta, 3)} px/s。</p>}</section>
      </div> : <div className="lab-layout">
        <aside className="lab-palette">
          <div className="lab-section-title"><span>FRUIT PALETTE</span><h3>投放水果</h3></div>
          <div className="lab-fruit-grid">
            {specs.map((spec) => <button key={spec.level} data-level={spec.level} title={`${spec.name} · 拖到画布后松开放置`} className={`${selectedLevel === spec.level ? "active" : ""}${gesture?.kind === "palette" && gesture.level === spec.level ? " dragging" : ""}`} onClick={() => { setSelectedLevel(spec.level); setEditTool("place"); }} onPointerDown={(event) => onPalettePointerDown(event, spec.level)} onPointerMove={onPalettePointerMove} onPointerUp={() => void finishDropGesture(false)} onPointerCancel={() => void finishDropGesture(true)}><img src={textures[spec.level]} alt="" /><span>{spec.name}</span><small>L{spec.level}</small></button>)}
          </div>
          <details className="lab-collapsible" open><summary><ChevronDown size={14} /> 队列 q0～q3</summary><div className="lab-queue-list">{(live?.queue ?? [1, 2, 3, 4]).map((level, index) => <button key={index} disabled={!live?.paused} onClick={() => void updateQueue(index, 1)} onContextMenu={(event) => { event.preventDefault(); void updateQueue(index, -1); }}><span>q{index}</span><img src={textures[level]} alt="" /><b>{specs.find((item) => item.level === level)?.name ?? `L${level}`}</b></button>)}</div></details>
          <details className="lab-collapsible"><summary><ChevronDown size={14} /> 场景预设</summary><div className="lab-preset-list"><button onClick={() => void loadPreset("empty")}><RotateCcw size={14} />空场景</button><button onClick={() => void loadPreset("stack")}><Boxes size={14} />同级堆积</button><button onClick={() => void loadPreset("gap")}><Crosshair size={14} />底层缺口</button></div></details>
        </aside>

        <main className="lab-canvas-stage">
          <div className="lab-canvas-toolbar">
            <div><button className={canvasMode === "live" ? "active" : ""} onClick={() => setCanvasMode("live")}>实时场景</button><button disabled={!evaluation} className={canvasMode === "after" ? "active" : ""} onClick={() => setCanvasMode("after")}>动作后结果</button></div>
            <span>分数 {live?.score ?? 0} · 投放 {live?.step_count ?? 0} · {live?.stable ? "已稳定" : "运动中"}</span>
          </div>
          <div className="lab-edit-toolbar" role="toolbar" aria-label="场景编辑工具">
            <button aria-pressed={editTool === "select"} className={editTool === "select" ? "active" : ""} disabled={canvasMode !== "live" || Boolean(live?.model_continuous?.running)} onClick={() => setEditTool("select")} title="选择工具（V）"><MousePointer2 size={14} />选择</button>
            <button aria-pressed={editTool === "place"} className={editTool === "place" ? "active" : ""} disabled={canvasMode !== "live" || Boolean(live?.model_continuous?.running)} onClick={() => setEditTool("place")} title="放置工具（B）"><Crosshair size={14} />放置</button>
            <button aria-pressed={editTool === "erase"} className={editTool === "erase" ? "active" : ""} disabled={canvasMode !== "live" || !live?.paused || Boolean(live?.model_continuous?.running)} onClick={() => setEditTool("erase")} title="擦除工具（E）"><Eraser size={14} />擦除</button>
            <i />
            <button disabled={!live?.paused || !history.length || Boolean(gesture)} onClick={() => void undo()} title="撤销（Ctrl+Z）"><Undo2 size={14} />撤销</button>
            <button disabled={!live?.paused || !future.length || Boolean(gesture)} onClick={() => void redo()} title="重做（Ctrl+Y）"><Redo2 size={14} />重做</button>
          </div>
          <div className="lab-board-wrap">
            <svg ref={boardRef} data-testid="lab-board" className={`lab-board is-${editTool}${gesture ? " is-dragging" : ""}`} viewBox={`0 0 ${geometry.board_width} ${geometry.board_height}`} onPointerDown={onBoardPointerDown} onPointerMove={onBoardPointerMove} onPointerUp={(event) => void onBoardPointerUp(event)} onPointerCancel={(event) => void onBoardPointerCancel(event)} onPointerLeave={() => { if (!gestureRef.current) setHoverPoint(null); }} onContextMenu={(event) => event.preventDefault()} onWheel={onBoardWheel} aria-label="合成大西瓜实时物理场景">
              <rect data-board="drop" width={geometry.board_width} height={geometry.board_height} className="lab-board-bg" />
              {settings.showGrid && Array.from({ length: 10 }, (_, index) => <line key={`v${index}`} x1={(index + 1) * geometry.board_width / 11} x2={(index + 1) * geometry.board_width / 11} y1="0" y2={geometry.board_height} className="lab-grid-line" />)}
              {settings.showGrid && Array.from({ length: 11 }, (_, index) => <line key={`h${index}`} y1={(index + 1) * geometry.board_height / 12} y2={(index + 1) * geometry.board_height / 12} x1="0" x2={geometry.board_width} className="lab-grid-line" />)}
              {settings.showDanger && <g><line x1={geometry.wall_width} x2={geometry.board_width - geometry.wall_width} y1={geometry.spawn_y} y2={geometry.spawn_y} className="lab-danger-line" /><text x={geometry.wall_width + 8} y={geometry.spawn_y - 10}>危险线</text></g>}
              {displayedVoronoi && <g className="lab-voronoi-layer" aria-label="完整圆障碍加权 Voronoi 图">
                {displayedVoronoi.edges.map((edge, index) => <line key={`ve-${index}`} x1={edge.x1} y1={edge.y1} x2={edge.x2} y2={edge.y2} stroke={voronoiColor(edge.clearance)} />)}
                {settings.showVoronoiVertices && displayedVoronoi.vertices.map((vertex, index) => <circle key={`vv-${index}`} cx={vertex.x} cy={vertex.y} r={3.4} fill={voronoiColor(vertex.clearance)}><title>{`交汇点 · clearance ${vertex.clearance.toFixed(1)} · ${vertex.owners.length}个站点`}</title></circle>)}
              </g>}
              {settings.showAnchors && Array.from({ length: geometry.action_count }, (_, index) => { const x = left + (right - left) * index / Math.max(1, geometry.action_count - 1); return <g key={index} className={index === selectedAction ? "lab-anchor is-selected" : "lab-anchor"}><line x1={x} x2={x} y1={geometry.spawn_y - 28} y2={geometry.spawn_y + 28} /><text x={x} y={geometry.spawn_y - 38}>A{index}</text></g>; })}
              <line x1={actionX} x2={actionX} y1="0" y2={geometry.spawn_y} className="lab-probe-line" />
              {displayFruits.map((fruit) => { const draft = gesture?.kind === "fruit" && gesture.fruitId === fruit.id ? gesture : fruit; const radius = specs.find((item) => item.level === fruit.level)?.radius ?? fruit.physics_radius; const risks = pairRiskByFruit.get(fruit.id) ?? []; const riskTitle = risks.length ? risks.map((pair) => { const otherId = pair.fruit_id_i === fruit.id ? pair.fruit_id_j : pair.fruit_id_i; return `与 #${otherId}：未来${displayedPairRisk?.forecast_horizon ?? 24}次投放内堵塞起始风险 ${(pair.probability * 100).toFixed(1)}%`; }).join("\n") : `L${fruit.level} #${fruit.id}`; return <g key={fruit.id} data-fruit-id={fruit.id} transform={`translate(${draft.x} ${draft.y}) rotate(${fruit.angle * 180 / Math.PI})`} className={`${selectedFruit === fruit.id ? "lab-fruit is-selected" : "lab-fruit"}${gesture?.kind === "fruit" && gesture.fruitId === fruit.id ? " is-dragging" : ""}`} onPointerDown={(event) => onFruitPointerDown(event, fruit)} onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); }}><title>{riskTitle}</title><circle r={radius + 4} /><image href={textures[fruit.level]} x={-radius} y={-radius} width={radius * 2} height={radius * 2} /><text y={4}>L{fruit.level}</text>{settings.showVelocity && <line x1="0" y1="0" x2={fruit.vx * .05} y2={fruit.vy * .05} className="lab-velocity" />}</g>; })}
              {displayedPairRisk && <g className="lab-pair-risk-layer" aria-label={`未来${displayedPairRisk.forecast_horizon}次投放内的水果对堵塞起始风险`}>
                {[...displayedPairRisk.pairs].sort((first, second) => first.probability - second.probability).map((pair) => { const first = displayFruits.find((fruit) => fruit.id === pair.fruit_id_i); const second = displayFruits.find((fruit) => fruit.id === pair.fruit_id_j); if (!first || !second) return null; const x = (first.x + second.x) * .5; const y = (first.y + second.y) * .5; const color = pairRiskColor(pair.probability); return <g key={`${pair.fruit_id_i}-${pair.fruit_id_j}`}><line x1={first.x} y1={first.y} x2={second.x} y2={second.y} stroke={color} opacity={.35 + pair.probability * .6} /><rect x={x - 22} y={y - 10} width="44" height="20" rx="10" fill={color} /><text x={x} y={y + 3.5}>{(pair.probability * 100).toFixed(0)}%</text><title>{`L${pair.level} #${pair.fruit_id_i} ↔ #${pair.fruit_id_j} · 未来${displayedPairRisk.forecast_horizon}次投放内堵塞起始风险 ${(pair.probability * 100).toFixed(1)}%`}</title></g>; })}
              </g>}
              {ghostPoint && dropGesture && <g transform={`translate(${ghostPoint.x} ${ghostPoint.y})`} className="lab-fruit is-ghost"><circle r={(ghostSpec?.radius ?? 20) + 4} /><image href={textures[dropGesture.level]} x={-(ghostSpec?.radius ?? 20)} y={-(ghostSpec?.radius ?? 20)} width={(ghostSpec?.radius ?? 20) * 2} height={(ghostSpec?.radius ?? 20) * 2} /><text y={4}>L{dropGesture.level}</text></g>}
              {settings.showPrediction && prediction?.first_contact?.position && <g className="lab-overlay prediction"><circle cx={prediction.first_contact.position.x} cy={prediction.first_contact.position.y} r="14" /><text x={prediction.first_contact.position.x + 18} y={prediction.first_contact.position.y - 12}>预测接触</text>{settings.showNormal && prediction.first_contact.normal && <line x1={prediction.first_contact.position.x} y1={prediction.first_contact.position.y} x2={prediction.first_contact.position.x + prediction.first_contact.normal.x * 54} y2={prediction.first_contact.position.y + prediction.first_contact.normal.y * 54} />}</g>}
              {settings.showActual && actual?.first_contact?.position && <g className="lab-overlay actual"><circle cx={actual.first_contact.position.x} cy={actual.first_contact.position.y} r="11" /><text x={actual.first_contact.position.x + 16} y={actual.first_contact.position.y + 24}>真实接触</text>{settings.showNormal && actual.first_contact.normal && <line x1={actual.first_contact.position.x} y1={actual.first_contact.position.y} x2={actual.first_contact.position.x + actual.first_contact.normal.x * 54} y2={actual.first_contact.position.y + actual.first_contact.normal.y * 54} />}</g>}
              {settings.showPrediction && prediction?.generations?.map((generation) => generation.exists_probability && generation.exists_probability > .35 && generation.position ? <g key={generation.rank} className="lab-overlay generation"><circle cx={generation.position.x} cy={generation.position.y} r={8 + Number(generation.rank)} /><text x={generation.position.x + 12} y={generation.position.y}>预测新L{generation.level?.index}</text></g> : null)}
              <rect x="0" y={geometry.board_height - geometry.wall_width} width={geometry.board_width} height={geometry.wall_width} className="lab-wall" /><rect x="0" y="0" width={geometry.wall_width} height={geometry.board_height} className="lab-wall" /><rect x={geometry.board_width - geometry.wall_width} y="0" width={geometry.wall_width} height={geometry.board_height} className="lab-wall" />
              {hoverPoint && <g className="lab-coordinate-readout" transform={`translate(${Math.min(geometry.board_width - 82, hoverPoint.x + 12)} ${Math.max(26, hoverPoint.y - 12)})`}><rect x="0" y="-20" width="76" height="24" rx="6" /><text x="7" y="-4">{hoverPoint.x.toFixed(0)}, {hoverPoint.y.toFixed(0)}</text></g>}
            </svg>
          </div>
          <div className="lab-interaction-hint"><Crosshair size={14} /><span>{interactionHint}</span><kbd>V</kbd><kbd>B</kbd><kbd>E</kbd></div>
          <div className="lab-action-strip"><span>A{selectedAction}</span><input type="range" min="0" max={geometry.action_count - 1} value={selectedAction} onChange={(event) => { setSelectedAction(Number(event.target.value)); setCanvasMode("live"); }} /><b>x {actionX.toFixed(1)}</b></div>
          <div className="lab-primary-actions"><button className="primary-button compact" disabled={!health?.model_available || Boolean(busy)} onClick={() => void evaluateModel()}><Sparkles size={15} />{busy === "model" ? "预测中" : "刷新模型预测"}</button><button disabled={Boolean(busy)} onClick={() => void evaluatePhysics()}><FlaskConical size={15} />{busy === "physics" ? "验证21动作中" : "真实验证21动作"}</button><button disabled={!live || Boolean(busy)} onClick={() => void sendCommand({ type: "drop_action", action: selectedAction })}><Play size={15} />执行A{selectedAction}</button><button disabled={!health?.model_continuous_available || Boolean(busy)} onClick={() => void modelControl()}>{live?.model_continuous?.running ? <Pause size={15} /> : <Activity size={15} />}{live?.model_continuous?.running ? "停止持续决策" : "启动持续决策"}</button></div>
        </main>

        <aside className="lab-inspector">
          <div className="lab-section-title"><span>ACTION EFFECT</span><h3>预测与真实结果</h3></div>
          <div className="lab-effect-summary">
            <div><span>首次接触</span><b>{CONTACT_LABELS[prediction?.first_contact?.primary?.label ?? "none"] ?? "—"} / {CONTACT_LABELS[actual?.first_contact?.primary ?? "none"] ?? "—"}</b></div>
            <div><span>接触置信度</span><b>{percentage(prediction?.first_contact?.primary?.confidence)}</b></div>
            <div><span>合成次数</span><b>{prediction?.merge?.count?.index ?? "—"} / {actual?.merge?.count ?? "—"}</b></div>
            <div><span>q0谱系深度</span><b>{prediction?.q0?.lineage_depth?.index ?? "—"} / {actual?.q0?.lineage_depth ?? "—"}</b></div>
            <div><span>q0最终等级</span><b>{prediction?.q0?.final_level?.index ?? "—"} / {actual?.q0?.final_level ?? "—"}</b></div>
            <div><span>得分增量</span><b>{number(prediction?.outcome?.score_delta)} / {actual?.outcome?.score_delta ?? "—"}</b></div>
          </div>
          <details className="lab-collapsible" open={settings.showVoronoi}><summary><ChevronDown size={14} /> Voronoi / Free-Space Graph</summary>{displayedVoronoi ? <><div className="lab-voronoi-summary"><span>构建设备<b>{displayedVoronoi.device}</b></span><span>完整站点<b>{displayedVoronoi.stats.fruit_site_count} 水果 + 3 边界</b></span><span>可见水果站点<b>{displayedVoronoi.stats.visible_fruit_site_count} / {displayedVoronoi.stats.fruit_site_count}</b></span><span>图规模<b>{displayedVoronoi.stats.edge_count} 边 · {displayedVoronoi.stats.vertex_count} 交汇点</b></span><span>采样间距<b>{number(displayedVoronoi.sample_spacing)} px</b></span><span>构建耗时<b>{number(displayedVoronoi.compute_ms, 2)} ms{displayedVoronoi.cache_hit ? " · cache" : ""}</b></span></div><div className="lab-voronoi-legend"><i className="is-narrow" />窄 clearance <i className="is-wide" />宽 clearance</div><p className="lab-muted">圆面加权距离；左右壁与桶底参与，顶部开放。规则栅格只用于 GPU 数值构图，画布节点是恢复出的空间交汇点。</p></> : <p className="lab-muted">{voronoiError ?? (voronoiBusy ? "正在构建当前场景的完整图……" : settings.showVoronoi ? "等待场景几何停止明显变化后构图。" : "在显示设置中开启该图层。")}</p>}</details>
          <details className="lab-collapsible" open={settings.showPairRisk}><summary><ChevronDown size={14} /> 水果对未来堵塞风险</summary>{displayedPairRisk ? <><div className="lab-pair-risk-summary"><span>预测范围<b>未来 {displayedPairRisk.forecast_horizon} 次投放</b></span><span>同级高等级对<b>{displayedPairRisk.pair_count} 对</b></span><span>最高风险<b>{displayedPairRisk.pairs.length ? percentage(displayedPairRisk.pairs[0].probability) : "无候选对"}</b></span><span>CPU推理耗时<b>{number(displayedPairRisk.inference_ms, 2)} ms</b></span></div>{displayedPairRisk.pairs.length > 0 && <div className="lab-pair-risk-list">{displayedPairRisk.pairs.slice(0, 6).map((pair) => <div key={`${pair.fruit_id_i}-${pair.fruit_id_j}`}><i style={{ background: pairRiskColor(pair.probability) }} /><span>L{pair.level} · #{pair.fruit_id_i} ↔ #{pair.fruit_id_j}</span><b>{percentage(pair.probability)}</b></div>)}</div>}<p className="lab-muted">这是未来窗口内“堵塞起始”的连续风险，不表示当前已经堵塞，也不作为最终因果判断。</p></> : <p className="lab-muted">{pairRiskError ?? (pairRiskBusy ? "正在批量预测当前水果对……" : !health?.pair_risk_available ? "本次场景服务未找到堵塞风险 checkpoint。" : settings.showPairRisk ? live?.stable ? "当前没有可显示结果。" : "等待场景稳定后预测。" : "在显示设置中开启该图层。")}</p>}</details>
          <details className="lab-collapsible" open><summary><ChevronDown size={14} /> 模型动作偏好</summary>{model ? <><div className="lab-model-head"><span>推荐 A{model.action}</span><b>Q {number(model.selected_q, 5)}</b><small>{number(model.inference_ms, 2)} ms</small></div><EChart option={qChart} className="lab-q-chart" /></> : <p className="lab-muted">加载带模型的服务后，稳定场景会自动预测全部21个动作。</p>}</details>
          <details className="lab-collapsible"><summary><ChevronDown size={14} /> 场景与模型身份</summary><div className="lab-identity"><span>实时物理<b>{live?.physics_backend === "tensor_cuda" ? "Tensor / CUDA" : live?.physics_backend === "tensor_cpu" ? "Tensor / CPU" : "—"}</b></span><span>物理帧率<b>{live?.physics_fps ?? "—"} FPS</b></span><span>训练物理同源<b>{live?.training_physics_equivalent ? "是" : "—"}</b></span><span>场上水果<b>{live?.fruits.length ?? 0} / {geometry.max_fruits}</b></span><span>模型checkpoint<b>{String(health?.model?.checkpoint ?? "未加载")}</b></span><span>模型训练量<b>{number(health?.model?.training_transitions, 0)}</b></span></div></details>
          <details className="lab-collapsible"><summary><ChevronDown size={14} /> 场景文件与编辑</summary><div className="lab-file-actions"><button onClick={exportScene}><Download size={14} />导出JSON</button><label><Import size={14} />导入JSON<input type="file" accept="application/json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importScene(file); }} /></label><button disabled={selectedFruit === null} onClick={() => void removeSelected()}><Trash2 size={14} />删除选中水果</button></div><p className="lab-muted">从水果栏拖到画布，松开即放置；场上水果可拖动，右键删除，滚轮切级。暂停后支持擦除、撤销与重做；触摸端可拖动或长按删除。</p></details>
        </aside>
      </div>}

      <AnimatePresence>
        {settingsOpen && <motion.div className="lab-settings-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onMouseDown={(event) => { if (event.target === event.currentTarget) setSettingsOpen(false); }}><motion.aside className="lab-settings-drawer" initial={{ x: 36, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 36, opacity: 0 }}><div className="lab-settings-head"><div><span>VISUAL LAYERS</span><h2>场景显示设置</h2><p>只勾选当前诊断需要的图层，避免预测、真实效果和几何辅助相互遮挡。</p></div><button onClick={() => setSettingsOpen(false)}><X size={17} /></button></div><div className="lab-settings-groups"><section><h3>基础画布</h3>{settingLabel("空间网格", settings.showGrid, (value) => setSettings((current) => ({ ...current, showGrid: value })))}{settingLabel("危险线", settings.showDanger, (value) => setSettings((current) => ({ ...current, showDanger: value })))}{settingLabel("21个动作锚点", settings.showAnchors, (value) => setSettings((current) => ({ ...current, showAnchors: value })))}{settingLabel("水果速度向量", settings.showVelocity, (value) => setSettings((current) => ({ ...current, showVelocity: value })))}</section><section><h3>自由空间几何</h3>{settingLabel("完整加权 Voronoi 图", settings.showVoronoi, (value) => setSettings((current) => ({ ...current, showVoronoi: value })))}{settingLabel("显示空间交汇节点", settings.showVoronoiVertices, (value) => setSettings((current) => ({ ...current, showVoronoiVertices: value })))}</section><section><h3>长期风险预测</h3>{settingLabel("L7～L11 同级水果对堵塞风险", settings.showPairRisk, (value) => setSettings((current) => ({ ...current, showPairRisk: value })))}<p className="lab-settings-note">稳定场景下每约 0.5 秒刷新；默认使用 CPU，不参与 Policy。</p></section><section><h3>辅助动作预测</h3>{settingLabel("稳定场景实时预测", settings.realtimePrediction, (value) => setSettings((current) => ({ ...current, realtimePrediction: value })))}{settingLabel("预测接触与生成位置", settings.showPrediction, (value) => setSettings((current) => ({ ...current, showPrediction: value })))}{settingLabel("真实动作效果", settings.showActual, (value) => setSettings((current) => ({ ...current, showActual: value })))}{settingLabel("接触法向量", settings.showNormal, (value) => setSettings((current) => ({ ...current, showNormal: value })))}</section></div><button className="ghost-button" onClick={() => setSettings(DEFAULT_SETTINGS)}><RefreshCw size={14} />恢复默认图层</button></motion.aside></motion.div>}
      </AnimatePresence>
    </section>
  );
}
