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
  FlaskConical,
  Import,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Settings2,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { EChart } from "./EChart";

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
  model_continuous?: { running?: boolean; decision_count?: number; message?: string; error?: string };
};

type Health = {
  ready: boolean;
  reward_version: string;
  device: string;
  model_available: boolean;
  model_continuous_available: boolean;
  model?: Record<string, unknown> | null;
};

type Point = { x: number; y: number };
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

type ViewSettings = {
  showGrid: boolean;
  showDanger: boolean;
  showAnchors: boolean;
  showVelocity: boolean;
  showPrediction: boolean;
  showActual: boolean;
  showNormal: boolean;
  realtimePrediction: boolean;
};

const DEFAULT_GEOMETRY: Geometry = { board_width: 560, board_height: 1120, wall_width: 8, spawn_y: 156, action_count: 21, queue_length: 4, max_fruits: 64 };
const DEFAULT_SETTINGS: ViewSettings = { showGrid: true, showDanger: true, showAnchors: true, showVelocity: false, showPrediction: true, showActual: true, showNormal: true, realtimePrediction: true };
const CONTACT_LABELS: Record<string, string> = { none: "未接触", floor: "地面", left_wall: "左墙", right_wall: "右墙", fruit: "水果", dynamic_fruit: "水果" };

function number(value: unknown, digits = 1) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(digits) : "—";
}

function percentage(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : "—";
}

function sceneKey(state: LiveState) {
  return JSON.stringify({
    queue: state.queue,
    fruits: state.fruits.map((fruit) => [fruit.id, fruit.level, Math.round(fruit.x * 2), Math.round(fruit.y * 2), Math.round(fruit.vx), Math.round(fruit.vy)]),
    score: state.score,
    step: state.step_count,
  });
}

function buildScene(state: LiveState, action: number, fruits = state.fruits, queue = state.queue) {
  return {
    name: "Xigua Atlas 场景",
    fps: state.physics_fps,
    queue,
    probe_action: action,
    score: state.score,
    step_count: state.step_count,
    fruits: fruits.map((fruit) => ({ ...fruit })),
  };
}

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
  const [selectedAction, setSelectedAction] = useState(10);
  const [model, setModel] = useState<ModelEvaluation | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [canvasMode, setCanvasMode] = useState<"live" | "after">("live");
  const [settings, setSettings] = useState<ViewSettings>(() => {
    if (typeof window === "undefined") return DEFAULT_SETTINGS;
    try { return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem("xigua-atlas-lab-settings") ?? "{}") }; } catch { return DEFAULT_SETTINGS; }
  });
  const predictionKey = useRef("");
  const boardRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<{ id: number; x: number; y: number } | null>(null);

  const geometry = config?.geometry ?? DEFAULT_GEOMETRY;
  const specs = config?.fruit_specs ?? [];
  const textures = config?.textures ?? [];

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
        setHealth(nextHealth); setConfig(nextConfig); setLive(nextLive); setConnected(true); setError(null);
        source?.close();
        source = new EventSource(`${baseUrl}/api/live/events`);
        source.onmessage = (event) => { try { setLive(JSON.parse(event.data) as LiveState); setConnected(true); } catch { /* 忽略不完整事件 */ } };
        source.onerror = () => {
          setConnected(false);
          source?.close();
          source = null;
          retry();
        };
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

  const sendCommand = useCallback(async (command: Record<string, unknown>) => {
    const result = await api<Record<string, unknown>>("/api/live/command", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command }) });
    const next = await api<LiveState>("/api/live/state");
    setLive(next);
    return result;
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

  const pointFromEvent = (event: React.PointerEvent<SVGSVGElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: Math.max(geometry.wall_width, Math.min(geometry.board_width - geometry.wall_width, (event.clientX - rect.left) / rect.width * geometry.board_width)), y: Math.max(0, Math.min(geometry.board_height, (event.clientY - rect.top) / rect.height * geometry.board_height)) };
  };

  const onBoardPointerUp = async (event: React.PointerEvent<SVGSVGElement>) => {
    if (!live || live.model_continuous?.running) return;
    if (drag) {
      const fruits = live.fruits.map((fruit) => fruit.id === drag.id ? { ...fruit, x: drag.x, y: drag.y, vx: 0, vy: 0 } : fruit);
      setDrag(null);
      await sendCommand({ type: "load_scene", scene: buildScene(live, selectedAction, fruits), paused: true });
      return;
    }
    if ((event.target as SVGElement).dataset.board !== "drop") return;
    const point = pointFromEvent(event);
    await sendCommand({ type: "drop", level: selectedLevel, x: point.x });
  };

  const onBoardPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!drag || !live?.paused) return;
    const point = pointFromEvent(event);
    setDrag({ ...drag, x: point.x, y: point.y });
  };

  const removeSelected = async () => {
    if (selectedFruit === null) return;
    await sendCommand({ type: "remove", fruit_id: selectedFruit }); setSelectedFruit(null);
  };

  const togglePause = async () => {
    if (!live) return;
    await sendCommand({ type: live.paused ? "resume" : "pause" });
  };

  const updateQueue = async (index: number, delta: number) => {
    if (!live?.paused) return;
    const queue = [...live.queue]; queue[index] = 1 + ((queue[index] - 1 + delta + 5) % 5);
    await sendCommand({ type: "load_scene", scene: buildScene(live, selectedAction, live.fruits, queue), paused: true });
  };

  const loadPreset = async (kind: "empty" | "stack" | "gap") => {
    if (!live) return;
    const fruits: Fruit[] = kind === "empty" ? [] : kind === "stack"
      ? [1, 2, 3, 4, 5, 6].map((level, index) => ({ id: index + 1, level, x: 150 + (index % 3) * 130, y: geometry.board_height - 50 - Math.floor(index / 3) * 115, vx: 0, vy: 0, angle: 0, angular_velocity: 0, age_frames: 20, physics_radius: specs.find((item) => item.level === level)?.merged_physics_radius ?? 20 }))
      : [1, 2, 2, 3].map((level, index) => ({ id: index + 1, level, x: [70, 170, 390, 490][index], y: geometry.board_height - 50 - (index % 2) * 95, vx: 0, vy: 0, angle: 0, angular_velocity: 0, age_frames: 20, physics_radius: specs.find((item) => item.level === level)?.merged_physics_radius ?? 20 }));
    await sendCommand({ type: "load_scene", scene: buildScene(live, selectedAction, fruits), paused: true });
    setEvaluation(null); setModel(null); predictionKey.current = "";
  };

  const exportScene = () => {
    if (!live) return;
    const blob = new Blob([JSON.stringify(buildScene(live, selectedAction), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = "xigua-scenario.json"; anchor.click(); URL.revokeObjectURL(url);
  };

  const importScene = async (file: File) => {
    if (!live) return;
    const scene = JSON.parse(await file.text());
    await sendCommand({ type: "load_scene", scene, paused: true });
    setEvaluation(null); setModel(null); predictionKey.current = "";
  };

  const selectedEvaluation = evaluation?.actions?.[selectedAction];
  const prediction = model?.action_effect_predictions?.[selectedAction] ?? null;
  const actual = selectedEvaluation?.action_effect ?? null;
  const afterFruits = selectedEvaluation?.result_fruits;
  const displayFruits = canvasMode === "after" && afterFruits?.length ? afterFruits : live?.fruits ?? [];
  const qChart = useMemo(() => modelChart(model, selectedAction), [model, selectedAction]);
  const selectedSpec = specs.find((item) => item.level === selectedLevel);
  const left = geometry.wall_width + (selectedSpec?.radius ?? 20) + 2;
  const right = geometry.board_width - geometry.wall_width - (selectedSpec?.radius ?? 20) - 2;
  const actionX = left + (right - left) * selectedAction / Math.max(1, geometry.action_count - 1);

  if (!running) {
    return <section className="lab-empty-state"><FlaskConical size={44} /><span>SCENARIO LAB</span><h2>场景实验室后端尚未启动</h2><p>在统一设置中选择物理设备、模型checkpoint和奖励缩放；启动后将在本页直接进入实验，不再打开旧页面。</p><button className="primary-button" onClick={onConfigure}><Settings2 size={16} /> 配置并启动</button></section>;
  }

  return (
    <section className="scenario-workspace">
      <div className="lab-commandbar">
        <div><span className={`training-live-indicator ${connected ? "is-online" : ""}`} /><b>{connected ? "实时物理已连接" : "正在重连"}</b><span>{health?.reward_version ?? "Reward V2.1"} · {health?.device ?? "—"}</span></div>
        <div><button onClick={() => void togglePause()}>{live?.paused ? <CirclePlay size={15} /> : <CirclePause size={15} />}{live?.paused ? "恢复" : "暂停"}</button><button onClick={() => setSettingsOpen(true)}><Settings2 size={15} /> 显示设置</button></div>
      </div>

      {error && <div className="lab-error"><AlertCircle size={16} />{error}<button onClick={() => setError(null)}><X size={14} /></button></div>}

      <div className="lab-layout">
        <aside className="lab-palette">
          <div className="lab-section-title"><span>FRUIT PALETTE</span><h3>投放水果</h3></div>
          <div className="lab-fruit-grid">
            {specs.map((spec) => <button key={spec.level} className={selectedLevel === spec.level ? "active" : ""} onClick={() => setSelectedLevel(spec.level)}><img src={textures[spec.level]} alt="" /><span>{spec.name}</span><small>L{spec.level}</small></button>)}
          </div>
          <details className="lab-collapsible" open><summary><ChevronDown size={14} /> 队列 q0～q3</summary><div className="lab-queue-list">{(live?.queue ?? [1, 2, 3, 4]).map((level, index) => <button key={index} disabled={!live?.paused} onClick={() => void updateQueue(index, 1)} onContextMenu={(event) => { event.preventDefault(); void updateQueue(index, -1); }}><span>q{index}</span><img src={textures[level]} alt="" /><b>{specs.find((item) => item.level === level)?.name ?? `L${level}`}</b></button>)}</div></details>
          <details className="lab-collapsible"><summary><ChevronDown size={14} /> 场景预设</summary><div className="lab-preset-list"><button onClick={() => void loadPreset("empty")}><RotateCcw size={14} />空场景</button><button onClick={() => void loadPreset("stack")}><Boxes size={14} />同级堆积</button><button onClick={() => void loadPreset("gap")}><Crosshair size={14} />底层缺口</button></div></details>
        </aside>

        <main className="lab-canvas-stage">
          <div className="lab-canvas-toolbar">
            <div><button className={canvasMode === "live" ? "active" : ""} onClick={() => setCanvasMode("live")}>实时场景</button><button disabled={!evaluation} className={canvasMode === "after" ? "active" : ""} onClick={() => setCanvasMode("after")}>动作后结果</button></div>
            <span>分数 {live?.score ?? 0} · 投放 {live?.step_count ?? 0} · {live?.stable ? "已稳定" : "运动中"}</span>
          </div>
          <div className="lab-board-wrap">
            <svg ref={boardRef} className="lab-board" viewBox={`0 0 ${geometry.board_width} ${geometry.board_height}`} onPointerMove={onBoardPointerMove} onPointerUp={(event) => void onBoardPointerUp(event)} onPointerCancel={() => setDrag(null)} aria-label="合成大西瓜实时物理场景">
              <rect data-board="drop" width={geometry.board_width} height={geometry.board_height} className="lab-board-bg" />
              {settings.showGrid && Array.from({ length: 10 }, (_, index) => <line key={`v${index}`} x1={(index + 1) * geometry.board_width / 11} x2={(index + 1) * geometry.board_width / 11} y1="0" y2={geometry.board_height} className="lab-grid-line" />)}
              {settings.showGrid && Array.from({ length: 11 }, (_, index) => <line key={`h${index}`} y1={(index + 1) * geometry.board_height / 12} y2={(index + 1) * geometry.board_height / 12} x1="0" x2={geometry.board_width} className="lab-grid-line" />)}
              {settings.showDanger && <g><line x1={geometry.wall_width} x2={geometry.board_width - geometry.wall_width} y1={geometry.spawn_y} y2={geometry.spawn_y} className="lab-danger-line" /><text x={geometry.wall_width + 8} y={geometry.spawn_y - 10}>危险线</text></g>}
              {settings.showAnchors && Array.from({ length: geometry.action_count }, (_, index) => { const x = left + (right - left) * index / Math.max(1, geometry.action_count - 1); return <g key={index} className={index === selectedAction ? "lab-anchor is-selected" : "lab-anchor"}><line x1={x} x2={x} y1={geometry.spawn_y - 28} y2={geometry.spawn_y + 28} /><text x={x} y={geometry.spawn_y - 38}>A{index}</text></g>; })}
              <line x1={actionX} x2={actionX} y1="0" y2={geometry.spawn_y} className="lab-probe-line" />
              {displayFruits.map((fruit) => { const draft = drag?.id === fruit.id ? drag : fruit; const radius = specs.find((item) => item.level === fruit.level)?.radius ?? fruit.physics_radius; return <g key={fruit.id} transform={`translate(${draft.x} ${draft.y}) rotate(${fruit.angle * 180 / Math.PI})`} className={selectedFruit === fruit.id ? "lab-fruit is-selected" : "lab-fruit"} onPointerDown={(event) => { event.stopPropagation(); setSelectedFruit(fruit.id); if (live?.paused && canvasMode === "live") { event.currentTarget.setPointerCapture(event.pointerId); setDrag({ id: fruit.id, x: fruit.x, y: fruit.y }); } }}><circle r={radius + 4} /><image href={textures[fruit.level]} x={-radius} y={-radius} width={radius * 2} height={radius * 2} /><text y={4}>L{fruit.level}</text>{settings.showVelocity && <line x1="0" y1="0" x2={fruit.vx * .05} y2={fruit.vy * .05} className="lab-velocity" />}</g>; })}
              {settings.showPrediction && prediction?.first_contact?.position && <g className="lab-overlay prediction"><circle cx={prediction.first_contact.position.x} cy={prediction.first_contact.position.y} r="14" /><text x={prediction.first_contact.position.x + 18} y={prediction.first_contact.position.y - 12}>预测接触</text>{settings.showNormal && prediction.first_contact.normal && <line x1={prediction.first_contact.position.x} y1={prediction.first_contact.position.y} x2={prediction.first_contact.position.x + prediction.first_contact.normal.x * 54} y2={prediction.first_contact.position.y + prediction.first_contact.normal.y * 54} />}</g>}
              {settings.showActual && actual?.first_contact?.position && <g className="lab-overlay actual"><circle cx={actual.first_contact.position.x} cy={actual.first_contact.position.y} r="11" /><text x={actual.first_contact.position.x + 16} y={actual.first_contact.position.y + 24}>真实接触</text>{settings.showNormal && actual.first_contact.normal && <line x1={actual.first_contact.position.x} y1={actual.first_contact.position.y} x2={actual.first_contact.position.x + actual.first_contact.normal.x * 54} y2={actual.first_contact.position.y + actual.first_contact.normal.y * 54} />}</g>}
              {settings.showPrediction && prediction?.generations?.map((generation) => generation.exists_probability && generation.exists_probability > .35 && generation.position ? <g key={generation.rank} className="lab-overlay generation"><circle cx={generation.position.x} cy={generation.position.y} r={8 + Number(generation.rank)} /><text x={generation.position.x + 12} y={generation.position.y}>预测新L{generation.level?.index}</text></g> : null)}
              <rect x="0" y={geometry.board_height - geometry.wall_width} width={geometry.board_width} height={geometry.wall_width} className="lab-wall" /><rect x="0" y="0" width={geometry.wall_width} height={geometry.board_height} className="lab-wall" /><rect x={geometry.board_width - geometry.wall_width} y="0" width={geometry.wall_width} height={geometry.board_height} className="lab-wall" />
            </svg>
          </div>
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
          <details className="lab-collapsible" open><summary><ChevronDown size={14} /> 模型动作偏好</summary>{model ? <><div className="lab-model-head"><span>推荐 A{model.action}</span><b>Q {number(model.selected_q, 5)}</b><small>{number(model.inference_ms, 2)} ms</small></div><EChart option={qChart} className="lab-q-chart" /></> : <p className="lab-muted">加载带模型的服务后，稳定场景会自动预测全部21个动作。</p>}</details>
          <details className="lab-collapsible"><summary><ChevronDown size={14} /> 场景与模型身份</summary><div className="lab-identity"><span>物理帧率<b>{live?.physics_fps ?? "—"} FPS</b></span><span>场上水果<b>{live?.fruits.length ?? 0} / {geometry.max_fruits}</b></span><span>模型checkpoint<b>{String(health?.model?.checkpoint ?? "未加载")}</b></span><span>模型训练量<b>{number(health?.model?.training_transitions, 0)}</b></span></div></details>
          <details className="lab-collapsible"><summary><ChevronDown size={14} /> 场景文件与编辑</summary><div className="lab-file-actions"><button onClick={exportScene}><Download size={14} />导出JSON</button><label><Import size={14} />导入JSON<input type="file" accept="application/json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importScene(file); }} /></label><button disabled={selectedFruit === null} onClick={() => void removeSelected()}><Trash2 size={14} />删除选中水果</button></div><p className="lab-muted">暂停后可拖动水果；点击队列升级，右键降级。</p></details>
        </aside>
      </div>

      <AnimatePresence>
        {settingsOpen && <motion.div className="lab-settings-backdrop" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onMouseDown={(event) => { if (event.target === event.currentTarget) setSettingsOpen(false); }}><motion.aside className="lab-settings-drawer" initial={{ x: 36, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 36, opacity: 0 }}><div className="lab-settings-head"><div><span>VISUAL LAYERS</span><h2>场景显示设置</h2><p>只勾选当前诊断需要的图层，避免预测、真实效果和几何辅助相互遮挡。</p></div><button onClick={() => setSettingsOpen(false)}><X size={17} /></button></div><div className="lab-settings-groups"><section><h3>基础画布</h3>{settingLabel("空间网格", settings.showGrid, (value) => setSettings((current) => ({ ...current, showGrid: value })))}{settingLabel("危险线", settings.showDanger, (value) => setSettings((current) => ({ ...current, showDanger: value })))}{settingLabel("21个动作锚点", settings.showAnchors, (value) => setSettings((current) => ({ ...current, showAnchors: value })))}{settingLabel("水果速度向量", settings.showVelocity, (value) => setSettings((current) => ({ ...current, showVelocity: value })))}</section><section><h3>辅助动作预测</h3>{settingLabel("稳定场景实时预测", settings.realtimePrediction, (value) => setSettings((current) => ({ ...current, realtimePrediction: value })))}{settingLabel("预测接触与生成位置", settings.showPrediction, (value) => setSettings((current) => ({ ...current, showPrediction: value })))}{settingLabel("真实动作效果", settings.showActual, (value) => setSettings((current) => ({ ...current, showActual: value })))}{settingLabel("接触法向量", settings.showNormal, (value) => setSettings((current) => ({ ...current, showNormal: value })))}</section></div><button className="ghost-button" onClick={() => setSettings(DEFAULT_SETTINGS)}><RefreshCw size={14} />恢复默认图层</button></motion.aside></motion.div>}
      </AnimatePresence>
    </section>
  );
}
