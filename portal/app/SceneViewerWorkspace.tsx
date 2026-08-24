"use client";

import { ChevronLeft, ChevronRight, FileJson2, Images, RotateCcw } from "lucide-react";
import { useState } from "react";
import { SceneCanvas } from "./scene/SceneCanvas";
import {
  DEFAULT_SCENE_GEOMETRY,
  type SceneFruit,
  type SceneGeometry,
  type SceneSnapshot,
} from "./scene/types";

const DEFAULT_RADII = [0, 23, 31, 42, 52, 63, 72, 81, 91, 104, 119, 142];

const DEMO_SCENES: SceneSnapshot[] = [{
  name: "内置示例终局",
  score: 6689,
  step_count: 812,
  fruits: [
    { id: 1, level: 11, x: 204, y: 949, physics_radius: 142 },
    { id: 2, level: 9, x: 397, y: 1005, physics_radius: 104 },
    { id: 3, level: 8, x: 410, y: 817, physics_radius: 91 },
    { id: 4, level: 7, x: 97, y: 765, physics_radius: 81 },
    { id: 5, level: 6, x: 201, y: 741, physics_radius: 72 },
    { id: 6, level: 5, x: 500, y: 696, physics_radius: 63 },
    { id: 7, level: 4, x: 72, y: 660, physics_radius: 52 },
    { id: 8, level: 3, x: 303, y: 661, physics_radius: 42 },
    { id: 9, level: 2, x: 342, y: 604, physics_radius: 31 },
    { id: 10, level: 1, x: 378, y: 569, physics_radius: 23 },
  ],
}];

function numberValue(value: unknown) {
  const result = typeof value === "number" ? value : Number(value);
  return Number.isFinite(result) ? result : null;
}

function normaliseFruit(value: unknown, index: number): SceneFruit | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const level = numberValue(raw.level);
  const x = numberValue(raw.x);
  const y = numberValue(raw.y);
  if (level === null || x === null || y === null || level < 1 || level > 11) return null;
  return {
    id: numberValue(raw.id) ?? index + 1,
    level,
    x,
    y,
    physics_radius: numberValue(raw.physics_radius ?? raw.radius) ?? DEFAULT_RADII[level] ?? 20,
    angle: numberValue(raw.angle) ?? 0,
    vx: numberValue(raw.vx) ?? 0,
    vy: numberValue(raw.vy) ?? 0,
  };
}

function normaliseScene(value: unknown, index: number): SceneSnapshot | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const rawFruits = Array.isArray(raw.fruits) ? raw.fruits : null;
  if (!rawFruits) return null;
  const fruits = rawFruits.map(normaliseFruit).filter((fruit): fruit is SceneFruit => fruit !== null);
  const rawGeometry = raw.geometry && typeof raw.geometry === "object"
    ? raw.geometry as Record<string, unknown>
    : {};
  const geometry: Partial<SceneGeometry> = {};
  for (const key of ["board_width", "board_height", "wall_width", "spawn_y"] as const) {
    const value = numberValue(rawGeometry[key] ?? raw[key]);
    if (value !== null) geometry[key] = value;
  }
  return {
    name: String(raw.name ?? raw.label ?? `场景 ${index + 1}`),
    score: numberValue(raw.score) ?? undefined,
    step_count: numberValue(raw.step_count ?? raw.drop) ?? undefined,
    physics_frame: numberValue(raw.physics_frame) ?? undefined,
    queue: Array.isArray(raw.queue) ? raw.queue.map(numberValue).filter((item): item is number => item !== null) : undefined,
    fruits,
    geometry,
    metadata: raw.metadata && typeof raw.metadata === "object" ? raw.metadata as Record<string, unknown> : undefined,
  };
}

function normaliseScenes(payload: unknown): SceneSnapshot[] {
  const raw = payload && typeof payload === "object" ? payload as Record<string, unknown> : null;
  const candidates = Array.isArray(payload)
    ? payload
    : Array.isArray(raw?.scenes)
      ? raw.scenes
      : Array.isArray(raw?.snapshots)
        ? raw.snapshots
        : Array.isArray(raw?.frames)
          ? raw.frames
          : raw?.state
            ? [raw.state]
            : [payload];
  return candidates.map(normaliseScene).filter((scene): scene is SceneSnapshot => scene !== null);
}

export function SceneViewerWorkspace() {
  const [scenes, setScenes] = useState(DEMO_SCENES);
  const [index, setIndex] = useState(0);
  const [filename, setFilename] = useState("内置示例");
  const [error, setError] = useState<string | null>(null);
  const [layers, setLayers] = useState({ grid: true, danger: true, labels: true });
  const scene = scenes[Math.min(index, scenes.length - 1)] ?? DEMO_SCENES[0];
  const geometry = { ...DEFAULT_SCENE_GEOMETRY, ...scene.geometry };

  const importFile = async (file: File) => {
    try {
      const parsed = JSON.parse(await file.text()) as unknown;
      const imported = normaliseScenes(parsed);
      if (!imported.length) throw new Error("没有找到包含 fruits 数组的场景快照");
      setScenes(imported);
      setIndex(0);
      setFilename(file.name);
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  return (
    <section className="scene-viewer-workspace">
      <header className="page-header">
        <span className="eyebrow"><Images size={14} /> SNAPSHOT VIEWER</span>
        <h1>轻量场景查看</h1>
        <p>只负责读取和展示一个或一组 JSON 场景快照；不启动模拟器、模型或场景实验室服务。</p>
      </header>
      <div className="scene-viewer-toolbar">
        <label><FileJson2 size={15} />导入场景 JSON<input type="file" accept="application/json,.json" onChange={(event) => { const file = event.target.files?.[0]; if (file) void importFile(file); }} /></label>
        <button onClick={() => { setScenes(DEMO_SCENES); setIndex(0); setFilename("内置示例"); setError(null); }}><RotateCcw size={14} />恢复示例</button>
        <span>{filename} · {scenes.length} 个快照</span>
      </div>
      {error && <div className="scene-viewer-error">{error}</div>}
      <div className="scene-viewer-layout">
        <main className="scene-viewer-board-shell">
          <SceneCanvas fruits={scene.fruits} geometry={geometry} showGrid={layers.grid} showDanger={layers.danger} showLabels={layers.labels} ariaLabel={scene.name || "场景快照"} />
        </main>
        <aside className="scene-viewer-inspector">
          <div><span>SCENE</span><h2>{scene.name || `场景 ${index + 1}`}</h2><p>第 {index + 1} / {scenes.length} 张</p></div>
          <section className="scene-viewer-stats">
            <div><span>分数</span><b>{scene.score?.toLocaleString("zh-CN") ?? "—"}</b></div>
            <div><span>投放</span><b>{scene.step_count?.toLocaleString("zh-CN") ?? "—"}</b></div>
            <div><span>水果</span><b>{scene.fruits.length}</b></div>
            <div><span>物理帧</span><b>{scene.physics_frame?.toLocaleString("zh-CN") ?? "—"}</b></div>
          </section>
          <section className="scene-viewer-layers">
            <h3>显示图层</h3>
            {([['grid', '空间网格'], ['danger', '危险线'], ['labels', '水果等级']] as const).map(([key, label]) => <label key={key}><span>{label}</span><input type="checkbox" checked={layers[key]} onChange={(event) => setLayers((current) => ({ ...current, [key]: event.target.checked }))} /></label>)}
          </section>
          <div className="scene-viewer-pager">
            <button disabled={index <= 0} onClick={() => setIndex((current) => Math.max(0, current - 1))}><ChevronLeft size={15} />上一张</button>
            <button disabled={index >= scenes.length - 1} onClick={() => setIndex((current) => Math.min(scenes.length - 1, current + 1))}>下一张<ChevronRight size={15} /></button>
          </div>
          <p className="scene-viewer-note">该页面与物理实验室解耦。后续检测器只需输出通用场景 JSON，即可复用同一画布和独立图层。</p>
        </aside>
      </div>
    </section>
  );
}
