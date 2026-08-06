"""自定义场景实验室的无依赖前端页面。

当前模块只负责浏览器中的场景编辑、几何提示和结果展示外壳。真实物理、模型推理与
奖励计算由后续本地服务接入，前端不会用近似动画伪造这些结果。
"""

from __future__ import annotations

from html import escape


def render_scenario_lab_document(
        *, title: str, fruit_specs_json: str, textures_json: str) -> str:
    """渲染一个可离线打开的场景实验室完整 HTML。"""

    replacements = {
        '__TITLE__': escape(str(title)),
        '__FRUIT_SPECS__': fruit_specs_json,
        '__TEXTURES__': textures_json,
    }
    document = _DOCUMENT
    for token, value in replacements.items():
        document = document.replace(token, value)
    return document


_DOCUMENT = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>__TITLE__</title>
<style>
:root{
  font-family:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif;
  color:#1b1b1f;background:#eef2f8;--accent:#0f6cbd;--accent-hover:#115ea3;
  --accent-soft:#e7f3ff;--surface:rgba(255,255,255,.82);--surface-solid:#fff;
  --stroke:rgba(34,40,55,.11);--stroke-strong:rgba(34,40,55,.18);
  --text:#1f232b;--muted:#667085;--danger:#c42b1c;--warning:#bc6a00;
  --success:#0f7b0f;--shadow:0 12px 36px rgba(42,52,73,.10);
}
*{box-sizing:border-box}
body{margin:0;min-width:1060px;min-height:100vh;color:var(--text);overflow:hidden;
  background:
    radial-gradient(circle at 10% 0%,rgba(15,108,189,.11),transparent 30%),
    radial-gradient(circle at 95% 95%,rgba(119,83,173,.09),transparent 31%),#eef2f8}
button,input,select{font:inherit}button{color:inherit}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #0f6cbd;outline-offset:2px}
.app{height:100vh;display:grid;grid-template-rows:68px minmax(0,1fr)}
.topbar{height:68px;display:flex;align-items:center;gap:18px;padding:0 22px;border-bottom:1px solid var(--stroke);
  background:rgba(248,250,253,.78);backdrop-filter:blur(24px) saturate(135%);z-index:10}
.brand{display:flex;align-items:center;gap:11px;min-width:265px}.brand-mark{width:36px;height:36px;display:grid;
  place-items:center;border-radius:10px;color:#fff;background:linear-gradient(145deg,#1683d8,#6750a4);
  box-shadow:0 5px 14px rgba(15,108,189,.24);font-size:19px}.brand-copy strong{display:block;font-size:15px;
  letter-spacing:.01em}.brand-copy span{display:block;color:var(--muted);font-size:11px;margin-top:1px}
.scene-title{flex:1;max-width:390px;height:36px;border:1px solid transparent;border-radius:8px;background:transparent;
  padding:0 10px;font-weight:600;color:var(--text)}.scene-title:hover,.scene-title:focus{background:#fff;border-color:var(--stroke-strong)}
.command-group{display:flex;align-items:center;gap:6px}.command-group.push{margin-left:auto}.divider{width:1px;height:24px;
  background:var(--stroke);margin:0 5px}.icon-button,.command,.primary{height:36px;border-radius:8px;border:1px solid transparent;
  display:inline-flex;align-items:center;justify-content:center;gap:7px;cursor:pointer;transition:.14s ease}
.icon-button{width:36px;background:transparent;font-size:19px}.icon-button:hover,.command:hover{background:rgba(0,0,0,.055)}
.icon-button:disabled{opacity:.35;cursor:default}.command{padding:0 12px;background:rgba(255,255,255,.66);border-color:var(--stroke);
  font-size:13px;font-weight:600}.primary{padding:0 15px;color:#fff;background:var(--accent);border-color:#075a9e;font-size:13px;
  font-weight:650;box-shadow:0 2px 5px rgba(15,108,189,.18)}.primary:hover{background:var(--accent-hover)}
.workspace{min-height:0;display:grid;grid-template-columns:236px minmax(510px,1fr) 328px;gap:12px;padding:12px}
.panel{min-height:0;border:1px solid var(--stroke);border-radius:14px;background:var(--surface);box-shadow:var(--shadow);
  backdrop-filter:blur(22px) saturate(125%);overflow:hidden}.panel-scroll{height:100%;overflow:auto;scrollbar-width:thin}
.section{padding:16px}.section+.section{border-top:1px solid var(--stroke)}.eyebrow{font-size:11px;font-weight:700;
  color:var(--muted);letter-spacing:.07em;text-transform:uppercase;margin-bottom:11px}.section-title{display:flex;align-items:center;
  justify-content:space-between;gap:12px;margin-bottom:10px}.section-title strong{font-size:13px}.hint{font-size:11px;color:var(--muted)}
.tool-row{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.tool{height:52px;border:1px solid var(--stroke);border-radius:9px;
  background:rgba(255,255,255,.72);cursor:pointer;font-size:11px;color:var(--muted);transition:.13s}.tool b{display:block;
  color:var(--text);font-size:17px;line-height:18px;margin-bottom:3px}.tool:hover{border-color:rgba(15,108,189,.35);background:#fff}
.tool.active{border-color:rgba(15,108,189,.55);background:var(--accent-soft);color:#075a9e;box-shadow:inset 0 0 0 1px rgba(15,108,189,.12)}
.fruit-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.fruit-choice{position:relative;min-width:0;height:68px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;border:1px solid transparent;border-radius:10px;
  background:transparent;cursor:pointer;transition:.14s}.fruit-choice:hover{background:rgba(255,255,255,.82);border-color:var(--stroke)}
.fruit-choice.active{background:#fff;border-color:rgba(15,108,189,.48);box-shadow:0 2px 8px rgba(36,67,102,.10)}
.fruit-choice img{width:38px;height:38px;object-fit:contain;filter:drop-shadow(0 3px 3px rgba(30,40,55,.16))}
.fruit-choice span{max-width:100%;font-size:10px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fruit-choice i{position:absolute;right:5px;top:5px;width:16px;height:16px;border-radius:999px;background:rgba(31,35,43,.72);
  color:#fff;font:600 9px/16px "Segoe UI";font-style:normal}.fruit-choice.active i{background:var(--accent)}
.option-list{display:grid;gap:9px}.switch-row{display:flex;align-items:center;justify-content:space-between;gap:12px;font-size:12px}
.switch{position:relative;width:36px;height:20px;flex:none}.switch input{position:absolute;opacity:0}.switch span{position:absolute;inset:0;
  border-radius:999px;background:#a9aeb6;cursor:pointer;transition:.16s}.switch span::after{content:"";position:absolute;width:14px;height:14px;
  left:3px;top:3px;border-radius:50%;background:#fff;box-shadow:0 1px 3px #0003;transition:.16s}.switch input:checked+span{background:var(--accent)}
.switch input:checked+span::after{transform:translateX(16px)}
.stage-panel{position:relative;display:grid;place-items:center;overflow:hidden;background:rgba(231,237,246,.54)}
.stage-glow{position:absolute;width:760px;height:760px;border-radius:50%;background:radial-gradient(circle,rgba(15,108,189,.08),transparent 68%);pointer-events:none}
.stage-shell{height:calc(100vh - 116px);aspect-ratio:1/2;max-width:calc(100% - 92px);position:relative;z-index:1;
  filter:drop-shadow(0 20px 34px rgba(25,34,50,.19))}.board-frame{height:100%;position:relative;padding:6px;border-radius:18px;
  background:linear-gradient(145deg,rgba(255,255,255,.88),rgba(207,218,234,.75));border:1px solid rgba(255,255,255,.9)}
canvas{display:block;width:100%;height:100%;border-radius:13px;background:#111b2a;touch-action:none;cursor:crosshair}
.canvas-badge{position:absolute;left:18px;top:18px;height:28px;display:flex;align-items:center;gap:7px;padding:0 10px;
  color:#dbe7f5;background:rgba(7,15,27,.67);border:1px solid rgba(255,255,255,.12);border-radius:8px;backdrop-filter:blur(10px);
  font-size:10px;pointer-events:none}.canvas-badge i{width:7px;height:7px;border-radius:50%;background:#79d7a8;box-shadow:0 0 0 3px rgba(121,215,168,.13)}
.canvas-tip{position:absolute;left:50%;bottom:18px;transform:translateX(-50%);max-width:84%;padding:7px 11px;border-radius:8px;
  color:#d8e5f3;background:rgba(7,15,27,.72);border:1px solid rgba(255,255,255,.10);backdrop-filter:blur(10px);
  font-size:10px;white-space:nowrap;pointer-events:none;transition:opacity .18s}.coordinate{position:absolute;display:none;padding:5px 7px;
  color:#fff;background:rgba(8,16,29,.84);border:1px solid rgba(255,255,255,.14);border-radius:6px;font:10px/1.2 ui-monospace,monospace;
  pointer-events:none;z-index:3}.stage-tools{position:absolute;right:16px;top:16px;z-index:2;display:flex;flex-direction:column;gap:6px}
.stage-tools button{width:34px;height:34px;border:1px solid rgba(255,255,255,.75);border-radius:9px;background:rgba(255,255,255,.80);
  box-shadow:0 3px 10px rgba(35,46,65,.12);cursor:pointer}.stage-tools button:hover{background:#fff}
.side-tabs{height:48px;display:grid;grid-template-columns:1fr 1fr;padding:6px;border-bottom:1px solid var(--stroke);gap:4px}
.side-tab{border:0;border-radius:8px;background:transparent;color:var(--muted);font-size:12px;font-weight:650;cursor:pointer}
.side-tab.active{background:#fff;color:var(--text);box-shadow:0 1px 5px rgba(45,56,74,.10)}.tab-page{display:none}.tab-page.active{display:block}
.template-select{width:100%;height:38px;border:1px solid var(--stroke);border-radius:8px;background:#fff;padding:0 10px;color:var(--text);font-size:12px}
.stat-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.mini-stat{padding:9px;border-radius:9px;background:rgba(246,248,252,.86);
  border:1px solid var(--stroke)}.mini-stat span{display:block;color:var(--muted);font-size:9px;margin-bottom:4px}.mini-stat strong{display:block;
  font-size:14px;font-variant-numeric:tabular-nums}.mini-stat.warning strong{color:var(--warning)}.mini-stat.danger strong{color:var(--danger)}
.validation{margin-top:9px;display:flex;align-items:flex-start;gap:8px;padding:9px 10px;border-radius:9px;background:#eef8ef;color:#176b1a;
  font-size:11px;line-height:1.45}.validation.warning{background:#fff4e5;color:#8a4b00}.validation i{width:7px;height:7px;margin-top:4px;
  border-radius:50%;background:currentColor;flex:none}
.selection-card{display:grid;grid-template-columns:48px 1fr auto;align-items:center;gap:10px;padding:10px;border:1px solid var(--stroke);
  border-radius:10px;background:#fff}.selection-card.empty{grid-template-columns:1fr;color:var(--muted);font-size:11px;text-align:center;min-height:62px}
.selection-card img{width:44px;height:44px;object-fit:contain}.selection-copy strong{display:block;font-size:12px}.selection-copy span{display:block;
  color:var(--muted);font-size:10px;margin-top:3px}.small-danger{width:30px;height:30px;border:0;border-radius:7px;background:#fff0ee;color:var(--danger);
  font-size:16px;cursor:pointer}.small-danger:hover{background:#fde3df}
.queue{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.queue-slot{height:67px;border:1px solid var(--stroke);border-radius:10px;
  background:#fff;cursor:pointer;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px;transition:.13s}
.queue-slot:hover{border-color:rgba(15,108,189,.4);transform:translateY(-1px)}.queue-slot:first-child{border-color:rgba(15,108,189,.46);
  background:var(--accent-soft)}.queue-slot em{font-style:normal;font-size:9px;color:var(--muted)}.queue-slot img{width:32px;height:32px;object-fit:contain}
.queue-slot span{font-size:9px;color:var(--muted)}
.probe-row{display:flex;align-items:center;gap:10px}.probe-readout{min-width:62px}.probe-readout strong{display:block;font-size:13px}
.probe-readout span{font-size:9px;color:var(--muted)}input[type=range]{flex:1;accent-color:var(--accent)}
.segmented{display:grid;grid-template-columns:1fr 1fr;padding:3px;border-radius:9px;background:#edf0f5}.segmented button{height:31px;border:0;
  border-radius:7px;background:transparent;color:var(--muted);font-size:11px;cursor:pointer}.segmented button.active{background:#fff;color:var(--text);
  box-shadow:0 1px 4px rgba(30,42,62,.12)}
.run-stack{display:grid;gap:7px}.run-stack button{width:100%;height:38px}.secondary{border:1px solid var(--stroke);border-radius:8px;
  background:#fff;font-size:12px;font-weight:650;cursor:pointer}.secondary:hover{background:#f6f8fb}.backend-note{display:flex;gap:7px;align-items:flex-start;
  color:var(--muted);font-size:10px;line-height:1.45;margin-top:8px}.backend-note i{width:7px;height:7px;margin-top:4px;border-radius:50%;
  background:#98a2b3;flex:none}
.empty-result{padding:28px 18px;text-align:center}.empty-illustration{width:76px;height:76px;margin:0 auto 14px;display:grid;place-items:center;
  border-radius:24px;background:linear-gradient(145deg,#e6f3ff,#f0eafb);color:#306ea8;font-size:31px;box-shadow:inset 0 0 0 1px rgba(15,108,189,.08)}
.empty-result strong{display:block;font-size:14px}.empty-result p{max-width:220px;margin:7px auto 0;color:var(--muted);font-size:11px;line-height:1.55}
.result-summary{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.result-card{padding:10px;border:1px solid var(--stroke);border-radius:9px;
  background:#fff}.result-card span{display:block;color:var(--muted);font-size:9px;margin-bottom:4px}.result-card strong{font-size:15px}
.action-results{height:134px;display:grid;grid-template-columns:repeat(21,minmax(5px,1fr));align-items:end;gap:3px;padding:9px 7px 20px;
  border:1px solid var(--stroke);border-radius:10px;background:#f7f9fc}.action-result{height:100%;position:relative;display:flex;align-items:flex-end}
.action-result i{display:block;width:100%;height:5%;min-height:3px;border-radius:3px 3px 1px 1px;background:#aeb8c6}.action-result.best i{
  background:linear-gradient(#62b6ee,#0f6cbd)}.action-result span{position:absolute;bottom:-14px;width:100%;text-align:center;font-size:7px;color:#8993a2}
.result-message{margin-top:10px;padding:9px;border-radius:8px;background:#f2f4f8;color:var(--muted);font-size:10px;line-height:1.45}
.toast-stack{position:fixed;right:20px;top:80px;z-index:50;display:grid;gap:8px;pointer-events:none}.toast{min-width:250px;max-width:360px;
  padding:11px 13px;border:1px solid rgba(255,255,255,.7);border-radius:10px;background:rgba(31,35,43,.91);color:#fff;
  box-shadow:0 10px 28px rgba(22,29,42,.20);backdrop-filter:blur(18px);font-size:11px;animation:toast-in .2s ease both}
.toast strong{display:block;font-size:12px;margin-bottom:2px}.toast.out{animation:toast-out .2s ease both}@keyframes toast-in{from{opacity:0;transform:translateY(-6px)}}
@keyframes toast-out{to{opacity:0;transform:translateY(-6px)}}
.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:1240px){.workspace{grid-template-columns:216px minmax(470px,1fr) 310px}.brand{min-width:230px}.command span{display:none}}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand"><div class="brand-mark" aria-hidden="true">◇</div><div class="brand-copy"><strong>场景实验室</strong><span>Scenario Lab · 前端预览</span></div></div>
    <input id="scene-title" class="scene-title" value="濒临出界压力测试" aria-label="场景名称">
    <div class="command-group">
      <button id="undo" class="icon-button" title="撤销 Ctrl+Z" aria-label="撤销">↶</button>
      <button id="redo" class="icon-button" title="重做 Ctrl+Y" aria-label="重做">↷</button>
      <div class="divider"></div>
      <button id="import" class="command">⇧ <span>导入场景</span></button>
      <button id="export" class="command">⇩ <span>导出 JSON</span></button>
      <input id="import-file" class="sr-only" type="file" accept="application/json,.json">
    </div>
    <div class="command-group push"><button id="evaluate-top" class="primary">评估 21 个动作</button></div>
  </header>

  <main class="workspace">
    <aside class="panel"><div class="panel-scroll">
      <section class="section">
        <div class="eyebrow">编辑工具</div>
        <div class="tool-row">
          <button class="tool" data-tool="select"><b>↖</b>选择</button>
          <button class="tool active" data-tool="place"><b>＋</b>放置</button>
          <button class="tool" data-tool="erase"><b>⌫</b>擦除</button>
        </div>
      </section>
      <section class="section">
        <div class="section-title"><strong>水果等级</strong><span class="hint">滚轮切换</span></div>
        <div id="fruit-grid" class="fruit-grid" aria-label="水果等级列表"></div>
      </section>
      <section class="section">
        <div class="section-title"><strong>画布辅助</strong></div>
        <div class="option-list">
          <label class="switch-row"><span>显示 21 个动作锚点</span><span class="switch"><input id="show-anchors" type="checkbox" checked><span></span></span></label>
          <label class="switch-row"><span>显示空间参考网格</span><span class="switch"><input id="show-grid" type="checkbox" checked><span></span></span></label>
          <label class="switch-row"><span>高亮几何异常</span><span class="switch"><input id="show-warnings" type="checkbox" checked><span></span></span></label>
        </div>
      </section>
    </div></aside>

    <section class="panel stage-panel" aria-label="场景编辑画布">
      <div class="stage-glow"></div>
      <div class="stage-shell">
        <div class="board-frame">
          <canvas id="board" width="560" height="1120"></canvas>
          <div class="canvas-badge"><i></i><span id="canvas-status">编辑模式 · 120 FPS</span></div>
          <div id="canvas-tip" class="canvas-tip">左键放置 · 拖动调整 · 右键删除 · 滚轮切换等级</div>
          <div id="coordinate" class="coordinate"></div>
        </div>
      </div>
      <div class="stage-tools">
        <button id="center-view" title="回到完整画布">⌂</button>
        <button id="clear-scene" title="清空场景">⌫</button>
      </div>
    </section>

    <aside class="panel"><div class="panel-scroll">
      <div class="side-tabs"><button class="side-tab active" data-tab="scene">场景设置</button><button class="side-tab" data-tab="result">评估结果</button></div>
      <div id="tab-scene" class="tab-page active">
        <section class="section">
          <div class="section-title"><strong>快速场景</strong><span class="hint">可继续编辑</span></div>
          <select id="template" class="template-select">
            <option value="critical">濒临出界压力测试</option>
            <option value="gap">底层缺口与小果陷落</option>
            <option value="duplicates">同级水果合并债务</option>
            <option value="empty">空白场景</option>
          </select>
        </section>
        <section class="section">
          <div class="section-title"><strong>场景概览</strong><span class="hint">仅几何检查</span></div>
          <div class="stat-strip">
            <div class="mini-stat"><span>水果</span><strong id="fruit-count">0</strong></div>
            <div class="mini-stat"><span>越线</span><strong id="danger-count">0</strong></div>
            <div class="mini-stat"><span>重叠</span><strong id="overlap-count">0</strong></div>
          </div>
          <div id="validation" class="validation"><i></i><span>场景几何检查通过，可以接入物理模拟。</span></div>
        </section>
        <section class="section">
          <div class="section-title"><strong>当前选择</strong><span class="hint">拖动可调整</span></div>
          <div id="selection" class="selection-card empty">尚未选择场上水果</div>
        </section>
        <section class="section">
          <div class="section-title"><strong>待投水果</strong><span class="hint">点击循环等级</span></div>
          <div id="queue" class="queue"></div>
        </section>
        <section class="section">
          <div class="section-title"><strong>投放探针</strong><span class="hint">21 个模型动作</span></div>
          <div class="probe-row"><input id="probe" type="range" min="0" max="20" value="10"><div class="probe-readout"><strong id="probe-action">A10</strong><span id="probe-x">x = 280</span></div></div>
        </section>
        <section class="section">
          <div class="section-title"><strong>物理档位</strong><span class="hint">评估参数</span></div>
          <div class="segmented"><button data-fps="30">30 FPS</button><button class="active" data-fps="120">120 FPS</button></div>
        </section>
        <section class="section">
          <div class="run-stack">
            <button id="settle" class="secondary">运行至稳定</button>
            <button id="evaluate" class="primary">并行评估 21 个动作</button>
          </div>
          <div class="backend-note"><i></i><span>当前为前端版。点击会生成标准场景请求；接入 CUDA 场景加载接口后返回真实物理与奖励。</span></div>
        </section>
      </div>

      <div id="tab-result" class="tab-page">
        <div id="result-empty" class="empty-result"><div class="empty-illustration">⌁</div><strong>等待第一次真实评估</strong><p>结果区不会展示模拟数据。后端接入后，这里将对比 21 个动作的奖励和终局风险。</p></div>
        <div id="result-content" hidden>
          <section class="section">
            <div class="section-title"><strong>动作摘要</strong><span id="result-fps" class="hint">—</span></div>
            <div class="result-summary">
              <div class="result-card"><span>推荐动作</span><strong id="best-action">—</strong></div>
              <div class="result-card"><span>最高即时奖励</span><strong id="best-reward">—</strong></div>
              <div class="result-card"><span>最高得分增量</span><strong id="best-score">—</strong></div>
              <div class="result-card"><span>危险动作</span><strong id="terminal-actions">—</strong></div>
            </div>
          </section>
          <section class="section"><div class="section-title"><strong>21 动作奖励</strong><span class="hint">当前值</span></div><div id="action-results" class="action-results"></div><div id="result-message" class="result-message"></div></section>
        </div>
      </div>
    </div></aside>
  </main>
</div>
<div id="toasts" class="toast-stack" aria-live="polite"></div>

<script>
(()=>{
'use strict';
const FRUITS=__FRUIT_SPECS__;
const TEXTURES=__TEXTURES__;
const BOARD={width:560,height:1120,spawnY:252,wall:20,actions:21};
const COLORS=['','#b74b95','#ed5b70','#7650ad','#f19c37','#ef7447','#ef5350','#d5b34c','#ef9d91','#e8a63d','#9fd45a','#45b933'];
const $=id=>document.getElementById(id);
const canvas=$('board'),ctx=canvas.getContext('2d');
const images=TEXTURES.map(source=>{if(!source)return null;const image=new Image();image.src=source;image.onload=draw;return image});
const templates={
  critical:{title:'濒临出界压力测试',queue:[2,1,4,3],fruits:[
    [7,93,1020],[6,230,1018],[7,410,1015],[5,150,900],[5,292,891],[4,405,892],
    [4,88,798],[3,193,790],[3,303,785],[2,390,770],[2,469,748],[1,275,681],
    [1,340,266],[1,386,302]
  ]},
  gap:{title:'底层缺口与小果陷落',queue:[1,3,2,4],fruits:[
    [8,127,994],[8,430,995],[5,268,1036],[3,291,928],[2,279,862],[7,104,783],[7,446,785],[4,256,755]
  ]},
  duplicates:{title:'同级水果合并债务',queue:[3,3,1,2],fruits:[
    [6,91,1014],[6,235,1017],[6,384,1015],[5,492,1028],[5,137,886],[5,276,876],[5,414,887],
    [4,81,782],[4,183,779],[4,287,776],[4,389,782],[3,473,765],[2,260,683]
  ]},
  empty:{title:'空白场景',queue:[1,2,3,4],fruits:[]}
};
let state={fruits:[],queue:[2,1,4,3],selectedId:null,level:1,tool:'place',fps:120,probe:10,
  showAnchors:true,showGrid:true,showWarnings:true,title:'濒临出界压力测试'};
let nextId=1,hover=null,drag=null,history=[],future=[],historyLock=false;

function spec(level){return FRUITS[level-1]}
function actionX(index,level=state.queue[0]){const radius=spec(level).radius,left=BOARD.wall+radius+2,right=BOARD.width-BOARD.wall-radius-2;return left+(right-left)*index/(BOARD.actions-1)}
function cloneScene(){return {version:1,board:{...BOARD},name:$('scene-title').value.trim()||'未命名场景',fps:state.fps,queue:[...state.queue],probe_action:state.probe,
  fruits:state.fruits.map(({id,level,x,y})=>({id,level,x:+x.toFixed(2),y:+y.toFixed(2),vx:0,vy:0,angle:0,angular_velocity:0}))}}
function snapshot(){return JSON.stringify(cloneScene())}
function remember(){if(historyLock)return;const value=snapshot();if(history.at(-1)!==value){history.push(value);if(history.length>80)history.shift()}future=[];updateUndo()}
function updateUndo(){$('undo').disabled=history.length<2;$('redo').disabled=!future.length}
function restore(raw){const scene=typeof raw==='string'?JSON.parse(raw):raw;loadScene(scene,{remember:false});draw();renderUi()}
function undo(){if(history.length<2)return;future.push(history.pop());restore(history.at(-1));updateUndo()}
function redo(){if(!future.length)return;const value=future.pop();history.push(value);restore(value);updateUndo()}
function clampFruit(fruit){const r=spec(fruit.level).radius;fruit.x=Math.max(BOARD.wall+r,Math.min(BOARD.width-BOARD.wall-r,fruit.x));fruit.y=Math.max(r,Math.min(BOARD.height-BOARD.wall-r,fruit.y))}
function loadTemplate(name){const item=templates[name]||templates.empty;nextId=1;state.fruits=item.fruits.map(([level,x,y])=>({id:nextId++,level,x,y}));state.queue=[...item.queue];state.selectedId=null;state.title=item.title;$('scene-title').value=item.title;remember();renderUi();draw();toast('场景已载入',item.title)}
function loadScene(scene,{remember:shouldRemember=true}={}){
  if(!scene||!Array.isArray(scene.fruits))throw new Error('场景文件缺少 fruits 数组');
  nextId=1;state.fruits=scene.fruits.slice(0,64).map(item=>{const level=Math.max(1,Math.min(11,Number(item.level)||1));const fruit={id:Number(item.id)||nextId,level,x:Number(item.x),y:Number(item.y)};nextId=Math.max(nextId,fruit.id+1);if(!Number.isFinite(fruit.x)||!Number.isFinite(fruit.y))throw new Error('水果坐标必须是有限数值');clampFruit(fruit);return fruit});
  state.queue=(Array.isArray(scene.queue)?scene.queue:[1,2,3,4]).slice(0,4).map(value=>Math.max(1,Math.min(11,Number(value)||1)));while(state.queue.length<4)state.queue.push(1);
  state.fps=Number(scene.fps)===30?30:120;state.probe=Math.max(0,Math.min(20,Number(scene.probe_action)||10));state.selectedId=null;
  $('scene-title').value=String(scene.name||'导入场景').slice(0,80);if(shouldRemember)remember();renderUi();draw()
}
function pointFromEvent(event){const rect=canvas.getBoundingClientRect();return {x:(event.clientX-rect.left)*BOARD.width/rect.width,y:(event.clientY-rect.top)*BOARD.height/rect.height,rect}}
function hitFruit(point){for(let i=state.fruits.length-1;i>=0;i--){const fruit=state.fruits[i],r=spec(fruit.level).radius;if(Math.hypot(point.x-fruit.x,point.y-fruit.y)<=r)return fruit}return null}
function overlaps(){const pairs=new Set;for(let a=0;a<state.fruits.length;a++)for(let b=a+1;b<state.fruits.length;b++){const x=state.fruits[a],y=state.fruits[b];if(Math.hypot(x.x-y.x,x.y-y.y)<spec(x.level).radius+spec(y.level).radius-1){pairs.add(x.id);pairs.add(y.id)}}return pairs}
function place(point){if(state.fruits.length>=64){toast('无法继续放置','场景最多包含 64 个水果');return}const fruit={id:nextId++,level:state.level,x:point.x,y:point.y};clampFruit(fruit);state.fruits.push(fruit);state.selectedId=fruit.id;remember();renderUi();draw()}
function removeFruit(id){const index=state.fruits.findIndex(item=>item.id===id);if(index<0)return;state.fruits.splice(index,1);if(state.selectedId===id)state.selectedId=null;remember();renderUi();draw()}
function cycleLevel(direction,target=null){if(target){target.level=1+(target.level-1+direction+11)%11;clampFruit(target);state.level=target.level}else state.level=1+(state.level-1+direction+11)%11;remember();renderUi();draw()}

function drawFruit(fruit,{ghost=false,invalid=false,selected=false}={}){const r=spec(fruit.level).radius;ctx.save();ctx.globalAlpha=ghost?.46:1;
  if(invalid){ctx.beginPath();ctx.arc(fruit.x,fruit.y,r+5,0,Math.PI*2);ctx.fillStyle='rgba(255,91,77,.22)';ctx.fill()}
  ctx.shadowColor='rgba(0,0,0,.32)';ctx.shadowBlur=ghost?0:12;ctx.shadowOffsetY=ghost?0:5;
  if(images[fruit.level]&&images[fruit.level].complete)ctx.drawImage(images[fruit.level],fruit.x-r,fruit.y-r,r*2,r*2);else{ctx.beginPath();ctx.arc(fruit.x,fruit.y,r,0,Math.PI*2);ctx.fillStyle=COLORS[fruit.level];ctx.fill()}
  ctx.shadowColor='transparent';if(selected){ctx.beginPath();ctx.arc(fruit.x,fruit.y,r+7,0,Math.PI*2);ctx.strokeStyle='#76c7ff';ctx.lineWidth=3;ctx.stroke();ctx.setLineDash([5,5]);ctx.beginPath();ctx.arc(fruit.x,fruit.y,r+12,0,Math.PI*2);ctx.strokeStyle='rgba(118,199,255,.48)';ctx.lineWidth=1.5;ctx.stroke()}
  ctx.restore()}
function draw(){ctx.clearRect(0,0,BOARD.width,BOARD.height);const bg=ctx.createLinearGradient(0,0,0,BOARD.height);bg.addColorStop(0,'#17263a');bg.addColorStop(1,'#0e1724');ctx.fillStyle=bg;ctx.fillRect(0,0,BOARD.width,BOARD.height);
  if(state.showGrid){ctx.strokeStyle='rgba(148,177,207,.085)';ctx.lineWidth=1;for(let y=BOARD.spawnY;y<BOARD.height;y+=56){ctx.beginPath();ctx.moveTo(BOARD.wall,y+.5);ctx.lineTo(BOARD.width-BOARD.wall,y+.5);ctx.stroke()}for(let x=BOARD.wall;x<BOARD.width;x+=56){ctx.beginPath();ctx.moveTo(x,BOARD.spawnY);ctx.lineTo(x,BOARD.height-BOARD.wall);ctx.stroke()}}
  ctx.fillStyle='#795b43';ctx.fillRect(0,BOARD.spawnY,BOARD.wall,BOARD.height-BOARD.spawnY);ctx.fillRect(BOARD.width-BOARD.wall,BOARD.spawnY,BOARD.wall,BOARD.height-BOARD.spawnY);ctx.fillRect(0,BOARD.height-BOARD.wall,BOARD.width,BOARD.wall);
  ctx.fillStyle='rgba(255,255,255,.12)';ctx.fillRect(BOARD.wall,BOARD.spawnY,2,BOARD.height-BOARD.spawnY);ctx.fillRect(BOARD.width-BOARD.wall-2,BOARD.spawnY,2,BOARD.height-BOARD.spawnY);
  ctx.save();ctx.setLineDash([8,7]);ctx.strokeStyle='rgba(255,112,94,.82)';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(BOARD.wall+4,BOARD.spawnY);ctx.lineTo(BOARD.width-BOARD.wall-4,BOARD.spawnY);ctx.stroke();ctx.restore();ctx.fillStyle='rgba(255,170,156,.82)';ctx.font='600 11px "Segoe UI"';ctx.fillText('危险线',BOARD.wall+10,BOARD.spawnY-10);
  if(state.showAnchors){for(let i=0;i<BOARD.actions;i++){const x=actionX(i);ctx.beginPath();ctx.arc(x,BOARD.spawnY,i===state.probe?4:2.2,0,Math.PI*2);ctx.fillStyle=i===state.probe?'#83d2ff':'rgba(189,214,238,.48)';ctx.fill()}ctx.fillStyle='rgba(194,216,237,.58)';ctx.font='9px "Segoe UI"';ctx.textAlign='center';ctx.fillText('A0',actionX(0),BOARD.spawnY+17);ctx.fillText('A10',actionX(10),BOARD.spawnY+17);ctx.fillText('A20',actionX(20),BOARD.spawnY+17)}
  const probeX=actionX(state.probe);ctx.save();ctx.setLineDash([7,8]);ctx.strokeStyle='rgba(109,203,255,.70)';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(probeX,74);ctx.lineTo(probeX,BOARD.height-BOARD.wall);ctx.stroke();ctx.restore();drawFruit({level:state.queue[0],x:probeX,y:118},{ghost:true});
  const invalid=state.showWarnings?overlaps():new Set;state.fruits.forEach(fruit=>drawFruit(fruit,{invalid:invalid.has(fruit.id),selected:fruit.id===state.selectedId}));
  if(hover&&state.tool==='place'&&!drag&&!hitFruit(hover)){const ghost={level:state.level,x:hover.x,y:hover.y};clampFruit(ghost);drawFruit(ghost,{ghost:true})}
}

function renderFruitGrid(){const grid=$('fruit-grid');grid.replaceChildren(...FRUITS.map(item=>{const button=document.createElement('button');button.className='fruit-choice'+(item.level===state.level?' active':'');button.title=`${item.level}级 · ${item.name} · 半径 ${item.radius}`;button.innerHTML=`<img src="${TEXTURES[item.level]}" alt=""><span>${item.name}</span><i>${item.level}</i>`;button.onclick=()=>{state.level=item.level;state.tool='place';renderUi();draw()};return button}))}
function renderQueue(){const root=$('queue');root.replaceChildren(...state.queue.map((level,index)=>{const button=document.createElement('button');button.className='queue-slot';button.title='点击升级；Shift+点击降级';button.innerHTML=`<em>q${index}</em><img src="${TEXTURES[level]}" alt=""><span>${spec(level).name}</span>`;button.onclick=event=>{state.queue[index]=1+(level-1+(event.shiftKey?-1:1)+11)%11;remember();renderUi();draw()};return button}))}
function renderSelection(){const fruit=state.fruits.find(item=>item.id===state.selectedId),root=$('selection');if(!fruit){root.className='selection-card empty';root.textContent='尚未选择场上水果';return}root.className='selection-card';root.innerHTML=`<img src="${TEXTURES[fruit.level]}" alt=""><div class="selection-copy"><strong>${fruit.level}级 · ${spec(fruit.level).name}</strong><span>x ${fruit.x.toFixed(1)} · y ${fruit.y.toFixed(1)} · 半径 ${spec(fruit.level).radius}</span></div><button class="small-danger" title="删除水果">⌫</button>`;root.querySelector('button').onclick=()=>removeFruit(fruit.id)}
function renderValidation(){const invalid=overlaps(),danger=state.fruits.filter(fruit=>fruit.y-spec(fruit.level).radius<BOARD.spawnY).length;$('fruit-count').textContent=state.fruits.length;$('danger-count').textContent=danger;$('overlap-count').textContent=invalid.size;
  $('danger-count').parentElement.className='mini-stat'+(danger?' danger':'');$('overlap-count').parentElement.className='mini-stat'+(invalid.size?' warning':'');const box=$('validation');
  if(invalid.size){box.className='validation warning';box.querySelector('span').textContent=`检测到 ${invalid.size} 个水果参与重叠。可以保留以测试极端状态，但后端载入时应再次校验。`}else if(danger){box.className='validation warning';box.querySelector('span').textContent=`有 ${danger} 个水果越过危险线，适合验证终局与奖励边界。`}else{box.className='validation';box.querySelector('span').textContent='场景几何检查通过，可以接入物理模拟。'}}
function renderProbe(){const x=actionX(state.probe);$('probe').value=state.probe;$('probe-action').textContent=`A${state.probe}`;$('probe-x').textContent=`x = ${x.toFixed(1)}`}
function renderUi(){document.querySelectorAll('.tool').forEach(button=>button.classList.toggle('active',button.dataset.tool===state.tool));renderFruitGrid();renderQueue();renderSelection();renderValidation();renderProbe();
  $('show-anchors').checked=state.showAnchors;$('show-grid').checked=state.showGrid;$('show-warnings').checked=state.showWarnings;document.querySelectorAll('[data-fps]').forEach(button=>button.classList.toggle('active',Number(button.dataset.fps)===state.fps));$('canvas-status').textContent=`编辑模式 · ${state.fps} FPS`;updateUndo()}
function toast(title,message){const item=document.createElement('div');item.className='toast';item.innerHTML=`<strong></strong><span></span>`;item.querySelector('strong').textContent=title;item.querySelector('span').textContent=message;$('toasts').append(item);setTimeout(()=>{item.classList.add('out');setTimeout(()=>item.remove(),220)},3200)}
function selectTab(name){document.querySelectorAll('.side-tab').forEach(tab=>tab.classList.toggle('active',tab.dataset.tab===name));document.querySelectorAll('.tab-page').forEach(page=>page.classList.toggle('active',page.id===`tab-${name}`))}
function requestEvaluation(mode='all'){const detail={mode,scene:cloneScene()};window.dispatchEvent(new CustomEvent('daxigua:scenario-request',{detail}));selectTab('result');toast('已生成场景请求','物理与模型后端尚未连接，未产生虚构评估结果。')}
function applyEvaluation(payload){if(!payload||!Array.isArray(payload.actions))throw new Error('评估结果必须包含 actions 数组');$('result-empty').hidden=true;$('result-content').hidden=false;const actions=payload.actions.slice(0,21),rewards=actions.map(x=>Number(x.reward)||0),best=rewards.length?Math.max(...rewards):0,bestIndex=rewards.length?rewards.indexOf(best):0,bestItem=actions[bestIndex]||{};
  $('best-action').textContent=`A${bestIndex}`;$('best-reward').textContent=best.toFixed(3);$('best-score').textContent=String(bestItem.score_delta??'—');$('terminal-actions').textContent=String(actions.filter(x=>x.done||x.terminal).length);$('result-fps').textContent=`${payload.physics_fps||state.fps} FPS`;
  const min=Math.min(0,...rewards),span=Math.max(1e-8,best-min),root=$('action-results');root.replaceChildren(...Array.from({length:21},(_,index)=>{const value=rewards[index]??0,item=document.createElement('div');item.className='action-result'+(index===bestIndex?' best':'');item.title=`A${index} · reward ${value.toFixed(4)}`;item.innerHTML=`<i style="height:${8+82*(value-min)/span}%"></i><span>${index}</span>`;return item}));$('result-message').textContent=payload.message||'结果来自已连接的真实物理与模型评估。';selectTab('result')}

canvas.addEventListener('contextmenu',event=>event.preventDefault());
canvas.addEventListener('pointerdown',event=>{const point=pointFromEvent(event),hit=hitFruit(point);canvas.setPointerCapture(event.pointerId);if(event.button===2||state.tool==='erase'){if(hit)removeFruit(hit.id);return}if(hit){state.selectedId=hit.id;drag={fruit:hit,dx:hit.x-point.x,dy:hit.y-point.y,before:snapshot()};renderUi();draw();return}if(state.tool==='select'){state.selectedId=null;renderUi();draw();return}place(point)});
canvas.addEventListener('pointermove',event=>{const point=pointFromEvent(event);hover=point;const frame=canvas.parentElement.getBoundingClientRect(),rect=canvas.getBoundingClientRect(),label=$('coordinate');label.style.display='block';label.style.left=`${event.clientX-frame.left+12}px`;label.style.top=`${event.clientY-frame.top+12}px`;label.textContent=`${point.x.toFixed(0)}, ${point.y.toFixed(0)}`;
  if(drag){drag.fruit.x=point.x+drag.dx;drag.fruit.y=point.y+drag.dy;clampFruit(drag.fruit);renderSelection();renderValidation()}draw()});
canvas.addEventListener('pointerup',()=>{if(drag){if(drag.before!==snapshot())remember();drag=null;renderUi();draw()}});canvas.addEventListener('pointerleave',()=>{hover=null;$('coordinate').style.display='none';if(!drag)draw()});
canvas.addEventListener('wheel',event=>{event.preventDefault();const point=pointFromEvent(event),target=hitFruit(point)||state.fruits.find(item=>item.id===state.selectedId);cycleLevel(event.deltaY>0?1:-1,target)},{passive:false});
document.addEventListener('keydown',event=>{if(/INPUT|SELECT|TEXTAREA/.test(event.target.tagName))return;if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();event.shiftKey?redo():undo();return}if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='y'){event.preventDefault();redo();return}if(event.key==='Delete'||event.key==='Backspace'){const selected=state.fruits.find(item=>item.id===state.selectedId);if(selected){event.preventDefault();removeFruit(selected.id)}}if(event.key.toLowerCase()==='v')state.tool='select';if(event.key.toLowerCase()==='b')state.tool='place';if(event.key.toLowerCase()==='e')state.tool='erase';renderUi()});
document.querySelectorAll('.tool').forEach(button=>button.onclick=()=>{state.tool=button.dataset.tool;renderUi();draw()});document.querySelectorAll('.side-tab').forEach(button=>button.onclick=()=>selectTab(button.dataset.tab));
document.querySelectorAll('[data-fps]').forEach(button=>button.onclick=()=>{state.fps=Number(button.dataset.fps);remember();renderUi()});
$('template').onchange=event=>loadTemplate(event.target.value);$('probe').oninput=event=>{state.probe=Number(event.target.value);renderProbe();draw()};$('probe').onchange=remember;
['show-anchors','show-grid','show-warnings'].forEach(id=>$(id).onchange=event=>{const key={"show-anchors":'showAnchors',"show-grid":'showGrid',"show-warnings":'showWarnings'}[id];state[key]=event.target.checked;draw()});
$('scene-title').onchange=remember;$('undo').onclick=undo;$('redo').onclick=redo;$('clear-scene').onclick=()=>loadTemplate('empty');$('center-view').onclick=()=>toast('视图已居中','当前画布始终保持完整场地比例。');$('settle').onclick=()=>requestEvaluation('settle');$('evaluate').onclick=()=>requestEvaluation('all');$('evaluate-top').onclick=()=>requestEvaluation('all');
$('export').onclick=()=>{const blob=new Blob([JSON.stringify(cloneScene(),null,2)],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=($('scene-title').value.trim()||'scenario').replace(/[\\/:*?"<>|]/g,'_')+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);toast('场景已导出',`${state.fruits.length} 个水果 · ${state.fps} FPS`)};
$('import').onclick=()=>$('import-file').click();$('import-file').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{loadScene(JSON.parse(await file.text()));$('template').value='empty';toast('场景已导入',file.name)}catch(error){toast('导入失败',String(error.message||error))}event.target.value=''};
window.daxiguaScenarioLab={getScene:cloneScene,loadScene,applyEvaluation,setBackendStatus(connected,message=''){const dot=document.querySelector('.canvas-badge i');dot.style.background=connected?'#79d7a8':'#f2c14e';$('canvas-status').textContent=connected?`物理已连接 · ${state.fps} FPS`:`编辑模式 · ${state.fps} FPS`;if(message)toast(connected?'后端已连接':'后端状态',message)}};
loadTemplate('critical');history=[snapshot()];future=[];renderUi();draw();
})();
</script>
</body>
</html>'''
