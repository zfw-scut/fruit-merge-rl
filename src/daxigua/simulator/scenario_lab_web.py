"""自定义场景实验室的无依赖前端页面与Reward V2诊断视图。"""

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
.fruit-choice.dragging{background:var(--accent-soft);border-color:rgba(15,108,189,.55)}
.fruit-drag-preview{position:fixed;z-index:80;width:54px;height:54px;object-fit:contain;pointer-events:none;
  transform:translate(-50%,-50%);filter:drop-shadow(0 8px 8px rgba(20,35,55,.30));opacity:.90}
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
.live-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:9px}.live-strip div{padding:8px 9px;
  border:1px solid var(--stroke);border-radius:9px;background:#f7f9fc}.live-strip span{display:block;color:var(--muted);font-size:9px}
.live-strip strong{display:block;margin-top:3px;font-size:12px;font-variant-numeric:tabular-nums}.live-toggle.running{color:#fff;
  border-color:#075a9e;background:var(--accent)}button:disabled{opacity:.42;cursor:not-allowed!important;transform:none!important}
.empty-result{padding:28px 18px;text-align:center}.empty-illustration{width:76px;height:76px;margin:0 auto 14px;display:grid;place-items:center;
  border-radius:24px;background:linear-gradient(145deg,#e6f3ff,#f0eafb);color:#306ea8;font-size:31px;box-shadow:inset 0 0 0 1px rgba(15,108,189,.08)}
.empty-result strong{display:block;font-size:14px}.empty-result p{max-width:220px;margin:7px auto 0;color:var(--muted);font-size:11px;line-height:1.55}
.result-summary{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.result-card{padding:10px;border:1px solid var(--stroke);border-radius:9px;
  background:#fff}.result-card span{display:block;color:var(--muted);font-size:9px;margin-bottom:4px}.result-card strong{font-size:15px}
.action-results{height:134px;display:grid;grid-template-columns:repeat(21,minmax(5px,1fr));align-items:end;gap:3px;padding:9px 7px 20px;
  border:1px solid var(--stroke);border-radius:10px;background:#f7f9fc}.action-result{height:100%;position:relative;display:flex;align-items:flex-end}
.action-result{cursor:pointer;border-radius:4px}.action-result:hover,.action-result.selected{background:rgba(15,108,189,.10)}
.action-result i{display:block;width:100%;height:5%;min-height:3px;border-radius:3px 3px 1px 1px;background:#aeb8c6}.action-result.best i{
  background:linear-gradient(#62b6ee,#0f6cbd)}.action-result span{position:absolute;bottom:-14px;width:100%;text-align:center;font-size:7px;color:#8993a2}
.space-slots{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.space-slot{min-height:42px;border:1px solid var(--stroke);border-radius:8px;background:#fff;cursor:pointer;color:var(--muted);font-size:9px}.space-slot strong{display:block;color:var(--text);font-size:11px}.space-slot.active{border-color:rgba(15,108,189,.55);background:var(--accent-soft);color:#075a9e}
.alignment-note{margin-top:8px;color:var(--muted);font-size:9px;line-height:1.5}.reward-positive{color:var(--success)!important}.reward-negative{color:var(--danger)!important}
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
    <div class="brand"><div class="brand-mark" aria-hidden="true">◇</div><div class="brand-copy"><strong>场景实验室</strong><span>Scenario Lab · Reward V2</span></div></div>
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
          <div class="canvas-badge"><i></i><span id="canvas-status">正在连接实时物理…</span></div>
          <div id="canvas-tip" class="canvas-tip">从左侧拖入水果，或在画布按下后拖动、松手投放</div>
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
          <div class="section-title"><strong>当前选择</strong><span class="hint">暂停后可调整</span></div>
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
          <div class="section-title"><strong>实时物理</strong><span id="live-state-label" class="hint">未连接</span></div>
          <div class="live-strip">
            <div><span>当前得分</span><strong id="live-score">0</strong></div>
            <div><span>物理帧</span><strong id="live-frame">0</strong></div>
            <div><span>世界状态</span><strong id="live-stable">—</strong></div>
          </div>
          <div class="run-stack">
            <button id="live-toggle" class="secondary live-toggle">暂停并进入编辑</button>
            <button id="settle" class="secondary">评估当前探针动作</button>
            <button id="evaluate" class="primary">并行评估 21 个动作</button>
          </div>
          <div class="backend-note"><i></i><span>实时世界与游戏一样持续推进；奖励评估只读取按下按钮时的状态快照，不会阻塞投放。</span></div>
        </section>
      </div>

      <div id="tab-result" class="tab-page">
        <div id="result-empty" class="empty-result"><div class="empty-illustration">⌁</div><strong>等待第一次真实评估</strong><p>连接本地服务后，这里会对比 21 个动作的空间奖励、终局风险与投放后局面。</p></div>
        <div id="result-content" hidden>
          <section class="section">
            <div class="section-title"><strong>动作摘要</strong><span id="result-fps" class="hint">—</span></div>
            <div class="result-summary">
              <div class="result-card"><span>推荐动作</span><strong id="best-action">—</strong></div>
              <div class="result-card"><span>最高空间奖励</span><strong id="best-reward">—</strong></div>
              <div class="result-card"><span>当前查看动作</span><strong id="selected-action">—</strong></div>
              <div class="result-card"><span>当前动作奖励</span><strong id="selected-reward">—</strong></div>
              <div class="result-card"><span>投放前势能</span><strong id="space-before">—</strong></div>
              <div class="result-card"><span>投放后势能</span><strong id="space-after">—</strong></div>
              <div class="result-card"><span>原始空间变化</span><strong id="space-delta">—</strong></div>
              <div class="result-card"><span>水果占用补偿</span><strong id="space-compensation">—</strong></div>
              <div class="result-card"><span>当前得分增量</span><strong id="best-score">—</strong></div>
              <div class="result-card"><span>终局动作</span><strong id="terminal-actions">—</strong></div>
            </div>
          </section>
          <section class="section"><div class="section-title"><strong>未来水果空间</strong><span class="hint">21列近似</span></div><div id="space-slots" class="space-slots"></div><div class="alignment-note">投放前 q1～q3 与投放后 q0～q2 严格对齐；新随机 q3 不参与当前奖励。画布中绿色为新增空间、红色为损失空间，虚线和实线分别表示投放前后可抵达深度。</div></section>
          <section class="section"><div class="section-title"><strong>21 动作空间奖励</strong><span class="hint">点击切换结果</span></div><div id="action-results" class="action-results"></div><div id="result-message" class="result-message"></div></section>
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
let state={fruits:[],queue:[1,2,3,4],selectedId:null,level:1,tool:'place',fps:120,probe:10,
  showAnchors:true,showGrid:true,showWarnings:true,title:'空白场景',livePaused:false,score:0,stepCount:0,physicsFrame:0,
  stable:true,done:false,liveSequence:-1};
let nextId=1,hover=null,drag=null,history=[],future=[],historyLock=false;
let evaluation=null,selectedResultAction=null,selectedSpaceSlot=0,backendConnected=false,liveEventSource=null;
let visualPreviousFrame=null,visualCurrentFrame=null,visualFruits=[],visualTransitionMs=0,visualFramesReady=false;
let liveRenderHandle=null,lastDynamicRender=0;
let autoEditPending=null,transientEditActive=false,autoEditFinishing=false;
const dropPreviews=new Map();

function spec(level){return FRUITS[level-1]}
function setVisualFruitsExact(fruits,physicsFrame=state.physicsFrame){const frame={fruits,physicsFrame:Number(physicsFrame)||0,receivedAt:performance.now()};visualPreviousFrame=frame;visualCurrentFrame=frame;visualFruits=fruits;visualTransitionMs=0;visualFramesReady=true}
function interpolateLiveFruits(now=performance.now()){if(!visualFramesReady||!visualCurrentFrame)return state.fruits;if(!visualPreviousFrame||visualTransitionMs<=0)return visualCurrentFrame.fruits;const alpha=Math.max(0,Math.min(1,(now-visualCurrentFrame.receivedAt)/visualTransitionMs)),previousById=new Map(visualPreviousFrame.fruits.map(fruit=>[fruit.id,fruit]));return visualCurrentFrame.fruits.map(fruit=>{const previous=previousById.get(fruit.id);if(!previous)return fruit;return {...fruit,x:previous.x+(fruit.x-previous.x)*alpha,y:previous.y+(fruit.y-previous.y)*alpha,angle:previous.angle+(fruit.angle-previous.angle)*alpha}})}
function visibleSceneFruits(){return backendConnected&&!state.livePaused&&visualFramesReady?visualFruits:state.fruits}
function renderLiveFrame(now){liveRenderHandle=requestAnimationFrame(renderLiveFrame);if(!backendConnected||state.livePaused||state.done||!visualFramesReady)return;visualFruits=interpolateLiveFruits(now);draw()}
function ensureLiveRenderLoop(){if(liveRenderHandle===null)liveRenderHandle=requestAnimationFrame(renderLiveFrame)}
function actionX(index,level=state.queue[0]){const radius=spec(level).radius,left=BOARD.wall+radius+2,right=BOARD.width-BOARD.wall-radius-2;return left+(right-left)*index/(BOARD.actions-1)}
function cloneScene(){return {version:1,board:{...BOARD},name:$('scene-title').value.trim()||'未命名场景',fps:state.fps,queue:[...state.queue],probe_action:state.probe,
  score:state.score||0,step_count:state.stepCount||0,
  fruits:state.fruits.map(({id,level,x,y,physicsRadius,vx=0,vy=0,angle=0,angularVelocity=0,ageFrames=0})=>({id,level,x:+x.toFixed(3),y:+y.toFixed(3),physics_radius:physicsRadius||spec(level).merged_physics_radius,vx:+vx.toFixed(3),vy:+vy.toFixed(3),angle:+angle.toFixed(6),angular_velocity:+angularVelocity.toFixed(6),age_frames:Math.max(0,Math.trunc(ageFrames))}))}}
function snapshot(){return JSON.stringify(cloneScene())}
function invalidateEvaluation(){evaluation=null;selectedResultAction=null;draw()}
function remember(){if(historyLock)return;invalidateEvaluation();const value=snapshot();if(history.at(-1)!==value){history.push(value);if(history.length>80)history.shift()}future=[];updateUndo()}
function updateUndo(){const editable=(!backendConnected||state.livePaused)&&!autoEditPending&&!transientEditActive&&!autoEditFinishing;$('undo').disabled=!editable||history.length<2;$('redo').disabled=!editable||!future.length}
function restore(raw){const scene=typeof raw==='string'?JSON.parse(raw):raw;loadScene(scene,{remember:false,sync:false});syncPausedScene();draw();renderUi()}
function undo(){if(!state.livePaused||history.length<2)return;future.push(history.pop());restore(history.at(-1));updateUndo()}
function redo(){if(!state.livePaused||!future.length)return;const value=future.pop();history.push(value);restore(value);updateUndo()}
function clampFruit(fruit){const r=spec(fruit.level).radius;fruit.x=Math.max(BOARD.wall+r,Math.min(BOARD.width-BOARD.wall-r,fruit.x));fruit.y=Math.max(r,Math.min(BOARD.height-BOARD.wall-r,fruit.y))}
function loadTemplate(name,{sync=true,notify=true}={}){const item=templates[name]||templates.empty;nextId=1;state.fruits=item.fruits.map(([level,x,y])=>({id:nextId++,level,x,y,vx:0,vy:0,angle:0,angularVelocity:0,ageFrames:0}));state.queue=[...item.queue];state.selectedId=null;state.score=0;state.stepCount=0;state.physicsFrame=0;state.title=item.title;state.livePaused=backendConnected?true:state.livePaused;$('scene-title').value=item.title;remember();renderUi();draw();if(sync&&backendConnected)syncLiveScene(true);if(notify)toast('场景已载入',backendConnected?`${item.title} · 已暂停供编辑`:item.title)}
function loadScene(scene,{remember:shouldRemember=true,sync=true}={}){
  if(!scene||!Array.isArray(scene.fruits))throw new Error('场景文件缺少 fruits 数组');
  nextId=1;state.fruits=scene.fruits.slice(0,64).map(item=>{const level=Math.max(1,Math.min(11,Number(item.level)||1));const fruit={id:Number(item.id)||nextId,level,x:Number(item.x),y:Number(item.y),physicsRadius:Number(item.physics_radius)||spec(level).merged_physics_radius,vx:Number(item.vx)||0,vy:Number(item.vy)||0,angle:Number(item.angle)||0,angularVelocity:Number(item.angular_velocity)||0,ageFrames:Math.max(0,Math.trunc(Number(item.age_frames)||0))};nextId=Math.max(nextId,fruit.id+1);if(!Number.isFinite(fruit.x)||!Number.isFinite(fruit.y))throw new Error('水果坐标必须是有限数值');clampFruit(fruit);return fruit});
  state.queue=(Array.isArray(scene.queue)?scene.queue:[1,2,3,4]).slice(0,4).map(value=>Math.max(1,Math.min(5,Number(value)||1)));while(state.queue.length<4)state.queue.push(1);
  state.fps=Number(scene.fps)===30?30:120;state.probe=Math.max(0,Math.min(20,Number(scene.probe_action)||10));state.selectedId=null;state.score=Number(scene.score)||0;state.stepCount=Math.max(0,Math.trunc(Number(scene.step_count)||0));
  $('scene-title').value=String(scene.name||'导入场景').slice(0,80);if(shouldRemember)remember();renderUi();draw();if(sync&&backendConnected){state.livePaused=true;syncLiveScene(true)}
}
function pointFromEvent(event){const rect=canvas.getBoundingClientRect();return {x:(event.clientX-rect.left)*BOARD.width/rect.width,y:(event.clientY-rect.top)*BOARD.height/rect.height,rect}}
function hitFruit(point,fruits=state.fruits){for(let i=fruits.length-1;i>=0;i--){const fruit=fruits[i],r=spec(fruit.level).radius;if(Math.hypot(point.x-fruit.x,point.y-fruit.y)<=r)return fruit}return null}
function overlaps(){const pairs=new Set;for(let a=0;a<state.fruits.length;a++)for(let b=a+1;b<state.fruits.length;b++){const x=state.fruits[a],y=state.fruits[b];if(Math.hypot(x.x-y.x,x.y-y.y)<spec(x.level).radius+spec(y.level).radius-1){pairs.add(x.id);pairs.add(y.id)}}return pairs}
function place(point,level=state.level){if(!state.livePaused&&backendConnected)return;if(state.fruits.length>=64){toast('无法继续放置','场景最多包含 64 个水果');return}const fruit={id:nextId++,level,x:point.x,y:point.y,physicsRadius:spec(level).merged_physics_radius,vx:0,vy:0,angle:0,angularVelocity:0,ageFrames:0};clampFruit(fruit);state.fruits.push(fruit);state.selectedId=fruit.id;remember();renderUi();draw();syncPausedScene()}
function removeFruit(id){if(!state.livePaused&&backendConnected)return;const index=state.fruits.findIndex(item=>item.id===id);if(index<0)return;state.fruits.splice(index,1);if(state.selectedId===id)state.selectedId=null;remember();renderUi();draw();syncPausedScene()}
function cycleLevel(direction,target=null){if(target&&backendConnected&&!state.livePaused)return;if(target){target.level=1+(target.level-1+direction+11)%11;target.physicsRadius=spec(target.level).merged_physics_radius;clampFruit(target);state.level=target.level}else state.level=1+(state.level-1+direction+11)%11;if(target){remember();syncPausedScene()}renderUi();draw()}

function drawFruit(fruit,{ghost=false,invalid=false,selected=false}={}){const r=spec(fruit.level).radius;ctx.save();ctx.globalAlpha=ghost?.46:1;
  if(invalid){ctx.beginPath();ctx.arc(fruit.x,fruit.y,r+5,0,Math.PI*2);ctx.fillStyle='rgba(255,91,77,.22)';ctx.fill()}
  ctx.shadowColor='rgba(0,0,0,.32)';ctx.shadowBlur=ghost?0:12;ctx.shadowOffsetY=ghost?0:5;
  if(images[fruit.level]&&images[fruit.level].complete){ctx.save();ctx.translate(fruit.x,fruit.y);ctx.rotate(Number(fruit.angle)||0);ctx.drawImage(images[fruit.level],-r,-r,r*2,r*2);ctx.restore()}else{ctx.beginPath();ctx.arc(fruit.x,fruit.y,r,0,Math.PI*2);ctx.fillStyle=COLORS[fruit.level];ctx.fill()}
  ctx.shadowColor='transparent';if(selected){ctx.beginPath();ctx.arc(fruit.x,fruit.y,r+7,0,Math.PI*2);ctx.strokeStyle='#76c7ff';ctx.lineWidth=3;ctx.stroke();ctx.setLineDash([5,5]);ctx.beginPath();ctx.arc(fruit.x,fruit.y,r+12,0,Math.PI*2);ctx.strokeStyle='rgba(118,199,255,.48)';ctx.lineWidth=1.5;ctx.stroke()}
  ctx.restore()}
function activeResult(){return evaluation&&selectedResultAction!==null?evaluation.actions[selectedResultAction]:null}
function drawSpaceOverlay(){if(backendConnected&&!state.livePaused)return;const action=activeResult(),slot=action?.space_slots?.[selectedSpaceSlot];if(!slot)return;const before=slot.before,after=slot.after,xs=after.drop_x||before.drop_x||[];ctx.save();ctx.lineWidth=1;
  xs.forEach((x,index)=>{const beforeEnd=BOARD.spawnY+(before.depths[index]||0),afterEnd=BOARD.spawnY+(after.depths[index]||0),common=Math.min(beforeEnd,afterEnd);ctx.fillStyle='rgba(76,180,224,.095)';ctx.fillRect(x-4,BOARD.spawnY,8,Math.max(0,common-BOARD.spawnY));if(afterEnd>beforeEnd){ctx.fillStyle='rgba(46,184,92,.28)';ctx.fillRect(x-4,beforeEnd,8,afterEnd-beforeEnd)}else if(afterEnd<beforeEnd){ctx.fillStyle='rgba(221,76,66,.30)';ctx.fillRect(x-4,afterEnd,8,beforeEnd-afterEnd)}ctx.save();ctx.setLineDash([3,3]);ctx.strokeStyle='rgba(177,213,239,.72)';ctx.beginPath();ctx.moveTo(x-6,beforeEnd);ctx.lineTo(x+6,beforeEnd);ctx.stroke();ctx.restore();ctx.strokeStyle=afterEnd>=beforeEnd?'rgba(83,222,133,.92)':'rgba(255,110,96,.94)';ctx.beginPath();ctx.moveTo(x-6,afterEnd);ctx.lineTo(x+6,afterEnd);ctx.stroke()});ctx.restore()}
function draw(){ctx.clearRect(0,0,BOARD.width,BOARD.height);const bg=ctx.createLinearGradient(0,0,0,BOARD.height);bg.addColorStop(0,'#17263a');bg.addColorStop(1,'#0e1724');ctx.fillStyle=bg;ctx.fillRect(0,0,BOARD.width,BOARD.height);
  if(state.showGrid){ctx.strokeStyle='rgba(148,177,207,.085)';ctx.lineWidth=1;for(let y=BOARD.spawnY;y<BOARD.height;y+=56){ctx.beginPath();ctx.moveTo(BOARD.wall,y+.5);ctx.lineTo(BOARD.width-BOARD.wall,y+.5);ctx.stroke()}for(let x=BOARD.wall;x<BOARD.width;x+=56){ctx.beginPath();ctx.moveTo(x,BOARD.spawnY);ctx.lineTo(x,BOARD.height-BOARD.wall);ctx.stroke()}}
  ctx.fillStyle='#795b43';ctx.fillRect(0,BOARD.spawnY,BOARD.wall,BOARD.height-BOARD.spawnY);ctx.fillRect(BOARD.width-BOARD.wall,BOARD.spawnY,BOARD.wall,BOARD.height-BOARD.spawnY);ctx.fillRect(0,BOARD.height-BOARD.wall,BOARD.width,BOARD.wall);
  ctx.fillStyle='rgba(255,255,255,.12)';ctx.fillRect(BOARD.wall,BOARD.spawnY,2,BOARD.height-BOARD.spawnY);ctx.fillRect(BOARD.width-BOARD.wall-2,BOARD.spawnY,2,BOARD.height-BOARD.spawnY);
  ctx.save();ctx.setLineDash([8,7]);ctx.strokeStyle='rgba(255,112,94,.82)';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(BOARD.wall+4,BOARD.spawnY);ctx.lineTo(BOARD.width-BOARD.wall-4,BOARD.spawnY);ctx.stroke();ctx.restore();ctx.fillStyle='rgba(255,170,156,.82)';ctx.font='600 11px "Segoe UI"';ctx.fillText('危险线',BOARD.wall+10,BOARD.spawnY-10);
  drawSpaceOverlay();
  if(state.showAnchors){for(let i=0;i<BOARD.actions;i++){const x=actionX(i);ctx.beginPath();ctx.arc(x,BOARD.spawnY,i===state.probe?4:2.2,0,Math.PI*2);ctx.fillStyle=i===state.probe?'#83d2ff':'rgba(189,214,238,.48)';ctx.fill()}ctx.fillStyle='rgba(194,216,237,.58)';ctx.font='9px "Segoe UI"';ctx.textAlign='center';ctx.fillText('A0',actionX(0),BOARD.spawnY+17);ctx.fillText('A10',actionX(10),BOARD.spawnY+17);ctx.fillText('A20',actionX(20),BOARD.spawnY+17)}
  const shownAction=selectedResultAction===null?state.probe:selectedResultAction,probeX=actionX(shownAction);ctx.save();ctx.setLineDash([7,8]);ctx.strokeStyle='rgba(109,203,255,.70)';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(probeX,74);ctx.lineTo(probeX,BOARD.height-BOARD.wall);ctx.stroke();ctx.restore();if(!activeResult()&&(!backendConnected||state.livePaused))drawFruit({level:state.queue[0],x:probeX,y:118},{ghost:true});
  const invalid=state.showWarnings&&!activeResult()?overlaps():new Set;visibleSceneFruits().forEach(fruit=>drawFruit(fruit,{invalid:invalid.has(fruit.id),selected:fruit.id===state.selectedId}));
  dropPreviews.forEach(preview=>{if(!preview.inside)return;const ghost={level:preview.level,x:preview.x,y:state.livePaused?preview.y:BOARD.spawnY};clampFruit(ghost);drawFruit(ghost,{ghost:true})});
  if(state.livePaused&&hover&&state.tool==='place'&&!drag&&!hitFruit(hover)&&!dropPreviews.size){const ghost={level:state.level,x:hover.x,y:hover.y};clampFruit(ghost);drawFruit(ghost,{ghost:true})}
}

function pointFromClient(clientX,clientY){const rect=canvas.getBoundingClientRect();return {x:(clientX-rect.left)*BOARD.width/rect.width,y:(clientY-rect.top)*BOARD.height/rect.height,rect,inside:clientX>=rect.left&&clientX<=rect.right&&clientY>=rect.top&&clientY<=rect.bottom}}
function updateDropPreview(pointerId,clientX,clientY){const preview=dropPreviews.get(pointerId);if(!preview)return;const point=pointFromClient(clientX,clientY);preview.x=point.x;preview.y=point.y;preview.inside=point.inside;if(preview.element){preview.element.style.left=`${clientX}px`;preview.element.style.top=`${clientY}px`;preview.element.style.opacity=point.inside?'.92':'.62'}draw()}
function finishDropPreview(pointerId,cancelled=false){const preview=dropPreviews.get(pointerId);if(!preview)return;dropPreviews.delete(pointerId);preview.element?.remove();preview.button?.classList.remove('dragging');if(!cancelled&&preview.inside){if(backendConnected&&!state.livePaused)sendLiveCommand({type:'drop',level:preview.level,x:preview.x}).catch(error=>toast('投放失败',String(error.message||error)));else place({x:preview.x,y:preview.y},preview.level)}draw()}
function startPaletteDrag(event,item,button){if(autoEditPending||transientEditActive||autoEditFinishing)return;if(event.button!==0&&event.pointerType==='mouse')return;event.preventDefault();state.level=item.level;state.tool='place';const element=document.createElement('img');element.className='fruit-drag-preview';element.src=TEXTURES[item.level];element.alt='';document.body.append(element);button.classList.add('dragging');button.setPointerCapture(event.pointerId);dropPreviews.set(event.pointerId,{level:item.level,x:0,y:0,inside:false,element,button});updateDropPreview(event.pointerId,event.clientX,event.clientY);document.querySelectorAll('.fruit-choice').forEach(choice=>choice.classList.toggle('active',choice===button));document.querySelectorAll('.tool').forEach(tool=>tool.classList.toggle('active',tool.dataset.tool==='place'));renderLiveStatus();draw()}
function renderFruitGrid(){const busy=Boolean(autoEditPending||transientEditActive||autoEditFinishing),grid=$('fruit-grid');grid.replaceChildren(...FRUITS.map(item=>{const button=document.createElement('button');button.className='fruit-choice'+(item.level===state.level?' active':'');button.disabled=busy;button.title=busy?'完成当前拖动后可继续投放':`${item.level}级 · ${item.name} · 拖到画布投放`;button.innerHTML=`<img src="${TEXTURES[item.level]}" alt=""><span>${item.name}</span><i>${item.level}</i>`;button.onpointerdown=event=>startPaletteDrag(event,item,button);button.onpointermove=event=>updateDropPreview(event.pointerId,event.clientX,event.clientY);button.onpointerup=event=>finishDropPreview(event.pointerId);button.onpointercancel=event=>finishDropPreview(event.pointerId,true);return button}))}
function renderQueue(){const editable=(!backendConnected||state.livePaused)&&!autoEditPending&&!transientEditActive&&!autoEditFinishing,root=$('queue');root.replaceChildren(...state.queue.map((level,index)=>{const button=document.createElement('button');button.className='queue-slot';button.disabled=!editable;button.title=editable?'点击升级；Shift+点击降级':'暂停物理后可编辑未来队列';button.innerHTML=`<em>q${index}</em><img src="${TEXTURES[level]}" alt=""><span>${spec(level).name}</span>`;button.onclick=event=>{if(!editable)return;state.queue[index]=1+(level-1+(event.shiftKey?-1:1)+5)%5;remember();renderUi();draw();syncPausedScene()};return button}))}
function renderSelection(){const fruit=state.fruits.find(item=>item.id===state.selectedId),root=$('selection');if(!fruit){root.className='selection-card empty';root.textContent=backendConnected&&!state.livePaused?'按住场上水果可临时编辑；右键可直接删除':'尚未选择场上水果';return}root.className='selection-card';root.innerHTML=`<img src="${TEXTURES[fruit.level]}" alt=""><div class="selection-copy"><strong>${fruit.level}级 · ${spec(fruit.level).name}</strong><span>x ${fruit.x.toFixed(1)} · y ${fruit.y.toFixed(1)} · 半径 ${spec(fruit.level).radius}</span></div><button class="small-danger" title="删除水果">⌫</button>`;const button=root.querySelector('button');button.disabled=Boolean(autoEditPending||transientEditActive||autoEditFinishing);button.onclick=()=>backendConnected&&!state.livePaused?removeLiveFruit(fruit.id):removeFruit(fruit.id)}
function renderValidation(){const invalid=overlaps(),danger=state.fruits.filter(fruit=>fruit.y-spec(fruit.level).radius<BOARD.spawnY).length;$('fruit-count').textContent=state.fruits.length;$('danger-count').textContent=danger;$('overlap-count').textContent=invalid.size;
  $('danger-count').parentElement.className='mini-stat'+(danger?' danger':'');$('overlap-count').parentElement.className='mini-stat'+(invalid.size?' warning':'');const box=$('validation');
  if(invalid.size){box.className='validation warning';box.querySelector('span').textContent=`检测到 ${invalid.size} 个水果参与重叠。可以保留以测试极端状态，但后端载入时应再次校验。`}else if(danger){box.className='validation warning';box.querySelector('span').textContent=`有 ${danger} 个水果越过危险线，适合验证终局与奖励边界。`}else{box.className='validation';box.querySelector('span').textContent='场景几何检查通过，可以接入物理模拟。'}}
function renderProbe(){const x=actionX(state.probe);$('probe').value=state.probe;$('probe-action').textContent=`A${state.probe}`;$('probe-x').textContent=`x = ${x.toFixed(1)}`}
function renderLiveStatus(){const autoEditing=Boolean(autoEditPending||transientEditActive||autoEditFinishing),label=!backendConnected?'离线编辑':state.done?'对局已结束':autoEditing?'按住编辑 · 松手恢复':state.livePaused?'已暂停 · 编辑模式':'实时运行';$('canvas-status').textContent=backendConnected?`${label} · 物理 ${state.physicsFps||120} FPS · 显示同步`:label;$('canvas-tip').textContent=autoEditing?'拖动水果调整位置；松手后自动同步并恢复实时物理':backendConnected&&!state.livePaused?(state.tool==='select'?'选择模式：按住水果直接编辑，右键水果直接删除':'放置模式：空白处拖动投放；按住已有水果直接编辑；右键删除'):'暂停编辑：左键放置 · 拖动调整 · 右键删除 · 滚轮切级';$('live-state-label').textContent=label;$('live-score').textContent=String(state.score||0);$('live-frame').textContent=String(state.physicsFrame||0);$('live-stable').textContent=state.done?'结束':state.stable?'稳定':'运动中';const toggle=$('live-toggle');toggle.disabled=!backendConnected||state.done||autoEditing;toggle.textContent=state.livePaused?'恢复实时运行':'暂停并进入编辑';toggle.classList.toggle('running',backendConnected&&!state.livePaused&&!state.done);document.querySelector('.canvas-badge i').style.background=!backendConnected?'#f2c14e':autoEditing?'#f2c14e':state.livePaused?'#75b9ee':'#79d7a8'}
function renderUi(){const editable=!backendConnected||state.livePaused,busy=Boolean(autoEditPending||transientEditActive||autoEditFinishing);document.querySelectorAll('.tool').forEach(button=>{button.classList.toggle('active',button.dataset.tool===state.tool);button.disabled=busy||(!editable&&button.dataset.tool==='erase')});renderFruitGrid();renderQueue();renderSelection();renderValidation();renderProbe();
  $('show-anchors').checked=state.showAnchors;$('show-grid').checked=state.showGrid;$('show-warnings').checked=state.showWarnings;document.querySelectorAll('[data-fps]').forEach(button=>{button.classList.toggle('active',Number(button.dataset.fps)===state.fps);button.disabled=busy});['template','clear-scene','import','scene-title','settle','evaluate','evaluate-top'].forEach(id=>$(id).disabled=busy);canvas.style.cursor=drag?'grabbing':state.tool==='select'?'default':state.tool==='erase'?'not-allowed':'crosshair';renderLiveStatus();updateUndo()}
function toast(title,message){const item=document.createElement('div');item.className='toast';item.innerHTML=`<strong></strong><span></span>`;item.querySelector('strong').textContent=title;item.querySelector('span').textContent=message;$('toasts').append(item);setTimeout(()=>{item.classList.add('out');setTimeout(()=>item.remove(),220)},3200)}
function selectTab(name){document.querySelectorAll('.side-tab').forEach(tab=>tab.classList.toggle('active',tab.dataset.tab===name));document.querySelectorAll('.tab-page').forEach(page=>page.classList.toggle('active',page.id===`tab-${name}`))}
function rewardClass(value){return value>0?'reward-positive':value<0?'reward-negative':''}
function renderEvaluation(){if(!evaluation)return;const actions=evaluation.actions.slice(0,21),rewards=actions.map(item=>Number(item.reward)||0),bestIndex=Number.isInteger(evaluation.best_action)?evaluation.best_action:rewards.indexOf(Math.max(...rewards)),best=rewards[bestIndex]||0,current=actions[selectedResultAction]||actions[bestIndex];$('best-action').textContent=`A${bestIndex}`;$('best-reward').textContent=best.toFixed(4);$('best-reward').className=rewardClass(best);$('selected-action').textContent=`A${selectedResultAction}`;$('selected-reward').textContent=Number(current.reward).toFixed(4);$('selected-reward').className=rewardClass(Number(current.reward));$('space-before').textContent=Number(current.previous_potential).toFixed(4);$('space-after').textContent=Number(current.next_potential).toFixed(4);$('space-delta').textContent=Number(current.raw_space_delta).toFixed(5);$('space-delta').className=rewardClass(Number(current.raw_space_delta));$('space-compensation').textContent=Number(current.compensation).toFixed(5);$('best-score').textContent=String(current.score_delta??'—');$('terminal-actions').textContent=String(actions.filter(item=>item.done).length);$('result-fps').textContent=`${evaluation.physics_fps||state.fps} FPS · Reward V2`;
  const min=Math.min(0,...rewards),max=Math.max(0,...rewards),span=Math.max(1e-8,max-min),root=$('action-results');root.replaceChildren(...Array.from({length:21},(_,index)=>{const value=rewards[index]??0,item=document.createElement('div');item.className='action-result'+(index===bestIndex?' best':'')+(index===selectedResultAction?' selected':'');item.title=`A${index} · Reward V2 ${value.toFixed(5)}`;item.innerHTML=`<i style="height:${8+82*(value-min)/span}%;background:${value<0?'linear-gradient(#f08b82,#c42b1c)':''}"></i><span>${index}</span>`;item.onclick=()=>{selectedResultAction=index;renderEvaluation();draw()};return item}));
  const slots=current.space_slots||[];$('space-slots').replaceChildren(...slots.map((slot,index)=>{const button=document.createElement('button'),area=slot.after.effective_normalized_area??slot.after.normalized_area;button.className='space-slot'+(index===selectedSpaceSlot?' active':'');button.innerHTML=`<strong>q${index+1} · ${spec(slot.level).name}</strong>权重 ${Number(slot.weight).toFixed(1)} · 有效面积 ${Number(area).toFixed(3)}${current.done?' · 终局':''}`;button.onclick=()=>{selectedSpaceSlot=index;renderEvaluation();draw()};return button}));$('result-message').textContent=evaluation.message||'结果来自已连接的真实物理与Reward V2后端。';draw()}
async function requestEvaluation(mode='all'){const requestMode=mode==='settle'?'probe':mode,detail={mode:requestMode,scene:cloneScene()};window.dispatchEvent(new CustomEvent('daxigua:scenario-request',{detail}));selectTab('result');if(!backendConnected){toast('后端未连接','请使用 --serve 启动真实物理与Reward V2服务。');return}toast('开始真实评估','正在并行执行21个投放动作…');try{const response=await fetch('/api/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(detail)}),payload=await response.json();if(!response.ok)throw new Error(payload.error||`HTTP ${response.status}`);applyEvaluation(payload);toast('评估完成',`已返回 ${payload.actions.length} 个真实物理结果`)}catch(error){toast('评估失败',String(error.message||error))}}
function applyEvaluation(payload){if(!payload||!Array.isArray(payload.actions)||payload.actions.length!==21)throw new Error('评估结果必须包含21个 actions');evaluation=payload;selectedResultAction=Number.isInteger(payload.selected_action)?payload.selected_action:(Number.isInteger(payload.best_action)?payload.best_action:0);selectedSpaceSlot=0;$('result-empty').hidden=true;$('result-content').hidden=false;renderEvaluation();selectTab('result')}

async function sendLiveCommand(command){if(!backendConnected)throw new Error('实时物理后端未连接');const response=await fetch('/api/live/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({command})}),payload=await response.json();if(!response.ok)throw new Error(payload.error||`HTTP ${response.status}`);return payload}
function pushLiveScene(paused=state.livePaused){if(!backendConnected)return Promise.resolve();return sendLiveCommand({type:'load_scene',scene:cloneScene(),paused:Boolean(paused)})}
function syncLiveScene(paused=state.livePaused){return pushLiveScene(paused).catch(error=>toast('场景同步失败',String(error.message||error)))}
function syncPausedScene(){if(backendConnected&&state.livePaused)return syncLiveScene(true);return Promise.resolve()}
function applyLivePayload(payload){if(!payload||!Array.isArray(payload.fruits)||!Array.isArray(payload.queue))return;if(Number(payload.sequence)<=state.liveSequence)return;const receivedAt=performance.now(),incomingFrame=Number(payload.physics_frame)||0,nextPaused=Boolean(payload.paused),nextFruits=payload.fruits.map(item=>({id:Number(item.id),level:Number(item.level),x:Number(item.x),y:Number(item.y),vx:Number(item.vx)||0,vy:Number(item.vy)||0,angle:Number(item.angle)||0,angularVelocity:Number(item.angular_velocity)||0,ageFrames:Number(item.age_frames)||0,physicsRadius:Number(item.physics_radius)||spec(Number(item.level)).merged_physics_radius}));if(!visualCurrentFrame||nextPaused||incomingFrame<=visualCurrentFrame.physicsFrame){setVisualFruitsExact(nextFruits,incomingFrame)}else{const startFruits=interpolateLiveFruits(receivedAt),frameDelta=Math.max(1,incomingFrame-visualCurrentFrame.physicsFrame),physicsFps=Number(payload.physics_fps)||120;visualPreviousFrame={fruits:startFruits,physicsFrame:visualCurrentFrame.physicsFrame,receivedAt};visualCurrentFrame={fruits:nextFruits,physicsFrame:incomingFrame,receivedAt};visualTransitionMs=Math.max(8,Math.min(50,frameDelta*1000/physicsFps));visualFruits=startFruits;visualFramesReady=true}state.liveSequence=Number(payload.sequence);state.fruits=nextFruits;state.queue=payload.queue.map(Number);state.score=Number(payload.score)||0;state.stepCount=Number(payload.step_count)||0;state.physicsFrame=incomingFrame;state.physicsFps=Number(payload.physics_fps)||120;state.publishFps=Number(payload.publish_fps)||60;state.livePaused=nextPaused;if(!state.livePaused&&state.tool==='erase')state.tool='select';state.stable=Boolean(payload.stable);state.done=Boolean(payload.done);nextId=Math.max(1,...state.fruits.map(item=>item.id+1));if(!state.fruits.some(item=>item.id===state.selectedId))state.selectedId=null;renderLiveStatus();if(state.livePaused||state.done)draw();else ensureLiveRenderLoop();const now=performance.now();if(now-lastDynamicRender>100){lastDynamicRender=now;renderQueue();renderSelection();renderValidation();updateUndo()}}
function openLiveStream(){liveEventSource?.close();liveEventSource=new EventSource('/api/live/events');liveEventSource.onmessage=event=>{try{applyLivePayload(JSON.parse(event.data))}catch(error){console.error(error)}};liveEventSource.onerror=()=>{$('live-state-label').textContent='状态流重连中…'}}
async function toggleLive(){if(!backendConnected)return;try{const targetPaused=!state.livePaused;await sendLiveCommand({type:targetPaused?'pause':'resume'});state.livePaused=targetPaused;state.selectedId=null;if(targetPaused){history=[snapshot()];future=[]}else if(state.tool==='erase')state.tool='select';renderUi();draw();toast(targetPaused?'物理已暂停':'实时物理已恢复',targetPaused?'现在可以拖动、删除或导入水果。':'世界将持续模拟，可随时连续投放。')}catch(error){toast('状态切换失败',String(error.message||error))}}

async function readLiveState(){const response=await fetch('/api/live/state',{cache:'no-store'}),payload=await response.json();if(!response.ok)throw new Error(payload.error||`HTTP ${response.status}`);return payload}
async function removeLiveFruit(fruitId){try{const result=await sendLiveCommand({type:'remove',fruit_id:fruitId});if(!result.accepted){toast('删除未执行','水果可能已经在合成过程中消失。');return}if(state.selectedId===fruitId)state.selectedId=null;renderSelection()}catch(error){toast('删除失败',String(error.message||error))}}
async function beginTransientEdit(pending){let paused=false;try{await sendLiveCommand({type:'pause'});paused=true;const payload=await readLiveState();applyLivePayload(payload);if(autoEditPending!==pending)throw new Error('临时编辑手势已失效');transientEditActive=true;state.livePaused=true;const fruit=state.fruits.find(item=>item.id===pending.fruitId);if(!fruit)throw new Error('水果已在暂停前合成或消失');drag={pointerId:pending.pointerId,fruit,dx:fruit.x-pending.initialPoint.x,dy:fruit.y-pending.initialPoint.y,before:snapshot(),autoResume:true};fruit.x=pending.latestPoint.x+drag.dx;fruit.y=pending.latestPoint.y+drag.dy;clampFruit(fruit);renderUi();draw();if(pending.released)finishTransientEdit(drag,pending.cancelled)}catch(error){if(autoEditPending===pending)autoEditPending=null;transientEditActive=false;drag=null;if(paused){try{await sendLiveCommand({type:'resume'});state.livePaused=false}catch(resumeError){console.error(resumeError)}}renderUi();draw();if(String(error.message||error)!=='临时编辑手势已失效')toast('无法开始临时编辑',String(error.message||error))}}
async function finishTransientEdit(activeDrag,cancelled=false){if(autoEditFinishing||!activeDrag)return;autoEditFinishing=true;if(cancelled)loadScene(JSON.parse(activeDrag.before),{remember:false,sync:false});else if(activeDrag.before!==snapshot())remember();drag=null;renderUi();draw();try{await pushLiveScene(true);await sendLiveCommand({type:'resume'});state.livePaused=false;transientEditActive=false;autoEditPending=null;autoEditFinishing=false;renderUi();draw()}catch(error){transientEditActive=false;autoEditPending=null;autoEditFinishing=false;state.livePaused=true;renderUi();draw();toast('编辑已保留但未恢复',`场景仍处于暂停状态：${String(error.message||error)}`)}}

canvas.addEventListener('contextmenu',event=>event.preventDefault());
canvas.addEventListener('pointerdown',event=>{const point=pointFromEvent(event),running=backendConnected&&!state.livePaused,fruits=running?visibleSceneFruits():state.fruits,hit=hitFruit(point,fruits);if(event.button===2&&event.pointerType==='mouse'){event.preventDefault();if(hit){running?removeLiveFruit(hit.id):removeFruit(hit.id)}return}if(event.button!==0&&event.pointerType==='mouse')return;if(autoEditPending||autoEditFinishing)return;event.preventDefault();canvas.setPointerCapture(event.pointerId);if(running){if(hit){const pending={pointerId:event.pointerId,fruitId:hit.id,initialPoint:point,latestPoint:point,released:false,cancelled:false};autoEditPending=pending;state.selectedId=hit.id;renderUi();draw();beginTransientEdit(pending);return}if(state.tool==='select'){state.selectedId=null;renderUi();draw();return}if(state.tool==='erase'){state.tool='select';renderUi();draw();return}dropPreviews.set(event.pointerId,{level:state.level,x:point.x,y:point.y,inside:true});draw();return}if(event.button===2||state.tool==='erase'){if(hit)removeFruit(hit.id);return}if(hit){state.selectedId=hit.id;drag={pointerId:event.pointerId,fruit:hit,dx:hit.x-point.x,dy:hit.y-point.y,before:snapshot(),autoResume:false};renderUi();draw();return}if(state.tool==='select'){state.selectedId=null;renderUi();draw();return}place(point)});
canvas.addEventListener('pointermove',event=>{const point=pointFromEvent(event);hover=point;const frame=canvas.parentElement.getBoundingClientRect(),rect=canvas.getBoundingClientRect(),label=$('coordinate');label.style.display='block';label.style.left=`${event.clientX-frame.left+12}px`;label.style.top=`${event.clientY-frame.top+12}px`;label.textContent=`${point.x.toFixed(0)}, ${point.y.toFixed(0)}`;
  if(autoEditPending&&autoEditPending.pointerId===event.pointerId)autoEditPending.latestPoint=point;if(dropPreviews.has(event.pointerId)){updateDropPreview(event.pointerId,event.clientX,event.clientY);return}if(drag&&drag.pointerId===event.pointerId){drag.fruit.x=point.x+drag.dx;drag.fruit.y=point.y+drag.dy;clampFruit(drag.fruit);renderSelection();renderValidation()}draw()});
canvas.addEventListener('pointerup',event=>{if(dropPreviews.has(event.pointerId)){finishDropPreview(event.pointerId);return}if(autoEditPending&&autoEditPending.pointerId===event.pointerId){autoEditPending.released=true;if(drag&&drag.pointerId===event.pointerId)finishTransientEdit(drag,false);else renderLiveStatus();return}if(drag&&drag.pointerId===event.pointerId){if(drag.before!==snapshot()){remember();syncPausedScene()}drag=null;renderUi();draw()}});canvas.addEventListener('pointercancel',event=>{if(dropPreviews.has(event.pointerId))finishDropPreview(event.pointerId,true);if(autoEditPending&&autoEditPending.pointerId===event.pointerId){autoEditPending.released=true;autoEditPending.cancelled=true;if(drag&&drag.pointerId===event.pointerId)finishTransientEdit(drag,true);return}if(drag&&drag.pointerId===event.pointerId){loadScene(JSON.parse(drag.before),{remember:false,sync:false});drag=null;renderUi();draw()}});canvas.addEventListener('pointerleave',()=>{hover=null;$('coordinate').style.display='none';if(!drag)draw()});
canvas.addEventListener('wheel',event=>{event.preventDefault();const point=pointFromEvent(event),target=backendConnected&&!state.livePaused?null:(hitFruit(point)||state.fruits.find(item=>item.id===state.selectedId));cycleLevel(event.deltaY>0?1:-1,target)},{passive:false});
document.addEventListener('keydown',event=>{if(/INPUT|SELECT|TEXTAREA/.test(event.target.tagName))return;if(event.key==='Escape'){if(autoEditPending){event.preventDefault();autoEditPending.released=true;autoEditPending.cancelled=true;if(drag)finishTransientEdit(drag,true);return}if(dropPreviews.size){event.preventDefault();[...dropPreviews.keys()].forEach(pointerId=>finishDropPreview(pointerId,true));return}}if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='z'){event.preventDefault();event.shiftKey?redo():undo();return}if((event.ctrlKey||event.metaKey)&&event.key.toLowerCase()==='y'){event.preventDefault();redo();return}if(event.key==='Delete'||event.key==='Backspace'){const selected=state.fruits.find(item=>item.id===state.selectedId);if(selected){event.preventDefault();backendConnected&&!state.livePaused?removeLiveFruit(selected.id):removeFruit(selected.id)}}if(event.key.toLowerCase()==='v')state.tool='select';if(event.key.toLowerCase()==='b')state.tool='place';if(event.key.toLowerCase()==='e'&&(!backendConnected||state.livePaused))state.tool='erase';renderUi();draw()});
document.querySelectorAll('.tool').forEach(button=>button.onclick=()=>{if(backendConnected&&!state.livePaused&&button.dataset.tool==='erase'){toast('擦除需暂停物理','选择和放置可在运行时随时切换。');return}state.tool=button.dataset.tool;renderUi();draw()});document.querySelectorAll('.side-tab').forEach(button=>button.onclick=()=>selectTab(button.dataset.tab));
document.querySelectorAll('[data-fps]').forEach(button=>button.onclick=()=>{state.fps=Number(button.dataset.fps);remember();renderUi()});
$('template').onchange=event=>loadTemplate(event.target.value);$('probe').oninput=event=>{state.probe=Number(event.target.value);renderProbe();draw()};$('probe').onchange=remember;
['show-anchors','show-grid','show-warnings'].forEach(id=>$(id).onchange=event=>{const key={"show-anchors":'showAnchors',"show-grid":'showGrid',"show-warnings":'showWarnings'}[id];state[key]=event.target.checked;draw()});
$('scene-title').onchange=()=>{if(!backendConnected||state.livePaused){remember();syncPausedScene()}};$('undo').onclick=undo;$('redo').onclick=redo;$('clear-scene').onclick=async()=>{if(!backendConnected){loadTemplate('empty');return}try{await sendLiveCommand({type:'clear',queue:[...state.queue]});state.selectedId=null;$('scene-title').value='空白场景';toast('场景已清空','实时物理保持当前运行状态。')}catch(error){toast('清空失败',String(error.message||error))}};$('center-view').onclick=()=>toast('视图已居中','当前画布始终保持完整场地比例。');$('live-toggle').onclick=toggleLive;$('settle').onclick=()=>requestEvaluation('probe');$('evaluate').onclick=()=>requestEvaluation('all');$('evaluate-top').onclick=()=>requestEvaluation('all');
$('export').onclick=()=>{const blob=new Blob([JSON.stringify(cloneScene(),null,2)],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=($('scene-title').value.trim()||'scenario').replace(/[\\/:*?"<>|]/g,'_')+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);toast('场景已导出',`${state.fruits.length} 个水果 · ${state.fps} FPS`)};
$('import').onclick=()=>$('import-file').click();$('import-file').onchange=async event=>{const file=event.target.files[0];if(!file)return;try{loadScene(JSON.parse(await file.text()));$('template').value='empty';toast('场景已导入',backendConnected?`${file.name} · 已暂停供编辑`:file.name)}catch(error){toast('导入失败',String(error.message||error))}event.target.value=''};
function setBackendStatus(connected,message=''){backendConnected=Boolean(connected);renderUi();if(message)toast(backendConnected?'实时后端已连接':'后端状态',message)}
async function connectBackend(){if(location.protocol==='file:'){setBackendStatus(false);return}try{const response=await fetch('/api/health',{cache:'no-store'}),payload=await response.json();if(!response.ok||!payload.ready||!payload.live_physics)throw new Error(payload.error||'实时物理后端未就绪');backendConnected=true;const stateResponse=await fetch('/api/live/state',{cache:'no-store'}),livePayload=await stateResponse.json();if(!stateResponse.ok)throw new Error(livePayload.error||'无法读取实时世界');applyLivePayload(livePayload);openLiveStream();setBackendStatus(true,`${payload.reward_version||'Reward V2'} · ${payload.device||'未知设备'} · 持续物理`)}catch(error){setBackendStatus(false,String(error.message||error))}}
window.daxiguaScenarioLab={getScene:cloneScene,loadScene,applyEvaluation,setBackendStatus};
loadTemplate('empty',{sync:false,notify:false});$('template').value='empty';history=[snapshot()];future=[];renderUi();draw();connectBackend();
})();
</script>
</body>
</html>'''
