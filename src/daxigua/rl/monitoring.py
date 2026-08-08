"""与训练热路径隔离的资源监控和只读 Web 面板。"""

from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import os
from pathlib import Path
import queue
import threading
import time
import math

from .curves import (
    CURVE_FILENAME,
    existing_curve_metadata,
    render_training_curve_snapshot,
)
from .event_analysis import EVENT_ANALYSIS_FILENAME


_DASHBOARD_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>合成大西瓜 - GNN-DQN 训练监视器</title>
<style>
:root{
  --desktop:#008080;--control:#c0c0c0;--light:#fff;--light2:#dfdfdf;
  --mid:#808080;--dark:#404040;--black:#0a0a0a;--blue:#000080;
  --title2:#1084d0;--text:#111;--screen:#fff;--good:#008000;
  --warn:#a06000;--bad:#b00020;--grid:#d7d7d7;
}
*{box-sizing:border-box}
html{min-width:360px;background:var(--desktop)}
body{margin:0;padding:12px;background:var(--desktop);color:var(--text);
  font:12px Tahoma,"Microsoft YaHei UI","Microsoft YaHei",sans-serif;
  font-variant-numeric:tabular-nums}
.window{max-width:1540px;margin:0 auto;background:var(--control);padding:3px;
  box-shadow:inset 1px 1px var(--light),inset -1px -1px var(--black),
  inset 2px 2px var(--light2),inset -2px -2px var(--dark),4px 5px 0 rgba(0,0,0,.28)}
.titlebar{height:30px;padding:4px 5px;display:flex;align-items:center;gap:7px;
  color:#fff;background:linear-gradient(90deg,var(--blue),var(--title2));font-weight:700}
.app-icon{width:19px;height:19px;position:relative;background:#fff;border:1px solid #000;
  box-shadow:inset 1px 1px #7fd6ff}
.app-icon:before,.app-icon:after{content:"";position:absolute;border-radius:50%}
.app-icon:before{width:10px;height:10px;left:2px;bottom:2px;background:#1ca33b}
.app-icon:after{width:8px;height:8px;right:1px;top:2px;background:#f2b400}
.title{font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.title-buttons{margin-left:auto;display:flex;gap:2px}
.title-button{width:21px;height:20px;display:grid;place-items:center;color:#000;background:var(--control);
  box-shadow:inset 1px 1px var(--light),inset -1px -1px var(--black),
  inset 2px 2px var(--light2),inset -2px -2px var(--mid);font:700 11px monospace}
.menubar{height:25px;display:flex;align-items:center;gap:20px;padding:0 8px;border-bottom:1px solid var(--mid)}
.menubar span:first-letter{text-decoration:underline}.toolbar{display:flex;align-items:center;gap:8px;
  margin:3px;padding:5px 7px;background:var(--control);box-shadow:inset 1px 1px var(--light),inset -1px -1px var(--mid)}
.lamp{width:10px;height:10px;border:1px solid #333;border-radius:50%;background:#d0a000;box-shadow:inset 1px 1px rgba(255,255,255,.7)}
.lamp.live{background:#16a016}.lamp.stale{background:#d0a000}.lamp.error{background:#d02020}
.toolbar strong{font-size:12px}.toolbar .right{margin-left:auto;color:#444}
main{padding:5px;display:grid;gap:8px}
.top-grid{display:grid;grid-template-columns:repeat(4,minmax(230px,1fr));gap:8px}
fieldset{min-width:0;margin:0;padding:12px 9px 9px;border:0;background:var(--control);
  box-shadow:inset 1px 1px var(--mid),inset -1px -1px var(--light)}
legend{padding:0 5px;color:#111;font-weight:700;background:var(--control)}
.metrics{display:grid;grid-template-columns:1fr;gap:3px}
.metric{min-height:24px;display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:10px;
  padding:3px 5px;border-bottom:1px dotted #888}
.metric:last-child{border-bottom:0}.metric-name{min-width:0}.metric-name small{display:block;color:#555;font-size:9px}
.value{min-width:90px;padding:3px 5px;text-align:right;white-space:nowrap;background:var(--screen);
  box-shadow:inset 1px 1px var(--dark),inset -1px -1px var(--light);font-weight:700}
.progress-shell{height:20px;margin:7px 4px 3px;padding:2px;background:#fff;
  box-shadow:inset 1px 1px var(--dark),inset -1px -1px var(--light)}
.progress-fill{height:100%;width:0;background:repeating-linear-gradient(90deg,#000080 0 11px,#fff 11px 13px);transition:width .3s}
.progress-text{text-align:center;font-weight:700}
.chart-grid{display:grid;grid-template-columns:2fr 1fr 1fr;gap:8px}.score-panel{grid-row:span 1}
.chart-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}.chart-head strong{font-size:12px}
.chart-legend{display:flex;flex-wrap:wrap;gap:10px;color:#333;font-size:10px}.legend-item{display:flex;align-items:center;gap:4px}
.legend-line{width:16px;height:3px;background:var(--line-color)}
.canvas-frame{height:246px;padding:2px;background:#fff;box-shadow:inset 1px 1px var(--dark),inset -1px -1px var(--light)}
.small-chart .canvas-frame{height:246px}canvas{display:block;width:100%;height:100%;background:#fff}
.action-panel{overflow:hidden}.action-summary{margin-bottom:6px;color:#444}.actions{height:150px;display:grid;
  grid-template-columns:repeat(21,minmax(18px,1fr));align-items:end;gap:3px;padding:8px 6px 0;background:#fff;
  box-shadow:inset 1px 1px var(--dark),inset -1px -1px var(--light);overflow-x:auto}
.action{height:100%;min-width:18px;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:3px}
.action-bar-shell{width:100%;height:105px;display:flex;align-items:flex-end;background:#eee;border-left:1px solid #aaa;border-bottom:1px solid #777}
.action-bar{width:100%;min-height:1px;background:#000080}.action-label{font-size:8px;color:#333}.action-value{font-size:8px;color:#555}
.events{max-height:145px;overflow:auto;background:#fff;box-shadow:inset 1px 1px var(--dark),inset -1px -1px var(--light)}
.event{display:grid;grid-template-columns:140px minmax(0,1fr) 80px;gap:8px;padding:5px 7px;border-bottom:1px solid #ddd}
.event-kind{font-weight:700;color:#000080}.event-time{text-align:right;color:#555}.empty{padding:12px;color:#666;text-align:center}
.statusbar{display:grid;grid-template-columns:1fr auto auto;gap:3px;padding:3px}
.status-cell{min-height:22px;padding:4px 7px;box-shadow:inset 1px 1px var(--mid),inset -1px -1px var(--light)}
.good{color:var(--good)}.warn{color:var(--warn)}.bad{color:var(--bad)}
@media(max-width:1180px){.top-grid{grid-template-columns:repeat(2,minmax(230px,1fr))}.chart-grid{grid-template-columns:1fr 1fr}.score-panel{grid-column:1/-1}}
@media(max-width:700px){body{padding:0}.window{padding:2px}.menubar{gap:10px}.toolbar .right{display:none}.top-grid,.chart-grid{grid-template-columns:1fr}.score-panel{grid-column:auto}.canvas-frame,.small-chart .canvas-frame{height:220px}.event{grid-template-columns:95px 1fr}.event-time{display:none}.statusbar{grid-template-columns:1fr}.status-cell:nth-child(n+2){display:none}}
</style>
<style>
/* Windows 11 Fluent 风格覆盖层 */
:root{
  --desktop:#eef3f8;--control:rgba(255,255,255,.72);--light:#fff;
  --mid:#d7dce2;--dark:#7c8591;--black:#202124;--blue:#0067c0;
  --title2:#60a5e8;--text:#1b1b1b;--screen:rgba(255,255,255,.86);
  --good:#0f7b0f;--warn:#9d5d00;--bad:#c42b1c;--grid:#e5e8eb;
}
html{background:#e9f0f7}
body{min-height:100vh;padding:20px;background:
  radial-gradient(circle at 12% -8%,rgba(96,165,232,.34),transparent 32%),
  radial-gradient(circle at 92% 12%,rgba(170,137,255,.22),transparent 30%),
  linear-gradient(145deg,#edf4fa 0%,#f7f8fa 48%,#e9f0f7 100%);
  color:var(--text);font:13px "Segoe UI Variable Text","Segoe UI","Microsoft YaHei UI",sans-serif}
.window{max-width:1540px;margin:0 auto;padding:0;overflow:hidden;border:1px solid rgba(0,0,0,.08);
  border-radius:14px;background:rgba(248,249,251,.78);backdrop-filter:blur(28px) saturate(135%);
  box-shadow:0 24px 65px rgba(31,45,61,.18),0 2px 8px rgba(31,45,61,.08)}
.titlebar{height:58px;padding:0 20px;gap:12px;color:#202020;background:rgba(255,255,255,.66);
  border-bottom:1px solid rgba(0,0,0,.06);font-weight:600}
.app-icon{width:30px;height:30px;border:0;border-radius:8px;background:linear-gradient(145deg,#e7f3ff,#cde7ff);
  box-shadow:inset 0 0 0 1px rgba(0,95,184,.15),0 1px 2px rgba(0,0,0,.08)}
.app-icon:before{width:15px;height:15px;left:4px;bottom:4px;background:#32a852}
.app-icon:after{width:12px;height:12px;right:3px;top:4px;background:#ffb900}
.title{font-size:15px;letter-spacing:.01em}.title-buttons{display:none}
.menubar{height:45px;padding:5px 18px;gap:6px;border:0;background:rgba(255,255,255,.42)}
.menubar span{padding:8px 14px;border-radius:7px;color:#3b3b3b;text-decoration:none;transition:background .15s}
.menubar span:first-letter{text-decoration:none}.menubar span.active{color:#005fb8;background:rgba(255,255,255,.92);
  box-shadow:0 1px 3px rgba(0,0,0,.08)}
.toolbar{min-height:44px;margin:0;padding:8px 20px;gap:9px;border-top:1px solid rgba(255,255,255,.55);
  border-bottom:1px solid rgba(0,0,0,.06);background:rgba(246,248,251,.72);box-shadow:none}
.lamp{width:9px;height:9px;border:0;box-shadow:0 0 0 4px rgba(255,185,0,.14)}
.lamp.live{background:#0f7b0f;box-shadow:0 0 0 4px rgba(15,123,15,.13)}
.lamp.error{background:#c42b1c;box-shadow:0 0 0 4px rgba(196,43,28,.13)}
.toolbar .right{color:#666}
main{padding:14px;gap:12px}.top-grid,.chart-grid{gap:12px}
fieldset{padding:15px 14px 14px;border:1px solid rgba(0,0,0,.07);border-radius:10px;
  background:rgba(255,255,255,.76);box-shadow:0 1px 2px rgba(0,0,0,.045),0 5px 18px rgba(44,62,80,.035)}
legend{padding:0 5px;color:#242424;background:transparent;font-size:14px;font-weight:600}
.metrics{gap:2px}.metric{min-height:31px;padding:5px 4px;border-bottom:1px solid rgba(0,0,0,.055)}
.metric-name{font-weight:500}.metric-name small{margin-top:1px;color:#777;font-size:10px;font-weight:400}
.value{min-width:102px;padding:5px 8px;border:1px solid rgba(0,0,0,.055);border-radius:6px;
  background:rgba(247,249,251,.88);box-shadow:none;color:#202020;font-weight:600}
.progress-shell{height:11px;margin:12px 3px 5px;padding:0;overflow:hidden;border:0;border-radius:999px;
  background:#e5e7ea;box-shadow:none}.progress-fill{border-radius:inherit;background:linear-gradient(90deg,#005fb8,#2b88d8);transition:width .35s ease}
.progress-text{color:#4a4a4a;font-size:11px;font-weight:500}
.chart-head{min-height:30px;margin-bottom:8px}.chart-head strong{font-size:13px;font-weight:600}.chart-legend{color:#606060}
.legend-line{height:3px;border-radius:3px}.canvas-frame{height:250px;padding:0;overflow:hidden;border:1px solid rgba(0,0,0,.065);
  border-radius:8px;background:rgba(255,255,255,.92);box-shadow:none}.small-chart .canvas-frame{height:250px}
canvas{background:#fff}.action-summary{margin:0 2px 9px;color:#5a5a5a}
.actions{height:164px;gap:5px;padding:12px 10px 7px;overflow-x:auto;border:1px solid rgba(0,0,0,.065);
  border-radius:8px;background:rgba(250,251,252,.9);box-shadow:none}
.action{gap:5px}.action-bar-shell{height:108px;overflow:hidden;border:0;border-radius:5px 5px 2px 2px;background:#e8ebef}
.action-bar{border-radius:5px 5px 0 0;background:linear-gradient(180deg,#2b88d8,#0067c0)}
.action-label{color:#555;font-size:9px}.action-value{color:#666;font-size:9px}
.events{max-height:155px;overflow:auto;border:1px solid rgba(0,0,0,.065);border-radius:8px;background:rgba(255,255,255,.8);box-shadow:none}
.event{min-height:36px;align-items:center;padding:7px 10px;border-bottom:1px solid rgba(0,0,0,.055)}
.event-kind{color:#005fb8}.statusbar{gap:7px;padding:8px 14px 12px;background:rgba(247,248,250,.7)}
.status-cell{min-height:27px;padding:6px 10px;border:1px solid rgba(0,0,0,.055);border-radius:6px;
  background:rgba(255,255,255,.62);box-shadow:none;color:#555}
.snapshot-layout{display:grid;grid-template-columns:minmax(0,1fr) 230px;gap:14px;align-items:start}
.snapshot-frame{min-height:280px;display:grid;place-items:center;overflow:hidden;border:1px solid rgba(0,0,0,.065);
  border-radius:9px;background:rgba(255,255,255,.94)}
.snapshot-frame img{display:block;width:100%;height:auto;object-fit:contain}
.snapshot-info{padding:13px;border:1px solid rgba(0,0,0,.055);border-radius:9px;background:rgba(247,249,251,.86);color:#555;line-height:1.7}
.snapshot-info strong{display:block;margin-bottom:5px;color:#202020;font-size:14px}.snapshot-info .snapshot-error{color:var(--bad)}
@media(max-width:700px){body{padding:0}.window{border-radius:0}.titlebar{height:54px;padding:0 14px}.menubar{padding-left:10px;overflow-x:auto}.menubar span{white-space:nowrap}.toolbar{padding:8px 14px}main{padding:9px}}
@media(max-width:880px){.snapshot-layout{grid-template-columns:1fr}.snapshot-info{order:-1}}
</style>
</head>
<body>
<section class="window">
  <header class="titlebar">
    <span class="app-icon" aria-hidden="true"></span>
    <span class="title">合成大西瓜 - GNN-DQN 训练监视器</span>
  </header>
  <nav class="menubar" aria-label="面板导航"><span class="active">概览</span><span>训练曲线</span><span>系统资源</span><span>评估记录</span><span>事件与告警</span></nav>
  <div class="toolbar">
    <span id="status-lamp" class="lamp"></span><strong id="connection">正在连接训练进程…</strong>
    <span id="phase-text">阶段：等待数据</span><span class="right" id="updated-at">最后更新：—</span>
  </div>
  <main>
    <section class="top-grid">
      <fieldset><legend>训练进度 / Progress</legend><div id="progress" class="metrics"></div><div class="progress-shell"><div id="progress-fill" class="progress-fill"></div></div><div id="progress-text" class="progress-text">0.00%</div></fieldset>
      <fieldset><legend>实时吞吐 / Throughput</legend><div id="throughput" class="metrics"></div></fieldset>
      <fieldset><legend>服务器资源 / Resources</legend><div id="resources" class="metrics"></div></fieldset>
      <fieldset><legend>学习状态 / Learning</legend><div id="learning" class="metrics"></div></fieldset>
    </section>

    <section class="chart-grid">
      <fieldset class="score-panel"><legend>训练效果曲线 / Training Effect</legend>
        <div class="chart-head"><strong>局分变化</strong><div class="chart-legend">
          <span class="legend-item"><i class="legend-line" style="--line-color:#0067c0"></i>窗口平均分</span>
          <span class="legend-item"><i class="legend-line" style="--line-color:#d13438"></i>窗口最高分</span>
          <span class="legend-item"><i class="legend-line" style="--line-color:#107c10"></i>滚动平均分</span>
          <span class="legend-item"><i class="legend-line" style="--line-color:#0099bc"></i>30 FPS 评估</span>
          <span class="legend-item"><i class="legend-line" style="--line-color:#8764b8"></i>120 FPS 评估</span>
        </div></div>
        <div class="canvas-frame"><canvas id="score-chart" aria-label="训练分数变化曲线"></canvas></div>
      </fieldset>
      <fieldset class="small-chart"><legend>优化曲线 / Optimization</legend><div class="chart-head"><strong>损失与 TD 误差</strong><span class="chart-legend"><span class="legend-item"><i class="legend-line" style="--line-color:#0067c0"></i>Huber 损失</span><span class="legend-item"><i class="legend-line" style="--line-color:#d13438"></i>TD 误差</span></span></div><div class="canvas-frame"><canvas id="loss-chart" aria-label="损失变化曲线"></canvas></div></fieldset>
      <fieldset class="small-chart"><legend>性能曲线 / Performance</legend><div class="chart-head"><strong>投放与学习吞吐</strong><span class="chart-legend"><span class="legend-item"><i class="legend-line" style="--line-color:#0067c0"></i>投放</span><span class="legend-item"><i class="legend-line" style="--line-color:#107c10"></i>学习样本</span></span></div><div class="canvas-frame"><canvas id="speed-chart" aria-label="训练吞吐变化曲线"></canvas></div></fieldset>
      <fieldset class="small-chart"><legend>辅助监督 / Auxiliary</legend><div class="chart-head"><strong>动作效果损失分组</strong></div><div class="canvas-frame"><canvas id="aux-loss-chart" aria-label="辅助动作效果损失曲线"></canvas></div></fieldset>
      <fieldset class="small-chart"><legend>评估事件 / Eval Events</legend><div class="chart-head"><strong>高等级生成密度（每千次投放）</strong></div><div class="canvas-frame"><canvas id="merge-density-chart" aria-label="高等级水果生成密度曲线"></canvas></div></fieldset>
    </section>

    <fieldset class="snapshot-panel"><legend>定期保存曲线 / Saved Curve Snapshot</legend>
      <div class="snapshot-layout">
        <div class="snapshot-frame"><img id="curve-snapshot" alt="训练曲线定期保存图片" hidden></div>
        <div id="curve-snapshot-info" class="snapshot-info"><strong>等待第一张曲线快照</strong>训练指标开始落盘后，后台会自动生成并刷新图片。</div>
      </div>
    </fieldset>

    <fieldset class="snapshot-panel"><legend>最终评估分布 / Evaluation Distribution</legend>
      <div class="snapshot-layout">
        <div class="snapshot-frame"><img id="event-snapshot" alt="评估分数密度与关键事件图片" hidden></div>
        <div id="event-snapshot-info" class="snapshot-info"><strong>等待最终评估</strong>两种物理帧率评估完成后自动生成分数密度和高等级关键事件图。</div>
      </div>
    </fieldset>

    <fieldset class="action-panel"><legend>投放动作分布 / Action Distribution</legend><div id="action-summary" class="action-summary">等待动作统计…</div><div id="actions" class="actions"></div></fieldset>
    <fieldset><legend>最近事件与告警 / Events</legend><div id="events" class="events"><div class="empty">等待训练事件…</div></div></fieldset>
  </main>
  <footer class="statusbar"><span id="status-message" class="status-cell">就绪</span><span id="run-time" class="status-cell">运行时间：—</span><span class="status-cell">只读监控 · 30 FPS 训练 / 120 FPS 仅评估</span></footer>
</section>
<script>
const $=id=>document.getElementById(id);
const finite=value=>{const n=Number(value);return Number.isFinite(n)?n:null};
const int=value=>{const n=finite(value);return n===null?'—':new Intl.NumberFormat('zh-CN',{maximumFractionDigits:0}).format(n)};
const decimal=(value,digits=3)=>{const n=finite(value);if(n===null)return '—';if(n!==0&&Math.abs(n)<.001)return n.toExponential(2);return new Intl.NumberFormat('zh-CN',{minimumFractionDigits:digits,maximumFractionDigits:digits}).format(n)};
const rate=value=>{const n=finite(value);return n===null?'—':`${new Intl.NumberFormat('zh-CN',{maximumFractionDigits:1}).format(n)} /秒`};
const percent=(value,digits=1)=>{const n=finite(value);return n===null?'—':`${decimal(n,digits)}%`};
const ratioPercent=value=>{const n=finite(value);return n===null?'—':percent(n*100,2)};
const score=value=>{const n=finite(value);return n===null?'—':decimal(n,1)};
const gib=value=>{const n=finite(value);return n===null?'—':`${decimal(n/1024,2)} GiB`};
const duration=value=>{const n=finite(value);if(n===null||n<0)return '—';const s=Math.round(n);const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60),r=s%60;if(d)return `${d}天 ${h}小时 ${m}分`;if(h)return `${h}小时 ${m}分 ${r}秒`;if(m)return `${m}分 ${r}秒`;return `${r}秒`};
const short=value=>{const n=finite(value);if(n===null)return '—';if(Math.abs(n)>=1e8)return `${decimal(n/1e8,2)}亿`;if(Math.abs(n)>=1e4)return `${decimal(n/1e4,1)}万`;return int(n)};
const escapeHtml=value=>String(value??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const phaseNames={training:'训练中',evaluation:'评估中',warmup:'Replay 预热',finished:'已完成',failed:'失败'};
const eventNames={preflight_finished:'预检完成',autotune_finished:'性能标定完成',evaluation_finished:'评估完成',training_started:'训练启动',training_finished:'训练完成',training_failed:'训练失败',warmup_started:'预热开始',warmup_finished:'预热完成',autoscale_trial:'扩容试运行',autoscale_commit:'扩容确认',autoscale_rollback:'扩容回退',checkpoint_saved:'模型已保存'};

const groups={
 progress:[
  ['transitions','累计投放','Transitions',(v)=>int(v)],['total_transitions','计划投放','Target',(v)=>int(v)],
  ['updates','模型更新','Updates',(v)=>int(v)],['episodes','已完成局数','Episodes',(v)=>int(v)],
  ['active_envs','活跃环境','Active Envs',(v)=>int(v)],['epsilon','探索率','Epsilon',(v)=>ratioPercent(v)],
  ['replay_size','Replay 占用','Replay',(v,d)=>`${int(v)} / ${int(d.replay_capacity)}`],
  ['eta_seconds','预计剩余','ETA',(v)=>duration(v)]],
 throughput:[
  ['env_steps_per_second','投放速度','Drops',(v)=>rate(v)],['updates_per_second','更新速度','Updates',(v)=>rate(v)],
  ['learner_samples_per_second','学习样本速度','Learner Samples',(v)=>rate(v)],
  ['physics_seconds','物理耗时（窗口）','Physics',(v)=>`${decimal(v,2)} 秒`],
  ['reward_seconds','奖励几何耗时（窗口）','Reward Geometry',(v)=>`${decimal(v,2)} 秒`],
  ['actor_seconds','决策耗时（窗口）','Actor',(v)=>`${decimal(v,2)} 秒`],
  ['learner_seconds','学习耗时（窗口）','Learner',(v)=>`${decimal(v,2)} 秒`],
  ['training_window_episodes','统计窗口局数','Window Episodes',(v)=>int(v)],
  ['training_window_mean_drops','窗口平均局长','Mean Drops',(v)=>decimal(v,1)]],
 resources:[
  ['gpu_utilization','GPU 使用率','GPU Utilization',(v)=>percent(v,1)],
  ['gpu_memory_used_mb','GPU 显存','VRAM',(v,d)=>`${gib(v)} / ${gib(d.gpu_memory_total_mb)}`],
  ['gpu_temperature','GPU 温度','Temperature',(v)=>`${decimal(v,1)} °C`],
  ['gpu_power_watts','GPU 功耗','Power',(v)=>`${decimal(v,1)} W`],
  ['cpu_utilization','CPU 使用率','CPU Utilization',(v)=>percent(v,1)],
  ['memory_used_mb','系统内存','System RAM',(v,d)=>`${gib(v)} / ${gib(d.memory_total_mb)}`],
  ['process_rss_mb','训练进程内存','Process RSS',(v)=>gib(v)]],
 learning:[
  ['loss','总损失','Total Loss',(v)=>decimal(v,5)],['dqn_loss','DQN 损失','DQN Loss',(v)=>decimal(v,5)],['aux_loss_total','辅助效果损失','Auxiliary Loss',(v)=>decimal(v,5)],['mean_reward','平均奖励','Mean Reward',(v)=>decimal(v,4)],
  ['spatial_reward','空间奖励','Spatial Reward',(v)=>decimal(v,5)],
  ['spatial_previous_potential','投放前空间势能','Space Before',(v)=>decimal(v,4)],
  ['spatial_next_potential','投放后空间势能','Space After',(v)=>decimal(v,4)],
  ['spatial_raw_delta','原始空间变化','Raw Space Delta',(v)=>decimal(v,5)],
  ['spatial_reference_loss','无合成参考损失','No-merge Reference',(v)=>decimal(v,5)],
  ['spatial_positive_rate','正空间奖励比例','Positive Rate',(v)=>ratioPercent(v)],
  ['mean_q','平均 Q 值','Mean Q',(v)=>decimal(v,4)],['mean_target','平均 TD 目标','TD Target',(v)=>decimal(v,4)],
  ['mean_abs_td_error','平均绝对 TD 误差','Abs TD Error',(v)=>decimal(v,4)],['policy_disagreement','策略头分歧','Policy Disagreement',(v)=>decimal(v,5)],['active_learning_action_fraction','主动学习动作比例','Active Learning',(v)=>ratioPercent(v)],['grad_norm','梯度范数','Grad Norm',(v)=>decimal(v,3)],
  ['training_window_mean_score','窗口局均分','Window Mean Score',(v)=>score(v)],
  ['training_window_max_score','窗口最高分','Window Max Score',(v)=>score(v)],
  ['training_rolling_mean_score','近 4096 局均分','Rolling Mean',(v)=>score(v)],
  ['best_training_score','训练历史最高分','Best Training Score',(v)=>score(v)],
  ['last_fast_eval_score','最近 30 FPS 评估','30 FPS Eval',(v)=>score(v)],
  ['last_accurate_eval_score','最近 120 FPS 评估','120 FPS Eval',(v)=>score(v)]]};

function renderGroup(id,definitions,data){$(id).innerHTML=definitions.map(([key,cn,en,formatter])=>`<div class="metric"><span class="metric-name">${cn}<small>${en}</small></span><span class="value">${formatter(data[key],data)}</span></div>`).join('')}
function renderActions(values){const list=Array.isArray(values)?values.map(v=>finite(v)??0):[];if(!list.length){$('actions').innerHTML='<div class="empty">等待动作统计…</div>';return}const max=Math.max(...list,1e-9),sum=list.reduce((a,b)=>a+b,0);let best=0;list.forEach((v,i)=>{if(v>list[best])best=i});const bestRatio=sum>1.01?list[best]/sum:list[best];$('action-summary').textContent=`当前最常选择：动作 A${best}（${percent(bestRatio*100,2)}）；分布来自最近一个统计窗口`;$('actions').innerHTML=list.map((v,i)=>{const p=sum>1.01?v/sum:v;return `<div class="action" title="动作 A${i}：${percent(p*100,2)}"><span class="action-value">${decimal(p*100,1)}%</span><span class="action-bar-shell"><i class="action-bar" style="height:${Math.max(1,v/max*100)}%"></i></span><span class="action-label">A${i}</span></div>`}).join('')}
function drawChart(id,history,series,formatter){const canvas=$(id),rect=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);if(!rect.width||!rect.height)return;canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);const c=canvas.getContext('2d');c.scale(ratio,ratio);const w=rect.width,h=rect.height;c.fillStyle='#fff';c.fillRect(0,0,w,h);const available=series.map(s=>({...s,points:history.map((row,index)=>({index,value:finite(row[s.key])})).filter(p=>p.value!==null)})).filter(s=>s.points.length);if(!available.length){c.fillStyle='#666';c.font='12px Tahoma';c.textAlign='center';c.fillText('等待足够的训练数据…',w/2,h/2);return}const all=available.flatMap(s=>s.points.map(p=>p.value));let min=Math.min(...all),max=Math.max(...all);if(min===max){min-=Math.max(1,Math.abs(min)*.05);max+=Math.max(1,Math.abs(max)*.05)}else{const pad=(max-min)*.1;min-=pad;max+=pad}const box={l:55,r:12,t:27,b:28},pw=w-box.l-box.r,ph=h-box.t-box.b;c.font='10px Tahoma';c.lineWidth=1;for(let i=0;i<=4;i++){const y=box.t+ph*i/4,value=max-(max-min)*i/4;c.strokeStyle='#d0d0d0';c.beginPath();c.moveTo(box.l,y);c.lineTo(w-box.r,y);c.stroke();c.fillStyle='#444';c.textAlign='right';c.textBaseline='middle';c.fillText(formatter(value),box.l-6,y)}const x=index=>box.l+(history.length<=1?0:index/(history.length-1))*pw,y=value=>box.t+ph-(value-min)/(max-min)*ph;available.forEach(s=>{c.strokeStyle=s.color;c.lineWidth=2;c.beginPath();s.points.forEach((p,i)=>{i?c.lineTo(x(p.index),y(p.value)):c.moveTo(x(p.index),y(p.value))});c.stroke();const last=s.points[s.points.length-1];c.fillStyle=s.color;c.fillRect(x(last.index)-2,y(last.value)-2,5,5)});c.fillStyle='#444';c.textBaseline='alphabetic';c.textAlign='left';const firstTransition=history.find(r=>finite(r.transitions)!==null)?.transitions;c.fillText(short(firstTransition),box.l,h-7);c.textAlign='right';const lastTransition=[...history].reverse().find(r=>finite(r.transitions)!==null)?.transitions;c.fillText(short(lastTransition),w-box.r,h-7)}
function renderEvents(events){const rows=(events||[]).slice(-12).reverse();$('events').innerHTML=rows.length?rows.map(event=>`<div class="event"><span class="event-kind">${escapeHtml(eventNames[event.kind]||event.kind||'训练事件')}</span><span>${escapeHtml(event.message||'')}</span><time class="event-time">${new Date((event.monitor_timestamp||0)*1000).toLocaleTimeString('zh-CN',{hour12:false})}</time></div>`).join(''):'<div class="empty">等待训练事件…</div>'}
function renderCurveSnapshot(plot){const image=$('curve-snapshot'),info=$('curve-snapshot-info');if(!plot){image.hidden=true;info.innerHTML='<strong>等待第一张曲线快照</strong>训练指标开始落盘后，后台会自动生成并刷新图片。';return}if(plot.error){info.innerHTML=`<strong>曲线快照暂不可用</strong><span class="snapshot-error">${escapeHtml(plot.error)}</span>`;return}const version=finite(plot.modified_at)??finite(plot.generated_at)??0,url=`${plot.url||'/plots/training_curves.png'}?v=${encodeURIComponent(version)}`;if(image.dataset.version!==String(version)){image.dataset.version=String(version);image.onload=()=>{image.hidden=false};image.src=url}const generated=finite(plot.generated_at);info.innerHTML=`<strong>已自动保存并同步到面板</strong>更新时间：${generated===null?'—':new Date(generated*1000).toLocaleString('zh-CN',{hour12:false})}<br>覆盖投放：${int(plot.source_last_transition)}<br>训练指标点：${int(plot.source_metric_rows)}<br>评估记录：${int(plot.source_evaluation_rows)}<br>图片大小：${int(plot.size_bytes)} 字节`}
function renderEventSnapshot(plot){const image=$('event-snapshot'),info=$('event-snapshot-info');if(!plot){image.hidden=true;return}const version=finite(plot.generated_at)??0,url=`${plot.url||'/plots/evaluation_event_analysis.png'}?v=${encodeURIComponent(version)}`;if(image.dataset.version!==String(version)){image.dataset.version=String(version);image.onload=()=>{image.hidden=false};image.src=url}info.innerHTML=`<strong>关键事件统计已归档</strong>30 FPS：${int(plot.episodes_30fps)} 局<br>120 FPS：${int(plot.episodes_120fps)} 局<br>生成 L11：${int(plot.created_l11_episodes_30fps)} 局<br>消除 L11：${int(plot.removed_l11_episodes_30fps)} 局`}
async function tick(){try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);const state=await response.json(),data={...(state.training||{}),...(state.resources||{})},history=(state.history||[]).slice(-900);renderGroup('progress',groups.progress,data);renderGroup('throughput',groups.throughput,data);renderGroup('resources',groups.resources,data);renderGroup('learning',groups.learning,data);renderActions(data.action_distribution);renderEvents(state.events);renderCurveSnapshot((state.plots||{}).training_curves);renderEventSnapshot((state.plots||{}).evaluation_event_analysis);const fraction=Math.max(0,Math.min(1,finite(data.progress_fraction)??((finite(data.transitions)??0)/Math.max(1,finite(data.total_transitions)??1))));$('progress-fill').style.width=`${fraction*100}%`;$('progress-text').textContent=`${decimal(fraction*100,2)}% · ${int(data.transitions)} / ${int(data.total_transitions)}`;const age=Date.now()/1000-(finite(state.timestamp)??0),lamp=$('status-lamp');lamp.className='lamp '+(age<5?'live':age<30?'stale':'error');$('connection').textContent=age<5?'实时连接正常':age<30?'数据更新延迟':'训练数据已失联';$('phase-text').textContent=`阶段：${phaseNames[data.phase]||data.phase||'等待数据'}`;$('updated-at').textContent=`最后更新：${new Date((state.timestamp||0)*1000).toLocaleString('zh-CN',{hour12:false})}`;$('status-message').textContent=`投放 ${rate(data.env_steps_per_second)} · 更新 ${rate(data.updates_per_second)} · 面板队列丢弃 ${int(data.dropped_messages||0)} 条`;$('run-time').textContent=`运行时间：${duration(data.uptime_seconds)}`;drawChart('score-chart',history,[{key:'training_window_mean_score',color:'#0067c0'},{key:'training_window_max_score',color:'#d13438'},{key:'training_rolling_mean_score',color:'#107c10'},{key:'last_fast_eval_score',color:'#0099bc'},{key:'last_accurate_eval_score',color:'#8764b8'}],v=>short(v));drawChart('loss-chart',history,[{key:'loss',color:'#0067c0'},{key:'dqn_loss',color:'#107c10'},{key:'mean_abs_td_error',color:'#d13438'}],v=>decimal(v,3));drawChart('speed-chart',history,[{key:'env_steps_per_second',color:'#0067c0'},{key:'learner_samples_per_second',color:'#107c10'}],v=>short(v));drawChart('aux-loss-chart',history,[{key:'aux_loss_merge',color:'#0067c0'},{key:'aux_loss_q0_lineage',color:'#107c10'},{key:'aux_loss_first_contact',color:'#ca5010'},{key:'aux_loss_generation',color:'#8764b8'},{key:'aux_loss_outcome',color:'#c239b3'}],v=>decimal(v,3));drawChart('merge-density-chart',history,[{key:'eval_created_l7_per_1000',color:'#69797e'},{key:'eval_created_l8_per_1000',color:'#0099bc'},{key:'eval_created_l9_per_1000',color:'#107c10'},{key:'eval_created_l10_per_1000',color:'#ca5010'},{key:'eval_created_l11_per_1000',color:'#d13438'}],v=>decimal(v,2))}catch(error){$('status-lamp').className='lamp error';$('connection').textContent='无法读取训练状态';$('status-message').textContent=String(error)}finally{setTimeout(tick,1000)}}tick();
</script>
</body>
</html>'''


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class ResourceSampler:
    def __init__(self, target_pid):
        self.target_pid = int(target_pid)
        self.psutil = None
        self.process = None
        self.nvml = None
        self.gpu_handle = None
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process(self.target_pid)
            psutil.cpu_percent(interval=None)
        except (ImportError, OSError):
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except (ImportError, OSError, RuntimeError):
            pass

    def sample(self):
        result = {}
        if self.psutil is not None:
            memory = self.psutil.virtual_memory()
            result.update({
                'cpu_utilization': self.psutil.cpu_percent(interval=None),
                'memory_used_mb': memory.used / 1024 ** 2,
                'memory_total_mb': memory.total / 1024 ** 2,
            })
            if self.process is not None:
                try:
                    result['process_rss_mb'] = (
                        self.process.memory_info().rss / 1024 ** 2
                    )
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    pass
        if self.nvml is not None and self.gpu_handle is not None:
            try:
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(
                    self.gpu_handle
                )
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                result.update({
                    'gpu_utilization': float(utilization.gpu),
                    'gpu_memory_used_mb': memory.used / 1024 ** 2,
                    'gpu_memory_total_mb': memory.total / 1024 ** 2,
                    'gpu_temperature': float(
                        self.nvml.nvmlDeviceGetTemperature(
                            self.gpu_handle,
                            self.nvml.NVML_TEMPERATURE_GPU,
                        )
                    ),
                    'gpu_power_watts': (
                        self.nvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                        / 1000.0
                    ),
                })
            except self.nvml.NVMLError:
                pass
        return result

    def close(self):
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except self.nvml.NVMLError:
                pass


class _DashboardState:
    def __init__(self, history_size):
        self.lock = threading.Lock()
        self.training = {}
        self.resources = {}
        self.plots = {}
        self.events = deque(maxlen=100)
        self.history = deque(maxlen=history_size)
        self.timestamp = time.time()

    def update_training(self, payload):
        with self.lock:
            self.training.update(payload)
            self.timestamp = time.time()
            if 'env_steps_per_second' in payload:
                history_entry = {
                    'timestamp': self.timestamp,
                    'transitions': payload.get(
                        'transitions', self.training.get('transitions')
                    ),
                    'env_steps_per_second': payload.get(
                        'env_steps_per_second', 0.0
                    ),
                    'updates_per_second': payload.get(
                        'updates_per_second', 0.0
                    ),
                }
                optional_names = (
                    'learner_samples_per_second',
                    'loss',
                    'mean_abs_td_error',
                    'training_window_mean_score',
                    'training_window_max_score',
                    'training_rolling_mean_score',
                    'training_rolling_max_score',
                    'best_training_score',
                    'last_fast_eval_score',
                    'last_accurate_eval_score',
                )
                for name in optional_names:
                    if name in payload:
                        history_entry[name] = payload[name]
                self.history.append(history_entry)

    def update_resources(self, payload):
        with self.lock:
            self.resources = dict(payload)

    def add_event(self, payload):
        with self.lock:
            self.events.append(dict(payload))

    def update_plot(self, name, payload):
        with self.lock:
            self.plots[str(name)] = dict(payload)

    def snapshot(self):
        with self.lock:
            return {
                'timestamp': self.timestamp,
                'training': dict(self.training),
                'resources': dict(self.resources),
                'plots': {
                    name: dict(payload)
                    for name, payload in self.plots.items()
                },
                'events': list(self.events),
                'history': list(self.history),
            }


def _dashboard_process_main(
        metric_queue,
        host,
        port,
        resource_interval,
        history_size,
        curve_snapshot_enabled,
        curve_snapshot_interval,
        plot_done_event,
        run_dir,
        target_pid):
    state = _DashboardState(history_size)
    stop_event = threading.Event()
    sampler = ResourceSampler(target_pid)
    run_dir = Path(run_dir)
    output_path = run_dir / 'monitoring.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_lock = threading.Lock()
    previous_plot = existing_curve_metadata(run_dir)
    if previous_plot is not None:
        state.update_plot('training_curves', previous_plot)

    def render_curves_once():
        if not curve_snapshot_enabled:
            return None
        with render_lock:
            try:
                metadata = render_training_curve_snapshot(run_dir)
            except Exception as error:
                fallback = existing_curve_metadata(run_dir) or {}
                state.update_plot('training_curves', {
                    **fallback,
                    'error': f'{type(error).__name__}: {error}',
                    'last_attempt_at': time.time(),
                })
                return None
            state.update_plot('training_curves', metadata)
            return metadata

    def consume():
        with output_path.open('a', encoding='utf-8', buffering=1) as log:
            while not stop_event.is_set():
                try:
                    payload = metric_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if payload is None:
                    stop_event.set()
                    break
                kind = payload.pop('_kind', 'training')
                if kind == 'render_curves':
                    render_curves_once()
                    plot_done_event.set()
                    continue
                payload['monitor_timestamp'] = time.time()
                if kind == 'event':
                    state.add_event(payload)
                elif kind == 'plot':
                    state.update_plot(payload.pop('name'), payload)
                else:
                    state.update_training(payload)
                log.write(json.dumps(
                    {'kind': kind, **payload}, ensure_ascii=False
                ) + '\n')

    def collect_resources():
        resource_path = run_dir / 'resources.jsonl'
        with resource_path.open('a', encoding='utf-8', buffering=1) as log:
            while not stop_event.wait(resource_interval):
                payload = sampler.sample()
                state.update_resources(payload)
                log.write(json.dumps(
                    {'timestamp': time.time(), **payload},
                    ensure_ascii=False,
                ) + '\n')

    def update_curve_snapshot():
        delay = min(5.0, curve_snapshot_interval)
        while not stop_event.is_set():
            if stop_event.wait(delay):
                return
            metadata = render_curves_once()
            delay = (
                curve_snapshot_interval
                if metadata is not None
                else min(15.0, curve_snapshot_interval)
            )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request_path = self.path.partition('?')[0]
            if request_path == '/api/status':
                body = json.dumps(
                    state.snapshot(), ensure_ascii=False
                ).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
            elif request_path in ('/', '/index.html'):
                body = _DASHBOARD_HTML.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
            elif request_path in (
                    f'/plots/{CURVE_FILENAME}',
                    f'/plots/{EVENT_ANALYSIS_FILENAME}'):
                plot_root = run_dir / 'plots'
                plot_path = plot_root / Path(request_path).name
                try:
                    valid = (
                        not plot_root.is_symlink()
                        and plot_root.is_dir()
                        and not plot_path.is_symlink()
                        and plot_path.is_file()
                        and plot_path.resolve(strict=True).parent
                        == plot_root.resolve(strict=True)
                    )
                    body = plot_path.read_bytes() if valid else b'not found'
                except OSError:
                    body = b'not found'
                    valid = False
                self.send_response(200 if valid else 404)
                self.send_header(
                    'Content-Type', 'image/png' if valid else 'text/plain'
                )
                self.send_header('Cache-Control', 'no-store, max-age=0')
            else:
                body = b'not found'
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    consumer = threading.Thread(target=consume, daemon=True)
    resource_thread = threading.Thread(target=collect_resources, daemon=True)
    curve_thread = threading.Thread(
        target=update_curve_snapshot,
        name='daxigua-curve-snapshot',
        daemon=True,
    )
    consumer.start()
    resource_thread.start()
    if curve_snapshot_enabled:
        curve_thread.start()
    server = ThreadingHTTPServer((host, port), Handler)
    server.timeout = 0.5
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        stop_event.set()
        server.server_close()
        consumer.join(2.0)
        resource_thread.join(2.0)
        if curve_snapshot_enabled:
            curve_thread.join(10.0)
        sampler.close()


class DashboardPublisher:
    """训练侧非阻塞发布器；任何面板故障都降级为丢弃指标。"""

    def __init__(self, config, run_dir):
        self.enabled = bool(config.enabled)
        self.curve_snapshot_enabled = bool(
            config.curve_snapshot_enabled
        )
        self.process = None
        self.queue = None
        self.plot_done_event = None
        self._last_curve_snapshot_request = 0.0
        self.dropped_messages = 0
        if not self.enabled:
            return
        context = multiprocessing.get_context('spawn')
        self.queue = context.Queue(maxsize=128)
        self.plot_done_event = context.Event()
        self.process = context.Process(
            target=_dashboard_process_main,
            args=(
                self.queue,
                config.host,
                config.port,
                config.resource_interval_seconds,
                config.history_size,
                config.curve_snapshot_enabled,
                config.curve_snapshot_interval_seconds,
                self.plot_done_event,
                str(run_dir),
                os.getpid(),
            ),
            name='daxigua-training-dashboard',
            daemon=True,
        )
        self.process.start()

    def publish(self, payload, *, kind='training'):
        if not self.enabled or self.queue is None:
            return False
        message = _json_safe({'_kind': kind, **payload})
        try:
            self.queue.put_nowait(message)
            return True
        except (queue.Full, OSError, ValueError):
            self.dropped_messages += 1
            return False

    def event(self, event_kind, message, **values):
        return self.publish(
            {'kind': event_kind, 'message': message, **values}, kind='event'
        )

    def plot(self, name, metadata):
        return self.publish(
            {'name': str(name), **dict(metadata)}, kind='plot'
        )

    def snapshot_curves(self, *, wait=False, timeout=30.0):
        """请求旁路进程立即更新曲线；正式收尾时可等待原子落盘。"""

        if (
                not self.enabled
                or not self.curve_snapshot_enabled
                or self.queue is None
                or self.plot_done_event is None):
            return False
        self.plot_done_event.clear()
        message = {'_kind': 'render_curves'}
        try:
            if wait:
                self.queue.put(message, timeout=min(1.0, timeout))
            else:
                self.queue.put_nowait(message)
        except (queue.Full, OSError, ValueError):
            self.dropped_messages += 1
            return False
        self._last_curve_snapshot_request = time.monotonic()
        if not wait:
            return True
        return self.plot_done_event.wait(timeout)

    def close(self, timeout=5.0):
        if not self.enabled or self.queue is None:
            return
        if (
                self.curve_snapshot_enabled
                and time.monotonic() - self._last_curve_snapshot_request > 5.0):
            self.snapshot_curves(wait=True, timeout=30.0)
        try:
            self.queue.put_nowait(None)
        except (queue.Full, OSError, ValueError):
            pass
        if self.process is not None:
            self.process.join(timeout)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(1.0)


def _read_jsonl(path):
    rows = []
    try:
        with Path(path).open('r', encoding='utf-8') as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except OSError:
        pass
    return rows


def _completed_dashboard_snapshot(run_dir):
    run_dir = Path(run_dir)
    state = _DashboardState(3600)
    for row in _read_jsonl(run_dir / 'monitoring.jsonl'):
        kind = row.get('kind')
        payload = {
            key: value for key, value in row.items()
            if key not in ('kind', 'monitor_timestamp')
        }
        if kind == 'event':
            state.add_event(row)
        elif kind == 'plot':
            name = payload.pop('name', 'plot')
            state.update_plot(name, payload)
        else:
            state.update_training(payload)
    resources = _read_jsonl(run_dir / 'resources.jsonl')
    if resources:
        state.update_resources(resources[-1])
    for name, filename in (
            ('training_curves', 'training_curves.json'),
            ('evaluation_event_analysis', EVENT_ANALYSIS_FILENAME.replace(
                '.png', '.json'))):
        try:
            metadata = json.loads(
                (run_dir / 'plots' / filename).read_text(encoding='utf-8')
            )
        except (OSError, json.JSONDecodeError):
            continue
        state.update_plot(name, metadata)
    state.update_training({'phase': 'finished'})
    return state


def serve_completed_dashboard(run_dir, *, host='127.0.0.1', port=8765):
    """训练进程退出后继续提供只读面板和最终图片。"""

    run_dir = Path(run_dir).resolve()
    state = _completed_dashboard_snapshot(run_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            request_path = self.path.partition('?')[0]
            if request_path == '/api/status':
                snapshot = state.snapshot()
                snapshot['timestamp'] = time.time()
                body = json.dumps(snapshot, ensure_ascii=False).encode('utf-8')
                status = 200
                content_type = 'application/json; charset=utf-8'
            elif request_path in ('/', '/index.html'):
                body = _DASHBOARD_HTML.encode('utf-8')
                status = 200
                content_type = 'text/html; charset=utf-8'
            elif request_path in (
                    f'/plots/{CURVE_FILENAME}',
                    f'/plots/{EVENT_ANALYSIS_FILENAME}'):
                plot_path = run_dir / 'plots' / Path(request_path).name
                try:
                    valid = (
                        plot_path.is_file()
                        and not plot_path.is_symlink()
                        and plot_path.resolve(strict=True).parent
                        == (run_dir / 'plots').resolve(strict=True)
                    )
                    body = plot_path.read_bytes() if valid else b'not found'
                except OSError:
                    valid = False
                    body = b'not found'
                status = 200 if valid else 404
                content_type = 'image/png' if valid else 'text/plain'
            else:
                body = b'not found'
                status = 404
                content_type = 'text/plain'
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store, max-age=0')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer((host, int(port)), Handler)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
