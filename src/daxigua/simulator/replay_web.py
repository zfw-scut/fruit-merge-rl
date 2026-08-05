"""回放播放器和多局目录的无依赖 HTML 模板。"""

from __future__ import annotations

from hashlib import sha1
from html import escape


def _root_id(payload_data):
    digest = sha1(payload_data.encode('utf-8')).hexdigest()[:12]
    return f'daxigua-replay-{digest}'


def _render_player(
        title, payload_data, payload_encoding, textures_json, *, fragment):
    root_id = _root_id(payload_data)
    replacements = {
        '__ROOT_ID__': root_id,
        '__TITLE__': escape(title),
        '__TRACE_DATA__': payload_data,
        '__TRACE_ENCODING__': payload_encoding,
        '__TEXTURE_DATA__': textures_json,
    }
    body = _PLAYER_MARKUP + _PLAYER_SCRIPT
    css = _PLAYER_CSS
    for token, value in replacements.items():
        body = body.replace(token, value)
        css = css.replace(token, value)
    if fragment:
        return f'<style>{css}</style>\n{body}'
    document = _DOCUMENT_TEMPLATE.replace('__TITLE__', escape(title))
    document = document.replace('__PLAYER_CSS__', css)
    return document.replace('__PLAYER_BODY__', body)


def render_replay_document(
        *, title, payload_data, payload_encoding, textures_json):
    return _render_player(
        title, payload_data, payload_encoding, textures_json, fragment=False
    )


def render_replay_fragment(
        *, title, payload_data, payload_encoding, textures_json):
    return _render_player(
        title, payload_data, payload_encoding, textures_json, fragment=True
    )


def render_replay_catalog(*, title, description, entries_json):
    return (
        _CATALOG_TEMPLATE
        .replace('__TITLE__', escape(title))
        .replace('__DESCRIPTION__', escape(description))
        .replace('__ENTRIES__', entries_json)
    )


_DOCUMENT_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:#0b1017;color:#eef4f8}
body::before{content:"";position:fixed;inset:0;pointer-events:none;background:radial-gradient(circle at 18% 5%,#234d442e,transparent 34%),radial-gradient(circle at 92% 88%,#523c7428,transparent 34%)}
__PLAYER_CSS__
</style>
</head>
<body>
__PLAYER_BODY__
</body>
</html>'''


_PLAYER_CSS = r'''
#__ROOT_ID__{--dx-bg:#111923;--dx-panel:#161f2b;--dx-panel-2:#1d2936;--dx-border:#314153;--dx-text:#eef4f8;--dx-muted:#9eb0c0;--dx-accent:#7bd8bd;--dx-accent-2:#f2c66d;--dx-danger:#ff6b72;--dx-wall:#63788b;position:relative;display:grid;grid-template-columns:minmax(300px,560px) minmax(300px,1fr);gap:18px;max-width:1180px;margin:0 auto;padding:18px;color:var(--dx-text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
#__ROOT_ID__ *{box-sizing:border-box}
#__ROOT_ID__ .dxr-stage{position:sticky;top:18px;align-self:start;min-width:0}
#__ROOT_ID__ canvas{display:block;width:100%;height:auto;max-height:calc(100vh - 36px);background:#101820;border:1px solid var(--dx-border);border-radius:16px;box-shadow:0 18px 46px #0008}
#__ROOT_ID__ .dxr-panel{display:flex;flex-direction:column;gap:13px;min-width:0;padding:4px 0 28px}
#__ROOT_ID__ h1{font-size:21px;line-height:1.25;margin:0;letter-spacing:.01em}
#__ROOT_ID__ .dxr-lead{margin:4px 0 0;color:var(--dx-muted);font-size:13px;line-height:1.55}
#__ROOT_ID__ .dxr-box{padding:12px;border:1px solid var(--dx-border);border-radius:12px;background:color-mix(in srgb,var(--dx-panel) 94%,transparent)}
#__ROOT_ID__ .dxr-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}
#__ROOT_ID__ .dxr-field{display:flex;flex-direction:column;gap:6px;color:var(--dx-muted);font-size:12px}
#__ROOT_ID__ button,#__ROOT_ID__ select{min-height:36px;border:1px solid var(--dx-border);border-radius:9px;background:var(--dx-panel-2);color:var(--dx-text);font:inherit;padding:7px 10px}
#__ROOT_ID__ button{cursor:pointer;font-weight:620;transition:border-color .15s,background .15s,transform .1s}
#__ROOT_ID__ button:hover{border-color:var(--dx-accent);background:#253543}
#__ROOT_ID__ button:active{transform:translateY(1px)}
#__ROOT_ID__ .dxr-primary{background:#285e55;border-color:#3e8c7c}
#__ROOT_ID__ .dxr-row{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
#__ROOT_ID__ .dxr-row button{flex:1 1 auto}
#__ROOT_ID__ input[type=range]{width:100%;accent-color:var(--dx-accent)}
#__ROOT_ID__ .dxr-timeline-label{display:flex;justify-content:space-between;gap:12px;color:var(--dx-muted);font-size:12px;font-variant-numeric:tabular-nums}
#__ROOT_ID__ .dxr-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
#__ROOT_ID__ .dxr-stat{padding:9px 10px;border:1px solid var(--dx-border);border-radius:9px;background:#101821}
#__ROOT_ID__ .dxr-stat span{display:block;color:var(--dx-muted);font-size:11px;margin-bottom:3px}
#__ROOT_ID__ .dxr-stat strong{display:block;font-size:14px;font-variant-numeric:tabular-nums;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#__ROOT_ID__ .dxr-status{display:flex;align-items:center;gap:8px;min-height:34px;padding:8px 10px;border-radius:9px;background:#101821;color:var(--dx-muted);font-size:12px;line-height:1.4}
#__ROOT_ID__ .dxr-status i{width:9px;height:9px;flex:none;border-radius:50%;background:var(--dx-accent)}
#__ROOT_ID__ .dxr-status[data-kind=timeout] i{background:var(--dx-accent-2)}
#__ROOT_ID__ .dxr-status[data-kind=done] i,#__ROOT_ID__ .dxr-status[data-kind=truncated] i{background:var(--dx-danger)}
#__ROOT_ID__ .dxr-model[hidden]{display:none}
#__ROOT_ID__ .dxr-model-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:10px}
#__ROOT_ID__ .dxr-model-head strong{font-size:13px}
#__ROOT_ID__ .dxr-model-head span{color:var(--dx-muted);font-size:10px;text-align:right;overflow-wrap:anywhere}
#__ROOT_ID__ .dxr-model-summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;margin-bottom:10px}
#__ROOT_ID__ .dxr-model-value{padding:8px;border-radius:8px;background:#101821;border:1px solid #293848}
#__ROOT_ID__ .dxr-model-value span{display:block;color:var(--dx-muted);font-size:10px;margin-bottom:3px}
#__ROOT_ID__ .dxr-model-value strong{display:block;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#__ROOT_ID__ .dxr-queue{display:flex;gap:5px;flex-wrap:wrap;margin:0 0 10px}
#__ROOT_ID__ .dxr-queue span{padding:4px 7px;border-radius:999px;background:#21313e;color:var(--dx-muted);font-size:10px}
#__ROOT_ID__ .dxr-queue span:first-child{background:#285e55;color:#ecfff9}
#__ROOT_ID__ .dxr-q-title{display:flex;justify-content:space-between;color:var(--dx-muted);font-size:10px;margin-bottom:5px}
#__ROOT_ID__ .dxr-q-bars{height:92px;display:grid;grid-template-columns:repeat(21,minmax(5px,1fr));align-items:end;gap:3px;padding:6px 5px 4px;border-radius:8px;background:#101821;border:1px solid #293848}
#__ROOT_ID__ .dxr-q-action{position:relative;height:100%;display:flex;align-items:flex-end;justify-content:center}
#__ROOT_ID__ .dxr-q-action i{display:block;width:100%;min-height:4px;border-radius:3px 3px 1px 1px;background:#587083;transition:height .18s}
#__ROOT_ID__ .dxr-q-action.selected i{background:linear-gradient(180deg,#8be6ca,#3aa98a);box-shadow:0 0 0 1px #a5f2db55,0 0 12px #5cc5a566}
#__ROOT_ID__ .dxr-q-action b{position:absolute;bottom:-17px;font-size:8px;color:#738696;font-weight:500}
#__ROOT_ID__ .dxr-q-action.selected b{color:var(--dx-accent);font-weight:700}
#__ROOT_ID__ .dxr-toggles{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 12px}
#__ROOT_ID__ .dxr-toggle{display:flex;align-items:center;gap:7px;color:var(--dx-muted);font-size:12px;cursor:pointer}
#__ROOT_ID__ .dxr-toggle input{accent-color:var(--dx-accent)}
#__ROOT_ID__ details{font-size:12px;color:var(--dx-muted)}
#__ROOT_ID__ summary{cursor:pointer;color:var(--dx-text);font-weight:620}
#__ROOT_ID__ .dxr-help{margin-top:8px;line-height:1.65}
#__ROOT_ID__ kbd{padding:1px 5px;border:1px solid var(--dx-border);border-bottom-width:2px;border-radius:5px;background:#0e151d;color:var(--dx-text);font:11px ui-monospace,monospace}
#__ROOT_ID__ .dxr-legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 10px}
#__ROOT_ID__ .dxr-legend-item{display:flex;align-items:center;gap:8px;min-width:0;color:var(--dx-muted);font-size:12px}
#__ROOT_ID__ .dxr-legend-item img{width:28px;height:28px;object-fit:contain;flex:none}
#__ROOT_ID__ .dxr-legend-item span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@media(max-width:820px){#__ROOT_ID__{grid-template-columns:1fr;padding:12px}#__ROOT_ID__ .dxr-stage{position:relative;top:auto}#__ROOT_ID__ canvas{max-height:none;max-width:560px;margin:auto}#__ROOT_ID__ .dxr-panel{max-width:560px;margin:auto;width:100%}}
@media(max-width:430px){#__ROOT_ID__ .dxr-grid,#__ROOT_ID__ .dxr-stats,#__ROOT_ID__ .dxr-model-summary{grid-template-columns:1fr}#__ROOT_ID__ .dxr-toggles{grid-template-columns:1fr}}
'''


_PLAYER_MARKUP = r'''
<main id="__ROOT_ID__">
  <section class="dxr-stage">
    <canvas class="dxr-board" role="img" aria-label="水果物理模拟逐帧回放"></canvas>
  </section>
  <section class="dxr-panel">
    <header>
      <h1>__TITLE__</h1>
      <p class="dxr-lead">贴图按显示半径旋转，碰撞圈按真实物理半径绘制；两者不会混用。</p>
    </header>
    <div class="dxr-box dxr-grid">
      <label class="dxr-field">环境样本<select class="dxr-clip"></select></label>
      <label class="dxr-field">跳到投放<select class="dxr-drop"></select></label>
      <label class="dxr-field">播放模式
        <select class="dxr-mode">
          <option value="review" selected>智能浏览（压缩长等待）</option>
          <option value="physics">真实物理时间</option>
          <option value="drops">逐次投放总览</option>
        </select>
      </label>
      <label class="dxr-field">播放速度
        <select class="dxr-speed">
          <option value="0.25">0.25×</option><option value="0.5">0.5×</option>
          <option value="1" selected>1×</option><option value="2">2×</option>
          <option value="4">4×</option><option value="8">8×</option>
          <option value="16">16×</option>
        </select>
      </label>
    </div>
    <div class="dxr-row">
      <button type="button" class="dxr-play dxr-primary">播放</button>
      <button type="button" class="dxr-prev-frame" title="上一采样帧">−1 帧</button>
      <button type="button" class="dxr-next-frame" title="下一采样帧">+1 帧</button>
    </div>
    <div class="dxr-row">
      <button type="button" class="dxr-prev-drop">上一投放</button>
      <button type="button" class="dxr-next-drop">下一投放</button>
      <button type="button" class="dxr-prev-merge">上一合成</button>
      <button type="button" class="dxr-next-merge">下一合成</button>
      <button type="button" class="dxr-prev-timeout">上一超时</button>
      <button type="button" class="dxr-next-timeout">下一超时</button>
    </div>
    <div>
      <div class="dxr-timeline-label"><span>物理时间轴</span><span class="dxr-position"></span></div>
      <input class="dxr-timeline" type="range" min="0" max="0" value="0" aria-label="回放时间轴">
    </div>
    <div class="dxr-stats">
      <div class="dxr-stat"><span>投放 / 动作</span><strong class="dxr-drop-stat"></strong></div>
      <div class="dxr-stat"><span>本局时间</span><strong class="dxr-time-stat"></strong></div>
      <div class="dxr-stat"><span>水果数 / 分数</span><strong class="dxr-score-stat"></strong></div>
      <div class="dxr-stat"><span>当前投放累计合成</span><strong class="dxr-merge-stat"></strong></div>
    </div>
    <section class="dxr-box dxr-model" hidden>
      <div class="dxr-model-head"><strong>GNN-DQN 模型决策</strong><span class="dxr-model-checkpoint"></span></div>
      <div class="dxr-model-summary">
        <div class="dxr-model-value"><span>当前水果 / 选定动作</span><strong class="dxr-model-action"></strong></div>
        <div class="dxr-model-value"><span>真实投放位置</span><strong class="dxr-model-position"></strong></div>
        <div class="dxr-model-value"><span>选定 Q / Q 均值</span><strong class="dxr-model-q"></strong></div>
        <div class="dxr-model-value"><span>危险进度 / 推理耗时</span><strong class="dxr-model-danger"></strong></div>
      </div>
      <div class="dxr-queue" aria-label="未来水果队列"></div>
      <div class="dxr-q-title"><span>21 个动作 Q 值</span><span class="dxr-q-range"></span></div>
      <div class="dxr-q-bars" aria-label="动作 Q 值分布"></div>
    </section>
    <div class="dxr-status" aria-live="polite"><i></i><span></span></div>
    <div class="dxr-box dxr-toggles">
      <label class="dxr-toggle"><input class="dxr-show-textures" type="checkbox" checked>水果贴图</label>
      <label class="dxr-toggle"><input class="dxr-show-collisions" type="checkbox" checked>碰撞半径与角度</label>
      <label class="dxr-toggle"><input class="dxr-show-velocity" type="checkbox">线速度向量</label>
      <label class="dxr-toggle"><input class="dxr-show-ids" type="checkbox">水果 ID / 等级</label>
    </div>
    <details class="dxr-box">
      <summary>快捷键与显示说明</summary>
      <div class="dxr-help"><kbd>Space</kbd> 播放/暂停；<kbd>←</kbd>/<kbd>→</kbd> 单帧；<kbd>Shift</kbd> + 方向键切换投放；<kbd>M</kbd>/<kbd>Shift+M</kbd> 跳到下/上一次合成；<kbd>T</kbd>/<kbd>Shift+T</kbd> 跳到下/上一次等待超时；<kbd>Home</kbd>/<kbd>End</kbd> 跳到首尾。智能浏览只压缩观看等待，不改变记录中的物理帧、速度或结果。</div>
    </details>
    <details class="dxr-box">
      <summary>水果等级</summary>
      <div class="dxr-legend"></div>
    </details>
  </section>
</main>
'''


_PLAYER_SCRIPT = r'''
<script>
;(async()=>{
'use strict';
const root=document.getElementById('__ROOT_ID__');
async function decodePayload(encoding,value){
 if(encoding==='json')return value;
 if(encoding!=='gzip-base64')throw new Error(`不支持的回放编码：${encoding}`);
 if(typeof DecompressionStream==='undefined')throw new Error('当前浏览器不支持 DecompressionStream，请重新生成未压缩回放或更换现代浏览器');
 const bytes=Uint8Array.from(atob(value),character=>character.charCodeAt(0));
 const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
 return JSON.parse(await new Response(stream).text());
}
let replay;
try{replay=await decodePayload('__TRACE_ENCODING__',__TRACE_DATA__)}catch(error){root.innerHTML=`<section class="dxr-panel"><h1>回放载入失败</h1><p class="dxr-lead"></p></section>`;root.querySelector('p').textContent=String(error);throw error}
const textureSources=__TEXTURE_DATA__;
const q=selector=>root.querySelector(selector);
const board=q('.dxr-board'),ctx=board.getContext('2d');
const clipSelect=q('.dxr-clip'),dropSelect=q('.dxr-drop'),timeline=q('.dxr-timeline');
const playButton=q('.dxr-play'),speedSelect=q('.dxr-speed'),modeSelect=q('.dxr-mode');
const textures=textureSources.map((source,level)=>{if(!source)return null;const image=new Image();image.decoding='async';image.src=source;image.addEventListener('load',()=>draw());return image});
const fallback=['','#b74b95','#ed5b70','#7650ad','#f19c37','#ef7447','#ef5350','#d5b34c','#ef9d91','#e8a63d','#9fd45a','#45b933'];
let clipIndex=0,recordIndex=0,playing=false,lastTime=0,accumulator=0;
let renderedDecisionKey='';
board.width=replay.board.width;board.height=replay.board.height;ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';

if(replay.model_viewer){const viewer=replay.model_viewer,episode=viewer.episode||{};q('.dxr-lead').textContent=`模型使用 ${episode.physics_fps||replay.physics_fps} FPS 真实物理 greedy 游玩；页面保存了每次投放的完整 Q 值与逐帧运动。`}

function currentClip(){return replay.clips[clipIndex]}
function decode(raw){
 if(!replay.compact_records)return raw;
 return {frame:raw[0],local_frame:raw[1],drop:raw[2],score:raw[3],merges:raw[4],fruits:raw[5]};
}
function currentRecord(){return decode(currentClip().records[recordIndex])}
function summaryFor(record){return currentClip().drop_summaries[record.drop-1]}
function prepareClip(){
 const clip=currentClip();clip._bounds=Array.from({length:clip.drops},()=>({start:-1,end:-1}));
 clip.records.forEach((raw,index)=>{const drop=decode(raw).drop-1,b=clip._bounds[drop];if(b.start<0)b.start=index;b.end=index});
 dropSelect.replaceChildren();clip.drop_summaries.forEach((summary,index)=>{const option=document.createElement('option');option.value=index+1;const suffix=summary.done?' · 失败':summary.truncated?' · 截断':summary.settle_timeout?' · 等待上限':summary.stable?' · 稳定':'';option.textContent=`第 ${index+1} 次 · 动作 ${summary.action} · +${summary.score_delta} 分${suffix}`;dropSelect.append(option)});
 timeline.max=Math.max(0,clip.records.length-1);recordIndex=Math.min(recordIndex,clip.records.length-1);
}
function statusText(summary){
 if(summary.done)return ['done','本次投放触发游戏失败'];
 if(summary.truncated)return ['truncated','本次投放因技术边界截断，需要重置'];
 if(summary.settle_timeout)return ['timeout',`等待达到 ${summary.frames} 帧上限，运动状态交给下一次投放继续模拟`];
 if(summary.stable)return ['stable',`连续稳定后结束等待，共模拟 ${summary.frames} 帧`];
 return ['active',`本次投放记录 ${summary.frames} 个物理帧`];
}
function drawArrow(x,y,vx,vy,r){
 const speed=Math.hypot(vx,vy);if(speed<.01)return;const length=Math.min(r*2.5,Math.max(5,speed*.08)),ux=vx/speed,uy=vy/speed,tx=x+ux*length,ty=y+uy*length;
 ctx.strokeStyle='#61d7ff';ctx.fillStyle='#61d7ff';ctx.lineWidth=1.8;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(tx,ty);ctx.stroke();ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-ux*7-uy*4,ty-uy*7+ux*4);ctx.lineTo(tx-ux*7+uy*4,ty-uy*7-ux*4);ctx.closePath();ctx.fill();
}
function drawFruit(fruit){
 const[id,level,x,y,physicsRadius,angle,vx,vy]=fruit;const displayRadius=replay.fruit_display_radii[level-1]||physicsRadius;
 ctx.save();ctx.translate(x,y);ctx.rotate(angle);
 if(q('.dxr-show-textures').checked&&textures[level]&&textures[level].complete){ctx.drawImage(textures[level],-displayRadius,-displayRadius,displayRadius*2,displayRadius*2)}else{ctx.fillStyle=fallback[level]||'#aaa';ctx.beginPath();ctx.arc(0,0,displayRadius,0,Math.PI*2);ctx.fill()}
 ctx.restore();
 if(q('.dxr-show-collisions').checked){ctx.strokeStyle='#f1f7fbcc';ctx.lineWidth=1.35;ctx.beginPath();ctx.arc(x,y,physicsRadius,0,Math.PI*2);ctx.stroke();ctx.strokeStyle='#12202dcc';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+Math.cos(angle)*physicsRadius*.72,y+Math.sin(angle)*physicsRadius*.72);ctx.stroke()}
 if(q('.dxr-show-velocity').checked)drawArrow(x,y,vx,vy,physicsRadius);
 if(q('.dxr-show-ids').checked){ctx.font=`600 ${Math.max(11,Math.min(18,physicsRadius*.34))}px ui-monospace,monospace`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.lineWidth=3;ctx.strokeStyle='#0b111bcc';ctx.strokeText(`${level}·${id}`,x,y);ctx.fillStyle='#fff';ctx.fillText(`${level}·${id}`,x,y)}
}
function renderDecision(summary){
 const panel=q('.dxr-model'),decision=summary.decision,viewer=replay.model_viewer||{};
 if(!decision){panel.hidden=true;renderedDecisionKey='';return}
 panel.hidden=false;const key=`${clipIndex}:${decision.drop}`;if(key===renderedDecisionKey)return;renderedDecisionKey=key;
 q('.dxr-model-checkpoint').textContent=`${viewer.checkpoint||'checkpoint'} · ${(viewer.checkpoint_sha256||'').slice(0,12)}`;
 q('.dxr-model-action').textContent=`${decision.current_fruit||`等级 ${decision.current_level}`} · A${decision.action}`;
 q('.dxr-model-position').textContent=`x = ${Number(decision.drop_x).toFixed(1)} px`;
 q('.dxr-model-q').textContent=`${Number(decision.selected_q).toFixed(4)} / ${Number(decision.q_mean).toFixed(4)}`;
 q('.dxr-model-danger').textContent=`${(Number(decision.danger_progress)*100).toFixed(1)}% / ${Number(decision.inference_ms).toFixed(2)} ms`;
 const queue=q('.dxr-queue');queue.replaceChildren(...decision.queue.map((level,index)=>{const item=document.createElement('span');item.textContent=`q${index} · ${replay.fruit_names[level-1]||`等级 ${level}`}`;return item}));
 const values=decision.q_values.map(Number),minimum=Math.min(...values),maximum=Math.max(...values),span=Math.max(1e-9,maximum-minimum),bars=q('.dxr-q-bars');
 bars.replaceChildren(...values.map((value,index)=>{const item=document.createElement('span');item.className='dxr-q-action'+(index===decision.action?' selected':'');item.title=`动作 A${index} · Q=${value.toFixed(6)}`;const bar=document.createElement('i');bar.style.height=`${10+(value-minimum)/span*90}%`;const label=document.createElement('b');label.textContent=index;item.append(bar,label);return item}));
 q('.dxr-q-range').textContent=`${minimum.toFixed(3)} ～ ${maximum.toFixed(3)}`;
}
function draw(){
 const clip=currentClip(),record=currentRecord(),summary=summaryFor(record),b=replay.board;
 const gradient=ctx.createLinearGradient(0,0,0,b.height);gradient.addColorStop(0,'#182735');gradient.addColorStop(1,'#0e171e');ctx.fillStyle=gradient;ctx.fillRect(0,0,b.width,b.height);
 ctx.fillStyle='#ff6b7214';ctx.fillRect(b.wall_width,0,b.width-b.wall_width*2,b.spawn_y);
 ctx.strokeStyle='#647a8c';ctx.lineWidth=b.wall_width;ctx.lineJoin='round';ctx.beginPath();ctx.moveTo(b.wall_width/2,0);ctx.lineTo(b.wall_width/2,b.height-b.wall_width/2);ctx.lineTo(b.width-b.wall_width/2,b.height-b.wall_width/2);ctx.lineTo(b.width-b.wall_width/2,0);ctx.stroke();
 ctx.strokeStyle='#ff6b72';ctx.lineWidth=2;ctx.setLineDash([10,8]);ctx.beginPath();ctx.moveTo(b.wall_width,b.spawn_y);ctx.lineTo(b.width-b.wall_width,b.spawn_y);ctx.stroke();ctx.setLineDash([]);
 record.fruits.forEach(drawFruit);
 timeline.value=recordIndex;dropSelect.value=String(record.drop);
 q('.dxr-position').textContent=`采样 ${recordIndex+1} / ${clip.records.length}`;
 q('.dxr-drop-stat').textContent=`${record.drop} / ${clip.drops} · #${summary.action}`;
 q('.dxr-time-stat').textContent=`${(record.frame/replay.physics_fps).toFixed(2)} 秒 · 局部 ${record.local_frame} 帧`;
 q('.dxr-score-stat').textContent=`${record.fruits.length} 个 · ${record.score} 分`;
 q('.dxr-merge-stat').textContent=`${record.merges} 次 · 本次 +${summary.score_delta} 分`;
 renderDecision(summary);
 const[kind,text]=statusText(summary),status=q('.dxr-status');status.dataset.kind=kind;status.querySelector('span').textContent=(summary.reset_before?'已在本次投放前重置；':'')+text;
}
function setRecord(value,{stopPlayback=true}={}){if(stopPlayback)stop();recordIndex=Math.max(0,Math.min(Number(value),currentClip().records.length-1));draw()}
function stop(){playing=false;playButton.textContent='播放'}
function togglePlay(){if(!playing&&recordIndex+1>=currentClip().records.length)recordIndex=0;playing=!playing;playButton.textContent=playing?'暂停':'播放';lastTime=performance.now();accumulator=0;if(playing)requestAnimationFrame(tick)}
function jumpDrop(delta){const record=currentRecord(),target=Math.max(1,Math.min(currentClip().drops,record.drop+delta)),bound=currentClip()._bounds[target-1];setRecord(bound.start)}
function hasMerge(index){if(index<=0)return false;return decode(currentClip().records[index]).score!==decode(currentClip().records[index-1]).score}
function jumpMerge(direction){let index=recordIndex+direction;while(index>=0&&index<currentClip().records.length){if(hasMerge(index)){setRecord(index);return}index+=direction}}
function jumpTimeout(direction){let drop=currentRecord().drop-1+direction;while(drop>=0&&drop<currentClip().drops){if(currentClip().drop_summaries[drop].settle_timeout){setRecord(currentClip()._bounds[drop].start);return}drop+=direction}}
function intervalMs(){const speed=Number(speedSelect.value),mode=modeSelect.value;if(mode==='drops')return 520/speed;const physical=1000*replay.frame_stride/replay.physics_fps/speed;if(mode==='physics')return physical;const record=currentRecord(),bounds=currentClip()._bounds[record.drop-1],count=Math.max(1,bounds.end-bounds.start+1);return Math.min(physical,1500/count/speed)}
function advance(){if(modeSelect.value==='drops'){const record=currentRecord();if(record.drop>=currentClip().drops)return false;recordIndex=currentClip()._bounds[record.drop].end;return true}if(recordIndex+1>=currentClip().records.length)return false;recordIndex++;return true}
function tick(now){if(!playing)return;accumulator+=now-lastTime;lastTime=now;let interval=intervalMs(),changed=false;while(accumulator>=interval){accumulator-=interval;if(!advance()){stop();draw();return}changed=true;interval=intervalMs();if(modeSelect.value==='drops')break}if(changed)draw();requestAnimationFrame(tick)}

replay.clips.forEach((clip,index)=>{const option=document.createElement('option');option.value=index;option.textContent=`环境 ${clip.env} · ${clip.drops} 次投放 · ${(clip.total_frames/replay.physics_fps).toFixed(1)} 秒`;clipSelect.append(option)});
q('.dxr-legend').replaceChildren(...replay.fruit_names.map((name,index)=>{const item=document.createElement('div');item.className='dxr-legend-item';const img=document.createElement('img');if(textureSources[index+1])img.src=textureSources[index+1];img.alt='';const text=document.createElement('span');text.textContent=`${index+1} · ${name}`;item.append(img,text);return item}));
prepareClip();draw();
clipSelect.addEventListener('change',()=>{clipIndex=Number(clipSelect.value);recordIndex=0;stop();prepareClip();draw()});
dropSelect.addEventListener('change',()=>{const bound=currentClip()._bounds[Number(dropSelect.value)-1];setRecord(bound.start)});
timeline.addEventListener('input',()=>setRecord(timeline.value));
playButton.addEventListener('click',togglePlay);
q('.dxr-prev-frame').addEventListener('click',()=>setRecord(recordIndex-1));q('.dxr-next-frame').addEventListener('click',()=>setRecord(recordIndex+1));
q('.dxr-prev-drop').addEventListener('click',()=>jumpDrop(-1));q('.dxr-next-drop').addEventListener('click',()=>jumpDrop(1));
q('.dxr-prev-merge').addEventListener('click',()=>jumpMerge(-1));q('.dxr-next-merge').addEventListener('click',()=>jumpMerge(1));
q('.dxr-prev-timeout').addEventListener('click',()=>jumpTimeout(-1));q('.dxr-next-timeout').addEventListener('click',()=>jumpTimeout(1));
root.querySelectorAll('.dxr-toggles input').forEach(input=>input.addEventListener('change',draw));
root.addEventListener('keydown',event=>event.stopPropagation());
document.addEventListener('keydown',event=>{if(!root.isConnected||['INPUT','SELECT','BUTTON'].includes(document.activeElement?.tagName))return;if(event.code==='Space'){event.preventDefault();togglePlay()}else if(event.key==='ArrowLeft'){event.preventDefault();event.shiftKey?jumpDrop(-1):setRecord(recordIndex-1)}else if(event.key==='ArrowRight'){event.preventDefault();event.shiftKey?jumpDrop(1):setRecord(recordIndex+1)}else if(event.key==='Home'){event.preventDefault();setRecord(0)}else if(event.key==='End'){event.preventDefault();setRecord(currentClip().records.length-1)}else if(event.key.toLowerCase()==='m'){event.preventDefault();jumpMerge(event.shiftKey?-1:1)}else if(event.key.toLowerCase()==='t'){event.preventDefault();jumpTimeout(event.shiftKey?-1:1)}});
})();
</script>
'''


_CATALOG_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}*{box-sizing:border-box}body{margin:0;background:#0b1017;color:#edf4f8}main{display:grid;grid-template-columns:340px minmax(0,1fr);min-height:100vh}.sidebar{padding:18px;border-right:1px solid #2d3d4c;background:#111923;display:flex;flex-direction:column;gap:12px;max-height:100vh;position:sticky;top:0}.sidebar h1{font-size:20px;margin:0}.sidebar p{font-size:12px;line-height:1.55;color:#9eb0c0;margin:0}.controls{display:grid;grid-template-columns:1fr 1fr;gap:8px}input,select,button,a.direct{min-height:36px;border:1px solid #314153;border-radius:8px;background:#1b2834;color:inherit;font:inherit;padding:7px 9px}input{grid-column:1/-1}button{cursor:pointer}.list{display:flex;flex-direction:column;gap:7px;overflow:auto;padding-right:3px}.entry{text-align:left;padding:10px;border:1px solid #2d3d4c;border-radius:10px;background:#151f29;color:inherit;cursor:pointer}.entry:hover,.entry.active{border-color:#68cdb2;background:#1c302f}.entry strong{display:block;font-size:13px;margin-bottom:4px}.entry span{display:block;color:#9eb0c0;font-size:11px;line-height:1.45}.empty{color:#9eb0c0;font-size:13px;padding:12px}.viewer{position:relative;min-width:0;background:#090d13}.viewer iframe{display:block;width:100%;height:100vh;border:0}.viewerbar{position:fixed;right:16px;top:12px;z-index:2;display:flex;gap:7px}.viewerbar button,.viewerbar a{box-shadow:0 5px 18px #0008;text-decoration:none}.counter{color:#9eb0c0;font-size:11px;text-align:center}.kind{color:#f2c66d}@media(max-width:780px){main{grid-template-columns:1fr}.sidebar{position:relative;max-height:none;border-right:0;border-bottom:1px solid #2d3d4c}.list{max-height:310px}.viewer iframe{height:86vh}.viewerbar{position:absolute}}
</style></head><body>
<main><aside class="sidebar"><h1>__TITLE__</h1><p>__DESCRIPTION__</p><div class="controls"><input class="search" type="search" placeholder="筛选环境编号或结束状态"><select class="sort"><option value="original">原抽样顺序</option><option value="timeout-rate-desc">超时比例：高到低</option><option value="timeout-count-desc">超时次数：多到少</option><option value="drops-desc">投放次数：多到少</option><option value="score-desc">分数：高到低</option><option value="env">环境编号</option></select><button class="previous" type="button">上一局</button><button class="next" type="button">下一局</button></div><div class="counter"></div><div class="list"></div></aside><section class="viewer"><div class="viewerbar"><a class="direct" target="_blank" rel="noopener">单独打开</a></div><iframe title="选中的完整局回放" sandbox="allow-scripts allow-same-origin"></iframe></section></main>
<script>
(()=>{'use strict';const entries=__ENTRIES__,list=document.querySelector('.list'),frame=document.querySelector('iframe'),direct=document.querySelector('.direct'),search=document.querySelector('.search'),sort=document.querySelector('.sort'),counter=document.querySelector('.counter');let visible=[],selected=-1;
const labels={terminated:'失败',truncated:'技术截断',capped:'投放上限',unknown:'未知'};
 function rebuild(){const query=search.value.trim().toLowerCase();visible=entries.map((entry,index)=>({...entry,_index:index})).filter(entry=>!query||String(entry.env_index).includes(query)||(labels[entry.end_kind]||entry.end_kind).toLowerCase().includes(query));if(sort.value==='timeout-rate-desc')visible.sort((a,b)=>b.settle_timeout_rate-a.settle_timeout_rate);else if(sort.value==='timeout-count-desc')visible.sort((a,b)=>b.settle_timeout_count-a.settle_timeout_count);else if(sort.value==='drops-desc')visible.sort((a,b)=>b.step_count-a.step_count);else if(sort.value==='score-desc')visible.sort((a,b)=>b.score-a.score);else if(sort.value==='env')visible.sort((a,b)=>Number(a.env_index)-Number(b.env_index));renderList();const current=visible.findIndex(entry=>entry._index===selected);if(current<0)selectVisible(0);else if(!frame.getAttribute('src'))selectVisible(current)}
 function renderList(){list.replaceChildren();visible.forEach((entry,index)=>{const button=document.createElement('button');button.type='button';button.className='entry'+(entry._index===selected?' active':'');const title=document.createElement('strong');title.textContent=`环境 ${entry.env_index} · ${labels[entry.end_kind]||entry.end_kind}`;const detail=document.createElement('span');detail.textContent=`${entry.step_count} 次投放 · ${entry.score} 分 · 超时 ${entry.settle_timeout_count} 次 (${(entry.settle_timeout_rate*100).toFixed(1)}%) · ${entry.physics_frames.toLocaleString()} 物理帧`;button.append(title,detail);button.addEventListener('click',()=>selectVisible(index));list.append(button)});counter.textContent=`显示 ${visible.length} / ${entries.length} 局`}
function selectVisible(index){if(!visible.length){selected=-1;frame.removeAttribute('src');direct.removeAttribute('href');list.innerHTML='<div class="empty">没有匹配的回放</div>';return}index=Math.max(0,Math.min(index,visible.length-1));const entry=visible[index];selected=entry._index;frame.src=entry.href;direct.href=entry.href;history.replaceState(null,'',`#env-${encodeURIComponent(entry.env_index)}`);renderList();list.children[index]?.scrollIntoView({block:'nearest'})}
function move(delta){const index=visible.findIndex(entry=>entry._index===selected);selectVisible(index+delta)}
search.addEventListener('input',rebuild);sort.addEventListener('change',rebuild);document.querySelector('.previous').addEventListener('click',()=>move(-1));document.querySelector('.next').addEventListener('click',()=>move(1));
const hash=decodeURIComponent(location.hash.replace(/^#env-/,''));const initial=entries.findIndex(entry=>String(entry.env_index)===hash);selected=initial>=0?initial:(entries[0]?0:-1);rebuild();if(initial>=0){const visibleIndex=visible.findIndex(entry=>entry._index===initial);selectVisible(visibleIndex)}
})();
</script></body></html>'''
