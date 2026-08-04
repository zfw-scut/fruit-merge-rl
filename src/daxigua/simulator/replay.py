"""把 CUDA 逐帧追踪导出为可独立打开的轻量回放页。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from daxigua.core import fruit_name

from .config import SimulatorConfig
from .types import BatchSimulationTrace


def trace_to_payload(trace, config=None):
    """把一次或连续多次投放追踪压缩为浏览器 JSON 数据。"""

    if isinstance(trace, BatchSimulationTrace):
        traces = (trace,)
    else:
        try:
            traces = tuple(trace)
        except TypeError as error:
            raise TypeError(
                'trace must be BatchSimulationTrace or a sequence of traces'
            ) from error
        if not traces or not all(
                isinstance(item, BatchSimulationTrace) for item in traces):
            raise TypeError(
                'trace sequence must contain BatchSimulationTrace values'
            )
    config = config or SimulatorConfig()
    if not isinstance(config, SimulatorConfig):
        raise TypeError('config must be SimulatorConfig')
    traces = tuple(item.cpu() for item in traces)
    first = traces[0]
    env_indices = first.env_indices.tolist()
    for item in traces[1:]:
        if item.env_indices.tolist() != env_indices:
            raise ValueError('all traces must contain the same environments')
        if (
                item.physics_fps != first.physics_fps
                or item.frame_stride != first.frame_stride):
            raise ValueError('all traces must use the same timing configuration')

    clips = [
        {
            'env': int(env_index),
            'drops': len(traces),
            'total_frames': 0,
            'records': [],
            'drop_summaries': [],
        }
        for env_index in env_indices
    ]
    frame_offsets = [0 for _ in clips]
    previous_finished = [False for _ in clips]
    for drop_index, item in enumerate(traces):
        for row, clip in enumerate(clips):
            record_count = int(item.record_counts[row])
            action = int(item.actions[row])
            final_local_frame = int(
                item.frame_numbers[row, record_count - 1]
            )
            clip['drop_summaries'].append({
                'drop': drop_index + 1,
                'action': action,
                'frames': final_local_frame,
                'score_delta': int(item.score_deltas[row]),
                'stable': bool(item.stable[row]),
                'done': bool(item.done[row]),
                'truncated': bool(item.truncated[row]),
                'settle_timeout': (
                    False
                    if item.settle_timeout is None
                    else bool(item.settle_timeout[row])
                ),
                'reset_before': previous_finished[row],
            })
            for record_index in range(record_count):
                active_slots = torch.nonzero(
                    item.active[row, record_index], as_tuple=False
                ).flatten()
                fruits = []
                for slot in active_slots.tolist():
                    fruits.append([
                        int(item.fruit_ids[row, record_index, slot]),
                        int(item.levels[row, record_index, slot]),
                        round(float(item.positions[row, record_index, slot, 0]), 3),
                        round(float(item.positions[row, record_index, slot, 1]), 3),
                        round(float(item.physics_radii[row, record_index, slot]), 3),
                        round(float(item.angles[row, record_index, slot]), 4),
                        round(float(item.velocities[row, record_index, slot, 0]), 3),
                        round(float(item.velocities[row, record_index, slot, 1]), 3),
                        round(
                            float(item.angular_velocities[
                                row, record_index, slot
                            ]),
                            4,
                        ),
                    ])
                local_frame = int(item.frame_numbers[row, record_index])
                clip['records'].append({
                    'frame': frame_offsets[row] + local_frame,
                    'local_frame': local_frame,
                    'drop': drop_index + 1,
                    'action': action,
                    'drop_start': record_index == 0,
                    'reset': record_index == 0 and previous_finished[row],
                    'score': int(item.scores[row, record_index]),
                    'merges': int(item.merge_counts[row, record_index]),
                    'fruits': fruits,
                })
            frame_offsets[row] += final_local_frame
            clip['total_frames'] = frame_offsets[row]
            previous_finished[row] = bool(
                item.done[row] or item.truncated[row]
            )
    return {
        'board': {
            'width': config.board_width,
            'height': config.board_height,
            'spawn_y': config.spawn_y,
            'wall_width': config.wall_width,
        },
        'physics_fps': first.physics_fps,
        'frame_stride': first.frame_stride,
        'fruit_names': [fruit_name(level) for level in range(1, 12)],
        'clips': clips,
    }


def write_replay_html(path, trace, config=None, *, title='CUDA physics replay'):
    """生成不依赖服务端或外部资源的回放 HTML。"""

    payload = trace_to_payload(trace, config)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    safe_title = str(title).replace('&', '&amp;').replace('<', '&lt;')
    html = _HTML_TEMPLATE.replace('__TITLE__', safe_title).replace(
        '__TRACE_DATA__', data.replace('</', '<\\/')
    )
    output_path.write_text(html, encoding='utf-8')
    return output_path


def write_replay_fragment(path, trace, config=None):
    """生成适合嵌入 Codex 对话的主题自适应回放片段。"""

    payload = trace_to_payload(trace, config)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    fragment = _FRAGMENT_TEMPLATE.replace(
        '__TRACE_DATA__', data.replace('</', '<\\/')
    )
    output_path.write_text(fragment, encoding='utf-8')
    return output_path


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
body{margin:0;background:#101318;color:#edf2f7}
main{max-width:1120px;margin:auto;padding:20px;display:grid;grid-template-columns:minmax(280px,560px) minmax(250px,1fr);gap:20px}
canvas{display:block;width:100%;height:auto;background:#171b22;border:1px solid #48515e;border-radius:10px}
.controls{display:flex;flex-direction:column;gap:14px}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button,select,input{font:inherit}button,select{padding:7px 11px;border:1px solid #596575;border-radius:7px;background:#242a33;color:inherit}
button:hover{background:#303844}input[type=range]{width:100%}.stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.stat{padding:10px;border:1px solid #3b4552;border-radius:8px;background:#1a1f27}.label{color:#9da9b7;font-size:12px}.value{font-variant-numeric:tabular-nums;margin-top:3px}
.legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;font-size:13px}.dot{display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:6px;vertical-align:-1px}
h1{font-size:20px;margin:0 0 2px}p{margin:0;color:#aeb8c5;font-size:13px;line-height:1.5}
@media(max-width:760px){main{grid-template-columns:1fr;padding:12px}.controls{order:-1}}
</style>
</head>
<body>
<main>
<canvas id="board" aria-label="水果物理模拟回放"></canvas>
<section class="controls">
<h1>__TITLE__</h1>
<p>圆表示真实碰撞半径；短线表示角度。红色横线是危险线。</p>
<label>样本 <select id="clip"></select></label>
<div class="row">
<button id="play" type="button">播放</button>
<button id="previous" type="button">上一帧</button>
<button id="next" type="button">下一帧</button>
<label>速度 <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
</div>
<input id="timeline" type="range" min="0" max="0" value="0" aria-label="回放时间轴">
<div class="stats">
<div class="stat"><div class="label">投放 / 物理时间</div><div class="value" id="frame"></div></div>
<div class="stat"><div class="label">水果数</div><div class="value" id="count"></div></div>
<div class="stat"><div class="label">分数</div><div class="value" id="score"></div></div>
<div class="stat"><div class="label">本次合并</div><div class="value" id="merges"></div></div>
</div>
<div class="legend" id="legend"></div>
</section>
</main>
<script>
const replay=__TRACE_DATA__;
const colors=['','#f3c969','#f4a261','#e76f51','#de5b8a','#ad68c9','#7a72d8','#5d8bd8','#48a9a6','#52b788','#8fbc55','#4eaa67'];
const names=['',...replay.fruit_names];
const board=document.getElementById('board'),ctx=board.getContext('2d');
const clipSelect=document.getElementById('clip'),timeline=document.getElementById('timeline');
const playButton=document.getElementById('play'),speedSelect=document.getElementById('speed');
let clipIndex=0,recordIndex=0,playing=false,lastTime=0,accumulator=0;
board.width=replay.board.width;board.height=replay.board.height;
replay.clips.forEach((clip,index)=>{const option=document.createElement('option');option.value=index;option.textContent=`环境 ${clip.env} · ${clip.drops} 次投放 · ${clip.total_frames} 帧`;clipSelect.append(option)});
document.getElementById('legend').innerHTML=names.slice(1).map((name,index)=>`<span><i class="dot" style="background:${colors[index+1]}"></i>${index+1} ${name}</span>`).join('');
function currentClip(){return replay.clips[clipIndex]}
function draw(){
 const clip=currentClip(),record=clip.records[recordIndex],b=replay.board;
 ctx.clearRect(0,0,board.width,board.height);ctx.fillStyle='#171b22';ctx.fillRect(0,0,board.width,board.height);
 ctx.strokeStyle='#657080';ctx.lineWidth=b.wall_width;ctx.beginPath();ctx.moveTo(b.wall_width/2,0);ctx.lineTo(b.wall_width/2,b.height-b.wall_width/2);ctx.lineTo(b.width-b.wall_width/2,b.height-b.wall_width/2);ctx.lineTo(b.width-b.wall_width/2,0);ctx.stroke();
 ctx.strokeStyle='#df5b61';ctx.lineWidth=2;ctx.setLineDash([10,8]);ctx.beginPath();ctx.moveTo(b.wall_width,b.spawn_y);ctx.lineTo(b.width-b.wall_width,b.spawn_y);ctx.stroke();ctx.setLineDash([]);
 record.fruits.forEach(f=>{const[id,level,x,y,r,angle]=f;ctx.fillStyle=colors[level]||'#aaa';ctx.strokeStyle='#edf2f7';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.strokeStyle='#24303c';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+Math.cos(angle)*r*.72,y+Math.sin(angle)*r*.72);ctx.stroke();ctx.fillStyle='#101318';ctx.font=`500 ${Math.max(13,Math.min(24,r*.45))}px system-ui`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(level,x,y)});
 timeline.max=Math.max(0,clip.records.length-1);timeline.value=recordIndex;
 document.getElementById('frame').textContent=`${record.drop}/${clip.drops} · 动作 ${record.action} · ${record.frame} 帧 / ${(record.frame/replay.physics_fps).toFixed(2)} 秒${record.reset?' · 已重置':''}`;
 document.getElementById('count').textContent=record.fruits.length;document.getElementById('score').textContent=record.score;document.getElementById('merges').textContent=record.merges;
}
function setRecord(value){recordIndex=Math.max(0,Math.min(Number(value),currentClip().records.length-1));draw()}
function stop(){playing=false;playButton.textContent='播放'}
clipSelect.addEventListener('change',()=>{clipIndex=Number(clipSelect.value);recordIndex=0;stop();draw()});
timeline.addEventListener('input',()=>{stop();setRecord(timeline.value)});
document.getElementById('previous').addEventListener('click',()=>{stop();setRecord(recordIndex-1)});
document.getElementById('next').addEventListener('click',()=>{stop();setRecord(recordIndex+1)});
playButton.addEventListener('click',()=>{if(!playing&&recordIndex+1>=currentClip().records.length)recordIndex=0;playing=!playing;playButton.textContent=playing?'暂停':'播放';lastTime=performance.now();accumulator=0;if(playing)requestAnimationFrame(tick)});
function tick(now){if(!playing)return;accumulator+=now-lastTime;lastTime=now;const interval=1000*replay.frame_stride/replay.physics_fps/Number(speedSelect.value);while(accumulator>=interval){accumulator-=interval;if(recordIndex+1>=currentClip().records.length){stop();return}recordIndex++}draw();requestAnimationFrame(tick)}
draw();
</script>
</body>
</html>'''


_FRAGMENT_TEMPLATE = r'''<div id="daxigua-cuda-replay" class="dxr-root">
<style>
#daxigua-cuda-replay{display:grid;grid-template-columns:minmax(260px,360px) minmax(240px,1fr);gap:16px;align-items:start;color:var(--foreground)}
#daxigua-cuda-replay .dxr-board{display:block;width:100%;height:auto;background:var(--card);border:1px solid var(--border);border-radius:8px}
#daxigua-cuda-replay .dxr-side{display:flex;flex-direction:column;gap:12px;min-width:0}
#daxigua-cuda-replay .dxr-status{font-variant-numeric:tabular-nums;color:var(--foreground)}
#daxigua-cuda-replay .dxr-range{width:100%}
@media(max-width:600px){#daxigua-cuda-replay{grid-template-columns:1fr}#daxigua-cuda-replay .dxr-board{max-width:360px;margin:auto}}
</style>
<canvas class="dxr-board" role="img" aria-label="CUDA 水果物理模拟逐帧回放"></canvas>
<div class="dxr-side">
<div class="viz-controls">
<label class="form-label">样本
<select class="form-select dxr-clip"></select>
</label>
<label class="form-label">速度
<select class="form-select dxr-speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
</label>
</div>
<div class="viz-row">
<button type="button" class="btn btn-primary dxr-play">播放</button>
<button type="button" class="btn dxr-previous">上一帧</button>
<button type="button" class="btn dxr-next">下一帧</button>
</div>
<label class="form-label">物理时间轴
<input class="form-range dxr-range" type="range" min="0" max="0" value="0">
</label>
<div class="dxr-status" aria-live="polite"></div>
<div class="text-small text-muted">圆为真实碰撞半径；圆内数字为等级；短线为旋转角；横线为危险线。</div>
</div>
<script>
(()=>{
const root=document.getElementById('daxigua-cuda-replay');
const replay=__TRACE_DATA__,board=root.querySelector('.dxr-board'),ctx=board.getContext('2d');
const clipSelect=root.querySelector('.dxr-clip'),timeline=root.querySelector('.dxr-range'),playButton=root.querySelector('.dxr-play'),speedSelect=root.querySelector('.dxr-speed'),status=root.querySelector('.dxr-status');
let clipIndex=0,recordIndex=0,playing=false,lastTime=0,accumulator=0;
board.width=replay.board.width;board.height=replay.board.height;
replay.clips.forEach((clip,index)=>{const option=document.createElement('option');option.value=index;option.textContent=`环境 ${clip.env} · ${clip.drops} 次投放 · ${clip.total_frames} 帧`;clipSelect.append(option)});
function token(name){const probe=document.createElement('span');probe.style.cssText=`position:absolute;visibility:hidden;color:var(${name})`;root.append(probe);const value=getComputedStyle(probe).color;probe.remove();return value}
const series=[1,2,3,4,5,6].map(i=>token(`--viz-series-${i}`)),foreground=token('--foreground'),card=token('--card'),border=token('--border'),danger=token('--destructive');
function color(level){return series[(level-1)%series.length]||foreground}
function currentClip(){return replay.clips[clipIndex]}
function draw(){
 const clip=currentClip(),record=clip.records[recordIndex],b=replay.board;
 ctx.clearRect(0,0,board.width,board.height);ctx.fillStyle=card;ctx.fillRect(0,0,board.width,board.height);
 ctx.strokeStyle=border;ctx.lineWidth=b.wall_width;ctx.beginPath();ctx.moveTo(b.wall_width/2,0);ctx.lineTo(b.wall_width/2,b.height-b.wall_width/2);ctx.lineTo(b.width-b.wall_width/2,b.height-b.wall_width/2);ctx.lineTo(b.width-b.wall_width/2,0);ctx.stroke();
 ctx.strokeStyle=danger;ctx.lineWidth=2;ctx.setLineDash([10,8]);ctx.beginPath();ctx.moveTo(b.wall_width,b.spawn_y);ctx.lineTo(b.width-b.wall_width,b.spawn_y);ctx.stroke();ctx.setLineDash([]);
 record.fruits.forEach(f=>{const[id,level,x,y,r,angle]=f;ctx.fillStyle=color(level);ctx.globalAlpha=.82;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;ctx.strokeStyle=foreground;ctx.lineWidth=1.5;ctx.stroke();ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+Math.cos(angle)*r*.72,y+Math.sin(angle)*r*.72);ctx.stroke();ctx.fillStyle=card;ctx.font=`500 ${Math.max(13,Math.min(24,r*.45))}px system-ui`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(level,x,y)});
 timeline.max=Math.max(0,clip.records.length-1);timeline.value=recordIndex;
 status.textContent=`投放 ${record.drop}/${clip.drops} · 动作 ${record.action} · 第 ${record.frame} 帧 · ${(record.frame/replay.physics_fps).toFixed(2)} 秒 · ${record.fruits.length} 个水果 · 分数 ${record.score} · 当前投放合并 ${record.merges}${record.reset?' · 新局已重置':''}`;
}
function setRecord(value){recordIndex=Math.max(0,Math.min(Number(value),currentClip().records.length-1));draw()}
function stop(){playing=false;playButton.textContent='播放'}
clipSelect.addEventListener('change',()=>{clipIndex=Number(clipSelect.value);recordIndex=0;stop();draw()});
timeline.addEventListener('input',()=>{stop();setRecord(timeline.value)});
root.querySelector('.dxr-previous').addEventListener('click',()=>{stop();setRecord(recordIndex-1)});
root.querySelector('.dxr-next').addEventListener('click',()=>{stop();setRecord(recordIndex+1)});
playButton.addEventListener('click',()=>{if(!playing&&recordIndex+1>=currentClip().records.length)recordIndex=0;playing=!playing;playButton.textContent=playing?'暂停':'播放';lastTime=performance.now();accumulator=0;if(playing)requestAnimationFrame(tick)});
function tick(now){if(!playing)return;accumulator+=now-lastTime;lastTime=now;const interval=1000*replay.frame_stride/replay.physics_fps/Number(speedSelect.value);while(accumulator>=interval){accumulator-=interval;if(recordIndex+1>=currentClip().records.length){stop();return}recordIndex++}draw();requestAnimationFrame(tick)}
draw();
})();
</script>
</div>'''
