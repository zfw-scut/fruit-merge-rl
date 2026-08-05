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


_DASHBOARD_HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>合成大西瓜训练面板</title>
<style>
body{margin:0;background:#0b1020;color:#dce7ff;font:14px system-ui,sans-serif}
header{padding:18px 24px;background:#111a30;position:sticky;top:0}
h1{font-size:20px;margin:0 0 4px}.muted{color:#8ea2c9}
main{padding:18px;display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.card{background:#121c33;border:1px solid #253454;border-radius:12px;padding:15px}
.card h2{font-size:15px;margin:0 0 12px;color:#a9c7ff}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 16px}
.metric{display:flex;justify-content:space-between;border-bottom:1px solid #22304d;padding:5px 0}
.value{font-variant-numeric:tabular-nums;color:#fff}.wide{grid-column:1/-1}
.ok{color:#72e6a5}.warn{color:#ffd166}.bad{color:#ff758f}
canvas{width:100%;height:130px;background:#0d1528;border-radius:8px}
</style></head><body>
<header><h1>GNN-DQN 云端训练</h1><div id="subtitle" class="muted">等待训练指标…</div></header>
<main>
<section class="card"><h2>训练进度</h2><div id="progress" class="grid"></div></section>
<section class="card"><h2>实时吞吐</h2><div id="throughput" class="grid"></div></section>
<section class="card"><h2>服务器资源</h2><div id="resources" class="grid"></div></section>
<section class="card"><h2>学习状态</h2><div id="learning" class="grid"></div></section>
<section class="card wide"><h2>吞吐历史</h2><canvas id="chart" width="1000" height="160"></canvas></section>
<section class="card wide"><h2>最近事件</h2><div id="events"></div></section>
</main><script>
const groups={
 progress:['phase','transitions','updates','episodes','epsilon','replay_size','active_envs','uptime_seconds','eta_seconds'],
 throughput:['env_steps_per_second','updates_per_second','learner_samples_per_second','physics_seconds','actor_seconds','learner_seconds'],
 resources:['gpu_utilization','gpu_memory_used_mb','gpu_memory_total_mb','gpu_temperature','gpu_power_watts','cpu_utilization','memory_used_mb','memory_total_mb'],
 learning:['loss','mean_reward','mean_q','mean_target','mean_abs_td_error','grad_norm','action_distribution','training_mean_score','training_mean_drops','last_fast_eval_score','last_accurate_eval_score']};
function fmt(v){if(v===null||v===undefined)return '—';if(typeof v==='number'){if(Math.abs(v)>=1e6)return (v/1e6).toFixed(2)+'M';if(Math.abs(v)>=1e3)return v.toFixed(0);return v.toFixed(3)}return String(v)}
function renderGroup(id,keys,d){document.getElementById(id).innerHTML=keys.map(k=>`<div class="metric"><span class="muted">${k}</span><span class="value">${fmt(d[k])}</span></div>`).join('')}
function chart(history){const c=document.getElementById('chart'),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);let vals=history.map(v=>v.env_steps_per_second||0);let max=Math.max(1,...vals);x.strokeStyle='#66a3ff';x.lineWidth=2;x.beginPath();vals.forEach((v,i)=>{let px=i*Math.max(1,c.width/(vals.length-1||1)),py=c.height-8-v/max*(c.height-16);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}
async function tick(){try{let r=await fetch('/api/status',{cache:'no-store'}),s=await r.json(),d={...(s.training||{}),...(s.resources||{})};document.getElementById('subtitle').textContent=`${d.phase||'unknown'} · ${new Date((s.timestamp||0)*1000).toLocaleTimeString()}`;Object.entries(groups).forEach(([id,keys])=>renderGroup(id,keys,d));chart(s.history||[]);document.getElementById('events').innerHTML=(s.events||[]).slice(-10).reverse().map(e=>`<div class="metric"><span>${e.kind||'event'}</span><span class="muted">${e.message||''}</span></div>`).join('')}catch(e){document.getElementById('subtitle').textContent='面板暂时无法读取训练状态'}setTimeout(tick,1000)}tick();
</script></body></html>'''


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
        self.events = deque(maxlen=100)
        self.history = deque(maxlen=history_size)
        self.timestamp = time.time()

    def update_training(self, payload):
        with self.lock:
            self.training.update(payload)
            self.timestamp = time.time()
            if 'env_steps_per_second' in payload:
                self.history.append({
                    'timestamp': self.timestamp,
                    'env_steps_per_second': payload.get(
                        'env_steps_per_second', 0.0
                    ),
                    'updates_per_second': payload.get(
                        'updates_per_second', 0.0
                    ),
                })

    def update_resources(self, payload):
        with self.lock:
            self.resources = dict(payload)

    def add_event(self, payload):
        with self.lock:
            self.events.append(dict(payload))

    def snapshot(self):
        with self.lock:
            return {
                'timestamp': self.timestamp,
                'training': dict(self.training),
                'resources': dict(self.resources),
                'events': list(self.events),
                'history': list(self.history),
            }


def _dashboard_process_main(
        metric_queue,
        host,
        port,
        resource_interval,
        history_size,
        run_dir,
        target_pid):
    state = _DashboardState(history_size)
    stop_event = threading.Event()
    sampler = ResourceSampler(target_pid)
    output_path = Path(run_dir) / 'monitoring.jsonl'
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                payload['monitor_timestamp'] = time.time()
                if kind == 'event':
                    state.add_event(payload)
                else:
                    state.update_training(payload)
                log.write(json.dumps(
                    {'kind': kind, **payload}, ensure_ascii=False
                ) + '\n')

    def collect_resources():
        resource_path = Path(run_dir) / 'resources.jsonl'
        with resource_path.open('a', encoding='utf-8', buffering=1) as log:
            while not stop_event.wait(resource_interval):
                payload = sampler.sample()
                state.update_resources(payload)
                log.write(json.dumps(
                    {'timestamp': time.time(), **payload},
                    ensure_ascii=False,
                ) + '\n')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/api/status':
                body = json.dumps(
                    state.snapshot(), ensure_ascii=False
                ).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
            elif self.path in ('/', '/index.html'):
                body = _DASHBOARD_HTML.encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
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
    consumer.start()
    resource_thread.start()
    server = ThreadingHTTPServer((host, port), Handler)
    server.timeout = 0.5
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        stop_event.set()
        server.server_close()
        sampler.close()


class DashboardPublisher:
    """训练侧非阻塞发布器；任何面板故障都降级为丢弃指标。"""

    def __init__(self, config, run_dir):
        self.enabled = bool(config.enabled)
        self.process = None
        self.queue = None
        self.dropped_messages = 0
        if not self.enabled:
            return
        context = multiprocessing.get_context('spawn')
        self.queue = context.Queue(maxsize=128)
        self.process = context.Process(
            target=_dashboard_process_main,
            args=(
                self.queue,
                config.host,
                config.port,
                config.resource_interval_seconds,
                config.history_size,
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

    def close(self, timeout=5.0):
        if not self.enabled or self.queue is None:
            return
        try:
            self.queue.put_nowait(None)
        except (queue.Full, OSError, ValueError):
            pass
        if self.process is not None:
            self.process.join(timeout)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(1.0)
