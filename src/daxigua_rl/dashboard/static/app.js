"use strict";

(() => {
  const POLL_MS = 2000;
  const HISTORY_LIMIT = 240;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const byId = (id) => document.getElementById(id);
  const finite = (value) => {
    const result = Number(value);
    return Number.isFinite(result) ? result : null;
  };
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  function read(object, paths, fallback = null) {
    for (const path of Array.isArray(paths) ? paths : [paths]) {
      let value = object;
      for (const part of path.split(".")) {
        if (value == null || typeof value !== "object" || !(part in value)) {
          value = undefined;
          break;
        }
        value = value[part];
      }
      if (value !== undefined && value !== null && value !== "") return value;
    }
    return fallback;
  }

  function number(object, paths, fallback = null) {
    const value = finite(read(object, paths));
    return value === null ? fallback : value;
  }

  function setText(id, value) {
    const element = byId(id);
    if (element) element.textContent = value ?? "—";
  }

  function compact(value, decimals = 0) {
    if (!Number.isFinite(value)) return "—";
    return new Intl.NumberFormat("zh-CN", {
      notation: Math.abs(value) >= 10000 ? "compact" : "standard",
      maximumFractionDigits: decimals,
      minimumFractionDigits: decimals,
    }).format(value);
  }

  function metric(value, decimals = 3) {
    if (!Number.isFinite(value)) return "—";
    if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
    return value.toFixed(decimals);
  }

  function duration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "计算中";
    if (seconds < 60) return `${Math.round(seconds)} 秒`;
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (days) return `${days} 天 ${hours} 小时`;
    if (hours) return `${hours} 小时 ${minutes} 分`;
    return `${minutes} 分钟`;
  }

  function timeLabel(value) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat("zh-CN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(date);
  }

  const animations = new Map();
  function animateText(id, target, formatter) {
    const element = byId(id);
    if (!element) return;
    if (!Number.isFinite(target)) {
      element.textContent = "—";
      delete element.dataset.value;
      return;
    }
    const start = finite(element.dataset.value);
    element.dataset.value = String(target);
    if (start === null || reduceMotion || Math.abs(start - target) < 1e-9) {
      element.textContent = formatter(target);
      return;
    }
    if (animations.has(id)) cancelAnimationFrame(animations.get(id));
    const began = performance.now();
    const draw = (now) => {
      const progress = clamp((now - began) / 520, 0, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      element.textContent = formatter(start + (target - start) * eased);
      if (progress < 1) animations.set(id, requestAnimationFrame(draw));
      else animations.delete(id);
    };
    animations.set(id, requestAnimationFrame(draw));
  }

  function setMeter(id, fillId, value) {
    const meter = byId(id);
    const fill = byId(fillId);
    const safe = Number.isFinite(value) ? clamp(value, 0, 100) : 0;
    if (meter) meter.setAttribute("aria-valuenow", safe.toFixed(1));
    if (fill) fill.style.width = `${safe}%`;
  }

  class LineChart {
    constructor(canvasId, emptyId) {
      this.canvas = byId(canvasId);
      this.empty = byId(emptyId);
      this.series = [];
      this.resizeObserver = new ResizeObserver(() => this.draw());
      this.resizeObserver.observe(this.canvas.parentElement);
    }

    update(series) {
      this.series = series
        .map((entry) => ({
          ...entry,
          points: entry.points.filter((point) =>
            Number.isFinite(point.x) && Number.isFinite(point.y)
          ).slice(-HISTORY_LIMIT),
        }))
        .filter((entry) => entry.points.length);
      this.empty.classList.toggle("is-visible", this.series.length === 0);
      this.canvas.setAttribute(
        "aria-label",
        this.series.length
          ? this.series.map((entry) => `${entry.name} 最新 ${metric(entry.points.at(-1).y, 2)}`).join("；")
          : "暂无曲线数据"
      );
      this.draw();
    }

    draw() {
      const box = this.canvas.getBoundingClientRect();
      if (!box.width || !box.height) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.round(box.width * ratio);
      this.canvas.height = Math.round(box.height * ratio);
      const ctx = this.canvas.getContext("2d");
      ctx.scale(ratio, ratio);
      ctx.clearRect(0, 0, box.width, box.height);
      if (!this.series.length) return;

      const pad = { left: 44, right: 15, top: 29, bottom: 25 };
      const width = box.width - pad.left - pad.right;
      const height = box.height - pad.top - pad.bottom;
      const points = this.series.flatMap((entry) => entry.points);
      let minX = Math.min(...points.map((point) => point.x));
      let maxX = Math.max(...points.map((point) => point.x));
      let minY = Math.min(...points.map((point) => point.y));
      let maxY = Math.max(...points.map((point) => point.y));
      if (minX === maxX) maxX = minX + 1;
      if (minY === maxY) {
        minY -= Math.max(1, Math.abs(minY) * 0.1);
        maxY += Math.max(1, Math.abs(maxY) * 0.1);
      }
      const yPadding = (maxY - minY) * 0.1;
      minY -= yPadding;
      maxY += yPadding;
      const px = (x) => pad.left + ((x - minX) / (maxX - minX)) * width;
      const py = (y) => pad.top + height - ((y - minY) / (maxY - minY)) * height;

      ctx.lineWidth = 1;
      ctx.font = '10px "Segoe UI", sans-serif';
      ctx.fillStyle = "rgba(144,170,161,.72)";
      ctx.strokeStyle = "rgba(180,255,228,.075)";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let index = 0; index <= 4; index += 1) {
        const y = pad.top + (height * index) / 4;
        const value = maxY - ((maxY - minY) * index) / 4;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + width, y);
        ctx.stroke();
        ctx.fillText(compact(value, Math.abs(value) < 10 ? 2 : 0), pad.left - 7, y);
      }

      this.series.forEach((entry, seriesIndex) => {
        const line = new Path2D();
        entry.points.forEach((point, index) => {
          const x = px(point.x);
          const y = py(point.y);
          if (index === 0) line.moveTo(x, y);
          else line.lineTo(x, y);
        });
        if (seriesIndex === 0 && entry.points.length > 1) {
          const fill = new Path2D(line);
          fill.lineTo(px(entry.points.at(-1).x), pad.top + height);
          fill.lineTo(px(entry.points[0].x), pad.top + height);
          fill.closePath();
          const gradient = ctx.createLinearGradient(0, pad.top, 0, pad.top + height);
          gradient.addColorStop(0, `${entry.color}24`);
          gradient.addColorStop(1, `${entry.color}00`);
          ctx.fillStyle = gradient;
          ctx.fill(fill);
        }
        ctx.strokeStyle = entry.color;
        ctx.lineWidth = 1.8;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.stroke(line);
        const last = entry.points.at(-1);
        ctx.fillStyle = entry.color;
        ctx.beginPath();
        ctx.arc(px(last.x), py(last.y), 2.8, 0, Math.PI * 2);
        ctx.fill();
      });

      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      let legendX = pad.left;
      for (const entry of this.series) {
        ctx.fillStyle = entry.color;
        ctx.fillRect(legendX, 8, 8, 2);
        ctx.fillStyle = "rgba(214,239,231,.72)";
        ctx.fillText(entry.name, legendX + 12, 4);
        legendX += Math.max(72, ctx.measureText(entry.name).width + 31);
      }
      ctx.fillStyle = "rgba(144,170,161,.62)";
      ctx.textAlign = "left";
      ctx.fillText(compact(minX), pad.left, box.height - 14);
      ctx.textAlign = "right";
      ctx.fillText(compact(maxX), pad.left + width, box.height - 14);
    }
  }

  function normalizePoints(value) {
    const source = Array.isArray(value) ? value : value?.points;
    if (!Array.isArray(source)) return [];
    return source.map((item, index) => {
      if (Array.isArray(item)) return { x: finite(item[0]) ?? index, y: finite(item[1]) };
      if (typeof item === "number") return { x: index, y: item };
      return {
        x: finite(item?.x ?? item?.step ?? item?.update_step ?? item?.timestamp) ?? index,
        y: finite(item?.y ?? item?.value ?? item?.score),
      };
    }).filter((point) => point.y !== null);
  }

  function series(state, aliases) {
    for (const alias of aliases) {
      const points = normalizePoints(read(state, [`series.${alias}`, alias]));
      if (points.length) return points;
    }
    return [];
  }

  const charts = {
    score: new LineChart("score-chart", "score-empty"),
    loss: new LineChart("loss-chart", "loss-empty"),
    throughput: new LineChart("throughput-chart", "throughput-empty"),
    util: new LineChart("util-chart", "util-empty"),
  };

  function updateCharts(state) {
    const score = series(state, ["score", "episode_score", "reward"]);
    const loss = series(state, ["loss", "total_loss"]);
    const td = series(state, ["td_loss"]);
    const structural = series(state, ["structural_loss", "structure_loss"]);
    const causal = series(state, ["causal_loss", "counterfactual_loss"]);
    const envSpeed = series(state, ["env_steps_per_second", "steps_per_second"]);
    const updateSpeed = series(state, ["updates_per_second"]);
    const cpu = series(state, ["cpu_util", "cpu_util_percent", "cpu"]);
    const gpu = series(state, ["gpu_util", "gpu_util_percent", "gpu"]);

    charts.score.update([{ name: "局分", color: "#48f5c4", points: score }]);
    charts.loss.update([
      { name: "总损失", color: "#a889ff", points: loss },
      { name: "TD", color: "#68b9ff", points: td },
      { name: "结构", color: "#48f5c4", points: structural },
      { name: "因果", color: "#ffc66d", points: causal },
    ]);
    charts.throughput.update([
      { name: "投放/s", color: "#48f5c4", points: envSpeed },
      { name: "更新/s", color: "#a889ff", points: updateSpeed },
    ]);
    charts.util.update([
      { name: "CPU %", color: "#68b9ff", points: cpu },
      { name: "GPU %", color: "#a889ff", points: gpu },
    ]);

    setText("score-latest", score.length ? compact(score.at(-1).y, 1) : "—");
    setText("loss-latest", loss.length ? metric(loss.at(-1).y, 3) : "—");
    setText("throughput-latest", envSpeed.length ? `${metric(envSpeed.at(-1).y, 1)} /s` : "—");
    setText("util-latest", gpu.length ? `GPU ${metric(gpu.at(-1).y, 0)}%` : "—");
    const longest = Math.max(score.length, loss.length, envSpeed.length, cpu.length, gpu.length);
    setText("chart-window-label", longest ? `最近 ${longest} 个采样点` : "最近训练窗口");
  }

  const plotSlots = {
    training: { image: "plot-training", placeholder: "plot-training-placeholder", time: "plot-training-time" },
    reward: { image: "plot-reward", placeholder: "plot-reward-placeholder", time: "plot-reward-time" },
    structure: { image: "plot-structure", placeholder: "plot-structure-placeholder", time: "plot-structure-time" },
  };

  function classifyPlot(plot) {
    const name = String(plot?.name ?? plot?.title ?? plot?.url ?? "").toLowerCase();
    if (name.includes("reward")) return "reward";
    if (name.includes("structure")) return "structure";
    if (name.includes("training")) return "training";
    return null;
  }

  function updatePlots(state) {
    const found = new Set();
    const plots = Array.isArray(state.plots) ? state.plots : [];
    for (const plot of plots) {
      const type = classifyPlot(plot);
      if (!type || found.has(type)) continue;
      found.add(type);
      const slot = plotSlots[type];
      const image = byId(slot.image);
      const rawUrl = plot.url || (plot.name ? `/plots/${encodeURIComponent(plot.name)}` : null);
      if (!rawUrl) continue;
      const version = plot.modified_at ?? plot.mtime ?? plot.size_bytes ?? "";
      const key = `${rawUrl}|${version}`;
      if (image.dataset.key !== key) {
        const separator = rawUrl.includes("?") ? "&" : "?";
        const candidate = new Image();
        candidate.onload = () => {
          image.src = candidate.src;
          image.dataset.key = key;
          image.classList.add("is-loaded");
        };
        candidate.onerror = () => image.classList.remove("is-loaded");
        candidate.src = `${rawUrl}${separator}v=${encodeURIComponent(version || Date.now())}`;
      }
      setText(slot.time, plot.modified_at ? `更新于 ${timeLabel(plot.modified_at)}` : "已生成");
    }
  }

  function updateEvents(state) {
    const raw = Array.isArray(state.alerts)
      ? state.alerts
      : Array.isArray(state.events) ? state.events : [];
    const list = byId("event-list");
    list.replaceChildren();
    const alerts = raw.slice(-8).reverse();
    setText("event-count", `${alerts.length} 条`);
    if (!alerts.length) {
      const item = document.createElement("li");
      item.className = "event-item event-success";
      const dot = document.createElement("span");
      dot.className = "event-dot";
      dot.setAttribute("aria-hidden", "true");
      const body = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = "当前没有活动告警";
      const message = document.createElement("p");
      message.textContent = "训练、资源和因果预算均未报告异常。";
      body.append(title, message);
      const stamp = document.createElement("time");
      stamp.textContent = "现在";
      item.append(dot, body, stamp);
      list.append(item);
      return;
    }
    for (const alert of alerts) {
      const severity = String(alert.severity ?? alert.level ?? alert.type ?? "neutral").toLowerCase();
      const style = severity.includes("error") || severity.includes("critical")
        ? "error"
        : severity.includes("warn") ? "warning"
          : severity.includes("success") || severity.includes("ok") ? "success" : "neutral";
      const item = document.createElement("li");
      item.className = `event-item event-${style}`;
      const dot = document.createElement("span");
      dot.className = "event-dot";
      dot.setAttribute("aria-hidden", "true");
      const body = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = String(
        alert.title ?? alert.code ?? alert.event ?? "训练事件"
      ).replaceAll("_", " ");
      const message = document.createElement("p");
      message.textContent = alert.message ?? alert.detail ?? alert.description ?? "";
      body.append(title, message);
      const stamp = document.createElement("time");
      const timestamp = alert.timestamp ?? alert.created_at ?? alert.time;
      stamp.dateTime = timestamp ?? "";
      stamp.textContent = timestamp ? timeLabel(timestamp) : "—";
      item.append(dot, body, stamp);
      list.append(item);
    }
  }

  function updateStatus(state) {
    const connection = byId("connection");
    const progress = state.progress ?? {};
    const active = Boolean(progress.process_active ?? read(state, "process_active", false));
    const complete = Boolean(progress.is_complete);
    const status = String(state.status ?? (complete ? "complete" : active ? "running" : "idle")).toLowerCase();
    connection.className = "connection";
    if (status.includes("fail") || status.includes("error")) {
      connection.classList.add("is-error");
      setText("connection-label", "训练异常");
    } else if (active || status.includes("run")) {
      connection.classList.add("is-live");
      setText("connection-label", "实时同步中");
    } else if (complete) {
      connection.classList.add("is-live");
      setText("connection-label", "训练已完成");
    } else {
      connection.classList.add("is-stale");
      setText("connection-label", "等待训练进程");
    }
    setText("last-updated", state.generated_at ? `更新 ${timeLabel(state.generated_at)}` : "刚刚更新");
  }

  function updateDashboard(state) {
    const p = state.progress ?? {};
    const r = state.resources ?? {};
    const t = state.training ?? {};
    const a = state.actor ?? {};
    const c = state.causal ?? {};
    const current = number(state, ["progress.current_update", "training.update_step", "update_step"]);
    const target = number(state, ["progress.target_updates", "training.total_updates", "total_updates"]);
    const percent = number(state, ["progress.percent"], current !== null && target ? current / target * 100 : 0);
    const envSteps = number(state, ["progress.env_steps", "training.env_steps", "env_steps"]);
    const updateSpeed = number(state, ["progress.updates_per_second", "training.updates_per_second"]);
    const envSpeed = number(state, ["progress.env_steps_per_second", "training.env_steps_per_second"]);

    updateStatus(state);
    setText("phase-pill", String(read(state, ["progress.phase", "phase"], "IDLE")).toUpperCase());
    setText(
      "run-name",
      read(
        state,
        ["identity.run", "paths.run_dir", "run.name", "run_name"],
        "未指定训练目录"
      )
    );
    setText("run-message", read(state, ["message", "progress.message"], "面板每 2 秒自动同步训练、硬件与因果归因状态。"));
    animateText("update-step", current, (value) => compact(Math.round(value)));
    setText("total-updates", target === null ? " / —" : ` / ${compact(target)}`);
    animateText("env-steps", envSteps, (value) => compact(Math.round(value)));
    setText("eta", duration(number(state, ["progress.eta_seconds", "eta_seconds"])));
    animateText("progress-value", clamp(percent ?? 0, 0, 100), (value) => `${value.toFixed(1)}%`);
    const ring = byId("progress-ring");
    ring.style.setProperty("--progress", `${clamp(percent ?? 0, 0, 100) * 3.6}deg`);
    ring.setAttribute("aria-valuenow", clamp(percent ?? 0, 0, 100).toFixed(1));
    animateText("updates-speed", updateSpeed, (value) => value.toFixed(2));
    animateText("steps-speed", envSpeed, (value) => value.toFixed(1));

    const cpuCount = number(state, ["resources.cpu_count", "resources.cpu_total_cores"], 25);
    const cpuCores = number(state, ["resources.cpu_cores_used", "resources.cpu_used_cores"]);
    const cpuPercent = number(state, ["resources.cpu_util_percent"], cpuCores !== null && cpuCount ? cpuCores / cpuCount * 100 : null);
    animateText("cpu-cores", cpuCores, (value) => value.toFixed(1));
    setText("cpu-total", `/ ${compact(cpuCount)} 核`);
    setText("cpu-percent", cpuPercent === null ? "—" : `${cpuPercent.toFixed(1)}%`);
    setText("cpu-state", cpuPercent === null ? "等待数据" : cpuPercent >= 90 ? "高负载" : "运行正常");
    setMeter("cpu-meter", "cpu-meter-fill", cpuPercent);

    const gpuPercent = number(state, ["resources.gpu_util_percent", "resources.gpu_utilization_percent"]);
    animateText("gpu-percent-value", gpuPercent, (value) => value.toFixed(0));
    setText("gpu-name", read(state, ["resources.gpu_name", "gpu.name"], "NVIDIA GPU"));
    const temperature = number(state, ["resources.gpu_temperature_c"]);
    setText("gpu-temperature", temperature === null ? "— °C" : `${temperature.toFixed(0)} °C`);
    setMeter("gpu-meter", "gpu-meter-fill", gpuPercent);

    const vramUsedMb = number(state, ["resources.gpu_memory_used_mb", "resources.vram_used_mb"]);
    const vramTotalMb = number(state, ["resources.gpu_memory_total_mb", "resources.vram_total_mb"]);
    const vramPercent = vramUsedMb !== null && vramTotalMb ? vramUsedMb / vramTotalMb * 100 : null;
    animateText("vram-used", vramUsedMb === null ? null : vramUsedMb / 1024, (value) => value.toFixed(1));
    setText("vram-total", vramTotalMb === null ? "/ — GB" : `/ ${(vramTotalMb / 1024).toFixed(1)} GB`);
    setText("vram-percent", vramPercent === null ? "—" : `${vramPercent.toFixed(1)}%`);
    setMeter("vram-meter", "vram-meter-fill", vramPercent);

    const power = number(state, ["resources.gpu_power_draw_w", "resources.gpu_power_w"]);
    const powerLimit = number(state, ["resources.gpu_power_limit_w"]);
    const powerPercent = power !== null && powerLimit ? power / powerLimit * 100 : null;
    animateText("power-draw", power, (value) => value.toFixed(0));
    setText("power-limit", powerLimit === null ? "— W" : `${powerLimit.toFixed(0)} W`);
    setMeter("power-meter", "power-meter-fill", powerPercent);

    const epsilon = number(state, ["progress.epsilon", "training.epsilon"]);
    const loss = number(state, ["training.loss", "training.total_loss"]);
    const tdLoss = number(state, ["training.td_loss"]);
    const structuralLoss = number(state, ["training.structural_loss", "training.weighted_structural_loss"]);
    const structuralValid = number(state, ["training.structural_valid_count", "training.structural_sample_count"]);
    const causalLoss = number(state, [
      "training.causal_loss",
      "training.counterfactual_loss",
      "causal.loss",
      "causal.counterfactual_loss",
      "causal.rule_rank_loss",
    ]);
    const causalBatch = number(state, ["training.causal_batch_size", "causal.batch_size", "causal.samples"]);
    animateText("epsilon", epsilon, (value) => value.toFixed(4));
    setText("epsilon-mode", read(state, ["training.epsilon_mode", "training.epsilon_schedule"], "epsilon greedy"));
    animateText("total-loss", loss, (value) => metric(value, 4));
    setText("td-loss", metric(tdLoss, 4));
    animateText("structural-loss", structuralLoss, (value) => metric(value, 4));
    setText("structural-valid", structuralValid === null ? "—" : compact(structuralValid));
    animateText("causal-loss", causalLoss, (value) => metric(value, 4));
    setText("causal-batch", causalBatch === null ? "—" : compact(causalBatch));

    const actorMean = number(state, ["actor.mean_batch_size", "actor.inference_mean_batch_size", "training.actor_inference_mean_batch_size"]);
    const actorRequests = number(state, ["actor.requests", "actor.inference_requests", "training.actor_inference_requests"]);
    const actorBatches = number(state, ["actor.batches", "actor.inference_batches", "training.actor_inference_batches"]);
    const actorSeconds = number(state, ["actor.inference_seconds", "actor.latency_seconds"]);
    const actorLatency = number(state, ["actor.latency_ms"], actorSeconds !== null && actorBatches ? actorSeconds / actorBatches * 1000 : null);
    animateText("actor-batch", actorMean, (value) => value.toFixed(1));
    setText("actor-requests", actorRequests === null ? "—" : compact(actorRequests));
    setText("actor-batches", actorBatches === null ? "—" : compact(actorBatches));
    setText("actor-latency", actorLatency === null ? "—" : actorLatency.toFixed(1));

    const commit = read(state, ["identity.commit", "git.commit", "training.commit"]);
    const branch = read(state, ["identity.branch", "git.branch"]);
    const runIdentity = read(state, ["identity.run"]);
    setText(
      "footer-identity",
      [branch, commit ? String(commit).slice(0, 9) : null, runIdentity]
        .filter(Boolean)
        .join(" · ") || "运行身份未提供"
    );
    updateCharts(state);
    updatePlots(state);
    updateEvents(state);
  }

  let timer = null;
  let failures = 0;
  async function poll() {
    clearTimeout(timer);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch("/api/state", {
        cache: "no-store",
        headers: { Accept: "application/json" },
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const state = await response.json();
      failures = 0;
      updateDashboard(state && typeof state === "object" ? state : {});
    } catch (error) {
      failures += 1;
      const connection = byId("connection");
      connection.className = "connection is-offline";
      setText("connection-label", failures > 1 ? "连接已中断" : "正在重连");
      setText("last-updated", error.name === "AbortError" ? "请求超时" : "2 秒后重试");
    } finally {
      clearTimeout(timeout);
      timer = setTimeout(poll, POLL_MS);
    }
  }

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) poll();
  });
  poll();
})();
