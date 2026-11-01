// marionette UI: vanilla JS, no build step. Talks to the FastAPI server under /api.
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = {
  info: null,
  messages: [],      // chat history sent to the model (user/assistant turns)
  busy: false,
  steers: new Map(), // feature id -> {label, strength}
  trace: [],         // per-token probe scores for the strip chart
  presetId: null,
};

// ---------------------------------------------------------------- colour helpers
function heat(p) {
  // 0 -> green, 0.5 -> amber, 1 -> red (matches the gauge gradient)
  const stops = [[0, [52, 211, 153]], [0.55, [251, 191, 36]], [1, [244, 63, 94]]];
  for (let i = 1; i < stops.length; i++) {
    if (p <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i];
      const t = (p - a) / (b - a);
      return `rgb(${ca.map((c, j) => Math.round(c + t * (cb[j] - c))).join(",")})`;
    }
  }
  return "rgb(244,63,94)";
}
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
const pct = (x) => (x === null || x === undefined ? "–" : `${(100 * x).toFixed(1)}%`);
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------------------------------------------------------------- gauge with a damped spring needle
const needle = { angle: -90, velocity: 0, target: -90 };
function setNeedle(p) { needle.target = -90 + 180 * Math.min(1, Math.max(0, p)); }
function animateNeedle() {
  // Critically under-damped spring: the needle overshoots a little and settles, like a real meter.
  const k = 0.06, damping = 0.82;
  needle.velocity = (needle.velocity + k * (needle.target - needle.angle)) * damping;
  needle.angle += needle.velocity;
  $("#needle").setAttribute("transform", `rotate(${needle.angle.toFixed(2)} 150 160)`);
  requestAnimationFrame(animateNeedle);
}
function drawTicks() {
  const g = $("#ticks");
  for (let i = 0; i <= 10; i++) {
    const a = Math.PI * (1 - i / 10);
    const r1 = 104, r2 = i % 5 === 0 ? 92 : 98;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", 150 + r1 * Math.cos(a)); line.setAttribute("y1", 160 - r1 * Math.sin(a));
    line.setAttribute("x2", 150 + r2 * Math.cos(a)); line.setAttribute("y2", 160 - r2 * Math.sin(a));
    g.appendChild(line);
  }
}
function placeThreshold(p) {
  const a = Math.PI * (1 - p);
  const m = $("#threshold-mark");
  m.setAttribute("x1", 150 + 132 * Math.cos(a)); m.setAttribute("y1", 160 - 132 * Math.sin(a));
  m.setAttribute("x2", 150 + 112 * Math.cos(a)); m.setAttribute("y2", 160 - 112 * Math.sin(a));
  m.innerHTML = `<title>flag threshold: 1% false positives on benign chat (p = ${p.toFixed(2)})</title>`;
}
function setVerdict(p, done, flagged) {
  $("#p-value").textContent = p === null ? "–" : p.toFixed(2);
  const v = $("#verdict");
  const thr = state.info ? state.info.threshold_probability : 0.5;
  if (p === null) { v.className = "verdict idle"; v.textContent = "waiting"; return; }
  const alert = done ? flagged : p > thr;
  v.className = `verdict ${alert ? "alert" : "ok"}`;
  v.textContent = alert ? (done ? "deception detected" : "deceptive?") : (done ? "no deception detected" : "listening…");
}

// ---------------------------------------------------------------- strip chart (polygraph paper)
function drawStrip() {
  const c = $("#strip"), ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#1b202b"; ctx.lineWidth = 1;
  for (let y = 0; y <= 4; y++) { ctx.beginPath(); ctx.moveTo(0, (h * y) / 4); ctx.lineTo(w, (h * y) / 4); ctx.stroke(); }
  for (let x = 0; x < w; x += 28) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
  if (state.info) {
    const ty = h - state.info.threshold_probability * (h - 10) - 5;
    ctx.setLineDash([4, 4]); ctx.strokeStyle = "#e6e8ee55";
    ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(w, ty); ctx.stroke(); ctx.setLineDash([]);
  }
  const pts = state.trace.slice(-90);
  if (!pts.length) return;
  const step = w / 90;
  ctx.lineWidth = 2; ctx.lineJoin = "round";
  for (let i = 1; i < pts.length; i++) {
    const x0 = (i - 1) * step, x1 = i * step;
    const y0 = h - pts[i - 1] * (h - 10) - 5, y1 = h - pts[i] * (h - 10) - 5;
    ctx.strokeStyle = heat(pts[i]);
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
  const last = pts[pts.length - 1];
  ctx.fillStyle = heat(last);
  ctx.beginPath(); ctx.arc((pts.length - 1) * step, h - last * (h - 10) - 5, 3.5, 0, 2 * Math.PI); ctx.fill();
}

// ---------------------------------------------------------------- thought stream
function showThoughts(features, tokenText) {
  const box = $("#thoughts");
  $("#thought-token").textContent = tokenText ? `“${tokenText.trim() || "␣"}”` : "";
  if (!state.info || !state.info.sae) {
    box.innerHTML = '<p class="hint">No SAE is loaded for this profile, so the thought stream is off.</p>';
    return;
  }
  if (!features || !features.length) { box.innerHTML = '<p class="hint">No labelled feature is active on this token.</p>'; return; }
  box.innerHTML = features.map((f) => `
    <div class="thought" title="feature #${f.id}">
      <span>${escapeHtml(f.label)} <span class="id">#${f.id}</span></span>
      <div class="bar" style="width:${Math.min(100, 100 * f.strength).toFixed(0)}%"></div>
    </div>`).join("");
}

// ---------------------------------------------------------------- transcript
function addMessage(role, text) {
  const empty = $(".empty-state"); if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `<span class="who">${role === "user" ? "you" : "model"}</span>`;
  if (text) div.appendChild(document.createTextNode(text));
  $("#transcript").appendChild(div);
  $("#transcript").scrollTop = 1e9;
  return div;
}

function tokenSpan(ev) {
  const span = document.createElement("span");
  span.className = "tok";
  span.textContent = ev.text;
  span.style.borderBottomColor = heat(ev.p_token);
  span.dataset.p = ev.p_token;
  span.addEventListener("mouseenter", (e) => {
    const tip = $("#tooltip");
    const feats = (ev.features || []).slice(0, 3).map((f) => escapeHtml(f.label)).join(" · ");
    tip.innerHTML = `<b>token score ${fmt(ev.p_token)}</b> · running verdict ${fmt(ev.p_response)}` +
      (feats ? `<br><span style="color:var(--muted)">${feats}</span>` : "");
    tip.hidden = false;
    tip.style.left = `${e.clientX + 12}px`; tip.style.top = `${e.clientY + 14}px`;
    showThoughts(ev.features, ev.text);
  });
  span.addEventListener("mouseleave", () => { $("#tooltip").hidden = true; });
  return span;
}

// ---------------------------------------------------------------- chat streaming
function buildMessages() {
  const sys = $("#system").value.trim();
  return [...(sys ? [{ role: "system", content: sys }] : []), ...state.messages];
}

async function generate() {
  if (state.busy) return;
  state.busy = true;
  for (const b of ["#send-btn", "#regen-btn"]) $(b).disabled = true;
  state.trace = []; drawStrip(); setNeedle(0); setVerdict(null);
  const bubble = addMessage("assistant", "");
  let reply = "";
  const body = {
    messages: buildMessages(),
    truth_serum: Number($("#serum").value),
    features: [...state.steers.entries()].filter(([, s]) => s.strength !== 0).map(([id, s]) => ({ id, strength: s.strength })),
    max_new_tokens: 160,
    temperature: 0,
  };
  try {
    const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl); buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === "token") {
          reply += ev.text;
          bubble.appendChild(tokenSpan(ev));
          state.trace.push(ev.p_token); drawStrip();
          setNeedle(ev.p_response); setVerdict(ev.p_response, false, false);
          showThoughts(ev.features, ev.text);
          $("#transcript").scrollTop = 1e9;
        } else if (ev.type === "done") {
          setNeedle(ev.p_response); setVerdict(ev.p_response, true, ev.flagged);
          bubble.classList.toggle("flagged", ev.flagged);
        } else if (ev.type === "error") {
          throw new Error(ev.message);
        }
      }
    }
    state.messages.push({ role: "assistant", content: reply });
  } catch (err) {
    bubble.appendChild(Object.assign(document.createElement("div"), { className: "hint", textContent: `error: ${err.message}` }));
  } finally {
    state.busy = false;
    for (const b of ["#send-btn", "#regen-btn"]) $(b).disabled = false;
  }
}

function send(text) {
  if (!text.trim() || state.busy) return;
  state.messages.push({ role: "user", content: text.trim() });
  addMessage("user", text.trim());
  $("#user-input").value = "";
  generate();
}

function regenerate() {
  if (state.busy || !state.messages.length) return;
  if (state.messages[state.messages.length - 1].role === "assistant") state.messages.pop();
  const bubbles = document.querySelectorAll(".msg.assistant");
  if (bubbles.length) bubbles[bubbles.length - 1].remove();
  generate();
}

function reset() {
  state.messages = [];
  $("#transcript").innerHTML = "";
  state.trace = []; drawStrip(); setNeedle(0); setVerdict(null); showThoughts([], "");
}

// ---------------------------------------------------------------- presets
async function loadPresets() {
  const presets = await (await fetch("/api/presets")).json();
  const list = $("#preset-list");
  for (const p of presets) {
    const b = document.createElement("button");
    b.className = "preset"; b.dataset.id = p.id;
    b.innerHTML = `<span class="title">${escapeHtml(p.title)}</span>
      <span class="meta"><span class="tag ${p.kind}">${p.kind}</span>${p.held_out ? "held-out family" : "training family"}</span>`;
    b.addEventListener("click", () => {
      document.querySelectorAll(".preset").forEach((x) => x.classList.toggle("active", x === b));
      state.presetId = p.id;
      reset();
      $("#system").value = p.system;
      send(p.user);
    });
    list.appendChild(b);
  }
}

// ---------------------------------------------------------------- feature sliders
let searchTimer = null;
async function searchFeatures(q) {
  const hits = await (await fetch(`/api/features?q=${encodeURIComponent(q)}&limit=12`)).json();
  const box = $("#feature-results");
  box.innerHTML = "";
  for (const f of hits) {
    const b = document.createElement("button");
    b.textContent = `#${f.id} ${f.label}`;
    b.addEventListener("click", () => addSlider(f));
    box.appendChild(b);
  }
}
function addSlider(f) {
  if (state.steers.has(f.id)) return;
  const max = state.info ? state.info.max_feature_strength : 3;
  state.steers.set(f.id, { label: f.label, strength: 0 });
  const row = document.createElement("div");
  row.className = "slider-row";
  row.innerHTML = `<span>#${f.id} ${escapeHtml(f.label)}</span><span class="val">0.0</span><button class="x" title="remove">×</button>
    <input type="range" min="${-max}" max="${max}" step="0.25" value="0">`;
  row.querySelector("input").addEventListener("input", (e) => {
    state.steers.get(f.id).strength = Number(e.target.value);
    row.querySelector(".val").textContent = Number(e.target.value).toFixed(2);
  });
  row.querySelector(".x").addEventListener("click", () => { state.steers.delete(f.id); row.remove(); });
  $("#feature-sliders").appendChild(row);
}

// ---------------------------------------------------------------- research dashboard
function svgEl(tag, attrs, parent) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function lineChart(container, series, opts) {
  const W = 560, H = 260, m = { l: 40, r: 12, t: 12, b: 32 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" });
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = opts.y;
  const X = (x) => m.l + ((x - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (y) => H - m.b - ((y - y0) / (y1 - y0)) * (H - m.t - m.b);
  for (const t of opts.yTicks) {
    svgEl("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridline" }, svg);
    svgEl("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, svg).textContent = opts.yFmt(t);
  }
  for (const t of opts.xTicks) svgEl("text", { x: X(t), y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = t;
  svgEl("text", { x: (W + m.l) / 2, y: H - 2, "text-anchor": "middle" }, svg).textContent = opts.xLabel;
  if (opts.marker !== undefined) {
    svgEl("line", { x1: X(opts.marker), x2: X(opts.marker), y1: m.t, y2: H - m.b, stroke: "#e6e8ee55", "stroke-dasharray": "4 4" }, svg);
  }
  for (const s of series) {
    const pts = s.points.filter((p) => p[1] !== null);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash || "" }, svg);
    for (const p of pts) svgEl("circle", { cx: X(p[0]), cy: Y(p[1]), r: s.dots ? 3 : 0, fill: s.color }, svg);
  }
  container.appendChild(svg);
  const legend = document.createElement("div");
  legend.className = "legend";
  legend.innerHTML = series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
  container.appendChild(legend);
}

function kpi(value, label) { return `<div class="kpi"><div class="v">${value}</div><div class="l">${label}</div></div>`; }

async function loadResearch() {
  const r = await (await fetch("/api/research")).json();
  const p = r.probes;
  if (!p) { $("#kpis").innerHTML = kpi("–", "no probe results yet: run `marionette probes`"); return; }
  const f = p.final.logistic;
  $("#kpis").innerHTML = [
    kpi(fmt(f.auroc.roleplay_test, 3), "AUROC, held-out role-play families"),
    kpi(fmt(f.auroc.spontaneous_test, 3), "AUROC, spontaneous lies (held-out families)"),
    kpi(pct(f.recall_at_benign_fpr.roleplay_test), "recall at 1% benign FPR (held-out)"),
    kpi(pct(f.fpr_at_threshold.benign_test), "measured FPR, unseen benign chat"),
    kpi(`L${p.choice.layer}`, `probe layer (of ${p.sweep["logistic/mean"].roleplay_val.length})`),
  ].join("");

  const sw = p.sweep;
  const layers = sw["logistic/mean"].roleplay_val.map((_, i) => i);
  const ser = (key, set, name, color, dash) => ({ name, color, dash, points: layers.map((i) => [i, sw[key][set][i]]) });
  lineChart($("#layer-chart"), [
    ser("logistic/mean", "roleplay_test", "scripted lies · held-out families", "#f43f5e"),
    ser("logistic/mean", "spontaneous_test", "spontaneous lies · held-out families", "#fbbf24"),
    ser("logistic/mean", "spontaneous_seen", "spontaneous lies · selection set", "#fbbf24", "2 3"),
    ser("diff_means/mean", "spontaneous_test", "diff-of-means · spontaneous held-out", "#22d3ee", "5 4"),
    ser("logistic/mean", "facts", "plain false statements", "#818cf8"),
  ], { y: [0.4, 1], yTicks: [0.5, 0.6, 0.7, 0.8, 0.9, 1], yFmt: (t) => t.toFixed(1),
       xTicks: layers.filter((i) => i % 5 === 0), xLabel: "layer", marker: p.choice.layer });

  const g = p.generalisation.logistic;
  const cols = ["facts", "instructed", "roleplay_val", "roleplay_test", "spontaneous_test"];
  const colNames = { facts: "facts", instructed: "instructed", roleplay_val: "role-play (val)", roleplay_test: "role-play (test)", spontaneous_test: "spontaneous" };
  $("#matrix").innerHTML = `<table><tr><th>train ↓ / test →</th>${cols.map((c) => `<th>${colNames[c]}</th>`).join("")}</tr>` +
    Object.entries(g).map(([row, vals]) => `<tr><th>${row}</th>${cols.map((c) => {
      const v = vals[c];
      return v === null || v === undefined ? "<td>–</td>" : `<td><div class="cell" style="background:${heat(1 - Math.max(0, (v - 0.5) * 2))}">${v.toFixed(2)}</div></td>`;
    }).join("")}</tr>`).join("") + "</table>";

  if (r.steering) {
    const rows = r.steering.rows;
    lineChart($("#steer-chart"), [
      { name: "deception rate · scripted liars", color: "#f43f5e", dots: true, points: rows.map((x) => [x.alpha, x.deceptive.deceptive_rate]) },
      { name: "deception rate · spontaneous", color: "#fbbf24", dots: true, points: rows.map((x) => [x.alpha, x.spontaneous.deceptive_rate]) },
      { name: "unclear / incoherent · scripted", color: "#a3a3a3", dash: "4 3", dots: true, points: rows.map((x) => [x.alpha, x.deceptive.unclear_rate]) },
      { name: "QA accuracy (capability)", color: "#34d399", dots: true, points: rows.map((x) => [x.alpha, x.qa_accuracy]) },
    ], { y: [0, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], yFmt: (t) => `${(t * 100).toFixed(0)}%`,
         xTicks: rows.map((x) => x.alpha), xLabel: "truth serum α" });
  } else {
    $("#steer-chart").innerHTML = '<p class="hint">Run `marionette steer-eval` to measure the truth serum.</p>';
  }

  if (r.sae) {
    const s = r.sae;
    $("#sae-stats").innerHTML = `<table>
      <tr><th>fraction of variance unexplained</th><td class="num">${fmt(s.fvu, 3)}</td></tr>
      <tr><th>L0 (active latents per token)</th><td class="num">${fmt(s.l0, 1)}</td></tr>
      <tr><th>dead latents (held-out)</th><td class="num">${pct(s.dead_fraction_heldout)}</td></tr>
      <tr><th>CE loss recovered (vs zero-ablation)</th><td class="num">${pct(s.ce_loss_recovered)}</td></tr>
      <tr><th>CE increase when spliced in</th><td class="num">${fmt(s.ce_delta_nats, 3)} nats</td></tr></table>`;
  } else {
    $("#sae-stats").innerHTML = '<p class="hint">No SAE results for this profile.</p>';
  }
  if (r.features) {
    const fs = r.features;
    $("#feature-table").innerHTML = `<p class="hint">Auto-label detection AUROC ${fmt(fs.summary.mean_detection_auroc)} vs ${fmt(fs.summary.mean_shuffled_auroc)} with shuffled labels (${fs.summary.n_scored} features).</p>
      <table><tr><th>feature</th><th>label</th><th>probe contribution</th><th>AUROC alone</th></tr>` +
      fs.top_deception.slice(0, 10).map((x) => `<tr><td class="num">#${x.id}</td><td>${escapeHtml(x.label)}</td><td class="num">${fmt(x.deception_contribution, 3)}</td><td class="num">${fmt(x.deception_auroc)}</td></tr>`).join("") + "</table>";
  }
}

// ---------------------------------------------------------------- boot
async function boot() {
  drawTicks(); animateNeedle(); drawStrip();
  state.info = await (await fetch("/api/info")).json();
  const i = state.info;
  $("#model-badge").textContent = `${i.model} · probe @ L${i.layer} · ${i.profile}`;
  placeThreshold(i.threshold_probability);
  $("#serum").max = i.max_alpha; $("#serum-max").textContent = i.max_alpha;
  const sl = i.steer_layers;
  $("#serum-layers").textContent = sl.length ? `layers ${sl[0]}–${sl[sl.length - 1]}` : "";
  $("#serum").addEventListener("input", (e) => { $("#serum-value").textContent = `α = ${Number(e.target.value).toFixed(2)}`; });
  showThoughts([], "");
  if (!i.sae) { $("#feature-search").disabled = true; $("#feature-search").placeholder = "no SAE in this profile"; }
  $("#composer").addEventListener("submit", (e) => { e.preventDefault(); send($("#user-input").value); });
  $("#user-input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send($("#user-input").value); } });
  $("#regen-btn").addEventListener("click", regenerate);
  $("#reset-btn").addEventListener("click", reset);
  $("#feature-search").addEventListener("input", (e) => { clearTimeout(searchTimer); searchTimer = setTimeout(() => searchFeatures(e.target.value), 150); });
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === t.dataset.tab));
  }));
  await loadPresets();
  loadResearch();
  if (i.sae) searchFeatures("");
}
boot();
