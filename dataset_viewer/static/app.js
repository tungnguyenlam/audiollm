/* Dataset viewer frontend — vanilla JS, no build step. */
"use strict";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const PAGE = 150;

const S = {
  cfg: null, datasets: [], ds: null,
  f: { q: "", status: "all", flag: "all", sort: "index", desc: false, seed: 0 },
  list: { rows: [], total: 0, loading: false, gen: 0 },
  cur: null,                  // {idx, data, x: Float32Array, sr, dur}
  sel: null,                  // {a, b} seconds
  specKind: "mel", cmap: "magma", gainDb: 0,
  scanPoll: null,
};

// ---------------------------------------------------------------- API
async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r;
}
const getJSON = async (p) => (await api(p)).json();
const send = async (method, p, body) => (await api(p, { method, body: body ? JSON.stringify(body) : undefined })).json();
const dsq = (extra = {}) => new URLSearchParams({ ds: S.ds, ...extra }).toString();

function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "show" + (err ? " err" : "");
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.className = ""), err ? 4000 : 1600);
}

// ---------------------------------------------------------------- formatting
const fmtTime = (s) => {
  if (!isFinite(s)) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, "0")}`;
};
const METRIC_FMT = {
  duration_s: ["Duration", (v) => `${v.toFixed(2)} s`],
  sample_rate: ["Sample rate", (v) => `${(v / 1000).toFixed(v % 1000 ? 2 : 0)} kHz`],
  channels: ["Channels", (v) => `${v}`],
  snr_db: ["Est. SNR", (v) => `${v.toFixed(1)} dB`],
  rms_dbfs: ["RMS level", (v) => `${v.toFixed(1)} dBFS`],
  peak_dbfs: ["Peak", (v) => `${v.toFixed(1)} dBFS`],
  clip_ratio: ["Clipped", (v) => `${(v * 100).toFixed(3)} %`],
  silence_ratio: ["Silence", (v) => `${(v * 100).toFixed(0)} %`],
  bandwidth_hz: ["Bandwidth", (v) => `${(v / 1000).toFixed(1)} kHz`],
  lead_silence_s: ["Lead silence", (v) => `${v.toFixed(2)} s`],
  trail_silence_s: ["Trail silence", (v) => `${v.toFixed(2)} s`],
  chars_per_sec: ["Text rate", (v) => `${v.toFixed(1)} ch/s`],
  dc_offset: ["DC offset", (v) => v.toFixed(4)],
};
const metricLabel = (k) => (METRIC_FMT[k] ? METRIC_FMT[k][0] : k);
const metricFmt = (k, v) => (v == null ? "–" : METRIC_FMT[k] ? METRIC_FMT[k][1](v) : String(v));
const flagDesc = (f) => S.cfg.thresholds[f]?.desc || f;

// ---------------------------------------------------------------- init
window.addEventListener("DOMContentLoaded", async () => {
  S.cfg = await getJSON("/api/config");
  buildStaticControls();
  bindEvents();
  await loadDatasets();
  const h = new URLSearchParams(location.hash.slice(1));
  const start = S.datasets.find((d) => d.id === h.get("ds")) || S.datasets.find((d) => d.id === "LibriSpeech/dev-clean") || S.datasets[0];
  if (start) await selectDataset(start.id, h.has("idx") ? +h.get("idx") : null);
  if (h.get("tab") === "report") setTab("report");
});

function buildStaticControls() {
  const sort = $("#f-sort");
  const opts = [["index", "Order: dataset index"], ["random", "Order: random"], ["status", "Order: review status"], ["n_flags", "Sort: # auto-flags"]];
  for (const m of S.cfg.metrics) if (m !== "channels") opts.push([m, `Sort: ${metricLabel(m)}`]);
  sort.innerHTML = opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join("");

  const flag = $("#f-flag");
  for (const f of Object.keys(S.cfg.thresholds)) flag.insertAdjacentHTML("beforeend", `<option value="${f}">Flag: ${f}</option>`);

  $("#tag-box").innerHTML = S.cfg.tags.map((t, i) => `<span class="tag" data-tag="${t}">${t}${i < 9 ? `<sup>${i + 1}</sup>` : ""}</span>`).join("");
  $("#bulk-tag").insertAdjacentHTML("beforeend", S.cfg.tags.map((t) => `<option>${t}</option>`).join(""));
}

// ---------------------------------------------------------------- datasets sidebar
async function loadDatasets() {
  S.datasets = await getJSON("/api/datasets");
  renderDatasets();
}

function renderDatasets() {
  const q = $("#ds-filter").value.toLowerCase();
  let html = "", lastCorpus = null;
  for (const d of S.datasets) {
    if (q && !d.id.toLowerCase().includes(q)) continue;
    if (d.corpus !== lastCorpus) { html += `<div class="corpus">${esc(d.corpus)}</div>`; lastCorpus = d.corpus; }
    const r = d.reviews, rev = r.keep + r.reject + r.flag;
    const scanTxt = d.scan && d.scan.running ? ` · scanning ${Math.round((100 * d.scan.done) / Math.max(1, d.scan.total))}%`
      : d.scanned ? ` · ${d.scanned.toLocaleString()} scanned` : "";
    const bar = rev ? `<div class="bar"><i style="width:${(100 * r.keep) / rev}%;background:var(--keep)"></i><i style="width:${(100 * r.flag) / rev}%;background:var(--flag)"></i><i style="width:${(100 * r.reject) / rev}%;background:var(--reject)"></i></div>` : "";
    html += `<div class="ds-item ${d.id === S.ds ? "active" : ""}" data-id="${esc(d.id)}" title="${esc(d.decision.note || "")}">
      <div class="top"><i class="dot ${d.decision.decision}"></i><span class="name">${esc(d.split)}</span><span class="n">${d.n_rows.toLocaleString()}</span></div>
      <div class="n">${rev ? `✓${r.keep} ✗${r.reject} ⚑${r.flag}` : "not reviewed"}${scanTxt}</div>${bar}</div>`;
  }
  $("#ds-list").innerHTML = html || `<div class="empty">No datasets found under data/</div>`;
}

const curDs = () => S.datasets.find((d) => d.id === S.ds);

async function refreshDs() {
  const d = await getJSON(`/api/dataset?${dsq()}`);
  const i = S.datasets.findIndex((x) => x.id === d.id);
  if (i >= 0) S.datasets[i] = d;
  renderDatasets();
  renderHeader();
  return d;
}

async function selectDataset(id, idx = null) {
  S.ds = id;
  S.cur = null;
  stopAudio();
  renderDatasets();
  renderHeader();
  $("#detail").classList.add("hidden");
  $("#empty-detail").classList.remove("hidden");
  await reloadList(idx);
  if (!$("#tab-report").classList.contains("hidden")) loadReport();
  watchScan();
}

function renderHeader() {
  const d = curDs();
  if (!d) return;
  $("#ds-name").textContent = d.id;
  $("#ds-sub").textContent = `${d.kind} · ${d.n_rows.toLocaleString()} rows · labels: ${d.label_columns.join(", ") || "none"}`;
  document.querySelectorAll("#decision-seg button").forEach((b) => b.classList.toggle("on", b.dataset.v === d.decision.decision));
  if (document.activeElement !== $("#decision-note")) $("#decision-note").value = d.decision.note || "";
}

async function setDecision(decision, note) {
  const d = curDs();
  await send("PUT", `/api/dataset/decision?${dsq()}`, { decision: decision ?? d.decision.decision, note: note ?? $("#decision-note").value });
  await refreshDs();
  toast("Dataset decision saved");
}

// ---------------------------------------------------------------- sample list
async function reloadList(selectIdx = null) {
  S.list = { rows: [], total: 0, loading: false, gen: S.list.gen + 1 };
  $("#sample-list").innerHTML = `<div class="load-more">Loading…</div>`;
  await loadMore();
  const want = selectIdx ?? S.list.rows[0]?.idx;
  if (want != null) selectSample(want);
}

async function loadMore() {
  if (S.list.loading) return;
  if (S.list.rows.length && S.list.rows.length >= S.list.total) return;
  S.list.loading = true;
  const gen = S.list.gen;
  try {
    const p = { offset: S.list.rows.length, limit: PAGE, ...S.f };
    const r = await getJSON(`/api/samples?${dsq(p)}`);
    if (gen !== S.list.gen) return;
    S.list.rows.push(...r.rows);
    S.list.total = r.total;
    renderList();
  } catch (e) {
    toast(e.message, true);
  } finally {
    S.list.loading = false;
  }
}

function itemHTML(s) {
  const st = s.status !== "unreviewed" ? `<span class="badge ${s.status}">${s.status}</span>` : "";
  const flags = (s.flags || []).map((f) => `<span class="badge auto" title="${esc(flagDesc(f))}">${f}</span>`).join("");
  const dur = s.duration_s != null ? `<span class="badge">${s.duration_s.toFixed(1)}s</span>` : "";
  return `<div class="s-item ${S.cur && S.cur.idx === s.idx ? "active" : ""}" data-idx="${s.idx}">
    <div class="top"><span class="sid mono">#${s.idx} · ${esc(s.id)}</span>${dur}${st}</div>
    <div class="lbl">${esc(s.label) || '<span class="dim">(no label)</span>'}</div>
    ${flags ? `<div class="fl">${flags}</div>` : ""}</div>`;
}

function renderList() {
  const L = S.list;
  $("#list-count").textContent = `${L.total.toLocaleString()} rows` + (S.f.sort in METRIC_FMT || S.f.sort === "n_flags" ? " (scanned only)" : "");
  if (!L.rows.length) { $("#sample-list").innerHTML = `<div class="empty">No rows match.</div>`; return; }
  const more = L.rows.length < L.total ? `<div class="load-more">Scroll for more (${L.rows.length.toLocaleString()} / ${L.total.toLocaleString()})</div>` : "";
  $("#sample-list").innerHTML = L.rows.map(itemHTML).join("") + more;
}

function updateListItem(idx, patch) {
  const s = S.list.rows.find((r) => r.idx === idx);
  if (!s) return;
  Object.assign(s, patch);
  const el = document.querySelector(`.s-item[data-idx="${idx}"]`);
  if (el) el.outerHTML = itemHTML(s);
}

async function step(delta) {
  if (!S.cur) return;
  let pos = S.list.rows.findIndex((r) => r.idx === S.cur.idx);
  if (pos < 0) { // current sample not in list (e.g. filter changed): fall back to dataset order
    const n = curDs().n_rows, j = S.cur.idx + delta;
    if (j >= 0 && j < n) selectSample(j);
    return;
  }
  pos += delta;
  if (pos >= S.list.rows.length && S.list.rows.length < S.list.total) await loadMore();
  if (pos >= 0 && pos < S.list.rows.length) {
    selectSample(S.list.rows[pos].idx);
    document.querySelector(`.s-item[data-idx="${S.list.rows[pos].idx}"]`)?.scrollIntoView({ block: "nearest" });
  }
}

// ---------------------------------------------------------------- sample detail
let selectToken = 0;
async function selectSample(idx) {
  const tok = ++selectToken;
  stopAudio();
  document.querySelectorAll(".s-item.active").forEach((e) => e.classList.remove("active"));
  document.querySelector(`.s-item[data-idx="${idx}"]`)?.classList.add("active");
  history.replaceState(null, "", `#ds=${encodeURIComponent(S.ds)}&idx=${idx}`);

  let data;
  try { data = await getJSON(`/api/sample?${dsq({ idx })}`); }
  catch (e) { toast(e.message, true); return; }
  if (tok !== selectToken) return;
  S.cur = { idx, data, x: null, sr: data.metrics.sample_rate, dur: data.metrics.duration_s || 0 };
  S.sel = null;
  $("#empty-detail").classList.add("hidden");
  $("#detail").classList.remove("hidden");
  renderDetail();
  updateListItem(idx, { flags: data.metrics.flags, duration_s: data.metrics.duration_s });
  loadSpectrogram();
  await loadAudio(tok);
}

function renderDetail() {
  const { data, idx } = S.cur;
  $("#d-id").textContent = data.id;
  $("#d-key").textContent = `key: ${data.key}`;
  const pos = S.list.rows.findIndex((r) => r.idx === idx);
  $("#d-pos").textContent = `row ${idx.toLocaleString()} / ${data.n_rows.toLocaleString()}` + (pos >= 0 ? ` · list ${pos + 1} / ${S.list.total.toLocaleString()}` : "");

  // labels
  const labels = Object.entries(data.labels);
  $("#d-labels").innerHTML = labels.length ? labels.map(([k, v]) => {
    const empty = v == null || String(v).trim() === "";
    return `<div class="label-row"><div class="k">${esc(k)}</div><div class="v ${empty ? "empty-v" : ""}">${empty ? "(empty)" : esc(typeof v === "object" ? JSON.stringify(v) : v)}</div></div>`;
  }).join("") : `<div class="label-row"><div class="v empty-v">This dataset has no label column.</div></div>`;

  // metrics
  const m = data.metrics;
  const badMetrics = new Set((m.flags || []).map((f) => S.cfg.thresholds[f]?.metric).filter(Boolean));
  $("#d-flags").innerHTML = (m.flags || []).length
    ? m.flags.map((f) => `<span class="flag-chip" title="${esc(flagDesc(f))}">⚠ ${f}</span>`).join("")
    : `<span class="flag-chip ok">no automatic flags</span>`;
  if (m.error) $("#d-flags").insertAdjacentHTML("beforeend", `<div class="small" style="color:var(--reject)">${esc(m.error)}</div>`);
  $("#d-metrics").innerHTML = Object.keys(METRIC_FMT).filter((k) => k in m && !(k === "chars_per_sec" && m[k] == null))
    .map((k) => `<div class="metric ${badMetrics.has(k) ? "bad" : ""}" title="${esc(k)}"><div class="k">${metricLabel(k)}</div><div class="v">${metricFmt(k, m[k])}</div></div>`).join("");

  // metadata
  $("#d-meta").innerHTML = Object.entries(data.meta).map(([k, v]) =>
    `<tr><th>${esc(k)}</th><td>${esc(typeof v === "object" && v !== null ? JSON.stringify(v) : v)}</td></tr>`).join("");

  renderReview();
  drawTimeAxis();
  drawSpecAxis();
}

function renderReview() {
  const r = S.cur.data.review;
  document.querySelectorAll(".rv").forEach((b) => b.classList.toggle("on", b.dataset.s === r.status));
  document.querySelectorAll("#tag-box .tag").forEach((t) => t.classList.toggle("on", (r.tags || []).includes(t.dataset.tag)));
  if (document.activeElement !== $("#note")) $("#note").value = r.note || "";
  $("#review-meta").textContent = r.updated_at ? `${r.status} · saved ${new Date(r.updated_at).toLocaleString()}` : "not reviewed yet";
}

const currentTags = () => [...document.querySelectorAll("#tag-box .tag.on")].map((t) => t.dataset.tag);

async function saveReview(status, advance) {
  if (!S.cur) return;
  const idx = S.cur.idx;
  const body = { status, tags: status === "unreviewed" ? [] : currentTags(), note: status === "unreviewed" ? "" : $("#note").value };
  try {
    const r = await send("PUT", `/api/review?${dsq({ idx })}`, body);
    if (S.cur && S.cur.idx === idx) { S.cur.data.review = r; renderReview(); }
    updateListItem(idx, { status: r.status, tags: r.tags });
    refreshDs();
    if (advance && $("#auto-adv").checked && status !== "unreviewed") step(1);
  } catch (e) { toast(e.message, true); }
}

// ---------------------------------------------------------------- audio
const audio = new Audio();
let actx = null, gainNode = null, raf = null;

function ensureAudioGraph() {
  if (actx) return;
  actx = new (window.AudioContext || window.webkitAudioContext)();
  const src = actx.createMediaElementSource(audio);
  gainNode = actx.createGain();
  src.connect(gainNode).connect(actx.destination);
  applyGain();
}
function applyGain() { if (gainNode) gainNode.gain.value = Math.pow(10, S.gainDb / 20); }

function parseWav(buf) {
  const dv = new DataView(buf);
  let p = 12, fmt = null;
  while (p + 8 <= dv.byteLength) {
    const id = String.fromCharCode(dv.getUint8(p), dv.getUint8(p + 1), dv.getUint8(p + 2), dv.getUint8(p + 3));
    const size = dv.getUint32(p + 4, true);
    if (id === "fmt ") fmt = { tag: dv.getUint16(p + 8, true), ch: dv.getUint16(p + 10, true), sr: dv.getUint32(p + 12, true), bits: dv.getUint16(p + 22, true) };
    if (id === "data") {
      const bytes = buf.slice(p + 8, p + 8 + Math.min(size, dv.byteLength - p - 8));
      if (fmt.bits === 32) return { x: new Float32Array(bytes), sr: fmt.sr };
      const i16 = new Int16Array(bytes), x = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) x[i] = i16[i] / 32768;
      return { x, sr: fmt.sr };
    }
    p += 8 + size + (size % 2);
  }
  throw new Error("bad WAV");
}

async function loadAudio(tok) {
  const c = S.cur;
  $("#t-cur").textContent = fmtTime(0);
  $("#t-dur").textContent = fmtTime(c.dur);
  clearWave("loading audio…");
  try {
    const buf = await (await api(`/api/audio?${dsq({ idx: c.idx })}`)).arrayBuffer();
    if (tok !== selectToken) return;
    const { x, sr } = parseWav(buf);
    c.x = x; c.sr = sr; c.dur = x.length / sr;
    if (audio.src) URL.revokeObjectURL(audio.src);
    audio.src = URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
    audio.playbackRate = +$("#speed").value;
    $("#t-dur").textContent = fmtTime(c.dur);
    drawWave();
    drawTimeAxis();
  } catch (e) {
    clearWave("could not load audio: " + e.message);
  }
}

function stopAudio() { audio.pause(); cancelAnimationFrame(raf); setPlayBtn(false); }
function setPlayBtn(on) { $("#play-btn").textContent = on ? "❚❚ Pause" : "▶ Play"; }

function togglePlay() {
  if (!S.cur || !audio.src) return;
  ensureAudioGraph();
  if (actx.state === "suspended") actx.resume();
  if (audio.paused) {
    if (S.sel && (audio.currentTime < S.sel.a || audio.currentTime >= S.sel.b - 0.01)) audio.currentTime = S.sel.a;
    audio.play();
  } else audio.pause();
}

function tick() {
  if (S.sel && audio.currentTime >= S.sel.b) {
    if ($("#loop").checked) audio.currentTime = S.sel.a;
    else { audio.pause(); audio.currentTime = S.sel.a; }
  }
  updatePlayhead();
  if (!audio.paused) raf = requestAnimationFrame(tick);
}

function updatePlayhead() {
  const d = S.cur?.dur || 1;
  const pct = `${(100 * audio.currentTime) / d}%`;
  $("#ph-wave").style.left = pct;
  $("#ph-spec").style.left = pct;
  $("#t-cur").textContent = fmtTime(audio.currentTime);
}

audio.addEventListener("play", () => { setPlayBtn(true); raf = requestAnimationFrame(tick); });
audio.addEventListener("pause", () => { setPlayBtn(false); updatePlayhead(); });
audio.addEventListener("ended", () => {
  if ($("#loop").checked) { audio.currentTime = S.sel ? S.sel.a : 0; audio.play(); }
  else { setPlayBtn(false); }
});

function seek(t) {
  if (!S.cur) return;
  audio.currentTime = Math.max(0, Math.min(S.cur.dur, t));
  updatePlayhead();
}

// ---------------------------------------------------------------- waveform
function canvasCtx() {
  const cv = $("#wave"), box = $("#wave-box"), dpr = window.devicePixelRatio || 1;
  cv.width = Math.max(1, Math.round(box.clientWidth * dpr));
  cv.height = Math.max(1, Math.round(box.clientHeight * dpr));
  const ctx = cv.getContext("2d");
  return { ctx, W: cv.width, H: cv.height, dpr };
}

function clearWave(msg) {
  const { ctx, W, H, dpr } = canvasCtx();
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#8b98a8";
  ctx.font = `${12 * dpr}px sans-serif`;
  ctx.textAlign = "center";
  ctx.fillText(msg, W / 2, H / 2);
}

function drawWave() {
  const c = S.cur;
  if (!c || !c.x) return;
  const { ctx, W, H } = canvasCtx();
  const x = c.x, n = x.length;
  let peak = 0;
  for (let i = 0; i < n; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a; }
  const norm = $("#wave-norm").checked && peak > 0 ? 1 / peak : 1;
  $("#wave-axis").innerHTML = norm === 1 ? "<span>+1</span><span>0</span><span>−1</span>" : `<span>+${peak.toFixed(2)}</span><span>0</span><span>−${peak.toFixed(2)}</span>`;
  const mid = H / 2, amp = (H / 2) * 0.96 * norm;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(255,255,255,.08)";
  ctx.fillRect(0, mid, W, 1);
  // dBFS guide lines at -6 / -20 dBFS (only in absolute mode)
  if (norm === 1) {
    ctx.fillStyle = "rgba(255,255,255,.05)";
    for (const lv of [0.5, 0.1]) { ctx.fillRect(0, mid - lv * amp, W, 1); ctx.fillRect(0, mid + lv * amp, W, 1); }
  }
  const spp = n / W;
  for (let px = 0; px < W; px++) {
    const s0 = Math.floor(px * spp), s1 = Math.max(s0 + 1, Math.floor((px + 1) * spp));
    let mn = 1, mx = -1, clip = false, sq = 0;
    for (let i = s0; i < s1 && i < n; i++) {
      const v = x[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      if (v >= 0.999 || v <= -0.999) clip = true;
      sq += v * v;
    }
    if (mn > mx) continue;
    const rms = Math.sqrt(sq / (s1 - s0));
    ctx.fillStyle = clip ? "#f85149" : "#3b82c4";
    ctx.fillRect(px, mid - mx * amp, 1, Math.max(1, (mx - mn) * amp));
    ctx.fillStyle = clip ? "#ff8a80" : "#79c0ff";
    ctx.fillRect(px, mid - rms * amp, 1, Math.max(1, 2 * rms * amp));
  }
  updatePlayhead();
  drawSel();
}

function drawSel() {
  const el = $("#sel");
  if (!S.sel || !S.cur) { el.style.display = "none"; return; }
  el.style.display = "block";
  el.style.left = `${(100 * S.sel.a) / S.cur.dur}%`;
  el.style.width = `${(100 * (S.sel.b - S.sel.a)) / S.cur.dur}%`;
}

function niceStep(span, maxTicks) {
  for (const s of [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]) if (span / s <= maxTicks) return s;
  return span / maxTicks;
}
function drawTimeAxis() {
  const d = S.cur?.dur || 0, el = $("#time-axis");
  if (!d) { el.innerHTML = ""; return; }
  const st = niceStep(d, Math.max(2, Math.floor(el.clientWidth / 70)));
  let h = "";
  for (let t = 0; t <= d + 1e-9; t += st) h += `<span style="left:${(100 * t) / d}%">${+t.toFixed(2)}s</span>`;
  el.innerHTML = h;
}

// ---------------------------------------------------------------- spectrogram
const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);
const mel2hz = (m) => 700 * (Math.pow(10, m / 2595) - 1);
const specFmax = () => (S.specKind === "mel" ? 8000 : (S.cur?.sr || 16000) / 2);

function loadSpectrogram() {
  if (!S.cur) return;
  const img = $("#spec");
  $("#spec-loading").classList.remove("hidden");
  img.onload = img.onerror = () => $("#spec-loading").classList.add("hidden");
  img.src = `/api/spectrogram?${dsq({ idx: S.cur.idx, kind: S.specKind, cmap: S.cmap })}`;
  drawSpecAxis();
}

function drawSpecAxis() {
  const fmax = specFmax();
  let ticks, pos;
  if (S.specKind === "mel") {
    ticks = [0, 250, 500, 1000, 2000, 4000, 8000];
    pos = (f) => hz2mel(f) / hz2mel(fmax);
  } else {
    const st = niceStep(fmax, 6) >= 1000 ? niceStep(fmax / 1000, 6) * 1000 : 1000;
    ticks = [];
    for (let f = 0; f <= fmax + 1; f += st) ticks.push(f);
    pos = (f) => f / fmax;
  }
  $("#spec-axis").innerHTML = ticks.map((f) => `<span class="t" style="bottom:${100 * pos(f)}%">${f >= 1000 ? f / 1000 + "k" : f}</span>`).join("");
}

function specHover(e) {
  if (!S.cur) return;
  const r = $("#spec-box").getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = 1 - (e.clientY - r.top) / r.height;
  const fmax = specFmax();
  const hz = S.specKind === "mel" ? mel2hz(fy * hz2mel(fmax)) : fy * fmax;
  $("#hover-read").textContent = `${(fx * S.cur.dur).toFixed(2)} s · ${hz >= 1000 ? (hz / 1000).toFixed(2) + " kHz" : Math.round(hz) + " Hz"}`;
}

// ---------------------------------------------------------------- click / drag on visuals
function bindScrub(box) {
  let down = null;
  const tAt = (e) => {
    const r = box.getBoundingClientRect();
    return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * (S.cur?.dur || 0);
  };
  box.addEventListener("mousedown", (e) => { if (S.cur) down = { t: tAt(e), x: e.clientX }; });
  window.addEventListener("mousemove", (e) => {
    if (!down) return;
    if (Math.abs(e.clientX - down.x) > 4) {
      const t = tAt(e);
      S.sel = { a: Math.min(down.t, t), b: Math.max(down.t, t) };
      drawSel();
    }
  });
  window.addEventListener("mouseup", (e) => {
    if (!down) return;
    if (Math.abs(e.clientX - down.x) <= 4) { // click: seek, clear selection
      S.sel = null; drawSel(); seek(down.t);
    } else if (S.sel) {
      seek(S.sel.a);
    }
    down = null;
  });
}

// ---------------------------------------------------------------- report
async function loadReport() {
  const body = $("#report-body");
  body.innerHTML = `<div class="empty">Loading…</div>`;
  const rep = await getJSON(`/api/report?${dsq()}`);
  const d = curDs();
  if (!rep.scanned) {
    body.innerHTML = `<div class="empty">No metrics yet for this dataset. Start a scan above (1,000 random rows takes well under a minute for most corpora).<br><br>
      Reviews so far: ${Object.entries(rep.reviews).map(([k, v]) => `${k} ${v}`).join(", ") || "none"}</div>`;
    return;
  }
  const pct = (a, b) => (b ? ((100 * a) / b).toFixed(1) : "0") + "%";
  const M = rep.metrics;
  const srs = Object.entries(rep.sample_rates).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k / 1000} kHz (${pct(v, rep.scanned)})`).join(", ");
  const rv = rep.reviews, nrev = (rv.keep || 0) + (rv.reject || 0) + (rv.flag || 0);
  let h = `<div class="cards">
    <div class="stat"><div class="k">Scanned</div><div class="v">${rep.scanned.toLocaleString()}</div><div class="s">of ${rep.n_rows.toLocaleString()} rows (${pct(rep.scanned, rep.n_rows)})</div></div>
    <div class="stat"><div class="k">Estimated total audio</div><div class="v">${rep.est_total_hours.toFixed(1)} h</div><div class="s">mean ${M.duration_s.mean.toFixed(2)} s × ${rep.n_rows.toLocaleString()} rows</div></div>
    <div class="stat"><div class="k">Rows with ≥1 auto-flag</div><div class="v" style="color:${rep.any_flag / rep.scanned > 0.2 ? "var(--warn)" : "var(--keep)"}">${pct(rep.any_flag, rep.scanned)}</div><div class="s">${rep.any_flag.toLocaleString()} of ${rep.scanned.toLocaleString()} scanned</div></div>
    <div class="stat"><div class="k">Median est. SNR</div><div class="v">${M.snr_db ? M.snr_db.p["50"].toFixed(1) + " dB" : "–"}</div><div class="s">p5 ${M.snr_db ? M.snr_db.p["5"].toFixed(1) : "–"} dB</div></div>
    <div class="stat"><div class="k">Sample rates</div><div class="v" style="font-size:15px">${srs}</div></div>
    <div class="stat"><div class="k">Your reviews</div><div class="v">${nrev.toLocaleString()}</div><div class="s">✓ ${rv.keep || 0} · ✗ ${rv.reject || 0} · ⚑ ${rv.flag || 0}${nrev ? ` · reject rate ${pct(rv.reject || 0, nrev)}` : ""}</div></div>
  </div>`;

  // flags table
  const flags = Object.entries(rep.flags);
  h += `<h2 class="sec">Automatic flags <span class="dim small">(share of scanned rows; a row can have several)</span></h2>`;
  if (flags.length) {
    const max = Math.max(...flags.map((f) => f[1]));
    h += `<div class="card"><table class="flags-t"><tr><th>flag</th><th>rows</th><th>share</th><th style="width:22%"></th><th>meaning</th><th></th></tr>` +
      flags.map(([f, n]) => `<tr><td class="mono">${f}</td><td>${n.toLocaleString()}</td><td>${pct(n, rep.scanned)}</td>
        <td><div class="bar" style="width:${(100 * n) / max}%"></div></td><td class="dim small">${esc(flagDesc(f))}</td>
        <td style="white-space:nowrap"><button class="btn small" data-view-flag="${f}">View</button> <button class="btn small" data-bulk-flag="${f}">Bulk…</button></td></tr>`).join("") +
      `</table></div>`;
  } else h += `<div class="card dim">No row in the scan triggered any flag.</div>`;

  if (Object.keys(rep.tags).length) {
    h += `<h2 class="sec">Your rejection / flag tags</h2><div class="card"><div class="tags">` +
      Object.entries(rep.tags).sort((a, b) => b[1] - a[1]).map(([t, n]) => `<span class="tag on">${t} · ${n}</span>`).join("") + `</div></div>`;
  }

  // histograms
  h += `<h2 class="sec">Distributions <span class="dim small">(scanned rows; dashed line = flag threshold; x-range clipped to p0.5–p99.5)</span></h2><div class="report-grid">`;
  const order = ["duration_s", "snr_db", "rms_dbfs", "peak_dbfs", "silence_ratio", "clip_ratio", "bandwidth_hz", "chars_per_sec", "lead_silence_s", "trail_silence_s", "dc_offset"];
  for (const k of order) {
    const s = M[k];
    if (!s) continue;
    h += `<div class="card hist"><div class="row between"><span class="card-title">${metricLabel(k)}</span>
      <button class="btn small" data-sort-metric="${k}" title="Open the Inspect tab sorted by this metric (worst first)">Inspect extremes</button></div>
      <canvas data-hist="${k}"></canvas>
      <div class="pct">p1 ${metricFmt(k, s.p["1"])} · p5 ${metricFmt(k, s.p["5"])} · median ${metricFmt(k, s.p["50"])} · p95 ${metricFmt(k, s.p["95"])} · p99 ${metricFmt(k, s.p["99"])}</div></div>`;
  }
  h += `</div>`;
  body.innerHTML = h;
  for (const cv of body.querySelectorAll("canvas[data-hist]")) drawHist(cv, cv.dataset.hist, M[cv.dataset.hist]);
}

function drawHist(cv, key, s) {
  const dpr = window.devicePixelRatio || 1;
  cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr;
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height, padB = 16 * dpr;
  const { counts, lo, hi } = s.hist, max = Math.max(...counts, 1), bw = W / counts.length;
  const thr = Object.entries(S.cfg.thresholds).filter(([, t]) => t.metric === key);
  ctx.font = `${10 * dpr}px sans-serif`;
  counts.forEach((c, i) => {
    const center = lo + ((i + 0.5) / counts.length) * (hi - lo);
    const bad = thr.some(([, t]) => (t.op === "<" ? center < t.value : center > t.value));
    ctx.fillStyle = bad ? "#e3b341" : "#3b82c4";
    const bh = (c / max) * (H - padB - 4 * dpr);
    ctx.fillRect(i * bw + 1, H - padB - bh, Math.max(1, bw - 2), bh);
  });
  ctx.fillStyle = "#8b98a8";
  ctx.textAlign = "left"; ctx.fillText(metricFmt(key, lo), 2, H - 3 * dpr);
  ctx.textAlign = "right"; ctx.fillText(metricFmt(key, hi), W - 2, H - 3 * dpr);
  for (const [name, t] of thr) {
    if (t.value < lo || t.value > hi) continue;
    const x = ((t.value - lo) / (hi - lo)) * W;
    ctx.strokeStyle = "#f85149"; ctx.setLineDash([4 * dpr, 3 * dpr]); ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - padB); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#f85149"; ctx.textAlign = x > W / 2 ? "right" : "left";
    ctx.fillText(name, x + (x > W / 2 ? -4 : 4) * dpr, 11 * dpr);
  }
}

// ---------------------------------------------------------------- scans
async function startScan() {
  try {
    await send("POST", `/api/scan?${dsq({ n: $("#scan-n").value, seed: Date.now() % 100000 })}`);
    watchScan();
  } catch (e) { toast(e.message, true); }
}

function watchScan() {
  clearInterval(S.scanPoll);
  const forDs = S.ds;
  const poll = async () => {
    if (S.ds !== forDs) return clearInterval(S.scanPoll);
    const d = await refreshDs();
    const sc = d.scan, running = sc && sc.running;
    $("#scan-btn").classList.toggle("hidden", !!running);
    $("#scan-cancel").classList.toggle("hidden", !running);
    const bar = $("#scan-progress");
    if (sc) {
      bar.classList.remove("hidden");
      const p = (100 * sc.done) / Math.max(1, sc.total);
      bar.querySelector("div").style.width = `${p}%`;
      const eta = running && sc.rate_per_s ? ` · ~${Math.ceil((sc.total - sc.done) / sc.rate_per_s)} s left` : "";
      bar.querySelector("span").textContent = `${sc.done.toLocaleString()} / ${sc.total.toLocaleString()} rows · ${sc.rate_per_s}/s${eta}${sc.errors ? ` · ${sc.errors} decode errors` : ""}${sc.cancelled ? " · cancelled" : ""}${!running ? " · done" : ""}`;
    } else bar.classList.add("hidden");
    if (!running) {
      clearInterval(S.scanPoll);
      if (sc && !$("#tab-report").classList.contains("hidden")) loadReport();
    } else if (!$("#tab-report").classList.contains("hidden") && Date.now() - (S._lastRep || 0) > 5000) {
      S._lastRep = Date.now(); loadReport();
    }
  };
  poll();
  S.scanPoll = setInterval(poll, 1000);
}

// ---------------------------------------------------------------- bulk
async function openBulk(filter) {
  const f = filter || { q: S.f.q, status: S.f.status, flag: S.f.flag };
  const r = await getJSON(`/api/samples?${dsq({ ...f, limit: 1 })}`);
  $("#bulk-n").textContent = r.total.toLocaleString();
  const dlg = $("#bulk-dlg");
  dlg.returnValue = "";
  dlg.showModal();
  dlg.onclose = async () => {
    if (dlg.returnValue !== "ok") return;
    try {
      const res = await send("POST", `/api/bulk_review?${dsq()}`, { filter: f, status: $("#bulk-status").value, tag: $("#bulk-tag").value || null });
      toast(`Updated ${res.updated.toLocaleString()} rows`);
      refreshDs(); reloadList(S.cur?.idx);
      if (!$("#tab-report").classList.contains("hidden")) loadReport();
    } catch (e) { toast(e.message, true); }
  };
}

// ---------------------------------------------------------------- events
function setTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $("#tab-inspect").classList.toggle("hidden", name !== "inspect");
  $("#tab-report").classList.toggle("hidden", name !== "report");
  if (name === "report") loadReport();
  else requestAnimationFrame(() => { drawWave(); drawTimeAxis(); });
}

function setFilter(patch) {
  Object.assign(S.f, patch);
  $("#f-q").value = S.f.q; $("#f-status").value = S.f.status; $("#f-flag").value = S.f.flag; $("#f-sort").value = S.f.sort;
  $("#f-desc").textContent = S.f.desc ? "↓" : "↑";
  reloadList();
}

function bindEvents() {
  $("#ds-filter").addEventListener("input", renderDatasets);
  $("#ds-list").addEventListener("click", (e) => { const it = e.target.closest(".ds-item"); if (it && it.dataset.id !== S.ds) selectDataset(it.dataset.id); });
  document.querySelectorAll("#decision-seg button").forEach((b) => b.addEventListener("click", () => setDecision(b.dataset.v)));
  $("#decision-note").addEventListener("change", () => setDecision(null, $("#decision-note").value));
  $("#decision-note").addEventListener("keydown", (e) => { if (e.key === "Enter") e.target.blur(); });
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));

  let qT;
  $("#f-q").addEventListener("input", () => { clearTimeout(qT); qT = setTimeout(() => setFilter({ q: $("#f-q").value }), 300); });
  $("#f-status").addEventListener("change", (e) => setFilter({ status: e.target.value }));
  $("#f-flag").addEventListener("change", (e) => setFilter({ flag: e.target.value }));
  $("#f-sort").addEventListener("change", (e) => setFilter({ sort: e.target.value }));
  $("#f-desc").addEventListener("click", () => setFilter({ desc: !S.f.desc }));
  $("#f-shuffle").addEventListener("click", () => setFilter({ sort: "random", seed: Math.floor(Math.random() * 1e9) }));
  $("#bulk-btn").addEventListener("click", () => openBulk());
  $("#sample-list").addEventListener("click", (e) => { const it = e.target.closest(".s-item"); if (it) selectSample(+it.dataset.idx); });
  $("#sample-list").addEventListener("scroll", (e) => { const el = e.target; if (el.scrollTop + el.clientHeight > el.scrollHeight - 300) loadMore(); });

  $("#prev-btn").addEventListener("click", () => step(-1));
  $("#next-btn").addEventListener("click", () => step(1));
  $("#play-btn").addEventListener("click", togglePlay);
  $("#speed").addEventListener("change", (e) => (audio.playbackRate = +e.target.value));
  $("#gain").addEventListener("change", (e) => { S.gainDb = +e.target.value; ensureAudioGraph(); applyGain(); });
  $("#spec-kind").addEventListener("change", (e) => { S.specKind = e.target.value; loadSpectrogram(); });
  $("#cmap").addEventListener("change", (e) => { S.cmap = e.target.value; loadSpectrogram(); });
  $("#wave-norm").addEventListener("change", drawWave);
  bindScrub($("#wave-box"));
  bindScrub($("#spec-box"));
  $("#spec-box").addEventListener("mousemove", specHover);
  $("#spec-box").addEventListener("mouseleave", () => ($("#hover-read").textContent = ""));

  document.querySelectorAll(".rv").forEach((b) => b.addEventListener("click", () => saveReview(b.dataset.s, true)));
  $("#tag-box").addEventListener("click", (e) => { const t = e.target.closest(".tag"); if (t) toggleTag(t); });
  $("#note").addEventListener("change", () => { if (S.cur && S.cur.data.review.status !== "unreviewed") saveReview(S.cur.data.review.status, false); });

  $("#scan-btn").addEventListener("click", startScan);
  $("#scan-cancel").addEventListener("click", async () => { await send("DELETE", `/api/scan?${dsq()}`); });
  $("#scan-clear").addEventListener("click", async () => {
    if (!confirm(`Delete all cached metrics for ${S.ds}? Reviews are kept.`)) return;
    await send("DELETE", `/api/metrics?${dsq()}`); refreshDs(); loadReport(); reloadList(S.cur?.idx);
  });
  $("#report-body").addEventListener("click", (e) => {
    const v = e.target.closest("[data-view-flag]"), b = e.target.closest("[data-bulk-flag]"), s = e.target.closest("[data-sort-metric]");
    if (v) { setTab("inspect"); setFilter({ flag: v.dataset.viewFlag, status: "all", q: "" }); }
    if (b) openBulk({ q: "", status: "all", flag: b.dataset.bulkFlag });
    if (s) {
      const k = s.dataset.sortMetric;
      // "worst first": low values are bad for these metrics, high values for the rest
      const lowBad = ["snr_db", "rms_dbfs", "bandwidth_hz", "duration_s"].includes(k);
      setTab("inspect"); setFilter({ sort: k, desc: !lowBad, flag: "all", status: "all", q: "" });
    }
  });

  window.addEventListener("resize", () => { drawWave(); drawTimeAxis(); });
  window.addEventListener("keydown", onKey);
}

function toggleTag(t) {
  t.classList.toggle("on");
  if (S.cur && S.cur.data.review.status !== "unreviewed") saveReview(S.cur.data.review.status, false);
}

function onKey(e) {
  if (e.target.matches("input, textarea, select") || document.querySelector("dialog[open]")) {
    if (e.key === "Escape") e.target.blur?.();
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const inspect = !$("#tab-inspect").classList.contains("hidden");
  const k = e.key;
  if (e.shiftKey && /^Digit[1-9]$/.test(e.code)) {
    const t = document.querySelectorAll("#tag-box .tag")[+e.code.slice(5) - 1];
    if (t && inspect) { toggleTag(t); e.preventDefault(); }
    return;
  }
  const act = {
    " ": togglePlay,
    ArrowLeft: () => seek(audio.currentTime - 1), ArrowRight: () => seek(audio.currentTime + 1),
    ArrowDown: () => step(1), j: () => step(1), ArrowUp: () => step(-1), k: () => step(-1),
    1: () => saveReview("keep", true), 2: () => saveReview("reject", true), 3: () => saveReview("flag", true), 0: () => saveReview("unreviewed", false),
    s: () => { const sel = $("#spec-kind"); sel.value = sel.value === "mel" ? "linear" : "mel"; sel.dispatchEvent(new Event("change")); },
    g: () => { const sel = $("#gain"); sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length; sel.dispatchEvent(new Event("change")); toast(`Gain ${sel.value} dB`); },
    l: () => { $("#loop").checked = !$("#loop").checked; toast(`Loop ${$("#loop").checked ? "on" : "off"}`); },
    n: () => $("#note").focus(),
    "?": () => $("#help-dlg").showModal(),
  }[k];
  if (act && (inspect || k === "?")) { e.preventDefault(); act(); }
}
