// Review view: filterable sample list, player (waveform + spectrogram), and the keep / reject / flag inspector.
import {
  $, $$, S, esc, on, emit, api, getJSON, sendJSON, dsq, toast, curDataset, n0, fmtTime, fmtHz,
  METRICS, metricLabel, metricFmt, flagDesc, flagLabel, freqTicks, yAxisHTML, timeAxisHTML, hz2mel, mel2hz, tok, fitCanvas, parseWav,
} from "./core.js";

const PAGE = 150;
const R = {
  f: { q: "", status: "all", flag: "all", sort: "index", desc: false, seed: 0 },
  list: { rows: [], total: 0, gen: 0, loading: false },
  x: null,               // decoded samples of the open sample
  spec: "mel", cmap: "magma", gainDb: 0,
  selectToken: 0,
};
const audio = new Audio();
let actx = null, gainNode = null, raf = 0;

// ---------------------------------------------------------------- init
export function init() {
  const sorts = { index: "Dataset order", random: "Shuffled", status: "Review status", n_flags: "Number of checks failed" };
  $("#f-sort").innerHTML = S.cfg.sorts.filter((s) => s !== "channels")
    .map((s) => `<option value="${s}">${sorts[s] || `By ${metricLabel(s).toLowerCase()}`}</option>`).join("");
  $("#f-flag").innerHTML = [["all", "Any checks"], ["any", "Fails a check"], ["none", "Passes all checks"], ["scanned", "Scanned"], ["unscanned", "Not scanned"]]
    .map(([v, t]) => `<option value="${v}">${t}</option>`).join("") +
    `<optgroup label="Fails this check">${Object.keys(S.cfg.thresholds).map((f) => `<option value="${f}">${flagLabel(f)}</option>`).join("")}</optgroup>`;
  $("#tags").innerHTML = S.cfg.tags.map((t, i) => `<button class="chip" data-tag="${t}">${flagLabel(t)}${i < 9 ? `<sup>${i + 1}</sup>` : ""}</button>`).join("");
  $("#bulk-tag").insertAdjacentHTML("beforeend", S.cfg.tags.map((t) => `<option value="${t}">${flagLabel(t)}</option>`).join(""));
  $("#cmap").innerHTML = S.cfg.cmaps.map((c) => `<option value="${c}">${c === "gray_r" ? "gray" : c}</option>`).join("");
  setSegment("#spec-kind", R.spec);

  // list
  let qT;
  $("#f-q").addEventListener("input", () => { clearTimeout(qT); qT = setTimeout(() => setFilter({ q: $("#f-q").value.trim() }), 250); });
  $("#f-status").addEventListener("change", (e) => setFilter({ status: e.target.value }));
  $("#f-flag").addEventListener("change", (e) => setFilter({ flag: e.target.value }));
  $("#f-sort").addEventListener("change", (e) => setFilter({ sort: e.target.value, seed: e.target.value === "random" ? Date.now() % 1e9 : R.f.seed }));
  $("#f-desc").addEventListener("click", () => setFilter({ desc: !R.f.desc }));
  $("#bulk-open").addEventListener("click", () => openBulk());
  $("#sample-list").addEventListener("click", (e) => { const it = e.target.closest(".s-item"); if (it) selectSample(+it.dataset.idx); });
  $("#sample-list").addEventListener("scroll", (e) => { const el = e.target; if (el.scrollTop + el.clientHeight > el.scrollHeight - 400) loadMore(); });

  // player
  $("#prev-btn").addEventListener("click", () => step(-1));
  $("#next-btn").addEventListener("click", () => step(1));
  $("#play-btn").addEventListener("click", togglePlay);
  $("#speed").addEventListener("change", (e) => (audio.playbackRate = +e.target.value));
  $("#gain").addEventListener("change", (e) => { R.gainDb = +e.target.value; ensureGraph(); applyGain(); });
  $("#spec-kind").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setSpec(b.dataset.v); });
  $("#cmap").addEventListener("change", (e) => { R.cmap = e.target.value; loadSpectrogram(); });
  $("#wave-norm").addEventListener("change", drawWave);
  bindScrub($("#wave-plot"));
  bindScrub($("#spec-plot"));
  $("#spec-plot").addEventListener("mousemove", specHover);
  $("#spec-plot").addEventListener("mouseleave", () => ($("#spec-read").textContent = ""));

  // inspector
  $$(".verdict, #clear-btn").forEach((b) => b.addEventListener("click", () => saveReview(b.dataset.s, true)));
  $("#tags").addEventListener("click", (e) => { const t = e.target.closest(".chip"); if (t) toggleTag(t); });
  $("#note").addEventListener("change", () => { if (S.cur && S.cur.data.review.status !== "unreviewed") saveReview(S.cur.data.review.status, false); });

  on("dataset-changed", (idx) => { stopAudio(); showSample(false); reloadList(idx); });
  on("resize", () => { if (S.view === "review") redraw(); });
  on("theme-changed", () => { if (S.view === "review") redraw(); });
  on("filter-review", (patch) => { emit("set-view", "review"); setFilter({ q: "", status: "all", flag: "all", ...patch }); });
}

export function show() { redraw(); }
export function hide() { stopAudio(); }

function redraw() { drawWave(); drawAxes(); }
function setSegment(sel, v) { $$(`${sel} button`).forEach((b) => b.classList.toggle("on", b.dataset.v === v)); }

// ---------------------------------------------------------------- list
function setFilter(patch) {
  Object.assign(R.f, patch);
  $("#f-q").value = R.f.q;
  $("#f-status").value = R.f.status;
  $("#f-flag").value = R.f.flag;
  $("#f-sort").value = R.f.sort;
  $("#f-desc").textContent = R.f.desc ? "↓ Desc" : "↑ Asc";
  reloadList(S.cur?.idx, true);
}

async function reloadList(selectIdx = null, keepCurrent = false) {
  R.list = { rows: [], total: 0, gen: R.list.gen + 1, loading: false };
  $("#sample-list").innerHTML = `<div class="list-more">Loading…</div>`;
  await loadMore();
  if (keepCurrent && S.cur) return;
  const want = selectIdx ?? R.list.rows[0]?.idx;
  if (want != null) selectSample(want);
  else if (!R.list.rows.length) showSample(false);
}

async function loadMore() {
  const L = R.list;
  if (L.loading || (L.rows.length && L.rows.length >= L.total) || !S.ds) return;
  L.loading = true;
  const gen = L.gen;
  try {
    const r = await getJSON(`/api/samples?${dsq({ offset: L.rows.length, limit: PAGE, ...R.f })}`);
    if (gen !== R.list.gen) return;
    L.rows.push(...r.rows);
    L.total = r.total;
    renderList();
  } catch (e) { toast(e.message, "error"); }
  finally { if (gen === R.list.gen) L.loading = false; }
}

function itemHTML(s) {
  const st = s.status !== "unreviewed" ? `<span class="badge ${s.status}">${s.status}</span>` : "";
  const dur = s.duration_s != null ? `<span class="badge">${s.duration_s.toFixed(1)} s</span>` : "";
  const flags = (s.flags || []).map((f) => `<span class="badge auto" title="${esc(flagDesc(f))}">${flagLabel(f)}</span>`).join("");
  return `<button class="s-item ${S.cur?.idx === s.idx ? "active" : ""}" data-idx="${s.idx}" data-status="${s.status}">
    <div class="top"><span class="sid">#${s.idx} · ${esc(s.id)}</span>${dur}${st}</div>
    <div class="lbl">${esc(s.label) || '<span class="muted">(no label)</span>'}</div>
    ${flags ? `<div class="fl">${flags}</div>` : ""}</button>`;
}

function renderList() {
  const L = R.list;
  const metricSort = R.f.sort in METRICS || R.f.sort === "n_flags";
  $("#list-count").textContent = `${n0(L.total)} ${L.total === 1 ? "row" : "rows"}${metricSort ? " (scanned only)" : ""}`;
  if (!L.rows.length) {
    $("#sample-list").innerHTML = `<div class="empty-state">No rows match.${metricSort ? "<br>Sorting by a metric shows scanned rows only: run a scan in Quality report." : ""}</div>`;
    return;
  }
  const more = L.rows.length < L.total ? `<div class="list-more">Showing ${n0(L.rows.length)} of ${n0(L.total)} · scroll for more</div>` : "";
  $("#sample-list").innerHTML = L.rows.map(itemHTML).join("") + more;
}

function patchListItem(idx, patch) {
  const s = R.list.rows.find((r) => r.idx === idx);
  if (!s) return;
  Object.assign(s, patch);
  const el = $(`.s-item[data-idx="${idx}"]`);
  if (el) el.outerHTML = itemHTML(s);
}

async function step(delta) {
  if (!S.cur) return;
  let pos = R.list.rows.findIndex((r) => r.idx === S.cur.idx);
  if (pos < 0) {  // the open sample left the list (filter changed): walk the dataset instead
    const j = S.cur.idx + delta;
    if (j >= 0 && j < curDataset().n_rows) selectSample(j);
    return;
  }
  pos += delta;
  if (pos >= R.list.rows.length && R.list.rows.length < R.list.total) await loadMore();
  if (pos < 0 || pos >= R.list.rows.length) { toast(pos < 0 ? "First sample in the list" : "Last sample in the list"); return; }
  selectSample(R.list.rows[pos].idx);
  $(`.s-item[data-idx="${R.list.rows[pos].idx}"]`)?.scrollIntoView({ block: "nearest" });
}

// ---------------------------------------------------------------- open a sample
function showSample(yes) {
  $("#stage-empty").hidden = yes;
  $("#stage-body").hidden = !yes;
  $("#insp-body").hidden = !yes;
}

async function selectSample(idx) {
  const tok = ++R.selectToken;
  stopAudio();
  $$(".s-item.active").forEach((e) => e.classList.remove("active"));
  $(`.s-item[data-idx="${idx}"]`)?.classList.add("active");
  let data;
  try { data = await getJSON(`/api/sample?${dsq({ idx })}`); }
  catch (e) { toast(e.message, "error"); return; }
  if (tok !== R.selectToken) return;
  S.cur = { idx, data, dur: data.metrics.duration_s || 0 };
  S.sel = null;
  R.x = null;
  showSample(true);
  renderSample();
  patchListItem(idx, { flags: data.metrics.flags, duration_s: data.metrics.duration_s });
  loadSpectrogram();
  emit("sample-changed");
  await loadAudio(tok);
}

function renderSample() {
  const { data, idx } = S.cur;
  $("#s-id").textContent = data.id;
  $("#s-key").textContent = data.key;
  const pos = R.list.rows.findIndex((r) => r.idx === idx);
  $("#s-pos").textContent = pos >= 0 ? `${n0(pos + 1)} of ${n0(R.list.total)} in list` : `row ${n0(idx)} of ${n0(data.n_rows)}`;

  const labels = Object.entries(data.labels);
  $("#s-labels").innerHTML = labels.length ? labels.map(([k, v]) => {
    const empty = v == null || String(v).trim() === "";
    const txt = empty ? "empty" : typeof v === "object" ? JSON.stringify(v) : v;
    return `<div class="label-row"><div class="k">${esc(k)}</div><div class="v ${empty ? "empty" : ""}">${esc(txt)}</div></div>`;
  }).join("") : `<div class="label-row"><div class="v muted">This dataset has no label column.</div></div>`;

  const m = data.metrics, flags = m.flags || [];
  const bad = new Set(flags.map((f) => S.cfg.thresholds[f]?.metric).filter(Boolean));
  $("#s-flags").innerHTML = flags.length
    ? flags.map((f) => `<span class="chip warn" title="${esc(flagDesc(f))}">${flagLabel(f)}</span>`).join("")
    : `<span class="chip ok">Passes every check</span>`;
  if (m.error) $("#s-flags").insertAdjacentHTML("beforeend", `<span class="chip bad">${esc(m.error)}</span>`);
  $("#s-metrics").innerHTML = Object.keys(METRICS).filter((k) => k in m && !(k === "chars_per_sec" && m[k] == null))
    .map((k) => `<div class="${bad.has(k) ? "bad" : ""}"><dt>${metricLabel(k)}</dt><dd>${metricFmt(k, m[k])}</dd></div>`).join("");
  $("#s-meta").innerHTML = Object.entries(data.meta).map(([k, v]) =>
    `<tr><th>${esc(k)}</th><td>${esc(typeof v === "object" && v !== null ? JSON.stringify(v) : v)}</td></tr>`).join("");
  renderReview();
  $("#t-cur").textContent = fmtTime(0);
  $("#t-dur").textContent = fmtTime(S.cur.dur);
  drawAxes();
}

// ---------------------------------------------------------------- review
function renderReview() {
  const r = S.cur.data.review;
  $$(".verdict").forEach((b) => b.classList.toggle("on", b.dataset.s === r.status));
  $$("#tags .chip").forEach((t) => t.classList.toggle("on", (r.tags || []).includes(t.dataset.tag)));
  if (document.activeElement !== $("#note")) $("#note").value = r.note || "";
  $("#review-meta").textContent = r.updated_at ? `Saved ${new Date(r.updated_at).toLocaleString()}` : "Not reviewed yet";
}

const currentTags = () => $$("#tags .chip.on").map((t) => t.dataset.tag);

async function saveReview(status, advance) {
  if (!S.cur) return;
  const idx = S.cur.idx, clear = status === "unreviewed";
  try {
    const r = await sendJSON("PUT", `/api/review?${dsq({ idx })}`, { status, tags: clear ? [] : currentTags(), note: clear ? "" : $("#note").value });
    if (S.cur?.idx === idx) { S.cur.data.review = r; renderReview(); }
    patchListItem(idx, { status: r.status });
    emit("refresh-dataset");
    if (advance && !clear && $("#auto-adv").checked) step(1);
  } catch (e) { toast(e.message, "error"); }
}

function toggleTag(t) {
  t.classList.toggle("on");
  if (S.cur && S.cur.data.review.status !== "unreviewed") saveReview(S.cur.data.review.status, false);
  else if (t.classList.contains("on")) toast("Reasons are saved with Keep, Reject or Flag");
}

// ---------------------------------------------------------------- audio
function ensureGraph() {
  if (actx) return;
  actx = new AudioContext();
  gainNode = actx.createGain();
  actx.createMediaElementSource(audio).connect(gainNode).connect(actx.destination);
  applyGain();
}
const applyGain = () => gainNode && (gainNode.gain.value = 10 ** (R.gainDb / 20));

async function loadAudio(tok) {
  const c = S.cur;
  waveMsg("Loading audio…");
  try {
    const buf = await (await api(`/api/audio?${dsq({ idx: c.idx })}`)).arrayBuffer();
    if (tok !== R.selectToken) return;
    const { x, sr } = parseWav(buf);
    R.x = x;
    c.dur = x.length / sr;
    if (audio.src) URL.revokeObjectURL(audio.src);
    audio.src = URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
    audio.playbackRate = +$("#speed").value;
    $("#t-dur").textContent = fmtTime(c.dur);
    waveMsg("");
    redraw();
  } catch (e) {
    if (tok === R.selectToken) waveMsg(`Could not load audio: ${e.message}`);
  }
}

function stopAudio() { audio.pause(); cancelAnimationFrame(raf); setPlayBtn(false); }
const setPlayBtn = (on) => ($("#play-btn").textContent = on ? "❚❚ Pause" : "▶ Play");

function togglePlay() {
  if (!S.cur || !audio.src) return;
  ensureGraph();
  if (actx.state === "suspended") actx.resume();
  if (audio.paused) {
    if (S.sel && (audio.currentTime < S.sel.a || audio.currentTime >= S.sel.b - 0.01)) audio.currentTime = S.sel.a;
    audio.play().catch((e) => toast(e.message, "error"));
  } else audio.pause();
}

function tick() {
  if (S.sel && audio.currentTime >= S.sel.b) {
    audio.currentTime = S.sel.a;
    if (!$("#loop").checked) audio.pause();
  }
  updatePlayhead();
  if (!audio.paused) raf = requestAnimationFrame(tick);
}
function updatePlayhead() {
  const p = `${(100 * audio.currentTime) / (S.cur?.dur || 1)}%`;
  $$("#scope .playhead").forEach((el) => (el.style.left = p));
  $("#t-cur").textContent = fmtTime(audio.currentTime);
}
audio.addEventListener("play", () => { setPlayBtn(true); raf = requestAnimationFrame(tick); });
audio.addEventListener("pause", () => { setPlayBtn(false); updatePlayhead(); });
audio.addEventListener("ended", () => {
  if ($("#loop").checked) { audio.currentTime = S.sel ? S.sel.a : 0; audio.play(); }
  else setPlayBtn(false);
});

function seek(t) {
  if (!S.cur) return;
  audio.currentTime = Math.max(0, Math.min(S.cur.dur, t));
  updatePlayhead();
}

// ---------------------------------------------------------------- waveform + axes
function waveMsg(m) { $("#wave-msg").textContent = m; }

function drawWave() {
  const cv = $("#wave");
  if (!cv.clientWidth) return;
  const { ctx, W, H } = fitCanvas(cv);
  const x = R.x;
  ctx.clearRect(0, 0, W, H);
  if (!x) return;
  let peak = 0;
  for (let i = 0; i < x.length; i++) { const a = Math.abs(x[i]); if (a > peak) peak = a; }
  const scale = $("#wave-norm").checked && peak > 0 ? 1 / peak : 1;
  const top = scale === 1 ? "+1" : `+${peak.toFixed(2)}`;
  $("#wave-y").innerHTML = yAxisHTML([{ pos: 0.98, label: top }, { pos: 0.5, label: "0" }, { pos: 0.02, label: top.replace("+", "−") }]);
  const mid = H / 2, amp = (H / 2) * 0.96 * scale;
  ctx.fillStyle = tok("--grid");
  ctx.fillRect(0, mid, W, 1);
  if (scale === 1) for (const lv of [0.5, 0.1]) { ctx.fillRect(0, mid - lv * amp, W, 1); ctx.fillRect(0, mid + lv * amp, W, 1); }
  const cWave = tok("--wave"), cRms = tok("--wave-rms"), cClip = tok("--clip");
  const spp = x.length / W;
  for (let px = 0; px < W; px++) {
    const s0 = Math.floor(px * spp), s1 = Math.min(x.length, Math.max(s0 + 1, Math.floor((px + 1) * spp)));
    let mn = 1, mx = -1, clip = false, sq = 0;
    for (let i = s0; i < s1; i++) {
      const v = x[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      if (v >= 0.999 || v <= -0.999) clip = true;
      sq += v * v;
    }
    if (mn > mx) continue;
    const rms = Math.sqrt(sq / (s1 - s0));
    ctx.fillStyle = clip ? cClip : cWave;
    ctx.fillRect(px, mid - mx * amp, 1, Math.max(1, (mx - mn) * amp));
    ctx.fillStyle = clip ? cClip : cRms;
    ctx.fillRect(px, mid - rms * amp, 1, Math.max(1, 2 * rms * amp));
  }
  updatePlayhead();
  drawRegion();
}

function drawRegion() {
  const el = $("#region");
  if (!S.sel || !S.cur?.dur) { el.style.display = "none"; return; }
  el.style.display = "block";
  el.style.left = `${(100 * S.sel.a) / S.cur.dur}%`;
  el.style.width = `${(100 * (S.sel.b - S.sel.a)) / S.cur.dur}%`;
}

const specFmax = () => (R.spec === "mel" ? 8000 : (S.cur?.data.metrics.sample_rate || 16000) / 2);

function drawAxes() {
  if (!S.cur) return;
  $("#spec-y").innerHTML = yAxisHTML(freqTicks(specFmax(), R.spec === "mel"));
  $("#time-x").innerHTML = timeAxisHTML($("#time-x"), S.cur.dur);
}

// ---------------------------------------------------------------- spectrogram
function setSpec(kind) {
  R.spec = kind;
  setSegment("#spec-kind", kind);
  loadSpectrogram();
  drawAxes();
}

function loadSpectrogram() {
  if (!S.cur) return;
  const img = $("#spec"), msg = $("#spec-msg");
  msg.textContent = "Rendering spectrogram…";
  img.onload = () => (msg.textContent = "");
  img.onerror = () => (msg.textContent = "Could not render the spectrogram");
  img.src = `/api/spectrogram?${dsq({ idx: S.cur.idx, kind: R.spec, cmap: R.cmap })}`;
}

function specHover(e) {
  if (!S.cur) return;
  const r = $("#spec-plot").getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width, fy = 1 - (e.clientY - r.top) / r.height, fmax = specFmax();
  const hz = R.spec === "mel" ? mel2hz(fy * hz2mel(fmax)) : fy * fmax;
  $("#spec-read").textContent = `${(fx * S.cur.dur).toFixed(2)} s · ${fmtHz(hz)}`;
}

// click = seek; drag = select a region
function bindScrub(box) {
  let down = null;
  const tAt = (e) => {
    const r = box.getBoundingClientRect();
    return Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * (S.cur?.dur || 0);
  };
  box.addEventListener("pointerdown", (e) => {
    if (!S.cur || e.button !== 0) return;
    down = { t: tAt(e), x: e.clientX };
    box.setPointerCapture(e.pointerId);
  });
  box.addEventListener("pointermove", (e) => {
    if (!down || Math.abs(e.clientX - down.x) <= 4) return;
    const t = tAt(e);
    S.sel = { a: Math.min(down.t, t), b: Math.max(down.t, t) };
    drawRegion();
  });
  box.addEventListener("pointerup", (e) => {
    if (!down) return;
    if (Math.abs(e.clientX - down.x) <= 4) { S.sel = null; drawRegion(); seek(down.t); }
    else if (S.sel) { seek(S.sel.a); toast(`Region ${S.sel.a.toFixed(2)}–${S.sel.b.toFixed(2)} s`); }
    down = null;
  });
}

// ---------------------------------------------------------------- bulk review
export async function openBulk(filter) {
  if (!S.ds) return;
  const f = filter || { q: R.f.q, status: R.f.status, flag: R.f.flag };
  let n;
  try { n = (await getJSON(`/api/samples?${dsq({ ...f, limit: 0 })}`)).total; }
  catch (e) { toast(e.message, "error"); return; }
  const parts = [f.flag && f.flag !== "all" ? `check “${flagLabel(f.flag)}”` : "", f.status !== "all" ? `status “${f.status}”` : "", f.q ? `search “${f.q}”` : ""].filter(Boolean);
  $("#bulk-n").textContent = n0(n);
  $("#bulk-desc").textContent = parts.length ? parts.join(", ") : "the current filter (no filter set)";
  const dlg = $("#bulk-dlg");
  dlg.returnValue = "";
  dlg.showModal();
  dlg.addEventListener("close", async () => {
    if (dlg.returnValue !== "ok") return;
    try {
      const res = await sendJSON("POST", `/api/bulk_review?${dsq()}`, { filter: f, status: $("#bulk-status").value, tag: $("#bulk-tag").value || null });
      toast(`Updated ${n0(res.updated)} rows`);
      emit("refresh-dataset");
      emit("bulk-done");
      reloadList(S.cur?.idx, true);
    } catch (e) { toast(e.message, "error"); }
  }, { once: true });
}

// ---------------------------------------------------------------- keys
export function onKey(e) {
  if (e.shiftKey && /^Digit[1-9]$/.test(e.code)) {
    const t = $$("#tags .chip")[+e.code.slice(5) - 1];
    if (t && S.cur) { e.preventDefault(); toggleTag(t); }
    return;
  }
  const act = {
    " ": togglePlay,
    ArrowLeft: () => seek(audio.currentTime - 1), ArrowRight: () => seek(audio.currentTime + 1),
    ArrowDown: () => step(1), j: () => step(1), ArrowUp: () => step(-1), k: () => step(-1),
    1: () => saveReview("keep", true), 2: () => saveReview("reject", true), 3: () => saveReview("flag", true), 0: () => saveReview("unreviewed", false),
    s: () => setSpec(R.spec === "mel" ? "linear" : "mel"),
    g: () => { const sel = $("#gain"); sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length; sel.dispatchEvent(new Event("change")); toast(`Gain ${sel.options[sel.selectedIndex].text}`); },
    l: () => { $("#loop").checked = !$("#loop").checked; toast(`Loop ${$("#loop").checked ? "on" : "off"}`); },
    n: () => $("#note").focus(),
  }[e.key];
  if (act) { e.preventDefault(); act(); }
}
