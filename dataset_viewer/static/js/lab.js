// Front-end lab: run the open sample through resample -> quantise -> STFT -> mel -> log -> precision -> MVN -> stack
// with any settings, and see / hear each stage.
//
// One source of truth: L.p holds the settings. Every control writes to it and the form is redrawn from it. Runs are
// serialised (at most one request in flight; the latest settings always win), and each result carries its own file
// key, so audio and images always belong to the settings that produced them.
import {
  $, $$, S, esc, on, api, getJSON, dsq, toast, n0, fmtHz, kLabel, freqTicks, yAxisHTML, timeAxisHTML, hz2mel, mel2hz, tok, fitCanvas,
} from "./core.js";

const L = {
  p: null, base: null,          // current settings; train.py defaults
  res: null, sent: null,        // last result and the settings it was run with
  inflight: false, again: false, timer: 0,
  sampleKey: null,              // which sample the region / result belong to
  patch: 0, norm: null, playing: null, raf: 0,
};
const player = new Audio();

// ---------------------------------------------------------------- settings
const SR_OPTS = [8000, 16000, 22050, 24000, 44100, 48000];
const FIELDS = [
  { group: "Waveform" },
  { k: "sr", label: "Sample rate", code: "SR", kind: "seg", opts: SR_OPTS, fmt: (v) => `${+(v / 1000).toFixed(2)}k`,
    hint: (p, st) => `Nyquist ${fmtHz(p.sr / 2)}${st && p.sr > st.native_sr ? ` · above the file's ${fmtHz(st.native_sr)}` : ""}` },
  { k: "bits", label: "Bit depth", code: "BITS", kind: "seg", opts: [32, 16, 12, 8, 6, 4, 2], fmt: (v) => (v >= 32 ? "float" : `${v}`),
    hint: (p) => (p.bits >= 32 ? "32-bit float: no quantisation" : `${2 ** p.bits} levels`) },
  { k: "quant", label: "Coding", kind: "seg", opts: ["pcm", "mulaw"], fmt: (v) => ({ pcm: "Linear PCM", mulaw: "μ-law" })[v],
    disabled: (p) => p.bits >= 32, hint: (p) => (p.bits >= 32 ? "Only applies below 32 bits" : p.quant === "mulaw" ? "Finer steps for quiet samples (telephony)" : "Equal steps everywhere") },
  { group: "Spectrogram" },
  { k: "n_fft", label: "Window", code: "N_FFT", kind: "num", min: 32, max: 4096, step: 1, quick: [200, 256, 400, 512, 640, 1024],
    hint: (p) => `${ms(p.n_fft, p.sr)} ms · ${(p.sr / p.n_fft).toFixed(1)} Hz per bin` },
  { k: "hop", label: "Hop", code: "HOP", kind: "num", min: 8, max: 4096, step: 1, quick: [80, 160, 240, 320],
    hint: (p) => `${ms(p.hop, p.sr)} ms · ${(p.sr / p.hop).toFixed(1)} frames/s · ${Math.round(100 * (1 - p.hop / p.n_fft))}% overlap` },
  { group: "Model input" },
  { k: "n_mels", label: "Mel channels", code: "N_MELS", kind: "range", min: 4, max: 256, step: 1, quick: [40, 64, 80, 128],
    hint: (p) => `${(p.n_fft / 2 + 1)} FFT bins → ${p.n_mels} channels` },
  { k: "stack", label: "Frames per token", code: "STACK", kind: "range", min: 1, max: 64, step: 1, quick: [4, 8, 16, 32],
    hint: (p) => `${((1000 * p.stack * p.hop) / p.sr).toFixed(0)} ms per token · ${(p.sr / p.hop / p.stack).toFixed(2)} tokens/s` },
  { k: "feat", label: "Feature precision", kind: "seg", opts: ["fp32", "bf16", "fp16", "int8"], fmt: (v) => v,
    hint: (p) => ({ fp32: "Exact", bf16: "train.py under PRECISION=bf16", fp16: "Half precision", int8: "Per-clip min / max, 256 levels" })[p.feat] },
  { k: "hidden", label: "LLM hidden size", kind: "num", min: 64, max: 16384, step: 1, quick: [768, 1024, 2048],
    hint: (p) => `patch ${p.stack * p.n_mels}-d = ${((p.stack * p.n_mels) / p.hidden).toFixed(2)}× hidden` },
];
const PARAM_KEYS = FIELDS.filter((f) => f.k).map((f) => f.k);
const ordinal = (n) => `${n}${[, "st", "nd", "rd"][(n % 100 >> 3) ^ 1 && n % 10] || "th"}`;
const ms = (n, sr) => +((1000 * n) / sr).toFixed(2);
const fieldOf = (k) => FIELDS.find((f) => f.k === k);

function clamp(k, v) {
  const f = fieldOf(k);
  if (k === "start") return +Math.max(0, Math.min(+v || 0, (S.cur?.dur || Infinity) - 0.2)).toFixed(2);
  if (k === "dur") return Math.min(S.cfg.lab.max_sec, Math.max(0.2, +v || 8));
  if (f.kind === "seg") return f.opts.map(String).includes(String(v)) ? (typeof f.opts[0] === "number" ? +v : v) : L.p[k];
  return Math.round(Math.min(f.max, Math.max(f.min, +v)));
}

export function init() {
  L.base = { ...S.cfg.lab.presets["train.py default"], hidden: 1024 };
  L.p = { ...L.base, start: 0, dur: 8 };
  $("#lab-preset").innerHTML = `<option value="">Custom</option>` +
    Object.keys(S.cfg.lab.presets).map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  buildFields();
  syncForm();

  const box = $("#lab-fields");
  box.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-k]");
    if (b && !b.disabled) setParam(b.dataset.k, b.dataset.v);
  });
  box.addEventListener("input", (e) => {
    const el = e.target, k = el.dataset.k;
    if (!k) return;
    if (el.type === "range") setParam(k, el.value);
    else if (el.value !== "" && el.validity.valid) setParam(k, el.value, el);  // typing: don't rewrite the box under the cursor
  });
  box.addEventListener("change", (e) => { const k = e.target.dataset.k; if (k) setParam(k, e.target.value); });
  for (const k of ["start", "dur"]) {
    $(`#lab-${k}`).addEventListener("change", (e) => setParam(k, e.target.value));
  }
  $("#lab-preset").addEventListener("change", (e) => {
    if (!e.target.value) return;
    Object.assign(L.p, L.base, S.cfg.lab.presets[e.target.value], { hidden: L.p.hidden });
    commit(true);
  });
  $("#lab-reset").addEventListener("click", () => { Object.assign(L.p, L.base); commit(true); });
  $("#lab-run").addEventListener("click", () => run());
  $("#lab-auto").addEventListener("change", (e) => e.target.checked && !upToDate() && run());
  $("#lab-use-sel").addEventListener("click", () => {
    if (!S.sel) { toast("Drag across the waveform in Review to select a region first"); return; }
    L.p.start = +S.sel.a.toFixed(2);
    L.p.dur = clamp("dur", +(S.sel.b - S.sel.a).toFixed(2));
    commit(true);
  });

  $("#lab-listen").addEventListener("click", (e) => { const b = e.target.closest(".play-card"); if (b) play(b.dataset.wav); });
  $("#lab-pipeline").addEventListener("click", onPipelineClick);
  $("#lab-pipeline").addEventListener("mousemove", onPipelineHover);
  $("#lab-pipeline").addEventListener("mouseleave", () => $$("#lab-pipeline .readout").forEach((el) => (el.textContent = "")));
  for (const id of ["#lab-fb", "#lab-ltas"]) {
    const cv = $(id);
    cv.addEventListener("mousemove", (e) => chartHover(cv, e));
    cv.addEventListener("mouseleave", () => { redrawChart(cv); cv.nextElementSibling.style.display = "none"; });
  }

  on("dataset-changed", () => { L.sampleKey = null; L.res = null; stopPlayer(); $("#lab-body").hidden = true; });
  on("sample-changed", () => S.view === "lab" && show());
  on("resize", () => S.view === "lab" && redraw());
  on("theme-changed", () => S.view === "lab" && redraw());
}

const sampleKey = () => (S.cur ? `${S.ds}#${S.cur.idx}` : null);

export function show() {
  const has = !!S.cur;
  $("#lab-empty").hidden = has;
  $("#lab-body").hidden = !has || !L.res;
  $("#lab-use-sel").disabled = !S.sel;
  if (!has) { setStatus("idle", "No sample"); return; }
  if (L.sampleKey !== sampleKey()) {   // a new sample: start from the Review region, or its first 8 s
    L.sampleKey = sampleKey();
    if (S.sel && S.sel.b - S.sel.a >= 0.2) { L.p.start = +S.sel.a.toFixed(2); L.p.dur = clamp("dur", S.sel.b - S.sel.a); }
    else { L.p.start = 0; L.p.dur = clamp("dur", Math.min(8, S.cur.dur || 8)); }
    syncForm();
    run();
  } else redraw();
}
export function hide() { stopPlayer(); }

// ---------------------------------------------------------------- form
function buildFields() {
  let html = "", open = false;
  for (const f of FIELDS) {
    if (f.group) {
      html += `${open ? "</fieldset>" : ""}<fieldset class="group"><legend>${f.group}</legend>`;
      open = true;
      continue;
    }
    const head = `<div class="field-head">${f.label}${f.code ? ` <span class="code">${f.code}</span>` : ""}
      <button class="reset" data-k="${f.k}" data-v="" title="Back to the train.py value" hidden></button></div>`;
    let ctl;
    if (f.kind === "seg") {
      ctl = `<div class="seg" role="radiogroup" aria-label="${f.label}">${f.opts.map((o) => `<button data-k="${f.k}" data-v="${o}" role="radio">${f.fmt(o)}</button>`).join("")}</div>`;
    } else {
      const range = f.kind === "range" ? `<input type="range" data-k="${f.k}" min="${f.min}" max="${f.max}" step="${f.step}" aria-label="${f.label}">` : "";
      ctl = `<div class="num-row">${range}<input type="number" id="lab-${f.k}" data-k="${f.k}" min="${f.min}" max="${f.max}" step="${f.step}" aria-label="${f.label}"></div>
        <div class="quick">${f.quick.map((q) => `<button data-k="${f.k}" data-v="${q}">${q}</button>`).join("")}</div>`;
    }
    html += `<div class="field" data-field="${f.k}">${head}${ctl}<div class="hint"></div></div>`;
  }
  $("#lab-fields").innerHTML = html + "</fieldset>";
  // the reset dot carries an empty value: map it to the default
  $$("#lab-fields .reset").forEach((b) => (b.dataset.v = L.base[b.dataset.k]));
}

/** redraw every control from L.p (skip `except`, the box the user is typing in) */
function syncForm(except = null) {
  for (const f of FIELDS) {
    if (!f.k) continue;
    const el = $(`[data-field="${f.k}"]`), v = L.p[f.k], changed = String(v) !== String(L.base[f.k]);
    el.classList.toggle("changed", changed);
    el.querySelector(".reset").hidden = !changed;
    const dis = f.disabled?.(L.p) ?? false;
    el.classList.toggle("disabled", dis);
    el.querySelectorAll("button[data-v]:not(.reset)").forEach((b) => {
      b.classList.toggle("on", String(b.dataset.v) === String(v));
      if (b.closest(".seg")) { b.disabled = dis; b.setAttribute("aria-checked", String(b.dataset.v) === String(v)); }
    });
    el.querySelectorAll("input").forEach((i) => { if (i !== except) i.value = v; });
    el.querySelector(".hint").textContent = f.hint(L.p, L.res?.stats);
  }
  for (const k of ["start", "dur"]) if ($(`#lab-${k}`) !== except) $(`#lab-${k}`).value = L.p[k];
  const match = Object.entries(S.cfg.lab.presets).find(([, pr]) => PARAM_KEYS.filter((k) => k !== "hidden").every((k) => String({ ...L.base, ...pr }[k]) === String(L.p[k])));
  $("#lab-preset").value = match ? match[0] : "";
  renderDerived();
}

function setParam(k, v, typingIn = null) {
  const nv = clamp(k, v);
  if (String(nv) === String(L.p[k]) && !typingIn) { syncForm(); return; }
  L.p[k] = nv;
  syncForm(typingIn);
  schedule();
}
function commit(now) { syncForm(); now ? run() : schedule(); }

const paramsKey = (p) => JSON.stringify([...PARAM_KEYS, "start", "dur"].map((k) => p[k]));
const upToDate = () => L.res && L.sent && paramsKey(L.sent) === paramsKey(L.p) && L.res.forSample === sampleKey();

function schedule() {
  clearTimeout(L.timer);
  markStale();
  if ($("#lab-auto").checked) L.timer = setTimeout(run, 280);
}

function markStale() {
  const stale = !!L.res && !upToDate();
  $("#lab-body").classList.toggle("stale", stale);
  if (!L.inflight) setStatus(stale ? "stale" : L.res ? "ok" : "idle", stale ? ($("#lab-auto").checked ? "Changed" : "Changed · press Run") : L.res ? "Up to date" : "Idle");
}

function setStatus(state, text) { const s = $("#lab-status"); s.dataset.state = state; s.textContent = text; }

// ---------------------------------------------------------------- run
async function run() {
  clearTimeout(L.timer);
  if (!S.cur) return;
  if (L.inflight) { L.again = true; return; }
  L.inflight = true;
  const sent = { ...L.p }, forSample = sampleKey();
  setStatus("busy", "Running…");
  $("#lab-run").disabled = true;
  try {
    const res = await getJSON(`/api/lab?${dsq({ idx: S.cur.idx, ...sent })}`);
    if (forSample !== sampleKey()) return;
    res.forSample = forSample;
    // the server clamps impossible values; show what it actually ran if the user hasn't moved on
    if (paramsKey(sent) === paramsKey(L.p)) {
      for (const k of [...PARAM_KEYS, "start", "dur"]) if (k in res.params) L.p[k] = res.params[k];
      Object.assign(sent, res.params);
    }
    sent.hidden = L.p.hidden;
    L.sent = sent;
    L.res = res;
    L.norm = null;
    L.patch = Math.min(L.patch, res.stats.tokens - 1);
    stopPlayer();
    player.removeAttribute("src");
    L.playing = null;
    $("#lab-body").hidden = false;
    render();
    syncForm();
  } catch (e) {
    setStatus("error", "Failed");
    toast(`Lab: ${e.message}`, "error");
    L.again = false;
  } finally {
    L.inflight = false;
    $("#lab-run").disabled = false;
    if (L.again) { L.again = false; if (!upToDate()) run(); }
    else if ($("#lab-status").dataset.state !== "error") markStale();
  }
}

const labFile = (name) => `/api/lab/file?key=${encodeURIComponent(L.res.key)}&name=${name}`;

// ---------------------------------------------------------------- derived numbers (instant, before the server answers)
function derived(p) {
  const fps = p.sr / p.hop, frames = Math.floor((p.dur * p.sr) / p.hop) + 1;
  return {
    "Window": [`${ms(p.n_fft, p.sr)} ms`, p.n_fft / p.sr],
    "Hop": [`${ms(p.hop, p.sr)} ms`, p.hop / p.sr],
    "Frequency step": [`${(p.sr / p.n_fft).toFixed(1)} Hz`, p.sr / p.n_fft],
    "Frame rate": [`${fps.toFixed(1)} /s`, fps],
    "Token rate": [`${(fps / p.stack).toFixed(2)} /s`, fps / p.stack],
    "Token length": [`${((1000 * p.stack) / fps).toFixed(0)} ms`, p.stack / fps],
    "Tokens in clip": [n0(Math.ceil(frames / p.stack)), null],
    "Patch size": [`${p.stack * p.n_mels}-d`, p.stack * p.n_mels],
  };
}
function renderDerived() {
  const now = derived(L.p), base = derived({ ...L.base, dur: L.p.dur });
  $("#lab-derived").innerHTML = Object.entries(now).map(([k, [v, raw]]) => {
    const changed = raw != null && Math.abs(raw - base[k][1]) > 1e-9;
    return `<div class="${changed ? "changed" : ""}"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${changed ? `train.py ${base[k][0]}` : "&nbsp;"}</div></div>`;
  }).join("");
}

// ---------------------------------------------------------------- render a result
function render() {
  const r = L.res, st = r.stats, p = r.params;
  $("#lab-clip").textContent = `${S.cur.data.id} · ${p.start.toFixed(2)}–${(p.start + st.seconds).toFixed(2)} s · native ${fmtHz(st.native_sr)}`;
  $("#lab-notes").innerHTML = r.notes.map((n) => `<div class="note-line">${esc(n)}</div>`).join("");

  const tile = (k, v, s, warn = false) => `<div class="tile ${warn ? "warn" : ""}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
  $("#lab-scores").innerHTML = [
    tile("Waveform loss", `${st.lsd_wave.toFixed(2)} dB`, "LSD after resample + quantise, vs the original on a fine 16 kHz grid", st.lsd_wave > 3),
    tile("Log-mel loss", `${st.lsd_mel.toFixed(2)} dB`, "LSD of what survives the whole front-end; lower keeps more"),
    tile("Quantisation SNR", st.quant_snr_db == null ? "none" : `${st.quant_snr_db} dB`, p.bits >= 32 ? "32-bit float input" : `${p.bits}-bit ${p.quant === "mulaw" ? "μ-law" : "PCM"}`, st.quant_snr_db != null && st.quant_snr_db < 30),
    tile("Dead mel filters", st.dead_filters, "filters that cover no FFT bin", st.dead_filters > 0),
    tile("Feature error", st.feat_max_err ? st.feat_max_err.toFixed(4) : "0", `max |Δ log-mel| from ${p.feat}`),
    tile("Patch vs hidden", `${st.dim_over_h.toFixed(2)}×`, `${st.patch_dim}-d patch into ${st.hidden_size}-d`, st.dim_over_h < 1),
  ].join("");

  const lis = [
    ["orig", "Original", `${fmtHz(st.native_sr)}, untouched`],
    ["wave", "Processed waveform", `${fmtHz(p.sr)}, ${p.bits >= 32 ? "float" : `${p.bits}-bit ${p.quant === "mulaw" ? "μ-law" : "PCM"}`}`],
    ...(p.bits < 32 ? [["err", "Quantisation error", "processed − resampled, normalised"]] : []),
    ["stft", "From |STFT|", "Griffin-Lim: loses phase only"],
    ["mel", "From the log-mel", "Griffin-Lim from the model input: phase + mel smoothing + precision"],
  ];
  $("#lab-listen").innerHTML = lis.map(([k, t, s]) =>
    `<button class="play-card" data-wav="${k}"><span class="ic">▶</span><span><b>${t}</b><br><span class="muted small">${esc(s)}</span></span></button>`).join("");

  const key = (c, t) => `<span class="key"><i style="background:var(${c})"></i>${t}</span>`;
  const melTop = `${fmtHz(p.sr / 2)} on a mel axis`;
  const rows = [
    ["wave", "Waveform", `${key("--c1", `original at ${fmtHz(p.sr)}`)} ${key("--c2", "processed")} ${p.bits < 32 ? key("--c3", "quantisation error") : ""}`, null],
    ["lin", "1 · STFT power", `${p.n_fft}-sample window every ${p.hop} samples · dB · linear axis to ${fmtHz(p.sr / 2)}`, "img_lin"],
    ["melpow", "2 · Mel power", `${p.n_mels} channels · linear energy, so a few loud bins dominate · ${melTop}`, "img_melpow"],
    ["logmel", "3 · Log-mel", `log(mel + 1e-6)${p.feat !== "fp32" ? `, stored as ${p.feat}` : ""}: quiet detail becomes visible`, "img_logmel"],
    ["norm", "4 · Normalised and stacked", `per-clip mean / variance (train.py uses train-set stats) · lines every ${p.stack} frames = one token (${st.ms_per_token.toFixed(0)} ms)`, "img_norm"],
  ];
  $("#lab-pipeline").innerHTML = rows.map(([k, t, s, img]) => `
    <div class="stage-head"><b>${t}</b> <span class="muted small">${s}</span></div>
    <div class="track">
      <div class="yaxis" id="lab-y-${k}"></div>
      <div class="plot ${img ? "spec" : "wave"}" data-row="${k}">
        ${img ? `<img src="${labFile(img)}" alt="${t}">` : `<canvas id="lab-wave"></canvas>`}
        ${k === "norm" ? `<div class="token-grid" id="lab-token-grid"></div><div class="patch-sel" id="lab-patch-sel"></div>` : ""}
        <div class="playhead"></div><div class="readout"></div>
      </div>
    </div>`).join("") + `<div class="track"><div class="yaxis"></div><div class="xaxis" id="lab-time-x"></div></div>`;

  const fmax = p.sr / 2;
  $("#lab-y-wave").innerHTML = yAxisHTML([{ pos: 0.98, label: "+1" }, { pos: 0.5, label: "0" }, { pos: 0.02, label: "−1" }]);
  $("#lab-y-lin").innerHTML = yAxisHTML(freqTicks(fmax, false));
  for (const k of ["melpow", "logmel", "norm"]) $(`#lab-y-${k}`).innerHTML = yAxisHTML(freqTicks(fmax, true));

  const tokPct = (100 * (p.stack * p.hop) / p.sr) / st.seconds;
  const grid = $("#lab-token-grid");
  grid.style.backgroundImage = `repeating-linear-gradient(to right, rgba(0,0,0,.7) 0 1px, transparent 1px ${tokPct}%)`;
  grid.hidden = tokPct < 0.5;  // too dense to read at tiny STACK

  $("#lab-fb-title").textContent = `${p.n_mels} filters${r.fb.step > 1 ? `, every ${ordinal(r.fb.step)} drawn` : ""} · ${p.n_fft / 2 + 1} FFT bins`;
  $("#lab-ltas-legend").innerHTML = [["--c1", `original (${fmtHz(st.native_sr)})`], ["--c2", `processed (${fmtHz(p.sr)})`], ["--c3", "rebuilt from the log-mel"]]
    .map(([c, t]) => `<span><i style="background:var(${c})"></i>${t}</span>`).join("");

  loadNorm();
  redraw();
}

function redraw() {
  if (!L.res || S.view !== "lab" || $("#lab-body").hidden) return;
  drawWave();
  $("#lab-time-x").innerHTML = timeAxisHTML($("#lab-time-x"), L.res.stats.seconds, L.res.params.start);
  drawPatch();
  drawFilterbank();
  drawLtas();
}

function drawWave() {
  const cv = $("#lab-wave");
  if (!cv) return;
  const { ctx, W, H } = fitCanvas(cv);
  const env = L.res.env, mid = H / 2, amp = (H / 2) * 0.95;
  ctx.fillStyle = tok("--grid"); ctx.fillRect(0, mid, W, 1);
  const band = (e, color, alpha = 1) => {
    if (!e.length) return;
    ctx.fillStyle = color; ctx.globalAlpha = alpha;
    const w = W / e.length;
    e.forEach(([mn, mx], i) => ctx.fillRect(i * w, mid - mx * amp, Math.max(1, w), Math.max(1, (mx - mn) * amp)));
    ctx.globalAlpha = 1;
  };
  // original as a filled band, processed as its outline on top, so both stay visible where they coincide
  band(env.orig, tok("--c1"), 0.85);
  const w = W / Math.max(1, env.proc.length);
  ctx.strokeStyle = tok("--c2"); ctx.lineWidth = 1;
  for (const side of [1, 0]) {
    ctx.beginPath();
    env.proc.forEach((mm, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, (i + 0.5) * w, mid - mm[side] * amp));
    ctx.stroke();
  }
  if (L.res.params.bits < 32) band(env.err, tok("--c3"));
}

// ---------------------------------------------------------------- token inspector
async function loadNorm() {
  const key = L.res.key;
  try {
    const buf = await (await api(labFile("norm_f32"))).arrayBuffer();
    if (L.res?.key !== key) return;
    L.norm = new Float32Array(buf);
    drawPatch();
  } catch (e) { toast(e.message, "error"); }
}

// RdBu reversed like the notebook: blue below the clip mean, red above, clamped to ±3 std
function rdbu(v) {
  const t = Math.max(-1, Math.min(1, v / 3)), u = Math.abs(t), a = [247, 247, 247], b = t < 0 ? [33, 102, 172] : [178, 24, 43];
  return `rgb(${a.map((c, i) => Math.round(c + (b[i] - c) * u)).join(",")})`;
}

function drawPatch() {
  if (!L.res) return;
  const p = L.res.params, st = L.res.stats, nm = p.n_mels, k = L.patch, tokSec = (p.stack * p.hop) / p.sr;
  $("#lab-patch-title").textContent = `token ${k + 1} of ${st.tokens} · ${(p.start + k * tokSec).toFixed(2)}–${(p.start + (k + 1) * tokSec).toFixed(2)} s`;
  const sel = $("#lab-patch-sel");
  if (sel) { sel.style.left = `${(100 * k * tokSec) / st.seconds}%`; sel.style.width = `${(100 * tokSec) / st.seconds}%`; }
  if (!L.norm) return;
  const vals = L.norm.subarray(k * p.stack * nm, (k + 1) * p.stack * nm);

  const img = fitCanvas($("#lab-patch-img"));
  const cw = img.W / p.stack, ch = img.H / nm;
  for (let f = 0; f < p.stack; f++)
    for (let b = 0; b < nm; b++) {
      img.ctx.fillStyle = rdbu(vals[f * nm + b] ?? 0);
      img.ctx.fillRect(f * cw, img.H - (b + 1) * ch, Math.ceil(cw), Math.ceil(ch));
    }

  const { ctx, W, H } = fitCanvas($("#lab-patch-vec"));
  const n = vals.length, padB = 16, y = (v) => (H - padB) / 2 - (Math.max(-4, Math.min(4, v)) / 4) * ((H - padB) / 2 - 6);
  ctx.strokeStyle = tok("--grid"); ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
  for (let f = 1; f < p.stack; f++) { const x = (f * nm * W) / n; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - padB); ctx.stroke(); }
  ctx.setLineDash([]);
  ctx.fillStyle = tok("--grid"); ctx.fillRect(0, y(0), W, 1);
  ctx.strokeStyle = tok("--c1"); ctx.lineWidth = 1.2; ctx.beginPath();
  for (let i = 0; i < n; i++) { const x = (i * W) / Math.max(1, n - 1); i ? ctx.lineTo(x, y(vals[i])) : ctx.moveTo(x, y(vals[i])); }
  ctx.stroke();
  ctx.fillStyle = tok("--muted");
  ctx.textAlign = "left"; ctx.fillText("dim 0", 2, H - 3);
  ctx.textAlign = "right"; ctx.fillText(`${n - 1}`, W - 2, H - 3);
  ctx.textAlign = "center"; ctx.fillText("±4 std", W / 2, 10);
}

function setPatch(k) {
  if (!L.res) return;
  L.patch = Math.max(0, Math.min(L.res.stats.tokens - 1, k));
  drawPatch();
}

// ---------------------------------------------------------------- line charts
function lineChart(cv, series, o) {
  const { ctx, W, H } = fitCanvas(cv);
  const m = { l: 40, r: 8, t: 8, b: 20 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
  const X = (v) => m.l + ((v - o.x0) / (o.x1 - o.x0)) * pw, Y = (v) => m.t + (1 - (v - o.y0) / (o.y1 - o.y0)) * ph;
  ctx.strokeStyle = tok("--grid"); ctx.fillStyle = tok("--muted"); ctx.lineWidth = 1;
  for (const v of o.yt) { ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(W - m.r, Y(v)); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(o.yf(v), m.l - 5, Y(v) + 3); }
  ctx.textAlign = "center";
  for (const v of o.xt) ctx.fillText(kLabel(v), Math.min(W - 14, X(v)), H - 5);
  ctx.save(); ctx.beginPath(); ctx.rect(m.l, m.t, pw, ph); ctx.clip();
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.globalAlpha = s.alpha ?? 1; ctx.beginPath();
    s.x.forEach((xv, i) => (i ? ctx.lineTo(X(xv), Y(s.y[i])) : ctx.moveTo(X(xv), Y(s.y[i]))));
    ctx.stroke();
  }
  ctx.restore(); ctx.globalAlpha = 1;
  cv._chart = { series, o, m, pw, W, H, X, Y };
}
const xTicks = (fmax) => freqTicks(fmax, false).map((t) => t.f);

function drawFilterbank() {
  const fb = L.res.fb, fmax = L.res.params.sr / 2, c = tok("--c1");
  lineChart($("#lab-fb"), fb.rows.map((y, i) => ({ x: fb.f, y, color: c, width: 1.2, alpha: i % 2 ? 0.55 : 1 })),
    { x0: 0, x1: fmax, y0: 0, y1: 1.05, xt: xTicks(fmax), yt: [0, 0.5, 1], yf: (v) => v, fmt: (v) => v.toFixed(2) });
}

function drawLtas() {
  const lt = L.res.ltas, fmax = Math.max(L.res.stats.native_sr, L.res.params.sr) / 2;
  const y1 = Math.ceil(Math.max(...lt.orig.db, ...lt.proc.db) / 10) * 10, y0 = y1 - 100, yt = [];
  for (let v = y0; v <= y1; v += 20) yt.push(v);
  lineChart($("#lab-ltas"), [
    { x: lt.orig.f, y: lt.orig.db, color: tok("--c1"), label: "original" },
    { x: lt.proc.f, y: lt.proc.db, color: tok("--c2"), label: "processed" },
    { x: lt.mel.f, y: lt.mel.db, color: tok("--c3"), label: "from log-mel" },
  ], { x0: 0, x1: fmax, y0, y1, xt: xTicks(fmax), yt, yf: (v) => `${v}`, fmt: (v) => `${v.toFixed(1)} dB` });
}

const redrawChart = (cv) => L.res && (cv.id === "lab-fb" ? drawFilterbank() : drawLtas());

function chartHover(cv, e) {
  const c = cv._chart;
  if (!c) return;
  redrawChart(cv);
  const px = e.clientX - cv.getBoundingClientRect().left, tip = cv.nextElementSibling;
  if (px < c.m.l || px > c.W - c.m.r) { tip.style.display = "none"; return; }
  const xv = c.o.x0 + ((px - c.m.l) / c.pw) * (c.o.x1 - c.o.x0), ctx = cv.getContext("2d");
  ctx.strokeStyle = tok("--muted"); ctx.beginPath(); ctx.moveTo(px, c.m.t); ctx.lineTo(px, c.H - c.m.b); ctx.stroke();
  const lines = [];
  for (const s of c.series.filter((s) => s.label)) {
    if (xv > s.x[s.x.length - 1]) continue;
    let j = 0;
    while (j < s.x.length - 1 && s.x[j + 1] <= xv) j++;
    ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(c.X(s.x[j]), c.Y(s.y[j]), 3.5, 0, 7); ctx.fill();
    lines.push(`${s.label}: ${c.o.fmt(s.y[j])}`);
  }
  tip.innerHTML = `<b>${fmtHz(xv)}</b>${lines.map((l) => `<br>${esc(l)}`).join("")}`;
  tip.style.display = "block";
  tip.style.left = `${Math.min(px + 12, c.W - 170)}px`;
}

// ---------------------------------------------------------------- pipeline interaction
function plotAt(e) {
  const box = e.target.closest(".plot");
  if (!box || !L.res) return null;
  const r = box.getBoundingClientRect();
  return { box, fx: Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), fy: 1 - (e.clientY - r.top) / r.height };
}

function onPipelineClick(e) {
  const at = plotAt(e);
  if (!at) return;
  const p = L.res.params, t = at.fx * L.res.stats.seconds;
  if (at.box.dataset.row === "norm") setPatch(Math.floor(t / ((p.stack * p.hop) / p.sr)));
  if (L.playing) { player.currentTime = t; tick(); }
}

function onPipelineHover(e) {
  const at = plotAt(e);
  if (!at) return;
  const p = L.res.params, row = at.box.dataset.row, fmax = p.sr / 2, sec = at.fx * L.res.stats.seconds;
  let txt = `${(p.start + sec).toFixed(2)} s`;
  if (row === "wave") txt += ` · ${(2 * at.fy - 1).toFixed(2)}`;
  else {
    txt += ` · ${fmtHz(row === "lin" ? at.fy * fmax : mel2hz(at.fy * hz2mel(fmax)))}`;
    if (row !== "lin") txt += ` · mel ${Math.min(p.n_mels - 1, Math.max(0, Math.floor(at.fy * p.n_mels)))}`;
    if (row === "norm") txt += ` · token ${Math.floor(sec / ((p.stack * p.hop) / p.sr)) + 1}`;
  }
  at.box.querySelector(".readout").textContent = txt;
}

// ---------------------------------------------------------------- playback
function play(name) {
  if (!L.res) return;
  if (L.playing === name && !player.paused) { player.pause(); return; }
  if (L.playing !== name) { player.src = labFile(`wav_${name}`); L.playing = name; }
  player.play().catch((e) => toast(e.message, "error"));
}
function stopPlayer() { player.pause(); cancelAnimationFrame(L.raf); tick(); }

function tick() {
  const d = player.duration || L.res?.stats.seconds || 1;
  const pos = `${(100 * player.currentTime) / d}%`;
  $$("#lab-pipeline .playhead").forEach((el) => (el.style.left = L.playing ? pos : "0"));
  $$(".play-card").forEach((b) => {
    const on = b.dataset.wav === L.playing && !player.paused;
    b.classList.toggle("on", b.dataset.wav === L.playing);
    b.querySelector(".ic").textContent = on ? "❚❚" : "▶";
  });
  if (!player.paused) L.raf = requestAnimationFrame(tick);
}
player.addEventListener("play", () => (L.raf = requestAnimationFrame(tick)));
player.addEventListener("pause", tick);
player.addEventListener("ended", tick);

export function onKey(e) {
  if (e.key === " ") { e.preventDefault(); play(L.playing || "orig"); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); setPatch(L.patch + (e.key === "ArrowLeft" ? -1 : 1)); }
  else if (e.key === "Enter") { e.preventDefault(); run(); }
}
