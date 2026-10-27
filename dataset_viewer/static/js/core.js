// Shared state, API helpers and formatting for every view. Plain ES modules, no build step.

export const $ = (s, el = document) => el.querySelector(s);
export const $$ = (s, el = document) => [...el.querySelectorAll(s)];
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// cfg: /api/bootstrap minus datasets; cur: the open sample {idx, data, dur}; sel: dragged region {a, b} in seconds
export const S = { cfg: null, datasets: [], ds: null, view: "review", cur: null, sel: null };

// ---------------------------------------------------------------- tiny event bus
const handlers = {};
export const on = (ev, fn) => (handlers[ev] ||= []).push(fn);
export const emit = (ev, ...args) => (handlers[ev] || []).forEach((fn) => fn(...args));

// ---------------------------------------------------------------- API
export async function api(path, { method = "GET", body, signal } = {}) {
  let r;
  try {
    r = await fetch(path, { method, signal, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new Error("The viewer server is not responding. Is `python -m dataset_viewer` still running?");
  }
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : j.detail?.map?.((d) => `${d.loc?.at(-1)}: ${d.msg}`).join("; ") || msg;
    } catch {}
    throw new Error(msg);
  }
  return r;
}
export const getJSON = async (path, opts) => (await api(path, opts)).json();
export const sendJSON = async (method, path, body) => (await api(path, { method, body })).json();
/** query string with the current dataset */
export const dsq = (extra = {}) => new URLSearchParams({ ds: S.ds, ...extra }).toString();

export const curDataset = () => S.datasets.find((d) => d.id === S.ds);

// ---------------------------------------------------------------- feedback
export function toast(msg, kind = "info") {
  const t = $("#toast");
  t.textContent = msg;
  t.dataset.kind = kind;
  t.classList.add("show");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("show"), kind === "error" ? 5000 : 1800);
}

/** In-page confirmation. Resolves true when the user presses the confirm button. */
export function confirmBox({ title, body, ok = "Confirm", danger = false }) {
  const dlg = $("#confirm-dlg");
  $("#confirm-title").textContent = title;
  $("#confirm-body").textContent = body;
  const btn = $("#confirm-ok");
  btn.textContent = ok;
  btn.classList.toggle("danger", danger);
  dlg.returnValue = "";
  dlg.showModal();
  return new Promise((res) => dlg.addEventListener("close", () => res(dlg.returnValue === "ok"), { once: true }));
}

// ---------------------------------------------------------------- formatting
export const fmtTime = (s) => {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, "0")}`;
};
export const fmtHz = (f) => (f >= 1000 ? `${+(f / 1000).toFixed(2)} kHz` : `${Math.round(f)} Hz`);
export const kLabel = (f) => (f >= 1000 ? `${+(f / 1000).toFixed(1)}k` : `${f}`);
export const n0 = (v) => Number(v).toLocaleString();
export const pct = (a, b) => (b ? `${((100 * a) / b).toFixed(1)}%` : "0%");

export const METRICS = {
  duration_s: ["Duration", (v) => `${v.toFixed(2)} s`],
  sample_rate: ["Sample rate", (v) => `${+(v / 1000).toFixed(3)} kHz`],
  channels: ["Channels", (v) => `${v}`],
  snr_db: ["Est. SNR", (v) => `${v.toFixed(1)} dB`],
  rms_dbfs: ["RMS level", (v) => `${v.toFixed(1)} dBFS`],
  peak_dbfs: ["Peak", (v) => `${v.toFixed(1)} dBFS`],
  clip_ratio: ["Clipped", (v) => `${(v * 100).toFixed(3)}%`],
  silence_ratio: ["Silence", (v) => `${(v * 100).toFixed(0)}%`],
  bandwidth_hz: ["Bandwidth", (v) => `${(v / 1000).toFixed(1)} kHz`],
  lead_silence_s: ["Lead silence", (v) => `${v.toFixed(2)} s`],
  trail_silence_s: ["Trail silence", (v) => `${v.toFixed(2)} s`],
  chars_per_sec: ["Text rate", (v) => `${v.toFixed(1)} ch/s`],
  dc_offset: ["DC offset", (v) => v.toFixed(4)],
};
export const metricLabel = (k) => METRICS[k]?.[0] ?? k;
export const metricFmt = (k, v) => (v == null ? "–" : METRICS[k] ? METRICS[k][1](v) : String(v));
export const flagDesc = (f) => S.cfg.thresholds[f]?.desc || f;
export const flagLabel = (f) => f.replaceAll("_", " ");

export function niceStep(span, maxTicks) {
  for (const s of [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]) if (span / s <= maxTicks) return s;
  return span / maxTicks;
}
export const hz2mel = (f) => 2595 * Math.log10(1 + f / 700);
export const mel2hz = (m) => 700 * (Math.pow(10, m / 2595) - 1);

/** frequency ticks for a y axis: mel-spaced or linear, positions in 0..1 */
export function freqTicks(fmax, mel) {
  let ticks;
  if (mel) ticks = [0, 250, 500, 1000, 2000, 4000, 8000, 16000].filter((f) => f <= fmax + 1);
  else {
    const st = Math.max(500, niceStep(fmax / 1000, 6) * 1000);
    ticks = [];
    for (let f = 0; f <= fmax + 1; f += st) ticks.push(f);
  }
  return ticks.map((f) => ({ f, pos: mel ? hz2mel(f) / hz2mel(fmax) : f / fmax }));
}
export const yAxisHTML = (ticks) => ticks.map(({ f, pos, label }) =>
  `<span class="${pos > 0.9 ? "top" : pos < 0.1 ? "bottom" : ""}" style="bottom:${100 * pos}%">${label ?? kLabel(f)}</span>`).join("");

export function timeAxisHTML(el, dur, offset = 0) {
  if (!dur) return "";
  const st = niceStep(dur, Math.max(2, Math.floor(el.clientWidth / 72)));
  let h = "";
  for (let t = 0; t <= dur + 1e-9; t += st) h += `<span style="left:${(100 * t) / dur}%">${+(offset + t).toFixed(2)} s</span>`;
  return h;
}

// ---------------------------------------------------------------- canvas
/** current value of a CSS colour token, e.g. tok("--accent") */
export const tok = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** size a canvas to its CSS box at device resolution; drawing happens in CSS pixels */
export function fitCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  cv.width = Math.max(1, Math.round(W * dpr));
  cv.height = Math.max(1, Math.round(H * dpr));
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = `11px ${tok("--font-mono") || "monospace"}`;
  return { ctx, W, H };
}

/** parse a mono float32 / int16 WAV into samples (the server always sends one of these) */
export function parseWav(buf) {
  const dv = new DataView(buf);
  let p = 12, fmt = null;
  while (p + 8 <= dv.byteLength) {
    const id = String.fromCharCode(dv.getUint8(p), dv.getUint8(p + 1), dv.getUint8(p + 2), dv.getUint8(p + 3));
    const size = dv.getUint32(p + 4, true);
    if (id === "fmt ") fmt = { ch: dv.getUint16(p + 10, true), sr: dv.getUint32(p + 12, true), bits: dv.getUint16(p + 22, true) };
    if (id === "data" && fmt) {
      const bytes = buf.slice(p + 8, p + 8 + Math.min(size, dv.byteLength - p - 8));
      if (fmt.bits === 32) return { x: new Float32Array(bytes), sr: fmt.sr };
      const i16 = new Int16Array(bytes), x = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) x[i] = i16[i] / 32768;
      return { x, sr: fmt.sr };
    }
    p += 8 + size + (size % 2);
  }
  throw new Error("unreadable WAV");
}

export const isTyping = (e) => e.target.matches?.("input:not([type=checkbox]):not([type=radio]):not([type=range]), textarea, select") || !!document.querySelector("dialog[open]");
