// Quality report: background scan control, summary tiles, automatic-check table and metric histograms.
import { $, S, esc, on, emit, getJSON, sendJSON, dsq, toast, confirmBox, curDataset, n0, pct, metricLabel, metricFmt, flagDesc, flagLabel, tok, fitCanvas } from "./core.js";
import { openBulk } from "./review.js";

let rep = null, poll = 0, lastLoad = 0;
// for these metrics a low value is the bad end; for the rest a high one
const LOW_IS_BAD = ["snr_db", "rms_dbfs", "bandwidth_hz", "duration_s"];
const HIST_ORDER = ["duration_s", "snr_db", "rms_dbfs", "peak_dbfs", "silence_ratio", "clip_ratio", "bandwidth_hz", "chars_per_sec", "lead_silence_s", "trail_silence_s", "dc_offset"];

export function init() {
  $("#scan-start").addEventListener("click", startScan);
  $("#scan-cancel").addEventListener("click", () => sendJSON("DELETE", `/api/scan?${dsq()}`).catch((e) => toast(e.message, "error")));
  $("#scan-clear").addEventListener("click", clearMetrics);
  $("#report-body").addEventListener("click", (e) => {
    const v = e.target.closest("[data-view-flag]"), b = e.target.closest("[data-bulk-flag]"), s = e.target.closest("[data-sort-metric]");
    if (v) emit("filter-review", { flag: v.dataset.viewFlag });
    if (b) openBulk({ q: "", status: "all", flag: b.dataset.bulkFlag });
    if (s) emit("filter-review", { sort: s.dataset.sortMetric, desc: !LOW_IS_BAD.includes(s.dataset.sortMetric) });
  });
  on("dataset-changed", () => { rep = null; if (S.view === "report") show(); else watchScan(); });
  on("dataset-updated", renderScan);
  on("bulk-done", () => S.view === "report" && load());
  on("resize", () => S.view === "report" && drawHists());
  on("theme-changed", () => S.view === "report" && drawHists());
}

export function show() {
  if (!S.ds) return;
  $("#csv-link").href = `/api/export/csv?${dsq()}`;
  load();
  watchScan();
}

async function load() {
  const body = $("#report-body");
  if (!rep) body.innerHTML = `<div class="empty-state">Loading…</div>`;
  lastLoad = Date.now();
  const forDs = S.ds;
  try {
    const r = await getJSON(`/api/report?${dsq()}`);
    if (forDs !== S.ds) return;
    rep = r;
    render();
  } catch (e) { body.innerHTML = `<div class="empty-state">${esc(e.message)}</div>`; }
}

function render() {
  const body = $("#report-body"), M = rep.metrics;
  const rv = rep.reviews, nrev = (rv.keep || 0) + (rv.reject || 0) + (rv.flag || 0);
  const reviewTile = tile("Your reviews", n0(nrev), nrev ? `${rv.keep || 0} kept · ${rv.reject || 0} rejected · ${rv.flag || 0} flagged` : "none yet");
  if (!rep.scanned) {
    body.innerHTML = `<div class="tiles">${reviewTile}</div>
      <div class="empty-state">No metrics for this dataset yet. Start a scan above: 1,000 random rows take well under a minute for most corpora.</div>`;
    return;
  }
  const srs = Object.entries(rep.sample_rates).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${+(k / 1000).toFixed(3)} kHz ${pct(v, rep.scanned)}`).join(" · ");
  const flagShare = rep.any_flag / rep.scanned;
  let h = `<div class="tiles">
    ${tile("Scanned", n0(rep.scanned), `of ${n0(rep.n_rows)} rows (${pct(rep.scanned, rep.n_rows)})`)}
    ${tile("Estimated audio", rep.est_total_hours == null ? "–" : `${rep.est_total_hours.toFixed(1)} h`, M.duration_s ? `mean ${M.duration_s.mean.toFixed(2)} s × ${n0(rep.n_rows)} rows` : "")}
    ${tile("Fail at least one check", pct(rep.any_flag, rep.scanned), `${n0(rep.any_flag)} of ${n0(rep.scanned)} scanned`, flagShare > 0.2 ? "warn" : "good")}
    ${tile("Median est. SNR", M.snr_db ? `${M.snr_db.p["50"].toFixed(1)} dB` : "–", M.snr_db ? `5th percentile ${M.snr_db.p["5"].toFixed(1)} dB` : "")}
    ${tile("Sample rates", Object.keys(rep.sample_rates).length, srs)}
    ${reviewTile}
  </div>`;

  const flags = Object.entries(rep.flags);
  h += `<h2 class="section-title">Automatic checks <span class="muted">share of scanned rows that fail each one; a row can fail several</span></h2>`;
  if (flags.length) {
    const max = Math.max(...flags.map((f) => f[1]));
    h += `<section class="panel"><div class="table-wrap"><table class="flag-table">
      <tr><th>Check</th><th>Rows</th><th>Share</th><th style="width:20%"></th><th>Fails when</th><th></th></tr>
      ${flags.map(([f, n]) => `<tr><td class="mono">${flagLabel(f)}</td><td>${n0(n)}</td><td>${pct(n, rep.scanned)}</td>
        <td><div class="meter" style="width:${(100 * n) / max}%"></div></td><td class="muted small">${esc(flagDesc(f))}</td>
        <td class="acts"><button class="btn small" data-view-flag="${f}">Review these</button> <button class="btn small" data-bulk-flag="${f}">Bulk…</button></td></tr>`).join("")}
    </table></div></section>`;
  } else h += `<section class="panel muted">No scanned row fails any check.</section>`;

  if (Object.keys(rep.tags).length) {
    h += `<h2 class="section-title">Reasons you gave</h2><section class="panel"><div class="chips">${Object.entries(rep.tags)
      .map(([t, n]) => `<span class="chip on">${flagLabel(t)} · ${n}</span>`).join("")}</div></section>`;
  }

  h += `<h2 class="section-title">Distributions <span class="muted">scanned rows · dashed line = check threshold · range clipped to the 0.5–99.5th percentile</span></h2><div class="hist-grid">`;
  for (const k of HIST_ORDER) {
    const s = M[k];
    if (!s) continue;
    h += `<section class="panel hist"><div class="hist-head"><h3 class="panel-title">${metricLabel(k)}</h3>
      <button class="link-btn" data-sort-metric="${k}" title="Open Review sorted by this metric, worst first">Review the worst</button></div>
      <canvas data-hist="${k}" role="img" aria-label="Histogram of ${metricLabel(k)}"></canvas>
      <div class="pcts">p5 ${metricFmt(k, s.p["5"])} · median ${metricFmt(k, s.p["50"])} · p95 ${metricFmt(k, s.p["95"])}</div></section>`;
  }
  body.innerHTML = h + `</div>`;
  drawHists();
}

const tile = (k, v, s, cls = "") => `<div class="tile ${cls}"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;

function drawHists() {
  if (!rep?.scanned) return;
  for (const cv of $("#report-body").querySelectorAll("canvas[data-hist]")) drawHist(cv, cv.dataset.hist, rep.metrics[cv.dataset.hist]);
}

function drawHist(cv, key, s) {
  const { ctx, W, H } = fitCanvas(cv);
  const padB = 16, { counts, lo, hi } = s.hist, max = Math.max(...counts, 1), bw = W / counts.length;
  const thr = Object.entries(S.cfg.thresholds).filter(([, t]) => t.metric === key);
  const cOk = tok("--c1"), cBad = tok("--flag"), cMuted = tok("--muted"), cLine = tok("--reject");
  ctx.fillStyle = tok("--grid");
  ctx.fillRect(0, H - padB, W, 1);
  counts.forEach((c, i) => {
    const center = lo + ((i + 0.5) / counts.length) * (hi - lo);
    const bad = thr.some(([, t]) => (t.op === "<" ? center < t.value : center > t.value));
    const bh = (c / max) * (H - padB - 14);
    ctx.fillStyle = bad ? cBad : cOk;
    ctx.fillRect(i * bw + 1, H - padB - bh, Math.max(1, bw - 2), bh);
  });
  ctx.fillStyle = cMuted;
  ctx.textAlign = "left"; ctx.fillText(metricFmt(key, lo), 2, H - 3);
  ctx.textAlign = "right"; ctx.fillText(metricFmt(key, hi), W - 2, H - 3);
  for (const [name, t] of thr) {
    if (t.value < lo || t.value > hi) continue;
    const x = ((t.value - lo) / (hi - lo)) * W, right = x > W / 2;
    ctx.strokeStyle = cLine; ctx.setLineDash([4, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - padB); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = cLine; ctx.textAlign = right ? "right" : "left";
    ctx.fillText(flagLabel(name), x + (right ? -4 : 4), 10);
  }
}

// ---------------------------------------------------------------- scans
async function startScan() {
  try {
    await sendJSON("POST", `/api/scan?${dsq({ n: $("#scan-n").value, seed: Date.now() % 100000 })}`);
    watchScan();
  } catch (e) { toast(e.message, "error"); }
}

async function clearMetrics() {
  const ok = await confirmBox({ title: "Reset metrics?", body: `Deletes every cached metric for ${S.ds}. Your keep / reject / flag reviews stay.`, ok: "Reset metrics", danger: true });
  if (!ok) return;
  try {
    await sendJSON("DELETE", `/api/metrics?${dsq()}`);
    rep = null;
    emit("refresh-dataset");
    load();
  } catch (e) { toast(e.message, "error"); }
}

/** poll the dataset summary once a second while a scan of it runs */
function watchScan() {
  clearInterval(poll);
  const forDs = S.ds;
  const step = () => (S.ds === forDs ? emit("refresh-dataset") : clearInterval(poll));  // renderScan stops it
  step();
  poll = setInterval(step, 1000);
}

function renderScan(d) {
  if (d.id !== S.ds) return;
  const sc = d.scan, running = !!sc?.running;
  if (!running) clearInterval(poll);
  $("#scan-start").hidden = running;
  $("#scan-cancel").hidden = !running;
  const bar = $("#scan-progress");
  bar.hidden = !sc;
  if (sc) {
    bar.querySelector(".fill").style.width = `${(100 * sc.done) / Math.max(1, sc.total)}%`;
    const eta = running && sc.rate_per_s ? ` · about ${Math.ceil((sc.total - sc.done) / sc.rate_per_s)} s left` : "";
    const end = sc.cancelled ? " · cancelled" : running ? "" : " · done";
    bar.querySelector("span").textContent = `${n0(sc.done)} of ${n0(sc.total)} rows · ${sc.rate_per_s} rows/s${eta}${sc.errors ? ` · ${sc.errors} decode errors` : ""}${end}`;
  }
  if (S.view !== "report") return;
  // refresh the report every 5 s during a scan, and once when it ends
  if ((running && Date.now() - lastLoad > 5000) || (!running && sc && rep && rep.scanned !== d.scanned)) load();
}
