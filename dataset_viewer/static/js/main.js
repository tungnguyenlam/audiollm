// App shell: datasets rail, dataset header, view switching, URL hash, keyboard routing, theme.
import { $, $$, S, esc, on, emit, getJSON, sendJSON, dsq, toast, curDataset, n0, isTyping } from "./core.js";
import * as review from "./review.js";
import * as report from "./report.js";
import * as lab from "./lab.js";

const VIEWS = { review, report, lab };

// ---------------------------------------------------------------- datasets rail
function renderRail() {
  const q = $("#ds-filter").value.trim().toLowerCase();
  let html = "", lastCorpus = null;
  for (const d of S.datasets) {
    if (q && !d.id.toLowerCase().includes(q)) continue;
    if (d.corpus !== lastCorpus) { html += `<div class="corpus eyebrow">${esc(d.corpus)}</div>`; lastCorpus = d.corpus; }
    const r = d.reviews, rev = r.keep + r.reject + r.flag;
    const scan = d.scan?.running ? `scanning ${Math.round((100 * d.scan.done) / Math.max(1, d.scan.total))}%` : d.scanned ? `${n0(d.scanned)} scanned` : "";
    const sub = [rev ? `${r.keep} kept · ${r.reject} rejected · ${r.flag} flagged` : "", scan].filter(Boolean).join(" · ");
    const bar = rev ? `<div class="bar"><i style="width:${(100 * r.keep) / rev}%;background:var(--keep)"></i><i style="width:${(100 * r.flag) / rev}%;background:var(--flag)"></i><i style="width:${(100 * r.reject) / rev}%;background:var(--reject)"></i></div>` : "";
    html += `<button class="ds-item ${d.id === S.ds ? "active" : ""}" data-id="${esc(d.id)}" title="${esc(d.decision.note || d.id)}">
      <div class="top"><i class="dot ${d.decision.decision}"></i><span class="name">${esc(d.split)}</span><span class="n">${n0(d.n_rows)}</span></div>
      ${sub ? `<div class="sub">${sub}</div>` : ""}${bar}</button>`;
  }
  $("#ds-list").innerHTML = html || `<div class="empty-state">${S.datasets.length ? "No dataset matches." : "No datasets found under data/."}</div>`;
}

function renderHeader() {
  const d = curDataset();
  if (!d) return;
  $("#ds-name").textContent = d.id;
  const labels = d.label_columns.length ? `labels: ${d.label_columns.join(", ")}` : "no label column";
  $("#ds-sub").textContent = `${d.kind} · ${n0(d.n_rows)} rows · ${labels}`;
  $$("#decision-seg button").forEach((b) => {
    b.classList.toggle("on", b.dataset.v === d.decision.decision);
    b.setAttribute("aria-checked", b.dataset.v === d.decision.decision);
  });
  if (document.activeElement !== $("#decision-note")) $("#decision-note").value = d.decision.note || "";
  document.title = `${d.id} · Dataset Viewer`;
}

/** re-fetch one dataset summary (review counts, scan state) and redraw what shows it */
async function refreshDataset() {
  if (!S.ds) return null;
  const d = await getJSON(`/api/dataset?${dsq()}`);
  const i = S.datasets.findIndex((x) => x.id === d.id);
  if (i >= 0) S.datasets[i] = d;
  renderRail();
  renderHeader();
  emit("dataset-updated", d);
  return d;
}
on("refresh-dataset", refreshDataset);

async function selectDataset(id, idx = null) {
  S.ds = id;
  S.cur = null;
  S.sel = null;
  $(".app").classList.remove("rail-open");
  renderRail();
  renderHeader();
  writeHash();
  emit("dataset-changed", idx);
}

async function saveDecision(decision, note) {
  const d = curDataset();
  try {
    await sendJSON("PUT", `/api/dataset/decision?${dsq()}`, { decision: decision ?? d.decision.decision, note: note ?? $("#decision-note").value });
    await refreshDataset();
    toast(decision ? `Marked ${decision}` : "Note saved");
  } catch (e) { toast(e.message, "error"); }
}

// ---------------------------------------------------------------- views + hash
function setView(name) {
  if (!VIEWS[name]) name = "review";
  const prev = S.view;
  S.view = name;
  $$("#views > button").forEach((b) => { b.classList.toggle("on", b.dataset.view === name); b.setAttribute("aria-selected", b.dataset.view === name); });
  for (const v of Object.keys(VIEWS)) $(`#view-${v}`).hidden = v !== name;
  if (prev !== name) VIEWS[prev]?.hide?.();
  VIEWS[name].show?.();
  writeHash();
}
on("set-view", setView);

function writeHash() {
  const p = new URLSearchParams();
  if (S.ds) p.set("ds", S.ds);
  if (S.cur) p.set("idx", S.cur.idx);
  if (S.view !== "review") p.set("view", S.view);
  history.replaceState(null, "", `#${p}`);
}
on("sample-changed", writeHash);

// ---------------------------------------------------------------- theme
function applyTheme(t) {
  if (t) document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  emit("theme-changed");
}
function toggleTheme() {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const t = dark ? "light" : "dark";
  try { localStorage.setItem("dv-theme", t); } catch {}
  applyTheme(t);
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => emit("theme-changed"));

// ---------------------------------------------------------------- boot
async function boot() {
  try { applyTheme(localStorage.getItem("dv-theme")); } catch {}
  let boot;
  try { boot = await getJSON("/api/bootstrap"); }
  catch (e) { $("#ds-name").textContent = "Cannot reach the viewer server"; $("#ds-sub").textContent = e.message; return; }
  S.datasets = boot.datasets;
  delete boot.datasets;
  S.cfg = boot;
  for (const v of Object.values(VIEWS)) v.init?.();

  $("#ds-filter").addEventListener("input", renderRail);
  $("#ds-list").addEventListener("click", (e) => { const it = e.target.closest(".ds-item"); if (it && it.dataset.id !== S.ds) selectDataset(it.dataset.id); });
  $$("#decision-seg button").forEach((b) => b.addEventListener("click", () => saveDecision(b.dataset.v)));
  $("#decision-note").addEventListener("change", () => saveDecision(null, $("#decision-note").value));
  $("#decision-note").addEventListener("keydown", (e) => { if (e.key === "Enter") e.target.blur(); });
  $$("#views > button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $("#help-btn").addEventListener("click", () => $("#help-dlg").showModal());
  $("#theme-btn").addEventListener("click", toggleTheme);
  $("#rail-toggle").addEventListener("click", () => $(".app").classList.toggle("rail-open"));

  window.addEventListener("keydown", (e) => {
    if (isTyping(e)) { if (e.key === "Escape") e.target.blur?.(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "?") { e.preventDefault(); $("#help-dlg").showModal(); return; }
    VIEWS[S.view].onKey?.(e);
  });
  let rT;
  window.addEventListener("resize", () => { clearTimeout(rT); rT = setTimeout(() => emit("resize"), 80); });

  const h = new URLSearchParams(location.hash.slice(1));
  const start = S.datasets.find((d) => d.id === h.get("ds")) || S.datasets.find((d) => d.id === "LibriSpeech/dev-clean") || S.datasets[0];
  setView(h.get("view") || "review");
  if (start) await selectDataset(start.id, h.has("idx") ? +h.get("idx") : null);
  else renderRail();
}
boot();
