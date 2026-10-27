"""FastAPI backend for the dataset viewer. All dataset ids go in the `ds` query parameter (they contain '/')."""
import csv
import io
import random
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from dataset_viewer import frontend_lab
from dataset_viewer.audio import THRESHOLDS, compute_metrics, linear_spec_db, log_mel, render_png
from dataset_viewer.sources import discover
from dataset_viewer.store import DATASET_DECISIONS, METRIC_FIELDS, SAMPLE_STATUSES, Store

STATIC = Path(__file__).parent / "static"
REVIEW_TAGS = ["wrong_transcript", "partial_transcript", "wrong_label", "noisy", "clipped_distorted",
               "music_background", "overlapping_speech", "mostly_silence", "wrong_language", "codec_artifacts",
               "truncated", "other"]
SORTABLE = ("index", "random", "status") + METRIC_FIELDS + ("n_flags",)


def create_app(data_root="data", db_path="data/curation.sqlite3", scan_workers=4):
    datasets = discover(data_root)
    store = Store(db_path)
    scans = {}  # dataset_id -> ScanJob
    audio_cache = OrderedDict()
    audio_lock = threading.Lock()
    lab_cache = OrderedDict()  # key -> {name: bytes} of the last front-end lab results
    app = FastAPI(title="Audio dataset viewer")

    # ---- helpers -------------------------------------------------------------------------------------------
    def get_ds(ds):
        d = datasets.get(ds)
        if d is None:
            raise HTTPException(404, f"unknown dataset {ds!r}")
        return d

    def check_idx(d, idx):
        if not 0 <= idx < d.n_rows:
            raise HTTPException(404, f"row {idx} out of range 0..{d.n_rows - 1}")

    def load_audio(d, idx):
        k = (d.id, idx)
        with audio_lock:
            if k in audio_cache:
                audio_cache.move_to_end(k)
                return audio_cache[k]
        val = d.audio(idx)
        with audio_lock:
            audio_cache[k] = val
            while len(audio_cache) > 8:
                audio_cache.popitem(last=False)
        return val

    def metrics_for(d, idx, x=None, sr=None, ch=1):
        if x is None:
            try:
                x, sr, ch = load_audio(d, idx)
            except Exception as e:
                return {f: None for f in METRIC_FIELDS} | {"flags": ["decode_error"], "error": str(e)}
        return compute_metrics(x, sr, ch, d.transcript(idx), bool(d.transcript_column))

    def label_text(d, idx):
        parts = []
        for c in d.label_columns:
            v = d.meta_column(c)[idx]
            if v is not None and v != "":
                parts.append(str(v) if len(d.label_columns) == 1 else f"{c}: {v}")
        return " | ".join(parts)

    def filtered_rows(d, q="", status="all", flag="all", sort="index", desc=False, seed=0):
        """Row indices matching the filters, in display order."""
        revs, mets = store.reviews(d.id), store.metrics(d.id)
        idxs = range(d.n_rows)
        if status != "all":
            idxs = [i for i in idxs if revs.get(i, {}).get("status", "unreviewed") == status]
        if flag == "unscanned":
            idxs = [i for i in idxs if i not in mets]
        elif flag == "scanned":
            idxs = [i for i in idxs if i in mets]
        elif flag == "any":
            idxs = [i for i in idxs if mets.get(i, {}).get("flags")]
        elif flag == "none":
            idxs = [i for i in idxs if i in mets and not mets[i]["flags"]]
        elif flag != "all":
            idxs = [i for i in idxs if flag in mets.get(i, {}).get("flags", ())]
        if q:
            ql = q.lower()
            cols = [d.meta_column(c) for c in d.label_columns]
            ids = d.meta_column(d.id_column) if getattr(d, "id_column", None) else d.meta_column("id")
            idxs = [i for i in idxs if ql in str(ids[i]).lower() or ql in d.key(i).lower()
                    or any(ql in str(col[i]).lower() for col in cols)]
        idxs = list(idxs)
        if sort == "random":
            random.Random(seed).shuffle(idxs)
        elif sort == "status":
            order = {"reject": 0, "flag": 1, "keep": 2}
            idxs.sort(key=lambda i: order.get(revs.get(i, {}).get("status"), 3), reverse=desc)
        elif sort == "n_flags":
            idxs = [i for i in idxs if i in mets]
            idxs.sort(key=lambda i: len(mets[i]["flags"]), reverse=desc)
        elif sort in METRIC_FIELDS:  # only scanned rows can be sorted by a metric
            idxs = [i for i in idxs if mets.get(i, {}).get(sort) is not None]
            idxs.sort(key=lambda i: mets[i][sort], reverse=desc)
        elif desc:
            idxs.reverse()
        return idxs

    def ds_summary(d, decisions):
        revs, mets = store.reviews(d.id), store.metrics(d.id)
        c = Counter(r["status"] for r in revs.values())
        flagged = sum(1 for m in mets.values() if m["flags"])
        job = scans.get(d.id)
        return d.info() | {
            "decision": decisions.get(d.id, {"decision": "undecided", "note": "", "updated_at": None}),
            "reviews": {s: c.get(s, 0) for s in SAMPLE_STATUSES},
            "scanned": len(mets), "auto_flagged": flagged,
            "scan": job.state() if job else None,
        }

    # ---- background quality scan ---------------------------------------------------------------------------
    class ScanJob:
        def __init__(self, d, idxs):
            self.d, self.idxs = d, idxs
            self.done = self.errors = 0
            self.lock = threading.Lock()
            self.cancelled = False
            self.started = time.time()
            self.finished = None
            threading.Thread(target=self.run, daemon=True).start()

        def run(self):
            # Contiguous chunks so each worker streams through its own row groups (good cache locality).
            chunk = max(1, min(64, len(self.idxs) // (scan_workers * 4) or 1))
            chunks = [self.idxs[i:i + chunk] for i in range(0, len(self.idxs), chunk)]

            def work(ch):
                rows = []
                for idx in ch:
                    if self.cancelled:
                        break
                    try:
                        x, sr, nch = self.d.audio(idx)
                        m = compute_metrics(x, sr, nch, self.d.transcript(idx), bool(self.d.transcript_column))
                    except Exception:
                        m = {"flags": ["decode_error"]}
                        with self.lock:
                            self.errors += 1
                    rows.append((idx, self.d.key(idx), m))
                    with self.lock:
                        self.done += 1
                store.put_metrics(self.d.id, rows)

            with ThreadPoolExecutor(scan_workers) as ex:
                list(ex.map(work, chunks))
            self.finished = time.time()

        def state(self):
            el = (self.finished or time.time()) - self.started
            return {"total": len(self.idxs), "done": self.done, "errors": self.errors,
                    "running": self.finished is None, "cancelled": self.cancelled,
                    "elapsed_s": round(el, 1), "rate_per_s": round(self.done / el, 1) if el > 0 else 0}

    # ---- API: metadata -------------------------------------------------------------------------------------
    @app.get("/api/config")
    def config():
        return {"thresholds": {k: {"metric": v[0], "op": v[1], "value": v[2], "desc": v[3]}
                               for k, v in THRESHOLDS.items()},
                "tags": REVIEW_TAGS, "statuses": SAMPLE_STATUSES, "decisions": DATASET_DECISIONS,
                "metrics": METRIC_FIELDS, "sortable": SORTABLE}

    @app.get("/api/datasets")
    def list_datasets():
        dec = store.decisions()
        return [ds_summary(d, dec) for d in datasets.values()]

    @app.get("/api/dataset")
    def dataset(ds: str):
        return ds_summary(get_ds(ds), store.decisions())

    @app.put("/api/dataset/decision")
    def set_decision(ds: str, body: dict = Body(...)):
        get_ds(ds)
        decision = body.get("decision", "undecided")
        if decision not in DATASET_DECISIONS:
            raise HTTPException(400, f"decision must be one of {DATASET_DECISIONS}")
        return store.set_decision(ds, decision, str(body.get("note", "")))

    # ---- API: samples --------------------------------------------------------------------------------------
    @app.get("/api/samples")
    def samples(ds: str, offset: int = 0, limit: int = Query(100, le=500), q: str = "", status: str = "all",
                flag: str = "all", sort: str = "index", desc: bool = False, seed: int = 0):
        d = get_ds(ds)
        if sort not in SORTABLE:
            raise HTTPException(400, f"sort must be one of {SORTABLE}")
        idxs = filtered_rows(d, q, status, flag, sort, desc, seed)
        revs, mets = store.reviews(d.id), store.metrics(d.id)
        rows = []
        for i in idxs[offset:offset + limit]:
            m = mets.get(i)
            r = revs.get(i, {})
            rows.append({"idx": i, "key": d.key(i), "id": d.display_id(i), "label": label_text(d, i)[:300],
                         "status": r.get("status", "unreviewed"), "tags": r.get("tags", []),
                         "duration_s": m and m["duration_s"], "snr_db": m and m["snr_db"],
                         "flags": m["flags"] if m else None})
        return {"total": len(idxs), "offset": offset, "rows": rows}

    @app.get("/api/sample")
    def sample(ds: str, idx: int):
        d = get_ds(ds)
        check_idx(d, idx)
        m = metrics_for(d, idx)
        if "error" not in m:
            store.put_metrics(d.id, [(idx, d.key(idx), m)])
        meta = d.row_meta(idx)
        labels = {c: meta.get(c) for c in d.label_columns}
        r = store.reviews(d.id).get(idx, {"status": "unreviewed", "tags": [], "note": ""})
        return {"idx": idx, "key": d.key(idx), "id": d.display_id(idx), "labels": labels, "meta": meta,
                "metrics": m, "review": r, "n_rows": d.n_rows}

    @app.get("/api/audio")
    def audio(ds: str, idx: int):
        d = get_ds(ds)
        check_idx(d, idx)
        x, sr, _ = load_audio(d, idx)
        buf = io.BytesIO()
        # float WAV: exact samples (no requantisation), so what you hear/see is what the model gets
        sf.write(buf, x, sr, format="WAV", subtype="FLOAT")
        return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/spectrogram")
    def spectrogram(ds: str, idx: int, kind: str = Query("mel", pattern="^(mel|linear)$"),
                    cmap: str = Query("magma", pattern="^(magma|viridis|inferno|gray_r|turbo)$")):
        d = get_ds(ds)
        check_idx(d, idx)
        x, sr, _ = load_audio(d, idx)
        if kind == "mel":
            png, fmax = render_png(log_mel(x, sr), cmap, dynamic_range=None), 8000
        else:
            png, fmax = render_png(linear_spec_db(x, sr), cmap, dynamic_range=90), sr / 2
        return Response(png, media_type="image/png",
                        headers={"X-Freq-Max": str(fmax), "Cache-Control": "max-age=3600"})

    # ---- API: front-end lab --------------------------------------------------------------------------------
    @app.get("/api/lab/presets")
    def lab_presets():
        return {"presets": frontend_lab.PRESETS, "feature_precisions": frontend_lab.FEATURE_PRECISIONS,
                "max_sec": frontend_lab.MAX_SEC}

    @app.get("/api/lab")
    def lab(ds: str, idx: int, sr: int = 16000, bits: int = 32, quant: str = "pcm", n_fft: int = 400, hop: int = 160,
            n_mels: int = 80, stack: int = 16, feat: str = "fp32", start: float = 0.0, dur: float = 8.0,
            hidden: int = 1024):
        """Resample / quantise / featurise one clip with the given front-end and return plots, scores and audio."""
        d = get_ds(ds)
        check_idx(d, idx)
        p = frontend_lab.validate(dict(sr=sr, bits=bits, quant=quant, n_fft=n_fft, hop=hop, n_mels=n_mels,
                                       stack=stack, feat=feat, start=start, dur=dur))
        key = f"{len(lab_cache)}-{time.time_ns()}"
        x, x_sr, _ = load_audio(d, idx)
        try:
            summary, files = frontend_lab.analyse(x, x_sr, p, hidden_size=hidden)
        except ValueError as e:
            raise HTTPException(400, str(e))
        with audio_lock:
            lab_cache[key] = files
            while len(lab_cache) > 12:
                lab_cache.popitem(last=False)
        return summary | {"key": key, "files": sorted(files)}

    @app.get("/api/lab/file")
    def lab_file(key: str, name: str):
        files = lab_cache.get(key)
        if files is None or name not in files:
            raise HTTPException(404, "lab result expired; run it again")
        kind = "image/png" if name.startswith("img_") else "audio/wav" if name.startswith("wav_") else "application/octet-stream"
        return Response(files[name], media_type=kind, headers={"Cache-Control": "max-age=3600"})

    @app.put("/api/review")
    def review(ds: str, idx: int, body: dict = Body(...)):
        d = get_ds(ds)
        check_idx(d, idx)
        status = body.get("status")
        if status not in SAMPLE_STATUSES + ("unreviewed",):
            raise HTTPException(400, f"status must be one of {SAMPLE_STATUSES + ('unreviewed',)}")
        r = store.set_reviews(d.id, [(idx, d.key(idx))], status, body.get("tags"), body.get("note"))[0]
        return r or {"status": "unreviewed", "tags": [], "note": ""}

    @app.post("/api/bulk_review")
    def bulk_review(ds: str, body: dict = Body(...)):
        """Apply one status (and optional tag) to every row matching the given list filter."""
        d = get_ds(ds)
        f = body.get("filter", {})
        status = body.get("status")
        if status not in SAMPLE_STATUSES + ("unreviewed",):
            raise HTTPException(400, "bad status")
        if f.get("flag", "all") in ("all", "unscanned") and not f.get("q") and f.get("status", "all") == "all":
            raise HTTPException(400, "refusing to bulk-review a whole dataset; filter by flag, search or status")
        idxs = filtered_rows(d, f.get("q", ""), f.get("status", "all"), f.get("flag", "all"))
        revs = store.reviews(d.id)
        tag = body.get("tag")
        if tag:  # add the tag per row, keep each row's existing tags/note
            for i in idxs:
                tags = sorted(set(revs.get(i, {}).get("tags", [])) | {tag})
                store.set_reviews(d.id, [(i, d.key(i))], status, tags, None)
        else:
            store.set_reviews(d.id, [(i, d.key(i)) for i in idxs], status)
        return {"updated": len(idxs)}

    # ---- API: quality scan + report ------------------------------------------------------------------------
    @app.post("/api/scan")
    def start_scan(ds: str, n: int = Query(500, ge=1), seed: int = 0):
        """Compute metrics for n random not-yet-scanned rows (n >= n_rows scans everything)."""
        d = get_ds(ds)
        job = scans.get(ds)
        if job and job.finished is None:
            raise HTTPException(409, "a scan of this dataset is already running")
        done = store.metrics(ds)
        todo = [i for i in range(d.n_rows) if i not in done]
        if n < len(todo):
            todo = sorted(random.Random(seed).sample(todo, n))
        scans[ds] = ScanJob(d, todo)
        return scans[ds].state()

    @app.delete("/api/scan")
    def cancel_scan(ds: str):
        job = scans.get(ds)
        if job:
            job.cancelled = True
        return job.state() if job else {}

    @app.delete("/api/metrics")
    def clear_metrics(ds: str):
        get_ds(ds)
        store.clear_metrics(ds)
        return {"ok": True}

    @app.get("/api/report")
    def report(ds: str):
        d = get_ds(ds)
        mets = store.metrics(ds)
        revs = store.reviews(ds)
        n = len(mets)
        out = {"n_rows": d.n_rows, "scanned": n, "metrics": {}, "flags": {}, "sample_rates": {},
               "reviews": dict(Counter(r["status"] for r in revs.values())),
               "tags": dict(Counter(t for r in revs.values() for t in r["tags"]))}
        if not n:
            return out
        vals = list(mets.values())
        for f in METRIC_FIELDS:
            arr = np.array([v[f] for v in vals if v.get(f) is not None], dtype=float)
            if f in ("sample_rate", "channels") or not len(arr):
                continue
            lo, hi = np.percentile(arr, [0.5, 99.5])
            if hi <= lo:
                lo, hi = float(arr.min()), float(arr.max())
            if hi <= lo:  # constant metric (e.g. no clipping anywhere): small range just above the value
                hi = lo + max(abs(lo) * 0.1, 1e-3)
            counts, edges = np.histogram(np.clip(arr, lo, hi), bins=40, range=(lo, hi))
            out["metrics"][f] = {
                "n": int(len(arr)), "mean": float(arr.mean()),
                "p": {str(q): float(np.percentile(arr, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
                "min": float(arr.min()), "max": float(arr.max()),
                "hist": {"counts": counts.tolist(), "lo": float(lo), "hi": float(hi)}}
        out["flags"] = dict(Counter(fl for v in vals for fl in v["flags"]).most_common())
        out["any_flag"] = sum(1 for v in vals if v["flags"])
        out["sample_rates"] = dict(Counter(str(int(v["sample_rate"])) for v in vals if v.get("sample_rate")))
        mean_dur = out["metrics"].get("duration_s", {}).get("mean", 0)
        out["est_total_hours"] = mean_dur * d.n_rows / 3600
        return out

    # ---- API: export ---------------------------------------------------------------------------------------
    @app.get("/api/export/manifest")
    def export_manifest():
        """Everything training code needs: per-dataset include/exclude + per-sample rejected/flagged keys."""
        dec = store.decisions()
        out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "key_format": "parquet: '<file relative to corpus dir>#<row in file>'; LibriSpeech: utterance id",
               "datasets": {}}
        for d in datasets.values():
            revs = store.reviews(d.id)
            by = {s: sorted(r["key"] for r in revs.values() if r["status"] == s) for s in SAMPLE_STATUSES}
            out["datasets"][d.id] = {
                "decision": dec.get(d.id, {}).get("decision", "undecided"),
                "note": dec.get(d.id, {}).get("note", ""),
                "n_rows": d.n_rows, "corpus_dir": str(getattr(d, "corpus_dir", "")),
                "rejected_keys": by["reject"], "flagged_keys": by["flag"], "n_kept": len(by["keep"])}
        return JSONResponse(out, headers={"Content-Disposition": 'attachment; filename="curation_manifest.json"'})

    @app.get("/api/export/csv")
    def export_csv(ds: str):
        d = get_ds(ds)
        revs, mets = store.reviews(ds), store.metrics(ds)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["row_idx", "key", "id", "status", "tags", "note", *METRIC_FIELDS, "flags", "label"])
        for i in sorted(set(revs) | set(mets)):
            r, m = revs.get(i, {}), mets.get(i, {})
            w.writerow([i, d.key(i), d.display_id(i), r.get("status", "unreviewed"), ";".join(r.get("tags", [])),
                        r.get("note", ""), *[m.get(f) for f in METRIC_FIELDS], ";".join(m.get("flags", [])),
                        label_text(d, i)])
        name = ds.replace("/", "__") + ".csv"
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    # ---- static UI -----------------------------------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app
