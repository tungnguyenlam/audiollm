"""FastAPI backend for the dataset viewer. Dataset ids contain '/', so they always travel in the `ds` query parameter.

Every endpoint is a plain function of (dataset, row, params); the only server-side state is the curation store, the
running quality scans and a few small LRU caches (decoded audio, front-end lab results)."""
import csv
import hashlib
import io
import json
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
REVIEW_TAGS = ("wrong_transcript", "partial_transcript", "wrong_label", "noisy", "clipped_distorted",
               "music_background", "overlapping_speech", "mostly_silence", "wrong_language", "codec_artifacts",
               "truncated", "other")
SORTS = ("index", "random", "status", "n_flags") + METRIC_FIELDS
STATUSES = SAMPLE_STATUSES + ("unreviewed",)
CMAPS = ("magma", "viridis", "inferno", "turbo", "gray_r")


class LRU(OrderedDict):
    """Thread-safe bounded dict; get() refreshes recency."""

    def __init__(self, size):
        super().__init__()
        self.size, self.lock = size, threading.Lock()

    def get(self, k, default=None):
        with self.lock:
            if k not in self:
                return default
            self.move_to_end(k)
            return self[k]

    def put(self, k, v):
        with self.lock:
            self[k] = v
            self.move_to_end(k)
            while len(self) > self.size:
                self.popitem(last=False)
        return v


class ScanJob:
    """Background quality scan: decode rows, compute metrics, write them to the store in chunks."""

    def __init__(self, d, idxs, store, workers):
        self.d, self.idxs, self.store, self.workers = d, idxs, store, workers
        self.done = self.errors = 0
        self.cancelled = False
        self.started, self.finished = time.time(), None
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _work(self, chunk):
        rows = []
        for idx in chunk:
            if self.cancelled:
                break
            try:
                x, sr, ch = self.d.audio(idx)
                m = compute_metrics(x, sr, ch, self.d.transcript(idx), bool(self.d.transcript_column))
            except Exception:
                m = {"flags": ["decode_error"]}
                with self.lock:
                    self.errors += 1
            rows.append((idx, self.d.key(idx), m))
            with self.lock:
                self.done += 1
        self.store.put_metrics(self.d.id, rows)

    def _run(self):
        # contiguous chunks: each worker streams through its own parquet row groups
        size = max(1, min(64, len(self.idxs) // (self.workers * 4) or 1))
        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(self._work, [self.idxs[i:i + size] for i in range(0, len(self.idxs), size)]))
        self.finished = time.time()

    @property
    def running(self):
        return self.finished is None

    def state(self):
        el = (self.finished or time.time()) - self.started
        return {"total": len(self.idxs), "done": self.done, "errors": self.errors, "running": self.running,
                "cancelled": self.cancelled, "elapsed_s": round(el, 1),
                "rate_per_s": round(self.done / el, 1) if el > 0 else 0}


def create_app(data_root="data", db_path="data/curation.sqlite3", scan_workers=4):
    datasets = discover(data_root)
    store = Store(db_path)
    scans = {}                      # dataset id -> ScanJob
    audio_cache = LRU(8)            # (ds, idx) -> (x, sr, channels)
    lab_cache = LRU(24)             # params hash -> (summary, {name: bytes})
    lab_slots = threading.Semaphore(2)
    app = FastAPI(title="Dataset viewer")

    # ------------------------------------------------------------------------------------------------ helpers
    def get_ds(ds):
        d = datasets.get(ds)
        if d is None:
            raise HTTPException(404, f"Unknown dataset {ds!r}")
        return d

    def get_row(ds, idx):
        d = get_ds(ds)
        if not 0 <= idx < d.n_rows:
            raise HTTPException(404, f"Row {idx} is outside 0..{d.n_rows - 1}")
        return d

    def load_audio(d, idx):
        hit = audio_cache.get((d.id, idx))
        if hit is not None:
            return hit
        try:
            return audio_cache.put((d.id, idx), d.audio(idx))
        except Exception as e:
            raise HTTPException(422, f"Could not decode row {idx}: {e}")

    def label_text(d, idx):
        vals = [(c, d.meta_column(c)[idx]) for c in d.label_columns]
        vals = [(c, v) for c, v in vals if v is not None and v != ""]
        return " | ".join(str(v) if len(d.label_columns) == 1 else f"{c}: {v}" for c, v in vals)

    def filtered_rows(d, q="", status="all", flag="all", sort="index", desc=False, seed=0):
        """Row indices that match the filters, in display order. Sorting by a metric keeps only scanned rows."""
        revs, mets = store.reviews(d.id), store.metrics(d.id)
        idxs = range(d.n_rows)
        if status != "all":
            idxs = [i for i in idxs if revs.get(i, {}).get("status", "unreviewed") == status]
        flag_test = {
            "all": None,
            "scanned": lambda i: i in mets,
            "unscanned": lambda i: i not in mets,
            "any": lambda i: bool(mets.get(i, {}).get("flags")),
            "none": lambda i: i in mets and not mets[i]["flags"],
        }.get(flag, lambda i: flag in mets.get(i, {}).get("flags", ()))
        if flag_test:
            idxs = [i for i in idxs if flag_test(i)]
        if q:
            ql = q.lower()
            cols = [d.meta_column(c) for c in d.label_columns]
            idxs = [i for i in idxs if ql in d.display_id(i).lower() or ql in d.key(i).lower()
                    or any(ql in str(col[i]).lower() for col in cols)]
        idxs = list(idxs)
        if sort == "random":
            random.Random(seed).shuffle(idxs)
        elif sort == "status":
            order = {"reject": 0, "flag": 1, "keep": 2}
            idxs.sort(key=lambda i: order.get(revs.get(i, {}).get("status"), 3), reverse=desc)
        elif sort == "n_flags":
            idxs = sorted((i for i in idxs if i in mets), key=lambda i: len(mets[i]["flags"]), reverse=desc)
        elif sort in METRIC_FIELDS:
            idxs = sorted((i for i in idxs if mets.get(i, {}).get(sort) is not None),
                          key=lambda i: mets[i][sort], reverse=desc)
        elif desc:
            idxs.reverse()
        return idxs

    def summary(d, decisions=None):
        decisions = store.decisions() if decisions is None else decisions
        revs, mets = store.reviews(d.id), store.metrics(d.id)
        c = Counter(r["status"] for r in revs.values())
        job = scans.get(d.id)
        return d.info() | {
            "decision": decisions.get(d.id, {"decision": "undecided", "note": "", "updated_at": None}),
            "reviews": {s: c.get(s, 0) for s in SAMPLE_STATUSES},
            "scanned": len(mets), "auto_flagged": sum(1 for m in mets.values() if m["flags"]),
            "scan": job.state() if job else None,
        }

    # ------------------------------------------------------------------------------------------------ catalogue
    @app.get("/api/bootstrap")
    def bootstrap():
        """Everything the page needs before the first dataset loads."""
        dec = store.decisions()
        return {
            "thresholds": {k: {"metric": m, "op": op, "value": v, "desc": desc}
                           for k, (m, op, v, desc) in THRESHOLDS.items()},
            "tags": REVIEW_TAGS, "statuses": SAMPLE_STATUSES, "decisions": DATASET_DECISIONS,
            "metrics": METRIC_FIELDS, "sorts": SORTS, "cmaps": CMAPS,
            "lab": {"presets": frontend_lab.PRESETS, "feature_precisions": frontend_lab.FEATURE_PRECISIONS,
                    "max_sec": frontend_lab.MAX_SEC},
            "datasets": [summary(d, dec) for d in datasets.values()],
        }

    @app.get("/api/datasets")
    def list_datasets():
        dec = store.decisions()
        return [summary(d, dec) for d in datasets.values()]

    @app.get("/api/dataset")
    def dataset(ds: str):
        return summary(get_ds(ds))

    @app.put("/api/dataset/decision")
    def set_decision(ds: str, body: dict = Body(...)):
        get_ds(ds)
        decision = body.get("decision", "undecided")
        if decision not in DATASET_DECISIONS:
            raise HTTPException(400, f"decision must be one of {DATASET_DECISIONS}")
        store.set_decision(ds, decision, str(body.get("note", "")))
        return summary(get_ds(ds))

    # ------------------------------------------------------------------------------------------------ samples
    @app.get("/api/samples")
    def samples(ds: str, offset: int = Query(0, ge=0), limit: int = Query(100, ge=0, le=500), q: str = "",
                status: str = "all", flag: str = "all", sort: str = "index", desc: bool = False, seed: int = 0):
        d = get_ds(ds)
        if sort not in SORTS:
            raise HTTPException(400, f"sort must be one of {SORTS}")
        idxs = filtered_rows(d, q, status, flag, sort, desc, seed)
        revs, mets = store.reviews(d.id), store.metrics(d.id)
        rows = []
        for i in idxs[offset:offset + limit]:
            m, r = mets.get(i), revs.get(i, {})
            rows.append({"idx": i, "id": d.display_id(i), "label": label_text(d, i)[:240],
                         "status": r.get("status", "unreviewed"),
                         "duration_s": m and m["duration_s"], "flags": m["flags"] if m else None})
        return {"total": len(idxs), "offset": offset, "rows": rows}

    @app.get("/api/sample")
    def sample(ds: str, idx: int):
        d = get_row(ds, idx)
        try:
            x, sr, ch = load_audio(d, idx)
            m = compute_metrics(x, sr, ch, d.transcript(idx), bool(d.transcript_column))
            store.put_metrics(d.id, [(idx, d.key(idx), m)])
        except HTTPException as e:
            m = {f: None for f in METRIC_FIELDS} | {"flags": ["decode_error"], "error": e.detail}
        meta = d.row_meta(idx)
        return {"idx": idx, "key": d.key(idx), "id": d.display_id(idx), "n_rows": d.n_rows,
                "labels": {c: meta.get(c) for c in d.label_columns}, "meta": meta, "metrics": m,
                "review": store.reviews(d.id).get(idx, {"status": "unreviewed", "tags": [], "note": ""})}

    @app.get("/api/audio")
    def audio(ds: str, idx: int):
        x, sr, _ = load_audio(get_row(ds, idx), idx)
        buf = io.BytesIO()
        sf.write(buf, x, sr, format="WAV", subtype="FLOAT")  # exact samples: what you hear is what the model gets
        return Response(buf.getvalue(), media_type="audio/wav", headers={"Cache-Control": "max-age=3600"})

    @app.get("/api/spectrogram")
    def spectrogram(ds: str, idx: int, kind: str = Query("mel", pattern="^(mel|linear)$"), cmap: str = "magma"):
        if cmap not in CMAPS:
            raise HTTPException(400, f"cmap must be one of {CMAPS}")
        x, sr, _ = load_audio(get_row(ds, idx), idx)
        if kind == "mel":
            png = render_png(log_mel(x, sr), cmap)
        else:
            png = render_png(linear_spec_db(x, sr), cmap, dynamic_range=90)
        return Response(png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})

    @app.put("/api/review")
    def review(ds: str, idx: int, body: dict = Body(...)):
        d = get_row(ds, idx)
        status = body.get("status")
        if status not in STATUSES:
            raise HTTPException(400, f"status must be one of {STATUSES}")
        r = store.set_reviews(d.id, [(idx, d.key(idx))], status, body.get("tags"), body.get("note"))[0]
        return r or {"status": "unreviewed", "tags": [], "note": ""}

    @app.post("/api/bulk_review")
    def bulk_review(ds: str, body: dict = Body(...)):
        """Apply one status (and optionally add one tag) to every row that matches a list filter."""
        d = get_ds(ds)
        f, status, tag = body.get("filter", {}), body.get("status"), body.get("tag")
        if status not in STATUSES:
            raise HTTPException(400, f"status must be one of {STATUSES}")
        if f.get("flag", "all") in ("all", "unscanned", "scanned") and not f.get("q") \
                and f.get("status", "all") == "all":
            raise HTTPException(400, "Narrow the filter first (a flag, a search or a status); "
                                     "bulk review never applies to a whole dataset")
        idxs = filtered_rows(d, f.get("q", ""), f.get("status", "all"), f.get("flag", "all"))
        if tag:  # add the tag per row and keep each row's other tags and note
            revs = store.reviews(d.id)
            for i in idxs:
                store.set_reviews(d.id, [(i, d.key(i))], status,
                                  sorted(set(revs.get(i, {}).get("tags", [])) | {tag}), None)
        else:
            store.set_reviews(d.id, [(i, d.key(i)) for i in idxs], status)
        return {"updated": len(idxs)}

    # ------------------------------------------------------------------------------------------------ front-end lab
    @app.get("/api/lab")
    def lab(ds: str, idx: int, sr: int = 16000, bits: int = 32, quant: str = "pcm", n_fft: int = 400,
            hop: int = 160, n_mels: int = 80, stack: int = 16, feat: str = "fp32", start: float = 0.0,
            dur: float = 8.0, hidden: int = 1024):
        """Run one clip through a front-end config. Results are cached by their parameters, so going back to a
        setting you already tried is instant and its files never vanish while the page shows them."""
        d = get_row(ds, idx)
        p = frontend_lab.validate(dict(sr=sr, bits=bits, quant=quant, n_fft=n_fft, hop=hop, n_mels=n_mels,
                                       stack=stack, feat=feat, start=start, dur=dur))
        key = hashlib.sha1(json.dumps([ds, idx, p, hidden], sort_keys=True).encode()).hexdigest()[:20]
        hit = lab_cache.get(key)
        if hit is None:
            x, x_sr, _ = load_audio(d, idx)
            with lab_slots:
                try:
                    hit = lab_cache.put(key, frontend_lab.analyse(x, x_sr, p, hidden_size=hidden))
                except ValueError as e:
                    raise HTTPException(400, str(e))
        return hit[0] | {"key": key, "files": sorted(hit[1])}

    @app.get("/api/lab/file")
    def lab_file(key: str, name: str):
        hit = lab_cache.get(key)
        if hit is None or name not in hit[1]:
            raise HTTPException(404, "This lab result is no longer cached; run it again")
        kind = {"img": "image/png", "wav": "audio/wav"}.get(name.split("_")[0], "application/octet-stream")
        return Response(hit[1][name], media_type=kind, headers={"Cache-Control": "max-age=86400, immutable"})

    # ------------------------------------------------------------------------------------------------ scans + report
    @app.post("/api/scan")
    def start_scan(ds: str, n: int = Query(1000, ge=1), seed: int = 0):
        """Compute metrics for n random rows that have not been scanned yet (n >= n_rows scans everything)."""
        d = get_ds(ds)
        if ds in scans and scans[ds].running:
            raise HTTPException(409, "A scan of this dataset is already running")
        done = store.metrics(ds)
        todo = [i for i in range(d.n_rows) if i not in done]
        if n < len(todo):
            todo = sorted(random.Random(seed).sample(todo, n))
        scans[ds] = ScanJob(d, todo, store, scan_workers)
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
        if ds in scans and scans[ds].running:
            raise HTTPException(409, "Cancel the running scan first")
        store.clear_metrics(ds)
        return {"ok": True}

    @app.get("/api/report")
    def report(ds: str):
        d = get_ds(ds)
        mets, revs = store.metrics(ds), store.reviews(ds)
        n = len(mets)
        out = {"n_rows": d.n_rows, "scanned": n, "metrics": {}, "flags": {}, "sample_rates": {}, "any_flag": 0,
               "est_total_hours": None,
               "reviews": dict(Counter(r["status"] for r in revs.values())),
               "tags": dict(Counter(t for r in revs.values() for t in r["tags"]).most_common())}
        if not n:
            return out
        vals = list(mets.values())
        for f in METRIC_FIELDS:
            if f in ("sample_rate", "channels"):
                continue
            arr = np.array([v[f] for v in vals if v.get(f) is not None], dtype=float)
            if not len(arr):
                continue
            lo, hi = np.percentile(arr, [0.5, 99.5])
            if hi <= lo:
                lo, hi = float(arr.min()), float(arr.max())
            if hi <= lo:  # constant metric: a small range just above the value
                hi = lo + max(abs(lo) * 0.1, 1e-3)
            counts, _ = np.histogram(np.clip(arr, lo, hi), bins=40, range=(lo, hi))
            out["metrics"][f] = {"n": int(len(arr)), "mean": float(arr.mean()),
                                 "p": {str(q): float(np.percentile(arr, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
                                 "hist": {"counts": counts.tolist(), "lo": float(lo), "hi": float(hi)}}
        out["flags"] = dict(Counter(fl for v in vals for fl in v["flags"]).most_common())
        out["any_flag"] = sum(1 for v in vals if v["flags"])
        out["sample_rates"] = dict(Counter(str(int(v["sample_rate"])) for v in vals if v.get("sample_rate")))
        if "duration_s" in out["metrics"]:
            out["est_total_hours"] = out["metrics"]["duration_s"]["mean"] * d.n_rows / 3600
        return out

    # ------------------------------------------------------------------------------------------------ export
    @app.get("/api/export/manifest")
    def export_manifest():
        """What training code needs: per-dataset include/exclude plus the rejected and flagged sample keys."""
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

    # ------------------------------------------------------------------------------------------------ page
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app
