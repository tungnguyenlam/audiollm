"""SQLite store for curation state: per-sample reviews, cached quality metrics, per-dataset decisions.

Everything is mirrored in memory (dicts keyed by row index) so list filtering/sorting never hits the disk.
"""
import json
import sqlite3
import threading
from datetime import datetime, timezone

SAMPLE_STATUSES = ("keep", "reject", "flag")
DATASET_DECISIONS = ("include", "exclude", "undecided")
METRIC_FIELDS = ("duration_s", "sample_rate", "channels", "peak_dbfs", "rms_dbfs", "clip_ratio", "silence_ratio",
                 "snr_db", "bandwidth_hz", "dc_offset", "lead_silence_s", "trail_silence_s", "chars_per_sec")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.RLock()
        cols = ", ".join(f"{f} REAL" for f in METRIC_FIELDS)
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS reviews (
                dataset_id TEXT NOT NULL, sample_key TEXT NOT NULL, row_idx INTEGER NOT NULL,
                status TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', note TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL, PRIMARY KEY (dataset_id, sample_key));
            CREATE TABLE IF NOT EXISTS metrics (
                dataset_id TEXT NOT NULL, sample_key TEXT NOT NULL, row_idx INTEGER NOT NULL,
                {cols}, flags TEXT NOT NULL DEFAULT '[]', PRIMARY KEY (dataset_id, sample_key));
            CREATE TABLE IF NOT EXISTS decisions (
                dataset_id TEXT PRIMARY KEY, decision TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL);
        """)
        self._reviews, self._metrics = {}, {}  # dataset_id -> {row_idx: dict}

    # ---- reviews -------------------------------------------------------------------------------------------
    def reviews(self, dataset_id):
        with self.lock:
            if dataset_id not in self._reviews:
                cur = self.conn.execute(
                    "SELECT row_idx, sample_key, status, tags, note, updated_at FROM reviews WHERE dataset_id=?",
                    (dataset_id,))
                self._reviews[dataset_id] = {
                    r[0]: {"key": r[1], "status": r[2], "tags": json.loads(r[3]), "note": r[4], "updated_at": r[5]}
                    for r in cur}
            return self._reviews[dataset_id]

    def set_reviews(self, dataset_id, items, status, tags=None, note=None):
        """items: [(row_idx, sample_key)]. status None/'unreviewed' clears. tags/note None = keep existing."""
        with self.lock:
            revs = self.reviews(dataset_id)
            now = _now()
            self.conn.execute("BEGIN")
            try:
                for idx, key in items:
                    if status in (None, "unreviewed"):
                        self.conn.execute("DELETE FROM reviews WHERE dataset_id=? AND sample_key=?", (dataset_id, key))
                        revs.pop(idx, None)
                        continue
                    old = revs.get(idx, {})
                    t = old.get("tags", []) if tags is None else tags
                    n = old.get("note", "") if note is None else note
                    self.conn.execute(
                        "INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?)",
                        (dataset_id, key, idx, status, json.dumps(t), n, now))
                    revs[idx] = {"key": key, "status": status, "tags": t, "note": n, "updated_at": now}
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return [revs.get(idx) for idx, _ in items]

    # ---- metrics -------------------------------------------------------------------------------------------
    def metrics(self, dataset_id):
        with self.lock:
            if dataset_id not in self._metrics:
                cur = self.conn.execute(
                    f"SELECT row_idx, {', '.join(METRIC_FIELDS)}, flags FROM metrics WHERE dataset_id=?",
                    (dataset_id,))
                d = {}
                for r in cur:
                    m = dict(zip(METRIC_FIELDS, r[1:-1]))
                    m["flags"] = json.loads(r[-1])
                    d[r[0]] = m
                self._metrics[dataset_id] = d
            return self._metrics[dataset_id]

    def put_metrics(self, dataset_id, rows):
        """rows: [(row_idx, sample_key, metrics_dict)]"""
        with self.lock:
            cache = self.metrics(dataset_id)
            ph = ",".join("?" * (len(METRIC_FIELDS) + 4))
            self.conn.execute("BEGIN")
            self.conn.executemany(
                f"INSERT OR REPLACE INTO metrics VALUES ({ph})",
                [(dataset_id, key, idx, *[m.get(f) for f in METRIC_FIELDS], json.dumps(m.get("flags", [])))
                 for idx, key, m in rows])
            self.conn.execute("COMMIT")
            for idx, _, m in rows:
                cache[idx] = {**{f: m.get(f) for f in METRIC_FIELDS}, "flags": m.get("flags", [])}

    def clear_metrics(self, dataset_id):
        with self.lock:
            self.conn.execute("DELETE FROM metrics WHERE dataset_id=?", (dataset_id,))
            self._metrics[dataset_id] = {}

    # ---- dataset decisions ---------------------------------------------------------------------------------
    def decisions(self):
        with self.lock:
            return {r[0]: {"decision": r[1], "note": r[2], "updated_at": r[3]}
                    for r in self.conn.execute("SELECT dataset_id, decision, note, updated_at FROM decisions")}

    def set_decision(self, dataset_id, decision, note):
        with self.lock:
            now = _now()
            self.conn.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?)", (dataset_id, decision, note, now))
            return {"decision": decision, "note": note, "updated_at": now}
