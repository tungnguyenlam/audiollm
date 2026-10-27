"""Discover the datasets under data/ and give uniform, random access to their rows.

One *dataset* = one corpus split, e.g. ``corpora/ami_ihm/train`` or ``LibriSpeech/dev-clean``, because splits of
the same corpus can have different quality and you may want to keep one and drop another.

Every row gets a stable ``key``: ``<file relative to corpus dir>#<row in file>`` for parquet,
the utterance id for LibriSpeech. Curation decisions are stored against that key.
"""
import bisect
import json
import re
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dataset_viewer.audio import decode

AUDIO_COLUMNS = ("audio", "input_values")
# Columns shown as the sample's label(s), in this order. Whatever exists in a dataset is shown.
LABEL_COLUMNS = ("text", "transcription", "raw_transcription", "question", "choices", "answer", "answer_gt",
                 "category", "emotion", "label", "labels", "task_name", "accent")
TRANSCRIPT_COLUMNS = ("text", "transcription")
ID_COLUMNS = ("id", "audio_id", "uniq_id", "filename", "file", "path")
SHARD_RE = re.compile(r"^(?P<split>.+?)-\d{5}-of-\d{5}")
# Raw audio row groups kept in memory, shared by all datasets. Row groups range from ~10 MB (ami) to ~400 MB (esc50,
# fleurs store a whole file in one row group), so the cache is bounded by bytes; the newest entry is always kept.
ROW_GROUP_CACHE_BYTES = 768 * 2**20
_rg_cache = OrderedDict()   # (file path, row group) -> pyarrow Array of the audio column
_rg_lock = threading.Lock()


def _cached_row_group(path, rg, column):
    k = (str(path), rg)
    with _rg_lock:
        if k in _rg_cache:
            _rg_cache.move_to_end(k)
            return _rg_cache[k]
    # Read outside the lock so scan workers decode different row groups in parallel. A fresh ParquetFile per read
    # keeps concurrent reads independent (footer parsing is ~1 ms).
    arr = pq.ParquetFile(path).read_row_group(rg, columns=[column]).column(0).combine_chunks()
    with _rg_lock:
        _rg_cache[k] = arr
        total = sum(a.nbytes for a in _rg_cache.values())
        while len(_rg_cache) > 1 and total > ROW_GROUP_CACHE_BYTES:
            _, old = _rg_cache.popitem(last=False)
            total -= old.nbytes
    return arr


class Dataset:
    id: str
    corpus: str
    split: str
    n_rows: int
    label_columns: list
    transcript_column: str | None

    @property
    def kind(self):
        if self.transcript_column:
            return "transcribed speech"
        if "question" in self.label_columns:
            return "audio QA"
        return "classification / tagged"

    def info(self):
        return {"id": self.id, "corpus": self.corpus, "split": self.split, "n_rows": self.n_rows,
                "kind": self.kind, "label_columns": self.label_columns, "transcript_column": self.transcript_column}

    def transcript(self, idx):
        if not self.transcript_column:
            return None
        v = self.meta_column(self.transcript_column)[idx]
        return "" if v is None else str(v)


class ParquetDataset(Dataset):
    def __init__(self, corpus, split, corpus_dir: Path, files):
        self.corpus, self.split, self.id = corpus, split, f"{corpus}/{split}"
        self.corpus_dir = corpus_dir
        self.files = files
        self._pfs = [pq.ParquetFile(f) for f in files]
        schema = self._pfs[0].schema_arrow
        self.columns = schema.names
        self.audio_column = next((c for c in AUDIO_COLUMNS if c in self.columns), None)
        self.meta_columns = [c for c in self.columns if c not in AUDIO_COLUMNS]
        self.label_columns = [c for c in LABEL_COLUMNS if c in self.columns]
        self.transcript_column = next((c for c in TRANSCRIPT_COLUMNS if c in self.columns), None)
        self.id_column = next((c for c in ID_COLUMNS if c in self.columns), None)
        self.class_names = _hf_class_labels(schema)

        # Global row -> (file, row group, offset) lookup from parquet footers only (no data read).
        self._file_starts, self._rg_starts = [], []
        total = 0
        for pf in self._pfs:
            self._file_starts.append(total)
            starts, s = [], 0
            for i in range(pf.metadata.num_row_groups):
                starts.append(s)
                s += pf.metadata.row_group(i).num_rows
            self._rg_starts.append(starts)
            total += pf.metadata.num_rows
        self.n_rows = total
        self._meta = None
        self._meta_lock = threading.Lock()

    def locate(self, idx):
        fi = bisect.bisect_right(self._file_starts, idx) - 1
        local = idx - self._file_starts[fi]
        rg = bisect.bisect_right(self._rg_starts[fi], local) - 1
        return fi, rg, local - self._rg_starts[fi][rg], local

    def key(self, idx):
        fi, _, _, local = self.locate(idx)
        return f"{self.files[fi].relative_to(self.corpus_dir)}#{local}"

    def _load_meta(self):
        """All non-audio columns of the whole split, read once (columnar, cheap)."""
        with self._meta_lock:
            if self._meta is None:
                cols = defaultdict(list)
                for pf in self._pfs:
                    if not self.meta_columns:
                        break
                    t = pf.read(columns=self.meta_columns)
                    for c in self.meta_columns:
                        cols[c].extend(t.column(c).to_pylist())
                for c, names in self.class_names.items():
                    if c in cols:
                        cols[c] = [_class_name(v, names) for v in cols[c]]
                self._meta = dict(cols)
        return self._meta

    def meta_column(self, name):
        return self._load_meta().get(name, [None] * self.n_rows)

    def row_meta(self, idx):
        meta = self._load_meta()
        fi, rg, _, local = self.locate(idx)
        row = {c: meta[c][idx] for c in self.meta_columns}
        row["_file"] = str(self.files[fi].relative_to(self.corpus_dir))
        row["_row_in_file"] = local
        row["_row_group"] = rg
        return row

    def display_id(self, idx):
        if self.id_column:
            v = self.meta_column(self.id_column)[idx]
            if v is not None:
                v = str(v)
                return v.rsplit("/", 1)[-1] if "/" in v else v  # absolute HF cache paths -> file name
        return self.key(idx)

    def audio(self, idx):
        """-> (mono float32 waveform, sample rate, channels)."""
        fi, rg, off, _ = self.locate(idx)
        arr = _cached_row_group(self.files[fi], rg, self.audio_column)
        if pa.types.is_struct(arr.type):
            cell = arr.field("bytes")[off].as_py() or arr.field("path")[off].as_py()
            return decode(cell)
        # list<float> of raw samples (speech_commands); 16 kHz per its dataset card
        return decode(arr[off].values.to_numpy(zero_copy_only=False).astype(np.float32))


class LibriSpeechDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.corpus, self.split = "LibriSpeech", split_dir.name
        self.id = f"LibriSpeech/{split_dir.name}"
        self.label_columns = ["text"]
        self.transcript_column = "text"
        rows = []
        for trans in sorted(split_dir.rglob("*.trans.txt")):
            for line in trans.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    uid, _, text = line.strip().partition(" ")
                    rows.append((uid, text, trans.parent / f"{uid}.flac"))
        self._rows = rows
        self.n_rows = len(rows)
        self._meta = {"id": [r[0] for r in rows], "text": [r[1] for r in rows]}

    def key(self, idx):
        return self._rows[idx][0]

    display_id = key

    def meta_column(self, name):
        return self._meta.get(name, [None] * self.n_rows)

    def row_meta(self, idx):
        uid, text, path = self._rows[idx]
        spk, chap, _ = uid.split("-")
        return {"id": uid, "text": text, "speaker_id": spk, "chapter_id": chap, "_file": str(path)}

    def audio(self, idx):
        return decode(str(self._rows[idx][2]))


def _hf_class_labels(schema):
    """ClassLabel names from the HuggingFace metadata stored in the parquet footer."""
    try:
        feats = json.loads(schema.metadata[b"huggingface"])["info"]["features"]
    except Exception:
        return {}
    out = {}
    for name, f in feats.items():
        if isinstance(f, dict) and f.get("_type") == "ClassLabel" and f.get("names"):
            out[name] = f["names"]
    return out


def _class_name(v, names):
    if isinstance(v, int) and 0 <= v < len(names):
        return f"{names[v]} ({v})"
    return v


def discover(data_root="data"):
    """-> OrderedDict[id, Dataset] for LibriSpeech splits and every parquet corpus under corpora/ and esb/."""
    root = Path(data_root)
    found = OrderedDict()

    libri = root / "LibriSpeech"
    if libri.is_dir():
        for d in sorted(p for p in libri.iterdir() if p.is_dir()):
            ds = LibriSpeechDataset(d)
            if ds.n_rows:
                found[ds.id] = ds

    for group in ("corpora", "esb"):
        gdir = root / group
        if not gdir.is_dir():
            continue
        for cdir in sorted(p for p in gdir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            files = sorted(f for f in cdir.rglob("*.parquet") if ".cache" not in f.parts)
            if not files:
                continue
            subdirs = {f.parent for f in files}
            splits = defaultdict(list)
            for f in files:
                m = SHARD_RE.match(f.stem)
                split = m.group("split") if m else f.stem
                if len(subdirs) > 1:
                    split = f"{f.parent.relative_to(cdir)}/{split}"
                splits[split].append(f)
            for split, fs in sorted(splits.items()):
                try:
                    ds = ParquetDataset(f"{group}/{cdir.name}", split, cdir, fs)
                except Exception as e:  # unreadable / partial download: skip but say so
                    print(f"[dataset_viewer] skipping {cdir}/{split}: {e}")
                    continue
                if ds.n_rows and ds.audio_column:
                    found[ds.id] = ds
    return found
