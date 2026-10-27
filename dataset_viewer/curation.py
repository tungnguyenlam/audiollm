"""Use curation results in training code without importing the web app.

    from dataset_viewer.curation import load_curation
    cur = load_curation("curation_manifest.json")   # exported from the viewer (Export -> manifest)
    cur.included_datasets()                          # ['LibriSpeech/train-clean-100', ...]
    cur.keep("corpora/ami_ihm/train", "ihm/train-00000-of-00042.parquet#17")   # False if that row was rejected

The manifest can also be built straight from the SQLite store: load_curation(db="data/curation.sqlite3").
"""
import json
import sqlite3
from dataclasses import dataclass, field


@dataclass
class Curation:
    decisions: dict = field(default_factory=dict)   # dataset_id -> 'include' | 'exclude' | 'undecided'
    rejected: dict = field(default_factory=dict)    # dataset_id -> set of sample keys
    flagged: dict = field(default_factory=dict)     # dataset_id -> set of sample keys

    def included_datasets(self):
        return sorted(k for k, v in self.decisions.items() if v == "include")

    def keep(self, dataset_id, key, drop_flagged=False):
        if self.decisions.get(dataset_id) == "exclude":
            return False
        if key in self.rejected.get(dataset_id, ()):
            return False
        return not (drop_flagged and key in self.flagged.get(dataset_id, ()))


def load_curation(manifest=None, db=None):
    c = Curation()
    if manifest:
        with open(manifest) as f:
            data = json.load(f)["datasets"]
        for ds, v in data.items():
            c.decisions[ds] = v["decision"]
            c.rejected[ds] = set(v["rejected_keys"])
            c.flagged[ds] = set(v["flagged_keys"])
    elif db:
        conn = sqlite3.connect(db)
        c.decisions = dict(conn.execute("SELECT dataset_id, decision FROM decisions"))
        for ds, key, status in conn.execute("SELECT dataset_id, sample_key, status FROM reviews"):
            if status == "reject":
                c.rejected.setdefault(ds, set()).add(key)
            elif status == "flag":
                c.flagged.setdefault(ds, set()).add(key)
        conn.close()
    else:
        raise ValueError("pass manifest= or db=")
    return c
