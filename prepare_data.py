"""
Fetch LibriSpeech from the HF parquet mirror (openslr/librispeech_asr) and unpack it into the standard
OpenSLR layout: data/LibriSpeech/<split>/<spk>/<chapter>/<id>.flac + <spk>-<chapter>.trans.txt
(OpenSLR itself is ~10 KB/s from here.) The parquet `audio.bytes` are the original FLAC files, written as-is.
"""
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

SPLITS = {  # HF name -> OpenSLR name
    "validation.clean": "dev-clean",
    "test.clean": "test-clean",
    "test.other": "test-other",
    "train.clean.100": "train-clean-100",
    "validation.other": "dev-other",
    "train.clean.360": "train-clean-360",
    "train.other.500": "train-other-500",
}
OUT = Path("data/LibriSpeech")

OPTIONAL = {"train.other.500"}  # ~31 GB, only fetched when named on the command line
# usage: prepare_data.py [split ...]  (HF or OpenSLR names; no args = every split except OPTIONAL)
splits = ({k: v for k, v in SPLITS.items() if k in sys.argv[1:] or v in sys.argv[1:]} if len(sys.argv) > 1
          else {k: v for k, v in SPLITS.items() if k not in OPTIONAL})

for hf_split, split in splits.items():
    if (OUT / split / ".done").exists():
        print("skip", split)
        continue
    local = snapshot_download("openslr/librispeech_asr", repo_type="dataset", allow_patterns=[f"all/{hf_split}/*"],
                              local_dir="data/hf_librispeech")
    trans = defaultdict(list)
    n = 0
    for f in sorted(Path(local, "all", hf_split).glob("*.parquet")):
        t = pq.read_table(f, columns=["id", "speaker_id", "chapter_id", "text", "audio"]).to_pylist()
        for r in t:
            d = OUT / split / str(r["speaker_id"]) / str(r["chapter_id"])
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{r['id']}.flac").write_bytes(r["audio"]["bytes"])
            trans[d].append((r["id"], r["text"]))
            n += 1
    for d, rows in trans.items():
        spk, chap = d.parent.name, d.name
        (d / f"{spk}-{chap}.trans.txt").write_text("".join(f"{i} {t}\n" for i, t in sorted(rows)))
    (OUT / split / ".done").touch()
    shutil.rmtree(Path(local, "all", hf_split), ignore_errors=True)
    print(split, n, "utterances")
