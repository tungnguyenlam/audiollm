"""
Fetch LibriSpeech from the HF parquet mirror (openslr/librispeech_asr) and unpack it into the standard
OpenSLR layout: data/LibriSpeech/<split>/<spk>/<chapter>/<id>.flac + <spk>-<chapter>.trans.txt
(OpenSLR itself is ~10 KB/s from here.) The parquet `audio.bytes` are the original FLAC files, written as-is.
"""
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

SPLITS = {  # HF name -> OpenSLR name
    "validation.clean": "dev-clean",
    "test.clean": "test-clean",
    "test.other": "test-other",
    "train.clean.100": "train-clean-100",
}
OUT = Path("data/LibriSpeech")

for hf_split, split in SPLITS.items():
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
    print(split, n, "utterances")
