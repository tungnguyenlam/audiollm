"""
WER per Vietnamese accent region (North / Central / South) from hypothesis files.

  ZAC 2026 (runs/<id>/hyp_*.jsonl or val_hyps/<zac split>/*.jsonl): region from the request's city
      (Hà Nội -> North; Huế, Đà Nẵng -> Central; TP.HCM, Cần Thơ -> South), joined on utterance id via the manifest.
  ViMD test (benchmark results/<model>/vimd.jsonl, ids "<shard>_<row>"): region from the parquet's region column.

Both sides are normalised with the Vietnamese ASR normaliser before scoring.

    python tools/vi_region_wer.py runs/<id>/hyp_zac2026_dev.jsonl results/mel_<id>/vimd.jsonl
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audiollm.vietnamese import normalize_vi_text  # noqa: E402

CITY_REGION = {"Hà Nội": "North", "Huế": "Central", "Đà Nẵng": "Central", "TP.HCM": "South", "Cần Thơ": "South"}


def zac_regions():
    regions = {}
    for manifest in glob.glob("data/asr_corpora/zac2026/*/manifest.jsonl"):
        for line in open(manifest):
            row = json.loads(line)
            regions[row["id"]] = (CITY_REGION.get(row["city"], "?"), row["city"])
    return regions


def vimd_regions():
    import pyarrow.parquet as pq
    regions = {}
    for path in sorted(glob.glob("data/corpora/vimd/data/test-*.parquet")):
        shard = Path(path).stem
        for row_idx, row in enumerate(pq.read_table(path, columns=["region", "province_name"]).to_pylist()):
            regions[f"{shard}_{row_idx}"] = (row["region"], row["province_name"])
    return regions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    maps = {}
    for path in args.files:
        rows = [json.loads(line) for line in open(path)]
        kind = "vimd" if rows and rows[0]["id"].startswith("test-") else "zac"
        if kind not in maps:
            maps[kind] = vimd_regions() if kind == "vimd" else zac_regions()
        groups, sub = defaultdict(lambda: ([], [])), defaultdict(lambda: ([], []))
        for row in rows:
            region, place = maps[kind].get(row["id"], ("?", "?"))
            ref, hyp = normalize_vi_text(row["ref"]), normalize_vi_text(row["hyp"])
            if not ref:
                continue
            for key, store in ((region, groups), ("all", groups), (place, sub)):
                store[key][0].append(ref)
                store[key][1].append(hyp)
        print(f"== {path} ({kind})")
        for key in ["all", "North", "Central", "South", "?"]:
            if key in groups:
                refs, hyps = groups[key]
                print(f"  {key:8s} WER {jiwer.wer(refs, hyps) * 100:6.2f}%  ({len(refs)} utts)")
        if kind == "zac":
            for key, (refs, hyps) in sorted(sub.items()):
                print(f"    {key:10s} WER {jiwer.wer(refs, hyps) * 100:6.2f}%  ({len(refs)} utts)")


if __name__ == "__main__":
    main()
