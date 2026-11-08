"""
ZAC 2026 LifeMate audio -> ASR corpus manifests under <asr_root>/zac2026/{train,dev,holdout,public_test}/manifest.jsonl.

Every user turn is one utterance (final request <id>.wav and history turns <id>_uN.wav). Partitions follow the ZaloAI
repo's frozen split (configs/validation_split.json): "train" -> train, "dev" -> dev (the 277-sample / 864-utterance
ASR dev set the Zipformer baseline is scored on), "tool_holdout" -> holdout. public_test has no transcripts (text "").

Each record also carries:
  hyp   Zipformer-30M-RNNT-6000h hypothesis from the ZaloAI cache, normalised like our targets (correction mode input)
  city  the user's city from the request context (accent proxy for per-region WER)
Also writes <asr_root>/zac2026/confusions.json: word substitutions ref -> hyp mined from the TRAIN partition only.

    python tools/prepare_zac_asr.py --zac /home/nguyenlt/ZaloAI --asr_root data/asr_corpora
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jiwer
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audiollm.vietnamese import normalize_vi_text  # noqa: E402


def user_turns(row):
    """[(utt_id, relative audio path, text or "")] for every user turn of a sample, final request last."""
    turns = []
    for turn in row.get("history", []):
        if turn.get("role") == "user" and turn.get("audio"):
            turns.append((Path(turn["audio"]).stem, turn["audio"], turn.get("audio_text", "")))
    turns.append((row["id"], row["audio"], row.get("audio_text", "")))
    return turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zac", default="/home/nguyenlt/ZaloAI")
    parser.add_argument("--asr_root", default="data/asr_corpora")
    args = parser.parse_args()
    zac = Path(args.zac)
    out_root = Path(args.asr_root, "zac2026")

    split_cfg = json.loads((zac / "configs/validation_split.json").read_text())["partitions"]
    partition = {sid: name for name, ids in split_cfg.items() for sid in ids}
    rename = {"train": "train", "dev": "dev", "tool_holdout": "holdout"}
    hyps = {}
    for line in open(zac / "data/asr_cache/zipformer_train.jsonl"):
        row = json.loads(line)
        for rel, text in row["transcripts"].items():
            hyps[rel] = text

    records = defaultdict(list)
    for split, data_dir in (("train", zac / "data/train"), ("public_test", zac / "data/public_test")):
        for line in open(data_dir / f"{split}.jsonl"):
            row = json.loads(line)
            part = rename[partition[row["id"]]] if split == "train" else "public_test"
            city = row.get("context", {}).get("user_profile", {}).get("city", "")
            for utt_id, rel, text in user_turns(row):
                path = (data_dir / rel).resolve()
                rec = {"id": utt_id, "path": str(path), "text": text, "sec": sf.info(str(path)).duration,
                       "city": city, "sample": row["id"]}
                if rel in hyps:
                    rec["hyp"] = normalize_vi_text(hyps[rel])
                records[part].append(rec)

    for part, recs in records.items():
        d = out_root / part
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        (d / ".done").touch()
        with_hyp = sum("hyp" in r for r in recs)
        print(f"{part}: {len(recs)} utts, {sum(r['sec'] for r in recs) / 3600:.2f} h, {with_hyp} with Zipformer hyp, "
              f"cities {dict(Counter(r['city'] for r in recs).most_common())}")

    # Zipformer WER per partition (our normaliser) = the baseline our model must beat
    for part in ("train", "dev", "holdout"):
        refs, hyps_ = [], []
        for r in records[part]:
            ref = normalize_vi_text(r["text"])
            if ref and "hyp" in r:
                refs.append(ref)
                hyps_.append(r["hyp"])
        print(f"Zipformer WER on zac2026/{part} (our normaliser): {jiwer.wer(refs, hyps_) * 100:.2f}% over {len(refs)} utts")

    # word confusions from the train partition
    pair_counts, ref_totals = Counter(), Counter()
    for r in records["train"]:
        ref, hyp = normalize_vi_text(r["text"]), r.get("hyp", "")
        if not ref or not hyp:
            continue
        ref_words, hyp_words = ref.split(), hyp.split()
        ref_totals.update(ref_words)
        out = jiwer.process_words(ref, hyp)
        for chunk in out.alignments[0]:
            if chunk.type == "substitute" and chunk.ref_end_idx - chunk.ref_start_idx == chunk.hyp_end_idx - chunk.hyp_start_idx:
                for i, j in zip(range(chunk.ref_start_idx, chunk.ref_end_idx), range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                    pair_counts[(ref_words[i], hyp_words[j])] += 1
    mined = [{"ref": a, "hyp": b, "count": n, "ref_total": ref_totals[a], "prob": round(n / ref_totals[a], 4)}
             for (a, b), n in pair_counts.most_common() if n >= 2]
    (out_root / "confusions.json").write_text(json.dumps({"source": "zac2026/train Zipformer hyps",
                                                          "categories": {"mined": mined}}, ensure_ascii=False, indent=1))
    print(f"confusions: {len(mined)} pairs with count >= 2 -> {out_root / 'confusions.json'}")


if __name__ == "__main__":
    main()
