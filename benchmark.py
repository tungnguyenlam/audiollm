"""
English ASR benchmark on the Open ASR Leaderboard test sets (the suite used in Table II of arXiv 2606.10231).

Data comes from hf-audio/esb-datasets-test-only-sorted (parquet, 16 kHz audio bytes). Text on both sides goes
through the Whisper EnglishTextNormalizer, which the leaderboard also uses, so numbers are directly comparable to the
leaderboard and to the paper. That's different from the `[a-z' ]` normalisation in train.py, so expect small
differences on LibriSpeech.

    python benchmark.py mel:runs/baseline_fixed12.5hz                      # our checkpoint, all 7 sets
    python benchmark.py mel:runs/latest --sets earnings22 ami              # newest run, some sets
    python benchmark.py hf:openai/whisper-large-v3-turbo --sets librispeech_clean --limit 100
    python benchmark.py --table                 # paper rows + hf baselines + the best run per base LLM, with its config
    python benchmark.py --table --all_runs      # ... plus every other benchmarked mel: run

"Best run per base LLM": among finished, non-smoke runs in runs/ with the same MODEL (230M, 350M, 1.2B, ...), the one
with the lowest in-training val WER (dev-clean subset). Selection never looks at test sets. If that run has no
benchmark results yet, the table shows train.py's own LibriSpeech test WER (marked †) and the command to benchmark it.

Outputs go to results/<tag>/: <set>.jsonl (id, ref, hyp, normalised ref/hyp) and summary.json (WER %, RTFx).
For mel: runs the per-set WERs are also stored in runs/<id>/summary.json ("benchmark"), shown in runs/INDEX.md and
plots.png. train.py runs this automatically after training (BENCHMARK_SETS, default "all").
"""
import argparse
import io
import json
import os
import time
from pathlib import Path

import jiwer
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import WhisperTokenizer

ESB_REPO = "hf-audio/esb-datasets-test-only-sorted"
ESB_DIR = "data/esb"
RESULTS = "results"
# set name -> (config dir, parquet file prefix). Sizes are approximate download sizes.
SETS = {
    "librispeech_clean": ("librispeech", "test.clean"),  # 0.35 GB
    "librispeech_other": ("librispeech", "test.other"),  # 0.33 GB
    "voxpopuli": ("voxpopuli", "test"),                  # 0.9 GB
    "ami": ("ami", "test"),                              # 1.3 GB
    "earnings22": ("earnings22", "test"),                # 1.1 GB
    "gigaspeech": ("gigaspeech", "test"),                # 4.3 GB
    "spgispeech": ("spgispeech", "test"),                # 11.6 GB
}
OPENASR_SETS = list(SETS)  # current leaderboard English short-form suite minus Common Voice (TED-LIUM was dropped)
# Extra test sets already on disk from download_corpora.py: set name -> (parquet glob, reference text column).
# Reported next to the leaderboard sets but kept out of its mean (and stored as "benchmark_extra" in a run's summary).
LOCAL_SETS = {
    "fleurs_en": ("data/corpora/fleurs_en/parquet-data/en_us/test-*.parquet", "raw_transcription"),  # read Wikipedia
    "edacc": ("data/corpora/edacc/data/test-*.parquet", "text"),  # accented conversational English, never trained on
}
ALL_SETS = OPENASR_SETS + list(LOCAL_SETS)
DECODE_KEYS = ("MAX_NEW_TOKENS", "MAX_TOKENS_PER_SEC", "NO_REPEAT_NGRAM_SIZE")  # audiollm/config.py keys used by mel: decoding

# Table II of arXiv 2606.10231 (WER %). Their Avg also includes TED-LIUM, so only compare per-set columns.
PAPER = {
    "[paper] Whisper-large-v3": dict(ami=15.95, earnings22=11.29, gigaspeech=10.02, librispeech_clean=2.01,
                                     librispeech_other=3.91, spgispeech=2.94, voxpopuli=9.54),
    "[paper] Phi-4-MM": dict(ami=11.69, earnings22=10.16, gigaspeech=9.78, librispeech_clean=1.68,
                             librispeech_other=3.83, spgispeech=3.13, voxpopuli=5.91),
    "[paper] Mel-LLM (Phi-4-MM init)": dict(ami=12.91, earnings22=12.99, gigaspeech=10.95, librispeech_clean=1.70,
                                            librispeech_other=4.83, spgispeech=2.28, voxpopuli=7.76),
    "[paper] Mel-LLM (random init)": dict(ami=13.65, earnings22=11.98, gigaspeech=11.38, librispeech_clean=1.83,
                                          librispeech_other=5.50, spgispeech=2.47, voxpopuli=8.25),
}


# ---------------------------------------------------------------- data
def load_set(name, limit=None):
    """Return list of dicts {id, bytes, text, sec}. The ESB parquets are sorted by duration, longest first."""
    if name in LOCAL_SETS:
        return load_local_set(name, limit)
    config, prefix = SETS[name]
    files = sorted(f for f in list_repo_files(ESB_REPO, repo_type="dataset")
                   if f.startswith(f"{config}/{prefix}-") and f.endswith(".parquet"))
    items = []
    for f in files:
        path = hf_hub_download(ESB_REPO, f, repo_type="dataset", local_dir=ESB_DIR)
        for r in pq.read_table(path, columns=["id", "text", "audio", "audio_length_s"]).to_pylist():
            items.append({"id": r["id"], "bytes": r["audio"]["bytes"], "text": r["text"], "sec": r["audio_length_s"]})
    if limit:  # evenly spaced subset so short and long utterances are both covered
        items = [items[i] for i in np.linspace(0, len(items) - 1, min(limit, len(items))).astype(int)]
    return items


def load_local_set(name, limit=None):
    """LOCAL_SETS test split as {id, bytes, text, sec}, sorted longest first like the ESB sets."""
    pattern, text_column = LOCAL_SETS[name]
    shards = sorted(Path().glob(pattern))
    if not shards:
        raise SystemExit(f"no parquet files for {name} ({pattern}); run download_corpora.py first")
    items = []
    for shard in shards:
        for index, r in enumerate(pq.read_table(shard, columns=[text_column, "audio"]).to_pylist()):
            text = (r[text_column] or "").strip()
            if not text or text == "IGNORE_TIME_SEGMENT_IN_SCORING":
                continue
            info = sf.info(io.BytesIO(r["audio"]["bytes"]))
            items.append({"id": f"{shard.stem}_{index}", "bytes": r["audio"]["bytes"], "text": text,
                          "sec": info.frames / info.samplerate})
    items.sort(key=lambda it: -it["sec"])
    if limit:
        items = [items[i] for i in np.linspace(0, len(items) - 1, min(limit, len(items))).astype(int)]
    return items


def read_wav(b, target_sr=16000):
    """Decode audio bytes to mono float32 at target_sr. Most ESB sets are 16 kHz, but earnings22 is mostly
    24 / 22.05 kHz, so resample (soxr, high quality) when needed."""
    wav, sr = sf.read(io.BytesIO(b), dtype="float32")
    wav = wav if wav.ndim == 1 else wav.mean(1)
    if sr != target_sr:
        import soxr
        wav = soxr.resample(wav, sr, target_sr).astype(np.float32)
    return wav


# ---------------------------------------------------------------- models
class MelLLM:
    """Our encoder-free checkpoint (audiollm.SpeechLLM), with the config overridden from the run's config.json."""

    def __init__(self, run_dir, batch_size, loaded=None):
        """loaded: optional SpeechLLM already in memory (train.py benchmarking right after training), otherwise it is
        loaded from run_dir."""
        from audiollm import config as C
        from audiollm.features import LogMelFeaturizer, load_run_mvn_stats
        from audiollm.model import SpeechLLM
        C.apply_config(C.with_legacy_defaults(json.load(open(Path(run_dir, "config.json")))))
        self.t = C  # the live config: decoding overrides are set on it
        self.run_dir = Path(run_dir)
        self.model = loaded or SpeechLLM.load(run_dir)
        self.model.eval()
        self.featurizer = LogMelFeaturizer(load_run_mvn_stats(run_dir))
        self.batch_size = batch_size or C.EVAL_BATCH_SIZE

    @torch.no_grad()
    def transcribe(self, batch):
        from audiollm.data import collate_speech_batch
        from audiollm.utils import compute_context
        items = [{"features": self.featurizer.from_waveform(read_wav(it["bytes"], self.t.SR)), "target": [],
                  "text": it["text"], "id": it["id"], "sec": it["sec"]} for it in batch]
        with compute_context():
            hypotheses, _ = self.model.transcribe(collate_speech_batch(items))
        return hypotheses


class HFPipeline:
    """Any transformers ASR checkpoint (Whisper, wav2vec2, Moonshine, ...) via the ASR pipeline."""

    run_dir = None

    def __init__(self, model_id, batch_size):
        from transformers import pipeline
        whisper = "whisper" in model_id
        # Whisper's encoder takes at most 30 s; a few LibriSpeech/ESB clips are longer, so chunk those
        self.pipe = pipeline("automatic-speech-recognition", model=model_id, dtype=torch.bfloat16, device="cuda",
                             chunk_length_s=30 if whisper else None)
        self.kwargs = {"generate_kwargs": {"language": "en", "task": "transcribe"}} if whisper else {}
        self.batch_size = batch_size or 16

    def transcribe(self, batch):
        audio = [{"raw": read_wav(it["bytes"]), "sampling_rate": 16000} for it in batch]
        return [o["text"] for o in self.pipe(audio, batch_size=len(audio), **self.kwargs)]


def load_model(spec, batch_size):
    kind, _, name = spec.partition(":")
    if kind == "mel":
        run_dir = Path(os.path.relpath(Path(name).resolve()))  # runs/latest -> runs/<id>, so results stay put
        return MelLLM(run_dir, batch_size), "mel_" + run_dir.name
    if kind == "hf":
        return HFPipeline(name, batch_size), "hf_" + name.replace("/", "__")
    raise SystemExit(f"model spec must be mel:<run_dir> or hf:<model_id>, got {spec}")


# ---------------------------------------------------------------- eval
def run(model, items, normalizer, out_path):
    refs, hyps = [], []
    t0 = time.time()
    with open(out_path, "w") as f:
        for i in range(0, len(items), model.batch_size):
            batch = items[i:i + model.batch_size]
            for it, hyp in zip(batch, model.transcribe(batch)):
                ref_n, hyp_n = normalizer(it["text"]), normalizer(hyp)
                f.write(json.dumps({"id": it["id"], "ref": it["text"], "hyp": hyp, "ref_n": ref_n, "hyp_n": hyp_n}) + "\n")
                if ref_n.strip():  # leaderboard drops references that normalise to empty
                    refs.append(ref_n)
                    hyps.append(hyp_n)
    wall = time.time() - t0
    audio_sec = sum(it["sec"] for it in items)
    alignment = jiwer.process_words(refs, hyps)
    n_ref = max(1, alignment.hits + alignment.substitutions + alignment.deletions)
    return {"wer": round(100 * alignment.wer, 2), "sub": round(100 * alignment.substitutions / n_ref, 2),
            "del": round(100 * alignment.deletions / n_ref, 2), "ins": round(100 * alignment.insertions / n_ref, 2),
            "n_utt": len(refs), "audio_hours": round(audio_sec / 3600, 2), "rtfx": round(audio_sec / wall, 1)}


def resolve_sets(sets):
    """"all" -> every set; also accepts a comma-separated string (train.py's BENCHMARK_SETS)."""
    if isinstance(sets, str):
        sets = [s for s in sets.split(",") if s]
    sets = ALL_SETS if sets == ["all"] else OPENASR_SETS if sets == ["openasr"] else list(sets)
    unknown = [s for s in sets if s not in ALL_SETS]
    if unknown:
        raise SystemExit(f"unknown benchmark sets {unknown}; choose from {ALL_SETS}, 'openasr' or 'all'")
    return sets


def update_run_summary(run_dir, **values):
    """Merge values into runs/<id>/summary.json (atomic), so runs.py / INDEX.md / plots.png see benchmark WERs."""
    path = Path(run_dir, "summary.json")
    summary = json.load(open(path)) if path.exists() else {}
    summary.update(values)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(path)


def benchmark(model, tag, sets, limit=None, log=print, spec=None, track=True):
    """
    Run `model` on `sets`, writing results/<tag>/<set>.jsonl + summary.json (merged with earlier sets). For a mel:
    run without --limit (and track=True, i.e. the run's own decoding settings), also store {set: WER} under "benchmark"
    in the run's own summary.json. Returns the summary.
    """
    out_dir = Path(RESULTS, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary["_model"] = f"mel:{model.run_dir}" if model.run_dir else spec  # mel: always the resolved run dir
    if model.run_dir:
        summary["_decode"] = {key: getattr(model.t, key) for key in DECODE_KEYS}
    normalizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3").normalize
    track_run = model.run_dir is not None and not limit and track
    for name in sets:
        if track_run:
            update_run_summary(model.run_dir, benchmark_status=f"running {name}")
        items = load_set(name, limit)
        summary[name] = run(model, items, normalizer, out_dir / f"{name}.jsonl")
        log(f"[benchmark] {name} {summary[name]}")
        json.dump(summary, open(summary_path, "w"), indent=2)
        if track_run:
            update_run_summary(model.run_dir, benchmark={s: summary[s]["wer"] for s in OPENASR_SETS if s in summary},
                               benchmark_extra={s: summary[s]["wer"] for s in LOCAL_SETS if s in summary},
                               benchmark_results=str(out_dir))

    done = [summary[s]["wer"] for s in OPENASR_SETS if s in summary]
    if len(done) == len(OPENASR_SETS):
        summary["mean_wer"] = round(sum(done) / len(done), 2)
        json.dump(summary, open(summary_path, "w"), indent=2)
    if track_run:
        update_run_summary(model.run_dir, benchmark_status="done")
    return summary


def load_results():
    """{tag: summary} for every results/<tag>/summary.json."""
    return {p.parent.name: json.load(open(p)) for p in sorted(Path(RESULTS).glob("*/summary.json"))}


def result_for_run(results, run_dir):
    """The benchmark result of a run dir: prefer full (non --limit) results, then the one covering most sets."""
    matches = [
        (tag, summary) for tag, summary in results.items()
        if summary.get("_model", "").startswith("mel:")
        and Path(summary["_model"][4:]).resolve() == Path(run_dir).resolve()
    ]
    if not matches:
        return None, None
    return max(matches, key=lambda m: ("_limit" not in m[0], sum(s in m[1] for s in ALL_SETS), m[0]))


def print_table(all_runs=False):
    """
    Markdown table: paper rows, hf: baselines, and the best run per base LLM (lowest in-training val WER, see
    runs.best_runs_by_model). --all_runs adds every other mel: result too. Then the config of each best run.
    """
    import runs as runs_lib

    results = load_results()
    best = runs_lib.best_runs_by_model()
    rows = dict(PAPER)
    for tag, summary in results.items():
        if not summary.get("_model", "").startswith("mel:"):
            rows[tag] = {k: v["wer"] for k, v in summary.items() if k in ALL_SETS}

    best_result_tags = set()
    train_eval_fallback = {"librispeech_clean": "test-clean", "librispeech_other": "test-other"}
    for model, (run, _) in best.items():
        tag, summary = result_for_run(results, run["dir"])
        best_result_tags.add(tag)
        row = {k: v["wer"] for k, v in summary.items() if k in ALL_SETS} if summary else {}
        for bench_set, split in train_eval_fallback.items():  # not benchmarked: show train.py's own test eval (†)
            if bench_set not in row and split in run["summary"].get("test_wer", {}):
                row[bench_set] = f"{run['summary']['test_wer'][split]:.2f}†"
        rows[f"**best {model.split('/')[-1]}** ({run['id']})"] = row
    if all_runs:
        for tag, summary in results.items():
            if summary.get("_model", "").startswith("mel:") and tag not in best_result_tags:
                rows[tag] = {k: v["wer"] for k, v in summary.items() if k in ALL_SETS}

    cols = [s for s in ALL_SETS if any(s in r for r in rows.values())]
    avg_cols = [c for c in cols if c in OPENASR_SETS]  # local extra sets are shown but not averaged
    print("| model | " + " | ".join(cols) + " | OpenASR avg (cols present) |")
    print("|---" * (len(cols) + 2) + "|")
    for name, r in rows.items():
        vals = [r.get(c) for c in cols]
        have = [r.get(c) for c in avg_cols if isinstance(r.get(c), (int, float))]
        avg = f"{sum(have) / len(have):.2f}" if have and len(have) == len(avg_cols) else "–"
        cells = ["–" if v is None else v if isinstance(v, str) else f"{v:.2f}" for v in vals]
        print(f"| {name} | " + " | ".join(cells) + f" | {avg} |")
    if any(isinstance(v, str) for r in rows.values() for v in r.values()):
        print("\n† not benchmarked yet: LibriSpeech WER from train.py's own test eval ([a-z' ] normalisation, "
              "not the Whisper normaliser)")

    if not best:
        print(f"\nno finished training runs in {runs_lib.RUNS_DIR}/ yet")
        return
    from audiollm import config as C  # only for the defaults, to mark which config values a run changed
    print("\n## Best run per base LLM (selected by in-training val WER; test sets are never used for selection)")
    for model, (run, n_candidates) in best.items():
        summary, meta = run["summary"], run["meta"]
        tag, _ = result_for_run(results, run["dir"])
        cfg = run["config"]
        print(f"\n### {model}: {run['dir']}")
        print(f"- selected: val WER {summary['best_val_wer']:.2f} on {cfg.get('VAL_SPLIT', '?')}[:{cfg.get('VAL_N', '?')}] "
              f"@ step {summary.get('best_val_step', '?')}/{summary.get('total_steps', '?')}, "
              f"best of {n_candidates} finished run(s) for this LLM")
        if summary.get("test_wer"):
            print("- train.py test eval: " + ", ".join(f"{k} {v:.2f}" for k, v in summary["test_wer"].items()))
        print(f"- benchmark: results/{tag}/" if tag else f"- benchmark: not run yet -> python benchmark.py mel:{run['dir']}")
        if meta.get("command"):
            print(f"- command: `{meta['command']}`")
        if meta.get("notes") or summary.get("notes"):
            print(f"- notes: {meta.get('notes') or summary.get('notes')}")
        if meta.get("git_commit"):
            print(f"- git: {meta['git_commit'][:10]}{' (dirty)' if meta.get('git_dirty') else ''}")
        items = [
            f"{key}={runs_lib.fmt(val)}" + ("*" if C.DEFAULT_CONFIG.get(key, val) != val else "")
            for key, val in cfg.items() if key not in runs_lib.IGNORED_KEYS
        ]
        if "PRECISION" not in cfg:
            items.append("PRECISION=fp32(implicit,pre-PRECISION run)*")
        lines, line = [], ""
        for item in items:
            if line and len(line) + len(item) + 1 > 110:
                lines.append(line)
                line = ""
            line = f"{line} {item}".strip()
        print("- config (* = differs from current train.py defaults):")
        print("\n".join("      " + line for line in lines + [line]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="mel:<run_dir> or hf:<model_id>")
    ap.add_argument("--sets", nargs="+", default=["all"],
                    help=f"subset of {ALL_SETS}, 'openasr' (leaderboard 7) or 'all' (leaderboard + local test sets)")
    ap.add_argument("--limit", type=int, default=None, help="evaluate N evenly spaced utterances per set (smoke test)")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--tag", default=None, help="results/<tag> (default derived from model)")
    ap.add_argument("--table", action="store_true",
                    help="print the results table (best run per base LLM + its config) and exit")
    ap.add_argument("--all_runs", action="store_true", help="with --table: also list non-best mel: results")
    decode = ap.add_argument_group("mel: decoding overrides (default: the run's config.json, else train.py defaults)")
    decode.add_argument("--max_new_tokens", type=int, default=None)
    decode.add_argument("--max_tokens_per_sec", type=float, default=None, help="per-utterance cap, 0 = off")
    decode.add_argument("--no_repeat_ngram_size", type=int, default=None, help="0 = off")
    args = ap.parse_args()

    if args.table:
        return print_table(args.all_runs)
    if not args.model:
        ap.error("model is required unless --table")
    sets = resolve_sets(args.sets)

    model, tag = load_model(args.model, args.batch_size)
    overrides = {key: getattr(args, key.lower()) for key in DECODE_KEYS if getattr(args, key.lower()) is not None}
    if overrides and not model.run_dir:
        ap.error("decoding overrides only apply to mel: models")
    for key, val in overrides.items():
        setattr(model.t, key, val)
    decode_suffix = "".join(f"_{key.lower()}{val:g}" for key, val in overrides.items())
    tag = args.tag or tag + decode_suffix + (f"_limit{args.limit}" if args.limit else "")
    benchmark(model, tag, sets, args.limit, log=lambda msg: print(msg, flush=True), spec=args.model,
              track=not overrides)
    if model.run_dir and not args.limit and not overrides:
        import runs as runs_lib
        runs_lib.refresh_run_views(model.run_dir)
    print_table()


if __name__ == "__main__":
    main()
