"""
English ASR benchmark on the Open ASR Leaderboard test sets (the suite used in Table II of arXiv 2606.10231).

Data comes from hf-audio/esb-datasets-test-only-sorted (parquet, 16 kHz audio bytes). Text on both sides goes
through the Whisper EnglishTextNormalizer, which the leaderboard also uses, so numbers are directly comparable to the
leaderboard and to the paper. That's different from the `[a-z' ]` normalisation in train.py, so expect small
differences on LibriSpeech.

    python benchmark.py mel:runs/baseline_fixed12.5hz                      # our checkpoint, default sets
    python benchmark.py hf:openai/whisper-large-v3-turbo --sets librispeech_clean --limit 100
    python benchmark.py --table                                           # all results/ + paper reference rows

Outputs go to results/<tag>/: <set>.jsonl (id, ref, hyp, normalised ref/hyp) and summary.json (WER %, RTFx).
"""
import argparse
import io
import json
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
DEFAULT_SETS = ["librispeech_clean", "librispeech_other", "voxpopuli", "ami", "earnings22"]
OPENASR_SETS = list(SETS)  # current leaderboard English short-form suite minus Common Voice (TED-LIUM was dropped)

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


def read_wav(b):
    wav, sr = sf.read(io.BytesIO(b), dtype="float32")
    assert sr == 16000, sr
    return wav if wav.ndim == 1 else wav.mean(1)


# ---------------------------------------------------------------- models
class MelLLM:
    """Our encoder-free checkpoint. Reuses train.py, with its constants overridden from the run's config.json."""

    def __init__(self, run_dir, batch_size):
        import train
        for k, v in json.load(open(Path(run_dir, "config.json"))).items():
            if k.isupper():
                setattr(train, k, v)
        self.t = train
        self.tok, self.model, self.proj = train.load_model_and_projector(run_dir)
        self.model.eval()
        self.mvn = torch.load(Path(train.DATA, f"mvn_{train.TRAIN_SPLIT}_n{train.MVN_N}.pt"))
        self.prompt_ids = self.tok(train.PROMPT, add_special_tokens=False).input_ids
        self.batch_size = batch_size or train.EVAL_BATCH_SIZE

    def features(self, b):
        t = self.t
        x = (t.extract_log_mel_spectrogram(io.BytesIO(b)) - self.mvn["mean"]) / self.mvn["std"]
        x = torch.cat([x, x.new_zeros((-x.shape[0]) % t.STACK, t.N_MELS)])
        return x.reshape(-1, t.STACK * t.N_MELS)

    @torch.no_grad()
    def transcribe(self, batch):
        t = self.t
        feats = [{"audio": self.features(it["bytes"])} for it in batch]
        x, mask, _ = t.build_multimodal_inputs(feats, self.model, self.proj, self.tok, self.prompt_ids, is_train=False)
        out = self.model.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=t.MAX_NEW_TOKENS,
                                  do_sample=False, eos_token_id=self.tok.eos_token_id,
                                  pad_token_id=self.tok.pad_token_id, use_cache=True)
        return [self.tok.decode(o, skip_special_tokens=True) for o in out]


class HFPipeline:
    """Any transformers ASR checkpoint (Whisper, wav2vec2, Moonshine, ...) via the ASR pipeline."""

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
        return MelLLM(name, batch_size), "mel_" + Path(name).name
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
    return {"wer": round(100 * jiwer.wer(refs, hyps), 2), "n_utt": len(refs), "audio_hours": round(audio_sec / 3600, 2),
            "rtfx": round(audio_sec / wall, 1)}


def print_table():
    rows = dict(PAPER)
    for p in sorted(Path(RESULTS).glob("*/summary.json")):
        rows[p.parent.name] = {k: v["wer"] for k, v in json.load(open(p)).items() if k in SETS}
    cols = [s for s in OPENASR_SETS if any(s in r for r in rows.values())]
    print("| model | " + " | ".join(cols) + " | avg (cols present) |")
    print("|---" * (len(cols) + 2) + "|")
    for name, r in rows.items():
        vals = [r.get(c) for c in cols]
        have = [v for v in vals if v is not None]
        avg = f"{sum(have) / len(have):.2f}" if len(have) == len(cols) else "–"
        print(f"| {name} | " + " | ".join("–" if v is None else f"{v:.2f}" for v in vals) + f" | {avg} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="mel:<run_dir> or hf:<model_id>")
    ap.add_argument("--sets", nargs="+", default=DEFAULT_SETS, help=f"subset of {list(SETS)} or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="evaluate N evenly spaced utterances per set (smoke test)")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--tag", default=None, help="results/<tag> (default derived from model)")
    ap.add_argument("--table", action="store_true", help="print a markdown table of all results and exit")
    args = ap.parse_args()

    if args.table:
        return print_table()
    if not args.model:
        ap.error("model is required unless --table")
    sets = OPENASR_SETS if args.sets == ["all"] else args.sets

    normalizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3").normalize
    model, tag = load_model(args.model, args.batch_size)
    tag = args.tag or tag + (f"_limit{args.limit}" if args.limit else "")
    out_dir = Path(RESULTS, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    summary["_model"] = args.model

    for name in sets:
        items = load_set(name, args.limit)
        summary[name] = run(model, items, normalizer, out_dir / f"{name}.jsonl")
        print(name, summary[name], flush=True)
        json.dump(summary, open(summary_path, "w"), indent=2)

    done = [summary[s]["wer"] for s in OPENASR_SETS if s in summary]
    if len(done) == len(OPENASR_SETS):
        summary["mean_wer"] = round(sum(done) / len(done), 2)
        json.dump(summary, open(summary_path, "w"), indent=2)
    print_table()


if __name__ == "__main__":
    main()
