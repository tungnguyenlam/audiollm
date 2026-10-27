"""
Baseline: encoder-free Mel -> LLM ASR, fixed-rate time-only chunking (Mel-LLM style, arXiv 2606.10231).

    wav -> 80-dim log-mel (10 ms hop) -> MVN (train stats) -> stack STACK=16 frames (time-only chunk, full freq)
        -> Linear(STACK*80 -> d_model) -> [BOS][audio tokens]["Transcription:"][text][EOS] -> LFM2.5-230M-Base + LoRA

The UPPERCASE constants at the top of this file are the defaults. Every one of them can be overridden with a
lower-case flag of the same name (LR -> --lr, TEST_SPLITS -> --test_splits a b, GRAD_CKPT -> --no-grad_ckpt).
`python train.py --help` lists them all.

Each training run gets its own directory runs/<YYYYmmdd-HHMMSS>_<run_name>/ with:
    config.json      resolved config (UPPERCASE keys) + "meta" (command, git commit/dirty, overrides, notes, ...)
    git_diff.patch   uncommitted changes at launch time (only if the tree was dirty)
    metrics.jsonl    time series: one JSON row per train log window / val pass / test split
    summary.json     status (running/done/failed/interrupted), best val WER, test WER, duration
    log.txt, lora/, proj.pt, hyp_<split>.jsonl, test_results.json
Compare runs with `python runs.py` (see runs.py --help).

    python train.py                                   # train with the defaults below
    python train.py --run_name lr3e-4 --lr 3e-4 --notes "higher lr"
    python train.py --model LiquidAI/LFM2.5-350M --stack 8       # run dir name defaults to <model>_stack<STACK>
    python train.py --config runs/<id>/config.json --stack 4   # start from a previous run's config, tweak one knob
    python train.py --max_steps 50                    # smoke test
    python train.py --eval_only runs/<id>             # re-run test eval with that run's saved config
"""
import argparse
import contextlib
import json
import math
import os
import random
import re
import shlex
import socket
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import runs as runs_lib

# ---------------------------------------------------------------- config
RUN_NAME = ""                # run directory suffix; empty = auto "<model>_stack<STACK>" (e.g. LFM2.5-230M-Base_stack16)
SEED = 0                     # Random seed for reproducibility
MODEL = "LiquidAI/LFM2.5-230M-Base"  # small model for fast experiments (2.6B-Base is the paper-scale target)
DATA = "data/LibriSpeech"    # Path to dataset directory
ASR_CORPORA = "data/asr_corpora"  # unpacked non-LibriSpeech corpora (unpack_parquet_asr_corpora.py), addressed as "<corpus>/<split>"
TRAIN_SPLIT = "train-clean-100+train-clean-360+ami_ihm/train+fleurs_en/train"  # Training split(s), "+"-joined: LibriSpeech names or "<corpus>/<split>" (100+360 h read speech, ~80 h AMI meetings, ~7 h FLEURS); "train-clean-100" = pre-2026-10-27 runs. "<split>*K" repeats a split K times per epoch (fractional K adds a seeded random subset), e.g. "fleurs_en/train*6"
MVN_SPLIT = ""               # split spec whose cached MVN stats normalise the mel features; "" = TRAIN_SPLIT without "*K" weights (INIT_FROM runs inherit the init run's)
INIT_FROM = ""               # run dir to initialise LoRA + projector weights from (fresh optimizer, warmup and schedule), e.g. runs/20261027-150147_mix544h_conv_stack8; "" = fresh init
MIN_TRAIN_SEC = 1.0          # drop shorter train utts (mostly AMI back-channels like "yeah", "mm-hmm")
VAL_SPLIT = "dev-clean"      # Validation dataset split
VAL_N = 200                  # fixed dev subset used for in-training WER
TEST_SPLITS = ["test-clean", "test-other"]  # Test dataset splits
MAX_TRAIN_SEC = 30.0         # drop the few very long utterances
TRAIN_N = 0                  # 0 = all train utts; N > 0 = a fixed random subset of N (seeded), e.g. 10 for an overfit test
VAL_FROM_TRAIN = False       # validate on the training utterances themselves (first VAL_N of the train set) instead of VAL_SPLIT: overfit / memorisation test
EVAL_AUDIO_ABLATIONS = ""    # comma list of extra decodes per val pass to test audio dependence: "shuffle" (each utt gets its batch neighbour's audio), "zeros" (all-zero mel); logged as wer_<ablation>

# Audio feature extraction configuration
SR = 16000                   # Target audio sample rate (Hz)
N_FFT = 400                  # STFT window size (400 samples = 25 ms at 16 kHz)
HOP = 160                    # STFT hop size (160 samples = 10 ms at 16 kHz -> 100 frames/s)
N_MELS = 80                  # Number of mel filterbank channels
MVN_N = 2000                 # Number of utterances sampled to compute dataset Mean-Variance Normalization stats
STACK = 16                   # Mel frames stacked per audio token: 16 -> 1280-d patch (>= 230M hidden 1024) at 100/16 = 6.25 Hz (8 = paper's 12.5 Hz)

# Model and optimization hyperparameters
PROMPT = "Transcription:"    # Text prompt prepended to transcriptions
LORA_R = 64                  # LoRA rank dimension
LORA_ALPHA = 128             # LoRA scaling factor (alpha)
LORA_DROPOUT = 0.05          # LoRA dropout probability
LORA_TARGETS = "all-linear"  # LLM modules that get LoRA (projector is always fully trained): "all-linear" (PEFT: every nn.Linear in the LLM blocks, i.e. attn q/k/v/out_proj + conv in/out_proj + MLP w1/w2/w3, not lm_head/embeddings) or a comma list e.g. "q_proj,k_proj,v_proj,out_proj"
LR = 1e-4                    # Peak learning rate: 1e-4, reached after linear warmup, then decayed per LR_SCHEDULE
LR_SCHEDULE = "cosine"       # Decay after warmup: "cosine" (half-cosine annealing to MIN_LR_RATIO * LR) | "linear" (to 0, pre-2026-10-27 runs)
MIN_LR_RATIO = 0.0           # Cosine floor as a fraction of LR (e.g. 0.1 -> anneal to 1e-5)
WEIGHT_DECAY = 0.0           # Weight decay for optimizer
PROJECTOR = "linear"         # audio adapter: "linear" (nn.Linear(STACK*N_MELS -> d)) | "mlp" (LayerNorm + 2-layer GELU MLP on stacked patches) | "conv" (Mel-LLM style strided Conv1d x log2(STACK) on continuous mel + LayerNorm + 2-layer GELU MLP)
PROJ_INIT = "match_emb"      # projector output init: "match_emb" (N(0, emb_std/sqrt(fan_in)), zero bias -> audio embeddings ~ the LLM's text-embedding scale) | "default" (PyTorch nn.Linear init, ~20x larger than text embeddings; pre-2026-10-27 runs)
PROJ_LR_MULT = 1.0           # projector lr = LR * PROJ_LR_MULT (LoRA uses LR)
PROJ_ONLY_STEPS = 0          # optimizer steps with LoRA frozen (only the projector trains) before LoRA joins with its own WARMUP_STEPS warmup
OPTIMIZER = "adamw"          # "adamw" (all trainable params) | "muon" (torch.optim.Muon on every trainable 2D linear weight = LoRA A/B + projector weight; no AdamW anywhere, so the projector bias is frozen at init; the LLM's depthwise conv is frozen either way)
MUON_ADJUST_LR = "match_rms_adamw"  # Muon per-matrix lr scaling: "match_rms_adamw" (update RMS ~ AdamW's, so LR/WEIGHT_DECAY transfer from AdamW) | "original" (Keller Jordan's sqrt(max(1, A/B)); wants LR ~ 0.02)
WARMUP_STEPS = 200           # Linear learning rate warmup steps
EPOCHS = 2                   # Total number of training epochs
TRAIN_STEPS = 0              # >0: train exactly this many optimizer steps (LR schedule spans them; epochs repeat as needed) instead of EPOCHS, e.g. to match budgets across data mixes
BATCH_SIZE = 8               # Training batch size per device
GRAD_ACCUM = 4               # Gradient accumulation steps -> effective batch size = 8 * 4 = 32
GRAD_CLIP = 1.0              # Maximum gradient norm clipping threshold
GRAD_CKPT = True             # Enable gradient checkpointing to reduce VRAM
LOG_EVERY = 20               # Step interval for logging training progress
EVAL_EVERY = 500             # Evaluate on validation split every N optimizer steps
EVAL_BATCH_SIZE = 16         # Evaluation batch size
MAX_NEW_TOKENS = 256         # Hard cap on generated tokens during evaluation (pre-2026-10-27 runs: 200)
MAX_TOKENS_PER_SEC = 6.0     # Per-utterance decode cap: ceil(MAX_TOKENS_PER_SEC * sec) + MAX_TOKENS_PAD tokens (0 = only MAX_NEW_TOKENS); no train transcript exceeds 6*sec+10
MAX_TOKENS_PAD = 10          # Constant slack added to the per-utterance cap
NO_REPEAT_NGRAM_SIZE = 4     # >0: forbid repeating any n-gram of this size in a hypothesis (HF generate); 0 = off (pre-2026-10-27 runs). 4 won the decode A/B (WORKLOG)
NUM_WORKERS = 8              # Number of DataLoader worker processes
LOG_N_EXAMPLES = 2           # Number of (ref, hyp) examples printed after each validation pass
PLOT_EVERY = 100             # Re-draw runs/<id>/plots.png (+ runs/INDEX.md) every N optimizer steps and after each val
BENCHMARK_SETS = "all"       # After training + test eval, run benchmark.py with the in-memory model: "all" (7 Open ASR sets + local fleurs_en/edacc test sets), "openasr", comma list (e.g. "librispeech_clean,ami") or "none"

DEVICE = "cuda"              # Compute device (cuda or cpu)
DTYPE = torch.bfloat16       # Base model torch dtype
PRECISION = "bf16"           # bf16: LoRA+proj weights/grads/AdamW states in DTYPE (fastest) | amp: fp32 master weights + DTYPE autocast | fp32: fp32 LoRA+proj, no autocast (pre-2026-10-27 runs)


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
PRECISIONS = ("bf16", "amp", "fp32")  # tuple, so it is not picked up as a config constant
OPTIMIZERS = ("adamw", "muon")
PROJECTORS = ("linear", "mlp", "conv")


def get_config_dictionary():
    """Extract all uppercase global constants into a dictionary for configuration saving."""
    config = {
        key: val
        for key, val in globals().items()
        if key.isupper() and isinstance(val, (int, float, str, list, bool))
    }
    config["DTYPE"] = str(DTYPE).removeprefix("torch.")  # JSON-serialisable; converted back in apply_config
    return config


DEFAULT_CONFIG = get_config_dictionary()  # snapshot of the constants above, before any override


def apply_config(overrides):
    """
    Override the module-level config constants (from CLI flags or a saved config.json) and rebuild the
    globals derived from them. Non-uppercase keys (e.g. "meta") are ignored.

    Returns:
        list[str]: uppercase keys that are not known config constants (ignored, e.g. from an older config.json).
    """
    global MEL_FILTERBANK, STFT_WINDOW
    unknown_keys = []
    for key, val in overrides.items():
        if not key.isupper():
            continue
        if key not in DEFAULT_CONFIG:
            unknown_keys.append(key)
            continue
        if key == "DTYPE" and isinstance(val, str):
            val = DTYPES[val]
        globals()[key] = val
    MEL_FILTERBANK = build_htk_mel_filterbank()
    STFT_WINDOW = torch.hann_window(N_FFT)
    return unknown_keys


def read_constant_comments():
    """Map each config constant to its trailing `# comment` in this file, used as --help text."""
    comments = {}
    for line in Path(__file__).read_text().splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]*)\s*=.*?#\s*(.+)$", line)
        if match:
            comments[match.group(1)] = match.group(2).strip()
    return comments


def build_arg_parser():
    """One flag per config constant (lower-cased name), plus run-control flags."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate encoder-free Mel-to-LLM ASR model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_group = parser.add_argument_group("run control")
    run_group.add_argument("--config", type=str, default=None,
                           help="JSON config to start from (e.g. runs/<id>/config.json); explicit flags still win")
    run_group.add_argument("--max_steps", type=int, default=None, help="Cap optimizer steps (for smoke tests)")
    run_group.add_argument("--eval_only", type=str, default=None,
                           help="Run directory to evaluate without training (loads its config.json)")
    run_group.add_argument("--notes", type=str, default="", help="Free-text description of the run, saved in config.json")

    hparam_group = parser.add_argument_group("config (defaults = UPPERCASE constants at the top of train.py)")
    comments = read_constant_comments()
    for key, default in DEFAULT_CONFIG.items():
        help_text = comments.get(key)
        kwargs = {
            "dest": key,
            "default": default,
            "help": help_text.replace("%", "%%") if help_text else "(default: %(default)s)",
        }
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        elif isinstance(default, list):
            kwargs.update(nargs="*", type=type(default[0]) if default else str)  # "*": a bare `--test_splits` = none
        elif key == "DTYPE":
            kwargs["choices"] = list(DTYPES)
        elif key == "PRECISION":
            kwargs["choices"] = PRECISIONS
        elif key == "OPTIMIZER":
            kwargs["choices"] = OPTIMIZERS
        elif key == "PROJECTOR":
            kwargs["choices"] = PROJECTORS
        else:
            kwargs["type"] = type(default)
        hparam_group.add_argument("--" + key.lower(), **kwargs)
    return parser


def parse_args(argv=None):
    """
    Resolve the config with precedence: explicit CLI flags > --config file (or the --eval_only run's config.json)
    > constants at the top of this file.

    Returns:
        tuple[argparse.Namespace, dict]: parsed args and the resolved config dict (UPPERCASE keys).
    """
    parser = build_arg_parser()
    pre_args, _ = parser.parse_known_args(argv)
    base_config_path = pre_args.config or (Path(pre_args.eval_only, "config.json") if pre_args.eval_only else None)
    if base_config_path:
        base_config = json.load(open(base_config_path))
        unknown_keys = [key for key in base_config if key.isupper() and key not in DEFAULT_CONFIG]
        if unknown_keys:
            print(f"warning: ignoring unknown keys from {base_config_path}: {unknown_keys}", file=sys.stderr)
        parser.set_defaults(**{key: val for key, val in base_config.items() if key in DEFAULT_CONFIG})
        if "PRECISION" not in base_config:
            parser.set_defaults(PRECISION="fp32")  # configs saved before PRECISION existed trained fp32 LoRA+proj
        if "PROJ_INIT" not in base_config:
            parser.set_defaults(PROJ_INIT="default")  # configs saved before PROJ_INIT existed used PyTorch's init
        if "MIN_TRAIN_SEC" not in base_config:
            parser.set_defaults(MIN_TRAIN_SEC=0.0)  # no minimum length before MIN_TRAIN_SEC existed
        if "PROJECTOR" not in base_config:
            parser.set_defaults(PROJECTOR="linear")
    args = parser.parse_args(argv)
    config = {key: getattr(args, key) for key in DEFAULT_CONFIG}
    return args, config


def get_git_info():
    """Commit hash, whether tracked files have uncommitted changes, and the diff against HEAD."""
    def git(*cmd):
        return subprocess.run(["git", *cmd], capture_output=True, text=True).stdout
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip()),
        "diff": git("diff", "HEAD"),
    }


def create_run_dir(run_name):
    """Create a fresh, never-reused directory runs/<YYYYmmdd-HHMMSS>_<run_name>[_N]."""
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{run_name}"
    run_dir, suffix = Path("runs", run_id), 1
    while run_dir.exists():
        suffix += 1
        run_dir = Path("runs", f"{run_id}_{suffix}")
    run_dir.mkdir(parents=True)
    return run_dir


class RunTracker:
    """Per-run bookkeeping: metrics.jsonl (append-only time series) and summary.json (status + headline numbers)."""

    def __init__(self, run_dir, **initial_summary):
        self.run_dir = Path(run_dir)
        summary_path = self.run_dir / "summary.json"
        self.summary = json.load(open(summary_path)) if summary_path.exists() else {}
        self.update(**initial_summary)

    def log_metrics(self, kind, **values):
        """Append one row, e.g. log_metrics("val", step=500, wer=0.12)."""
        row = {"type": kind, "time": round(time.time(), 1), **values}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def update(self, **values):
        """Merge values into summary.json (atomic rewrite, so a crash never leaves it half-written)."""
        self.summary.update(values)
        tmp_path = self.run_dir / "summary.json.tmp"
        tmp_path.write_text(json.dumps(self.summary, indent=2))
        tmp_path.replace(self.run_dir / "summary.json")


def seed_everything(seed):
    """Seed all pseudo-random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_progress_bar(**tqdm_kwargs):
    """
    Create a tqdm progress bar. When stderr is not a terminal (e.g. `> train.log 2>&1`),
    refresh rarely so the redirected log is not flooded with carriage-return updates.
    """
    is_interactive = sys.stderr.isatty()
    return tqdm(
        dynamic_ncols=True,
        mininterval=0.5 if is_interactive else 60.0,
        file=sys.stderr,
        **tqdm_kwargs,
    )


def format_duration(seconds):
    """Format a duration in seconds as H:MM:SS."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------- data
def load_librispeech_split(split_name):
    """
    Scan LibriSpeech directory for the given split and return a deterministic list of utterances.

    Returns:
        list[dict]: A list of dicts with keys:
            - 'id': utterance ID (e.g., '103-1240-0000')
            - 'path': absolute path to the .flac file
            - 'text': ground-truth transcript (lowercased)
            - 'sec': audio duration in seconds
    """
    split_dir = Path(DATA, split_name)
    manifest_path = split_dir / "manifest.jsonl"
    if manifest_path.exists():
        return [json.loads(line) for line in manifest_path.read_text().splitlines()]

    utterance_records = []
    for transcript_file in sorted(split_dir.rglob("*.trans.txt")):
        for line in transcript_file.read_text().splitlines():
            utterance_id, text = line.split(" ", 1)
            audio_path = transcript_file.parent / f"{utterance_id}.flac"
            utterance_records.append({
                "id": utterance_id,
                "path": str(audio_path),
                "text": text.lower(),
                "sec": sf.info(str(audio_path)).duration,
            })
    if utterance_records and (split_dir / ".done").exists():
        manifest_path.write_text("".join(json.dumps(r) + "\n" for r in utterance_records))
    return utterance_records


def normalize_training_text(text):
    """Training target / reference text: strip annotation tags, lowercase ASCII, keep letters/digits/apostrophes."""
    if text.strip() == "IGNORE_TIME_SEGMENT_IN_SCORING":
        return ""
    text = re.sub(r"<[^>]+>|\[[^\]]+\]", " ", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text)).strip()


def load_split(split_spec):
    """
    Utterances for a split spec: a LibriSpeech split name ("train-clean-100"), an unpacked corpus split
    ("ami_ihm/train", see unpack_parquet_asr_corpora.py), or several of them joined with "+". A "*K" suffix
    ("fleurs_en/train*6") repeats that split's utterances K times; a fractional K adds a SEED-fixed random subset.
    Returns the same dicts as load_librispeech_split(), with text normalised by normalize_training_text().
    """
    records = []
    for part in split_spec.split("+"):
        name, _, weight_text = part.partition("*")
        weight = float(weight_text) if weight_text else 1.0
        if "/" in name:
            manifest = Path(ASR_CORPORA, name, "manifest.jsonl")
            split_records = [json.loads(line) for line in manifest.read_text().splitlines()]
        else:
            split_records = load_librispeech_split(name)
        for record in split_records:
            record["text"] = normalize_training_text(record["text"])
        split_records = [record for record in split_records if record["text"]]
        repeats, fraction = int(weight), weight - int(weight)
        records += split_records * repeats
        if fraction > 0:
            records += random.Random(SEED).sample(split_records, round(fraction * len(split_records)))
    return records


def unweighted_split(split_spec):
    """Split spec without "*K" weights: "a*2+b" -> "a+b"."""
    return re.sub(r"\*[0-9.]+", "", split_spec)


def build_htk_mel_filterbank():
    """
    Construct an HTK-style triangular mel filterbank matrix.

    Returns:
        torch.Tensor: Filterbank weights of shape (N_MELS, N_FFT // 2 + 1) = (80, 201).
    """
    hz_to_mel = lambda freq_hz: 2595.0 * np.log10(1.0 + freq_hz / 700.0)
    mel_to_hz = lambda mel_val: 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)

    # Linearly spaced points in mel scale converted back to Hz
    mel_bin_frequencies_hz = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2), N_MELS + 2))
    fft_frequencies_hz = np.fft.rfftfreq(N_FFT, 1.0 / SR)

    filterbank_weights = np.zeros((N_MELS, len(fft_frequencies_hz)))
    for mel_idx in range(N_MELS):
        left_hz = mel_bin_frequencies_hz[mel_idx]
        center_hz = mel_bin_frequencies_hz[mel_idx + 1]
        right_hz = mel_bin_frequencies_hz[mel_idx + 2]
        # Triangular filter response
        filterbank_weights[mel_idx] = np.maximum(
            0.0,
            np.minimum(
                (fft_frequencies_hz - left_hz) / (center_hz - left_hz),
                (right_hz - fft_frequencies_hz) / (right_hz - center_hz),
            ),
        )
    return torch.tensor(filterbank_weights, dtype=torch.float32)


MEL_FILTERBANK = build_htk_mel_filterbank()
STFT_WINDOW = torch.hann_window(N_FFT)


def extract_log_mel_spectrogram(audio_filepath):
    """
    Load an audio file and extract 80-channel log mel-spectrogram features.

    Args:
        audio_filepath (str or Path): Path to the .flac audio file.

    Returns:
        torch.Tensor: Log mel-spectrogram tensor of shape (num_frames, N_MELS).
    """
    waveform, sample_rate = sf.read(audio_filepath, dtype="float32")
    assert sample_rate == SR, f"Expected sample rate {SR}, but got {sample_rate} for {audio_filepath}"
    return log_mel_from_waveform(waveform)


def log_mel_from_waveform(waveform):
    """(num_samples,) float32 mono waveform at SR -> (num_frames, N_MELS) log mel-spectrogram."""
    # Short-Time Fourier Transform (STFT) -> Power Spectrogram
    stft_complex = torch.stft(
        torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)),
        n_fft=N_FFT,
        hop_length=HOP,
        window=STFT_WINDOW,
        return_complex=True,
    )
    power_spectrogram = stft_complex.abs() ** 2

    # Project power spectrogram onto mel filterbank bins and compute log
    mel_energies = MEL_FILTERBANK @ power_spectrogram
    log_mel = torch.log(mel_energies + 1e-6).T  # Transpose to shape (num_frames, N_MELS)
    return log_mel


def mvn_split():
    """Split spec whose MVN stats are used: MVN_SPLIT, else TRAIN_SPLIT without "*K" weights."""
    return MVN_SPLIT or unweighted_split(TRAIN_SPLIT)


def mvn_cache_path():
    """MVN stats depend on the split, sample size and every feature-extraction knob, so all of them are in the name."""
    return Path(DATA, f"mvn_{mvn_split().replace('/', '-')}_n{MVN_N}_sr{SR}_fft{N_FFT}_hop{HOP}_mel{N_MELS}.pt")


def compute_or_load_mvn_stats(dataset_items):
    """
    Compute global Mean-Variance Normalization (MVN) statistics over a sample of train utterances.
    Caches the resulting mean and std vectors in mvn_cache_path(). dataset_items is only used when mvn_split() is
    the (unweighted) TRAIN_SPLIT; otherwise the MVN split is loaded.

    Returns:
        dict: {"mean": torch.Tensor(N_MELS,), "std": torch.Tensor(N_MELS,)}
    """
    cache_filepath = mvn_cache_path()
    if cache_filepath.exists():
        return torch.load(cache_filepath)
    if mvn_split() != TRAIN_SPLIT:
        dataset_items = [item for item in load_split(mvn_split()) if MIN_TRAIN_SEC <= item["sec"] <= MAX_TRAIN_SEC]

    deterministic_rng = random.Random(SEED)
    sampled_items = deterministic_rng.sample(dataset_items, min(MVN_N, len(dataset_items)))
    concatenated_mel_features = torch.cat(
        [extract_log_mel_spectrogram(item["path"]) for item in sampled_items]
    )
    mvn_stats = {
        "mean": concatenated_mel_features.mean(dim=0),
        "std": concatenated_mel_features.std(dim=0),
    }
    torch.save(mvn_stats, cache_filepath)
    return mvn_stats


class SpeechDataset(torch.utils.data.Dataset):
    """
    Dataset that loads audio waveforms, computes normalized mel-spectrograms,
    stacks STACK consecutive time frames into 1 token representation, and encodes text targets.
    """
    def __init__(self, dataset_items, mvn_stats, tokenizer):
        self.dataset_items = dataset_items
        self.mvn_stats = mvn_stats
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, index):
        item = self.dataset_items[index]

        # 1. Compute and normalize log mel-spectrogram: (num_frames, 80)
        normalized_mel = (extract_log_mel_spectrogram(item["path"]) - self.mvn_stats["mean"]) / self.mvn_stats["std"]

        # 2. Pad tail with zeros so the total frame count is divisible by STACK.
        # (Zero represents the mean value after MVN normalization)
        padding_frames_needed = (-normalized_mel.shape[0]) % STACK
        if padding_frames_needed > 0:
            zero_padding = normalized_mel.new_zeros(padding_frames_needed, N_MELS)
            normalized_mel = torch.cat([normalized_mel, zero_padding], dim=0)

        # 3. Stack STACK consecutive frames across all N_MELS bins -> (num_tokens, STACK * N_MELS), e.g. 16 * 80 = 1280
        stacked_audio_features = normalized_mel.reshape(-1, STACK * N_MELS)

        # 4. Tokenize target transcript: append a leading space and trailing [EOS] token
        target_token_ids = (
            self.tokenizer(" " + item["text"], add_special_tokens=False).input_ids
            + [self.tokenizer.eos_token_id]
        )

        return {
            "audio": stacked_audio_features,  # Shape: (num_audio_tokens, STACK * N_MELS)
            "target": target_token_ids,       # List of token IDs
            "text": item["text"],
            "id": item["id"],
        }


def collate_raw_batch(batch):
    """
    Pass raw list of items through; GPU handles dynamic projection and padding in build_multimodal_inputs.
    """
    return batch


def create_length_batched_sampler(dataset_items, batch_size, seed):
    """
    Bucket utterances by audio duration to minimize padding overhead:
    shuffle items, sort within chunks of 50 batches, then shuffle the resulting batches.
    """
    rng = random.Random(seed)
    item_indices = list(range(len(dataset_items)))
    rng.shuffle(item_indices)

    chunk_size = batch_size * 50
    batches = []
    for chunk_start in range(0, len(item_indices), chunk_size):
        chunk_indices = item_indices[chunk_start : chunk_start + chunk_size]
        sorted_chunk = sorted(chunk_indices, key=lambda idx: dataset_items[idx]["sec"])
        batches += [
            sorted_chunk[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(sorted_chunk), batch_size)
        ]
    rng.shuffle(batches)
    return batches


def init_dataloader_worker(_worker_id):
    """Set torch single-threaded inside each DataLoader worker process to prevent CPU oversubscription."""
    torch.set_num_threads(1)


def create_dataloader(speech_dataset, batch_sampler):
    """Create a PyTorch DataLoader configured with multi-worker prefetching."""
    return torch.utils.data.DataLoader(
        speech_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_raw_batch,
        num_workers=NUM_WORKERS,
        worker_init_fn=init_dataloader_worker,
        persistent_workers=False,
    )


# ---------------------------------------------------------------- model
def trainable_dtype():
    """dtype of the trainable weights (LoRA + projector), their grads and the AdamW states."""
    return DTYPE if PRECISION == "bf16" else torch.float32


def compute_context():
    """
    Context for forward passes. In "amp" mode, autocast runs the fp32 LoRA/projector matmuls in DTYPE
    (the weights and optimizer stay fp32); otherwise every op already runs in its weights' dtype.
    """
    if PRECISION == "amp" and DTYPE != torch.float32:
        return torch.autocast(device_type=torch.device(DEVICE).type, dtype=DTYPE)
    return contextlib.nullcontext()


def build_multimodal_inputs(batch, model, audio_projector, tokenizer, prompt_token_ids, is_train):
    """
    Project stacked mel features into token embedding space, concatenate with text prompt/target,
    and build padded input embeddings and attention masks.

    Sequence structure:
        [BOS] + [Audio Embeddings] + [Prompt: "Transcription:"] + [Target Text Tokens] + [EOS]

    Args:
        batch (list[dict]): Batch items containing 'audio' and 'target'.
        model (nn.Module): The base causal language model.
        audio_projector (nn.Linear): Linear layer mapping (STACK * N_MELS -> hidden_size).
        tokenizer: Hugging Face tokenizer.
        prompt_token_ids (list[int]): Token IDs for the prompt prefix (e.g., 'Transcription:').
        is_train (bool): If True, right-pads sequences and constructs training labels (with audio/prompt masked to -100).
                         If False, left-pads sequences for autoregressive generation.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - padded_inputs_embeds: Float tensor of shape (batch_size, max_seq_len, hidden_size).
            - attention_mask: Long tensor of shape (batch_size, max_seq_len) with 1 for tokens, 0 for pad.
            - target_labels: Long tensor of shape (batch_size, max_seq_len) with -100 on masked tokens.
    """
    text_embeddings_layer = model.get_input_embeddings()
    bos_token_tensor = torch.tensor([tokenizer.bos_token_id], device=DEVICE)
    bos_token_embed = text_embeddings_layer(bos_token_tensor)  # Shape: (1, hidden_size)

    sequence_embeddings_list = []
    token_labels_list = []
    proj_dtype = next(audio_projector.parameters()).dtype

    for item in batch:
        # 1. Project stacked mel features (num_audio_tokens, STACK * N_MELS) -> (num_audio_tokens, hidden_size)
        audio_features = item["audio"].to(DEVICE, proj_dtype)
        projected_audio_embeds = audio_projector(audio_features).to(DTYPE)

        # 2. Embed prompt and (if training) target text
        text_token_ids = prompt_token_ids + (item["target"] if is_train else [])
        text_token_embeds = text_embeddings_layer(torch.tensor(text_token_ids, device=DEVICE))

        # 3. Concatenate: [BOS] (1 token) + [Audio Tokens] + [Text Tokens]
        full_sequence_embeds = torch.cat([bos_token_embed, projected_audio_embeds, text_token_embeds], dim=0)
        sequence_embeddings_list.append(full_sequence_embeds)

        # 4. Construct labels: mask [BOS], all audio tokens, and prompt tokens with -100
        if is_train:
            masked_prefix_length = 1 + projected_audio_embeds.shape[0] + len(prompt_token_ids)
            utterance_labels = ([-100] * masked_prefix_length) + item["target"]
            token_labels_list.append(utterance_labels)

    # Maximum sequence length in this batch
    max_sequence_len = max(seq.shape[0] for seq in sequence_embeddings_list)
    batch_size = len(sequence_embeddings_list)
    hidden_dimension = sequence_embeddings_list[0].shape[1]

    pad_token_embed = text_embeddings_layer(torch.tensor([tokenizer.pad_token_id], device=DEVICE))

    # Initialize batch tensors
    padded_inputs_embeds = torch.zeros(batch_size, max_sequence_len, hidden_dimension, device=DEVICE, dtype=DTYPE)
    attention_mask = torch.zeros(batch_size, max_sequence_len, device=DEVICE, dtype=torch.long)
    target_labels = torch.full((batch_size, max_sequence_len), -100, device=DEVICE, dtype=torch.long)

    for batch_idx, seq_embed in enumerate(sequence_embeddings_list):
        seq_len = seq_embed.shape[0]
        # Right-padding for training; Left-padding for batched causal generation
        position_slice = slice(0, seq_len) if is_train else slice(max_sequence_len - seq_len, max_sequence_len)

        # Fill with pad token embeddings, then place valid sequence embeddings
        padded_inputs_embeds[batch_idx] = pad_token_embed.expand(max_sequence_len, -1)
        padded_inputs_embeds[batch_idx, position_slice] = seq_embed
        attention_mask[batch_idx, position_slice] = 1

        if is_train:
            target_labels[batch_idx, :seq_len] = torch.tensor(token_labels_list[batch_idx], device=DEVICE)

    return padded_inputs_embeds, attention_mask, target_labels


class ConvAudioProjector(nn.Module):
    """
    Mel-LLM / NeMo-style strided 1D conv subsampler + 2-layer GELU MLP over continuous mel frames.
    Unpacks SpeechDataset's (num_tokens, STACK * N_MELS) back to (num_tokens * STACK, N_MELS) so kernel_size=3
    windows cross patch boundaries, downsamples by STACK in log2(STACK) stride-2 stages, and outputs
    (num_tokens, hidden_size). Implemented via unfold + Linear (rocBLAS GEMM) to avoid MIOpen dynamic-length stalls.
    """

    def __init__(self, n_mels, stack, hidden_size):
        super().__init__()
        n_stages = int(round(math.log2(stack)))
        if 2 ** n_stages != stack or n_stages < 1:
            raise ValueError(f"PROJECTOR='conv' requires STACK to be a power of 2 >= 2, got {stack}")
        self.n_mels = n_mels
        self.convs = nn.ModuleList()
        in_channels = n_mels
        for stage in range(n_stages):
            out_channels = hidden_size // 2 if stage == 0 and n_stages > 1 else hidden_size
            self.convs.append(nn.Linear(3 * in_channels, out_channels))
            in_channels = out_channels
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, stacked_audio):
        x = stacked_audio.reshape(-1, self.n_mels)  # (T * STACK, N_MELS)
        for conv_linear in self.convs:
            windows = F.pad(x, (0, 0, 1, 1)).unfold(0, 3, 2).reshape(x.shape[0] // 2, -1)
            x = self.act(conv_linear(windows))
        return self.out(self.act(self.fc1(self.norm(x))))


def build_audio_projector(hidden_dimension, embedding_std):
    """Build PROJECTOR ('linear', 'mlp', or 'conv') and initialize its final linear layer per PROJ_INIT."""
    patch_dim = STACK * N_MELS
    if PROJECTOR == "linear":
        projector = nn.Linear(patch_dim, hidden_dimension)
        out_linear = projector
    elif PROJECTOR == "mlp":
        out_linear = nn.Linear(hidden_dimension, hidden_dimension)
        projector = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, hidden_dimension), nn.GELU(), out_linear)
    elif PROJECTOR == "conv":
        projector = ConvAudioProjector(N_MELS, STACK, hidden_dimension)
        out_linear = projector.out
    else:
        raise ValueError(f"unknown PROJECTOR {PROJECTOR!r} (expected one of {PROJECTORS})")

    if PROJ_INIT == "match_emb":
        for module in projector.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)) and module is not out_linear:
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")  # keeps GELU activations ~ unit variance
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(out_linear.weight, std=embedding_std / math.sqrt(out_linear.in_features))
        nn.init.zeros_(out_linear.bias)
    elif PROJ_INIT != "default":
        raise ValueError(f"unknown PROJ_INIT {PROJ_INIT!r} (expected 'match_emb' or 'default')")
    return projector.to(DEVICE, trainable_dtype())


def load_model_and_projector(saved_run_dir=None, trainable=False):
    """
    Load base language model, tokenizer, and audio projector.
    If saved_run_dir is provided, loads trained LoRA weights and projector state dict from disk (trainable=True keeps
    the loaded LoRA weights trainable, for INIT_FROM). Otherwise, initializes a fresh LoRA adapter and randomly
    initialized projector.

    Returns:
        tuple: (tokenizer, model, audio_projector)
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    hidden_dimension = model.config.hidden_size
    embedding_std = model.get_input_embeddings().weight.float().std().item()
    audio_projector = build_audio_projector(hidden_dimension, embedding_std)
    # PEFT upcasts LoRA weights to fp32 by default; in "bf16" mode keep them in the base model's DTYPE
    upcast_lora = trainable_dtype() == torch.float32

    if saved_run_dir is None:
        lora_configuration = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS if LORA_TARGETS == "all-linear" else LORA_TARGETS.split(","),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_configuration, autocast_adapter_dtype=upcast_lora)
    else:
        lora_weights_path = os.path.join(saved_run_dir, "lora")
        projector_weights_path = os.path.join(saved_run_dir, "proj.pt")
        model = PeftModel.from_pretrained(model, lora_weights_path, is_trainable=trainable,
                                          autocast_adapter_dtype=upcast_lora)
        audio_projector.load_state_dict(torch.load(projector_weights_path, map_location=DEVICE))

    return tokenizer, model, audio_projector


def describe_lora_placement(model):
    """One line saying where LoRA sits in the LLM: which module names, in how many of the decoder layers."""
    layer_ids_by_name = {}
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            layer = re.search(r"\.layers\.(\d+)\.", name)
            layer_ids_by_name.setdefault(name.rsplit(".", 1)[-1], set()).add(int(layer.group(1)) if layer else -1)
    n_layers = model.config.num_hidden_layers
    parts = [f"{name} ({len(ids)}/{n_layers} layers)" for name, ids in sorted(layer_ids_by_name.items())]
    return (f"lora on LLM ({LORA_TARGETS!r}, r{LORA_R} a{LORA_ALPHA} dropout {LORA_DROPOUT}): " + ", ".join(parts)
            + " | LLM base weights frozen, audio projector fully trained")


# ---------------------------------------------------------------- eval
def normalize_text_for_wer(text):
    """
    Standardize text for Word Error Rate (WER) computation:
    lowercase, retain only letters/digits/apostrophes/spaces, and collapse multiple spaces.
    """
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def decode_token_budget(audio_seconds):
    """Max new tokens for one utterance: MAX_TOKENS_PER_SEC * sec + MAX_TOKENS_PAD, never above MAX_NEW_TOKENS."""
    if MAX_TOKENS_PER_SEC <= 0:
        return MAX_NEW_TOKENS
    return min(MAX_NEW_TOKENS, math.ceil(MAX_TOKENS_PER_SEC * audio_seconds) + MAX_TOKENS_PAD)


def generate_transcripts(model, tokenizer, inputs_embeds, attention_mask, audio_seconds):
    """
    Greedy-decode a left-padded batch (train.py eval and benchmark.py share this). The batch runs up to the largest
    per-utterance budget (decode_token_budget), then each hypothesis is cut to its own budget, so a looping
    utterance cannot emit more than ~MAX_TOKENS_PER_SEC tokens per second of audio.

    Returns:
        list[str]: decoded hypotheses (special tokens skipped), one per batch row.
    """
    budgets = [decode_token_budget(sec) for sec in audio_seconds]
    extra_kwargs = {"no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE} if NO_REPEAT_NGRAM_SIZE > 0 else {}
    generated_token_ids = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max(budgets),
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        **extra_kwargs,
    )
    return [tokenizer.decode(tokens[:budget], skip_special_tokens=True)
            for tokens, budget in zip(generated_token_ids, budgets)]


@torch.no_grad()
def evaluate_dataset(dataset_items, mvn_stats, tokenizer, model, audio_projector, output_hypotheses_path=None,
                     progress_desc="eval", audio_ablation="none"):
    """
    Evaluate Word Error Rate (WER) and audio token throughput on a set of utterances using greedy decoding.

    Args:
        dataset_items (list[dict]): Utterances to evaluate.
        mvn_stats (dict): Normalization stats.
        tokenizer: Pretrained tokenizer.
        model (nn.Module): Language model.
        audio_projector (nn.Linear): Audio projector.
        output_hypotheses_path (str, optional): If specified, write {id, ref, hyp} JSONL to this file.
        progress_desc (str): Label shown on the tqdm progress bar.
        audio_ablation (str): "none" | "shuffle" (each utt is decoded from its length-sorted batch neighbour's audio,
            scored against its own reference) | "zeros" (all-zero mel patches). A model that uses the audio gets a
            much higher WER under either ablation; one that ignores it scores about the same.

    Returns:
        dict: Evaluation metrics containing 'wer', 'sub', 'del', 'ins' (rates), 'n_utt', 'n_words',
              'audio_tokens_per_sec', 'audio_tokens', 'eval_sec', 'rtfx' and 'examples' (a few ref/hyp pairs).
    """
    model.eval()
    eval_start_time = time.time()
    prompt_token_ids = tokenizer(PROMPT, add_special_tokens=False).input_ids
    speech_dataset = SpeechDataset(dataset_items, mvn_stats, tokenizer)

    # Sort evaluation utterances by audio length to minimize left-padding
    sorted_indices = sorted(range(len(dataset_items)), key=lambda idx: dataset_items[idx]["sec"])
    evaluation_batches = [
        sorted_indices[i : i + EVAL_BATCH_SIZE]
        for i in range(0, len(sorted_indices), EVAL_BATCH_SIZE)
    ]

    reference_transcripts = []
    hypothesis_transcripts = []
    output_rows = []
    total_audio_tokens = 0

    progress_bar = make_progress_bar(total=len(dataset_items), desc=progress_desc, unit="utt", leave=False)
    for batch in create_dataloader(speech_dataset, evaluation_batches):
        if audio_ablation == "shuffle":
            batch = [dict(item, audio=batch[(i + 1) % len(batch)]["audio"]) for i, item in enumerate(batch)]
        elif audio_ablation == "zeros":
            batch = [dict(item, audio=torch.zeros_like(item["audio"])) for item in batch]
        elif audio_ablation != "none":
            raise ValueError(f"unknown audio ablation {audio_ablation!r} (expected 'none', 'shuffle' or 'zeros')")
        with compute_context():
            inputs_embeds, attention_mask, _ = build_multimodal_inputs(
                batch,
                model=model,
                audio_projector=audio_projector,
                tokenizer=tokenizer,
                prompt_token_ids=prompt_token_ids,
                is_train=False,
            )

            decoded_texts = generate_transcripts(
                model, tokenizer, inputs_embeds, attention_mask,
                audio_seconds=[item["audio"].shape[0] * STACK * HOP / SR for item in batch],
            )

        for item, decoded_text in zip(batch, decoded_texts):
            normalized_hyp = normalize_text_for_wer(decoded_text)
            normalized_ref = normalize_text_for_wer(item["text"])

            reference_transcripts.append(normalized_ref)
            hypothesis_transcripts.append(normalized_hyp)
            output_rows.append({"id": item["id"], "ref": normalized_ref, "hyp": normalized_hyp})
            total_audio_tokens += item["audio"].shape[0]

        progress_bar.update(len(batch))
    progress_bar.close()

    total_audio_seconds = sum(item["sec"] for item in dataset_items)
    word_alignment = jiwer.process_words(reference_transcripts, hypothesis_transcripts)
    word_error_rate = word_alignment.wer
    n_reference_words = word_alignment.hits + word_alignment.substitutions + word_alignment.deletions
    eval_seconds = time.time() - eval_start_time

    if output_hypotheses_path:
        with open(output_hypotheses_path, "w") as f:
            for row in output_rows:
                f.write(json.dumps(row) + "\n")

    model.train()
    return {
        "wer": word_error_rate,
        "sub": word_alignment.substitutions / max(1, n_reference_words),
        "del": word_alignment.deletions / max(1, n_reference_words),
        "ins": word_alignment.insertions / max(1, n_reference_words),
        "n_utt": len(dataset_items),
        "n_words": n_reference_words,
        "audio_tokens_per_sec": total_audio_tokens / total_audio_seconds,
        "audio_tokens": total_audio_tokens,
        "eval_sec": eval_seconds,
        "rtfx": total_audio_seconds / eval_seconds,  # seconds of audio decoded per wall-clock second
        "examples": random.Random(SEED).sample(output_rows, min(LOG_N_EXAMPLES, len(output_rows))),
    }


def format_eval_metrics(metrics):
    """One-line human-readable summary of evaluate_dataset() output."""
    return (
        f"WER {metrics['wer'] * 100:.2f} (sub {metrics['sub'] * 100:.2f} del {metrics['del'] * 100:.2f} "
        f"ins {metrics['ins'] * 100:.2f}) | {metrics['n_utt']} utts {metrics['n_words']} words | "
        f"{metrics['eval_sec']:.0f}s RTFx {metrics['rtfx']:.0f} | audio tok/s {metrics['audio_tokens_per_sec']:.2f}"
    )


def run_test_evaluation(run_dir, mvn_stats, tokenizer, model, audio_projector, log_fn=None):
    """
    Run evaluation across all configured test splits (e.g. test-clean, test-other) and save results.
    log_fn (callable, optional) receives one summary line per split; defaults to tqdm.write.
    """
    log_fn = log_fn or (lambda message: tqdm.write(message, file=sys.stdout))
    test_results = {}
    for split_name in TEST_SPLITS:
        hypotheses_filepath = os.path.join(run_dir, f"hyp_{split_name.replace('/', '-')}.jsonl")
        split_items = load_split(split_name)
        test_results[split_name] = evaluate_dataset(
            split_items,
            mvn_stats=mvn_stats,
            tokenizer=tokenizer,
            model=model,
            audio_projector=audio_projector,
            output_hypotheses_path=hypotheses_filepath,
            progress_desc=f"test {split_name}",
        )
        test_results[split_name].pop("examples")
        tqdm_message = f"[test] {split_name} {format_eval_metrics(test_results[split_name])}"
        log_fn(tqdm_message)

    json.dump(test_results, open(os.path.join(run_dir, "test_results.json"), "w"), indent=2)
    return test_results


# ---------------------------------------------------------------- train
def main():
    args, config = parse_args()
    if config["INIT_FROM"] and not args.eval_only:
        config = resolve_init_from(config)
    apply_config(config)
    seed_everything(SEED)

    # Fast path: evaluation only mode from existing checkpoint (config comes from its config.json + any flags)
    if args.eval_only:
        tokenizer, model, audio_projector = load_model_and_projector(args.eval_only)
        mvn_stats = torch.load(mvn_cache_path())
        test_results = run_test_evaluation(args.eval_only, mvn_stats, tokenizer, model, audio_projector)
        RunTracker(args.eval_only).update(
            test_wer={split_name: round(metrics["wer"] * 100, 2) for split_name, metrics in test_results.items()}
        )
        return

    # Set up a fresh run directory: config + git state are written before anything can fail
    run_name = RUN_NAME or f"{MODEL.split('/')[-1]}_stack{STACK}{override_name_suffix(config)}"
    run_dir = create_run_dir(run_name + (f"_smoke{args.max_steps}" if args.max_steps else ""))
    run_id = run_dir.name
    runs_lib.update_latest_link(run_dir)
    git_info = get_git_info()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    config = get_config_dictionary()
    config["meta"] = {
        "run_id": run_id,
        "command": shlex.join([sys.executable.split("/")[-1], *sys.argv]),
        "notes": args.notes,
        "base_config": args.config,
        "overrides": {key: val for key, val in config.items() if val != DEFAULT_CONFIG[key]},
        "max_steps": args.max_steps,
        "started_at": started_at,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
    }
    json.dump(config, open(run_dir / "config.json", "w"), indent=2)
    if git_info["diff"]:
        (run_dir / "git_diff.patch").write_text(git_info["diff"])

    tracker = RunTracker(run_dir, run_id=run_id, status="running", started_at=started_at, notes=args.notes)
    log_file = open(run_dir / "log.txt", "a")
    runs_lib.write_index()

    def log_and_print(*print_args):
        message = " ".join(str(arg) for arg in print_args)
        tqdm.write(message, file=sys.stdout)  # prints above any active progress bar
        sys.stdout.flush()
        log_file.write(message + "\n")
        log_file.flush()

    dirty_marker = " (dirty, see git_diff.patch)" if git_info["dirty"] else ""
    log_and_print(f"===== run {run_id} | {started_at} | git {git_info['commit'][:10]}{dirty_marker} =====")
    log_and_print(f"command: {config['meta']['command']}")
    if args.notes:
        log_and_print(f"notes: {args.notes}")
    log_and_print(f"overrides vs defaults: {json.dumps(config['meta']['overrides'])}")

    run_start_time = time.time()
    try:
        trained = train(args, run_dir, tracker, log_and_print)
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        tracker.update(status=status, error=repr(error), finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                       duration_sec=round(time.time() - run_start_time))
        log_and_print(f"run {status}: {error!r}")
        runs_lib.refresh_run_views(run_dir, log_and_print)
        raise
    tracker.update(status="done", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   duration_sec=round(time.time() - run_start_time))
    runs_lib.refresh_run_views(run_dir, log_and_print)

    # Open ASR benchmark with the model still in memory. The run is already "done": a benchmark error is recorded
    # in summary.json (rerun with `python benchmark.py mel:runs/<id> --sets ...`), it does not fail the run.
    if BENCHMARK_SETS.lower() != "none" and not args.max_steps:
        import benchmark as benchmark_lib
        sets = benchmark_lib.resolve_sets(BENCHMARK_SETS)
        log_and_print(f"benchmark: {','.join(sets)} -> results/mel_{run_id}/")
        try:
            bench_model = benchmark_lib.MelLLM(run_dir, EVAL_BATCH_SIZE, loaded=trained)
            bench_summary = benchmark_lib.benchmark(bench_model, f"mel_{run_id}", sets, log=log_and_print)
            log_and_print("BENCHMARK", json.dumps({s: bench_summary[s]["wer"] for s in sets}))
        except Exception as error:  # noqa: BLE001
            log_and_print(f"benchmark failed: {error!r}")
            benchmark_lib.update_run_summary(run_dir, benchmark_status="failed", benchmark_error=repr(error))
        runs_lib.refresh_run_views(run_dir, log_and_print)
    log_file.close()


INIT_MUST_MATCH = ("MODEL", "STACK", "N_MELS", "SR", "N_FFT", "HOP", "MVN_N", "PROJECTOR", "LORA_R", "LORA_ALPHA",
                   "LORA_TARGETS", "PROMPT")  # tuple, so it is not picked up as a config constant


def resolve_init_from(config):
    """
    INIT_FROM: check the init run's architecture / feature keys match this config, and (unless MVN_SPLIT is set)
    reuse the init run's MVN stats so the warm-started weights see identically normalised mel features.
    """
    init_config = json.load(open(Path(config["INIT_FROM"], "config.json")))
    init_config.setdefault("PROJECTOR", "linear")
    mismatched = {key: (init_config.get(key), config[key]) for key in INIT_MUST_MATCH
                  if key in init_config and init_config[key] != config[key]}
    if mismatched:
        raise SystemExit(f"INIT_FROM {config['INIT_FROM']}: config differs from the init run as (init, now): "
                         f"{mismatched}; pass e.g. --config {config['INIT_FROM']}/config.json")
    if not config["MVN_SPLIT"]:
        config = dict(config, MVN_SPLIT=init_config.get("MVN_SPLIT") or unweighted_split(init_config["TRAIN_SPLIT"]))
    return config


def override_name_suffix(config):
    """
    Short run-name suffix from the hyperparameters that differ from the defaults, so runs/ is readable at a glance:
    --epochs 10 --batch_size 20 -> "_epochs10_bs20". MODEL/STACK are already in the name; bookkeeping keys are skipped.
    """
    short = {"BATCH_SIZE": "bs", "GRAD_ACCUM": "ga", "LORA_R": "r", "LORA_ALPHA": "a", "LORA_DROPOUT": "ldrop",
             "LORA_TARGETS": "lora", "WARMUP_STEPS": "warmup", "WEIGHT_DECAY": "wd", "GRAD_CLIP": "clip",
             "LR_SCHEDULE": "sched", "MIN_LR_RATIO": "minlr", "OPTIMIZER": "opt", "MUON_ADJUST_LR": "muonlr",
             "TRAIN_SPLIT": "train", "PRECISION": "prec", "MAX_TRAIN_SEC": "maxsec", "MIN_TRAIN_SEC": "minsec",
             "TRAIN_N": "n", "VAL_FROM_TRAIN": "valtrain", "PROJECTOR": "proj", "PROJ_INIT": "pinit",
             "PROJ_LR_MULT": "plr", "PROJ_ONLY_STEPS": "ponly", "TRAIN_STEPS": "steps"}
    skip = runs_lib.IGNORED_KEYS | {"MODEL", "STACK", "EVAL_EVERY", "VAL_N", "TEST_SPLITS", "DATA", "MAX_NEW_TOKENS",
                                    "MAX_TOKENS_PER_SEC", "MAX_TOKENS_PAD", "NO_REPEAT_NGRAM_SIZE", "INIT_FROM", "MVN_SPLIT",
                                    "EVAL_AUDIO_ABLATIONS"}
    parts = []
    for key, val in config.items():
        if key in skip or key not in DEFAULT_CONFIG or val == DEFAULT_CONFIG[key]:
            continue
        val = runs_lib.fmt(val).replace("_proj", "").replace(",", "+")
        parts.append(short.get(key, key.lower().replace("_", "")) + re.sub(r"[^A-Za-z0-9.+-]", "", val))
    suffix = "_".join(parts)[:60]
    return f"_{suffix}" if suffix else ""


def train(args, run_dir, tracker, log_and_print):
    """Full training loop + final test eval. Writes metrics to tracker and checkpoints to run_dir."""
    log_and_print(f"model {MODEL} | device {torch.cuda.get_device_name(0)} | dtype {DTYPE} | precision {PRECISION}")

    # Load dataset splits
    train_items = [item for item in load_split(TRAIN_SPLIT) if MIN_TRAIN_SEC <= item["sec"] <= MAX_TRAIN_SEC]
    mvn_stats = compute_or_load_mvn_stats(train_items)  # always from the full split, so TRAIN_N runs share the cache
    if TRAIN_N:
        train_items = sorted(random.Random(SEED).sample(train_items, TRAIN_N), key=lambda it: it["id"])
    if VAL_FROM_TRAIN:
        validation_items = train_items[:VAL_N]
    else:
        validation_items = sorted(load_split(VAL_SPLIT), key=lambda it: it["id"])[:VAL_N]
    val_name = f"{TRAIN_SPLIT}(train)" if VAL_FROM_TRAIN else VAL_SPLIT
    val_ablations = [name.strip() for name in EVAL_AUDIO_ABLATIONS.split(",") if name.strip()]

    total_train_hours = sum(item["sec"] for item in train_items) / 3600.0
    log_and_print(f"train utts {len(train_items)} ({total_train_hours:.1f} h), val utts {len(validation_items)}")
    log_and_print(
        f"batch {BATCH_SIZE} x accum {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} | {OPTIMIZER} lr {LR:.1e} {LR_SCHEDULE} warmup {WARMUP_STEPS} "
        f"| epochs {EPOCHS}{f' (train_steps {TRAIN_STEPS})' if TRAIN_STEPS else ''} | stack {STACK} ({SR / HOP / STACK:.2f} Hz audio tokens, {STACK * N_MELS}-d patches) | lora r{LORA_R} a{LORA_ALPHA} on LLM {LORA_TARGETS}"
    )

    # Initialize model and linear projector
    tokenizer, model, audio_projector = load_model_and_projector(INIT_FROM or None, trainable=True)
    if INIT_FROM:
        log_and_print(f"init: LoRA + projector weights from {INIT_FROM} (fresh optimizer and LR schedule)")
    if GRAD_CKPT:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # Muon only takes 2D matrices and nothing falls back to AdamW, so the (1D) projector bias stays at its init
    if OPTIMIZER == "muon":
        if PROJECTOR != "linear":
            raise ValueError(f"OPTIMIZER='muon' requires PROJECTOR='linear' (2D weights only), got {PROJECTOR!r}")
        audio_projector.bias.requires_grad_(False)

    # Collect trainable parameters (LoRA adapter weights + linear projector)
    trainable_lora_params = [param for param in model.parameters() if param.requires_grad]
    trainable_proj_params = [p for p in audio_projector.parameters() if p.requires_grad]
    all_trainable_params = trainable_lora_params + trainable_proj_params

    lora_param_count_m = sum(param.numel() for param in trainable_lora_params) / 1e6
    proj_param_count_m = sum(param.numel() for param in audio_projector.parameters()) / 1e6
    log_and_print(f"trainable: lora {lora_param_count_m:.1f}M, proj {proj_param_count_m:.2f}M "
                  f"({sorted({str(p.dtype).removeprefix('torch.') for p in all_trainable_params})})")
    log_and_print(describe_lora_placement(model))

    # Two param groups so the projector can get its own lr (PROJ_LR_MULT) and LoRA can start late (PROJ_ONLY_STEPS)
    param_groups = [{"params": trainable_lora_params}, {"params": trainable_proj_params}]
    if OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    elif OPTIMIZER == "muon":
        optimizer = torch.optim.Muon(param_groups, lr=LR, weight_decay=WEIGHT_DECAY, adjust_lr_fn=MUON_ADJUST_LR)
        log_and_print(f"optimizer: Muon ({MUON_ADJUST_LR}) on all {len(all_trainable_params)} trainable 2D weights, "
                      "no AdamW (projector bias frozen)")
    else:
        raise ValueError(f"unknown OPTIMIZER {OPTIMIZER!r} (expected 'adamw' or 'muon')")

    # Calculate steps and learning rate schedule: linear warmup, then cosine or linear decay
    steps_per_epoch = max(1, math.ceil(len(train_items) / BATCH_SIZE) // GRAD_ACCUM)
    total_optimizer_steps = args.max_steps or TRAIN_STEPS or (steps_per_epoch * EPOCHS)
    num_epochs = math.ceil(total_optimizer_steps / steps_per_epoch)

    def compute_lr_multiplier(step_idx):
        if step_idx < WARMUP_STEPS:
            return (step_idx + 1) / WARMUP_STEPS
        decay_progress = min(1.0, (step_idx - WARMUP_STEPS) / max(1, total_optimizer_steps - WARMUP_STEPS))
        if LR_SCHEDULE == "cosine":
            return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        if LR_SCHEDULE == "linear":
            return 1.0 - decay_progress
        raise ValueError(f"unknown LR_SCHEDULE {LR_SCHEDULE!r} (expected 'cosine' or 'linear')")

    def compute_lora_lr_multiplier(step_idx):
        """0 while only the projector trains, then a fresh linear warmup into the shared schedule."""
        if step_idx < PROJ_ONLY_STEPS:
            return 0.0
        return min(1.0, (step_idx - PROJ_ONLY_STEPS + 1) / max(1, WARMUP_STEPS)) * compute_lr_multiplier(step_idx)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[compute_lora_lr_multiplier if PROJ_ONLY_STEPS else compute_lr_multiplier,
                   lambda step_idx: PROJ_LR_MULT * compute_lr_multiplier(step_idx)],
    )
    log_and_print(f"steps/epoch {steps_per_epoch}, total steps {total_optimizer_steps} | proj {PROJECTOR} "
                  f"(init {PROJ_INIT}, lr x{PROJ_LR_MULT}), LoRA frozen for first {PROJ_ONLY_STEPS} steps")
    if TRAIN_N or VAL_FROM_TRAIN or val_ablations:
        log_and_print(f"diagnostics: train subset {TRAIN_N or 'all'}, val on {val_name}[:{VAL_N}], "
                      f"audio ablations {val_ablations or 'none'}")

    prompt_token_ids = tokenizer(PROMPT, add_special_tokens=False).input_ids
    train_dataset = SpeechDataset(train_items, mvn_stats, tokenizer)

    optimizer_step = 0
    micro_step = 0
    train_start_time = time.time()
    best_val_wer, best_val_step = float("inf"), None

    # Per-logging-window accumulators (reset every LOG_EVERY optimizer steps)
    def reset_window_stats():
        return {
            "start_time": time.time(), "loss_sum": 0.0, "micro_steps": 0, "tokens": 0, "padded_tokens": 0,
            "target_tokens": 0, "utts": 0, "audio_sec": 0.0, "data_wait_sec": 0.0, "grad_norms": [],
            "lora_grad_norms": [], "proj_grad_norms": [],
        }

    window = reset_window_stats()
    step_loss_sum = 0.0  # loss summed over the micro-batches of the current optimizer step (for the bar)

    progress_bar = make_progress_bar(total=total_optimizer_steps, desc="train", unit="step")

    for epoch in range(num_epochs):
        epoch_sampler = create_length_batched_sampler(train_items, BATCH_SIZE, seed=SEED + epoch)
        dataloader = create_dataloader(train_dataset, epoch_sampler)
        progress_bar.set_description(f"train ep {epoch + 1}/{num_epochs}")

        data_wait_start = time.time()
        for batch in dataloader:
            window["data_wait_sec"] += time.time() - data_wait_start

            with compute_context():
                inputs_embeds, attention_mask, target_labels = build_multimodal_inputs(
                    batch,
                    model=model,
                    audio_projector=audio_projector,
                    tokenizer=tokenizer,
                    prompt_token_ids=prompt_token_ids,
                    is_train=True,
                )
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=target_labels)
            loss = outputs.loss
            (loss / GRAD_ACCUM).backward()

            loss_value = loss.item()
            step_loss_sum += loss_value
            window["loss_sum"] += loss_value
            window["micro_steps"] += 1
            window["tokens"] += int(attention_mask.sum())
            window["padded_tokens"] += attention_mask.numel()
            window["target_tokens"] += int((target_labels != -100).sum())
            window["utts"] += len(batch)
            window["audio_sec"] += sum(item["audio"].shape[0] for item in batch) * STACK * HOP / SR
            micro_step += 1

            if micro_step % GRAD_ACCUM != 0:
                data_wait_start = time.time()
                continue

            if optimizer_step < PROJ_ONLY_STEPS:
                for param in trainable_lora_params:  # LoRA frozen: drop its grads so they neither step nor eat the clip budget
                    param.grad = None
            # Clip LoRA and projector independently so neither component starves the other's gradient budget
            lora_grad_norm = (torch.nn.utils.clip_grad_norm_(trainable_lora_params, GRAD_CLIP).item()
                              if trainable_lora_params and optimizer_step >= PROJ_ONLY_STEPS else 0.0)
            proj_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_proj_params, GRAD_CLIP).item()
            grad_norm = math.hypot(lora_grad_norm, proj_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            window["grad_norms"].append(grad_norm)
            window["lora_grad_norms"].append(lora_grad_norm)
            window["proj_grad_norms"].append(proj_grad_norm)

            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{step_loss_sum / GRAD_ACCUM:.3f}",
                gnorm=f"{grad_norm:.2f}",
                lr=f"{current_lr:.1e}",
                best_wer="-" if best_val_step is None else f"{best_val_wer * 100:.2f}",
                refresh=False,
            )
            step_loss_sum = 0.0

            if optimizer_step % LOG_EVERY == 0:
                elapsed_time = time.time() - window["start_time"]
                average_loss = window["loss_sum"] / window["micro_steps"]
                grad_norms = window["grad_norms"]
                total_elapsed = time.time() - train_start_time
                eta_seconds = total_elapsed / optimizer_step * (total_optimizer_steps - optimizer_step)
                epoch_progress = micro_step / len(epoch_sampler)  # fractional epochs completed (all epochs)
                num_clipped = sum(norm > GRAD_CLIP for norm in grad_norms)

                log_and_print(
                    f"ep {epoch_progress:.2f} step {optimizer_step}/{total_optimizer_steps} "
                    f"({optimizer_step / total_optimizer_steps * 100:.1f}%) | "
                    f"loss {average_loss:.4f} ppl {math.exp(min(average_loss, 20)):.2f} | "
                    f"gnorm {grad_norm:.2f} (avg {np.mean(grad_norms):.2f} max {np.max(grad_norms):.2f} "
                    f"clipped {num_clipped}/{len(grad_norms)}) | "
                    f"pre-clip gnorm lora {np.mean(window['lora_grad_norms']):.2f} proj {np.mean(window['proj_grad_norms']):.2f} | "
                    f"lr {current_lr:.2e} proj lr {lr_scheduler.get_last_lr()[1]:.2e} | "
                    f"elapsed {format_duration(total_elapsed)} eta {format_duration(eta_seconds)}"
                )
                log_and_print(
                    f"    perf: {elapsed_time / len(grad_norms):.2f}s/step tok/s {window['tokens'] / elapsed_time:.0f} "
                    f"tgt tok/s {window['target_tokens'] / elapsed_time:.0f} utt/s {window['utts'] / elapsed_time:.1f} "
                    f"audio {window['audio_sec'] / elapsed_time:.0f}x realtime | "
                    f"pad eff {window['tokens'] / max(1, window['padded_tokens']) * 100:.0f}% "
                    f"data wait {window['data_wait_sec'] / elapsed_time * 100:.0f}% | "
                    f"mem {torch.cuda.memory_allocated() / 2**30:.1f}G "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G "
                    f"reserved {torch.cuda.memory_reserved() / 2**30:.1f}G"
                )
                tracker.log_metrics(
                    "train", step=optimizer_step, epoch=round(epoch_progress, 4), loss=average_loss,
                    lr=current_lr, grad_norm=grad_norm, grad_norm_avg=float(np.mean(grad_norms)),
                    grad_norm_max=float(np.max(grad_norms)), clipped=num_clipped,
                    lora_grad_norm_avg=float(np.mean(window["lora_grad_norms"])),
                    proj_grad_norm_avg=float(np.mean(window["proj_grad_norms"])),
                    proj_lr=lr_scheduler.get_last_lr()[1],
                    sec_per_step=elapsed_time / len(grad_norms), tok_per_sec=window["tokens"] / elapsed_time,
                    peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, elapsed_sec=total_elapsed,
                )
                tracker.update(step=optimizer_step, total_steps=total_optimizer_steps, train_loss=average_loss)
                window = reset_window_stats()
                if optimizer_step % PLOT_EVERY == 0:
                    runs_lib.refresh_run_views(run_dir, log_and_print)

            if optimizer_step % EVAL_EVERY == 0 or optimizer_step == total_optimizer_steps:
                val_metrics = evaluate_dataset(
                    validation_items,
                    mvn_stats=mvn_stats,
                    tokenizer=tokenizer,
                    model=model,
                    audio_projector=audio_projector,
                    progress_desc=f"val @{optimizer_step}",
                )
                is_best = val_metrics["wer"] < best_val_wer
                if is_best:
                    best_val_wer, best_val_step = val_metrics["wer"], optimizer_step
                ablation_wers = {}
                for ablation in val_ablations:
                    ablation_wers[f"wer_{ablation}"] = evaluate_dataset(
                        validation_items, mvn_stats=mvn_stats, tokenizer=tokenizer, model=model,
                        audio_projector=audio_projector, progress_desc=f"val {ablation} @{optimizer_step}",
                        audio_ablation=ablation,
                    )["wer"]
                ablation_text = "".join(f" | {key} {val * 100:.2f}" for key, val in ablation_wers.items())
                log_and_print(
                    f"[val] step {optimizer_step} {val_name}[:{VAL_N}] {format_eval_metrics(val_metrics)} | "
                    f"best {best_val_wer * 100:.2f} @ step {best_val_step}{' (new best)' if is_best else ''}"
                    f"{ablation_text}"
                )
                for example in val_metrics["examples"]:
                    log_and_print(f"    [{example['id']}] REF: {example['ref']}")
                    log_and_print(f"    [{example['id']}] HYP: {example['hyp']}")
                tracker.log_metrics(
                    "val", step=optimizer_step,
                    **{key: val for key, val in val_metrics.items() if key != "examples"}, **ablation_wers,
                )
                tracker.update(
                    step=optimizer_step, total_steps=total_optimizer_steps,
                    val_wer=round(val_metrics["wer"] * 100, 2),
                    best_val_wer=round(best_val_wer * 100, 2), best_val_step=best_val_step,
                )
                runs_lib.refresh_run_views(run_dir, log_and_print)
                window = reset_window_stats()  # exclude eval time from training throughput

            if optimizer_step >= total_optimizer_steps:
                break

            data_wait_start = time.time()

        if optimizer_step >= total_optimizer_steps:
            break

    progress_bar.close()
    log_and_print(
        f"training done: {optimizer_step} steps in {format_duration(time.time() - train_start_time)}, "
        f"best val WER {best_val_wer * 100:.2f} @ step {best_val_step}"
    )

    # Save trained LoRA adapter and linear projector
    model.save_pretrained(os.path.join(run_dir, "lora"))
    torch.save(audio_projector.state_dict(), os.path.join(run_dir, "proj.pt"))
    log_and_print("saved", run_dir)

    if args.max_steps:
        return tokenizer, model, audio_projector  # Smoke test run: skip full evaluation on test splits

    test_results = run_test_evaluation(run_dir, mvn_stats, tokenizer, model, audio_projector, log_fn=log_and_print)
    wer_summary = {split_name: round(metrics["wer"] * 100, 2) for split_name, metrics in test_results.items()}
    for split_name, metrics in test_results.items():
        tracker.log_metrics("test", step=optimizer_step, split=split_name, **metrics)
    tracker.update(test_wer=wer_summary)
    log_and_print("TEST", json.dumps(wer_summary))
    return tokenizer, model, audio_projector


if __name__ == "__main__":
    main()
