"""
Run configuration: one flat set of UPPERCASE constants, shared by every module in the package.

The constants below are the defaults. train.py turns each one into a lower-case CLI flag (LR -> --lr), and a run's
resolved values are saved as runs/<id>/config.json with the same keys. Modules read them at call time through
`from audiollm import config as C` -> `C.STACK`, so apply_config() (a CLI flag, a saved config.json, a notebook
override) takes effect everywhere at once.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

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
VAL_SPLIT = "dev-clean,ami_ihm/validation,fleurs_en/validation"  # comma list of dev splits, each scored separately (one per training corpus); the FIRST one picks best_val_wer, "dev-clean" = pre-2026-10-28 runs
VAL_N = 200                  # utts per dev split: LibriSpeech = first VAL_N by id (comparable to older runs), corpora = SEED-fixed sample of utts in [MIN_TRAIN_SEC, MAX_TRAIN_SEC]
TEST_SPLITS = ["test-clean", "test-other"]  # Test dataset splits
MAX_TRAIN_SEC = 30.0         # drop the few very long utterances
TRAIN_N = 0                  # 0 = all train utts; N > 0 = a fixed random subset of N (seeded), e.g. 10 for an overfit test
VAL_FROM_TRAIN = False       # validate on the training utterances themselves (first VAL_N of the train set) instead of VAL_SPLIT: overfit / memorisation test
EVAL_AUDIO_ABLATIONS = ""    # comma list of extra decodes per val pass to test audio dependence: "shuffle" (each utt gets its batch neighbour's audio), "zeros" (all-zero mel); logged as wer_<ablation>

# Audio feature extraction configuration
# ------------------------------------------------------------------------------------------------
# SR: Sampling rate of input audio waveforms in Hz.
#   - 16 kHz audio captures frequencies up to 8 kHz (Nyquist), covering all fundamental frequencies
#     and critical formant/fricative acoustic cues for human speech. LibriSpeech/AMI/FLEURS standard.
SR = 16000

# N_FFT: Short-Time Fourier Transform (STFT) window length in samples.
#   - 400 samples = 25.0 ms window (N_FFT / SR). Fits the quasi-stationary assumption of human speech (vocal tract shape is steady over ~20-30 ms), balancing frequency resolution against temporal blurring.
#   - Yields N_FFT // 2 + 1 = 201 linear frequency bins spanning 0 to SR/2 (8 kHz) (~39.8 Hz/bin).
N_FFT = 400

# HOP: STFT hop length (stride between successive analysis windows) in samples.
#   - 160 samples = 10.0 ms frame shift (HOP / SR), giving an STFT frame rate of SR / HOP = 100 Hz.
#   - Overlap = (1 - HOP / N_FFT) = 60%, providing smooth temporal transitions without framing artifacts.
HOP = 160

# N_MELS: Number of triangular bandpass filters in the Mel-scale filterbank.
#   - Projects 201 linear STFT bins to 80 perceptually warped frequency channels (denser in low frequencies,
#     log-spaced in high frequencies, matching human cochlear frequency selectivity).
#   - Standard in modern ASR (Whisper, Conformer, Mel-LLM). (Higher, e.g. 128 at N_FFT=400, risks dead filter bins).
N_MELS = 80

# MVN_N: Number of training utterances randomly sampled to compute global Mean-Variance Normalization stats.
#   - Samples MVN_N utterances across the training set to derive per-channel mean and std vectors (shape: 80,).
#   - Applied as (mel - mean) / std to zero-center features and scale variance to 1, stabilizing initial gradients.
#   - Cached to disk based on split, MVN_N, SR, N_FFT, HOP, and N_MELS.
MVN_N = 2000

# STACK: Number of consecutive 10 ms mel frames grouped into a single audio patch/token for the LLM.
#   - Patch dimension: STACK * N_MELS (e.g. 16 * 80 = 1280-d input vector to Linear/MLP projector; or downsampled
#     by log2(STACK) = 4 strided Conv1d stages in Conv projector).
#   - Effective audio token rate: (SR / HOP) / STACK = 100 / 16 = 6.25 tokens/s (1 token per 160 ms).
#   - Trade-off: STACK=8 yields 12.5 Hz (paper default, finer phonetic resolution, 2x longer LLM sequence length),
#     whereas STACK=16 halves sequence length and compute while yielding 1280-d patches (>= 230M LLM hidden dim 1024).
STACK = 16
# ------------------------------------------------------------------------------------------------

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
PROJECTOR = "linear"         # audio adapter, a class in audiollm/projectors/: "linear" (nn.Linear(STACK*N_MELS -> d)) | "mlp" (LayerNorm + 2-layer GELU MLP on stacked patches) | "conv" (Mel-LLM style strided Conv1d x log2(STACK) on continuous mel + LayerNorm + 2-layer GELU MLP)
PROJ_INIT = "match_emb"      # projector output init: "match_emb" (N(0, emb_std/sqrt(fan_in)), zero bias -> audio embeddings ~ the LLM's text-embedding scale) | "default" (PyTorch nn.Linear init, ~20x larger than text embeddings; pre-2026-10-27 runs)
PROJ_LR_MULT = 1.0           # projector lr = LR * PROJ_LR_MULT (LoRA uses LR)
PROJ_ONLY_STEPS = 0          # optimizer steps with LoRA frozen (only the projector trains) before LoRA joins with its own WARMUP_STEPS warmup
OPTIMIZER = "adamw"          # "adamw" (all trainable params) | "muon" (torch.optim.Muon on every trainable 2D linear weight = LoRA A/B + projector weight; no AdamW anywhere, so the projector bias is frozen at init; the LLM's depthwise conv is frozen either way)
MUON_ADJUST_LR = "match_rms_adamw"  # Muon per-matrix lr scaling: "match_rms_adamw" (update RMS ~ AdamW's, so LR/WEIGHT_DECAY transfer from AdamW) | "original" (Keller Jordan's sqrt(max(1, A/B)); wants LR ~ 0.02)
WARMUP_STEPS = 200           # Linear learning rate warmup steps
EPOCHS = 2                   # Total number of training epochs
TRAIN_STEPS = 0              # >0: train exactly this many optimizer steps (LR schedule spans them; epochs repeat as needed) instead of EPOCHS, e.g. to match budgets across data mixes
BATCH_SIZE = 8               # Training batch size per device
BATCH_TOKENS = 0             # >0: build train batches by padded sequence length (BATCH_TOKENS >= batch size x longest [BOS][audio][prompt][text][EOS] in it; audio tokens from the projector, text tokens counted once at startup) instead of a fixed BATCH_SIZE; 0 = fixed BATCH_SIZE (pre-2026-10-28 runs)
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
# Keys whose value changed after older runs were saved: a config.json without the key gets the old behaviour back.
LEGACY_DEFAULTS = {
    "PRECISION": "fp32",     # runs from before PRECISION existed trained fp32 LoRA + projector
    "PROJ_INIT": "default",  # PyTorch's nn.Linear init before PROJ_INIT existed
    "MIN_TRAIN_SEC": 0.0,    # no minimum length before MIN_TRAIN_SEC existed
    "PROJECTOR": "linear",
}


def get_config_dictionary():
    """All UPPERCASE config constants as a JSON-serialisable dict (the format of runs/<id>/config.json)."""
    config = {
        key: val
        for key, val in globals().items()
        if key.isupper() and isinstance(val, (int, float, str, list, bool))
    }
    config["DTYPE"] = str(DTYPE).removeprefix("torch.")  # JSON-serialisable; converted back in apply_config
    return config


DEFAULT_CONFIG = get_config_dictionary()  # snapshot of the constants above, before any override


def with_legacy_defaults(saved_config):
    """A saved config.json with LEGACY_DEFAULTS filled in for keys it predates."""
    return {**LEGACY_DEFAULTS, **saved_config}


def apply_config(overrides):
    """
    Override the config constants (from CLI flags or a saved config.json). Non-uppercase keys (e.g. "meta") are
    ignored.

    Returns:
        list[str]: uppercase keys that are not known config constants (ignored, e.g. from an older config.json).
    """
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
    from audiollm.projectors import PROJECTORS

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

    hparam_group = parser.add_argument_group("config (defaults = UPPERCASE constants in audiollm/config.py)")
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
            kwargs["choices"] = list(PROJECTORS)
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
        base_config = with_legacy_defaults(json.load(open(base_config_path)))
        unknown_keys = [key for key in base_config if key.isupper() and key not in DEFAULT_CONFIG]
        if unknown_keys:
            print(f"warning: ignoring unknown keys from {base_config_path}: {unknown_keys}", file=sys.stderr)
        parser.set_defaults(**{key: val for key, val in base_config.items() if key in DEFAULT_CONFIG})
    args = parser.parse_args(argv)
    config = {key: getattr(args, key) for key in DEFAULT_CONFIG}
    return args, config
