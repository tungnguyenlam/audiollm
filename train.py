"""
Baseline: encoder-free Mel -> LLM ASR, fixed-rate time-only chunking (Mel-LLM style, arXiv 2606.10231).

    wav -> 80-dim log-mel (10 ms hop) -> MVN (train stats) -> projector (PROJECTOR, e.g. stack STACK frames + Linear)
        -> [BOS][audio tokens]["Transcription:"][text][EOS] -> LFM2.5-230M-Base + LoRA

This is the command-line entry point; the pieces live in the audiollm/ package:
    config.py      UPPERCASE defaults (every one is a lower-case flag: LR -> --lr, GRAD_CKPT -> --no-grad_ckpt)
    data.py        split specs -> utterances, SpeechDataset, length / token-budget batching
    features.py    waveform -> MVN-normalised log-mel frames (LogMelFeaturizer)
    projectors/    frames -> LLM embeddings, one class per PROJECTOR (linear, mlp, conv)
    backbone.py    MODEL + tokenizer + LoRA
    model.py       SpeechLLM = backbone + projector: input layout, loss, decoding, model summary
    trainer.py     the training loop; evaluate.py: WER evaluation
`python train.py --help` lists every flag.

Each training run gets its own directory runs/<YYYYmmdd-HHMMSS>_<run_name>/ with:
    config.json      resolved config (UPPERCASE keys) + "meta" (command, git commit/dirty, overrides, notes, ...)
    git_diff.patch   uncommitted changes at launch time (only if the tree was dirty)
    metrics.jsonl    time series: one JSON row per train log window / val pass / test split
    summary.json     status (running/done/failed/interrupted), best val WER, test WER, parameter counts, duration
    model_summary.txt  LLM shape, parameter counts per component, projector layers
    log.txt, lora/, proj.pt, mvn.pt (feature normalisation stats), hyp_<test split label>.jsonl (full test hyps), test_results.json
    val_hyps/<dev split label>/step_<N>.jsonl   {id, ref, hyp} of every val utt at each val pass
    checkpoint/      resumable state while training (CHECKPOINT_EVERY), removed at the end
Split labels name the dataset (runs.split_label): dev-clean_librispeech, validation_ami_ihm, validation_fleurs_en.
runs/latest -> the newest run, runs/latest_model -> the newest run with saved weights.
Compare runs with `python runs.py` (see runs.py --help).

    python train.py                                   # train with the defaults
    python train.py --run_name lr3e-4 --lr 3e-4 --notes "higher lr"
    python train.py --model LiquidAI/LFM2.5-350M --stack 8       # run dir name defaults to <model>_stack<STACK>
    python train.py --config runs/<id>/config.json --stack 4   # start from a previous run's config, tweak one knob
    python train.py --max_steps 50                    # smoke test
    python train.py --eval_only runs/<id>             # re-run test eval with that run's saved config
    python train.py --checkpoint_every 500 --hub_repo user/audiollm-runs   # resumable + mirrored to the HF Hub
    python train.py --resume runs/<id>                # continue an interrupted run from runs/<id>/checkpoint/
"""
import json
import re
import shlex
import shutil
import socket
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

import runs as runs_lib
from audiollm import config as C
from audiollm.data import unweighted_split
from audiollm.evaluate import run_test_evaluation
from audiollm.features import LogMelFeaturizer, load_run_mvn_stats
from audiollm.hub import HubSync
from audiollm.model import SpeechLLM
from audiollm.tracking import RunTracker, create_run_dir, get_git_info
from audiollm.trainer import train
from audiollm.utils import seed_everything


def main():
    args, config = C.parse_args()
    if (config["INIT_FROM"] or config["INIT_PROJ_FROM"]) and not (args.eval_only or args.resume):  # a resumed config is already resolved
        config = resolve_init_from(config)
    C.apply_config(config)
    seed_everything(C.SEED)

    # Fast path: evaluation only mode from existing checkpoint (config comes from its config.json + any flags)
    if args.eval_only:
        model = SpeechLLM.load(args.eval_only)
        featurizer = LogMelFeaturizer(load_run_mvn_stats(args.eval_only))
        test_results = run_test_evaluation(args.eval_only, featurizer, model)
        RunTracker(args.eval_only).update(
            test_wer={split_name: round(metrics["wer"] * 100, 2) for split_name, metrics in test_results.items()}
        )
        return

    # Set up the run directory (fresh, or the --resume one): config + git state are written before anything can fail
    git_info = get_git_info()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    command = shlex.join([sys.executable.split("/")[-1], *sys.argv])
    if args.resume:
        run_dir = Path(args.resume)
        meta = json.load(open(run_dir / "config.json"))["meta"]
        if args.max_steps is None:
            args.max_steps = meta.get("max_steps")  # the step budget (and so the LR schedule) must not change
        meta.setdefault("resumes", []).append({"command": command, "at": started_at, "host": socket.gethostname(),
                                               "git_commit": git_info["commit"], "git_dirty": git_info["dirty"]})
        args.notes = meta["notes"]
    else:
        run_name = C.RUN_NAME or f"{C.MODEL.split('/')[-1]}_stack{C.STACK}{override_name_suffix(config)}"
        run_dir = create_run_dir(run_name + (f"_smoke{args.max_steps}" if args.max_steps else ""))
        meta = {
            "run_id": run_dir.name,
            "command": command,
            "notes": args.notes,
            "base_config": args.config,
            "overrides": {key: val for key, val in config.items() if val != C.DEFAULT_CONFIG[key]},
            "max_steps": args.max_steps,
            "started_at": started_at,
            "host": socket.gethostname(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
        }
    run_id = run_dir.name
    runs_lib.update_latest_link(run_dir)

    config = dict(C.get_config_dictionary(), meta=meta)
    json.dump(config, open(run_dir / "config.json", "w"), indent=2)
    if git_info["diff"]:
        (run_dir / ("git_diff_resume.patch" if args.resume else "git_diff.patch")).write_text(git_info["diff"])

    tracker = RunTracker(run_dir, run_id=run_id, status="running", notes=args.notes,
                         **({"resumed_at": started_at} if args.resume else {"started_at": started_at}))
    log_file = open(run_dir / "log.txt", "a")
    runs_lib.write_index()

    def log_and_print(*print_args):
        message = " ".join(str(arg) for arg in print_args)
        tqdm.write(message, file=sys.stdout)  # prints above any active progress bar
        sys.stdout.flush()
        log_file.write(message + "\n")
        log_file.flush()

    dirty_marker = f" (dirty, see {'git_diff_resume.patch' if args.resume else 'git_diff.patch'})" if git_info["dirty"] else ""
    log_and_print(f"===== {'resume' if args.resume else 'run'} {run_id} | {started_at} | "
                  f"git {git_info['commit'][:10]}{dirty_marker} =====")
    log_and_print(f"command: {command}")
    if args.notes:
        log_and_print(f"notes: {args.notes}")
    log_and_print(f"overrides vs defaults: {json.dumps(config['meta']['overrides'])}")

    hub = HubSync.from_config(run_dir, log_and_print)
    run_start_time = time.time()
    try:
        trained_model, _ = train(args, run_dir, tracker, log_and_print, hub)
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        tracker.update(status=status, error=repr(error), finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                       duration_sec=round(time.time() - run_start_time))
        log_and_print(f"run {status}: {error!r}")
        runs_lib.refresh_run_views(run_dir, log_and_print)
        if hub:
            hub.push(f"{status} at step {tracker.summary.get('step')}", final=False)
            hub.wait()
        raise
    tracker.update(status="done", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                   duration_sec=round(time.time() - run_start_time))
    if hub:
        hub.wait()  # an upload may still be reading checkpoint/
    shutil.rmtree(run_dir / "checkpoint", ignore_errors=True)  # final weights are in lora/ + proj.pt
    runs_lib.refresh_run_views(run_dir, log_and_print)

    # Open ASR benchmark with the model still in memory. The run is already "done": a benchmark error is recorded
    # in summary.json (rerun with `python benchmark.py mel:runs/<id> --sets ...`), it does not fail the run.
    if C.BENCHMARK_SETS.lower() != "none" and not args.max_steps:
        import benchmark as benchmark_lib
        sets = benchmark_lib.resolve_sets(C.BENCHMARK_SETS)
        log_and_print(f"benchmark: {','.join(sets)} -> results/mel_{run_id}/")
        try:
            bench_model = benchmark_lib.MelLLM(run_dir, C.EVAL_BATCH_SIZE, loaded=trained_model)
            bench_summary = benchmark_lib.benchmark(bench_model, f"mel_{run_id}", sets, log=log_and_print)
            log_and_print("BENCHMARK", json.dumps({s: bench_summary[s]["wer"] for s in sets}))
        except Exception as error:  # noqa: BLE001
            log_and_print(f"benchmark failed: {error!r}")
            benchmark_lib.update_run_summary(run_dir, benchmark_status="failed", benchmark_error=repr(error))
        runs_lib.refresh_run_views(run_dir, log_and_print)
    if hub:
        hub.push("final", final=True)
        hub.wait()
        log_and_print(f"hub: final upload done -> {C.HUB_REPO}/{run_id}")
    log_file.close()


INIT_MUST_MATCH = ("MODEL", "STACK", "N_MELS", "SR", "N_FFT", "HOP", "MVN_N", "PROJECTOR", "LORA_R", "LORA_ALPHA",
                   "LORA_TARGETS", "PROMPT")
INIT_PROJ_MUST_MATCH = ("STACK", "N_MELS", "SR", "N_FFT", "HOP", "MVN_N", "PROJECTOR")


def resolve_init_from(config):
    """
    INIT_FROM / INIT_PROJ_FROM: check the init run's architecture / feature keys match this config, and (unless
    MVN_SPLIT is set) reuse the init run's MVN stats so the warm-started weights see identically normalised mel features.
    """
    if config["INIT_FROM"] and config["INIT_PROJ_FROM"]:
        raise SystemExit("INIT_FROM already loads the projector; pass only one of INIT_FROM / INIT_PROJ_FROM")
    key, must_match = ("INIT_FROM", INIT_MUST_MATCH) if config["INIT_FROM"] else ("INIT_PROJ_FROM", INIT_PROJ_MUST_MATCH)
    init_config = C.with_legacy_defaults(json.load(open(Path(config[key], "config.json"))))
    mismatched = {name: (init_config.get(name), config[name]) for name in must_match
                  if name in init_config and init_config[name] != config[name]}
    if mismatched:
        raise SystemExit(f"{key} {config[key]}: config differs from the init run as (init, now): "
                         f"{mismatched}; pass e.g. --config {config[key]}/config.json")
    if not config["MVN_SPLIT"]:
        config = dict(config, MVN_SPLIT=init_config.get("MVN_SPLIT") or unweighted_split(init_config["TRAIN_SPLIT"]))
    return config


def override_name_suffix(config):
    """
    Short run-name suffix from the hyperparameters that differ from the defaults, so runs/ is readable at a glance:
    --epochs 10 --batch_size 20 -> "_epochs10_bs20". MODEL/STACK are already in the name; bookkeeping keys are skipped.
    """
    short = {"BATCH_SIZE": "bs", "BATCH_TOKENS": "btok", "GRAD_ACCUM": "ga", "LORA_R": "r", "LORA_ALPHA": "a",
             "LORA_DROPOUT": "ldrop", "LORA_TARGETS": "lora", "WARMUP_STEPS": "warmup", "WEIGHT_DECAY": "wd",
             "GRAD_CLIP": "clip", "LR_SCHEDULE": "sched", "MIN_LR_RATIO": "minlr", "OPTIMIZER": "opt",
             "MUON_ADJUST_LR": "muonlr", "TRAIN_SPLIT": "train", "PRECISION": "prec", "MAX_TRAIN_SEC": "maxsec",
             "MIN_TRAIN_SEC": "minsec", "TRAIN_N": "n", "VAL_FROM_TRAIN": "valtrain", "PROJECTOR": "proj",
             "PROJ_INIT": "pinit", "PROJ_LR_MULT": "plr", "PROJ_ONLY_STEPS": "ponly", "TRAIN_STEPS": "steps",
             "SPECAUG_FREQ_MASKS": "fm", "SPECAUG_FREQ_WIDTH": "fw", "SPECAUG_TIME_MASKS": "tm",
             "SPECAUG_TIME_WIDTH": "tw", "SPECAUG_MAX_TIME_RATIO": "tr"}
    skip = runs_lib.IGNORED_KEYS | {"MODEL", "STACK", "EVAL_EVERY", "VAL_N", "TEST_SPLITS", "DATA", "MAX_NEW_TOKENS",
                                    "MAX_TOKENS_PER_SEC", "MAX_TOKENS_PAD", "NO_REPEAT_NGRAM_SIZE", "INIT_FROM", "INIT_PROJ_FROM", "MVN_SPLIT",
                                    "EVAL_AUDIO_ABLATIONS"}
    parts = []
    for key, val in config.items():
        if key in skip or key not in C.DEFAULT_CONFIG or val == C.DEFAULT_CONFIG[key]:
            continue
        val = runs_lib.fmt(val).replace("_proj", "").replace(",", "+")
        parts.append(short.get(key, key.lower().replace("_", "")) + re.sub(r"[^A-Za-z0-9.+-]", "", val))
    suffix = "_".join(parts)[:60]
    return f"_{suffix}" if suffix else ""


def __getattr__(name):
    """
    `import train as T; T.STACK, T.apply_config(...)` (notebooks): config constants and package functions are still
    reachable through this module, always reading the live config.
    """
    import audiollm.backbone, audiollm.data, audiollm.evaluate, audiollm.features, audiollm.model, audiollm.projectors, audiollm.utils
    for module in (C, audiollm.features, audiollm.data, audiollm.model, audiollm.backbone, audiollm.evaluate,
                   audiollm.projectors, audiollm.utils):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module 'train' has no attribute {name!r}")


if __name__ == "__main__":
    main()
