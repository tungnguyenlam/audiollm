# Vietnamese Stage 1 readiness — 2026-10-31

Both backbone pipelines pass training, validation, checkpoint writing, model reload, and all five Vietnamese
benchmark smoke tests. Full A/B training was launched on 2026-10-31: LFM run
`20261031-135333_vi500h_lfm_projE2_matched19k`, followed automatically by Qwen after successful training and benchmark.
The persistent systemd user service is `vi-stage1-ab-20261031`; frozen configs, logs and queue state are under
`results/vi_ab_20261031/`. No datasets were downloaded or deleted.
After both arms finish, the attached completion service writes `comparison.md` and `comparison.json` in that
directory, recording test scores, elapsed runtime and the backbone favored by primary GigaSpeech2-vi WER.

## Data

Audited all 16 manifests / 303,787 utterances: every audio header/path, 16 kHz mono format, duration agreement,
and normalized transcript. All passed; no empty normalized references. Fully decoded 32 samples per split.
An exhaustive SHA-256 scan found 94 duplicate groups within splits and no identical encoded audio across splits.
GigaSpeech2 dev/test have 148 shared IDs; their decoded audio differs. Treat IDs as split-local.
These checks establish structural integrity and exact-file separation; they do not establish transcript accuracy
or rule out near-duplicate recordings. A few identical transcript strings across official splits were retained.

The planned source mix actually has 708.29 usable hours. PhoAudiobook's first 100 shards contain 421.38 h,
not the earlier estimated 186 h. The prepared view uses seed 0 to select PhoAudiobook by cumulative duration:

| Corpus | Hours | Utterances |
|---|---:|---:|
| PhoAudiobook selection | 213.09 | 65,696 |
| VLSP2020 | 98.51 | 55,483 |
| LSVSC | 80.60 | 45,357 |
| ViMD | 77.65 | 14,578 |
| FPT FOSD | 30.15 | 25,863 |
| Total | 500.00 | 206,977 |

Manifests refer to existing audio. Evaluation directories link to official held-out splits. The view is under
`results/vi_readiness_20261031/mix500h/`; source/selection hashes are in `mix500h.json` beside it.
The adjacent `build_mix.py` reproduces the selection using the completed audio-overlap audit.

## Matched experiment

- Configs: `configs/vi_stage1_lfm.json` and `configs/vi_stage1_qwen.json`.
- Both initialize only the projector from `20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k`, with fresh
  LoRA and identical E2 MVN tensors. English LoRA is not transferred into either arm.
- Same selected utterances, seed, duration-based batching, microbatch 4 × accumulation 8, 19,200 steps,
  AMP, LoRA rank 64, LR 2e-4, projector LR multiplier 2.5, warmup 500, cosine floor 0.1.
- Fixed utterance batches match audio exposure. Equal token-budget batches would expose Qwen to more audio.
- Four dev sets, 200 utterances each; evaluation/checkpoint every 1,000 steps and final evaluation at 19,200.
- Greedy decode, no n-gram suppression, hard cap 1,024; LFM 32 tokens/s and Qwen 12 tokens/s, both +10 slack.
  These cover all audited dev/test targets and 1,000 training references per corpus. Initial 20 tokens/s for LFM
  would truncate 13 dev/test references. Token ceilings are safeguards, not matched compute budgets.
- Compare final saved weights on GigaSpeech2-vi test first, with FLEURS, ViMD, LSVSC and PhoAudiobook as supporting
  sets. Record runtime with WER; early smoke WER cannot select a backbone.

## Smoke results and runtime

| Arm | Training smoke | Steady seconds/step | Peak training GiB |
|---|---|---:|---:|
| LFM2.5-230M | `20261031-132803_vi_lfm_readiness_smoke30` | 1.96 | 3.20 |
| Qwen3-0.6B-Base | `20261031-133023_vi_qwen_readiness_smoke30` | 3.23 | 5.63 |

Each ran 30 optimizer steps on the same 640-utterance sample, validated four sets (4 utterances each), and saved
weights. Losses and both projector/LoRA gradients remained finite and nonzero; checkpoint steps 10/20 succeeded.
An initial LFM microbatch-16 probe had recovered allocation failures, so prepared runs use microbatch 4.
Saved-model benchmark CLI then decoded 4 utterances on each of all five test sets at the final decode caps.
Smoke WER is poor (frequent repetitions), as expected for this short training test; it establishes execution only.

At the measured rate, 19,200 training steps take approximately **10.5 h LFM + 17.2 h Qwen**. Budget roughly
**2–3 days sequentially including validation and full benchmarks**. Direct extrapolation of smoke decode rates
gives about 56 h combined. This is a planning estimate: tiny validation batches include worker startup, LFM's
training smoke used its earlier 20 tokens/s decode ceiling, benchmark smoke used batch 2 vs full batch 4, the
smoke mix weights differ from the full mix, and trained models may stop decoding much earlier than these probes.
Startup, thermal variation and repeated checkpoint/plot work add uncertainty.

## Commands and artifacts

The persistent queue is already running. These commands reproduce its arms manually; avoid launching duplicates:

```bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
.venv/bin/python train.py --config configs/vi_stage1_lfm.json
.venv/bin/python train.py --config configs/vi_stage1_qwen.json
```

Each completed training run automatically benchmarks all five available Vietnamese test sets. Offline mode uses
cached model weights and existing data. The generated manifest view must remain present for these configs.
Resumable checkpoints are written every 1,000 steps; use `train.py --resume runs/<run_id>` after interruption.

Audit scripts, logs, hashes, token coverage, smoke verification and runtime calculations are in
`results/vi_readiness_20261031/`. Benchmark smoke transcripts are in `results/vi_readiness_20261031_lfm/` and
`results/vi_readiness_20261031_qwen/`. Runtime artifacts stay gitignored.

## English E2 benchmark

Verified all nine D/E2 benchmark scores and utterance counts by recomputing WER from saved normalized hypotheses,
following benchmark.py's empty-reference exclusion. Seven-set Open ASR mean improves 52.35 → 41.36%; nine-set
mean improves 59.24 → 49.15%. All nine sets improve. E2 changed both data and training duration, so this does not
isolate the data contribution. GigaSpeech/SPGISpeech are in-domain; other held-out test splits remain held out.
Per-set scores are recorded in WORKLOG.md and `results/vi_readiness_20261031/english_benchmark_verified.json`;
the adjacent `english_benchmark.html` visualizes them.
