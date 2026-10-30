# Worklog

Oldest entry first; append new entries at the bottom with `cat >> WORKLOG.md`.
Never read this file whole. See "Worklog" in `AGENTS.md`.
Earlier months: `docs/worklog_archive/YYYY-MM.md`.

## 2026-10-27 - Diagnosis: every Mel-LLM run ignores the audio

### Findings
- All finished runs (baseline_fixed12.5hz 350M, 230M default, 230M muon lr1e-5, 350M muon) sit at 112-125% WER.
  `hyp_test-clean.jsonl` shows near-constant, audio-independent outputs ("and the other was a little more
  careful" for "a story", "venice", "marie sighed", ...). The LLM falls back to a LibriSpeech text prior;
  train loss ~3.2 is consistent with unconditional text modelling.
- Scale mismatch: LFM2.5-230M text embeddings have per-dim std 0.028 (row norm 0.89); default-init
  `nn.Linear(1280->1024)` on MVN'd mel gives per-dim std 0.58 (row norm 18.5), ~20x off-distribution.
- Muon run used lr 1e-5 with bf16 trainable weights: most updates round away in bf16. LoRA dropout 0.5 also high.

### Decisions
- Before more full runs: fix projector init scale, add per-component LR / projector-only warmup,
  per-component grad norms, a TRAIN_N subset + eval-on-train overfit mode, and audio-shuffle/zero eval ablations.
- Gate: a 10-100 utterance overfit run must reach near-0 train WER and the shuffled-audio WER must be high.
  If it fails after the fixes, move to architecture changes (conv/MLP adapter, then frozen Whisper-tiny encoder).

## 2026-10-27 - Diagnostic knobs in train.py

### Decisions
- New constants (all default to old behaviour except PROJ_INIT): `TRAIN_N` (seeded train subset; MVN stats still
  from the full split), `VAL_FROM_TRAIN` (validate on the train utts), `EVAL_AUDIO_ABLATIONS` ("shuffle": each utt
  decoded from its batch neighbour's audio; "zeros": zero mel; logged as `wer_<name>` per val pass),
  `PROJ_LR_MULT`, `PROJ_ONLY_STEPS` (LoRA grads dropped, then its own warmup), `PROJ_INIT`.
- `PROJ_INIT="match_emb"` is the new default: weight N(0, emb_std/sqrt(fan_in)), zero bias, so audio embeddings
  start at the text-embedding scale. Old configs loaded via `--config`/`--eval_only` fall back to "default".
- Optimizer now has two param groups (LoRA, projector); train log/metrics add pre-clip `lora_grad_norm_avg`,
  `proj_grad_norm_avg`, `proj_lr`. List flags take `nargs="*"`, so a bare `--test_splits` skips test eval.

## 2026-10-27 - Overfit test (10 utts) passes: pipeline is not broken

- Run `20261027-141342_overfit10_matchemb` (killed at step ~115 once answered): TRAIN_N 10, val on same 10,
  bs 10, fp32, lr 1e-4, proj lr x10, LoRA dropout 0, match_emb.
- Loss 4.6 -> 0.004 by step 30; **WER 0.00 at step 50**; `wer_shuffle` 106.6, `wer_zeros` 201.8/140.7.
- Rules out label/masking/decoding/generate() bugs and shows the audio path carries utterance identity.
  Rules in: the failure is generalisation (acoustic -> text mapping on 100 h), not plumbing.
- Note: step-1..10 projector pre-clip grad norm ~150 vs LoRA ~13; projector dominates early updates.
- Next: 1500-step full-data run with the fixes (amp, match_emb, proj lr x10, 300 projector-only steps) +
  shuffle ablation on dev-clean, compared to `20261027-005138` val (232 @500, 135 @1000, 146 @1500).

## 2026-10-27 - Wire multi-corpus ASR training (544 h) and local benchmark test sets

### Decisions
- Added `unpack_parquet_asr_corpora.py` (wired into `setup_dataset.sh`) unpacking `ami_ihm/{train,validation}` and
  `fleurs_en/{train,validation}` (`raw_transcription`) into `data/asr_corpora/<corpus>/<split>/{audio/*.flac,manifest.jsonl}`.
- Updated `train.py` (`load_split`, `normalize_training_text`, `MIN_TRAIN_SEC=1.0`, cached `manifest.jsonl` per
  LibriSpeech split): default `TRAIN_SPLIT="train-clean-100+train-clean-360+ami_ihm/train+fleurs_en/train"`
  yields **204,286 utts / 544.0 h** in `[1, 30] s` (drops 39.4k sub-1s AMI backchannels; strips `<...>`/`[...]` tags).
  Kept `speech_commands` (isolated 1s words) and audio-QA/event corpora out of ASR training.
- Updated `benchmark.py` (`LOCAL_SETS`): `BENCHMARK_SETS="all"` now runs the 7 Open ASR sets (which already cover
  LibriSpeech clean/other + AMI test) plus local `fleurs_en` (647 utts, 1.77 h) and `edacc` (9,228 utts, 16.38 h after
  dropping 61 `IGNORE_TIME_SEGMENT_IN_SCORING` rows), keeping `OpenASR avg` unchanged (`--sets openasr` for 7 only).
- Updated `runs.py` to exclude `TRAIN_N` / `VAL_FROM_TRAIN` diagnostic runs from best-model selection and plot
  component grad norms + `wer_shuffle`/`wer_zeros`.

## 2026-10-27 - 1500-step scale-matched + projector-warmup run still ignores audio

- Run `20261027-142314_fix_matchemb_plr10_ponly300_smoke1500` (100 h, amp, `match_emb`, `proj_lr_mult=10`,
  `proj_only_steps=300`, `eval_audio_ablations=shuffle`):
  - Steps 1-300 (LoRA frozen): loss barely moved (4.71 -> 4.62), `proj_grad_norm` decayed 1.16 -> 0.30; val @250
    WER 400.8 (`wer_shuffle` 413.9).
  - Steps 300-1500 (LoRA unfrozen): `lora_grad_norm` jumped to ~2.2 vs `proj_grad_norm` ~0.18 (12x ratio, clipping
    20/20 steps), loss plateaued at ~4.18. Val WER: 167.7 @500, 171.7 @750, **119.15 @1000 (`wer_shuffle` 116.38)**,
    125.6 @1250 (`136.3`), 119.81 @1500 (`125.67`).
- Rules out projector init scale, bf16 rounding, and a 300-step linear-projector warmup as sufficient fixes on
  `STACK=16` single-`nn.Linear`. Rules in architectural/alignment bottleneck (no shift-equivariant nonlinearity in
  160 ms patches + LoRA text-prior shortcut starving the projector under shared grad clip).

## 2026-10-27 - Add strided-Conv / MLP projectors and independent component grad clipping

### Decisions
- Added `PROJECTOR` (`"linear"` default | `"mlp"` | `"conv"`): `"conv"` (`ConvAudioProjector`) reshapes stacked
  patches back to continuous `(1, N_MELS, T*STACK)`, applies `log2(STACK)` stride-2 `Conv1d(k=3,p=1)+GELU` stages +
  `LayerNorm` + 2-layer GELU MLP (6.95M params @ stack 8, 10.09M @ stack 16; Kaiming-ReLU internal init + `match_emb`
  output linear -> row norm ~0.83 matching LFM2.5's 0.89). Zero changes to `SpeechDataset` or `benchmark.py` interface.
- Replaced combined `clip_grad_norm_(all_trainable_params, GRAD_CLIP)` with independent clipping on LoRA and projector
  so LoRA's ~2.2 text-prior grad norm no longer scales down projector updates.

## 2026-10-27 - Launch 544.1 h Mel-LLM strided-conv + 12.5 Hz full run

- `nn.Conv1d` on ROCm MIOpen stalled at >=8 s/step compiling kernels per dynamic utterance length; replaced the
  strided conv stages inside `ConvAudioProjector` with mathematically identical `F.pad + unfold(0,3,2) + nn.Linear`
  (pure rocBLAS GEMM: 100 dynamic lengths fwd+bwd in 1.36 s total, zero MIOpen compilations).
- Run `20261027-150147_mix544h_conv_stack8` (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`): 204,286 utts / 544.1 h
  (`train-clean-100 + train-clean-360 + ami_ihm/train + fleurs_en/train`, 1-30 s), `LFM2.5-230M-Base`, `STACK=8`
  (12.5 Hz), `PROJECTOR="conv"` (6.95M params, `match_emb`), `PRECISION="amp"`, `LR=2e-4` (`proj_lr=5e-4`),
  `warmup=500`, `bs=16 x ga=2`, `epochs=2` (12,768 steps), `eval_every=500` (`shuffle` ablation), 9-set benchmark.

## 2026-10-27 - Result: 544.1 h + strided-conv (12.5 Hz) breaks the 120% wall (real phonetic ASR)

- Run `20261027-150147_mix544h_conv_stack8` finished (12,768 steps / 2h 45m + 9-set benchmark):
  - **Phase transition at step ~3,000:** steps 0-2,500 sat on the text-prior plateau (`loss ~4.1`, `proj_grad_norm ~0.26`,
    `wer == wer_shuffle`). At step ~3,000 monotonic alignment ignited (`proj_grad_norm` jumped 5x to `1.3-2.3`),
    `train_loss` fell from `4.12 -> 1.47`, and `dev-clean` WER dropped steadily to **72.03% @ step 11,500** vs
    **`wer_shuffle` 108.96%** (37-point audio-dependence gap), plateauing only because cosine LR hit ~0 at 2 epochs.
  - **LibriSpeech test (`train.py`):** **test-clean 70.98%**, **test-other 85.38%** (down from 117.6-125.4% across all
    prior runs). Short/medium utterances show clear phonetic decoding (`"poor alice" -> "poor alice"`,
    `"there just in front" -> "there just in front"`, `"verse two" -> "first too"`).
  - **9-set benchmark (`results/mel_20261027-150147_mix544h_conv_stack8/`):** `librispeech_clean` **71.41**,
    `librispeech_other` **85.69**, `ami` **92.79** (down from 135.8-166.4), `gigaspeech` **105.73**, `voxpopuli` 118.85,
    `spgispeech` 141.29, `earnings22` 202.63; extra: `fleurs_en` 146.26, `edacc` 145.10 (repetition loops on >30 s clips).
- Rules **in**: strided conv + MLP + independent grad clip unlocks genuine encoder-free mel->phoneme alignment on 230M;
  2 epochs (only ~8k post-ignition steps before LR->0) strongly undertrains it. Next: longer training (6-10 epochs or
  warm-restart from `runs/20261027-150147_mix544h_conv_stack8`) and/or isolate 100 h vs 544 h with `PROJECTOR="conv"`.

## 2026-10-27 - Review of mix544h_conv_stack8 benchmark: out-of-domain failure is not just loops

- Re-scored `results/mel_20261027-150147_mix544h_conv_stack8/*.jsonl`, setting aside hyps with >=5x repeated 1-3-gram loops:
  non-looped WER `librispeech_clean` 71.3, `ami` 77.2, `edacc` 100.0, `fleurs_en` 109.8, `spgispeech` 112.5,
  `earnings22` 118.5. Looped share of hyps: earnings22 19%, edacc 12%, spgi 11%, fleurs 9%, ami 2%, ls_clean 0%.
- Rules **out** "high OOD WER is only loops on >30 s clips": even loop-free hyps on unfamiliar sets are >100%.
  On FLEURS (`fleurs_en/train` is in the mix) hyps fall back to an AMI-style chat prior ("okay so you know the the ...").
  Alignment is real only on LibriSpeech/AMI-like acoustics so far.
- `mean_wer` across sets with >100% WER (insertion-driven) can't be compared with older runs (e.g. spgi 115 -> 141 is not
  a real regression). Compare per set using sub/del/ins and the `wer - wer_shuffle` gap instead.
- `train.py` has no `--resume_from`. A warm restart would need new code (load `lora/` + `proj.pt`).

## 2026-10-27 - Notebook: front-end sweep (SR / N_FFT / HOP / N_MELS / STACK), section 4b
- `explore_pipeline.ipynb`: new `FRONTEND_SWEEP` + `FRONTEND_PROBE_N` in the PARAMETERS cell; new section 4b compares
  configs on 200 train utts without training (MVN_N untouched; per-config MVN from the probe set, cache not written).
- Proxies per config: window/bin resolution, frame/token rate, dead mel filters, log-spectral distance (LSD) vs a fine
  16 kHz reference, patch dim vs H, PCA dims (PC95/PC99), audio/text ratio, seq len / LLM cost; plus side-by-side
  log-mels with token boundaries and Griffin-Lim resynthesis to listen to; prints the `train.py` command per config.
- First readout (LSD dB, lower keeps more): current 5.62 | N_MELS=128 5.11 (but 4 dead filters at N_FFT=400) |
  N_FFT=512 5.29 | HOP=320/N_FFT=640/STACK=8 5.72 | N_MELS=40/STACK=32 6.43 (audio/text 1.02: too few tokens) |
  SR=8000 9.96. STACK=8 = same info, 4.05 audio/text, 1.65x LLM cost. Proxies only, not WER.
- Caveat: `extract_log_mel_spectrogram` asserts file SR == SR, so SR != 16000 is not trainable without a resample step.

## 2026-10-27 - Decode A/B: per-utterance token cap + no-repeat-4-gram become defaults

- Added `MAX_TOKENS_PER_SEC=6.0` + `MAX_TOKENS_PAD=10` (per-utt cap; no train transcript exceeds `6*sec+10`),
  `NO_REPEAT_NGRAM_SIZE`, `MAX_NEW_TOKENS` 200 -> 256; shared `generate_transcripts()` for train eval + benchmark.
  `benchmark.py` now reports sub/del/ins and takes `--max_tokens_per_sec/--no_repeat_ngram_size/--max_new_tokens`.
- A/B on `mix544h_conv_stack8`, 400 evenly spaced utts/set (`results/decab_*`), WER:

  | cfg | ls_c | ls_o | vox | ami | earn22 | giga | spgi | fleurs | edacc | mean9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | old (200 tok) | 71.5 | 88.5 | 114.0 | 96.0 | 207.6 | 107.6 | 145.2 | 134.7 | 130.5 | 121.7 |
  | cap6 | 71.5 | 87.5 | 109.5 | 78.4 | 140.1 | 96.5 | 119.0 | 117.2 | 109.9 | 103.3 |
  | cap6+nr3 | 71.8 | 87.5 | 104.5 | 76.0 | 112.7 | 92.7 | 109.5 | 106.7 | 96.2 | 95.3 |
  | cap6+nr4 | 71.6 | 87.3 | 104.7 | 74.7 | 113.7 | 92.7 | 109.8 | 107.4 | 96.2 | 95.3 |
  | cap6+nr6 | 71.5 | 87.4 | 105.0 | 74.8 | 115.5 | 92.8 | 111.4 | 107.8 | 97.1 | 95.9 |
- Decision: default `NO_REPEAT_NGRAM_SIZE=4` (ties nr3 on mean, better on LS/AMI, safer for real 3-gram repeats).
  Decoding is inference-only so old run configs pick up the new defaults; old full results moved to
  `results/mel_20261027-150147_mix544h_conv_stack8_maxnew200/`. LibriSpeech unchanged: the fix removes loop
  insertions only; OOD sets stay ~93-114% (real acoustic generalisation gap).

## 2026-10-27 - Queue: re-benchmark + warm-start continuation + rebalanced mix + 100 h ablation

- Added `INIT_FROM` (warm-start LoRA + projector, fresh optimizer/schedule; checks arch keys, inherits the init run's
  MVN via new `MVN_SPLIT`), `"<split>*K"` weights in `TRAIN_SPLIT`, `TRAIN_STEPS` (fixed budget). Smoke: warm start
  resumes at loss 1.54 (parent ended ~1.47), 15.5M trainable LoRA params.
- `results/queue_20261027.sh` (log `results/queue_20261027.log`), sequential:
  1. full 9-set re-benchmark of `mix544h_conv_stack8` with cap6+nr4 decoding;
  2. `mix544h_conv_stack8_cont19k`: init from it, same mix, 19.2k steps, lr 1e-4 (proj x2.5), warmup 300, cosine -> 0.1;
  3. `mix654h_rebal_conv_stack8_cont19k`: same budget, `ami_ihm/train*2+fleurs_en/train*6` (654 h/epoch);
  4. `ls100h_conv_stack8_12768`: cold start, train-clean-100 only, same 12,768-step schedule as the 544 h run.
- 2 vs 3 isolates data balance at equal steps; 4 vs `mix544h_conv_stack8` isolates data volume for the conv projector.

## 2026-10-27 - Dataset viewer: "Front-end lab" tab
- New tab in `dataset_viewer` (backend `dataset_viewer/frontend_lab.py`, `/api/lab`, UI `static/lab.js`): runs the
  selected sample through resample -> PCM/mu-law quantisation -> STFT -> mel -> log -> feature precision
  (fp32/bf16/fp16/int8) -> per-clip MVN -> stack, with presets matching the notebook's section 4b sweep.
- Shows waveform (+ quantisation error), STFT power, mel power (linear), log-mel, normalised log-mel with token
  boundaries, a clickable patch inspector, the mel filterbank and long-term spectra; plays original / processed /
  error / Griffin-Lim from |STFT| / Griffin-Lim from log-mel. Scores LSD vs a 16 kHz reference like the notebook.
- Spot check (one LibriSpeech clip, 8 s): 8-bit PCM qSNR 23-31 dB (LSD +6-8 dB); 8-bit mu-law LSD 2.2 dB;
  4-bit PCM ~20 dB; bf16/int8 log-mel max error ~0.03 (log-mel LSD unchanged).

## 2026-10-27 - Full re-benchmark of mix544h_conv_stack8 with cap6+nr4 decoding

- `results/mel_20261027-150147_mix544h_conv_stack8/` (old decode kept in `..._maxnew200/`), WER (sub/del/ins):
  ls_clean 71.29 (52.1/7.3/11.9), ls_other 85.30, voxpopuli 105.96, ami **77.72** (was 92.79), earnings22
  **114.66** (was 202.63), gigaspeech 92.25, spgispeech 110.75 (was 141.29); extra fleurs_en 107.84, edacc 97.73.
  Open ASR mean 116.91 -> **93.99**. Benchmark wall time ~43 min (RTFx 190-325, was 79-280).
- Substitutions are still 53-82% everywhere except LibriSpeech/AMI: the remaining OOD gap is acoustic, not decoding.
- Queue step 2 (`20261027-230948_mix544h_conv_stack8_cont19k`) started cleanly: val @1000 70.79 (shuffle 106.79),
  loss 1.44, ETA ~3.7 h.

## 2026-10-27 - Dataset viewer rebuilt (server + UI)
- Why: changing settings in the Front-end lab looked like it did nothing. The play buttons kept the previous run's
  audio (`L.playing` was never reset after a re-run). Also, `python -m dataset_viewer` with the system python died on
  a missing `uvicorn`.
- `server.py` rewritten: one `/api/bootstrap`; lab results are cached by a hash of (dataset, row, settings), so files
  never expire while shown and repeat settings return in ~10 ms (vs ~0.6 s); at most 2 lab runs at a time; clearer
  error messages. `__main__` now explains the venv when imports fail. Data/DSP modules, the SQLite schema and the
  manifest are unchanged.
- `static/` replaced by `index.html`, `style.css` and ES modules in `static/js/` (core, main, review, report, lab).
  In the lab, the settings object is the only state, runs are serialised so the latest settings win, and changed
  values are marked against `train.py` with derived numbers that update instantly. Light and dark themes, phone width.
- Verified with headless Chromium: settings clicks, typing, rapid changes, presets, reset and audio after a change
  (fetched for the new settings key); review next/flag/clear; report tiles and histograms; no console errors.

## 2026-10-28 - Notebooks: explore_pipeline reorganised, new projector_anatomy.ipynb
- `explore_pipeline.ipynb` updated to the current setup: loads the `TRAIN_SPLIT` mix via `T.load_split` (it still
  assumed train-clean-100 only), builds any `PROJECTOR` via `T.build_audio_projector` (was hard-coded `nn.Linear`, so
  conv runs failed to load), and matches runs on `PROJECTOR` too (missing key in old configs = default). Front-end sweep
  moved before the projector (now §4), projector §5. §7 (LLM decode) is skipped on CPU: LFM2's short-conv kernel is
  CUDA-only. Verified with Run All on CPU for default (linear/stack16) and conv/stack8.
- New `projector_anatomy.ipynb`: follows one audio token through the conv projector (family tree / receptive field,
  per-layer activations, exact Jacobian, occlusion, whole-utterance influence map, nearest vocab, conv1 kernels).
  Findings on `mix544h_conv_stack8`, dev-clean[10], token 47:
  - Receptive field = 15 frames: the token's own patch + the previous patch's last 7 frames; ~45% of its sensitivity
    comes from the previous patch. Per-frame sensitivity follows the stride-2 tree's 1-3-path comb.
  - Sparse units: ~9-15% active after conv2 / conv3 / fc1.
  - Audio tokens have median norm ~40 vs ~0.9 for text embeddings (44x; init `match_emb` was ~0.8), and cosine
    ~0.1-0.15 to their nearest vocab entry: a separate region of embedding space. Candidate experiment: norm
    penalty or an output LayerNorm on the projector.

## 2026-10-28 - Warm-start continuation `mix544h_conv_stack8_cont19k` cuts `test-clean` WER in half (71.0% -> 36.8%)
- Run: `runs/20261027-230948_mix544h_conv_stack8_cont19k` (`results/mel_20261027-230948_mix544h_conv_stack8_cont19k/summary.json`).
- Config delta vs `20261027-150147_mix544h_conv_stack8`: `INIT_FROM` parent LoRA + `ConvAudioProjector`, +19,200 steps (~3.0 epochs) on the same 544.1 h mix, `LR=1e-4` (`PROJ_LR_MULT=2.5`), `WARMUP_STEPS=300`, `MIN_LR_RATIO=0.1`, `cap6+nr4` decode (`MAX_NEW_TOKENS=256`, `MAX_TOKENS_PER_SEC=6.0`, `NO_REPEAT_NGRAM_SIZE=4`).
- Trajectory: `train_loss` `1.47 -> 0.380`; `dev-clean[:200]` WER `73.48% -> 36.60%` best @ step 19,000 (`sub 26.9 / del 4.2 / ins 5.4`, `wer_shuffle 105.22%`, 68.6-pt acoustic gap). LibriSpeech `test-clean` **36.83%** (`sub 26.35 / del 4.97 / ins 5.51`, vs 70.98% parent) and `test-other` **65.43%** (`sub 50.04 / del 7.44 / ins 7.94`, vs 85.38% parent).
- 9-set benchmark (vs parent under identical `cap6+nr4` decode): Open ASR 7-set `mean_wer` **93.99% -> 76.83%** (-17.16 pts; -40.08 pts vs original 116.91%): `librispeech_clean` 71.29 -> **36.92**, `librispeech_other` 85.30 -> **66.18**, `ami` 77.72 -> **57.20**, `gigaspeech` (unseen) 92.25 -> **73.26**, `voxpopuli` 105.96 -> **94.18**, `spgispeech` 110.75 -> **103.36**, `earnings22` 114.66 -> **106.69**; extra: `fleurs_en` 107.84 -> **91.43**, `edacc` 97.73 -> **93.19**.
- Rules IN: post-alignment optimization starvation (`MIN_LR_RATIO=0` after step ~3k ignition) was the primary bottleneck on LibriSpeech, AMI, and general read/podcast speech (`gigaspeech`), and `LFM2.5-230M` + `ConvAudioProjector` (`STACK=8`) has substantial headroom well below 40% clean WER without an audio encoder.
- Rules IN: even at `train_loss=0.38`, unseen accented/spontaneous/telephony domains (`earnings22`, `spgispeech`, `voxpopuli`, `edacc`) and 1.4%-weight `fleurs_en` remain substitution-dominated (59-76% `sub`), isolating corpus coverage/balance as the remaining bottleneck (tested next by `20261028-055047_mix654h_rebal_conv_stack8_cont19k`).

## 2026-10-28 - Log exact-match rate on validation
- `evaluate_dataset()` now returns `exact` (fraction of utterances whose normalised hyp equals the ref), shown in
  the `[val]`/`[test]` log line and in `metrics.jsonl`; `runs.plot_run()` draws it on a right-hand axis of the val panel.
- Applies to runs started from now on; earlier runs did not save val hypotheses, so they can't be backfilled.
- Note: in-training val is still `dev-clean[:200]` only (read LibriSpeech). `dev-other`, `ami_ihm/validation`
  (13,098 utts) and `fleurs_en/validation` (394 utts) exist on disk but are not used, so val WER does not track the
  AMI/FLEURS upweighting in `mix654h_rebal`.

## 2026-10-28 - Validate on one dev set per training corpus
- `VAL_SPLIT` is now a comma list scored separately; default `dev-clean,ami_ihm/validation,fleurs_en/validation`
  (200 utts each: dev-clean = first 200 ids as before; AMI/FLEURS = SEED-fixed sample within [MIN_TRAIN_SEC,
  MAX_TRAIN_SEC], AMI sample spans 18 meetings). Val rows in `metrics.jsonl` carry `split`; summary gets `val_wers`.
- `best_val_wer` / best step still come from the first split (dev-clean), so they stay comparable with older runs.
- Val cost ~3x per pass (plus ablations per split). `plots.png` val panels are now one per metric (WER, exact
  match, ablations, sub/del/ins) with a line per dev split; `runs.py curve` shows a split column.

## 2026-10-28 - Save val hypotheses per dev split and val pass
- Each val pass now writes `runs/<id>/val_hyps/<split>/step_<N>.jsonl` (`{id, ref, hyp}` for all 200 utts,
  ~50 KB; ~3 MB per 19k-step run with 3 dev splits). Test hyps were already saved in full (`runs/<id>/hyp_<split>.jsonl`,
  benchmark sets in `results/mel_<id>/<set>.jsonl`).

## 2026-10-28 - README: add model list, include LFM3-1.7B-Base
- New `## Models` table in README.md: models used by runs (230M-Base, 350M-Base, 350M) and targets (2.6B-Base).
- Added `LiquidAI/LFM3-1.7B-Base` (hidden 2048, 24 layers, released 2026-10-27) as untested. It ships custom code
  (`Lfm3ForCausalLM` via `auto_map`), and `train.py` loads without `trust_remote_code`, so a run needs that flag
  unless transformers adds native `lfm3`. Its tokenizer EOS is `<|im_end|>`, which is what gets appended to targets.

## 2026-10-28 - Rebalanced continuation `mix654h_rebal_conv_stack8_cont19k` (AMI x2, FLEURS x6) loses across all 9 sets at fixed steps
- Run: `runs/20261028-055047_mix654h_rebal_conv_stack8_cont19k` (`results/mel_20261028-055047_mix654h_rebal_conv_stack8_cont19k/summary.json`).
- Config delta vs `20261027-230948_mix544h_conv_stack8_cont19k`: identical warm start (`INIT_FROM=20261027-150147_mix544h_conv_stack8`), 19,200 steps, `LR=1e-4`, `MIN_LR_RATIO=0.1`, `MVN_SPLIT`, and `cap6+nr4` decode, changing only `TRAIN_SPLIT` to `train-clean-100+train-clean-360+ami_ihm/train*2+fleurs_en/train*6` (653.7 weighted h/pass -> 2.15x LS, 4.29x AMI, 12.87x FLEURS vs 3.02x uniform in `cont19k`).
- Trajectory & LibriSpeech test: `train_loss` `0.413` (vs `0.380`), `dev-clean[:200]` best WER **42.18%** @ step 19,000 (`wer_shuffle 105.69%`, vs `36.60%` in `cont19k`); `test-clean` **41.52%** (vs `36.83%`), `test-other` **68.59%** (vs `65.43%`).
- 9-set benchmark vs `cont19k`: Open ASR 7-set `mean_wer` **79.91%** vs **76.83%** (+3.08 pts worse across every set, *including* the two upsampled corpora): `librispeech_clean` 41.71 (+4.79), `librispeech_other` 69.46 (+3.28), `ami` **60.09** vs 57.20 (+2.89), `fleurs_en` **94.79** vs 91.43 (+3.36, `sub` 68.98 vs 64.23), `gigaspeech` 76.50 (+3.24), `voxpopuli` 97.75 (+3.57), `earnings22` 108.35 (+1.66), `spgispeech` 105.50 (+2.14), `edacc` 94.39 (+1.20).
- Rules OUT: reweighting the existing 544.1 h mix via corpus repetition (`*K`) as a fix for OOD/accented WER. Repeating 7.5 h FLEURS ~13x and noisy AMI ~4.3x while cutting clean, lexically rich LibriSpeech updates by 29% degrades general phonetic learning enough to hurt even `ami` and `fleurs_en` test sets.
- Rules IN: adding *unique* diverse hours (not repeating 2.6k FLEURS clips) while preserving clean phonetic step budget.

## 2026-10-28 - Refactor: `train.py` split into the `audiollm/` package (projector classes, SpeechLLM, trainer)
- `train.py` is now a thin CLI over `audiollm/`: `config.py` (same flat UPPERCASE keys, so old `config.json`, `runs.py`
  diffs and `INIT_FROM` keep working), `data.py`, `features.py`, `projectors/` (`linear`/`mlp`/`conv`, one class each,
  `@register_projector`), `backbone.py`, `model.py` (`SpeechLLM`), `trainer.py`, `evaluate.py`, `tracking.py`.
- Stacking moved out of the dataset into the projectors: the dataset emits 100 Hz frames, projectors run batched and
  return `ProjectorOutput(embeds, lengths per utterance, aux_loss, stats)`, ready for variable-rate projectors.
- Old `proj.pt` files load unchanged (legacy key prefix). Verified on CPU fp32 vs the pre-refactor code: same seeded
  init; batched projectors equal per-utterance ones to <1e-7 for all 3 projectors x STACK 4/8/16; a 4-step conv run
  gives the same loss at every step (6 decimals), identical val hyps, `proj.pt` diff 3e-7; `benchmark.py` hyps identical.
- New: model summary (LLM shape, params per component, trainable/dtype, projector layers) logged at start, saved as
  `model_summary.txt`, counts in `summary.json`; `runs/latest_model` -> newest run with weights, `runs.py best`,
  `runs.py latest --model`; `--batch_tokens N` token-budget batching (default 0 = off, unchanged behaviour).
- Small behaviour change: val/test decode budgets use the true utterance duration (as benchmark.py already did)
  instead of the STACK-padded one (<= 1 token of cap difference).
- Notebooks: `import train as T` still reaches config + functions, but cells using old signatures
  (`T.SpeechDataset(items, mvn, tok)`, `T.build_multimodal_inputs`, `T.ConvAudioProjector(M, S, H)`) need updating.

## 2026-10-28 - 100 h vs 544 h cold-start ablation (`ls100h_conv_stack8_12768`): `ConvAudioProjector` ignites on 100 h, data volume prevents transcript memorization
- Run: `runs/20261028-103311_ls100h_conv_stack8_12768` (`summary.json`).
- Config delta vs `20261027-150147_mix544h_conv_stack8`: identical `LFM2.5-230M-Base`, `PROJECTOR="conv"`, `STACK=8`, `LR=2e-4` (`PROJ_LR_MULT=2.5`), `WARMUP_STEPS=500`, `MIN_LR_RATIO=0`, and 12,768 steps, changing only `TRAIN_SPLIT="train-clean-100"` (100.6 h / 28,539 utts -> 14.31 epochs vs 2.00 epochs on 544.1 h) plus `cap6+nr4` decode.
- Trajectory & test WER: hits the exact same cold-start plateau (`proj_grad_norm_avg` `0.26` @ steps 1k-1.5k, `WER ~ wer_shuffle`), then ignites at steps 2.5k-3.5k (`proj_grad_norm_avg` jumps `0.26 -> 1.18` @ 5k; `dev-clean[:200]` WER `97.2%` vs `wer_shuffle 107.3%` @ 3.5k -> best **82.03%** vs `wer_shuffle 104.73%` @ 11.5k). However, from step 7k to 12.7k (epochs 8-14), `train_loss` collapses `0.638 -> 0.035` (`ppl 1.04`) while `val` WER flatlines at 82-83%. Final LibriSpeech `test-clean` **84.00%** (`sub 64.22 / del 11.87 / ins 7.91`, vs **70.98%** on 544 h @ 12.7k and **36.83%** @ 32k) and `test-other` **91.42%** (vs **85.38%** / **65.43%**).
- Rules IN (`ConvAudioProjector` + >=3k steps): breaking the 120% wall is an optimization + projector inductive-bias threshold (~3k steps with `conv` `STACK=8`), *not* something requiring >100 h to trigger (earlier 100 h runs used `linear`/`mlp` or stopped <=4.5k steps).
- Rules IN (data scale): once aligned, 100 h (28.5k utts) is memorized by the 230M LLM + 15.5M LoRA within ~8 epochs (`loss -> 0.035`), capping clean WER at 84%. Scaling unique utterances (544 h -> >1,000 h) is what converts post-ignition steps into phonetic generalization (84.0% -> 71.0% -> 36.8%).

## 2026-10-28 - Refactor verified on GPU; notebooks committed without outputs; smoke runs excluded from `runs/latest_model`

- GPU smoke of the `audiollm/` refactor (`20261028-145754_gpu_smoke_refactor_smoke50`, defaults, bf16): 50 steps in
  1:05, loss 5.60 -> 4.80, val on 3 dev sets ran, weights saved. With the earlier CPU fp32 old-vs-new equivalence
  checks, this is the go-ahead to commit the refactor.
- `.gitattributes` + `tools/strip_notebook_outputs.py` (git clean filter): committed `*.ipynb` lose outputs, execution
  counts and `language_info` run metadata; working copies keep their outputs. Per-clone setup is in README.
  History check: no commit so far contained notebook outputs.
- `runs/latest_model` skips `--max_steps` smoke runs (they save weights and were taking the link over).

## 2026-10-28 - SpecAugment added; queued +train-other-500 continuation with a same-data control
- Review of the ls100h ablation conclusions: "conv projector ignites" is confounded (the earlier 100 h runs also differ
  in LR / PROJ_LR_MULT / warmup / precision); "36.8 is data-bound" mixes steps with data (cont19k = +19.2k steps on the
  same 544 h, best val at step 19000/19200, so still improving). The clean finding is 100 h vs 544 h at a fixed
  config and step count: test-clean 84.00 vs 70.98.
- Code: `SPECAUG_FREQ_MASKS/FREQ_WIDTH/TIME_MASKS/TIME_WIDTH/MAX_TIME_RATIO` (`features.SpecAugment`, passed to the
  train `SpeechDataset` only). Masked bins = 0 = MVN mean. Off by default, so old configs are unchanged.
  GPU smoke run OK (1.05 s/step).
- Run A `20261028-152147_ls100h_conv_stack8_12768_specaug`: ls100h_conv_stack8_12768 config + 2x F27 + 2x T100
  (<=0.2 utt). Tests whether augmentation fixes the 100 h memorisation (train ppl 1.04, dev plateau ~82, ~22 pt over
  shuffled audio). ETA ~3 h.
- Queued after A finishes and train-other-500 is unpacked (`prepare_data.py train-other-500`):
  - B `mix1040h_conv_stack8_cont2_19k`: warm start from cont19k, +train-other-500 (~1040 h), same 19.2k-step schedule.
  - C `mix544h_conv_stack8_cont2_19k`: the same, but on the 544 h mix. This control separates +500 h from +steps.
  - Both add dev-other as the 4th val split (train-other-500 should mostly help test-other).
- Output-LayerNorm idea deferred: no run has shown the audio/text norm gap hurts.

## 2026-10-28 - Val/test splits named with their dataset (`runs.split_label`)
- "dev-clean" did not say it is LibriSpeech, and corpus val folders dropped the split ("ami_ihm"). Now every legend,
  log line, `runs.py curve/best` row and file name uses `<split>_<dataset>`: `dev-clean_librispeech`,
  `validation_ami_ihm`, `validation_fleurs_en`, `hyp_test-clean_librispeech.jsonl`. `metrics.jsonl` keeps the raw split spec.
- Plots already drew one line per dev split. ls100h_* runs show a single line only because their VAL_SPLIT is just
  dev-clean (inherited from the mix544h config).
- Existing run dirs renamed (val_hyps/*, hyp_*.jsonl) and all plots redrawn. The SpecAugment run is renamed when it exits.

## 2026-10-28 - Resumable checkpoints + HF Hub run mirror (Colab prep)
- `--checkpoint_every N` saves `runs/<id>/checkpoint/` (LoRA, projector, optimizer, LR schedule, loop position, RNG),
  written atomically and deleted at the end. `train.py --resume runs/<id>` continues in the same run dir and drops
  metrics rows logged after the checkpoint.
- Test (CPU, 230M, 8 steps, checkpoint every 3, `kill -9` at step 5, resume): every per-step loss and the final val
  WER are identical to the uninterrupted run. Found and fixed on the way: the RNG has to be restored after the epoch's
  DataLoader iterator is created; otherwise dropout masks shift.
- `--hub_repo user/name` mirrors the run dir to a private model repo after every checkpoint and at the end, in the
  background. Failures are only logged. `runs.py pull <repo> <run_id>` downloads a run back. Round trip tested:
  kill, delete the local dir, pull, resume, final push (Hub copy keeps lora/ + proj.pt and drops checkpoint/).
  An upload in flight at the kill is lost, so the previous checkpoint is used.
- Each run now saves its MVN stats as `runs/<id>/mvn.pt`; eval/benchmark/resume prefer that copy over the data/ cache.
- `requirements.txt` (pinned, torch excluded) and a Colab section in the README. No wandb: the Hub mirror already
  carries plots.png / metrics.jsonl / log.txt.

## 2026-10-28 - SpecAugment on 100 h: worse, stopped at step ~8650
- Run `20261028-152147_ls100h_conv_stack8_12768_specaug` (ls100h_conv_stack8_12768 config + 2x F27 + 2x T100).
  Stopped by hand (status interrupted), no weights saved.

  | step | baseline dev-clean / shuffled / loss | SpecAugment dev-clean / shuffled / loss |
  |---|---|---|
  | 4000 | 99.4 / 111.7 / 2.14 | 110.1 / 112.1 / 3.01 |
  | 6000 | 87.7 / 105.7 / 0.99 | 108.6 / 114.8 / 2.16 |
  | 8500 | 84.4 / 104.5 / 0.24 | 108.6 / 111.8 / 1.18 |
- Reading: it does slow memorisation (loss 1.18 vs 0.24), but the model never starts using the audio. The gap to
  shuffled audio is ~3 pt vs ~20 pt for the baseline. At 100 h and this LR schedule, masking up to 2 s of audio plus
  27/80 mel bins removes the signal the projector needs to align in the first place. Rules out SpecAugment from step 0
  as the 100 h fix. Untested: milder policy, or turning it on only after alignment (warm start).
- The queue went on to B `mix1040h_conv_stack8_cont2_19k` (+train-other-500), then C (544 h control).

## 2026-10-28 - Parked idea: loss-based soft data pruning (InfoBatch-style)
- Reviewed BLS (arXiv:2604.04681, EMA of batch loss as a per-sample score): rejected. The per-sample signal is
  1/B of the batch loss and the noise dominates. It also lacks a random-score control, and per-utterance CE is free here anyway.
- Parked instead: skip below-corpus-mean-loss utterances with prob r per epoch, reweight kept ones by 1/(1-r),
  no pruning in the last ~12.5% of steps. Selection cost is negligible.
- Not now: ~3-epoch runs would save only ~14% (epoch 1 is unscored), and data volume, not compute, is our lever.
  Best testbed is the ~14-epoch 100 h config. Planned arms: baseline / real loss / random scores.
- Design, implementation sketch and verified citations: `docs/idea_loss_based_soft_pruning.md`.

## 2026-10-29 - +train-other-500 vs same-data control: unique hours beat extra steps on all 9 sets
- B `20261028-171756_mix1040h_conv_stack8_cont2_19k` (1040 h) vs C `20261028-215900_mix544h_conv_stack8_cont2_19k`
  (544 h control). Both: warm start from cont19k, +19.2k steps, same config; only TRAIN_SPLIT differs.

  | WER % | cont19k | C 544 h | B 1040 h |
  |---|---|---|---|
  | librispeech clean / other | 36.9 / 66.2 | 31.8 / 62.5 | **25.1 / 49.4** |
  | ami / gigaspeech / voxpopuli | 57.2 / 73.3 / 94.2 | 54.2 / 70.2 / 91.0 | **52.2 / 64.4 / 80.4** |
  | spgispeech / earnings22 | 103.4 / 106.7 | 101.0 / 103.0 | **90.4 / 99.6** |
  | fleurs_en / edacc | 91.4 / 93.2 | 88.6 / 91.1 | 87.5 / 91.4 |
  | Open ASR 7-set mean | 76.83 | 73.38 | **65.95** |
- Extra steps alone: -3.5 mean. +500 h unique: another -7.4 mean, with gains also on sets train-other-500 does
  not cover (voxpopuli -10.5, spgispeech -10.6, gigaspeech -5.8). Accents (edacc, fleurs) barely move.
- Insertion analysis on B (earnings22 / spgispeech / voxpopuli ins 19-23% vs 4% on LS-clean): not loops (<25 looping
  hyps per set), not normaliser; hyps run 14-18% longer than refs because the LLM writes fluent audiobook-style prose
  ("i can not", "aint") over unfamiliar acoustics. Data problem first; a MAX_TOKENS_PER_SEC sweep (6/5/4/3.5, 600
  utts x 5 sets, tags `results/capsweep_B_cap*`) is running to see how much a decode cap can recover.
- Decision: keep adding unique, varied hours. Downloading VoxPopuli-en (~540 h) + TED-LIUM 3 (~450 h); GigaSpeech /
  SPGISpeech are gated (need terms accepted on HF). Queued: E `mix1040h_conv_stack8_cont3_19k` (B +19.2k steps, same
  data, control) then D `mix2000h_vox_ted_conv_stack8_cont3_19k` (B + VoxPopuli + TED-LIUM). voxpopuli stops being
  zero-shot in D.
- Code: `unpack_parquet_asr_corpora.py` gains voxpopuli_en / tedlium3 sources (TED-LIUM cleaner: drop
  ignore_time_segment_in_scoring, rejoin "it 's"), streams row batches. FLEURS re-unpack verified identical.
  `backbone.py`: LFM3-1.7B-Base loads with trust_remote_code pinned to an audited commit (not yet smoke-tested).

## 2026-10-29 - MAX_TOKENS_PER_SEC sweep on B: decode cap cannot remove the OOD insertions
- B `mix1040h_conv_stack8_cont2_19k`, 600 evenly spaced utts per set, `results/capsweep_B_cap{6.0,5.0,4.0,3.5}`.
  WER (ins) at cap 6.0 -> 3.5: earnings22 98.3 (22.6) -> 97.7 (22.0), voxpopuli 79.2 (21.0) -> 78.9 (20.6),
  spgispeech 89.5 (18.7) -> 89.4 (18.5), LS clean 25.3 -> 25.3, LS other 49.7 -> 49.7.
- Rules OUT: the length cap as a fix (<=0.6 pt even at 3.5 tok/s). Extra words are spread through utterances, not
  run-on tails. Keep cap 6. Rules IN (again): domain data. Control E `20261029-040621_mix1040h_conv_stack8_cont3_19k`
  started 04:06; D queued behind it and the VoxPopuli / TED-LIUM downloads (74 / 152 GB at 04:10).

## 2026-10-29 - Session OOM-kill lost control run E; plan now D -> 350M, GigaSpeech M + SPGISpeech S downloading
- 05:38 the VoxPopuli unpack (8 workers, parquet pre-buffering, ~4.8 GB RSS each) pushed RAM+swap past 90% and
  systemd-oomd killed the whole login session, including E (`20261029-040621_mix1040h_conv_stack8_cont3_19k`, step 6083/19200, no checkpoint).
- Fix: `unpack_parquet_asr_corpora.py` streams row batches with `pre_buffer=False` (2.1 GB peak/worker) and uses 4 workers.
  Long jobs now run as separate `systemd-run --user` units (unpack capped at MemoryMax=14G), and training uses `--checkpoint_every 1000` plus auto `--resume`.
- E is dropped, not rerun: B vs C already showed unique hours beat steps. D vs B is read with C's +3.5-point steps-only gain as the reference.
- User decision: no budget for LFM3-1.7B, so the bigger-backbone test is LFM2.5-350M-Base (16 layers vs 14, same 1024 width).
  New `INIT_PROJ_FROM` (config/train/trainer): loads only proj.pt from a run (feature/projector keys must match) and keeps LoRA fresh,
  so the 350M starts from B's projector. Smoke test: 60 steps OK, ~1.05-1.2 s/step (230M: ~0.8).
  The first smoke attempt died once with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (ROCm) at step ~30; the retry was clean. Not reproduced.
- Queue (`queue-d-350m` user unit, log `runs_queue_d_350m.out`):
  - D = B + VoxPopuli-en + TED-LIUM 3 (~2000 h), warm from B, 19.2k steps.
  - Then `lfm350m_mix2000h_projinit_19k`: same data and steps as D, LR 2e-4, warmup 500.
- The user accepted the GigaSpeech/SPGISpeech terms. `download_corpora.py` gains `gigaspeech_m` (parquet M, 1000 h, 120 GB)
  and `spgispeech_s` (S, 200 h, 45 GB), now downloading. Training on them makes the gigaspeech/spgispeech benchmarks in-domain; earnings22 stays OOD.

## 2026-10-29 - D (B + VoxPopuli-en + TED-LIUM 3, ~1983 h): big gains everywhere; 350M started
- Run `20261029-074100_mix2000h_vox_ted_conv_stack8_cont3_19k`: B warm start, 19.2k steps, same LR schedule,
  train = B's mix + voxpopuli_en/train (501 h) + tedlium3/train (454 h) = 796k utts. 4.2 h train, ~0.8-1 s/step.
- Benchmark WER, B -> D: libri clean 25.12 -> 19.99, other 49.42 -> 44.58, voxpopuli 80.44 -> 40.45 (now in-domain),
  ami 52.17 -> 46.13, earnings22 99.62 -> 87.72, gigaspeech 64.44 -> 50.54, spgispeech 90.41 -> 77.04,
  fleurs 87.51 -> 82.95, edacc 91.35 -> 83.75. Mean of 9: 71.16 -> 59.24.
- LibriSpeech -5 vs C's steps-only ~-3.5: new data is worth more than extra steps. Unseen sets
  (earnings22, gigaspeech, spgispeech, edacc) improve 7-14 points, so this is not just in-domain fitting.
- Insertions dropped a lot on OOD (earnings22 ins 22.7 -> 17.3, spgispeech 18.9 -> 14.6).
- GigaSpeech M (910k utts, 999.9 h; val 5715 / 11.4 h) and SPGISpeech S (77k utts, 196.2 h; val 13.1k / 33.3 h)
  unpacked. No leftover tags or empty texts. 0 source/call overlap with the ESB test shards.
  From the next stage on, the gigaspeech/spgispeech benchmarks are in-domain; earnings22/edacc/fleurs-test stay unseen.
- 350M run `20261029-123151_lfm350m_mix2000h_projinit_19k` (same data/steps as D, projector from B, fresh LoRA,
  lr 2e-4) started 12:32, ~0.8 s/step, ETA ~16:45 + benchmark.
- Next: compare 350M vs D, then a ~3200 h stage (+gigaspeech_m/train +spgispeech_s/train) on the winner.

## 2026-10-29 - 350M (projector from B, fresh LoRA, 19.2k steps) far behind D; not a matched comparison
- Run `20261029-123151_lfm350m_mix2000h_projinit_19k`: same data/steps as D, lr 2e-4, warmup 500. 4:58 train (~1.0 s/step vs ~0.8 for 230M).
  Finished 17:33, benchmark done ~18:14 (one HIP OOM warning on edacc, recovered).
- Benchmark WER, D (230M) -> 350M: libri clean 19.99 -> 39.53, other 44.58 -> 62.54, voxpopuli 40.45 -> 59.40,
  ami 46.13 -> 57.43, earnings22 87.72 -> 99.51, gigaspeech 50.54 -> 65.62, spgispeech 77.04 -> 93.09,
  fleurs 82.95 -> 95.38, edacc 83.75 -> 91.06. Mean of 9: 59.24 -> 73.73. Final train loss 0.66 vs 1.08.
- Not a fair verdict on size: D is the 4th warm-start stage of the 230M (~70k cumulative steps), the 350M had 19.2k with fresh LoRA.
  On the 230M's own path, test-clean was 71.0 after stage 1 (12.8k steps) and 36.8 after cont19k (~32k), so the 350M at
  19.2k (39.5) learns about as fast per step. It does not learn clearly faster, and each step costs ~25% more.
- Rules OUT switching to 350M now. Data was the big lever (B->D -12 mean), so the next stage stays on the 230M.
- Next: E2 = D warm start + gigaspeech_m/train + spgispeech_s/train (~3200 h), 19.2k steps, same schedule as D.

## 2026-10-29 - Survey: Vietnamese ASR / audio-understanding datasets (no code change)
- Wrote `docs/DATASETS_VI.md` (tiers, access checked against HF today, shard-level audits with PhoWhisper-small CER spot-checks, eval sets, WER references, normalisation recipe).
- Blocker to note first: the LFM2.5 tokenizer has no Vietnamese coverage (~3.0 tokens/word vs 1.1–1.4 for Qwen3/Gemma3/Llama3.2). Any vi run should A/B LFM2.5 against a vi-capable backbone of similar size.
- Clean and downloadable today: PhoAudiobook (0.9% CER spot-check, 1.5k h), VLSP2020 100 h (2.2%), LSVSC 100 h (numbers need verbalising), ViMD 102 h (dialects, NC-ND).
- Large but label C: GigaSpeech2-vi 6k h (gate + Google Form; its 11 h professional test set is the main benchmark), LEMAS-vi 6.6k h (not yet audited), VietSpeech, Bud500 (PhoASR kept about 50%).
- Reject as is: VietSuperSpeech (pseudo-labels, 17% English gibberish, 51.6% CER; card's hours wrong).
- Not started: vi data loader, vi WER normaliser in benchmark.py, choice of backbone.
