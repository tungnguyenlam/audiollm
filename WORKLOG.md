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
