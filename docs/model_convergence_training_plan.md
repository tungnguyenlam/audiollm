# Model Convergence Training Plan

**Date:** 2026-11-05  
**Target Model / Checkpoint:** E2 continuation (`runs/20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k`, LFM2.5-230M-Base, Conv projector, stack 8 = 12.5 tok/s).  
**Reviewer:** Claude Opus 5.5 (`claude-opus-5-5` via Claude Code CLI).  
**Review Status:** **APPROVED WITH MINOR FIXES** (Incorporated on 2026-11-05).

---

## 1. Problem Statement & Convergence Analysis

### Current Status
- **Baseline Checkpoint:** E2 (`20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k`) trained on a 3,179.4-hour multi-corpus English mix (1,783,428 utterances) across LibriSpeech (100+360+500), AMI IHM, FLEURS-en, VoxPopuli-en, TED-LIUM 3, GigaSpeech M, and SPGISpeech S.
- **Current Performance:**
  - 8-split Dev Mean WER: **39.85%** (dev-clean 20.61%, dev-other 34.99%, TED-LIUM 30.19%, VoxPopuli 37.18%, SPGISpeech 37.06%, AMI 41.48%, GigaSpeech 49.04%, FLEURS 67.35%).
  - Open ASR Leaderboard 7-set Test Mean WER: **41.36%** (test-clean 17.97%, test-other 41.96%).
- **Why is the Error Rate Too High?**
  1. **Premature Stopping / Under-exposure:** E2 was trained for only 19,200 optimizer steps with effective batch size 32 (16 microbatch × 2 grad accum).
     $$\text{Utterances Seen} = 19,200 \times 32 = 614,400 \text{ utterances}$$
     $$\text{Epoch Reached} = \frac{614,400}{1,783,428} \approx \mathbf{0.3528 \text{ epochs (35.3\%)}}.$$
     Over **64.7% (~2,060 hours / 1.17 million utterances)** of the dataset has *never been seen once* by the model!
  2. **Premature LR Annealing:** The cosine schedule annealed the learning rate down to the $0.1\times$ floor ($1.0 \times 10^{-5}$) at step 19,200 while the training loss was still declining steeply ($0.831 \to 0.567$) without flattening.
  3. **High-Noise Out-of-Domain Datasets:** Corpora such as FLEURS (67.4% dev WER) and GigaSpeech M (49.0% dev WER) require repeated exposure to adapt acoustic representations.

### Contrast with Vietnamese Pipeline (Why Not Vietnamese?)
- The Vietnamese Qwen arm (`runs/20261101-051531_vi500h_qwen_projE2_matched19k`) was trained for 19,200 steps on 206,977 utterances, completing **2.97 full epochs**.
- As audited on 2026-11-04, its training loss plateaued at ~0.27 with sharp drops at epoch boundaries (6.5k and 13k steps), indicating memorization of the 500 h set. Continuing training on the same 500 h dataset yields negligible gain without adding new data.
- The repo's primary research program (adaptive token merging on encoder-free speech LLMs) is anchored on the English E2 baseline.
- Therefore, training to convergence on the current dataset specifically targets **completing exposure on the 3,180 h English dataset**.

---

## 2. Convergence Strategy & Training Schedule

### Target Training Budget
- To reach **1.0 full epoch** of the 3,180 h dataset from scratch requires:
  $$\frac{1,783,428 \text{ utterances}}{32 \text{ utterances/step}} \approx 55,732 \text{ steps}.$$
- Since E2 completed 19,200 steps, completing full-dataset exposure requires an additional **36,532 steps**.
- Running a continuation budget of **38,400 steps** (two 19.2k blocks):
  - Total cumulative exposure: $19,200 + 38,400 = 57,600 \text{ steps} \approx \mathbf{1.03 \text{ epochs}}$.
  - The model will see all 3,180 hours of diverse acoustic data at least once.
  - Runtime: At ~0.55 s/step, 38,400 steps takes approximately **5.8 hours** of GPU time on the AMD ROCm device (7.0 GB peak VRAM).

### Learning Rate & Warmup Architecture
- **Warm-start (`--init_from`):** Initialize all weights (LoRA adapters and Conv audio projector) from `runs/20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k`.
- **Optimizer:** Fresh AdamW state, zero weight decay.
- **Schedule:**
  - Peak LR: $\eta_{\text{peak}} = 1.0 \times 10^{-4}$ (projector multiplier 2.5 $\implies 2.5 \times 10^{-4}$).
  - Warmup: 500 steps linear warmup from 0 to $\eta_{\text{peak}}$.
  - Decay: Cosine annealing from step 500 to 38,400, decaying to $\eta_{\text{min}} = 0.1 \times \eta_{\text{peak}} = 1.0 \times 10^{-5}$.
- **Data Shuffling Fix (per Claude review):**
  - Use `--seed 1` so that `create_length_batched_sampler(..., seed=SEED + epoch)` generates a distinct random permutation of the 1,783,428 utterances, immediately drawing samples uniformly across the entire pool rather than replaying the prefix seen during E2.

---

## 3. Detailed Configuration Specifications

| Parameter | Value | Justification |
|---|---|---|
| **Base Config** | `runs/20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k/config.json` | Exact match with E2 architecture and pipeline |
| **`INIT_FROM`** | `runs/20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k` | Continuation from E2 checkpoint |
| **`RUN_NAME`** | `mix3200h_conv_stack8_cont5_converge38k` | Clear naming following repo standards |
| **`SEED`** | `1` | Distinct shuffle from E2 to maximize new utterance exposure |
| **`TRAIN_SPLIT`** | `train-clean-100+train-clean-360+train-other-500+ami_ihm/train+fleurs_en/train+voxpopuli_en/train+tedlium3/train+gigaspeech_m/train+spgispeech_s/train` | Full 3,180 h multi-corpus dataset |
| **`MVN_SPLIT`** | `train-clean-100+train-clean-360+ami_ihm/train+fleurs_en/train` | Preserves identical MVN statistics as E2 |
| **`VAL_SPLIT`** | `dev-clean,ami_ihm/validation,fleurs_en/validation,dev-other,voxpopuli_en/validation,tedlium3/validation,gigaspeech_m/validation,spgispeech_s/validation` | 8-split balanced dev suite (200 utts each) |
| **`TRAIN_STEPS`** | `38400` | Reaches ~1.03 cumulative epochs of 3,180 h data |
| **`BATCH_SIZE`** | `16` | Microbatch size per accumulation |
| **`GRAD_ACCUM`** | `2` | Effective batch size = 32 utterances/step |
| **`WARMUP_STEPS`** | `500` | 500-step linear warmup for smooth optimizer re-entry |
| **`LR`** | `1e-4` | Matched with all previous successful continuations (cont2, cont3, cont4) |
| **`MIN_LR_RATIO`** | `0.1` | Decays to 1e-5 |
| **`PROJ_LR_MULT`** | `2.5` | Projector peak LR 2.5e-4 |
| **`EVAL_EVERY`** | `1000` | Regular 8-set validation pass (every ~10 min) |
| **`CHECKPOINT_EVERY`**| `1000` | Resumable checkpoint written every 1000 steps |
| **`LOG_EVERY`** | `20` | Loss, grad norms, throughput logging |
| **`BENCHMARK_SETS`** | `all` | Full 7/9-set Open ASR Leaderboard evaluation at completion |
| **`PRECISION`** | `amp` (`bfloat16`) | Fast ROCm execution, ~7.0 GB VRAM |

### Launch Command
```bash
python train.py \
  --config runs/20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k/config.json \
  --init_from runs/20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k \
  --seed 1 \
  --train_steps 38400 \
  --warmup_steps 500 \
  --lr 1e-4 \
  --min_lr_ratio 0.1 \
  --checkpoint_every 1000 \
  --eval_batch_size 16 \
  --eval_every 1000 \
  --run_name mix3200h_conv_stack8_cont5_converge38k \
  --notes "E2 continuation to ~1.03 epochs of 3180h English mix (convergence run)"
```

---

## 4. Verification Gates & Execution Protocol

### Pre-Flight Verification Gates (Before launching full 38.4k steps)
- **Gate 0 (Weight & State Continuity):**
  - Verify that loading `--init_from` successfully loads both `proj.pt` and `lora/` without key mismatches or missing layers.
- **Gate 1 (Zero-Step Loss / Gradient Equivalence):**
  - Run a 10-step smoke test (`--max_steps 10 --eval_every 5 --val_n 10`) to confirm that initial loss falls in the range $[0.50, 0.70]$ (matching the tail loss of E2 $\approx 0.58$) and gradients are finite and non-zero.
- **Gate 2 (VRAM & Speed Stability):**
  - Confirm memory consumption remains stable at $\le 7.5\text{ GB}$ (within the 16 GB hardware limit) and step speed is $\le 0.70\text{ s/step}$.

### Monitoring & Milestones
- Track the 8-split dev mean WER at each 1,000-step interval.
- **Target Milestones:**
  - Dev-clean WER: Drop from 20.61% to $< 17.0\%$.
  - Dev-other WER: Drop from 34.99% to $< 30.0\%$.
  - 8-split Dev Mean WER: Drop from 39.85% to $< 34.0\%$.
  - 7-set Open ASR Leaderboard Mean: Drop from 41.36% to $< 36.0\%$.

---

## 5. Review History
- **2026-11-05:** Reviewed and approved by Claude Opus 5.5 (`claude-opus-5-5`). All recommendations (seed offset for uniform sampling across pool, explicit command, Gate 1 loss range definition) incorporated.
