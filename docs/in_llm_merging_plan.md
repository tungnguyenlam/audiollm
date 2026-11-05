# In-LLM Training-Free Audio Token Merging: Experiment & Implementation Plan

**Date:** 2026-11-05  
**Baseline / Checkpoint:** E2 (`20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k`, LFM2.5-230M-Base, stack 8, 12.5 tok/s, dev mean WER 39.85).  
**Context:** Builds on Phase 0 (`results/merge_sweep_20261104/`) and the layer redundancy probe (`results/layer_redundancy_20261104/probe.py`).  
**Review Status:** Approved for execution by Claude Opus 5.5 on 2026-11-05.

---

## 1. Research Question & Hypotheses

- **Primary Question:** In an encoder-free mel -> LLM ASR model, does merging audio tokens after an intermediate layer $k$ suffer less WER degradation than merging at the projector output ($k=0$) at equivalent computational savings?
- **Hypothesis:** Because adjacent token redundancy doubles inside the LLM (rising from ~0.20-0.28 at input to ~0.50 around layers 6-9 as the model learns an implicit encoder), representations in deeper layers are more semantic and compressible. Merging at $k \in \{2, 4, 6, 8, 10\}$ will yield a better Pareto frontier (lower WER for the same effective compute / KV cache reduction) than $k=0$.
- **Downstream Purpose:** Identifies the optimal boundary layer where a learned CIF / adaptive merge module should be placed in Phase 3.

---

## 2. Fair Comparison & Metric Formulation

**Critical Principle:** Merging before layer $k$ only compresses tokens for the remaining $(14 - k)$ layers. At an identical audio token keep ratio $R = T' / T$, deeper layers automatically do less compression overall and would trivially show lower WER.

Therefore:
1. **Primary Metric (Effective Token Ratio $E$):**
   $$E = \frac{\bar{T}_{\text{eff}}}{T_{\text{audio}}} = \frac{k + (14 - k) \cdot R_k}{14}$$
   The sweep grid is defined over target **$E \in \{0.93, 0.87, 0.80, 0.75\}$**, ensuring overlapping, directly comparable points across all layers.  
   The required layer keep ratio $R_k$ is:
   $$R_k = 1 - (1 - E) \cdot \frac{14}{14 - k}$$
   
   | $k$ | $E=0.93$ | $E=0.87$ | $E=0.80$ | $E=0.75$ |
   |:---:|:---:|:---:|:---:|:---:|
   | 0 | 0.930 | 0.870 | 0.800 | 0.750 |
   | 2 | 0.918 | 0.848 | 0.767 | 0.708 |
   | 4 | 0.902 | 0.818 | 0.720 | 0.650 |
   | 6 | 0.878 | 0.772 | 0.650 | 0.562 |
   | 8 | 0.837 | 0.697 | 0.533 | 0.417 |
   | 10 | 0.755 | 0.545 | 0.300 | 0.125 |

   *Rule:* All 24 grid points are evaluated; a setting is dropped only if no calibrated $\tau_k$ falls within $\pm 0.01$ of $R_k$ on the calibration slice.
2. **Attention KV Cache Entries Saved:**
   $$\Delta \text{KV}_{\text{attn}} = \sum_{l \in \{2, 4, 6, 8, 10, 12\}, l \ge k} (T_{\text{audio}} - T'_{\text{audio}})$$
   *Note on architecture:* Since layers 0 and 1 are conv, **$k=2$ saves exactly as much attention KV as $k=0$** (6 attention layers each). On this axis, $k=2$ is a strictly free representation upgrade.
3. **Fair Twin Control:** Every adaptive `affinity` arm is paired with a `uniform_matched` twin that keeps the exact same per-utterance token count $T'_{\text{audio}}$ at layer $k$, spaced evenly. This isolates *where* tokens are merged from *how many* are merged.
4. **All-Audio Dropped Control ($R=0$ Floor):** Run an arm per layer $k \in \{2, 4, 6, 8, 10\}$ where all audio tokens are completely dropped after layer $k-1$ ($R=0$). This tests whether upper layers still depend on audio representations after lower layers have attended to them, establishing the performance floor.
5. **Silence Confound Monitoring:** Track the fraction of removed tokens falling in the quietest 20% of frame energy (using `where.py` energy quantile methodology) at each layer $k$.
6. **Statistical Significance:** Stratified paired bootstrap test (resampling within each of the 8 dev sets and taking the mean) and one-sided sign test across the 8 dev sets (requiring $\ge 7/8$ sets agreeing in sign, $p \le 0.035$).

---

## 3. Architecture & Layer Selection

LFM2.5-230M has 14 decoder layers ($0 \dots 13$) with hybrid operator structure:
- **Causal Conv1d (`conv`, kernel size 3):** Layers 0, 1, 3, 5, 7, 9, 11, 13
- **Full Attention (`self_attn`):** Layers 2, 4, 6, 8, 10, 12

**Probe Layers:** Even $k \in \{0, 2, 4, 6, 8, 10\}$.
- Merging immediately before an attention layer preserves local conv context in the lower block.
- Covers early encoder ($k=2, 4$), the implicit encoder boundary suggested by arXiv 2610.24239 ($k=4$), the redundancy plateau ($k=6, 8$), and late-layer compression ($k=10$).
- $k=0$ serves as the input baseline (re-evaluated in the exact same harness).

---

## 4. Implementation Design: No-Cache Exact Loop

To avoid brittle cache manipulation across heterogeneous conv/attention layers during the exploratory sweep, evaluation uses a **no-cache greedy decode loop**. It is mathematically exact and enforces clean separation between the lower and upper blocks.

### Step-by-Step Execution:
1. **Lower Block Forward (Layers $0 \dots k-1$):**
   - Input: Left-padded batch `[BOS, audio_tokens, prompt_tokens, gen_tokens_so_far]`.
   - Pass lower position IDs: `position_ids = attention_mask.cumsum(-1) - 1`.
   - Generate causal mask dict `{"full_attention": 4D, "conv": 2D}`.
   - Run layers $0 \dots k-1$.
2. **Audio Extraction & Merging (Prefill Phase):**
   - For row $r$, isolate audio token hidden states $H_{\text{audio}}[r]$ at layer $k-1$ output.
   - **Similarity Definition:** Raw residual-stream cosine (uncentered), matching Phase 0 and probe uncentered metrics.
   - Group boundaries, counts, and pooled representations $H'_{\text{audio}}[r]$ are computed **once** at prefill and frozen for subsequent decode steps.
3. **Upper Block Sequence Reconstruction (Layers $k \dots 13$):**
   - Re-assemble each utterance: `[BOS, H'_{audio}[r], H_{prompt}, H_{gen}]`.
   - Re-pad to uniform length with left-padding.
   - Construct upper attention mask and **recomputed upper rotary position embeddings**:
     - Primary setting (*Compact Re-indexing*): contiguous 0-indexed positions $0 \dots L'-1$.
     - Rebuild 2D conv padding mask to ensure no causal conv leakage into BOS or initial audio tokens.
   - Run layers $k \dots 13$, final `embedding_norm`, and `lm_head`.
4. **Token Generation:**
   - Apply `NoRepeatNGramLogitsProcessor(4)` to generated token IDs only (excluding prompt tokens).
   - Select greedy token (`argmax`).
   - Enforce per-utterance `decode_token_budget(audio_seconds)`.
   - Dynamic batch pruning: drop finished utterances from the active batch to optimize runtime.

---

## 5. Verification & Equivalence Gates (Pre-Sweep)

Before launching the full sweep across dev sets, the implementation must pass four strict gates:

- **Gate 0 (Identity Split & Rebuild):** For each $k \in \{2, 4, 6, 8, 10\}$, run with groups of size 1 (identity split and rebuild, no merging) in fp32. Output must be **100% token-identical** to standard unmerged $k=0$.
- **Gate 1 (No Merge Baseline in bf16):** With $k=0$ (no merge), the custom loop must reproduce the baseline dev mean WER of **39.85** ($\pm 0.1$). Also report the % of hypotheses identical to HF `generate()`.
- **Gate 2 (Phase 0 Reproduction):** At $k=0$ with raw-cosine `affinity` at $\tau=0.70$ (yielding keep ratio ~0.87), the custom loop must reproduce Phase 0 dev mean WER of **39.77** ($\pm 0.1$).
- **Gate 3 (Batch Invariance under Merging):** In fp32 with merging turned on at an intermediate layer ($k=4$), transcribing batch size 1 must be **100% token-identical** to transcribing batch size 16 (verifying left-padding and conv-mask zeroing under heterogeneous merged lengths).

---

## 6. Experimental Sweep Matrix

- **Evaluated Checkpoint:** E2 (`20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k`).
- **Dev Sets (8 sets $\times$ 200 utts):**
  1. `dev-clean` (LibriSpeech)
  2. `dev-other` (LibriSpeech)
  3. `ami_ihm/validation`
  4. `fleurs_en/validation`
  5. `voxpopuli/validation`
  6. `tedlium3/validation`
  7. `gigaspeech_m/validation`
  8. `spgispeech_s/validation`
- **Calibration of $\tau_k$:** Calibrate $\tau_k$ on a held-out slice of 64 utterances per set, strictly disjoint from the 200 eval utterances:
  - For LibriSpeech splits: sorted IDs `[200:264]`.
  - For other corpus splits: length-filtered pool minus the 200 eval IDs, sampled with `random.Random(C.SEED + 1).sample(..., 64)`.
  - An assertion verifies zero overlap between calibration and evaluation sets.
- **Evaluated Arms:**
  - `affinity` ($\omega=1$).
  - `uniform_matched` (identical per-utterance token count as affinity at layer $k$).
  - `all_dropped` ($R=0$ floor, 1 run per $k \in \{2, 4, 6, 8, 10\}$).
- Actual keep ratio and effective token ratio $E$ are recorded and plotted directly from the evaluation run.

---

## 7. Secondary Ablations (at Best Performing $k^*$)

Once the optimal layer $k^*$ is identified from the primary sweep:
1. **Window $\omega=3$ vs $\omega=1$:** Test whether multi-token lookback helps deep layers (as in Affinity Pooling AP_deep).
2. **Position Encoding Strategy:** Compare *Compact Re-indexing* against *Original Timing Positions* (where merged tokens retain the group's initial index and the prompt retains original absolute positions).
3. **Proportional Attention:** Add ToMe-style $\log(\text{group\_size})$ bias to the merged-key columns of the float 4D additive attention mask for layers $\ge k$.
4. **Similarity Metric:** Compare raw residual cosine against layer $k$'s `operator_norm` and mean-centered cosine.

---

## 8. Success Criteria & Deliverables

1. **Success Condition:**
   - At equal effective token ratio $E$ (or equal KV cache saved), an intermediate layer $k^* > 0$ achieves lower WER than input merging ($k=0$).
   - `affinity` retains a statistically significant margin over `uniform_matched` at $k^*$, exceeding the $R=0$ baseline and what can be explained by silence collapse alone.
2. **Deliverables:**
   - Verification test script `results/merge_in_llm_20261105/verify_gates.py`.
   - Self-contained sweep script `results/merge_in_llm_20261105/sweep.py`.
   - Comprehensive results table, Pareto curves ($E$ vs WER, $\Delta \text{KV}$ vs WER), and silence breakdown in `results/merge_in_llm_20261105/summary.md`.
   - WORKLOG entry documenting findings and guiding Phase 3 learned merge placement.
