# Adaptive audio-token merging: experiment plan

Started 2026-11-04. This is the repo's main research question (README: the fixed-rate baseline is "the control that
the dynamic-resolution tokenizer will be compared against"). Background: `docs/literature_review.md`.

**Question.** In an encoder-free mel → LLM ASR model, does content-adaptive merging of audio tokens beat fixed-rate
reduction at the same average token rate? Equivalently, does it reach the same WER with fewer tokens?

**Baseline / control.** E2 = `20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k`: LFM2.5-230M, conv
projector, stack 8 = 12.5 audio tokens/s, ~3,200 h English. Dev mean WER 39.85 (8 sets × 200 utts). 7-set Open ASR
test mean 41.36.

**Metric.** WER against average audio tokens per second (a Pareto curve), dev mean over the 8 E2 dev sets plus
per-set values. Final claims use `benchmark.py` test sets. Report prefill cost (audio + prompt tokens) next to WER.

**Fair comparison.** Every adaptive arm gets a `uniform_matched` twin: the same number of tokens per utterance,
evenly spaced. The difference then measures *where* merging happens, not *how much*. A trained arm also needs a
same-steps, no-merge continuation control, so extra training is not credited to merging.

**Positioning (novelty search 2026-11-04, `docs/literature_review.md`).**
- "Encoder-free LLMs learn implicit encoders" is already published (arXiv 2610.24239, Oct 2026). It has no token
  redundancy or merging analysis.
- Every in-LLM speech-token merge so far is training-free (FastAdaSP, AudioTETRIS, Affinity Pooling).
- Every learned-rate method (CIF, ACIF) sits before the LLM. None is encoder-free.
- Target contribution: a learned merge at the end of the implicit encoder in an encoder-free model, plus a
  redundancy-across-depth analysis and the en/vi rate comparison.

## Phases

0. **Training-free merging on E2** (`MERGE=affinity|uniform_matched|uniform`, `audiollm/merging.py`). Sweep τ and
   record WER against tokens/s. Cheap: ~40 s per setting. Expect degradation everywhere, because E2 never saw merged
   tokens. The useful signal is whether affinity degrades less than `uniform_matched` at equal budget.
1. **Merge-aware fine-tune.** Continue E2 for a short budget (~4.8k steps, ~1 h) with merging in the loop, at one or
   two keep ratios (~0.6 and ~0.4). Arms: affinity, `uniform_matched`, no-merge control. Win condition: affinity
   beats `uniform_matched` on the dev mean at equal tokens/s, by more than noise (repeat with a second seed if
   close).
2. **Where to measure similarity.** Merge on the projector-output embeddings, on intermediate conv features, or on
   raw mel. Also try a deep-LLM-layer merge (Affinity Pooling's AP_deep). For a 14-layer model that mostly saves
   KV cache.
3. **Learned boundaries.** A boundary predictor with a rate penalty through `ProjectorOutput.aux_loss` (CIF-style
   integrate-and-fire, or CTC-guided collapse). This is the real "dynamic-resolution tokenizer" if phases 0–1 show
   that position matters.

## Log

Results are appended to `WORKLOG.md`; sweep artifacts are under `results/merge_sweep_*`.

## Parked: cross-language compression (supervisor's suggestion)

Compare how much each language can be compressed. English and Vietnamese differ in syllable structure:
Vietnamese is monosyllabic and tonal. Questions:
- At equal WER degradation, does adaptive or learned merging keep a different number of tokens per second in vi
  than in en?
- Is the kept-token count per syllable or word more stable across languages than per second?
- Does the layer at which redundancy appears move between languages?

The Vietnamese 230M and Qwen3-0.6B runs from the backbone A/B can be reused for training-free sweeps and the
layer probe. A learned-rate model should be trained on both languages under matched settings.

## Parked: task-dependent compressibility (ASR vs speech translation)

Idea from a Gemini chat (2026-11-04). Verbatim ASR needs every phone and tone. A semantic task such as speech
translation may tolerate far more merging. Hypothesis: the en/vi rate gap is set by phonology (tones, syllables)
in ASR, but by information rate in translation.

Parked because:
- the 230M models are trained for ASR only;
- en↔vi parallel speech is scarce (FLEURS only; CoVoST 2 and MuST-C appear to lack vi, not re-checked);
- it adds BLEU as a second axis before the main ASR result exists.

Revive only if the learned merge works on ASR. Until then, mention it as a limitation or future work.

Gemini's accompanying roadmap is rejected. It proposed a Whisper-large-v3 + Qwen-2.5-7B model:
- it is encoder-based, which drops our differentiator;
- it duplicates arXiv 2606.10439 (Whisper → CIF → Qwen-2.5 7B, 11 languages incl. vi);
- it does not fit our 16 GB GPU.

## Parked: merge-aware fine-tune with the in-LLM merge (2026-11-05)

Proposal: after Stage A converges, fine-tune with fixed affinity merging after LLM layer 2, the best point in the
training-free in-LLM sweep. Question: does adaptation recover the WER lost at aggressive keep ratios? This is
Phase 1 moved from the projector output (k=0) to k=2. It is a stepping stone to Phase 3, not the contribution.
Fixed merging adapts the representations, but the discrete boundaries get no gradient. Learning them still needs
CIF-style aggregation or gates with a token-budget loss.

Caveats before reviving:
- The k=2 result is weak evidence. It is one run at E=0.93 (-0.33 dev mean, 200 utts per set) on the
  under-trained E2. Redo the training-free k × E sweep on the Stage A checkpoint before choosing k.
- In LFM2.5-230M, layers 0–1 are conv blocks and attention starts at layer 2. So k=2 saves exactly as much KV as
  k=0 and is barely "inside" the implicit encoder. A deeper k is the paper's claim, so keep one deeper arm
  (e.g. k=4 or 6) at mild E.
- Fine-tune at E ≈ 0.6 and 0.4, where training-free merging fails. E=0.93 has nothing to recover.
- Required arms (see "Fair comparison"): affinity and `uniform_matched` at the same k and E, plus a no-merge
  continuation with the same number of steps. Add a k=0 arm. It needs no new code (`MERGE=affinity` already runs
  in training) and shows whether the in-LLM position matters after adaptation.

Code needed for k>0: `audiollm/layer_merging.py` only supports inference (`@torch.no_grad`, greedy decode). A
training forward must run layers [0, k), merge only the audio span, rebuild the masks, position ids and label
alignment, run [k, 14) and compute CE. Reuse its pooling and mask logic, and gate it with the same fp32
split/rebuild identity check (Gate 0) on the loss instead of tokens.

Revive when Stage A (`runs/20261105-111014_mix3200h_conv_stack8_stageA_55k`) is benchmarked.

## Next step (as of 2026-11-04)

Training-free merge **inside** E2: merge after LLM layer 4, 6, or 8 instead of at the input. Use Phase 0 keep
ratios, each with a `uniform_matched` twin. Question: is merging at the end of the implicit encoder cheaper in
WER than merging at the input? The answer decides where the learned CIF merge goes. Not implemented yet.
