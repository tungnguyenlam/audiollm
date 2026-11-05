# Worklog

Oldest entry first; append new entries at the bottom with `cat >> WORKLOG.md`.
Never read this file whole. See "Worklog" in `AGENTS.md`.
Earlier months: `docs/worklog_archive/YYYY-MM.md`.


## 2026-11-01 - Vietnamese backbone A/B lfm completed
- Run `20261031-135333_vi500h_lfm_projE2_matched19k`: selected 500 h; E2 projector + fresh LoRA; 19,200 steps, matched batch 4 x 8.
- Final dev WER (%): {"gigaspeech2_vi/validation": 34.24, "fleurs_vi/validation": 71.53, "vimd/validation": 62.2, "lsvsc/validation": 16.45}.
- Test benchmark WER (%): {"gigaspeech2_vi": 48.37, "fleurs_vi": 80.28, "vimd": 58.36, "lsvsc": 19.02, "phoaudiobook": 19.89}.
- Duration (training + dev evaluation, seconds): 49722; compare both arms before choosing a backbone.
- Logs/configs/queue state: `results/vi_ab_20261031/`; detailed benchmark: `results/mel_20261031-135333_vi500h_lfm_projE2_matched19k`.

## 2026-11-02 - Vietnamese backbone A/B qwen completed
- Run `20261101-051531_vi500h_qwen_projE2_matched19k`: selected 500 h; E2 projector + fresh LoRA; 19,200 steps, matched batch 4 x 8.
- Final dev WER (%): {"gigaspeech2_vi/validation": 25.42, "fleurs_vi/validation": 42.99, "vimd/validation": 33.99, "lsvsc/validation": 11.35}.
- Test benchmark WER (%): {"gigaspeech2_vi": 35.48, "fleurs_vi": 44.35, "vimd": 31.65, "lsvsc": 11.88, "phoaudiobook": 11.64}.
- Duration (training + dev evaluation, seconds): 74023; compare both arms before choosing a backbone.
- Logs/configs/queue state: `results/vi_ab_20261031/`; detailed benchmark: `results/mel_20261101-051531_vi500h_qwen_projE2_matched19k`.

## 2026-11-02 - Vietnamese backbone A/B comparison complete
- qwen has lower GigaSpeech2-vi test WER and is the preferred backbone for the next stage.
- GigaSpeech2-vi WER (%): LFM 48.37, Qwen 35.48.
- Full test scores, run IDs and runtime: `results/vi_ab_20261031/comparison.md`.

## 2026-11-04 - Vietnamese A/B convergence check (no new run)
- Both arms saw identical data per step: batch 4 x accum 8 = 32 utterances, same seed, epoch 2.968 at step 19,200. Step count equals sample count; LFM only has longer *target text* (~3 vs ~1.2 tokens/word).
- Qwen reaches LFM's final dev WER at ~6-9k steps (~1/3-1/2 of the samples). GigaSpeech2-vi dev: Qwen 36.8 @6k, 32.0 @9k vs LFM 34.2 @19.2k. 4-set mean: Qwen 39.2 @9k vs LFM 46.1 final.
- Wall clock is also faster: Qwen @9k = 9.7 h vs LFM 19.2k = 13.7 h (training elapsed).
- Late trend, 14k->19.2k: Qwen giga 26.2->25.4, mean 32.1->28.4 (near plateau); LFM giga 39.4->34.2, mean 54.0->46.1 (still dropping).
- Cosine LR is down to 0.1x, so the late flattening is partly the schedule, not a data limit. Train loss drops at epoch boundaries (6.5k, 13k), a sign of memorizing the 500 h set.
- Continuing the same 500 h is unlikely to help Qwen much. More or new data is the lever. Per-token train loss is not comparable across tokenizers, so the comparison uses dev WER (200 utt/set, noisy; FLEURS swings ±20).

## 2026-11-04 - Feasibility: graft LFM3 tokenizer onto LFM2.5-230M + prune to en/vi (no code change)
- LFM3-1.7B-Base tokenizer (pinned rev 545fc8f) is an in-place expansion of LFM2.5's: same pre-tokenizer, all 63,683 LFM2.5 merges included, all 63,893 regular LFM2.5 tokens present (IDs shifted by -501; specials moved to the end). 125,033 tokens total.
- On normalized training targets (1.87M en + 304K vi utterances): vi 3.23 -> 1.14 tokens/word; en unchanged (1.11 -> 1.10).
- Merge-closed pruned vocab (tokens emitted on en+vi targets, all their BPE ancestors, and all bytes) = 37,347. LFM2.5 has 64,402, so tied embeddings drop from ~66M to ~38M params.
- 9,719 kept tokens are new to LFM2.5 and need initialization. Only 2,096 of them are emitted on vi text, but they cover 78% of vi target tokens.
- Script and output: `results/tokenizer_graft_20261104/usage.{py,txt}`. Not built or trained yet; awaiting a go/no-go on the plan.

## 2026-11-04 - Parked: LFM3 tokenizer graft + en/vi pruning
- User decision: park the idea. Write-up and revival recipe in `docs/idea_lfm3_tokenizer_graft.md`.

## 2026-11-04 - Main goal back to adaptive token merging; Phase 0 training-free sweep on E2
- User decision: Vietnamese track parked. The main research is adaptive audio-token merging against the fixed-rate E2 baseline. Plan: `docs/adaptive_merging_plan.md`.
- Code: new `audiollm/merging.py`, applied after the projector in `SpeechLLM.build_inputs`.
  - Config `MERGE` = none|affinity|uniform|uniform_matched, plus `MERGE_TAU`, `MERGE_WINDOW`, `MERGE_RATIO`. Default none, so existing behaviour is unchanged.
  - Affinity merging is Affinity Pooling (arXiv 2604.06871) on projector outputs. `uniform_matched` uses the same per-utterance token count, evenly spaced.
  - Unit-tested against a reference Algorithm 1; batched output equals unbatched; gradients flow through the pooling.
- Sweep (no training) on E2, 8 dev sets x 200 utterances. Full table: `results/merge_sweep_20261104/summary.md`.
  - No merge: 39.85 dev mean at 12.58 tok/s (reproduces E2).
  - Dev mean WER, affinity vs `uniform_matched` at equal tok/s:
    - keep 0.87 (10.9 tok/s): 39.77 vs 47.87
    - keep 0.82: 40.60 vs 50.70
    - keep 0.75: 44.94 vs 56.48
    - keep 0.61: 57.47 vs 68.58
  - Plain uniform keeping 0.75: 57.91.
  - Affinity is better on every set at every budget. Up to ~18% fewer tokens costs at most +0.75 WER, without retraining.
- Adjacent cosine similarity of projector outputs is low (median ~0.33), so useful thresholds are 0.4-0.7, not the paper's 0.8-0.9. Window ω=3 does not help (matches the paper's input-layer result).
- Confound: at τ=0.7, 30-85% of the removed tokens lie in the quietest 20% of tokens (`where.txt`), so mild merging is mostly silence collapse. At τ=0.4-0.5 removal spreads into speech.
- Next: a silence-aware fixed-rate control (collapse low-energy runs, uniform elsewhere, same budget). Then Phase 1: merge-aware fine-tune (affinity / uniform_matched / no-merge control).

## 2026-11-04 - Probe: audio-token redundancy vs LLM depth on E2 (no training)
- Context: the user is reframing toward a research paper; the question is whether encoder-free and merging connect.
- Measured the median adjacent cosine similarity of audio positions at each of the 15 hidden states (embeddings + 14 layers), 64 utterances x 8 dev sets. Script: `results/layer_redundancy_20261104/probe.py`.
- Mean-centered per utterance: input 0.19-0.28 -> last layers 0.46-0.53 on every set, roughly doubling. Most of the rise is at layers 1, 6, 8 and 9; the last ~5 layers plateau. Uncentered: 0.33-0.42 -> 0.54-0.61.
- Reading: the input is not redundant (it is acoustic) and redundancy grows inside the LLM, consistent with the lower LLM layers doing the encoder's job.
- The effect is modest next to Affinity Pooling's "near-identical within a word" deep tokens. Single checkpoint; medians only; no boundary alignment yet.

## 2026-11-04 - Parked: cross-language token compression ratio (supervisor's suggestion)
- The user recalled a supervisor suggestion: compare token compression ratios between languages (English vs Vietnamese). This likely explains why the Vietnamese pipeline exists, so Vietnamese is not abandoned. It is a future comparison axis for the merging research, not a separate track.
- Parked for now. When revived: run the same merging sweeps / learned-rate models on en and vi, and compare kept tokens per second and per word/syllable.
- Assets already exist: the vi 230M (`20261031-135333_vi500h_lfm_projE2_matched19k`) and Qwen3-0.6B (`20261101-051531_vi500h_qwen_projE2_matched19k`) runs, vi dev/test sets, the vi normaliser.
- Plan section added to `docs/adaptive_merging_plan.md`.

## 2026-11-04 - Novelty search: encoder-free analysis, learned in-LLM merging, cross-language rate

Search used three Gemini 3.8 Flash threads plus my own searches. All ids were checked on the arXiv API. Details
and the table are in `docs/literature_review.md` ("Novelty search 2026-11-04").
- **Our framing title is taken.** arXiv 2610.24239 (Harwath, Livescu, Khudanpur et al., Oct 16 2026) is
  "Encoder-Free Multimodal LLMs Learn Implicit Encoders", on Gemma-4 12B and Inkling-Small.
  - Early layers (up to about 12 of 48) act as an encoder. Phonetic content peaks at layer 8, lexical at layer 9.
  - The paper has no redundancy, merging, or multilingual analysis.
- **Still open:** learned merging *inside* a speech LLM. All in-LLM methods are training-free (FastAdaSP,
  AudioTETRIS, Affinity Pooling). All learned-rate methods sit before the LLM (CIF-PT, ACIF, MoE+CIF 2606.10439).
  No encoder-free model has any adaptive rate.
- **Cross-language:** no learned en/vi rate comparison was found.
  - Whisper-ToMe (2609.13151) finds redundancy profiles that are the same across 16 languages (std ≤ 0.03),
    with a static rate. This is the null result to beat.
  - Coupé 2019 gives the syllable-rate hypothesis.
- Gemma-4 12B is open-weight and encoder-free for audio (25 Hz raw waveform). It is a possible strong model for
  training-free analysis, but needs quantising for our 16 GB GPU.
- New framing: build on 2610.24239. Contribution = a learned CIF-style merge at the end of the implicit encoder,
  plus redundancy-across-depth analysis, plus the en/vi learned-rate comparison.

## 2026-11-04 - Scope: stay on ASR; park ASR-vs-translation; next = in-LLM training-free merge
- The user asked (in a Gemini chat) whether to expand beyond ASR, since speech LLMs do other tasks too.
- Gemini suggested ASR plus en↔vi speech translation as a contrast (acoustic vs semantic compressibility).
  - Parked, not adopted: our models are ASR-only, en↔vi parallel speech is scarce (FLEURS), and it adds a second
    metric before the main result exists.
- Gemini's roadmap (Whisper-large-v3 + Qwen-2.5-7B) is rejected: encoder-based, duplicates arXiv 2606.10439,
  and too large for 16 GB. It exists only in Gemini's private T3 artifact folder; nothing in the repo was
  changed by it.
- Next step: training-free merge after LLM layer 4/6/8 in E2, against `uniform_matched`, to locate where the
  learned merge should go. Details: `docs/adaptive_merging_plan.md` ("Next step").

## 2026-11-05 - In-LLM Training-Free Token Merging: implementation, verification gates, and sweep
- Authored formal specification `docs/in_llm_merging_plan.md` for training-free intermediate layer merging inside LFM2.5-230M. Iteratively reviewed and approved by Claude Opus 5.5 ("Verdict: approved for execution").
- Implemented exact no-cache greedy decoding engine in `audiollm/layer_merging.py` (`greedy_decode_with_layer_merge`), supporting layer range splitting $[0, k) \to \text{merge} \to [k, 14)$ with causal/conv mask reconstruction, position ID recalculation, and RoPE offset updates.
- All 4 verification gates passed (`results/merge_in_llm_20261105/verify_gates.py`):
  - Gate 0 (fp32 split/rebuild identity): 100.0% match across $k \in \{2, 4, 6, 8, 10\}$ (16/16 utterances).
  - Gate 3 (fp32 batch invariance under merging): 100.0% match between bs=1 and bs=16 at $k=4$.
  - Gate 1 (bf16 unmerged dev mean WER): 39.87% (target $39.85 \pm 0.15\%$).
  - Gate 2 (bf16 Phase 0 reproduction at $k=0, \tau=0.70$): 39.76% (target $39.77 \pm 0.15\%$).
- Disjoint calibration completed on 512 held-out utterances (`results/merge_in_llm_20261105/calibration.json`). All target effective token ratios $E \in \{0.93, 0.87, 0.80, 0.75\}$ match within $|R_{calib} - R_{target}| \le 0.01$.
- Floor controls evaluated: $R=0$ (`all_dropped`) collapses WER to 97.9–102.2% across all $k \in \{2, 4, 6, 8, 10\}$, confirming strong reliance on intermediate audio representations.
- Full sweep completed (54 runs across $k \in \{0, 2, 4, 6, 8, 10\}$ and $E \in \{0.93, 0.87, 0.80, 0.75\}$; `results/merge_in_llm_20261105/results.jsonl`, `summary.md`, `report.md`):
  - In-LLM merge at Layer $k=2$ (post-conv) is the global optimum at $E=0.93$: 39.35% dev mean WER (-0.33% vs baseline 39.68%, -0.21% vs projector $k=0$ 39.56%). On `dev-clean`, $k=2$ drops WER by -1.26% (20.28% vs 21.54%).
  - Identical attention KV savings: because all 6 self-attention layers are at layers 2, 4, 6, 8, 10, 12, merging at $k=2$ reduces KV cache across all attention layers, matching $k=0$.
  - Acoustic affinity outclasses uniform decimation across all operating points (WER gap up to -11.58%).
  - At aggressive compression ($E \le 0.80$), input-level merging ($k=0$) is superior because downstream sequence starvation ($R_k = \frac{14E-k}{14-k}$) hits deeper layers severely ($k=6 \to 57.64\%, k=8 \to 86.13\%, k=10 \to 100.46\%$).



## 2026-11-05 - Model Convergence Training Plan & Execution
- User request: Train model to convergence on current dataset first, as error rate is too high (~40% dev mean WER) before doing further experiments.
- Baseline audit: English E2 baseline (`runs/20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k`) was severely under-trained: only 19,200 steps (epoch 0.3528), leaving 64.7% (~2,060 h / 1.17M utts) of the 3,180 h multi-corpus dataset completely unseen. Train loss was still actively descending ($0.83 \to 0.57$) when the cosine schedule annealed to the floor. In contrast, Vietnamese 500 h already finished 2.97 epochs and plateaued.
- Authored formal specification `docs/model_convergence_training_plan.md` detailing continuation budget (38,400 steps, cumulative 57,600 steps $\approx 1.03$ epochs of the 3,180 h pool), LR schedule (1e-4 peak, 500-step warmup, cosine to 1e-5), and 3 verification gates.
- Reviewer Claude Opus 5.5 (`claude -p`) verified and approved the plan ("APPROVED WITH MINOR FIXES"), recommending seed offset (`--seed 1`) for uniform data pool sampling and defining Gate 1 finite loss bounds. All suggestions incorporated.
- Verification gates validated via 10-step smoke test (`converge38k_smoke10`): Gate 0 (132/132 LoRA + projector weights loaded cleanly), Gate 1 (step 5 loss 0.396, step 10 loss 0.557, finite grad norms 1.7–2.9), Gate 2 (tuned `--eval_batch_size 16` to maintain VRAM $< 7.0$ GB). Smoke run artifacts pruned.
- Full 38,400-step convergence run launched: `runs/20261105-100647_mix3200h_conv_stack8_cont5_converge38k`.
