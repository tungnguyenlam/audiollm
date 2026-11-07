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



## 2026-11-05 - Model Convergence Training Plan, Claude Review & Stage A Execution
- User request: Train model to convergence on current dataset first, as error rate is too high (~40% dev mean WER) before doing further experiments.
- Baseline audit: English E2 baseline (`runs/20261030-153841_mix3200h_giga_spgi_conv_stack8_cont4_19k`) was severely under-trained: only 19,200 steps (epoch 0.3528), leaving 64.7% (~2,060 h / 1.17M utts) of the 3,180 h multi-corpus dataset completely unseen. Train loss was still actively descending ($0.83 \to 0.57$) when the cosine schedule annealed to the floor. In contrast, Vietnamese 500 h already finished 2.97 epochs and plateaued.
- Authored formal specification `docs/model_convergence_training_plan.md`. Reviewed with Claude Opus 5.5 (`claude -p`).
- Incorporated Claude's recommendations:
  - Added `SHUFFLE_SEED_OFFSET` to `audiollm/config.py` and `audiollm/trainer.py` so that `seed = C.SEED + C.SHUFFLE_SEED_OFFSET + epoch` produces a 100% fresh batch permutation across the 1.78M utterance pool while strictly preserving `SEED=0` for dev set sampling (`VAL_N=200`) and MVN statistics.
  - Staged Convergence Protocol: Replaced ad-hoc step budget with Stage A (1 full epoch = 55,732 steps) with 1,000 warmup steps, cosine decay to 1e-5. Full 7-set test benchmark at conclusion determines whether Stage B (second epoch with peak LR 5e-5) is warranted.
  - Removed early stopping rule to ensure full cosine weight annealing.
  - Disabled redundant `shuffle` audio ablation (`--eval_audio_ablations ""`) to save ~2 hours of decoding time.
- Pre-flight Gate Verification:
  - Gate 0: LoRA (132/132 tensors) and Conv projector loaded with 0 missing/unexpected keys. MVN parameters identical.
  - Gate 1: Verified step-1 full 8-dev-set reproduction against E2 step-19200 within $\pm 0.3\%$ WER (dev-clean 21.13% vs 21.49%, dev-other 34.58% vs 34.99%, AMI 41.25% vs 41.48%, Fleurs 67.26% vs 67.35%, VoxPopuli 37.70% vs 37.18%, Tedlium 29.24% vs 30.19%, GigaSpeech 48.89% vs 49.04%, SPGISpeech 37.53% vs 37.06%).
  - Gate 2: Tuned `--eval_batch_size 16` avoiding allocator warnings; memory steady at peak 6.7 GB.
- Execution: Active Stage A run `runs/20261105-111014_mix3200h_conv_stack8_stageA_55k` (55,732 steps, ~8.2 h wallclock, ~400x realtime).
  - Predecessor run (`20261105-101022`) reached step 4,240 before being killed when the transient T3 ACP systemd scope (`t3code-acp-*.scope`) was recycled at 10:53.
  - Active run launched in user terminal session (`session-16.scope`, PID 503131 on pts/13), immune to agent scope teardowns.
  - Active throughput: ~0.50–0.52 s/step (~400x realtime), peak VRAM 6.7 GB, dev eval every 2,000 steps, checkpoint every 1,000 steps.

## 2026-11-05 - Parked: merge-aware fine-tune with in-LLM merge at layer 2
- Proposal reviewed: after Stage A, fine-tune with fixed affinity merging after LLM layer 2 to test whether adaptation recovers WER. Then move to a learned merge policy.
- Verdict: worth doing as Phase 1 of `docs/adaptive_merging_plan.md` at k=2, not as a separate track. Parked until Stage A is benchmarked.
- Caveats recorded in the plan ("Parked: merge-aware fine-tune with the in-LLM merge"):
  - the k=2 optimum is one under-trained-E2 run at E=0.93, so re-sweep on Stage A first;
  - k=2 is only two conv layers past the input, with the same KV saving as k=0;
  - fine-tune at E ≈ 0.6/0.4 with `uniform_matched`, no-merge and k=0 arms;
  - k>0 training needs a new grad-enabled split forward, because `layer_merging.py` only supports inference.
- No code changed.

## 2026-11-06 - Prepared Stage C continuation command
- User requested a command to launch manually in tmux after Stage B benchmarking finishes; no training launched.
- Proposed Stage C: initialize from final Stage B weights (`20261105-211932_mix3200h_conv_stack8_stageB_55k`), 55,732 steps, peak LR 2.5e-5, cosine floor ratio 0.1, warmup 1,000, shuffle offset 3000; inherit remaining Stage B settings.
- Stage B final 8-set dev mean WER 30.31% versus Stage A 32.25%; late Stage B validation fluctuates, so further gains remain uncertain.

## 2026-11-07 - Stage C result; Stage C becomes the baseline
- Run `runs/20261106-101155_mix3200h_conv_stack8_stageC_55k` (from Stage B, 55,732 steps, peak LR 2.5e-5) finished and was benchmarked.
- Final 8-set dev mean WER 29.12% (Stage B 30.31%, Stage A 32.25%).
- 9-set test mean WER: A 41.85 → B 40.13 → C 39.26. Per set, C vs B: LS-clean 11.87 (12.68), LS-other 32.73 (33.32),
  VoxPopuli 28.87 (29.80), AMI 35.93 (36.90), Earnings22 53.38 (54.53), GigaSpeech 33.23 (33.89), SPGI 24.07 (25.09),
  FLEURS 58.51 (59.56), EdAcc 74.76 (75.39).
- Decision: stop adding epochs. Each epoch gains about half the previous one (-1.7, then -0.9), so a Stage D would give about -0.4
  for ~8 h. Stage C replaces E2 (`20261030-153841`) as the baseline for all merging work.
- Next: rerun the training-free in-LLM merge sweep on Stage C, then choose k for the Phase 1 merge-aware fine-tune
  (`docs/adaptive_merging_plan.md`, "Parked: merge-aware fine-tune").

## 2026-11-07 - Launched training-free in-LLM merge sweep on Stage C
- `results/merge_in_llm_stageC_20261107/` is a copy of the 2026-11-05 sweep scripts. Checkpoint and output dir now come from
  `MERGE_RUN_DIR` / `MERGE_OUT_DIR` (defaults: Stage C and this dir). The merge code (`audiollm/layer_merging.py`) is unchanged.
- Grid: k ∈ {0,2,4,6,8,10} × E ∈ {0.93,0.87,0.80,0.75}, the same as before so results compare directly. Added E=0.60 and 0.40
  to measure the training-free starting point for the Phase 1 fine-tune. Pairs where the keep ratio after merging,
  R=(14E-k)/(14-k), falls below 0.2 are skipped (added pairs: k0/k2 at 0.6 and 0.4, k4/k6 at 0.6).
- Fresh 512-utt disjoint calibration. Sanity check: the `none` baseline should be within about 0.3 of the trainer's 29.12% dev mean.
- tmux session `merge_stageC`, log `results/merge_in_llm_stageC_20261107/sweep.log`. Expected runtime ~3.5 h.

## 2026-11-07 - Stage C merge sweep: sanity checks pass
- Unmerged (`none`) dev mean WER on Stage C is 29.23%; the trainer's final dev eval gave 29.12%, so the gap is +0.11, inside the ±0.3 tolerance.
- Calibration (`results/merge_in_llm_stageC_20261107/calibration.json`) hits every feasible (k, E) target within 0.01 on R.
- When every audio token is dropped (`all_dropped`), WER is 98.3–101.3% at k=2..8, so decoding still depends on the audio.
- The sweep keeps running in tmux `merge_stageC`.
