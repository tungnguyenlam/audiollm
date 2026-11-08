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

## 2026-11-07 - Stage C merge sweep was killed at 09:48; resumed at 22:08
- It did not finish. At 09:48 the agent's systemd scope was stopped, and the tmux server died with it because it had been
  started from inside that scope. Only k=0 finished (E 0.93/0.87/0.80/0.75, 14 rows in `results.jsonl`).
- Partial k=0 results on Stage C (affinity vs uniform_matched, dev mean WER; unmerged baseline 29.23):
  E=0.93 29.21 vs 34.37, E=0.87 29.48 vs 38.27, E=0.80 32.57 vs 43.66, E=0.75 36.77 vs 48.85.
  Affinity at E=0.93 is free, at 0.87 it costs +0.25, and the knee is between 0.87 and 0.80.
- Resumed with `systemd-run --user --scope --unit=merge-stageC-tmux` + tmux `merge_stageC`. The runner skips completed rows.
  25 (k, E) pairs are left (50 runs), ~2.5-3 h.
- AGENTS.md "Long-running jobs" now says to launch tmux via `systemd-run --user --scope`.

## 2026-11-08 - Pivot to Vietnamese ASR for ZAC 2026 (until 11-15); LFM3 en/vi tokenizer graft
- Merging research paused until 2026-11-15 for the Zalo AI Challenge LifeMate track (repo /home/nguyenlt/ZaloAI: ASR -> Qwen3.5-4B
  agent, 6B cap). Target: a vi ASR that holds up on N/C/S accents and beats the Zipformer-30M baseline.
- Parked (not feasible on the 230M model by 11-15): thinking mode (the Base model has none), looped/Dory-style depth recurrence,
  on-policy distillation from LFM3-Audio-2B (the graft keeps LFM3 ids in graft_meta.json `lfm3_ids` so it can be done later).
- `tools/build_vi_graft.py`: LFM2.5-230M + LFM3 tokenizer (same as LFM3-Audio-2B) pruned to en/vi/digits/punct, merge-closed:
  45,755 vocab (vs 64.4k LFM2.5), 10,477 new rows initialised as the mean of their LFM2.5 sub-tokens. vi 3.23 -> 1.14 tok/word
  (1.19 on ZAC). Tokenization matches LFM3 on 8999/9000 texts.
- `tools/warmup_graft_embeddings.py`: text-only training of the new rows only (1500 steps): vi 1.119 nats/char vs 1.270 for LFM2.5;
  English unchanged (2.659 vs 2.675 nats/tok). Log: results/vi_zac_20261108/warmup.log.
- ZAC data (`tools/prepare_zac_asr.py`): train/dev(864)/holdout/public_test manifests with Zipformer hyps; 871 confusions mined
  from train only. Zipformer on ZAC dev (our normaliser): 18.56% WER (N 15.95, C 22.86, S 19.72).
- Old LFM vi run on ViMD test: 58.36 (N 53.87, C 60.54, S 61.92). Qwen vi run: 31.65 (N 27.54, C 36.19, S 32.46).
  Per-region scorer: `tools/vi_region_wer.py`.

## 2026-11-08 - vi_zac_graft run: augmentation + prompt-triggered ASR correction; launched
- Code: WaveAugment (speed 0.9/1.0/1.1, ESC-50+Clotho noise bank 7.8 h at SNR 5-25 dB p 0.3, gain +-6 dB; `tools/build_noise_bank.py`);
  correction mode (CORRECTION_P), prompt "ASR hypothesis:{hyp}\nCorrected transcription:", only when the prompt asks for it.
  Hyps: real Zipformer (p 0.7 where available), else synthetic `audiollm/asr_noise.py` (mined confusions + dialect initial/final/tone
  swaps), with 15% clean. Val/test splits with a `#corr` suffix evaluate correction mode.
- Trainable fp32 tied embedding (TRAIN_EMBEDDINGS, lr x0.25, saved as embed.pt). The loss now projects only target positions through
  lm_head. train.py accepts INIT_FROM whose MODEL is the graft's base.
- Smoke run 20261108-092603_vi_zac_graft_smoke_smoke30: train/ckpt/eval OK in both modes; reload is exact.
- Launched `configs/vi_zac_graft.json` (INIT_FROM vi500h_lfm_projE2_matched19k; mix500h + gigaspeech2/fleurs val + zac2026/train x10
  + vimd x2; 16x2, 24k steps, lr 2e-4) at ~1.0 s/step, ETA ~7 h. Running in tmux `-L vi_zac -s vi_zac_graft` under systemd scope
  vi-zac-graft2. Log: results/vi_zac_20261108/train.log.
- Launch gotcha: the login shell is fish, so a tmux command with `source .venv/bin/activate` dies silently. Wrap it in `bash -c`.

## 2026-11-08 - vi_zac_graft completed; continuation assessment
- Run `20261108-093006_vi_zac_graft_aug_corr` completed 24,000 steps in 5:57:07.
- Fixed 300-utt ZAC dev ASR best 39.20% WER at 12k; final 41.56%. Correction best 16.33% at 6k; final 23.35%.
- Final full 864-utt dev: ASR 43.64%, correction 24.37%; 619-utt holdout: ASR 57.37%, correction 34.96%.
- Assessment: further unchanged training is unsupported by the late validation trend. Recommend full-dev evaluation of earlier
  checkpoints and untouched Zipformer hypotheses on identical scoring before choosing a targeted follow-up; no training launched.
- Evidence: run metrics.jsonl, test_results.json, and results/vi_zac_20261108/train.log. Subset and full-dev WER are not interchangeable.

## 2026-11-08 - Early ZAC checkpoint comparison blocked; matched prediction audit
- Requested full-dev 6k/12k evaluation cannot run: intermediate weights were overwritten by checkpoint rotation; completion deleted checkpoint/. HUB_REPO was empty; local Programming/HF-cache search found final/smoke ZAC adapters only.
- Run `20261108-093006_vi_zac_graft_aug_corr`: rescored saved outputs with identical Vietnamese normalization, unique IDs, exact coverage, and reference-equality checks against dev manifest.
- Full dev (864): untouched Zipformer 18.56% WER; final ASR 43.64%, correction 24.37%.
- Matched subset (300): Zipformer 18.87%; ASR 6k/12k 42.84/39.20%; correction 6k/12k 16.33/19.92%.
- 6k correction improves the subset by 2.54 points but cannot establish full-dev superiority. No full-dev early-checkpoint winner is claimed.
- Decision: keep Zipformer baseline; recover external early weights or use a future run retaining checkpoints. No training launched; no training defaults changed.
- Reproducible scorer and detailed report: [audit](results/vi_zac_checkpoint_audit_20261108/report.md).

## 2026-11-08 - Independent Sol 6.1 medium audit verification
- PASS for accuracy of the matched-prediction audit and transparency of its missing-weights blocker; original full-dev 6k/12k evaluation remains incomplete.
- Scorer rerun reproduced all WERs; separately confirmed identical 300-ID early subsets, inspected checkpoint retention, and repeated local weight search.
- Details: [verification](results/vi_zac_checkpoint_audit_20261108/verification.md). No training, inference, or default changes.

## 2026-11-08 - Public per-run Hugging Face backups and portable quickstart
- User requested public backups, one repository per training run, with training information/progress in README.
- Decision: new default HUB_REPO=auto resolves authenticated account/audiollm-<run_id>; custom user/prefix gets the run id suffix; empty disables uploads. Configured backups enable 500-step checkpoints if unset.
- Replace rotating-only remote mirror with immutable upload snapshots and retained checkpoints/step-N/ including optimizer/scheduler/RNG; keep checkpoint/ after completion and upload final weights before test evaluation.
- Bundle inference/training source, requirements, MVN and local base/tokenizer; add scripts/hf_backup.py (push/selective pull/revision/resume) and scripts/transcribe.py (audio resampling, CPU/CUDA, correction).
- Publish root model card with configuration, progress, WER and artifact links at every upload; retry uploads three times and record local hub_status.json instead of falsely reporting final success.
- Validation so far: three offline integrity/retry/restore tests pass; CLI help and compile checks pass. No training or WER experiment launched.

## 2026-11-08 - Full Vietnamese continuation with rotating 20% English replay
- User requested continuation on all Vietnamese training data with 20% English cycling through the entire English catalog.
- Added REPLAY_SPLIT/REPLAY_FRACTION: each epoch covers all primary examples and takes the next seeded English slice,
  exhausting the catalog before reshuffling. Ratio is by utterance count over each epoch, not audio hours or each batch.
- English targets use English normalization; English replay bypasses Vietnamese synthetic correction. Existing Vietnamese
  augmentation/correction stays enabled. No dev/test manifests are added to training; GigaSpeech2 VI has no local train split.
- Config: configs/vi_full_en20.json; starts from final vi_zac_graft weights (24k), fresh optimizer, LR 5e-5, warmup 1000,
  shuffle offset 4000. Removed ZAC x10 / ViMD x2 weighting; added FLEURS VI train for full available unweighted VI coverage.
- Eligible catalog: 277,723 VI + 1,741,670 EN; 69,431 EN examples per VI epoch; 26 epochs / about 282,074 steps to cover EN once.
- Run: 20261108-162118_vi_full_en20_cycle; tmux -L vi_en20, session vi_en20; systemd vi-en20-cycle-tmux.scope verified outside t3code.
- Log: results/vi_en_replay_20261108/train.log; counts: results/vi_en_replay_20261108/catalog.json. WER pending.
- Verification: deterministic replay/coverage across cycle boundary, epoch batching remapping, English normalization and correction
  exclusion passed; persistent regression tests in tests/test_replay.py. Estimated runtime ~3 days, pending observed throughput.
- Follow-up decision: enable auto backups in checked-in training presets as well; old saved run configs still honor their explicit setting.
- Real upload: completed run 20261108-093006_vi_zac_graft_aug_corr backed up publicly to https://huggingface.co/tungnguyenlam/audiollm-20261108-093006_vi_zac_graft_aug_corr at bb6f02f2a0b76d0565e6895873eb84573fe1fb1d; public visibility verified via Hub API. No lost intermediate weights were recovered.

## 2026-11-08 - Training catalog audio-hour audit
- Manifest durations for vi_full_en20: VI 728.89 h raw / 722.15 h eligible; EN 3197.71 h raw / 3169.36 h eligible (1–30 s).
- Counts exclude validation/test and do not multiply by replay/epochs. PhoAudiobook directory name says 200h but its train
  manifest totals 421.38 h; reporting manifest durations rather than directory labels.
- Detailed per-corpus totals: results/vi_en_replay_20261108/duration_totals.json. No training settings changed.
- Final validation: four backup tests pass; downloaded final bundle at the pinned Hub commit, verified SHA-256 of adapter/projector/embed/MVN/base/tokenizer, and ran standalone inference from /tmp with exact local transcript agreement. No new WER measurement. Evidence: [report](results/hf_backup_20261108/report.md).

## 2026-11-08 - Restarted English replay continuation with 1.5x microbatch
- User authorized stopping and continuing with batch size 24 (from 16), GRAD_ACCUM stays 2: effective batch 48 vs 32.
- Stopped run 20261108-162118_vi_full_en20_cycle with SIGINT at step 314, before its first checkpoint; discarded 4m14s of updates.
- Restart initializes from the completed 24k Vietnamese run, retaining the full 26-pass replay plan and learning rate 5e-5.
- configs/vi_full_en20.json now uses RUN_NAME vi_full_en20_cycle_b24, BATCH_SIZE 24, WARMUP_STEPS 667 (similar example exposure).
- Epoch-derived budget automatically falls from 282074 to about 188058 steps; no sampler cursor/optimizer state is reused across
  different batch layouts. 80/20 utterance ratio and complete English catalog coverage are preserved.
- Detached tmux -L vi_en20_b24 / session vi_en20_b24, systemd vi-en20-b24-tmux.scope.
- Log: results/vi_en_replay_20261108/train_b24.log. Startup/data loading underway; throughput and WER pending.
- Initial b24 launch 20261108-162654 failed before training: concurrent Hub auto-backup default requested whoami in offline mode.
  Relaunched with explicit --hub_repo "" to preserve the original local-only job behavior, socket vi_en20_b24_retry,
  scope vi-en20-b24-retry-tmux; log results/vi_en_replay_20261108/train_b24_retry.log.

## 2026-11-08 - Batch 32 x accum 3 relaunch; precision speed test
- User asked for BATCH_SIZE 32, GRAD_ACCUM 3 (effective 96, was 48) because the GPU looked underused. Stopped b24 run
  20261108-162722 at step ~1700 (checkpoint at step 1000 not reused, since the batch layout changed). WARMUP_STEPS 667 -> 333
  (same number of warmup examples), LR unchanged at 5e-5, RUN_NAME vi_full_en20_cycle_b32a3.
- Speed test (b32 x a3, 400-step smoke runs killed after 6 perf lines; logs in results/vi_en_replay_20261108/speed/):
  amp ~2.4 s/step, ~40 utt/s; fp32 (fp32 weights and compute) ~20 s/step, ~4.7 utt/s (~8.5x slower);
  PRECISION=bf16 crashes at step 0 (bf16 vs fp32 matmul dtype mismatch when initializing from INIT_FROM); amp without
  GRAD_CKPT OOMs at b32 on 16 GB.
- Finding: b24 x a2 already ran at ~40 utt/s, so the larger batch does not raise throughput. rocm-smi showed 95% busy;
  the run was compute-bound, and the unused VRAM did not mean idle compute. Pad efficiency ~74% is the visible slack.
- Pure bf16 rejected: Adam steps of size ~lr (5e-5, decaying to 5e-6; embeddings 1.25e-5) fall below the bf16 rounding
  threshold for weights larger than ~0.01 in magnitude. Larger batches lower gradient noise but not the step size, so
  they do not fix this. amp already runs matmuls in bf16.
- Relaunched: tmux -L vi_en20_b32a3, scope vi-en20-b32a3-tmux; log results/vi_en_replay_20261108/train_b32a3.log.
  ~2.45 s/step, ETA ~65-70 h (b24 run: ~60 h).
