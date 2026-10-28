# Idea (parked): loss-based soft data pruning

Status: not implemented. Parked 2026-10-28; revisit once the model and data mix are fixed, or as a cheap test on
the 100 h config. Origin: discussion of BLS (arXiv:2604.04681), see "Why not BLS" below.

## The idea

Lower the chance that well-learned utterances are trained on in an epoch, without removing them for good
(InfoBatch-style). Each epoch:

- Score = latest per-utterance loss (mean token CE over the transcript), recorded for free during the forward pass.
- Utterances whose score is below their **corpus** mean are skipped with probability `r` (e.g. 0.5).
  The ones kept get loss weight `1 / (1 - r)` so the expected gradient is unchanged.
- Never-seen utterances are always kept. No pruning for the last `PRUNE_ANNEAL` (~12.5%) of steps.
- A skipped utterance keeps its old score until it is drawn again (it is still drawn with probability `1 - r`),
  so nothing is removed permanently.

The selection costs nothing measurable (one O(N) pass per epoch); the saving is the forward + backward + feature
loading of every skipped utterance.

## Expected payoff

- ~3-epoch runs (544 h+, 19.2k steps at 6384 steps/epoch): epoch 1 has no scores, so only ~14% compute saved.
  Not worth it there.
- 100 h config (`ls100h_conv_stack8_12768`, ~14 epochs): close to the regime where InfoBatch reports 20-40%.
  Best testbed. Caution: low-loss utterances there are the memorised ones; skipping them may help or hurt.

## Implementation sketch

1. `audiollm/model.py` `forward`: per-utterance CE from logits (`reduction='none'`), return it in `aux`; training
   loss = `sum_i w_i * sum_tokens(CE) / total tokens` (all `w_i = 1` reproduces today's loss exactly).
2. `audiollm/data.py`: add `"index"` to `SpeechDataset.__getitem__` and to the keys `collate_speech_batch` keeps.
3. New `audiollm/selection.py`: `scores` tensor (NaN = unseen); `soft_prune(scores, rng, r)` -> kept indices + weights,
   threshold = mean per corpus (a global mean would prune LibriSpeech and keep AMI, i.e. the rebalancing that lost).
4. `audiollm/trainer.py`: `make_epoch_sampler` batches the kept subset; epoch loop runs until the step budget instead
   of a precomputed `num_epochs` (pruned epochs are shorter); anneal off at the end.
5. Checkpoint: save `scores` and the current epoch's batch list (the sampler is no longer a pure function of
   `SEED + epoch`).
6. Config: `PRUNE_RATIO = 0` (off), `PRUNE_ANNEAL = 0.125`. Log kept fraction per corpus per epoch; dump
   `runs/<id>/utt_loss.npy` at the end (also useful on its own to inspect the high-loss tail for bad transcripts).

Test: CPU smoke run with `PRUNE_RATIO = 0` gives identical per-step losses to the current code.

## Experiment

Three 100 h arms at equal steps: baseline / soft pruning on real loss / soft pruning on random scores.
Keep it only if real-loss pruning matches baseline WER with fewer samples **and** beats random scores
(the control the BLS paper skipped).

## Literature (quotes checked against the PDFs)

- Qin et al., *InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning*, ICLR 2024, arXiv:2303.04947.
  Abstract p.1: "On CIFAR10/100, ImageNet1K, and ADE20K, InfoBatch losslessly saves 40% overall cost."
  The method followed here.
- Jiang et al., *Accelerating Deep Learning by Focusing on the Biggest Losers* (Selective-Backprop), arXiv:1910.00762.
  P(backprop) = CDF_R(loss)^beta (Eq. 1, p.4). Abstract p.1: "converges to target error rates up to 3.5x faster than
  with standard SGD" (CIFAR/SVHN, time-to-target, still pays the forward pass for every sample).
- Loshchilov & Hutter, *Online Batch Selection for Faster Training of Neural Networks*, ICLR 2016 workshop,
  arXiv:1511.06343. Rank-based, exponentially decaying selection probability. Abstract p.1: "speeds up both AdaDelta
  and Adam by a factor of about 5" (MNIST only).
- Mindermann et al., *Prioritized Training on Points that are Learnable, Worth Learning, and Not Yet Learnt*
  (RHO-LOSS), ICML 2022, arXiv:2206.07137. p.2: "points selected by high loss or gradient norm are overwhelmingly
  those with noise-corrupted labels." Risk for AMI / FLEURS; a reason to prune only the low-loss side.

None of these is on ASR, and all report equal accuracy for less compute, not better accuracy.

## Why not BLS (arXiv:2604.04681)

BLS scores a sample with an EMA of the *batch* mean loss over the batches it appeared in. A sample's own loss is
only 1/B of that value; the paper's Fig. 2 shows the batch-composition noise dominating at all frequencies. Rough
SNR with B=128, alpha=0.7 (~6 visits): residual noise ~0.04 vs. easy-vs-hard signal ~0.02, so the ranking is close to
random. No InfoBatch-with-random-scores control and no rank correlation with true loss are reported. Its motivation
(per-sample loss is hard to get) does not apply here: token CE per utterance is free.
