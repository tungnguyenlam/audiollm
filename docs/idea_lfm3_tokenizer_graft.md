# Parked idea: graft the LFM3 tokenizer onto LFM2.5-230M and prune it to en + vi

Parked 2026-11-04. Measurements: `results/tokenizer_graft_20261104/usage.{py,txt}` (gitignored), WORKLOG 2026-11-04.

**Why:** LFM2.5's 64K tokenizer has no Vietnamese merges. Vietnamese diacritics fall back to bytes: 3.23 tokens per
word on normalized targets, against 1.14 for LFM3. The Qwen3-0.6B A/B win mixes tokenizer and model size (2.6x the
parameters). A grafted LFM2.5-230M would isolate the tokenizer effect at equal size.

**Facts measured:**
- The `LiquidAI/LFM3-1.7B-Base` tokenizer (rev `545fc8f`) is an in-place expansion of LFM2.5's: same pre-tokenizer,
  all 63,683 LFM2.5 merges, all 63,893 regular tokens (IDs shifted by -501). 125,033 tokens.
- Merge-closed pruning to tokens emitted on en + vi training targets (plus BPE ancestors and bytes) gives 37,347
  tokens. Tied embeddings go from ~66M to ~38M params. 9,719 kept tokens are new to LFM2.5. 2,096 of them cover
  78% of Vietnamese target tokens.

**Recipe if revived** (Liquid, arXiv 2607.15232):
1. Copy the old rows.
2. Initialize each new row as the mean of its LFM2.5 sub-token embeddings.
3. Run a text-only embedding warmup.
4. Train ASR with the embeddings (tied lm_head) unfrozen, plus LoRA and the projector.
5. Use the same 500 h, 19.2k-step matched config as the Qwen/LFM A/B.

**Open issues:**
- Vietnamese text for the warmup: only ~300K transcripts are on disk, and downloads are paused.
- Pruning to transcripts drops general text tokens. That hurts later spoken-understanding tasks; prune against a
  broader text sample instead.
