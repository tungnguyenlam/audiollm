# Literature Review: Token Merging

Focus: token merging and adaptive-rate compression, with emphasis on what applies to this repo's
fixed-rate, encoder-free mel → LLM baseline.

Compiled 2026-10-24. Initial search delegated to Gemini 3.8 Flash; claims marked ✅ were checked
against the paper directly. Unmarked numbers come from the research pass and have not been verified.

## Takeaway for this project

- No paper found does adaptive frame merging in an **encoder-free** mel → LLM model — this looks like an open gap.
- A-ToMe shows adaptive adjacent merging beats fixed conv subsampling at equal reduction (see table below).
- Obvious first experiment: A-ToMe-style adjacent merging on the projected mel frames (or inside the
  first few LLM layers), compared against fixed frame stacking at the same average rate.
- For learned variable rate: CIF, or CTC-guided collapse (needs an auxiliary CTC head).

## Core method and vision successors

| Paper | Mechanism | Training | Reported tradeoff |
|---|---|---|---|
| **ToMe** (ICLR 2023) [2210.09461](https://arxiv.org/abs/2210.09461) | Bipartite soft matching: split tokens into sets A/B, match by cosine similarity of attention keys, merge top-r pairs by size-weighted average. Proportional attention `softmax(QKᵀ/√d + log s)` keeps merged tokens' weight. | Training-free (fine-tuning optional) | ~2× throughput on ViT-L/H, ~0.2–0.3% acc drop |
| **ToMe for Stable Diffusion** (CVPRW 2023) [2303.17604](https://arxiv.org/abs/2303.17604) | Merge before attention/MLP, then **unmerge** back to the original grid | Training-free | Up to 2× (5.4× at 2048²) |
| **ToFu** (CVPR 2024) [2312.01026](https://arxiv.org/abs/2312.01026) | Prune in non-linear layers, merge in near-linear ones; merge with spherical interpolation (MLERP) to preserve feature norms | Training-free or fine-tuned | Beats ToMe and pruning alone |
| **DiffRate** (ICCV 2023) [2305.17997](https://arxiv.org/abs/2305.17997) | Per-layer prune and merge rates learned by backprop through the task loss | Rate search; backbone can stay frozen | ViT-H: ~40% fewer FLOPs, 0.16% drop |
| **PPT** (2023) [2310.01812](https://arxiv.org/abs/2310.01812) | Adaptively choose pruning (inattentive tokens) or pooling (duplicated tokens) per layer | Training-free | Beats pure prune/pool on DeiT/Swin |
| **LTMP** (TMLR 2023) [2307.10780](https://arxiv.org/abs/2307.10780) | Learned thresholds: prune below an importance threshold, merge above a similarity threshold | ~1 epoch fine-tune | Smooth acc/throughput curve |
| **DTEM** (NeurIPS 2024) [2412.10569](https://arxiv.org/abs/2412.10569) | Side module learns merge embeddings; differentiable soft grouping | Trains side module only | DeiT-S: 37% fewer FLOPs |
| **TCFormer / CTM** (CVPR 2022) [2204.08680](https://arxiv.org/abs/2204.08680) | Density-peak clustering (DPC-kNN), merges non-local clusters | End-to-end | Keeps detail in dense prediction |
| **EViT** (ICLR 2022) [2202.07800](https://arxiv.org/abs/2202.07800) | Fuse inattentive tokens (by CLS attention) into one token | Fine-tune | Baseline |
| **DynamicViT** (NeurIPS 2021) [2106.02034](https://arxiv.org/abs/2106.02034) | Learned pruning heads, Gumbel-softmax | Trained | Pruning baseline |

### Token reduction inside LLMs and multimodal LLMs

- **LLaVA-PruMerge** [2403.15388](https://arxiv.org/abs/2403.15388): keep visual tokens with high attention, merge the rest into them by key similarity; 4–18× fewer visual tokens.
- **KVMerger** [2407.08454](https://arxiv.org/abs/2407.08454): merge similar KV-cache entries with Gaussian-kernel weights instead of evicting them. Training-free.
- **CaM** [2405.03175](https://arxiv.org/abs/2405.03175): merge evicted KV entries into the ones that remain.

### Open disagreement

- Plain size-weighted averaging plus proportional attention (ToMe), or norm-preserving interpolation (ToFu)?

## Speech and audio

### A-ToMe: Accelerating Transducers through Adjacent Token Merging (Interspeech 2023) ✅

[2306.16009](https://arxiv.org/abs/2306.16009) · the most directly relevant paper.

- Merges only **adjacent** frames, using cosine similarity of the attention keys, then averages them
  (at most 50% merged per layer, i.e. at most average pooling).
- Placed between attention and FFN at layers 2, 5, 8, 11, 14, 17 of an 18-layer encoder.
- Two schedules: fixed similarity threshold (adaptive count) or fixed ratio per layer.
- **Trained from scratch with merging** — not training-free.
- LibriSpeech results (Table 1):

| Config | Merged | test-clean | test-other |
|---|---|---|---|
| Baseline | 0% | 2.79 | 6.01 |
| Ratio 10%/layer | 46% | 2.79 | 5.90 |
| Ratio 15%/layer | 61% | 2.88 | 6.02 |
| Ratio 20%/layer | 73% | 2.88 | 6.20 |
| Threshold 0.90 | 42% | 2.90 | 6.17 |
| Threshold 0.85 | 57% | 2.89 | 5.96 |
| Threshold 0.80 | 72% | 3.01 | 6.04 |
| Conv subsampling ×2 | 50% | 2.90 | 6.29 |
| Conv subsampling ×4 | 75% | 3.15 | 6.86 |

At threshold 0.85: 1.70× faster on GPU, 1.35× on CPU.

### Mel-LLM: LLM can Read Spectrogram (SLT 2026) ✅

[2606.10231](https://arxiv.org/abs/2606.10231) · closest to this repo's baseline.

- 80-dim log-mel (MVN) → optional NeMoConv (8× downsampling, 1024 channels) → projection to the LLM
  (the paper describes it as a single linear layer in the text but as a 2-layer MLP in Table I).
- Fixed 12.5 Hz in the main setting; ablates 100/50/25/6.25 Hz. **No merging, no adaptive rate.**
- OpenASR average WER: 7.12 (Phi-4-MM init), 7.44 (random init), against 5.61 for the
  encoder-based Phi-4-MM and 7.44 for Whisper-large-v3.
- 1.57× faster **training** than the encoder-based model (not an inference figure).

### Other speech work

| Paper | Mechanism | Where | Compression | Pretrained encoder? |
|---|---|---|---|---|
| **CIF** (ICASSP 2020) [1905.11235](https://arxiv.org/abs/1905.11235) | Learned per-frame weight α accumulates until it reaches 1, then fires one integrated vector; trained with a quantity loss | Encoder → decoder connector | ~1 token per output unit | No |
| **CTC compression** (Gaido et al., EACL 2021) [2102.01578](https://arxiv.org/abs/2102.01578) | Collapse runs of identical CTC predictions (blanks included) by averaging | Mid-encoder | Variable | No |
| **CJST** (2024) [2411.07607](https://arxiv.org/abs/2411.07607) | CTC peaky alignment: drop blanks, merge non-blank runs; for decoder-only ASR | Adapter into the LLM | Variable | No (supports frozen encoder too) |
| **Affinity Pooling** (Apr 2026) ✅ exists [2604.06871](https://arxiv.org/abs/2604.06871) | Training-free similarity merging of speech tokens at the input and in deep LLM layers | Speech LLM | ~27% fewer prefill FLOPs | Yes (Whisper-based) |
| **ToMe for multilingual Whisper** (arXiv 2026) ✅ exists [2609.13151](https://arxiv.org/abs/2609.13151) | ToMe at inference in the Whisper encoder; 16 languages, 3 sizes, DoRA | Encoder | 20–50% | Yes |
| **FastAST** (Interspeech 2024) [2406.07676](https://arxiv.org/abs/2406.07676) | ToMe in Audio Spectrogram Transformer + distillation | Encoder | ~2× throughput | No |
| **VARSTok** (2025) [2509.04685](https://arxiv.org/abs/2509.04685) | Temporal density-peak clustering → variable-length units with implicit duration | Tokenizer | 3–12.5 Hz | No |
| **SALMONN** (ICLR 2024) [2310.13289](https://arxiv.org/abs/2310.13289) | Window Q-Former, 1 query per 17-frame window | Encoder → LLM | Fixed 17:1 | Yes |
| **Qwen-Audio** (2023) [2311.07919](https://arxiv.org/abs/2311.07919) | Stride-2 pooling | Encoder → LLM | Fixed 2× (25 Hz) | Yes |
| **LLaMA-Omni** (2024) [2409.06666](https://arxiv.org/abs/2409.06666) | Stack 5 frames + MLP | Encoder → LLM | Fixed 5× | Yes |

### Open questions

- **Global or local matching for speech:** ToMe matches across the whole sequence; A-ToMe restricts
  matching to neighbours to keep frames in time order.
- **CTC blank handling:** drop blanks entirely, or average runs of them? Dropping them can lose
  boundary and prosody information.
