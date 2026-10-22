# audiollm: encoder-free Mel → LLM baseline

Fixed-rate, time-only-chunked Mel-LLM baseline ([arXiv 2606.10231](https://arxiv.org/abs/2606.10231)) on
`LiquidAI/LFM2.5-2.6B-Base`, trained on LibriSpeech train-clean-100, evaluated by WER on test-clean / test-other.
This is the control that the dynamic-resolution tokenizer will be compared against.

```
wav ─► 80-d log-mel (25ms win / 10ms hop) ─► MVN (train stats) ─► stack 8 frames = 1 token (12.5 Hz, full freq)
    ─► Linear(640 → 2048) ─► [BOS][audio…]["Transcription:"][ text ][EOS] ─► LFM2.5-2.6B-Base + LoRA
```

## Run

```bash
./setup.sh                                   # uv venv + ROCm torch
uv run --no-project --python 3.12 --with pyarrow --with huggingface_hub python prepare_data.py   # ~7 GB
source .venv/bin/activate
python train.py --max_steps 10               # smoke test
python train.py                              # full run, ~6 h on RX 9060 XT
python train.py --eval_only runs/baseline_fixed12.5hz
```

Outputs in `runs/<RUN_NAME>/`: `config.json` (all constants + git hash), `log.txt`, `lora/`, `proj.pt`,
`hyp_<split>.jsonl`, `test_results.json` (WER + measured `audio_tokens_per_sec`).

## Benchmark

`benchmark.py` scores any checkpoint on the Open ASR Leaderboard test sets (the suite of the paper's Table II), using
the leaderboard's Whisper English normalizer, so the numbers line up with the paper and the leaderboard.

```bash
python benchmark.py mel:runs/baseline_fixed12.5hz                 # LS clean/other, VoxPopuli, AMI, Earnings-22 (~4 GB)
python benchmark.py mel:runs/baseline_fixed12.5hz --sets all      # + GigaSpeech, SPGISpeech (~20 GB total)
python benchmark.py hf:openai/whisper-large-v3 --sets librispeech_clean librispeech_other
python benchmark.py --table                                       # every results/<tag>/summary.json + paper rows
```

`--limit N` evaluates N evenly spaced utterances per set for smoke tests. Data is cached in `data/esb/`.

## What is fixed (the "control")

All knobs are UPPERCASE constants at the top of `train.py`. Change them there, commit, run. Nothing comes from flags.

| | value | note |
|---|---|---|
| data | train-clean-100 (≤30 s), val = first 200 dev-clean by id, test = test-clean + test-other | |
| features | 80 log-mel, HTK filterbank, `log(mel+1e-6)`, global per-bin MVN from 2000 seeded train utts | cached in `data/mvn_*.pt` |
| tokenization | stack 8 frames → 12.5 Hz, zero-pad tail | paper's ASR rate |
| projector | single `nn.Linear`, fp32, random init | |
| LLM | LFM2.5-2.6B-Base, bf16, frozen + LoRA r64/α128 on all linear layers | no quantization |
| optim | AdamW lr 1e-4, 200 warmup → linear decay, clip 1.0, eff. batch 32, 2 epochs | paper's lr/clip |
| decode | greedy, batched left-pad, max 200 new tokens | |
| WER | lowercase, keep `[a-z' ]`, corpus-level jiwer | |

## Deviations from the paper (on purpose)

- **No conv downsampler.** The paper keeps Phi-4-MM's NeMo conv layers to do the ×8 reduction. Here the ×8 is
  plain frame stacking + one linear layer: the simplest "chunk in time and project" possible.
- **LoRA r=64**, not 320 (they had a 3.8B LLM on 16×H100).
- **No multimodal init.** LFM2.5-Base is text-only. The paper shows this costs them about 0.3 WER at scale. Expect a
  bigger gap at 100 h.
- **bf16 LoRA instead of QLoRA.** 2.6B fits in 16 GB, and bitsandbytes on gfx1200 is unreliable. One less confound.

## Notes

- Left-padded batched decoding was checked against unbatched decoding: last-step logits match to bf16 precision.
  LFM2's short-conv zeros out padded positions, so padding doesn't leak.
- `causal_conv1d` kernels aren't available on ROCm, so transformers falls back to `F.conv1d`. It's correct, just slower.
