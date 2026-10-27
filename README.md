# audiollm: encoder-free Mel → LLM baseline

Fixed-rate, time-only-chunked Mel-LLM baseline ([arXiv 2606.10231](https://arxiv.org/abs/2606.10231)) on
`LiquidAI/LFM2.5-230M-Base`, trained on LibriSpeech train-clean-100, evaluated by WER on test-clean / test-other and
on the Open ASR Leaderboard test sets (`benchmark.py`, run automatically after training).
This is the control that the dynamic-resolution tokenizer will be compared against.

> **Experiments use the 230M model** (`LiquidAI/LFM2.5-230M-Base`, hidden 1024, 14 layers) so runs are fast and
> cheap to iterate on. `LiquidAI/LFM2.5-2.6B-Base` (hidden 2048) is the paper-scale target: `--model LiquidAI/LFM2.5-2.6B-Base`.

```
wav ─► 80-d log-mel (25ms win / 10ms hop) ─► MVN (train stats) ─► stack 16 frames = 1 token (6.25 Hz, full freq)
    ─► Linear(1280 → 1024) ─► [BOS][audio…]["Transcription:"][ text ][EOS] ─► LFM2.5-230M-Base (frozen) + LoRA
```

Trained: the `Linear` projector (all of it) and LoRA adapters inside the LLM. The LLM's own weights stay frozen.

`explore_pipeline.ipynb` is the guide: it walks through this pipeline step by step (data and splits, batching, mel
patches, the `Linear` adapter, where LoRA sits, the final input sequence, run tracking) using the functions from
`train.py`.

## Run

```bash
./setup_rocm.sh                              # uv venv + ROCm torch
./setup_dataset.sh                           # all of data/ (~83 GB); --minimal = train.py's LibriSpeech only (~7 GB)
./setup_dataset.sh --check                   # verify data/ without downloading
source .venv/bin/activate
python train.py --max_steps 10               # smoke test
python train.py                              # full run (230M model)
python train.py --run_name lr3e-4 --lr 3e-4 --notes "higher peak lr"    # override any constant via its flag
python train.py --config runs/<id>/config.json --stack 4                # start from a previous run's config
python train.py --eval_only runs/<id>        # re-run test eval with that run's saved config
python train.py --benchmark_sets none        # skip the post-training Open ASR benchmark
```

Every UPPERCASE constant at the top of `train.py` is the default for a lower-case flag of the same name
(`LR` → `--lr`, `TEST_SPLITS` → `--test_splits a b`, `GRAD_CKPT` → `--no-grad_ckpt`); see `python train.py --help`.
Precedence: explicit flags > `--config` file > constants in `train.py`.

## Run tracking

Each training run gets a fresh directory `runs/<YYYYmmdd-HHMMSS>_<run_name>/` (never overwritten). The default
run name is `<model>_stack<N>` plus the hyperparameters you changed, e.g. `--epochs 10 --batch_size 16` →
`..._LFM2.5-230M-Base_stack16_epochs10_bs16`.

- **`runs/latest/`** is a symlink to the newest run: `tail -f runs/latest/log.txt`, open `runs/latest/plots.png`.
- **`runs/INDEX.md`**: every run, newest first, with status/step, what it changed, best val, test and benchmark WER.
- After training + test eval, `train.py` benchmarks the in-memory model on all Open ASR sets (`BENCHMARK_SETS`,
  `--benchmark_sets none` to skip; smoke runs skip it). A benchmark error is recorded, it doesn't fail the run.

| file | contents |
|---|---|
| `config.json` | resolved config (UPPERCASE keys) + `meta`: command, notes, overrides vs defaults, git commit/dirty, host, GPU |
| `git_diff.patch` | uncommitted changes at launch (only if the tree was dirty), so dirty runs stay reproducible |
| `metrics.jsonl` | one row per train log window (loss, lr, grad norm, speed, mem), val pass and test split |
| `summary.json` | status (`running`/`done`/`failed`/`interrupted`), current step, best val WER, test WER, duration, `benchmark` {set: WER} |
| `plots.png` | train loss (+EMA), grad norm vs clip threshold (% clipped), lr, val WER (+sub/del/ins, test WER), throughput, memory; redrawn every `PLOT_EVERY` steps and after each val |
| `log.txt`, `lora/`, `proj.pt`, `hyp_<split>.jsonl`, `test_results.json` | as before |

```bash
python runs.py                          # all runs: status, WERs, notes + every config key that differs between runs
python runs.py -f lr --sort test-clean  # filter by id substring, sort by a column
python runs.py show <run>               # full config + summary
python runs.py diff <run_a> <run_b>     # what changed between two runs
python runs.py curve <run> --every 5    # train loss / val WER over steps
python runs.py plot [<run> ...]         # (re)draw plots.png (default: all runs) + INDEX.md
python runs.py latest                   # path of the newest run
```

`<run>` is a path, a run id, a unique substring of one (e.g. the `--run_name`), or `latest`.

## Data

| role | data | size | used for |
|---|---|---|---|
| train | `train-clean-100+train-clean-360+ami_ihm/train+fleurs_en/train` (1–30 s) | 204,286 utts, 544.0 h | default training mixture (`--train_split train-clean-100` for the 100 h single-corpus control) |
| val | first 200 `dev-clean` utts by id | 200 utts, 23 min, **3 of 40 speakers** | in-training WER, picking the best run per LLM |
| test | `test-clean`, `test-other` | 5.4 h + 5.3 h | `train.py`'s own end-of-run WER |
| benchmark | 7 Open ASR sets (`data/esb/`) + local `fleurs_en`, `edacc` test sets | ~59 h total | `benchmark.py`, after training |

All splits are speaker-disjoint. VoxPopuli, Earnings-22, GigaSpeech, SPGISpeech and EdAcc are test-only (zero-shot,
out of domain); LibriSpeech, AMI IHM and FLEURS have train/dev/test splits. The val subset is deterministic but narrow
(3 speakers, clean read speech), so treat small val WER differences as noise.

**More corpora:** [docs/DATASETS.md](docs/DATASETS.md) ranks English ASR corpora and audio-understanding datasets
by annotation quality, with a short summary and a citation (BibTeX) for each. `download_corpora.py --list` shows
the registry. The default set (≈ 58 GB, keeping `data/` ≤ 100 GB) adds:
- LibriSpeech `train-clean-360` and `dev-other` (via `prepare_data.py`);
- AMI IHM and FLEURS en_us (unpacked to `data/asr_corpora/<corpus>/<split>/` by `unpack_parquet_asr_corpora.py` and wired into `train.py`), plus EdAcc (wired into `benchmark.py`);
- Speech Commands, Clotho-AQA, ESC-50, VocalSound, CREMA-D, MMAU test-mini and AIR-Bench (kept out of ASR training).

### Dataset viewer (quality check before pretraining)

```bash
.venv/bin/python -m dataset_viewer  # http://127.0.0.1:8765  (--port, --data, --db)
```

A local web page over everything in `data/`. Each corpus split is a separate dataset (37 at the moment).
- **Review samples**: play the audio (speed, gain, loop a dragged region), with the waveform (clipped samples in
  red) and the spectrogram. The spectrogram has two views: the *model view* (16 kHz HTK log-mel 80 that `train.py`
  computes) and a *linear STFT at the native rate*, which shows the file's true bandwidth. The page also shows the
  transcript/labels, every metadata column, and signal metrics with automatic flags. You mark each sample
  keep / reject / flag, with tags and a note (keys `1/2/3/0`, `j/k` to move; `?` lists all shortcuts).
- **Quality report**: scans a random subset (or the whole split) and shows histograms, percentiles and
  flag rates (clipping, low SNR, narrowband, mostly silent, quiet, text rate too fast or too slow, …). From there you
  can open the worst rows in the inspector, or bulk-reject every row that has a given flag.
- **Front-end lab**: runs the open sample through resample → quantise → STFT → mel → log → feature precision →
  MVN → stack with any `SR / BITS / N_FFT / HOP / N_MELS / STACK` and precision. Results update as you change a
  setting; values that differ from `train.py` are marked, and the derived numbers (window and hop in ms, token rate,
  patch size) update instantly. You can listen to every stage and inspect single tokens.
- **Decision per dataset**: *Include / Exclude / Undecided* for pretraining, with a note.

State is kept in `data/curation.sqlite3`. **Export curation manifest** (sidebar) writes JSON that lists, for each
dataset, the decision and the rejected/flagged sample keys. A key is `<parquet file relative to corpus dir>#<row>`,
or the utterance id for LibriSpeech. To use it in training code:

```python
from dataset_viewer.curation import load_curation
cur = load_curation("curation_manifest.json")      # or load_curation(db="data/curation.sqlite3")
cur.included_datasets(); cur.keep(dataset_id, key)
```

The metrics are heuristics for *finding* bad rows quickly, so listen before you reject. "Est. SNR" is the
p95−p10 spread of frame energy, not a true SNR. "Text rate" is alphanumeric characters per second of audio.
The thresholds are in `dataset_viewer/audio.py` (`THRESHOLDS`).

## Benchmark

`benchmark.py` scores any checkpoint on the Open ASR Leaderboard test sets (the suite of the paper's Table II), using
the leaderboard's Whisper English normalizer, so the numbers line up with the paper and the leaderboard. `train.py`
runs it on all sets right after training (with the model already in memory) and stores the WERs in
`runs/<id>/summary.json`. Audio that isn't 16 kHz (most of Earnings-22) is resampled with `soxr`.

```bash
python benchmark.py mel:runs/latest                               # 7 Open ASR sets + local fleurs_en/edacc test sets
python benchmark.py mel:runs/latest --sets openasr                # only the 7 Open ASR Leaderboard sets
python benchmark.py mel:runs/baseline_fixed12.5hz --sets earnings22 ami fleurs_en edacc   # selected sets (merged)
python benchmark.py hf:openai/whisper-large-v3 --sets librispeech_clean librispeech_other
python benchmark.py --table                                       # paper rows + hf baselines + best run per base LLM (by val WER) with its config
python benchmark.py --table --all_runs                            # ... plus every other benchmarked mel: run
```

`--limit N` evaluates N evenly spaced utterances per set for smoke tests. ESB data is cached in `data/esb/`; `fleurs_en` and `edacc` are read from `data/corpora/` and shown alongside the leaderboard columns without altering `OpenASR avg`.

## What is fixed (the "control")

All knobs are UPPERCASE constants at the top of `train.py` (the defaults); any of them can be overridden per run
with a flag, and the resolved values are saved in each run's `config.json`.

| | value | note |
|---|---|---|
| data | `train-clean-100+train-clean-360+ami_ihm/train+fleurs_en/train` (1–30 s, 544 h), val = first 200 dev-clean by id, test = test-clean + test-other | `--train_split train-clean-100` for 100 h only; see [Data](#data) |
| features | 80 log-mel, HTK filterbank, `log(mel+1e-6)`, global per-bin MVN from 2000 seeded train utts | cached in `data/LibriSpeech/mvn_*.pt`, keyed by split/N/SR/N_FFT/HOP/N_MELS |
| tokenization | stack 16 frames → 1280-d patches at 6.25 Hz, zero-pad tail | paper's ASR rate is stack 8 / 12.5 Hz (640-d < 1024 hidden); `--stack 8` reproduces it |
| projector | single `nn.Linear`, `PROJ_INIT=match_emb` (`N(0, emb_std/sqrt(fan_in))`, zero bias) | `--proj_init default` for PyTorch's default init (~20× larger than text embeddings) |
| LLM | LFM2.5-230M-Base, bf16, frozen + LoRA r64/α128 (`LORA_TARGETS=all-linear`: conv `in_proj`/`out_proj`, attn `q/k/v/out_proj`, MLP `w1/w2/w3`; not embeddings/`lm_head`) | no quantization; 2.6B-Base for paper-scale runs |
| precision | `PRECISION=bf16`: LoRA + projector weights, grads and AdamW states all bf16 | `amp` = fp32 master weights + bf16 autocast; `fp32` = runs before 2026-10-27 (fp32 LoRA/projector, ~1.65x slower/step) |
| optim | AdamW lr 1e-4, 200 warmup → cosine decay (`--lr_schedule linear` for the old schedule), clip 1.0, eff. batch 32, 2 epochs; `--optimizer muon` = `torch.optim.Muon` on every trainable 2D weight (LoRA A/B + projector weight), no AdamW at all (projector bias frozen), `--muon_adjust_lr match_rms_adamw` so the AdamW lr/wd carry over | paper's lr/clip |
| decode | greedy, batched left-pad, max 200 new tokens | |
| WER | lowercase, keep `[a-z' ]`, corpus-level jiwer | |

## Deviations from the paper (on purpose)

- **No conv downsampler.** The paper keeps Phi-4-MM's NeMo conv layers to do the ×8 reduction. Here the ×8 is
  plain frame stacking + one linear layer: the simplest "chunk in time and project" possible.
- **LoRA r=64**, not 320 (they had a 3.8B LLM on 16×H100).
- **No multimodal init.** LFM2.5-Base is text-only. The paper shows this costs them about 0.3 WER at scale. Expect a
  bigger gap at 100 h.
- **Much smaller LLM for experiments.** 230M instead of the paper's 3.8B, so WER will be higher in absolute terms;
  comparisons between tokenizers are made at the same LLM size.
- **bf16 LoRA instead of QLoRA.** Even 2.6B fits in 16 GB, and bitsandbytes on gfx1200 is unreliable. One less confound.

## Notes

- Left-padded batched decoding was checked against unbatched decoding: last-step logits match to bf16 precision.
  LFM2's short-conv zeros out padded positions, so padding doesn't leak.
- `causal_conv1d` kernels aren't available on ROCm, so transformers falls back to `F.conv1d`. It's correct, just slower.
- On gfx1200, PyTorch's AOTriton flash SDPA kernel can be enabled with
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. It is ~1.2x faster end-to-end than the math fallback on
  LFM2.5-350M (batch 4, 512 tokens), with lower peak memory. A prebuilt CK `flash-attn` wheel is available at
  <https://github.com/tungnguyenlam/flash-attn-gfx1200>, but its backward is slower than AOTriton on this GPU;
  it is mainly useful for inference and `attn_implementation="flash_attention_2"` compatibility.
