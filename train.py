"""
Baseline: encoder-free Mel -> LLM ASR, fixed-rate time-only chunking (Mel-LLM style, arXiv 2606.10231).

    wav -> 80-dim log-mel (10 ms hop) -> MVN (train stats) -> stack STACK frames (time-only chunk, full freq)
        -> Linear(STACK*80 -> d_model) -> [BOS][audio tokens]["Transcription:"][text][EOS] -> LFM2.5-2.6B-Base + LoRA

Everything that defines a run is a constant at the top of this file. Change it here, not via flags.

    python train.py                       # train on train-clean-100, val on dev-clean subset, test on test-clean/other
    python train.py --max_steps 50        # smoke test
    python train.py --eval_only runs/X    # re-run test eval from a saved run
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- config
RUN_NAME = "baseline_fixed12.5hz"
SEED = 0
MODEL = "LiquidAI/LFM2.5-2.6B-Base"
DATA = "data/LibriSpeech"
TRAIN_SPLIT = "train-clean-100"
VAL_SPLIT = "dev-clean"
VAL_N = 200                  # fixed dev subset used for in-training WER
TEST_SPLITS = ["test-clean", "test-other"]
MAX_TRAIN_SEC = 30.0         # drop the few very long utterances

# features
SR = 16000
N_FFT = 400                  # 25 ms
HOP = 160                    # 10 ms -> 100 frames/s
N_MELS = 80
MVN_N = 2000                 # utterances used to estimate MVN stats
STACK = 8                    # frames per audio token -> 100/8 = 12.5 Hz (paper's ASR setting)

# model / optimisation
PROMPT = "Transcription:"
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LR = 1e-4                    # paper: 1e-4 peak, linear warmup/decay, clip 1.0
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 200
EPOCHS = 2
BATCH_SIZE = 8
GRAD_ACCUM = 4               # effective batch 32
GRAD_CLIP = 1.0
GRAD_CKPT = True
LOG_EVERY = 20
EVAL_EVERY = 500             # optimizer steps
EVAL_BATCH_SIZE = 16
MAX_NEW_TOKENS = 200
NUM_WORKERS = 8

DEVICE = "cuda"
DTYPE = torch.bfloat16


def config_dict():
    return {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list, bool))}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def load_split(split):
    """Return list of dicts {id, path, text, sec}, sorted by id (deterministic)."""
    items = []
    for trans in sorted(Path(DATA, split).rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            uid, text = line.split(" ", 1)
            path = trans.parent / f"{uid}.flac"
            items.append({"id": uid, "path": str(path), "text": text.lower(), "sec": sf.info(str(path)).duration})
    return items


def mel_filterbank():
    """HTK-style triangular mel filterbank, shape (N_MELS, N_FFT//2+1)."""
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    hz = mel2hz(np.linspace(hz2mel(0.0), hz2mel(SR / 2), N_MELS + 2))
    bins = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    fb = np.zeros((N_MELS, len(bins)))
    for i in range(N_MELS):
        l, c, r = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((bins - l) / (c - l), (r - bins) / (r - c)))
    return torch.tensor(fb, dtype=torch.float32)


FBANK = mel_filterbank()
WINDOW = torch.hann_window(N_FFT)


def log_mel(path):
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SR
    spec = torch.stft(torch.from_numpy(wav), N_FFT, HOP, window=WINDOW, return_complex=True).abs() ** 2
    return torch.log(FBANK @ spec + 1e-6).T  # (T, N_MELS)


def compute_mvn(items):
    path = Path(DATA, f"mvn_{TRAIN_SPLIT}_n{MVN_N}.pt")
    if path.exists():
        return torch.load(path)
    rng = random.Random(SEED)
    feats = torch.cat([log_mel(it["path"]) for it in rng.sample(items, min(MVN_N, len(items)))])
    mvn = {"mean": feats.mean(0), "std": feats.std(0)}
    torch.save(mvn, path)
    return mvn


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, items, mvn, tok):
        self.items, self.mvn, self.tok = items, mvn, tok

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x = (log_mel(it["path"]) - self.mvn["mean"]) / self.mvn["std"]
        pad = (-x.shape[0]) % STACK                       # pad with zeros (= mean after MVN)
        x = torch.cat([x, x.new_zeros(pad, N_MELS)])
        x = x.reshape(-1, STACK * N_MELS)                   # time-only chunks spanning all mel bins
        target = self.tok(" " + it["text"], add_special_tokens=False).input_ids + [self.tok.eos_token_id]
        return {"audio": x, "target": target, "text": it["text"], "id": it["id"]}


def collate(batch):
    return batch  # embedding/padding happens on GPU in build_inputs


def length_batches(items, batch_size, seed):
    """Bucket by duration to cut padding: shuffle, sort within chunks of 50 batches, shuffle batches."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for s in range(0, len(idx), chunk):
        c = sorted(idx[s:s + chunk], key=lambda i: items[i]["sec"])
        batches += [c[j:j + batch_size] for j in range(0, len(c), batch_size)]
    rng.shuffle(batches)
    return batches


def worker_init(_):
    torch.set_num_threads(1)


def make_loader(ds, batches):
    return torch.utils.data.DataLoader(ds, batch_sampler=batches, collate_fn=collate,
                                       num_workers=NUM_WORKERS, worker_init_fn=worker_init, persistent_workers=False)


# ---------------------------------------------------------------- model
def build_inputs(batch, model, proj, tok, prompt_ids, train):
    """Build padded inputs_embeds. train=True: right-pad + labels. train=False: left-pad, no target."""
    embed = model.get_input_embeddings()
    bos = torch.tensor([tok.bos_token_id], device=DEVICE)
    seqs, labels = [], []
    for b in batch:
        a = proj(b["audio"].to(DEVICE)).to(DTYPE)
        text_ids = prompt_ids + (b["target"] if train else [])
        t = embed(torch.tensor(text_ids, device=DEVICE))
        seqs.append(torch.cat([embed(bos), a, t]))
        if train:
            labels.append([-100] * (1 + a.shape[0] + len(prompt_ids)) + b["target"])
    L = max(s.shape[0] for s in seqs)
    pad_emb = embed(torch.tensor([tok.pad_token_id], device=DEVICE))
    x = torch.zeros(len(seqs), L, seqs[0].shape[1], device=DEVICE, dtype=DTYPE)
    mask = torch.zeros(len(seqs), L, device=DEVICE, dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, device=DEVICE, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = s.shape[0]
        sl = slice(0, n) if train else slice(L - n, L)
        x[i] = pad_emb.expand(L, -1)
        x[i, sl] = s
        mask[i, sl] = 1
        if train:
            lab[i, :n] = torch.tensor(labels[i], device=DEVICE)
    return x, mask, lab


def load_model(run_dir=None):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    d = model.config.hidden_size
    proj = nn.Linear(STACK * N_MELS, d).to(DEVICE)          # fp32, randomly initialised
    if run_dir is None:
        model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                                 target_modules="all-linear", task_type="CAUSAL_LM"))
    else:
        model = PeftModel.from_pretrained(model, os.path.join(run_dir, "lora"))
        proj.load_state_dict(torch.load(os.path.join(run_dir, "proj.pt")))
    return tok, model, proj


# ---------------------------------------------------------------- eval
def normalize(s):
    s = re.sub(r"[^a-z' ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


@torch.no_grad()
def evaluate(items, mvn, tok, model, proj, out_path=None):
    model.eval()
    prompt_ids = tok(PROMPT, add_special_tokens=False).input_ids
    ds = SpeechDataset(items, mvn, tok)
    order = sorted(range(len(items)), key=lambda i: items[i]["sec"])   # length-sorted -> little padding
    batches = [order[i:i + EVAL_BATCH_SIZE] for i in range(0, len(order), EVAL_BATCH_SIZE)]
    refs, hyps, rows, n_audio_tok, n_sec = [], [], [], 0, 0.0
    for batch in make_loader(ds, batches):
        x, mask, _ = build_inputs(batch, model, proj, tok, prompt_ids, train=False)
        out = model.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, use_cache=True)
        for b, o in zip(batch, out):
            hyp = normalize(tok.decode(o, skip_special_tokens=True))
            ref = normalize(b["text"])
            refs.append(ref)
            hyps.append(hyp)
            rows.append({"id": b["id"], "ref": ref, "hyp": hyp})
            n_audio_tok += b["audio"].shape[0]
    n_sec = sum(it["sec"] for it in items)
    wer = jiwer.wer(refs, hyps)
    if out_path:
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    model.train()
    return {"wer": wer, "n_utt": len(items), "audio_tokens_per_sec": n_audio_tok / n_sec, "audio_tokens": n_audio_tok}


def run_test(run_dir, mvn, tok, model, proj):
    results = {}
    for split in TEST_SPLITS:
        t0 = time.time()
        results[split] = evaluate(load_split(split), mvn, tok, model, proj, os.path.join(run_dir, f"hyp_{split}.jsonl"))
        results[split]["eval_sec"] = time.time() - t0
        print(split, results[split], flush=True)
    json.dump(results, open(os.path.join(run_dir, "test_results.json"), "w"), indent=2)
    return results


# ---------------------------------------------------------------- train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=None, help="cap optimizer steps (smoke test)")
    ap.add_argument("--eval_only", type=str, default=None, help="run dir to evaluate")
    args = ap.parse_args()
    seed_everything(SEED)

    if args.eval_only:
        tok, model, proj = load_model(args.eval_only)
        mvn = torch.load(Path(DATA, f"mvn_{TRAIN_SPLIT}_n{MVN_N}.pt"))
        run_test(args.eval_only, mvn, tok, model, proj)
        return

    run_dir = os.path.join("runs", RUN_NAME + (f"_smoke{args.max_steps}" if args.max_steps else ""))
    os.makedirs(run_dir, exist_ok=True)
    cfg = config_dict()
    cfg["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    cfg["max_steps"] = args.max_steps
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    log = open(os.path.join(run_dir, "log.txt"), "a")

    def p(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log.write(s + "\n")
        log.flush()

    train_items = [it for it in load_split(TRAIN_SPLIT) if it["sec"] <= MAX_TRAIN_SEC]
    val_items = sorted(load_split(VAL_SPLIT), key=lambda it: it["id"])[:VAL_N]
    mvn = compute_mvn(train_items)
    p(f"train utts {len(train_items)} ({sum(i['sec'] for i in train_items) / 3600:.1f} h), val utts {len(val_items)}")

    tok, model, proj = load_model()
    if GRAD_CKPT:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    lora_params = [q for q in model.parameters() if q.requires_grad]
    params = lora_params + list(proj.parameters())
    p(f"trainable: lora {sum(q.numel() for q in lora_params) / 1e6:.1f}M, proj {sum(q.numel() for q in proj.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

    steps_per_epoch = math.ceil(len(train_items) / BATCH_SIZE) // GRAD_ACCUM
    total_steps = args.max_steps or steps_per_epoch * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / WARMUP_STEPS, max(0.0, (total_steps - s) / max(1, total_steps - WARMUP_STEPS))))
    p(f"steps/epoch {steps_per_epoch}, total steps {total_steps}")

    prompt_ids = tok(PROMPT, add_special_tokens=False).input_ids
    ds = SpeechDataset(train_items, mvn, tok)
    step, micro, t0, run_loss, run_tok = 0, 0, time.time(), 0.0, 0
    best = None
    for epoch in range(EPOCHS):
        for batch in make_loader(ds, length_batches(train_items, BATCH_SIZE, SEED + epoch)):
            x, mask, lab = build_inputs(batch, model, proj, tok, prompt_ids, train=True)
            loss = model(inputs_embeds=x, attention_mask=mask, labels=lab).loss
            (loss / GRAD_ACCUM).backward()
            run_loss += loss.item()
            run_tok += int(mask.sum())
            micro += 1
            if micro % GRAD_ACCUM:
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % LOG_EVERY == 0:
                dt = time.time() - t0
                p(f"ep {epoch} step {step}/{total_steps} loss {run_loss / (LOG_EVERY * GRAD_ACCUM):.4f} "
                  f"gnorm {gn:.2f} lr {sched.get_last_lr()[0]:.2e} tok/s {run_tok / dt:.0f} "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G")
                t0, run_loss, run_tok = time.time(), 0.0, 0
            if step % EVAL_EVERY == 0 or step == total_steps:
                r = evaluate(val_items, mvn, tok, model, proj)
                p(f"[val] step {step} {VAL_SPLIT}[:{VAL_N}] WER {r['wer'] * 100:.2f}")
                t0 = time.time()
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    model.save_pretrained(os.path.join(run_dir, "lora"))
    torch.save(proj.state_dict(), os.path.join(run_dir, "proj.pt"))
    p("saved", run_dir)
    if args.max_steps:
        return  # smoke run: skip full test eval
    res = run_test(run_dir, mvn, tok, model, proj)
    p("TEST", json.dumps({k: round(v["wer"] * 100, 2) for k, v in res.items()}))


if __name__ == "__main__":
    main()
