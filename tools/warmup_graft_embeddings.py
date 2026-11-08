"""
Text-only warmup of the NEW embedding rows of a grafted model (tools/build_vi_graft.py): causal-LM loss on Vietnamese
transcripts (normalised + raw) + ZAC 2026 texts, with ~20% English so new rows do not steal probability mass in English
contexts. Only rows flagged is_new in graft_meta.json get gradient; every other weight stays frozen. Overwrites the
model's weights in place (keeps a copy as model.init.safetensors the first time).

    python tools/warmup_graft_embeddings.py --model data/models/lfm25_230m_lfm3envi --steps 1500
"""
import argparse
import glob
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_vi_graft import VI_CORPORA, corpus_texts  # noqa: E402


def packed_batches(tokenizer, texts, seq_len, batch_size, rng):
    """Endless stream of (batch, seq_len) id tensors from shuffled texts joined with EOS."""
    eos = tokenizer.eos_token_id
    buffer = []
    while True:
        rng.shuffle(texts)
        for start in range(0, len(texts), 2048):
            for ids in tokenizer(texts[start:start + 2048], add_special_tokens=False).input_ids:
                buffer += ids + [eos]
                while len(buffer) >= seq_len * batch_size:
                    chunk, buffer = buffer[:seq_len * batch_size], buffer[seq_len * batch_size:]
                    yield torch.tensor(chunk).view(batch_size, seq_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/lfm25_230m_lfm3envi")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--en_frac", type=float, default=0.2)
    args = parser.parse_args()

    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.model)
    if not (out / "model.init.safetensors").exists():
        shutil.copy(out / "model.safetensors", out / "model.init.safetensors")
    meta = json.loads((out / "graft_meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(out)
    model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.bfloat16).cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings()
    emb.weight = torch.nn.Parameter(emb.weight.data.float(), requires_grad=True)
    model.tie_weights()
    assert model.lm_head.weight is emb.weight
    new_mask = torch.tensor(meta["is_new"], device="cuda", dtype=torch.float32)[:, None]
    emb.weight.register_hook(lambda grad: grad * new_mask)

    raw_vi, vi_norm, en_norm = corpus_texts()
    rng = random.Random(0)
    en = rng.sample(en_norm, min(len(en_norm), int(len(vi_norm + raw_vi) * args.en_frac / (1 - args.en_frac))))
    texts = vi_norm + raw_vi + en
    print(f"warmup texts: vi norm {len(vi_norm)}, raw vi/ZAC {len(raw_vi)}, en {len(en)}", flush=True)
    held = rng.sample([t for t in vi_norm if t], 500)

    def eval_vi():
        model.eval()
        total, n_chars = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for text in held:
                ids = torch.tensor([[tok.bos_token_id] + tok(text, add_special_tokens=False).input_ids]).cuda()
                total += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
                n_chars += len(text)
        model.train()
        return total / n_chars

    print(f"held-out vi nats/char before: {eval_vi():.3f}", flush=True)
    opt = torch.optim.AdamW([emb.weight], lr=args.lr, betas=(0.9, 0.98), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    model.train()
    stream = packed_batches(tok, texts, args.seq_len, args.batch, rng)
    start = time.time()
    for step in range(1, args.steps + 1):
        ids = next(stream).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ids, labels=ids).loss
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.3f} lr {sched.get_last_lr()[0]:.1e} "
                  f"{time.time() - start:.0f}s", flush=True)
    print(f"held-out vi nats/char after: {eval_vi():.3f}", flush=True)

    state = load_file(out / "model.init.safetensors")
    key = next(k for k in state if k.endswith("embed_tokens.weight"))
    state[key] = emb.weight.data.to(torch.bfloat16).cpu().contiguous()
    save_file(state, out / "model.safetensors", metadata={"format": "pt"})
    meta["warmup"] = vars(args)
    (out / "graft_meta.json").write_text(json.dumps(meta))
    print(f"saved warmed embeddings into {out / 'model.safetensors'}")


if __name__ == "__main__":
    main()
