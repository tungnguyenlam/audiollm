"""
Graft the LFM3 tokenizer (shared by LFM3-1.7B-Base and LFM3-Audio-2B) onto LFM2.5-230M-Base, pruned to English +
Vietnamese + digits + punctuation, and write a loadable HF model dir (docs/idea_lfm3_tokenizer_graft.md).

Vocab selection:
  1. every LFM3 token emitted on the local en + vi ASR transcripts (raw and normalised) and on the ZAC 2026 texts
     (user requests, assistant replies, tool arguments, tool catalog),
  2. plus the frequent general tokens: LFM3 ids below --common_ids (ids follow BPE merge order, so low ids are the
     most frequent merges) whose text uses only allowed characters (ASCII printable + Vietnamese letters + a few
     typographic marks), so common cased English / code pieces outside the lowercase transcripts survive,
  3. closed under BPE merges (every merge parent of a kept token is kept), plus all 256 byte tokens and all LFM3
     special tokens. Text made only of kept tokens tokenises exactly as LFM3 does; anything else falls back to bytes.

Embeddings (tied with lm_head in LFM2.5): a token LFM2.5 already has keeps its LFM2.5 row; a new token gets the mean
of the LFM2.5 rows its text splits into under LFM2.5's BPE (Liquid, arXiv 2607.15232). Specials are matched by name.

Writes <out>/ (config, weights, tokenizer) and <out>/graft_meta.json with lfm3_ids[new_id] = LFM3 id (for
on-policy distillation from LFM3-Audio-2B later) and is_new[new_id].

    python tools/build_vi_graft.py --out data/models/lfm25_230m_lfm3envi [--stats_only]
"""
import argparse
import glob
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LFM3 = "LiquidAI/LFM3-1.7B-Base"
LFM3_REV = "545fc8f4c7b5fa9f405105274cd29c4f8f44b526"
BASE = "LiquidAI/LFM2.5-230M-Base"
VI_CORPORA = {"fleurs_vi", "fpt_fosd", "gigaspeech2_vi", "lsvsc", "phoaudiobook_200h", "vimd", "vlsp2020_100h"}
ZAC_ROOT = Path("/home/nguyenlt/ZaloAI/data")


def bytes_to_unicode():
    """GPT-2 byte <-> printable-unicode table used by ByteLevel BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


def token_text(token):
    """Byte-level token string -> decoded text, or None if it is not complete UTF-8."""
    try:
        return bytes(BYTE_DECODER[ch] for ch in token).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def allowed_chars():
    base = "aăâbcdđeêfghijklmnoôơpqrstuưvwxyz"
    tones = ["", "̀", "́", "̃", "̉", "̣"]
    chars = set(chr(c) for c in range(0x20, 0x7F)) | set("\n\t\r")
    for letter in base:
        for tone in tones:
            for ch in (letter, letter.upper()):
                chars.add(unicodedata.normalize("NFC", ch + tone))
    chars |= set("“”‘’…–—•·€£¥₫°×÷²³½¼¾«»")
    return chars


def corpus_texts():
    """(raw vi + ZAC texts, normalised vi texts, normalised en texts) from local ASR manifests, LibriSpeech and ZAC."""
    from audiollm import config as C
    from audiollm.data import normalize_training_text

    raw, vi, en = [], [], []
    for manifest in glob.glob("data/asr_corpora/*/*/manifest.jsonl"):
        corpus = manifest.split("/")[2]
        texts = [json.loads(line)["text"] for line in open(manifest)]
        if corpus in VI_CORPORA:
            raw += texts
            vi += texts
        else:
            en += texts
    for path in glob.glob("data/LibriSpeech/*/*/*/*.trans.txt"):
        texts = [line.split(" ", 1)[1].strip().lower() for line in open(path)]
        en += texts
    zac = []
    for split in ("train", "public_test"):
        jsonl = ZAC_ROOT / split / f"{split}.jsonl"
        if not jsonl.exists():
            continue
        for line in open(jsonl):
            row = json.loads(line)
            if row.get("audio_text"):
                zac.append(row["audio_text"])
            zac.append(json.dumps(row.get("context", {}), ensure_ascii=False))
            for turn in row.get("history", []):
                zac += [turn[k] for k in ("audio_text", "text") if turn.get(k)]
                zac += [json.dumps(turn.get("tool_calls", []), ensure_ascii=False)]
            if "label" in row:
                zac.append(json.dumps(row["label"]["tool_calls"], ensure_ascii=False))
        catalog = ZAC_ROOT / split / f"{split}_tools_catalog.json"
        if catalog.exists():
            zac.append(catalog.read_text())
    raw += zac
    vi += [t for t in zac if len(t) < 400]

    C.LANG = "vi"
    vi_norm = [normalize_training_text(t) for t in vi]
    C.LANG = "en"
    en_norm = [normalize_training_text(t) for t in en]
    return raw, vi_norm, en_norm


def count_tokens(tokenizer, texts, chunk=4096):
    counts = Counter()
    for i in range(0, len(texts), chunk):
        for ids in tokenizer(texts[i:i + chunk], add_special_tokens=False).input_ids:
            counts.update(ids)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/models/lfm25_230m_lfm3envi")
    parser.add_argument("--stats_only", action="store_true")
    parser.add_argument("--common_ids", type=int, default=16000, help="clause 2 cutoff on LFM3 ids")
    args = parser.parse_args()

    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

    tok3 = AutoTokenizer.from_pretrained(LFM3, revision=LFM3_REV)
    tok25 = AutoTokenizer.from_pretrained(BASE)
    J = json.loads(tok3.backend_tokenizer.to_str())
    vocab3 = J["model"]["vocab"]
    inv3 = {i: t for t, i in vocab3.items()}
    special3 = {a["content"]: a["id"] for a in J["added_tokens"]}
    vocab25 = tok25.get_vocab()

    # 1. usage on corpora
    raw, vi_norm, en_norm = corpus_texts()
    print(f"texts: raw {len(raw)}, vi norm {len(vi_norm)}, en norm {len(en_norm)}", flush=True)
    used = set()
    usage = {}
    for name, texts in (("raw", raw), ("vi", vi_norm), ("en", en_norm)):
        counts = count_tokens(tok3, texts)
        usage[name] = counts
        used |= set(counts)
    used = {inv3[i] for i in used if i in inv3}

    # 2. allowed-character tokens LFM2.5 already had
    chars = allowed_chars()
    char_ok = set()
    for token in vocab3:
        text = token_text(token)
        if text is not None and text and all(ch in chars for ch in text):
            char_ok.add(token)
    char_ok_old = {t for t in char_ok if vocab3[t] < args.common_ids}

    # 3. merge closure + bytes
    parents = {}
    for merge in J["model"]["merges"]:
        a, b = merge.split(" ", 1) if isinstance(merge, str) else merge
        parents.setdefault(a + b, (a, b))
    keep = set(used) | char_ok_old | set(BYTE_ENCODER.values())
    stack = list(keep)
    while stack:
        token = stack.pop()
        for parent in parents.get(token, ()):
            if parent not in keep:
                keep.add(parent)
                stack.append(parent)
    keep = {t for t in keep if t in vocab3}
    n_new = sum(1 for t in keep if t not in vocab25)
    print(f"LFM3 vocab {len(vocab3)} | used on corpora {len(used)} | allowed-char {len(char_ok)} "
          f"(id < {args.common_ids}: {len(char_ok_old)}) | kept regular {len(keep)} + specials {len(special3)} | new vs LFM2.5 {n_new}")
    for name in ("vi", "en"):
        words = sum(len(t.split()) for t in (vi_norm if name == "vi" else en_norm))
        print(f"  {name} normalised: {sum(usage[name].values()) / max(1, words):.3f} LFM3 tokens/word")
    if args.stats_only:
        return

    # ---- new tokenizer: kept regular tokens in LFM3 id order, then LFM3 specials
    kept_sorted = sorted(keep, key=lambda t: vocab3[t])
    new_vocab = {t: i for i, t in enumerate(kept_sorted)}
    merges = []
    for merge in J["model"]["merges"]:
        a, b = merge.split(" ", 1) if isinstance(merge, str) else merge
        if a in new_vocab and b in new_vocab and a + b in new_vocab:
            merges.append([a, b])
    added = []
    for a in sorted(J["added_tokens"], key=lambda a: a["id"]):
        a = dict(a, id=len(new_vocab) + len(added))
        added.append(a)
    J_new = dict(J)
    J_new["model"] = dict(J["model"], vocab=new_vocab, merges=merges)
    J_new["added_tokens"] = added
    new_tok_backend = Tokenizer.from_str(json.dumps(J_new, ensure_ascii=False))
    special_ids = {a["content"]: a["id"] for a in added}
    new_tok = PreTrainedTokenizerFast(tokenizer_object=new_tok_backend, bos_token="<|startoftext|>",
                                      eos_token="<|im_end|>", pad_token="<|pad|>")
    vocab_size = len(new_vocab) + len(added)
    lfm3_ids = [vocab3[t] for t in kept_sorted] + [special3[a["content"]] for a in added]

    # ---- check: tokenisation equals LFM3's on corpus samples (token strings), and round-trips
    import random
    rng = random.Random(0)
    sample = rng.sample(vi_norm, 3000) + rng.sample(en_norm, 3000) + rng.sample(raw, 3000)
    mismatches = 0
    for text in sample:
        ids3 = tok3(text, add_special_tokens=False).input_ids
        ids_new = new_tok(text, add_special_tokens=False).input_ids
        if [lfm3_ids[i] for i in ids_new] != ids3:
            mismatches += 1
    roundtrip = sum(new_tok.decode(new_tok(t, add_special_tokens=False).input_ids) == t for t in sample)
    print(f"check: {mismatches}/{len(sample)} samples tokenise differently from LFM3; "
          f"round-trip exact {roundtrip}/{len(sample)}")

    # ---- model with grafted embeddings
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    old = model.get_input_embeddings().weight.data.clone()
    mean_row = old[:len(vocab25)].mean(0)
    rows = torch.empty(vocab_size, old.shape[1])
    is_new = []
    bpe25 = tok25.backend_tokenizer.model
    for new_id, token in enumerate(kept_sorted):
        if token in vocab25:
            rows[new_id] = old[vocab25[token]]
            is_new.append(False)
        else:
            pieces = [vocab25[p.value] for p in bpe25.tokenize(token) if p.value in vocab25]
            rows[new_id] = old[pieces].mean(0) if pieces else mean_row
            is_new.append(True)
    for a in added:
        content = a["content"]
        rows[a["id"]] = old[vocab25[content]] if content in vocab25 else mean_row
        is_new.append(content not in vocab25)
    model.resize_token_embeddings(vocab_size)
    model.get_input_embeddings().weight.data.copy_(rows)
    model.tie_weights()
    model.config.vocab_size = vocab_size
    model.config.bos_token_id = special_ids["<|startoftext|>"]
    model.config.eos_token_id = special_ids["<|im_end|>"]
    model.config.pad_token_id = special_ids["<|pad|>"]
    if model.generation_config is not None:
        model.generation_config.bos_token_id = model.config.bos_token_id
        model.generation_config.eos_token_id = model.config.eos_token_id
        model.generation_config.pad_token_id = model.config.pad_token_id
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out)
    new_tok.save_pretrained(out)
    meta = {
        "base": BASE, "tokenizer_from": f"{LFM3}@{LFM3_REV}", "vocab_size": vocab_size,
        "n_regular": len(new_vocab), "n_special": len(added), "n_new_rows": sum(is_new),
        "check_mismatches": mismatches, "check_samples": len(sample),
        "lfm3_ids": lfm3_ids, "is_new": is_new,
    }
    (out / "graft_meta.json").write_text(json.dumps(meta))
    print(f"wrote {out}: vocab {vocab_size} ({sum(is_new)} new rows), tied embeddings "
          f"{vocab_size * old.shape[1] / 1e6:.1f}M params (was {old.shape[0] * old.shape[1] / 1e6:.1f}M)")


if __name__ == "__main__":
    main()
