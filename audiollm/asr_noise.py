"""
Synthetic ASR hypotheses for correction-mode training (CORRECTION_P): corrupt a normalised Vietnamese reference the way a
first-pass recogniser does on regional accents.

Per word, with an utterance-level error rate drawn from U(0.03, 0.30):
  - mined substitution (CORRECTION_CONFUSIONS, e.g. Zipformer's giùm -> dùm, ni -> nay), sampled by P(hyp | ref);
  - accent-driven initial swaps: tr/ch, s/x, r/d/gi, n/l, v/d (southern v ~ d), gi/d;
  - final swaps merged in central/southern speech: -n/-ng, -t/-c, -nh/-n, -i/-y;
  - tone confusions: hỏi <-> ngã (merged in the south and centre), other tone swaps, tone dropped;
  - word deletion, filler insertion ("à", "ờ", "ừ"), word repetition.
English references get a light character-level version (deletion / repetition / letter swaps) only through the generic
ops. With probability CORRECTION_CLEAN_P (applied by the caller) the reference is returned unchanged.
"""
import json
import random
import unicodedata

TONE_MARKS = ["̀", "́", "̃", "̉", "̣"]  # huyền, sắc, ngã, hỏi, nặng
INITIAL_SWAPS = [("tr", "ch"), ("ch", "tr"), ("s", "x"), ("x", "s"), ("r", "d"), ("d", "r"), ("gi", "d"), ("d", "gi"),
                 ("r", "gi"), ("n", "l"), ("l", "n"), ("v", "d"), ("d", "v")]
FINAL_SWAPS = [("ng", "n"), ("n", "ng"), ("t", "c"), ("c", "t"), ("nh", "n"), ("i", "y"), ("y", "i")]
FILLERS = ["à", "ờ", "ừ", "ạ", "thì", "là"]


VOWELS = set("aăâeêioôơuưy")
HAT_VOWELS = set("ăâêôơư")


def _split_tone(word):
    """(NFC word without its tone mark, tone mark or "")."""
    nfd = unicodedata.normalize("NFD", word)
    tone = next((ch for ch in nfd if ch in TONE_MARKS), "")
    return unicodedata.normalize("NFC", nfd.replace(tone, "")) if tone else word, tone


def _with_tone(base, tone):
    """Place a tone mark on the syllable nucleus (new-style placement: hòa, thủy, nguyễn, thuốc)."""
    base = unicodedata.normalize("NFC", base)
    if not tone:
        return base
    start = 2 if (base.startswith("qu") or base.startswith("gi")) and len(base) > 2 and base[2] in VOWELS else 0
    positions = [i for i in range(start, len(base)) if base[i] in VOWELS]
    if not positions:
        positions = [i for i in range(len(base)) if base[i] in VOWELS]
        if not positions:
            return base
    run = [positions[0]]
    for i in positions[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            break
    hats = [i for i in run if base[i] in HAT_VOWELS]
    if hats:
        idx = hats[-1]
    elif len(run) >= 2 and run[-1] == len(base) - 1:
        idx = run[-2]
    else:
        idx = run[-1]
    return base[:idx] + unicodedata.normalize("NFC", base[idx] + tone) + base[idx + 1:]


def change_tone(word, rng):
    base, tone = _split_tone(word)
    if tone == "̉" and rng.random() < 0.6:
        new = "̃"
    elif tone == "̃" and rng.random() < 0.6:
        new = "̉"
    else:
        new = rng.choice([t for t in TONE_MARKS + [""] if t != tone])
    return _with_tone(base, new)


def swap_initial(word, rng):
    options = [(a, b) for a, b in INITIAL_SWAPS
               if word.startswith(a) and not (a == "n" and word[1:2] in ("h", "g"))]
    if not options:
        return None
    a, b = rng.choice(options)
    return b + word[len(a):]


def swap_final(word, rng):
    base, tone = _split_tone(word)
    options = [(a, b) for a, b in FINAL_SWAPS if base.endswith(a) and len(base) > len(a)]
    if not options:
        return None
    a, b = rng.choice(options)
    return _with_tone(base[:-len(a)] + b, tone)


class Noiser:
    def __init__(self, confusions_path=""):
        self.confusions = {}
        if confusions_path:
            data = json.load(open(confusions_path))
            for rules in data["categories"].values():
                for rule in rules:
                    if " " in rule["ref"] or not rule["hyp"]:
                        continue
                    self.confusions.setdefault(rule["ref"].lower(), []).append((rule["hyp"].lower(), rule["prob"]))

    def corrupt(self, text, rng=random):
        words = text.split()
        if not words:
            return text
        rate = rng.uniform(0.03, 0.30)
        out = []
        for word in words:
            if rng.random() >= rate:
                out.append(word)
                continue
            op = rng.random()
            if word in self.confusions and op < 0.45:
                choices, weights = zip(*self.confusions[word])
                out.append(rng.choices(choices, weights=weights)[0])
                continue
            if op < 0.60:
                new = swap_initial(word, rng) or swap_final(word, rng)
                out.append(new or change_tone(word, rng))
            elif op < 0.75:
                new = swap_final(word, rng) or swap_initial(word, rng)
                out.append(new or change_tone(word, rng))
            elif op < 0.85:
                out.append(change_tone(word, rng))
            elif op < 0.93:
                continue  # deletion
            elif op < 0.97:
                out += [word, rng.choice(FILLERS)]
            else:
                out += [word, word]
        return " ".join(out) if out else words[0]
