"""
Vietnamese ASR text normalisation (docs/DATASETS_VI.md Section 6), shared by training targets, in-training val/test WER
(LANG="vi"), and benchmark.py on VI_SETS:

  1. NFC Unicode composition + HTML superscript conversion (km<sup>2</sup> -> km2) + annotation tag stripping.
  2. Digits, dates, currencies, percentages, and units -> spoken Vietnamese words via vietnormalizer
     (with foreign-word transliteration disabled so loanwords/proper nouns like "springboks" stay intact).
  3. Old/new tone-placement unification ("hoà" -> "hòa", "thuỷ" -> "thủy") via underthesea.text_normalize.
  4. Lowercase, strip punctuation while keeping all Unicode letters/digits/apostrophes, collapse whitespace.
  5. Canonicalise spoken-number variants ("trăm linh <1..9>" -> "trăm lẻ <1..9>", "ngàn" -> "nghìn", "mươi bốn" -> "mươi tư").

underthesea and vietnormalizer are imported lazily on first call so English-only runs do not require them.
"""
import functools
import re
import unicodedata

NEEDS_VERBALISER_RE = re.compile(r"[\d$%°²³€£¥₫]")
HTML_SUP_RE = re.compile(r"<sup>\s*(\d+)\s*</sup>", flags=re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>|\[[^\]]+\]")
DOUBLE_NGAY_RE = re.compile(r"\bngày\s+ngày\b", flags=re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s']|_")
TRAM_LINH_RE = re.compile(r"\btrăm linh (?=(?:một|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín)\b)")
MUOI_BON_RE = re.compile(r"\bmươi bốn\b")
NGAN_RE = re.compile(r"\bngàn\b")


@functools.lru_cache(maxsize=1)
def _backends():
    from underthesea import text_normalize
    from vietnormalizer import VietnameseNormalizer

    return text_normalize, VietnameseNormalizer(enable_transliteration=False)


def normalize_vi_text(text):
    """Normalise a Vietnamese transcript or hypothesis to canonical lowercase spoken-form words."""
    if not text or text.strip() == "IGNORE_TIME_SEGMENT_IN_SCORING":
        return ""
    tone_normalize, verbaliser = _backends()

    text = unicodedata.normalize("NFC", text)
    text = HTML_SUP_RE.sub(r"\1", text)
    text = TAG_RE.sub(" ", text)

    if NEEDS_VERBALISER_RE.search(text):
        text = verbaliser.normalize(text)
        text = DOUBLE_NGAY_RE.sub("ngày", text)

    text = tone_normalize(text).lower()
    text = NON_WORD_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = TRAM_LINH_RE.sub("trăm lẻ ", text)
    text = MUOI_BON_RE.sub("mươi tư", text)
    text = NGAN_RE.sub("nghìn", text)
    return text
