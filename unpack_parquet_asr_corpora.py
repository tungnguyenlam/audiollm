"""
Unpack the HF-parquet ASR corpora in data/corpora/ (download_corpora.py) into the file layout train.py reads:

    data/asr_corpora/<corpus>/<split>/audio/<id>.flac     16 kHz mono FLAC (resampled / downmixed if needed)
    data/asr_corpora/<corpus>/<split>/manifest.jsonl      {"id", "path", "text", "sec"} per utterance, raw text

train.py then accepts "<corpus>/<split>" anywhere it takes a split name (e.g. --train_split
"train-clean-100+train-clean-360+ami_ihm/train+fleurs_en/train"). Text is stored raw; train.py normalises it.

    python unpack_parquet_asr_corpora.py                            # every corpus/split below (skips finished ones)
    python unpack_parquet_asr_corpora.py --workers 2 lsvsc/train    # specific splits with capped worker pool
"""
import argparse
import io
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
OUT = Path("data/asr_corpora")


def clean_tedlium(text):
    """TED-LIUM: drop untranscribed segments, re-attach split contractions ("it 's" -> "it's"). <unk> (mostly
    hesitations / noise) is kept and stripped later by train.py's normaliser, like the AMI / LibriSpeech tags."""
    if "ignore_time_segment_in_scoring" in text:
        return ""
    return re.sub(r" '(?=(?:s|re|ve|ll|d|m|t)\b)", "'", text)


GIGASPEECH_PUNCTUATION = {"<COMMA>": ",", "<PERIOD>": ".", "<QUESTIONMARK>": "?", "<EXCLAMATIONPOINT>": "!"}


def clean_gigaspeech(text):
    """GigaSpeech: punctuation tags -> marks; drop segments that are only garbage tags (<SIL>, <MUSIC>, <NOISE>, <OTHER>)."""
    for tag, mark in GIGASPEECH_PUNCTUATION.items():
        text = text.replace(f" {tag}", mark).replace(tag, mark)
    return text if re.sub(r"<[^>]+>|[,.?!]", "", text).strip() else ""


TEDLIUM = "data/corpora/tedlium3/data"
# "<corpus>/<split>" -> (parquet glob, id column or None [uses audio.path stem], text column, text cleaner or None, suffix ids with shard + row?)
SOURCES = {
    # English corpora
    "ami_ihm/train": ("data/corpora/ami_ihm/ihm/train-*.parquet", "audio_id", "text", None, False),
    "ami_ihm/validation": ("data/corpora/ami_ihm/ihm/validation-*.parquet", "audio_id", "text", None, False),
    # FLEURS ids repeat (one id per sentence, several speakers)
    "fleurs_en/train": ("data/corpora/fleurs_en/parquet-data/en_us/train-*.parquet", "id", "raw_transcription", None, True),
    "fleurs_en/validation": ("data/corpora/fleurs_en/parquet-data/en_us/validation-*.parquet", "id", "raw_transcription", None, True),
    "voxpopuli_en/train": ("data/corpora/voxpopuli_en/en/train-*.parquet", "audio_id", "raw_text", None, False),
    "voxpopuli_en/validation": ("data/corpora/voxpopuli_en/en/validation-*.parquet", "audio_id", "raw_text", None, False),
    "tedlium3/train": (f"{TEDLIUM}/train-*.parquet", "id", "text", clean_tedlium, False),
    "tedlium3/validation": (f"{TEDLIUM}/validation-*.parquet", "id", "text", clean_tedlium, False),
    # train/validation only: the test-* shards next to them are the gigaspeech / spgispeech benchmark sets
    "gigaspeech_m/train": ("data/corpora/gigaspeech/parquet-data/m/train-*.parquet", "segment_id", "text", clean_gigaspeech, False),
    "gigaspeech_m/validation": ("data/corpora/gigaspeech/parquet-data/m/validation-*.parquet", "segment_id", "text", clean_gigaspeech, False),
    "spgispeech_s/train": ("data/corpora/spgispeech/S/train-*.parquet", "wav_filename", "transcript", None, False),
    "spgispeech_s/validation": ("data/corpora/spgispeech/S/validation-00000-*.parquet", "wav_filename", "transcript", None, False),

    # Vietnamese corpora (docs/DATASETS_VI.md; text kept raw UTF-8 in manifest.jsonl, normalised when LANG="vi")
    "gigaspeech2_vi/validation": ("data/corpora/gigaspeech2_vi/data/validation-*.parquet", None, "transcription", None, True),
    "gigaspeech2_vi/test": ("data/corpora/gigaspeech2_vi/data/test-*.parquet", None, "transcription", None, True),
    "fleurs_vi/train": ("data/corpora/fleurs_vi/parquet-data/vi_vn/train-*.parquet", "id", "raw_transcription", None, True),
    "fleurs_vi/validation": ("data/corpora/fleurs_vi/parquet-data/vi_vn/validation-*.parquet", "id", "raw_transcription", None, True),
    "fleurs_vi/test": ("data/corpora/fleurs_vi/parquet-data/vi_vn/test-*.parquet", "id", "raw_transcription", None, True),
    "vlsp2020_100h/train": ("data/corpora/vlsp2020_100h/data/train-*.parquet", None, "transcription", None, True),
    "lsvsc/train": ("data/corpora/lsvsc/data/train-*.parquet", None, "transcription", None, True),
    "lsvsc/validation": ("data/corpora/lsvsc/data/validation-*.parquet", None, "transcription", None, True),
    "lsvsc/test": ("data/corpora/lsvsc/data/test-*.parquet", None, "transcription", None, True),
    "vimd/train": ("data/corpora/vimd/data/train-*.parquet", "filename", "text", None, True),
    "vimd/validation": ("data/corpora/vimd/data/valid-*.parquet", "filename", "text", None, True),
    "vimd/test": ("data/corpora/vimd/data/test-*.parquet", "filename", "text", None, True),
    "fpt_fosd/train": ("data/corpora/fpt_fosd/data/train-*.parquet", None, "transcription", None, True),
    "phoaudiobook_200h/train": ("data/corpora/phoaudiobook_200h/data/train-*.parquet", None, "text", None, True),
    "phoaudiobook_200h/validation": ("data/corpora/phoaudiobook_200h/data/validation-*.parquet", None, "text", None, True),
    "phoaudiobook_200h/test": ("data/corpora/phoaudiobook_200h/data/test-*.parquet", None, "text", None, True),
    "phoaudiobook/train": ("data/corpora/phoaudiobook/data/train-*.parquet", None, "text", None, True),
    "phoaudiobook/validation": ("data/corpora/phoaudiobook/data/validation-*.parquet", None, "text", None, True),
    "phoaudiobook/test": ("data/corpora/phoaudiobook/data/test-*.parquet", None, "text", None, True),
}


def unpack_shard(args):
    """Decode one parquet shard to FLAC files; return its manifest rows."""
    shard_path, id_column, text_column, clean, suffix_ids, audio_dir = args
    rows = []
    columns = ([id_column] if id_column else []) + [text_column, "audio"]
    # row batches without pre-buffering, not the whole table: VoxPopuli / TED-LIUM shards are GBs and this runs next to
    # training (~2 GB peak per worker on a VoxPopuli shard; 8 workers with pre-buffering got the session OOM-killed)
    batches = pq.ParquetFile(shard_path, pre_buffer=False, buffer_size=1 << 22).iter_batches(batch_size=64, columns=columns)
    records = (record for batch in batches for record in batch.to_pylist())
    shard_tag = Path(shard_path).stem.split("-")[1] if "-" in Path(shard_path).stem else Path(shard_path).stem
    for index, record in enumerate(records):
        text = re.sub(r"\s+", " ", (record[text_column] or "")).strip()
        if clean:
            text = clean(text).strip()
        if not text:
            continue
        raw_id = record[id_column] if id_column else Path(record["audio"].get("path") or f"utt_{index:06d}").stem
        utt_id = re.sub(r"[^\w.-]", "", str(raw_id)) or f"utt_{index:06d}"  # TED-LIUM ids contain "<NA>"
        if suffix_ids:
            utt_id = f"{utt_id}_{shard_tag}_{index}"
        waveform, sample_rate = sf.read(io.BytesIO(record["audio"]["bytes"]), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(1)
        if sample_rate != SR:
            import soxr
            waveform = soxr.resample(waveform, sample_rate, SR).astype(np.float32)
        audio_path = audio_dir / f"{utt_id}.flac"
        sf.write(audio_path, waveform, SR)
        rows.append({"id": str(utt_id), "path": str(audio_path), "text": text, "sec": round(len(waveform) / SR, 3)})
    return rows


def unpack(name, workers=4):
    split_dir = OUT / name
    if (split_dir / ".done").exists():
        print("skip", name)
        return
    pattern, id_column, text_column, clean, suffix_ids = SOURCES[name]
    shards = sorted(Path().glob(pattern))
    if not shards:
        raise SystemExit(f"no parquet shards for {name} ({pattern}); run download_corpora.py first")
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        shard_rows = list(pool.map(unpack_shard, [(str(s), id_column, text_column, clean, suffix_ids, audio_dir) for s in shards]))
    rows = sorted((row for rows in shard_rows for row in rows), key=lambda row: row["id"])
    with open(split_dir / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (split_dir / ".done").touch()
    print(f"{name}: {len(rows)} utterances, {sum(r['sec'] for r in rows) / 3600:.1f} h")


def main():
    parser = argparse.ArgumentParser(description="Unpack downloaded HF-parquet ASR corpora to 16 kHz FLAC + manifest.jsonl.")
    parser.add_argument("splits", nargs="*", help=f"Specific <corpus>/<split> targets (default: all {len(SOURCES)} registered)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel shard worker processes (use 2 alongside GPU training)")
    parser.add_argument("--list", action="store_true", help="List registered corpus splits and unpack status")
    args = parser.parse_args()

    if args.list:
        for name, (pattern, *_) in SOURCES.items():
            done = (OUT / name / ".done").exists()
            raw = bool(list(Path().glob(pattern)))
            state = "done" if done else ("ready" if raw else "no_raw")
            print(f"{name:28} {state:8} {pattern}")
        return

    for name in args.splits or SOURCES:
        unpack(name, workers=args.workers)


if __name__ == "__main__":
    main()
