"""
Unpack the HF-parquet ASR corpora in data/corpora/ (download_corpora.py) into the file layout train.py reads:

    data/asr_corpora/<corpus>/<split>/audio/<id>.flac     16 kHz mono FLAC (resampled / downmixed if needed)
    data/asr_corpora/<corpus>/<split>/manifest.jsonl      {"id", "path", "text", "sec"} per utterance, raw text

train.py then accepts "<corpus>/<split>" anywhere it takes a split name (e.g. --train_split
"train-clean-100+train-clean-360+ami_ihm/train+fleurs_en/train"). Text is stored raw; train.py normalises it.

    python unpack_parquet_asr_corpora.py                     # every corpus/split below (skips finished ones)
    python unpack_parquet_asr_corpora.py ami_ihm/train       # just one
"""
import io
import json
import re
import sys
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
# "<corpus>/<split>" -> (parquet glob, id column, text column, text cleaner or None, suffix ids with shard + row?)
SOURCES = {
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
}


def unpack_shard(args):
    """Decode one parquet shard to FLAC files; return its manifest rows."""
    shard_path, id_column, text_column, clean, suffix_ids, audio_dir = args
    rows = []
    # row batches without pre-buffering, not the whole table: VoxPopuli / TED-LIUM shards are GBs and this runs next to
    # training (~2 GB peak per worker on a VoxPopuli shard; 8 workers with pre-buffering got the session OOM-killed)
    batches = pq.ParquetFile(shard_path, pre_buffer=False, buffer_size=1 << 22).iter_batches(batch_size=64, columns=[id_column, text_column, "audio"])
    records = (record for batch in batches for record in batch.to_pylist())
    for index, record in enumerate(records):
        text = (record[text_column] or "").strip()
        if clean:
            text = clean(text).strip()
        if not text:
            continue
        utt_id = re.sub(r"[^\w.-]", "", str(record[id_column]))  # TED-LIUM ids contain "<NA>"
        if suffix_ids:
            utt_id = f"{utt_id}_{Path(shard_path).stem.split('-')[1]}_{index}"
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


def unpack(name):
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
    with ProcessPoolExecutor(max_workers=4) as pool:
        shard_rows = list(pool.map(unpack_shard, [(str(s), id_column, text_column, clean, suffix_ids, audio_dir) for s in shards]))
    rows = sorted((row for rows in shard_rows for row in rows), key=lambda row: row["id"])
    with open(split_dir / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (split_dir / ".done").touch()
    print(f"{name}: {len(rows)} utterances, {sum(r['sec'] for r in rows) / 3600:.1f} h")


if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        unpack(name)
