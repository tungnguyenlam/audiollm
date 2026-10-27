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
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

SR = 16000
OUT = Path("data/asr_corpora")
# "<corpus>/<split>" -> (parquet glob, id column or None = derive, text column)
SOURCES = {
    "ami_ihm/train": ("data/corpora/ami_ihm/ihm/train-*.parquet", "audio_id", "text"),
    "ami_ihm/validation": ("data/corpora/ami_ihm/ihm/validation-*.parquet", "audio_id", "text"),
    "fleurs_en/train": ("data/corpora/fleurs_en/parquet-data/en_us/train-*.parquet", "id", "raw_transcription"),
    "fleurs_en/validation": ("data/corpora/fleurs_en/parquet-data/en_us/validation-*.parquet", "id", "raw_transcription"),
}


def unpack_shard(args):
    """Decode one parquet shard to FLAC files; return its manifest rows."""
    shard_path, id_column, text_column, audio_dir = args
    rows = []
    table = pq.read_table(shard_path, columns=[id_column, text_column, "audio"]).to_pylist()
    for index, record in enumerate(table):
        text = (record[text_column] or "").strip()
        if not text:
            continue
        # FLEURS ids repeat (one id per sentence, several speakers), so suffix shard + row index
        utt_id = f"{record[id_column]}_{Path(shard_path).stem.split('-')[1]}_{index}" if id_column == "id" else record[id_column]
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
    pattern, id_column, text_column = SOURCES[name]
    shards = sorted(Path().glob(pattern))
    if not shards:
        raise SystemExit(f"no parquet shards for {name} ({pattern}); run download_corpora.py first")
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        shard_rows = list(pool.map(unpack_shard, [(str(s), id_column, text_column, audio_dir) for s in shards]))
    rows = sorted((row for rows in shard_rows for row in rows), key=lambda row: row["id"])
    with open(split_dir / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (split_dir / ".done").touch()
    print(f"{name}: {len(rows)} utterances, {sum(r['sec'] for r in rows) / 3600:.1f} h")


if __name__ == "__main__":
    for name in sys.argv[1:] or SOURCES:
        unpack(name)
