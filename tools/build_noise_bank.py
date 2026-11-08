"""
Noise bank for additive-noise augmentation (NOISE_BANK): ESC-50 (2,000 environmental clips) + Clotho-AQA clips
(environmental recordings, deduplicated), mono, band-limited resampled to 16 kHz, each clip RMS-normalised, near-silent
clips dropped, concatenated into one float16 .npy that DataLoader workers memory-map and crop at random.

    python tools/build_noise_bank.py --out data/noise_bank_16k.npy
"""
import argparse
import glob
import io

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

SR = 16000


def resample(x, sr):
    """Exact band-limited resampling via rFFT truncation / zero-padding."""
    if sr == SR:
        return x
    n_out = int(round(len(x) * SR / sr))
    spec = torch.fft.rfft(torch.from_numpy(x).double())
    keep = n_out // 2 + 1
    spec = spec[:keep] if spec.shape[0] >= keep else torch.cat([spec, spec.new_zeros(keep - spec.shape[0])])
    return (torch.fft.irfft(spec, n=n_out) * (n_out / len(x))).float().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/noise_bank_16k.npy")
    args = parser.parse_args()
    clips, seen = [], set()
    files = sorted(glob.glob("data/corpora/esc50/**/*.parquet", recursive=True)) + \
        sorted(glob.glob("data/corpora/clotho_aqa/**/*.parquet", recursive=True))
    for path in files:
        table = pq.read_table(path, columns=["audio"]).to_pylist()
        for row in table:
            audio = row["audio"]
            key = audio.get("path") or hash(audio["bytes"][:4096])
            if key in seen:
                continue
            seen.add(key)
            x, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            x = resample(x.mean(1), sr)
            rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
            if rms < 1e-4 or len(x) < SR:
                continue
            clips.append((x / rms * 0.1).astype(np.float16))
    bank = np.concatenate(clips)
    np.save(args.out, bank)
    print(f"{len(clips)} clips, {len(bank) / SR / 3600:.2f} h -> {args.out}")


if __name__ == "__main__":
    main()
