"""Front-end lab: what resampling, bit-depth quantisation, the STFT, the mel filterbank, the log and the feature
precision each do to one clip. Mirrors section 3 / 4b of explore_pipeline.ipynb (numpy only, no torch import)."""
import io

import numpy as np
import soundfile as sf

from dataset_viewer.audio import _frames, _htk_mel_filterbank, _periodic_hann, render_png, resample

REF_SR, REF_FFT, REF_HOP = 16000, 512, 64  # fine reference grid (31 Hz x 4 ms) every config is scored against
FLOOR_DB = 80                               # ignore detail more than 80 dB below the clip's peak in the LSD
MAX_SEC = 30.0
FEATURE_PRECISIONS = ("fp32", "bf16", "fp16", "int8")
PRESETS = {
    "train.py default": dict(sr=16000, bits=32, quant="pcm", n_fft=400, hop=160, n_mels=80, stack=16, feat="fp32"),
    "paper rate (STACK 8)": dict(stack=8),
    "128 mels": dict(n_mels=128),
    "40 mels, STACK 32": dict(n_mels=40, stack=32),
    "32 ms window": dict(n_fft=512),
    "20 ms frames": dict(hop=320, n_fft=640, stack=8),
    "telephone 8 kHz": dict(sr=8000, n_fft=200, hop=80),
    "8-bit PCM": dict(bits=8),
    "8-bit mu-law": dict(bits=8, quant="mulaw"),
    "4-bit PCM": dict(bits=4),
    "bf16 features": dict(feat="bf16"),
    "int8 features": dict(feat="int8"),
}


# ---------------------------------------------------------------- signal helpers
def stft(x, n_fft, hop):
    """(samples,) -> (frames, n_fft/2+1) complex, torch.stft(center=True, reflect pad, periodic Hann) layout."""
    return np.fft.rfft(_frames(x, n_fft, hop) * _periodic_hann(n_fft), axis=1)


def istft(spec, n_fft, hop, length):
    """Inverse of stft() by windowed overlap-add."""
    w = _periodic_hann(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * w
    n = n_fft + hop * (len(frames) - 1)
    y, wsum = np.zeros(n), np.zeros(n)
    for i, f in enumerate(frames):
        y[i * hop:i * hop + n_fft] += f
        wsum[i * hop:i * hop + n_fft] += w ** 2
    y /= np.maximum(wsum, 1e-8)
    return y[n_fft // 2:n_fft // 2 + length].astype(np.float32)


def griffin_lim(mag, n_fft, hop, length, n_iter=48, momentum=0.99):
    """Fast Griffin-Lim (same update as torchaudio): a waveform whose STFT magnitude matches `mag` (frames, F)."""
    angles = np.exp(2j * np.pi * np.random.default_rng(0).random(mag.shape))
    prev = 0
    for _ in range(n_iter):
        rebuilt = stft(istft(mag * angles, n_fft, hop, length), n_fft, hop)
        angles = rebuilt - prev * (momentum / (1 + momentum))
        angles /= np.abs(angles) + 1e-16
        prev = rebuilt
    return istft(mag * angles, n_fft, hop, length)


def quantize(x, bits, mode):
    """Uniform PCM (or mu-law companded) quantisation to `bits` bits, like storing the clip at that bit depth."""
    if bits >= 32:
        return x
    levels = 2 ** (bits - 1)
    if mode == "mulaw":
        mu = 2 ** bits - 1
        y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        y = np.clip(np.round(y * levels), -levels, levels - 1) / levels
        return (np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu).astype(np.float32)
    return (np.clip(np.round(x * levels), -levels, levels - 1) / levels).astype(np.float32)


def quantize_features(log_mel, prec):
    """Storage / compute precision of the log-mel the adapter receives (train.py: bf16 under PRECISION=bf16)."""
    if prec == "bf16":  # round-to-nearest-even on the top 16 bits of float32
        u = log_mel.astype(np.float32).view(np.uint32)
        u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
        return u.view(np.float32)
    if prec == "fp16":
        return log_mel.astype(np.float16).astype(np.float32)
    if prec == "int8":  # per-clip min/max, 256 levels
        lo, hi = float(log_mel.min()), float(log_mel.max())
        step = max(hi - lo, 1e-6) / 255
        return (np.round((log_mel - lo) / step) * step + lo).astype(np.float32)
    return log_mel


def mel_to_linear(mel_power, fb):
    """(N_MELS, T) mel power -> (F, T) linear power: each FFT bin gets the filter-weighted average of the mean power
    per filter (a flat spectrum inside each filter); bins no filter covers get 0."""
    per_filter = mel_power / np.maximum(fb.sum(1, keepdims=True), 1e-8)
    return fb.T @ per_filter / np.maximum(fb.sum(0), 1e-8)[:, None]


def psd_db(power, n_fft, sr):
    """|X|^2 / sum(w^2) / sr in dB: a power spectral density, comparable across window lengths and sample rates."""
    return 10 * np.log10(power / (_periodic_hann(n_fft) ** 2).sum() / sr + 1e-20)


def _interp_matrix(src, dst):
    """Linear interpolation from sample points `src` to `dst` as a (len(dst), len(src)) matrix."""
    m = np.zeros((len(dst), len(src)))
    j = np.clip(np.searchsorted(src, dst) - 1, 0, len(src) - 2) if len(src) > 1 else np.zeros(len(dst), int)
    if len(src) == 1:
        m[:, 0] = 1
        return m
    t = np.clip((dst - src[j]) / (src[j + 1] - src[j]), 0, 1)
    m[np.arange(len(dst)), j] = 1 - t
    m[np.arange(len(dst)), j + 1] = t
    return m


def to_ref_grid(db, sr, n_fft, hop, ref_shape, fill):
    """(F, T) dB on a config's grid -> the (REF_FFT/2+1, T_ref) reference grid. Bins above the config's Nyquist carry
    no information and get `fill` (the clip's mean reference level, as in the notebook's sweep)."""
    src_f, dst_f = np.fft.rfftfreq(n_fft, 1 / sr), np.fft.rfftfreq(REF_FFT, 1 / REF_SR)
    src_t, dst_t = np.arange(db.shape[1]) * hop / sr, np.arange(ref_shape[1]) * REF_HOP / REF_SR
    keep = dst_f <= sr / 2
    out = np.full(ref_shape, fill)
    out[keep] = _interp_matrix(src_f, dst_f[keep]) @ db @ _interp_matrix(src_t, dst_t).T
    return out


def lsd(db, ref):
    """Log-spectral distance (dB): RMS over frequency per frame, averaged over frames, with an 80 dB floor."""
    lo = ref.max() - FLOOR_DB
    return float(np.sqrt(((np.maximum(db, lo) - np.maximum(ref, lo)) ** 2).mean(0)).mean())


def envelope(x, n=1200):
    """(n, 2) min/max per pixel column, for drawing a waveform without shipping every sample."""
    if len(x) == 0:
        return []
    edges = np.linspace(0, len(x), n + 1).astype(int)
    return [[round(float(x[a:max(b, a + 1)].min()), 4), round(float(x[a:max(b, a + 1)].max()), 4)]
            for a, b in zip(edges[:-1], edges[1:])]


def wav_bytes(x, sr, subtype="PCM_16"):
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if subtype == "PCM_16" and peak > 0.99:  # Griffin-Lim output can overshoot; scale instead of clipping
        x = x * (0.99 / peak)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def ltas_db(power, n_fft, sr, max_f):
    """Long-term average spectrum in dB at <= 400 points up to max_f."""
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    db = psd_db(power.mean(1), n_fft, sr)
    keep = np.flatnonzero(f <= max_f + 1e-6)
    keep = keep[np.linspace(0, len(keep) - 1, min(400, len(keep))).astype(int)]
    return {"f": [round(float(v), 1) for v in f[keep]], "db": [round(float(v), 2) for v in db[keep]]}


# ---------------------------------------------------------------- main entry
def validate(p):
    """Clamp a parameter dict into a safe range; returns the cleaned dict."""
    q = dict(PRESETS["train.py default"]) | {k: v for k, v in p.items() if v is not None}
    q["sr"] = int(np.clip(int(q["sr"]), 4000, 48000))
    q["bits"] = int(np.clip(int(q["bits"]), 2, 32))
    q["quant"] = q["quant"] if q["quant"] in ("pcm", "mulaw") else "pcm"
    q["n_fft"] = int(np.clip(int(q["n_fft"]), 32, 4096))
    q["hop"] = int(np.clip(int(q["hop"]), 8, 4096))
    q["n_mels"] = int(np.clip(int(q["n_mels"]), 4, 256))
    q["stack"] = int(np.clip(int(q["stack"]), 1, 128))
    q["feat"] = q["feat"] if q["feat"] in FEATURE_PRECISIONS else "fp32"
    q["start"] = max(0.0, float(q.get("start") or 0.0))
    q["dur"] = float(np.clip(float(q.get("dur") or 8.0), 0.2, MAX_SEC))
    return q


def analyse(x0, sr0, p, hidden_size=1024):
    """Run one clip through resample -> quantise -> STFT -> mel -> log -> feature precision -> MVN -> stack, and every
    reconstruction back to audio. Returns (json-able summary, {name: bytes} of PNGs and WAVs)."""
    a = int(round(p["start"] * sr0))
    x0 = x0[a:a + int(round(p["dur"] * sr0))]
    if len(x0) < 64:
        raise ValueError("selected region is too short")
    sr, n_fft, hop, n_mels, stack = p["sr"], p["n_fft"], p["hop"], p["n_mels"], p["stack"]
    files, notes = {}, []

    # 1. waveform: resample, then quantise (= a recording stored at SR with BITS bits)
    xr = resample(x0, sr0, sr)
    xq = quantize(xr, p["bits"], p["quant"])
    err = xq - xr
    q_snr = 10 * np.log10((xr.astype(np.float64) ** 2).sum() / max((err.astype(np.float64) ** 2).sum(), 1e-20))

    # 2. STFT -> power -> mel -> log, exactly train.log_mel_from_waveform at these settings
    spec = stft(xq, n_fft, hop)
    power = (np.abs(spec) ** 2).T                                   # (F, T)
    fb = _htk_mel_filterbank(sr, n_fft, n_mels)
    mel_power = fb @ power                                          # (N_MELS, T)
    log_mel = np.log(mel_power + 1e-6)
    log_mel_q = quantize_features(log_mel, p["feat"])
    feat_err = float(np.abs(log_mel_q - log_mel).max())
    # per-clip MVN (train.py uses train-set stats; per-clip is close enough to see the patch structure)
    norm = (log_mel_q - log_mel_q.mean(1, keepdims=True)) / np.maximum(log_mel_q.std(1, keepdims=True), 1e-5)
    n_frames = norm.shape[1]
    pad = (-n_frames) % stack
    patches = np.concatenate([norm, np.zeros((n_mels, pad))], 1).T.reshape(-1, stack * n_mels)

    # 3. reconstructions to listen to
    length = len(xq)
    gl_stft = griffin_lim(np.abs(spec), n_fft, hop, length)                      # loses phase only
    lin_from_mel = mel_to_linear(np.exp(log_mel_q) - 1e-6, fb).clip(min=0)      # loses phase + mel smoothing
    gl_mel = griffin_lim(np.sqrt(lin_from_mel.T), n_fft, hop, length)

    # 4. scores against the fine 16 kHz reference of the original
    x16 = resample(x0, sr0, REF_SR)
    ref = psd_db((np.abs(stft(x16, REF_FFT, REF_HOP)) ** 2).T, REF_FFT, REF_SR)
    fill = float(ref.mean())
    wave_db = psd_db((np.abs(stft(resample(xq, sr, REF_SR), REF_FFT, REF_HOP)) ** 2).T, REF_FFT, REF_SR)
    if sr < REF_SR:
        wave_db[np.fft.rfftfreq(REF_FFT, 1 / REF_SR) > sr / 2] = fill
    mel_db = to_ref_grid(psd_db(lin_from_mel, n_fft, sr), sr, n_fft, hop, ref.shape, fill)

    dead = int(((fb > 0).sum(1) == 0).sum())
    if dead:
        notes.append(f"{dead} mel filter(s) cover no FFT bin: raise N_FFT or lower N_MELS")
    if hop > n_fft:
        notes.append("HOP > N_FFT: some samples fall between windows and are never seen")
    if sr > sr0:
        notes.append(f"SR above the file's native {sr0} Hz: upsampling adds an empty band, no information")
    if sr != 16000:
        notes.append("train.py asserts file SR == SR; training at this SR needs a resampling step first")
    seconds = len(xq) / sr
    stats = {
        "native_sr": sr0, "seconds": round(seconds, 3), "quant_snr_db": round(float(q_snr), 1) if p["bits"] < 32 else None,
        "win_ms": 1000 * n_fft / sr, "hz_per_bin": sr / n_fft, "fps": sr / hop, "token_hz": sr / hop / stack,
        "ms_per_token": 1000 * stack * hop / sr, "frames": n_frames, "tokens": int(patches.shape[0]),
        "patch_dim": stack * n_mels, "dim_over_h": stack * n_mels / hidden_size, "hidden_size": hidden_size,
        "dead_filters": dead, "feat_max_err": round(feat_err, 5),
        "lsd_wave": round(lsd(wave_db, ref), 2), "lsd_mel": round(lsd(mel_db, ref), 2),
        "overlap": round(1 - hop / n_fft, 3),
    }

    # images (no axes; the UI draws them). Mel images use the mel axis, linear ones Hz.
    files["img_lin"] = render_png(psd_db(power, n_fft, sr), dynamic_range=90)
    files["img_melpow"] = render_png(mel_power, cmap="magma")       # linear energy: a few loud bins dominate
    files["img_logmel"] = render_png(log_mel_q, cmap="magma")
    files["img_norm"] = render_png(np.clip(norm, -3, 3), cmap="RdBu_r")
    files["wav_orig"] = wav_bytes(x0, sr0, "FLOAT")
    files["wav_wave"] = wav_bytes(xq, sr, "FLOAT")                  # exact quantised samples
    files["wav_err"] = wav_bytes(err * (0.5 / max(float(np.abs(err).max()), 1e-9)) if p["bits"] < 32 else err, sr, "FLOAT")
    files["wav_stft"] = wav_bytes(gl_stft, sr)
    files["wav_mel"] = wav_bytes(gl_mel, sr)

    fb_step = max(1, int(np.ceil(n_mels / 40)))
    fb_f = np.fft.rfftfreq(n_fft, 1 / sr)
    fb_keep = np.linspace(0, len(fb_f) - 1, min(len(fb_f), 600)).astype(int)
    summary = {
        "params": p, "stats": stats, "notes": notes,
        "env": {"orig": envelope(resample(x0, sr0, sr)), "proc": envelope(xq), "err": envelope(err)},
        "ltas": {"orig": ltas_db((np.abs(stft(x0, 1024 if sr0 > 24000 else 512, max(1, sr0 // 100))) ** 2).T,
                                 1024 if sr0 > 24000 else 512, sr0, sr0 / 2),
                 "proc": ltas_db(power, n_fft, sr, sr / 2),
                 "mel": ltas_db(lin_from_mel, n_fft, sr, sr / 2)},
        "fb": {"f": [round(float(v), 1) for v in fb_f[fb_keep]], "step": fb_step,
               "rows": [[round(float(v), 3) for v in fb[i, fb_keep]] for i in range(0, n_mels, fb_step)],
               "dead": [int(i) for i in np.flatnonzero((fb > 0).sum(1) == 0)]},
        # normalised log-mel for the patch inspector: (frames + pad, N_MELS) row-major float32, base64 in the server
        "norm_shape": [int(patches.shape[0] * stack), n_mels],
    }
    files["norm_f32"] = np.concatenate([norm, np.zeros((n_mels, pad))], 1).T.astype(np.float32).tobytes()
    return summary, files
