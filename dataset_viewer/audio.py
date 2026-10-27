"""Audio decoding, quality metrics and spectrogram rendering (numpy only, no torch import)."""
import io

import numpy as np
import soundfile as sf

MODEL_SR = 16000  # train.py: SR
N_FFT = 400       # train.py: 25 ms window
HOP = 160         # train.py: 10 ms hop
N_MELS = 80

# Thresholds behind each automatic flag. Served to the UI so the meaning of every flag is visible.
THRESHOLDS = {
    "too_short": ("duration_s", "<", 0.5, "shorter than 0.5 s"),
    "too_long": ("duration_s", ">", 30.0, "longer than 30 s"),
    "low_sr": ("sample_rate", "<", 16000, "native sample rate below 16 kHz"),
    "clipping": ("clip_ratio", ">", 0.001, "more than 0.1% of samples sit in flat-topped runs at the peak"),
    "quiet": ("rms_dbfs", "<", -40.0, "RMS level below -40 dBFS"),
    "low_snr": ("snr_db", "<", 15.0, "estimated SNR below 15 dB"),
    "mostly_silent": ("silence_ratio", ">", 0.6, "more than 60% of 25 ms frames are silent"),
    "narrowband": ("bandwidth_hz", "<", 4500.0, "no energy above ~4.5 kHz although the file could hold it"),
    "dc_offset": ("dc_offset", ">", 0.02, "|mean sample value| above 0.02"),
    "fast_speech": ("chars_per_sec", ">", 25.0, "transcript has >25 chars/s: text may not match audio"),
    "slow_speech": ("chars_per_sec", "<", 4.0, "transcript has <4 chars/s: text may be truncated or audio padded"),
    "empty_label": (None, None, None, "the transcript / label is empty"),
    "decode_error": (None, None, None, "the audio could not be decoded"),
}


def decode(source, default_sr=MODEL_SR):
    """bytes | path | {'bytes','path'} | float array -> (mono float32, sr, n_channels)."""
    if isinstance(source, dict):
        source = source.get("bytes") or source.get("path")
    if isinstance(source, np.ndarray):
        x, sr = source.astype(np.float32, copy=False), default_sr
    else:
        if isinstance(source, (bytes, bytearray, memoryview)):
            source = io.BytesIO(bytes(source))
        x, sr = sf.read(source, dtype="float32", always_2d=True)
        channels = x.shape[1]
        return np.ascontiguousarray(x.mean(axis=1)), int(sr), channels
    return x, int(sr), 1


def resample(x, sr, target_sr):
    """Band-limited FFT resampling. Good enough for viewing; avoids a scipy/torchaudio dependency."""
    if sr == target_sr or len(x) == 0:
        return x
    n_out = int(round(len(x) * target_sr / sr))
    spec = np.fft.rfft(x)
    n_bins = n_out // 2 + 1
    if n_bins <= len(spec):
        spec = spec[:n_bins]
    else:
        spec = np.concatenate([spec, np.zeros(n_bins - len(spec), dtype=spec.dtype)])
    return (np.fft.irfft(spec, n=n_out) * (n_out / len(x))).astype(np.float32)


def _frames(x, n_fft, hop, center=True):
    if center:
        pad = n_fft // 2
        mode = "reflect" if len(x) > pad else "constant"
        x = np.pad(x, (pad, pad), mode=mode)
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n = 1 + (len(x) - n_fft) // hop
    return np.lib.stride_tricks.as_strided(x, (n, n_fft), (x.strides[0] * hop, x.strides[0]))


def _periodic_hann(n):
    # torch.hann_window default (periodic=True), as used by train.py
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def _htk_mel_filterbank(sr=MODEL_SR, n_fft=N_FFT, n_mels=N_MELS):
    """Same construction as train.build_htk_mel_filterbank."""
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    edges = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        l, c, r = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.maximum(0.0, np.minimum((freqs - l) / (c - l), (r - freqs) / (r - c)))
    return fb


_MEL_FB = _htk_mel_filterbank()


def log_mel(x, sr):
    """The exact feature the model sees: 16 kHz, 80-bin HTK log-mel, 25 ms / 10 ms. Returns (80, T)."""
    x16 = resample(x, sr, MODEL_SR)
    power = np.abs(np.fft.rfft(_frames(x16, N_FFT, HOP) * _periodic_hann(N_FFT), axis=1)) ** 2
    return np.log(_MEL_FB @ power.T + 1e-6)


def linear_spec_db(x, sr):
    """Linear-frequency STFT at the native rate, in dB (F, T). Shows the true bandwidth of the file."""
    n_fft = 1024 if sr > 24000 else 512
    hop = max(1, int(sr * 0.01))
    power = np.abs(np.fft.rfft(_frames(x, n_fft, hop) * _periodic_hann(n_fft), axis=1)) ** 2
    return 10 * np.log10(power.T + 1e-10)


def render_png(mat, cmap="magma", max_width=3000, dynamic_range=None):
    """(F, T) matrix -> PNG bytes, low frequencies at the bottom. No axes: the UI draws them."""
    import matplotlib
    from PIL import Image

    if mat.shape[1] > max_width:  # max-pool time so long clips stay small without hiding events
        k = int(np.ceil(mat.shape[1] / max_width))
        t = mat.shape[1] // k * k
        mat = mat[:, :t].reshape(mat.shape[0], -1, k).max(axis=2)
    hi = np.percentile(mat, 99.5)
    lo = hi - dynamic_range if dynamic_range else np.percentile(mat, 1)
    norm = np.clip((mat - lo) / max(hi - lo, 1e-6), 0, 1)[::-1]
    rgb = (matplotlib.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _clip_ratio(x, peak):
    """Fraction of samples in runs of >=3 consecutive samples at (or within 0.1% of) the peak level."""
    if peak < 0.05:
        return 0.0
    at_peak = np.abs(x) >= peak * 0.999
    if at_peak.sum() < 3:
        return 0.0
    d = np.diff(np.concatenate([[0], at_peak.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    runs = ends - starts
    return float(runs[runs >= 3].sum() / len(x))


def _bandwidth(x, sr):
    """Highest frequency whose long-term level is within 50 dB of the spectral peak (smoothed)."""
    if len(x) < 2048:
        return float(sr / 2)
    frames = _frames(x, 2048, 1024, center=False)
    if len(frames) > 400:
        frames = frames[np.linspace(0, len(frames) - 1, 400).astype(int)]
    psd = (np.abs(np.fft.rfft(frames * _periodic_hann(2048), axis=1)) ** 2).mean(axis=0)
    db = 10 * np.log10(np.convolve(psd, np.ones(9) / 9, mode="same") + 1e-12)
    above = np.flatnonzero(db[4:] > db[4:].max() - 50) + 4  # ignore DC bins
    freqs = np.fft.rfftfreq(2048, 1.0 / sr)
    return float(freqs[above[-1]]) if len(above) else 0.0


def compute_metrics(x, sr, channels=1, text=None, is_transcript=False):
    """Quality metrics + automatic flags for one clip. All values are JSON-safe floats."""
    n = len(x)
    duration = n / sr if sr else 0.0
    m = {"duration_s": round(duration, 3), "sample_rate": sr, "channels": channels}
    if n == 0 or not np.isfinite(x).all():
        m.update(rms_dbfs=-120.0, peak_dbfs=-120.0, clip_ratio=0.0, silence_ratio=1.0, snr_db=0.0,
                 bandwidth_hz=0.0, dc_offset=0.0, lead_silence_s=0.0, trail_silence_s=0.0, chars_per_sec=None)
        m["flags"] = ["decode_error"] if n else ["too_short"]
        return m

    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    frame_db = 10 * np.log10(np.mean(_frames(x, int(sr * 0.025), int(sr * 0.010), center=False) ** 2, axis=1) + 1e-12)
    p95, p10 = np.percentile(frame_db, 95), np.percentile(frame_db, 10)
    active = frame_db > max(-55.0, p95 - 35.0)
    act_idx = np.flatnonzero(active)
    chars_per_sec = None
    if is_transcript and text is not None:
        n_chars = sum(ch.isalnum() for ch in text)
        chars_per_sec = round(n_chars / duration, 2) if duration > 0 else None

    m.update(
        peak_dbfs=round(20 * np.log10(peak + 1e-12), 2),
        rms_dbfs=round(20 * np.log10(rms + 1e-12), 2),
        clip_ratio=round(_clip_ratio(x, peak), 6),
        silence_ratio=round(1.0 - float(active.mean()), 4),
        # Frame-energy dynamic range: loud (speech/event) frames vs the quietest frames (noise floor).
        snr_db=round(float(p95 - p10), 2),
        bandwidth_hz=round(_bandwidth(x, sr), 1),
        dc_offset=round(float(x.mean()), 5),
        lead_silence_s=round(act_idx[0] * 0.010, 3) if len(act_idx) else round(duration, 3),
        trail_silence_s=round(max(0.0, duration - (act_idx[-1] * 0.010 + 0.025)), 3) if len(act_idx) else round(duration, 3),
        chars_per_sec=chars_per_sec,
    )

    flags = []
    for name, (key, op, thr, _) in THRESHOLDS.items():
        if key is None:
            continue
        v = m.get(key)
        if v is None:
            continue
        if key == "dc_offset":
            v = abs(v)
        if name == "narrowband" and sr < 11000:
            continue  # a 8 kHz file cannot hold more; low_sr covers it
        if name == "slow_speech" and (text is None or len(text.strip()) < 10):
            continue
        if (op == "<" and v < thr) or (op == ">" and v > thr):
            flags.append(name)
    if is_transcript and (text is None or not text.strip()):
        flags.append("empty_label")
    m["flags"] = flags
    return m
