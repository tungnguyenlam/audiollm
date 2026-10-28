"""
Featurizer: waveform -> MVN-normalised frame-level features (num_frames, feature_dim) at SR / HOP frames per second.

Features stay at the full frame rate; grouping frames into tokens is the projector's job (audiollm/projectors/).
"""
import functools
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audiollm import config as C
from audiollm.data import load_split, unweighted_split


@functools.lru_cache(maxsize=None)
def build_htk_mel_filterbank(sr, n_fft, n_mels):
    """
    Construct an HTK-style triangular mel filterbank matrix.

    Returns:
        torch.Tensor: Filterbank weights of shape (n_mels, n_fft // 2 + 1), e.g. (80, 201).
    """
    hz_to_mel = lambda freq_hz: 2595.0 * np.log10(1.0 + freq_hz / 700.0)
    mel_to_hz = lambda mel_val: 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)

    # Linearly spaced points in mel scale converted back to Hz
    mel_bin_frequencies_hz = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fft_frequencies_hz = np.fft.rfftfreq(n_fft, 1.0 / sr)

    filterbank_weights = np.zeros((n_mels, len(fft_frequencies_hz)))
    for mel_idx in range(n_mels):
        left_hz = mel_bin_frequencies_hz[mel_idx]
        center_hz = mel_bin_frequencies_hz[mel_idx + 1]
        right_hz = mel_bin_frequencies_hz[mel_idx + 2]
        # Triangular filter response
        filterbank_weights[mel_idx] = np.maximum(
            0.0,
            np.minimum(
                (fft_frequencies_hz - left_hz) / (center_hz - left_hz),
                (right_hz - fft_frequencies_hz) / (right_hz - center_hz),
            ),
        )
    return torch.tensor(filterbank_weights, dtype=torch.float32)


@functools.lru_cache(maxsize=None)
def stft_window(n_fft):
    return torch.hann_window(n_fft)


def log_mel_from_waveform(waveform):
    """(num_samples,) float32 mono waveform at SR -> (num_frames, N_MELS) log mel-spectrogram."""
    # Short-Time Fourier Transform (STFT) -> Power Spectrogram
    stft_complex = torch.stft(
        torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)),
        n_fft=C.N_FFT,
        hop_length=C.HOP,
        window=stft_window(C.N_FFT),
        return_complex=True,
    )
    power_spectrogram = stft_complex.abs() ** 2

    # Project power spectrogram onto mel filterbank bins and compute log
    mel_energies = build_htk_mel_filterbank(C.SR, C.N_FFT, C.N_MELS) @ power_spectrogram
    log_mel = torch.log(mel_energies + 1e-6).T  # Transpose to shape (num_frames, N_MELS)
    return log_mel


def extract_log_mel_spectrogram(audio_filepath):
    """Load an SR audio file -> (num_frames, N_MELS) log mel-spectrogram."""
    waveform, sample_rate = sf.read(audio_filepath, dtype="float32")
    assert sample_rate == C.SR, f"Expected sample rate {C.SR}, but got {sample_rate} for {audio_filepath}"
    return log_mel_from_waveform(waveform)


def mvn_split():
    """Split spec whose MVN stats are used: MVN_SPLIT, else TRAIN_SPLIT without "*K" weights."""
    return C.MVN_SPLIT or unweighted_split(C.TRAIN_SPLIT)


def mvn_cache_path():
    """MVN stats depend on the split, sample size and every feature-extraction knob, so all of them are in the name."""
    return Path(C.DATA, f"mvn_{mvn_split().replace('/', '-')}_n{C.MVN_N}_sr{C.SR}_fft{C.N_FFT}_hop{C.HOP}_mel{C.N_MELS}.pt")


def compute_or_load_mvn_stats(dataset_items):
    """
    Compute global Mean-Variance Normalization (MVN) statistics over a sample of train utterances.
    Caches the resulting mean and std vectors in mvn_cache_path(). dataset_items is only used when mvn_split() is
    the (unweighted) TRAIN_SPLIT; otherwise the MVN split is loaded.

    Returns:
        dict: {"mean": torch.Tensor(N_MELS,), "std": torch.Tensor(N_MELS,)}
    """
    cache_filepath = mvn_cache_path()
    if cache_filepath.exists():
        return torch.load(cache_filepath)
    if mvn_split() != C.TRAIN_SPLIT:
        dataset_items = [item for item in load_split(mvn_split()) if C.MIN_TRAIN_SEC <= item["sec"] <= C.MAX_TRAIN_SEC]

    deterministic_rng = random.Random(C.SEED)
    sampled_items = deterministic_rng.sample(dataset_items, min(C.MVN_N, len(dataset_items)))
    concatenated_mel_features = torch.cat(
        [extract_log_mel_spectrogram(item["path"]) for item in sampled_items]
    )
    mvn_stats = {
        "mean": concatenated_mel_features.mean(dim=0),
        "std": concatenated_mel_features.std(dim=0),
    }
    torch.save(mvn_stats, cache_filepath)
    return mvn_stats


class LogMelFeaturizer:
    """
    MVN-normalised 80-d HTK log-mel at 100 Hz (SR, N_FFT, HOP, N_MELS). The single featurizer for now; a new one
    (another mel count, raw waveform patches, ...) implements the same three members.
    """

    def __init__(self, mvn_stats):
        self.mvn_stats = mvn_stats
        self.feature_dim = C.N_MELS
        self.frame_rate = C.SR / C.HOP  # frames per second

    def from_waveform(self, waveform):
        return (log_mel_from_waveform(waveform) - self.mvn_stats["mean"]) / self.mvn_stats["std"]

    def __call__(self, audio_filepath):
        """(num_frames, feature_dim) normalised features of one audio file."""
        return (extract_log_mel_spectrogram(audio_filepath) - self.mvn_stats["mean"]) / self.mvn_stats["std"]
