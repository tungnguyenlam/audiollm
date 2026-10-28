"""Mel-LLM / NeMo-style strided conv subsampler on continuous frames."""
import math

import torch.nn as nn
import torch.nn.functional as F

from audiollm import config as C
from audiollm.projectors.base import AudioProjector, ProjectorOutput, register_projector, stack_frames


@register_projector("conv")
class ConvAudioProjector(AudioProjector):
    """
    log2(STACK) kernel-3 stride-2 1D convs over time (100 -> 50 -> ... Hz) + LayerNorm + 2-layer GELU MLP. Windows
    cross STACK-patch boundaries, so each token sees 2 * STACK - 1 frames. Each conv is unfold + Linear (a rocBLAS
    GEMM) to avoid MIOpen dynamic-length stalls.
    """

    def __init__(self, feature_dim, hidden_size):
        super().__init__(feature_dim, hidden_size)
        self.stack = C.STACK
        n_stages = int(round(math.log2(self.stack)))
        if 2 ** n_stages != self.stack or n_stages < 1:
            raise ValueError(f"PROJECTOR='conv' requires STACK to be a power of 2 >= 2, got {self.stack}")
        self.convs = nn.ModuleList()
        in_channels = feature_dim
        for stage in range(n_stages):
            out_channels = hidden_size // 2 if stage == 0 and n_stages > 1 else hidden_size
            self.convs.append(nn.Linear(3 * in_channels, out_channels))
            in_channels = out_channels
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_size, hidden_size)

    @property
    def output_layer(self):
        return self.out

    def num_tokens(self, num_frames):
        return -(-num_frames // self.stack)

    def describe(self):
        rates = " -> ".join(f"{C.SR / C.HOP / 2 ** stage:g}" for stage in range(len(self.convs) + 1))
        widths = " -> ".join(str(conv.out_features) for conv in self.convs)
        return (f"conv: {len(self.convs)} k3/s2 stages {rates} Hz, widths {self.feature_dim} -> {widths}, "
                f"receptive field {2 * self.stack - 1} frames, then LayerNorm + MLP -> {self.hidden_size}-d")

    def forward(self, features, feature_lengths):
        # Pad each utterance to a multiple of STACK, as patches; the convs then halve the time axis exactly
        patches, lengths = stack_frames(features, feature_lengths, self.stack)
        x = patches.reshape(patches.shape[0], -1, self.feature_dim)  # (batch, tokens * STACK, feature_dim)
        valid_frames = lengths * self.stack
        for conv_linear in self.convs:
            windows = F.pad(x, (0, 0, 1, 1)).unfold(1, 3, 2).reshape(x.shape[0], x.shape[1] // 2, -1)
            x = self.act(conv_linear(windows))
            valid_frames = valid_frames // 2
            # Zero positions past each utterance's end, so its last windows see the same zero padding as unbatched
            x = x * (x.new_ones(x.shape[1]).cumsum(0) <= valid_frames[:, None]).unsqueeze(-1).to(x.dtype)
        return ProjectorOutput(self.out(self.act(self.fc1(self.norm(x)))), lengths)
