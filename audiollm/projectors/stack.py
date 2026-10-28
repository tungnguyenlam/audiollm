"""Fixed-rate projectors on stacked frames: STACK consecutive frames -> one patch -> one token."""
import torch.nn as nn

from audiollm import config as C
from audiollm.projectors.base import AudioProjector, ProjectorOutput, register_projector, stack_frames


class StackProjector(AudioProjector):
    """Base for projectors that see one STACK-frame patch per token, independently of its neighbours."""

    def __init__(self, feature_dim, hidden_size):
        super().__init__(feature_dim, hidden_size)
        self.stack = C.STACK
        self.patch_dim = self.stack * feature_dim

    def num_tokens(self, num_frames):
        return -(-num_frames // self.stack)

    def describe(self):
        return (f"{self.name}: stack {self.stack} frames -> {self.patch_dim}-d patches "
                f"({C.SR / C.HOP / self.stack:.2f} Hz tokens) -> {self.hidden_size}-d")

    def forward(self, features, feature_lengths):
        patches, lengths = stack_frames(features, feature_lengths, self.stack)
        return ProjectorOutput(self.project(patches), lengths)


@register_projector("linear")
class LinearProjector(StackProjector):
    """nn.Linear(STACK * N_MELS -> hidden_size) on each patch."""

    legacy_key_prefix = "linear."

    def __init__(self, feature_dim, hidden_size):
        super().__init__(feature_dim, hidden_size)
        self.linear = nn.Linear(self.patch_dim, hidden_size)

    @property
    def output_layer(self):
        return self.linear

    def project(self, patches):
        return self.linear(patches)


@register_projector("mlp")
class MLPProjector(StackProjector):
    """LayerNorm + 2-layer GELU MLP on each patch."""

    legacy_key_prefix = "mlp."

    def __init__(self, feature_dim, hidden_size):
        super().__init__(feature_dim, hidden_size)
        self.mlp = nn.Sequential(nn.LayerNorm(self.patch_dim), nn.Linear(self.patch_dim, hidden_size), nn.GELU(),
                                 nn.Linear(hidden_size, hidden_size))

    @property
    def output_layer(self):
        return self.mlp[3]

    def project(self, patches):
        return self.mlp(patches)
