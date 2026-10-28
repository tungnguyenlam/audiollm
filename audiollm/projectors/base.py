"""AudioProjector: the interface every projector implements, and the registry PROJECTOR names are looked up in."""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

PROJECTORS = {}  # PROJECTOR name -> AudioProjector subclass, filled by @register_projector


def register_projector(name):
    """Class decorator: make a projector selectable as PROJECTOR=name (--projector name)."""
    def decorator(cls):
        cls.name = name
        PROJECTORS[name] = cls
        return cls
    return decorator


@dataclass
class ProjectorOutput:
    embeds: torch.Tensor   # (batch, max_tokens, hidden_size), rows past their length are padding
    lengths: torch.Tensor  # (batch,) audio tokens per utterance; may differ per utterance at equal duration
    aux_loss: torch.Tensor | None = None  # added to the LM loss in training (e.g. a CTC head, a token-rate penalty)
    stats: dict = field(default_factory=dict)  # scalars logged with the train metrics (e.g. merge ratio)


class AudioProjector(nn.Module):
    """
    Frame-level features -> LLM input embeddings. Subclasses implement forward(), num_tokens() and the
    `output_layer` property (the final nn.Linear, initialised to the text-embedding scale under PROJ_INIT="match_emb").

    forward(features (batch, max_frames, feature_dim), feature_lengths (batch,)) -> ProjectorOutput.
    Frames past an utterance's length are zero (= the MVN mean) and must not change its valid tokens.
    """

    name = "?"
    # Older proj.pt files saved a bare module (e.g. nn.Linear: "weight", "bias"); their keys get this prefix on load.
    legacy_key_prefix = ""

    def __init__(self, feature_dim, hidden_size):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size

    def num_tokens(self, num_frames):
        """Audio tokens for an utterance of num_frames frames (expected count for variable-rate projectors)."""
        raise NotImplementedError

    def describe(self):
        """One line for the run log: what the projector does to the frame rate."""
        return self.name

    def init_weights(self, embedding_std):
        """PROJ_INIT="match_emb": Kaiming for hidden layers, output scaled so embeddings ~ the LLM's text embeddings."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)) and module is not self.output_layer:
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")  # keeps GELU activations ~ unit variance
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.output_layer.weight, std=embedding_std / math.sqrt(self.output_layer.in_features))
        nn.init.zeros_(self.output_layer.bias)

    def load_checkpoint(self, state_dict):
        """load_state_dict() that also accepts proj.pt files saved before the projector classes existed."""
        own_keys = set(self.state_dict())
        if self.legacy_key_prefix and not own_keys & set(state_dict):
            state_dict = {self.legacy_key_prefix + key: val for key, val in state_dict.items()}
        self.load_state_dict(state_dict)


def stack_frames(features, feature_lengths, stack):
    """
    (batch, frames, dim) -> (batch, ceil(frames / stack), stack * dim) patches of `stack` consecutive frames, each
    utterance's tail zero-padded to a multiple of `stack`. Returns (patches, tokens per utterance).
    """
    batch_size, num_frames, feature_dim = features.shape
    padding = (-num_frames) % stack
    if padding:
        features = torch.cat([features, features.new_zeros(batch_size, padding, feature_dim)], dim=1)
    return features.reshape(batch_size, -1, stack * feature_dim), (feature_lengths + stack - 1) // stack
