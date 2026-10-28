"""
Audio projectors: frame-level features -> LLM input embeddings. One class per file, selected by PROJECTOR.

Adding one: subclass AudioProjector in a new module here, decorate it with @register_projector("<name>"), and
import the module below. `python train.py --projector <name>` then trains it.
"""
from audiollm import config as C
from audiollm.projectors.base import PROJECTORS, AudioProjector, ProjectorOutput, register_projector, stack_frames
from audiollm.projectors import conv, stack  # noqa: F401  (registers "linear", "mlp", "conv")
from audiollm.projectors.conv import ConvAudioProjector


def build_projector(feature_dim, hidden_size, embedding_std):
    """The PROJECTOR class, initialised per PROJ_INIT."""
    if C.PROJECTOR not in PROJECTORS:
        raise ValueError(f"unknown PROJECTOR {C.PROJECTOR!r} (expected one of {list(PROJECTORS)})")
    projector = PROJECTORS[C.PROJECTOR](feature_dim, hidden_size)
    if C.PROJ_INIT == "match_emb":
        projector.init_weights(embedding_std)
    elif C.PROJ_INIT != "default":
        raise ValueError(f"unknown PROJ_INIT {C.PROJ_INIT!r} (expected 'match_emb' or 'default')")
    return projector
