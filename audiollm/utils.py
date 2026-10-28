"""Small helpers shared by training and evaluation: seeding, progress bars, precision."""
import contextlib
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

from audiollm import config as C


def seed_everything(seed):
    """Seed all pseudo-random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_progress_bar(**tqdm_kwargs):
    """
    Create a tqdm progress bar. When stderr is not a terminal (e.g. `> train.log 2>&1`),
    refresh rarely so the redirected log is not flooded with carriage-return updates.
    """
    is_interactive = sys.stderr.isatty()
    return tqdm(
        dynamic_ncols=True,
        mininterval=0.5 if is_interactive else 60.0,
        file=sys.stderr,
        **tqdm_kwargs,
    )


def format_duration(seconds):
    """Format a duration in seconds as H:MM:SS."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def trainable_dtype():
    """dtype of the trainable weights (LoRA + projector), their grads and the AdamW states."""
    return C.DTYPE if C.PRECISION == "bf16" else torch.float32


def compute_context():
    """
    Context for forward passes. In "amp" mode, autocast runs the fp32 LoRA/projector matmuls in DTYPE
    (the weights and optimizer stay fp32); otherwise every op already runs in its weights' dtype.
    """
    if C.PRECISION == "amp" and C.DTYPE != torch.float32:
        return torch.autocast(device_type=torch.device(C.DEVICE).type, dtype=C.DTYPE)
    return contextlib.nullcontext()
