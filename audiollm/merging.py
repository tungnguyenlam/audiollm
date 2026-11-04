"""
Adjacent merging of the projector's audio tokens (MERGE), applied to a ProjectorOutput before the LLM sees it.

Every mode partitions each utterance's tokens into contiguous groups and mean-pools each group into one token, so
time order is kept and padding never leaks into a valid token. Modes differ only in where the group boundaries go:
"affinity" puts them where adjacent embeddings stop being similar (content-adaptive), "uniform" and
"uniform_matched" space them evenly (fixed-rate controls; "uniform_matched" keeps the same per-utterance count as
"affinity", so the two differ only in *where* tokens are merged, not in *how many*).
"""
import torch
import torch.nn.functional as F

from audiollm import config as C
from audiollm.projectors.base import ProjectorOutput

MERGE_MODES = ("none", "affinity", "uniform", "uniform_matched")


def affinity_groups(embeds, lengths, tau, window):
    """
    Affinity Pooling group ids (arXiv 2604.06871, Algorithm 1), one left-to-right pass.

    Token t joins the current group if cos(x_t, x_s) > tau for any of the group's last `window` tokens s; otherwise
    it starts a new group. Returns group ids (batch, tokens), non-decreasing from 0 per row; padding gets -1.
    """
    batch_size, num_tokens, _ = embeds.shape
    valid = torch.arange(num_tokens, device=embeds.device)[None, :] < lengths[:, None]
    unit = F.normalize(embeds.float(), dim=-1)
    if window == 1:
        # Adjacent case: the group's last token is always t - 1, so boundaries are independent and vectorise
        similar = (unit[:, 1:] * unit[:, :-1]).sum(-1) > tau
        starts = torch.cat([torch.ones_like(similar[:, :1]), ~similar], dim=1)
    else:
        starts = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=embeds.device)
        starts[:, 0] = True
        group_start = torch.zeros(batch_size, dtype=torch.long, device=embeds.device)
        for t in range(1, num_tokens):
            joins = torch.zeros(batch_size, dtype=torch.bool, device=embeds.device)
            for lag in range(1, min(window, t) + 1):
                in_group = t - lag >= group_start
                joins |= in_group & ((unit[:, t] * unit[:, t - lag]).sum(-1) > tau)
            starts[:, t] = ~joins
            group_start = torch.where(joins, group_start, torch.full_like(group_start, t))
    group_ids = starts.long().cumsum(1) - 1
    return group_ids.masked_fill(~valid, -1)


def uniform_groups(lengths, keep_counts, num_tokens):
    """Group ids splitting each row's `lengths` tokens into `keep_counts` near-equal contiguous groups; padding = -1."""
    positions = torch.arange(num_tokens, device=lengths.device)[None, :]
    group_ids = positions * keep_counts[:, None] // lengths.clamp(min=1)[:, None]
    return group_ids.masked_fill(positions >= lengths[:, None], -1)


def pool_groups(embeds, group_ids):
    """Mean-pool tokens sharing a group id. Returns (batch, max_groups, hidden) embeddings and groups per row."""
    batch_size, _, hidden = embeds.shape
    new_lengths = group_ids.max(1).values + 1
    max_groups = int(new_lengths.max())
    index = group_ids.masked_fill(group_ids < 0, max_groups)  # padding -> a scratch slot, dropped below
    sums = embeds.new_zeros(batch_size, max_groups + 1, hidden).float()
    sums.scatter_add_(1, index[..., None].expand(-1, -1, hidden), embeds.float())
    counts = torch.zeros(batch_size, max_groups + 1, device=embeds.device)
    counts.scatter_add_(1, index, torch.ones_like(index, dtype=counts.dtype))
    pooled = sums[:, :max_groups] / counts[:, :max_groups, None].clamp(min=1)
    return pooled.to(embeds.dtype), new_lengths


def merge_tokens(projected):
    """Apply C.MERGE to a ProjectorOutput; logs the kept-token fraction as stats["merge_keep"]."""
    if C.MERGE == "none":
        return projected
    if C.MERGE not in MERGE_MODES:
        raise ValueError(f"unknown MERGE {C.MERGE!r} (expected one of {MERGE_MODES})")
    embeds, lengths = projected.embeds, projected.lengths
    with torch.no_grad():  # boundaries are discrete; gradients flow through the mean-pooling only
        if C.MERGE == "uniform":
            keep = torch.ceil(lengths.float() * C.MERGE_RATIO).long().clamp(min=1)
            group_ids = uniform_groups(lengths, torch.minimum(keep, lengths), embeds.shape[1])
        else:
            group_ids = affinity_groups(embeds, lengths, C.MERGE_TAU, C.MERGE_WINDOW)
            if C.MERGE == "uniform_matched":
                group_ids = uniform_groups(lengths, group_ids.max(1).values + 1, embeds.shape[1])
    pooled, new_lengths = pool_groups(embeds, group_ids)
    stats = {**projected.stats, "merge_keep": float(new_lengths.sum() / lengths.sum().clamp(min=1))}
    return ProjectorOutput(pooled, new_lengths, projected.aux_loss, stats)
