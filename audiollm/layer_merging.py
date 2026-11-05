"""
In-LLM audio token merging and exact no-cache greedy decoding for SpeechLLM.

Implements the verified no-cache forward split between layer k-1 and layer k:
- Lower block (layers 0..k-1): processes left-padded [BOS, audio, prompt, generated].
- Prefill: computes audio grouping and pooled representations H'_audio once and freezes them.
- Upper block (layers k..13): processes left-padded [BOS, H'_audio, prompt, generated] with
  recomputed rotary embeddings, cumsum position IDs, and 2D conv padding masks.
"""

from collections.abc import Sequence
import torch
from transformers import NoRepeatNGramLogitsProcessor
from transformers.models.lfm2.modeling_lfm2 import create_causal_mask, create_recurrent_attention_mask

from audiollm import config as C
from audiollm.merging import affinity_groups, uniform_groups, pool_groups
from audiollm.model import decode_token_budget


def make_lfm2_mask_map(model, inputs_embeds, attention_mask, position_ids):
    """Constructs the mask dict {'full_attention': 4D, 'conv': 2D} expected by Lfm2DecoderLayer."""
    mask_kwargs = {
        "config": model.config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    return {
        "full_attention": create_causal_mask(**mask_kwargs),
        "conv": create_recurrent_attention_mask(**mask_kwargs),
    }


def forward_layer_range(model, hidden_states, attention_mask, position_ids, start_layer, end_layer):
    """Runs a contiguous sub-range of decoder layers [start_layer, end_layer)."""
    if start_layer >= end_layer:
        return hidden_states
    mask_map = make_lfm2_mask_map(model, hidden_states, attention_mask, position_ids)
    pos_emb = model.rotary_emb(hidden_states, position_ids=position_ids)
    for i in range(start_layer, end_layer):
        layer = model.layers[i]
        ltype = model.config.layer_types[i]
        hidden_states = layer(
            hidden_states,
            attention_mask=mask_map[ltype],
            position_embeddings=pos_emb,
            position_ids=position_ids,
        )
    return hidden_states


def rebuild_left_padded_batch(rows: Sequence[torch.Tensor], pad_vector: torch.Tensor | None = None):
    """
    Collates a list of 2D row tensors (L_r, D) into a left-padded batch (B, L_max, D)
    along with its 2D binary attention_mask (B, L_max) and position_ids.
    """
    batch_size = len(rows)
    max_len = max(r.shape[0] for r in rows)
    hidden_dim = rows[0].shape[-1]
    device = rows[0].device
    dtype = rows[0].dtype

    if pad_vector is not None:
        batch_tensor = pad_vector.to(device=device, dtype=dtype).expand(batch_size, max_len, hidden_dim).clone()
    else:
        batch_tensor = torch.zeros(batch_size, max_len, hidden_dim, device=device, dtype=dtype)

    attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
    for r, row_tensor in enumerate(rows):
        length = row_tensor.shape[0]
        start = max_len - length
        batch_tensor[r, start:] = row_tensor
        attention_mask[r, start:] = 1

    position_ids = attention_mask.cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return batch_tensor, attention_mask, position_ids


@torch.no_grad()
def greedy_decode_with_layer_merge(
    speech_llm,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_seconds: list[float],
    audio_lengths: torch.Tensor,
    k: int = 0,
    merge_mode: str = "none",
    tau: float = 0.70,
    window: int = 1,
    uniform_keep_counts: list[int] | None = None,
    position_strategy: str = "compact",
):
    """
    Greedy decode a left-padded batch with audio token merging at layer k.
    
    Args:
        speech_llm: SpeechLLM instance.
        inputs_embeds: Initial batch embeds [BOS, audio, prompt] (B, L, D).
        attention_mask: Left-padded mask (B, L).
        audio_seconds: Audio durations for decode budget.
        audio_lengths: Original projected audio lengths per utterance (B,).
        k: Merge layer boundary (0 <= k <= 14). If k=0, merges before layer 0.
        merge_mode: 'none' | 'affinity' | 'uniform_matched' | 'all_dropped' | 'identity'.
        tau: Cosine similarity threshold for affinity.
        window: Lookback window for affinity.
        uniform_keep_counts: Target token counts for uniform_matched.
        position_strategy: 'compact' (0..L'-1) or 'original'.
    """
    llm = speech_llm.llm
    causal_lm = llm.base_model.model if hasattr(llm, "base_model") else llm
    model = causal_lm.model
    lm_head = causal_lm.lm_head
    tokenizer = speech_llm.tokenizer
    device = inputs_embeds.device
    batch_size = inputs_embeds.shape[0]
    budgets = [decode_token_budget(sec) for sec in audio_seconds]
    max_budget = max(budgets)
    eos_token_id = tokenizer.eos_token_id
    pad_embed = causal_lm.get_input_embeddings()(torch.tensor([speech_llm.pad_token_id], device=device))

    has_bos = tokenizer.bos_token_id is not None
    n_bos = int(has_bos)
    n_prompt = len(speech_llm.prompt_token_ids)

    # Calculate token slice boundaries for each utterance in the initial left-padded batch
    audio_spans = []
    prompt_spans = []
    for r in range(batch_size):
        valid_len = int(attention_mask[r].sum().item())
        pad_offset = attention_mask.shape[1] - valid_len
        a0 = pad_offset + n_bos
        a1 = a0 + int(audio_lengths[r].item())
        p0 = a1
        p1 = p0 + n_prompt
        audio_spans.append((a0, a1))
        prompt_spans.append((p0, p1))

    # --- PREFILL STEP ---
    pos_ids_lower = attention_mask.cumsum(-1) - 1
    pos_ids_lower.masked_fill_(attention_mask == 0, 0)

    # Lower block forward (layers 0..k-1)
    h_lower = forward_layer_range(model, inputs_embeds, attention_mask, pos_ids_lower, 0, k)

    # Extract and merge audio representations
    frozen_h_audio = []
    group_stats = []
    for r in range(batch_size):
        a0, a1 = audio_spans[r]
        h_audio_r = h_lower[r, a0:a1]  # (T_audio, D)
        T = h_audio_r.shape[0]

        if merge_mode == "none":
            h_merged = h_audio_r
            g_count = T
        elif merge_mode == "identity":
            h_merged = h_audio_r
            g_count = T
        elif merge_mode == "all_dropped":
            h_merged = h_audio_r[:0]  # Empty tensor (0, D)
            g_count = 0
        elif merge_mode == "affinity":
            # Uncentered raw residual cosine similarity
            embeds = h_audio_r.unsqueeze(0)
            lens = torch.tensor([T], device=device)
            groups = affinity_groups(embeds, lens, tau=tau, window=window)
            pooled, new_lens = pool_groups(embeds, groups)
            h_merged = pooled[0, :new_lens[0]]
            g_count = int(new_lens[0].item())
        elif merge_mode == "uniform_matched":
            embeds = h_audio_r.unsqueeze(0)
            lens = torch.tensor([T], device=device)
            target_count = uniform_keep_counts[r] if uniform_keep_counts is not None else T
            target_counts = torch.tensor([target_count], device=device)
            groups = uniform_groups(lens, target_counts, num_tokens=T)
            pooled, new_lens = pool_groups(embeds, groups)
            h_merged = pooled[0, :new_lens[0]]
            g_count = int(new_lens[0].item())
        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")

        frozen_h_audio.append(h_merged)
        group_stats.append({"orig_len": T, "merged_len": g_count})

    # Assemble upper prefill batch: [BOS, H'_audio, prompt]
    upper_rows = []
    for r in range(batch_size):
        a0, a1 = audio_spans[r]
        p0, p1 = prompt_spans[r]
        bos_part = h_lower[r, a0 - n_bos : a0] if n_bos > 0 else h_lower[r, :0]
        audio_part = frozen_h_audio[r]
        prompt_part = h_lower[r, p0:p1]
        upper_rows.append(torch.cat([bos_part, audio_part, prompt_part], dim=0))

    h_upper, mask_upper, pos_ids_upper = rebuild_left_padded_batch(upper_rows, pad_vector=pad_embed)

    # Upper block forward (layers k..13)
    h_out = forward_layer_range(model, h_upper, mask_upper, pos_ids_upper, k, len(model.layers))
    h_out = model.embedding_norm(h_out)
    logits = lm_head(h_out[:, -1:])  # (B, 1, V)

    # Initialize logits processor for repeat blocking
    logits_processor = (
        NoRepeatNGramLogitsProcessor(C.NO_REPEAT_NGRAM_SIZE)
        if C.NO_REPEAT_NGRAM_SIZE > 0
        else None
    )

    generated_ids = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    # First token selection
    scores = logits[:, -1, :]  # (B, V)
    first_tokens = scores.argmax(dim=-1)  # (B,)
    for r in range(batch_size):
        tok = first_tokens[r].item()
        generated_ids[r].append(tok)
        if tok == eos_token_id or len(generated_ids[r]) >= budgets[r]:
            finished[r] = True

    embed_tokens = causal_lm.get_input_embeddings()

    # --- AUTOREGRESSIVE DECODE LOOP (NO CACHE) ---
    for step in range(1, max_budget):
        if all(finished):
            break

        # Re-build lower input with generated tokens so far
        lower_rows = []
        for r in range(batch_size):
            a0, a1 = audio_spans[r]
            p0, p1 = prompt_spans[r]
            bos_part = inputs_embeds[r, a0 - n_bos : a0] if n_bos > 0 else inputs_embeds[r, :0]
            audio_part = inputs_embeds[r, a0:a1]
            prompt_part = inputs_embeds[r, p0:p1]
            gen_tensor = embed_tokens(torch.tensor(generated_ids[r], device=device, dtype=torch.long))
            lower_rows.append(torch.cat([bos_part, audio_part, prompt_part, gen_tensor], dim=0))

        h_step_lower, mask_step_lower, pos_ids_step_lower = rebuild_left_padded_batch(lower_rows, pad_vector=pad_embed)
        h_step_lower = forward_layer_range(model, h_step_lower, mask_step_lower, pos_ids_step_lower, 0, k)

        # Upper input: [BOS, frozen H'_audio, prompt_layer_k_minus_1, gen_layer_k_minus_1]
        upper_step_rows = []
        for r in range(batch_size):
            valid_lower_len = int(mask_step_lower[r].sum().item())
            pad_off = mask_step_lower.shape[1] - valid_lower_len
            b_start = pad_off
            b_end = b_start + n_bos
            # Audio starts after BOS: length is original unmerged length
            orig_audio_len = audio_spans[r][1] - audio_spans[r][0]
            suffix_start = b_end + orig_audio_len  # prompt + gen start
            
            bos_part = h_step_lower[r, b_start:b_end] if n_bos > 0 else h_step_lower[r, :0]
            audio_part = frozen_h_audio[r]
            suffix_part = h_step_lower[r, suffix_start:]  # contains prompt + all gen tokens at layer k-1
            upper_step_rows.append(torch.cat([bos_part, audio_part, suffix_part], dim=0))

        h_step_upper, mask_step_upper, pos_ids_step_upper = rebuild_left_padded_batch(upper_step_rows, pad_vector=pad_embed)
        h_step_out = forward_layer_range(model, h_step_upper, mask_step_upper, pos_ids_step_upper, k, len(model.layers))
        h_step_out = model.embedding_norm(h_step_out)
        next_logits = lm_head(h_step_out[:, -1:])  # (B, 1, V)

        # Score & select
        step_scores = next_logits[:, -1, :].clone()
        for r in range(batch_size):
            if finished[r]:
                continue
            if logits_processor is not None and len(generated_ids[r]) >= C.NO_REPEAT_NGRAM_SIZE:
                cur_ids_tensor = torch.tensor([generated_ids[r]], device=device, dtype=torch.long)
                step_scores[r:r+1] = logits_processor(cur_ids_tensor, step_scores[r:r+1])
            next_token = step_scores[r].argmax().item()
            generated_ids[r].append(next_token)
            if next_token == eos_token_id or len(generated_ids[r]) >= budgets[r]:
                finished[r] = True

    # Decode hypotheses
    transcripts = []
    for r in range(batch_size):
        tokens = generated_ids[r][:budgets[r]]
        transcripts.append(tokenizer.decode(tokens, skip_special_tokens=True))

    return transcripts, group_stats
