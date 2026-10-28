"""
SpeechLLM = backbone (LLM + LoRA + tokenizer) + audio projector, laid out as
[BOS][audio tokens][PROMPT][text][EOS], with the loss on [text][EOS] only.
"""
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn

from audiollm import config as C
from audiollm.backbone import attach_lora, describe_llm, describe_lora_placement, load_llm
from audiollm.projectors import build_projector
from audiollm.utils import trainable_dtype


def decode_token_budget(audio_seconds):
    """Max new tokens for one utterance: MAX_TOKENS_PER_SEC * sec + MAX_TOKENS_PAD, never above MAX_NEW_TOKENS."""
    if C.MAX_TOKENS_PER_SEC <= 0:
        return C.MAX_NEW_TOKENS
    return min(C.MAX_NEW_TOKENS, math.ceil(C.MAX_TOKENS_PER_SEC * audio_seconds) + C.MAX_TOKENS_PAD)


class SpeechLLM(nn.Module):
    def __init__(self, tokenizer, llm, projector):
        super().__init__()
        self.tokenizer = tokenizer
        self.llm = llm
        self.projector = projector
        self.prompt_token_ids = tokenizer(C.PROMPT, add_special_tokens=False).input_ids

    @classmethod
    def load(cls, saved_run_dir=None, trainable=False):
        """
        Fresh LoRA + projector (saved_run_dir=None), or a run's lora/ + proj.pt (trainable=True keeps the loaded
        LoRA trainable, for INIT_FROM). The projector is built before LoRA, so seeded inits match older runs.
        """
        tokenizer, llm = load_llm()
        embedding_std = llm.get_input_embeddings().weight.float().std().item()
        projector = build_projector(C.N_MELS, llm.config.hidden_size, embedding_std).to(C.DEVICE, trainable_dtype())
        llm = attach_lora(llm, saved_run_dir, trainable)
        if saved_run_dir is not None:
            projector.load_checkpoint(torch.load(os.path.join(saved_run_dir, "proj.pt"), map_location=C.DEVICE))
        return cls(tokenizer, llm, projector)

    def save(self, run_dir):
        self.llm.save_pretrained(os.path.join(run_dir, "lora"))
        torch.save(self.projector.state_dict(), os.path.join(run_dir, "proj.pt"))

    def lora_parameters(self):
        return [param for param in self.llm.parameters() if param.requires_grad]

    def build_inputs(self, batch, is_train, audio_ablation="none"):
        """
        Project a collated batch's features and lay out each utterance as [BOS][audio][prompt][target if is_train],
        padded with the pad-token embedding: right-padded for training, left-padded for batched generation.

        audio_ablation: "none" | "shuffle" (each utterance gets its batch neighbour's audio) | "zeros" (all-zero
        features, the MVN mean).

        Returns:
            inputs_embeds (batch, len, hidden), attention_mask (batch, len), labels (batch, len; -100 on everything
            but the target), and the ProjectorOutput.
        """
        features, feature_lengths = batch["features"], batch["feature_lengths"]
        if audio_ablation == "shuffle":
            features, feature_lengths = features.roll(-1, 0), feature_lengths.roll(-1, 0)
        elif audio_ablation == "zeros":
            features = torch.zeros_like(features)
        elif audio_ablation != "none":
            raise ValueError(f"unknown audio ablation {audio_ablation!r} (expected 'none', 'shuffle' or 'zeros')")
        proj_dtype = next(self.projector.parameters()).dtype
        projected = self.projector(features.to(C.DEVICE, proj_dtype), feature_lengths.to(C.DEVICE))

        embed_tokens = self.llm.get_input_embeddings()
        bos_token_embed = embed_tokens(torch.tensor([self.tokenizer.bos_token_id], device=C.DEVICE))
        sequences, labels = [], []
        for row, target in enumerate(batch["target"]):
            audio_embeds = projected.embeds[row, :int(projected.lengths[row])].to(C.DTYPE)
            text_token_ids = self.prompt_token_ids + (target if is_train else [])
            text_embeds = embed_tokens(torch.tensor(text_token_ids, device=C.DEVICE))
            sequences.append(torch.cat([bos_token_embed, audio_embeds, text_embeds], dim=0))
            if is_train:
                labels.append([-100] * (1 + audio_embeds.shape[0] + len(self.prompt_token_ids)) + target)

        max_len, hidden = max(seq.shape[0] for seq in sequences), sequences[0].shape[1]
        pad_token_embed = embed_tokens(torch.tensor([self.tokenizer.pad_token_id], device=C.DEVICE))
        inputs_embeds = pad_token_embed.expand(len(sequences), max_len, hidden).clone()
        attention_mask = torch.zeros(len(sequences), max_len, device=C.DEVICE, dtype=torch.long)
        target_labels = torch.full((len(sequences), max_len), -100, device=C.DEVICE, dtype=torch.long)
        for row, seq in enumerate(sequences):
            positions = slice(0, len(seq)) if is_train else slice(max_len - len(seq), max_len)
            inputs_embeds[row, positions] = seq
            attention_mask[row, positions] = 1
            if is_train:
                target_labels[row, :len(seq)] = torch.tensor(labels[row], device=C.DEVICE)
        return inputs_embeds, attention_mask, target_labels, projected

    def forward(self, batch):
        """Training loss (LM loss + projector aux loss) for a collated batch; also returns the masks and stats."""
        inputs_embeds, attention_mask, target_labels, projected = self.build_inputs(batch, is_train=True)
        loss = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=target_labels).loss
        if projected.aux_loss is not None:
            loss = loss + projected.aux_loss
        return loss, {"attention_mask": attention_mask, "target_labels": target_labels, "projected": projected}

    @torch.no_grad()
    def generate_transcripts(self, inputs_embeds, attention_mask, audio_seconds):
        """
        Greedy-decode a left-padded batch. The batch runs up to the largest per-utterance budget
        (decode_token_budget), then each hypothesis is cut to its own budget, so a looping utterance cannot emit more
        than ~MAX_TOKENS_PER_SEC tokens per second of audio.
        """
        budgets = [decode_token_budget(sec) for sec in audio_seconds]
        extra_kwargs = {"no_repeat_ngram_size": C.NO_REPEAT_NGRAM_SIZE} if C.NO_REPEAT_NGRAM_SIZE > 0 else {}
        generated_token_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max(budgets),
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **extra_kwargs,
        )
        return [self.tokenizer.decode(tokens[:budget], skip_special_tokens=True)
                for tokens, budget in zip(generated_token_ids, budgets)]

    @torch.no_grad()
    def transcribe(self, batch, audio_ablation="none"):
        """Hypotheses for a collated batch, plus the ProjectorOutput (for token counts)."""
        inputs_embeds, attention_mask, _, projected = self.build_inputs(batch, is_train=False,
                                                                        audio_ablation=audio_ablation)
        return self.generate_transcripts(inputs_embeds, attention_mask, batch["sec"]), projected

    def summary(self):
        """
        Model summary for the run log: LLM shape, parameter counts per component (frozen vs trainable, dtype), the
        projector's layers and where LoRA sits.

        Returns:
            tuple[str, dict]: the printable text and {"params_*": count} for summary.json.
        """
        def count(params):
            params = list(params)
            return sum(p.numel() for p in params), sum(p.numel() for p in params if p.requires_grad), \
                sorted({str(p.dtype).removeprefix("torch.") for p in params})

        llm_groups = defaultdict(list)
        lora_params = []
        for name, param in self.llm.named_parameters():
            if "lora_" in name:
                lora_params.append(param)
            elif ".layers." in name:
                llm_groups["decoder layers"].append(param)
            elif "embed" in name:
                llm_groups["embeddings"].append(param)
            elif "lm_head" in name:
                llm_groups["lm_head (untied)"].append(param)
            else:
                llm_groups["final norm / other"].append(param)

        rows = []  # (label, total, trainable, dtypes)
        llm_base = [p for group in llm_groups.values() for p in group]
        rows.append(("LLM base", *count(llm_base)))
        rows += [(f"  {label}", *count(params)) for label, params in llm_groups.items()]
        rows.append(("LoRA adapters", *count(lora_params)))
        rows.append((f"projector ({self.projector.name})", *count(self.projector.parameters())))
        for child_name, child in self.projector.named_children():
            n_params = sum(p.numel() for p in child.parameters())
            if n_params:
                rows.append((f"  {child_name}", *count(child.parameters())))
        total = count(list(self.llm.parameters()) + list(self.projector.parameters()))
        rows.append(("total", *total))

        def fmt_count(n):
            return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.2f}M" if n >= 1e5 else f"{n / 1e3:.1f}K" if n else "0"

        lines = ["model summary", f"  {describe_llm(self.llm)}", f"  projector {self.projector.describe()}",
                 f"  {describe_lora_placement(self.llm)}", "",
                 f"  {'component':<28} {'params':>10} {'trainable':>10}  dtype"]
        for label, n_total, n_trainable, dtypes in rows:
            lines.append(f"  {label:<28} {fmt_count(n_total):>10} {fmt_count(n_trainable):>10}  {','.join(dtypes)}")
        lines.append(f"  trainable fraction {total[1] / total[0] * 100:.2f}%")
        lines += ["", "  projector modules:"] + [f"    {line}" for line in repr(self.projector).splitlines()]
        counts = {
            "params_total": total[0],
            "params_trainable": total[1],
            "params_llm": count(llm_base)[0],
            "params_lora": count(lora_params)[0],
            "params_projector": count(self.projector.parameters())[0],
        }
        return "\n".join(lines), counts
