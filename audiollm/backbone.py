"""Backbone: the pretrained causal LLM (MODEL) + its tokenizer, frozen, with a LoRA adapter (LORA_*)."""
import os
import re

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from audiollm import config as C
from audiollm.utils import trainable_dtype

# Backbones that ship their own modelling code. trust_remote_code runs the repo's Python, so it is pinned to the
# commit whose .py files were read (imports only torch / transformers / triton).
REMOTE_CODE_REVISIONS = {"LiquidAI/LFM3-1.7B-Base": "545fc8f4c7b5fa9f405105274cd29c4f8f44b526"}


def load_llm():
    """(tokenizer, base LLM on DEVICE in DTYPE), no adapter yet."""
    remote = {"trust_remote_code": True, "revision": REMOTE_CODE_REVISIONS[C.MODEL]} if C.MODEL in REMOTE_CODE_REVISIONS else {}
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL, **remote)
    llm = AutoModelForCausalLM.from_pretrained(C.MODEL, dtype=C.DTYPE, **remote).to(C.DEVICE)
    return tokenizer, llm


def attach_lora(llm, saved_run_dir=None, trainable=False):
    """
    Wrap llm in a fresh LoRA adapter, or load runs/<id>/lora (trainable=True keeps it trainable, for INIT_FROM).
    """
    # PEFT upcasts LoRA weights to fp32 by default; in "bf16" mode keep them in the base model's DTYPE
    upcast_lora = trainable_dtype() == torch.float32
    if saved_run_dir is None:
        lora_configuration = LoraConfig(
            r=C.LORA_R,
            lora_alpha=C.LORA_ALPHA,
            lora_dropout=C.LORA_DROPOUT,
            target_modules=C.LORA_TARGETS if C.LORA_TARGETS == "all-linear" else C.LORA_TARGETS.split(","),
            task_type="CAUSAL_LM",
        )
        return get_peft_model(llm, lora_configuration, autocast_adapter_dtype=upcast_lora)
    return PeftModel.from_pretrained(llm, os.path.join(saved_run_dir, "lora"), is_trainable=trainable,
                                     autocast_adapter_dtype=upcast_lora)


def describe_lora_placement(llm):
    """One line saying where LoRA sits in the LLM: which module names, in how many of the decoder layers."""
    layer_ids_by_name = {}
    for name, module in llm.named_modules():
        if hasattr(module, "lora_A"):
            layer = re.search(r"\.layers\.(\d+)\.", name)
            layer_ids_by_name.setdefault(name.rsplit(".", 1)[-1], set()).add(int(layer.group(1)) if layer else -1)
    n_layers = llm.config.num_hidden_layers
    parts = [f"{name} ({len(ids)}/{n_layers} layers)" for name, ids in sorted(layer_ids_by_name.items())]
    return (f"lora on LLM ({C.LORA_TARGETS!r}, r{C.LORA_R} a{C.LORA_ALPHA} dropout {C.LORA_DROPOUT}): " + ", ".join(parts)
            + " | LLM base weights frozen, audio projector fully trained")


def describe_llm(llm):
    """One line on the LLM's shape: layers (by type, for hybrid conv/attention models), hidden size, vocab."""
    cfg = llm.config
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types:
        counts = {}
        for layer_type in layer_types:
            counts[layer_type] = counts.get(layer_type, 0) + 1
        layers = f"{cfg.num_hidden_layers} layers (" + ", ".join(f"{n} {t}" for t, n in counts.items()) + ")"
    else:
        layers = f"{cfg.num_hidden_layers} layers"
    return f"LLM {C.MODEL}: {layers}, hidden {cfg.hidden_size}, vocab {cfg.vocab_size}"
