"""Training loop: data -> SpeechLLM -> optimizer/schedule, with val passes, checkpoint and final test eval."""
import json
import math
import random
import time

import numpy as np
import torch

import runs as runs_lib
from audiollm import config as C
from audiollm.data import (SpeechDataset, create_dataloader, create_length_batched_sampler,
                           create_token_batched_sampler, load_train_items, load_validation_items, parse_val_splits)
from audiollm.evaluate import evaluate_dataset, format_eval_metrics, run_test_evaluation, val_hypotheses_path
from audiollm.features import LogMelFeaturizer, SpecAugment, compute_or_load_mvn_stats
from audiollm.model import SpeechLLM
from audiollm.utils import compute_context, format_duration, make_progress_bar


def build_optimizer(lora_params, proj_params, log_and_print):
    # Two param groups so the projector can get its own lr (PROJ_LR_MULT) and LoRA can start late (PROJ_ONLY_STEPS)
    param_groups = [{"params": lora_params}, {"params": proj_params}]
    if C.OPTIMIZER == "adamw":
        return torch.optim.AdamW(param_groups, lr=C.LR, weight_decay=C.WEIGHT_DECAY, betas=(0.9, 0.999))
    if C.OPTIMIZER == "muon":
        log_and_print(f"optimizer: Muon ({C.MUON_ADJUST_LR}) on all {len(lora_params) + len(proj_params)} trainable "
                      "2D weights, no AdamW (projector bias frozen)")
        return torch.optim.Muon(param_groups, lr=C.LR, weight_decay=C.WEIGHT_DECAY, adjust_lr_fn=C.MUON_ADJUST_LR)
    raise ValueError(f"unknown OPTIMIZER {C.OPTIMIZER!r} (expected 'adamw' or 'muon')")


def build_lr_scheduler(optimizer, total_optimizer_steps):
    """Linear warmup, then cosine or linear decay; LoRA held at 0 for PROJ_ONLY_STEPS, then its own warmup."""
    def compute_lr_multiplier(step_idx):
        if step_idx < C.WARMUP_STEPS:
            return (step_idx + 1) / C.WARMUP_STEPS
        decay_progress = min(1.0, (step_idx - C.WARMUP_STEPS) / max(1, total_optimizer_steps - C.WARMUP_STEPS))
        if C.LR_SCHEDULE == "cosine":
            return C.MIN_LR_RATIO + (1.0 - C.MIN_LR_RATIO) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        if C.LR_SCHEDULE == "linear":
            return 1.0 - decay_progress
        raise ValueError(f"unknown LR_SCHEDULE {C.LR_SCHEDULE!r} (expected 'cosine' or 'linear')")

    def compute_lora_lr_multiplier(step_idx):
        """0 while only the projector trains, then a fresh linear warmup into the shared schedule."""
        if step_idx < C.PROJ_ONLY_STEPS:
            return 0.0
        return min(1.0, (step_idx - C.PROJ_ONLY_STEPS + 1) / max(1, C.WARMUP_STEPS)) * compute_lr_multiplier(step_idx)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[compute_lora_lr_multiplier if C.PROJ_ONLY_STEPS else compute_lr_multiplier,
                   lambda step_idx: C.PROJ_LR_MULT * compute_lr_multiplier(step_idx)],
    )


def make_epoch_sampler(train_items, sequence_lengths, epoch):
    """Batches (lists of item indices) for one epoch: fixed BATCH_SIZE, or BATCH_TOKENS padded positions."""
    if C.BATCH_TOKENS:
        return create_token_batched_sampler(train_items, sequence_lengths, C.BATCH_TOKENS,
                                            chunk_size=C.BATCH_SIZE * 50, seed=C.SEED + epoch)
    return create_length_batched_sampler(train_items, C.BATCH_SIZE, seed=C.SEED + epoch)


def estimate_sequence_lengths(train_items, model, frame_rate):
    """[BOS][audio][prompt][text][EOS] length of each train utterance, for BATCH_TOKENS batching."""
    texts = [" " + item["text"] for item in train_items]
    text_lengths = [len(ids) for ids in model.tokenizer(texts, add_special_tokens=False).input_ids]
    prefix = 1 + len(model.prompt_token_ids)
    return [prefix + model.projector.num_tokens(int(item["sec"] * frame_rate) + 1) + n_text + 1
            for item, n_text in zip(train_items, text_lengths)]


def train(args, run_dir, tracker, log_and_print):
    """Full training loop + final test eval. Writes metrics to tracker and checkpoints to run_dir."""
    log_and_print(f"model {C.MODEL} | device {torch.cuda.get_device_name(0)} | dtype {C.DTYPE} | precision {C.PRECISION}")

    # Load dataset splits
    train_items = load_train_items()
    mvn_stats = compute_or_load_mvn_stats(train_items)  # always from the full split, so TRAIN_N runs share the cache
    featurizer = LogMelFeaturizer(mvn_stats)
    if C.TRAIN_N:
        train_items = sorted(random.Random(C.SEED).sample(train_items, C.TRAIN_N), key=lambda it: it["id"])
    if C.VAL_FROM_TRAIN:
        validation_sets = {f"{C.TRAIN_SPLIT}(train)": train_items[:C.VAL_N]}
    else:
        validation_sets = {name: load_validation_items(name) for name in parse_val_splits(C.VAL_SPLIT)}
    val_ablations = [name.strip() for name in C.EVAL_AUDIO_ABLATIONS.split(",") if name.strip()]

    total_train_hours = sum(item["sec"] for item in train_items) / 3600.0
    log_and_print(f"train utts {len(train_items)} ({total_train_hours:.1f} h), val utts " +
                  ", ".join(f"{runs_lib.split_label(name)} {len(items)}" for name, items in validation_sets.items()))
    batch_text = f"tokens {C.BATCH_TOKENS}" if C.BATCH_TOKENS else f"{C.BATCH_SIZE}"
    log_and_print(
        f"batch {batch_text} x accum {C.GRAD_ACCUM} | {C.OPTIMIZER} lr {C.LR:.1e} {C.LR_SCHEDULE} warmup {C.WARMUP_STEPS} "
        f"| epochs {C.EPOCHS}{f' (train_steps {C.TRAIN_STEPS})' if C.TRAIN_STEPS else ''} | features {featurizer.feature_dim}-d "
        f"at {featurizer.frame_rate:g} Hz | lora r{C.LORA_R} a{C.LORA_ALPHA} on LLM {C.LORA_TARGETS}"
    )

    # Backbone + projector -> one trainable model
    model = SpeechLLM.load(C.INIT_FROM or None, trainable=True)
    if C.INIT_FROM:
        log_and_print(f"init: LoRA + projector weights from {C.INIT_FROM} (fresh optimizer and LR schedule)")
    if C.GRAD_CKPT:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # Muon only takes 2D matrices and nothing falls back to AdamW, so the (1D) projector bias stays at its init
    if C.OPTIMIZER == "muon":
        if C.PROJECTOR != "linear":
            raise ValueError(f"OPTIMIZER='muon' requires PROJECTOR='linear' (2D weights only), got {C.PROJECTOR!r}")
        for param in model.projector.parameters():
            if param.ndim < 2:
                param.requires_grad_(False)

    trainable_lora_params = model.lora_parameters()
    trainable_proj_params = [p for p in model.projector.parameters() if p.requires_grad]
    summary_text, param_counts = model.summary()
    for line in summary_text.splitlines():
        log_and_print(line)
    (run_dir / "model_summary.txt").write_text(summary_text + "\n")
    tracker.update(**param_counts)

    optimizer = build_optimizer(trainable_lora_params, trainable_proj_params, log_and_print)

    # Steps: fixed batches give ceil(utts / BATCH_SIZE) per epoch; token batches are counted on epoch 0's sampler
    sequence_lengths = estimate_sequence_lengths(train_items, model, featurizer.frame_rate) if C.BATCH_TOKENS else None
    if C.BATCH_TOKENS:
        epoch_batches = make_epoch_sampler(train_items, sequence_lengths, 0)
        log_and_print(f"token batching: {len(epoch_batches)} batches/epoch, "
                      f"{len(train_items) / len(epoch_batches):.1f} utts/batch on average")
        steps_per_epoch = max(1, len(epoch_batches) // C.GRAD_ACCUM)
    else:
        steps_per_epoch = max(1, math.ceil(len(train_items) / C.BATCH_SIZE) // C.GRAD_ACCUM)
    total_optimizer_steps = args.max_steps or C.TRAIN_STEPS or (steps_per_epoch * C.EPOCHS)
    num_epochs = math.ceil(total_optimizer_steps / steps_per_epoch)
    lr_scheduler = build_lr_scheduler(optimizer, total_optimizer_steps)
    log_and_print(f"steps/epoch {steps_per_epoch}, total steps {total_optimizer_steps} | proj {C.PROJECTOR} "
                  f"(init {C.PROJ_INIT}, lr x{C.PROJ_LR_MULT}), LoRA frozen for first {C.PROJ_ONLY_STEPS} steps")
    if C.TRAIN_N or C.VAL_FROM_TRAIN or val_ablations:
        log_and_print(f"diagnostics: train subset {C.TRAIN_N or 'all'}, val on {', '.join(validation_sets)}, "
                      f"audio ablations {val_ablations or 'none'}")

    augment = SpecAugment.from_config()
    log_and_print(f"train augmentation: {augment or 'none'}")
    train_dataset = SpeechDataset(train_items, featurizer, model.tokenizer, augment=augment)

    optimizer_step = 0
    micro_step = 0
    train_start_time = time.time()
    best_val_wer, best_val_step = float("inf"), None

    # Per-logging-window accumulators (reset every LOG_EVERY optimizer steps)
    def reset_window_stats():
        return {
            "start_time": time.time(), "loss_sum": 0.0, "micro_steps": 0, "tokens": 0, "padded_tokens": 0,
            "target_tokens": 0, "audio_tokens": 0, "utts": 0, "audio_sec": 0.0, "data_wait_sec": 0.0,
            "grad_norms": [], "lora_grad_norms": [], "proj_grad_norms": [], "proj_stats": {},
        }

    window = reset_window_stats()
    step_loss_sum = 0.0  # loss summed over the micro-batches of the current optimizer step (for the bar)

    progress_bar = make_progress_bar(total=total_optimizer_steps, desc="train", unit="step")

    for epoch in range(num_epochs):
        epoch_sampler = make_epoch_sampler(train_items, sequence_lengths, epoch)
        dataloader = create_dataloader(train_dataset, epoch_sampler)
        progress_bar.set_description(f"train ep {epoch + 1}/{num_epochs}")

        data_wait_start = time.time()
        for batch in dataloader:
            window["data_wait_sec"] += time.time() - data_wait_start

            with compute_context():
                loss, aux = model(batch)
            (loss / C.GRAD_ACCUM).backward()

            attention_mask, projected = aux["attention_mask"], aux["projected"]
            loss_value = loss.item()
            step_loss_sum += loss_value
            window["loss_sum"] += loss_value
            window["micro_steps"] += 1
            window["tokens"] += int(attention_mask.sum())
            window["padded_tokens"] += attention_mask.numel()
            window["target_tokens"] += int((aux["target_labels"] != -100).sum())
            window["audio_tokens"] += int(projected.lengths.sum())
            window["utts"] += len(batch["id"])
            window["audio_sec"] += sum(batch["sec"])
            for key, val in projected.stats.items():
                window["proj_stats"][key] = window["proj_stats"].get(key, 0.0) + float(val)
            micro_step += 1

            if micro_step % C.GRAD_ACCUM != 0:
                data_wait_start = time.time()
                continue

            if optimizer_step < C.PROJ_ONLY_STEPS:
                for param in trainable_lora_params:  # LoRA frozen: drop its grads so they neither step nor eat the clip budget
                    param.grad = None
            # Clip LoRA and projector independently so neither component starves the other's gradient budget
            lora_grad_norm = (torch.nn.utils.clip_grad_norm_(trainable_lora_params, C.GRAD_CLIP).item()
                              if trainable_lora_params and optimizer_step >= C.PROJ_ONLY_STEPS else 0.0)
            proj_grad_norm = torch.nn.utils.clip_grad_norm_(trainable_proj_params, C.GRAD_CLIP).item()
            grad_norm = math.hypot(lora_grad_norm, proj_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            window["grad_norms"].append(grad_norm)
            window["lora_grad_norms"].append(lora_grad_norm)
            window["proj_grad_norms"].append(proj_grad_norm)

            current_lr = lr_scheduler.get_last_lr()[0]
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{step_loss_sum / C.GRAD_ACCUM:.3f}",
                gnorm=f"{grad_norm:.2f}",
                lr=f"{current_lr:.1e}",
                best_wer="-" if best_val_step is None else f"{best_val_wer * 100:.2f}",
                refresh=False,
            )
            step_loss_sum = 0.0

            if optimizer_step % C.LOG_EVERY == 0:
                elapsed_time = time.time() - window["start_time"]
                average_loss = window["loss_sum"] / window["micro_steps"]
                grad_norms = window["grad_norms"]
                total_elapsed = time.time() - train_start_time
                eta_seconds = total_elapsed / optimizer_step * (total_optimizer_steps - optimizer_step)
                epoch_progress = micro_step / len(epoch_sampler)  # fractional epochs completed (all epochs)
                num_clipped = sum(norm > C.GRAD_CLIP for norm in grad_norms)
                proj_stats = {f"proj_{key}": val / window["micro_steps"] for key, val in window["proj_stats"].items()}

                log_and_print(
                    f"ep {epoch_progress:.2f} step {optimizer_step}/{total_optimizer_steps} "
                    f"({optimizer_step / total_optimizer_steps * 100:.1f}%) | "
                    f"loss {average_loss:.4f} ppl {math.exp(min(average_loss, 20)):.2f} | "
                    f"gnorm {grad_norm:.2f} (avg {np.mean(grad_norms):.2f} max {np.max(grad_norms):.2f} "
                    f"clipped {num_clipped}/{len(grad_norms)}) | "
                    f"pre-clip gnorm lora {np.mean(window['lora_grad_norms']):.2f} proj {np.mean(window['proj_grad_norms']):.2f} | "
                    f"lr {current_lr:.2e} proj lr {lr_scheduler.get_last_lr()[1]:.2e} | "
                    f"elapsed {format_duration(total_elapsed)} eta {format_duration(eta_seconds)}"
                )
                log_and_print(
                    f"    perf: {elapsed_time / len(grad_norms):.2f}s/step tok/s {window['tokens'] / elapsed_time:.0f} "
                    f"tgt tok/s {window['target_tokens'] / elapsed_time:.0f} utt/s {window['utts'] / elapsed_time:.1f} "
                    f"audio {window['audio_sec'] / elapsed_time:.0f}x realtime "
                    f"({window['audio_tokens'] / max(1e-9, window['audio_sec']):.2f} audio tok/s of speech) | "
                    f"pad eff {window['tokens'] / max(1, window['padded_tokens']) * 100:.0f}% "
                    f"data wait {window['data_wait_sec'] / elapsed_time * 100:.0f}% | "
                    f"mem {torch.cuda.memory_allocated() / 2**30:.1f}G "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G "
                    f"reserved {torch.cuda.memory_reserved() / 2**30:.1f}G"
                    + "".join(f" | {key} {val:.4g}" for key, val in proj_stats.items())
                )
                tracker.log_metrics(
                    "train", step=optimizer_step, epoch=round(epoch_progress, 4), loss=average_loss,
                    lr=current_lr, grad_norm=grad_norm, grad_norm_avg=float(np.mean(grad_norms)),
                    grad_norm_max=float(np.max(grad_norms)), clipped=num_clipped,
                    lora_grad_norm_avg=float(np.mean(window["lora_grad_norms"])),
                    proj_grad_norm_avg=float(np.mean(window["proj_grad_norms"])),
                    proj_lr=lr_scheduler.get_last_lr()[1],
                    sec_per_step=elapsed_time / len(grad_norms), tok_per_sec=window["tokens"] / elapsed_time,
                    audio_tok_per_speech_sec=window["audio_tokens"] / max(1e-9, window["audio_sec"]),
                    peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, elapsed_sec=total_elapsed, **proj_stats,
                )
                tracker.update(step=optimizer_step, total_steps=total_optimizer_steps, train_loss=average_loss)
                window = reset_window_stats()
                if optimizer_step % C.PLOT_EVERY == 0:
                    runs_lib.refresh_run_views(run_dir, log_and_print)

            if optimizer_step % C.EVAL_EVERY == 0 or optimizer_step == total_optimizer_steps:
                val_wers = {}
                for val_index, (val_name, validation_items) in enumerate(validation_sets.items()):
                    val_metrics = evaluate_dataset(
                        validation_items,
                        featurizer,
                        model,
                        progress_desc=f"val {runs_lib.split_label(val_name)} @{optimizer_step}",
                        output_hypotheses_path=val_hypotheses_path(run_dir, val_name, optimizer_step),
                    )
                    val_wers[val_name] = round(val_metrics["wer"] * 100, 2)
                    best_text = ""
                    if val_index == 0:  # the first dev split selects the best step
                        is_best = val_metrics["wer"] < best_val_wer
                        if is_best:
                            best_val_wer, best_val_step = val_metrics["wer"], optimizer_step
                        best_text = f" | best {best_val_wer * 100:.2f} @ step {best_val_step}{' (new best)' if is_best else ''}"
                    ablation_wers = {}
                    for ablation in val_ablations:
                        ablation_wers[f"wer_{ablation}"] = evaluate_dataset(
                            validation_items, featurizer, model,
                            progress_desc=f"val {runs_lib.split_label(val_name)} {ablation} @{optimizer_step}", audio_ablation=ablation,
                        )["wer"]
                    ablation_text = "".join(f" | {key} {val * 100:.2f}" for key, val in ablation_wers.items())
                    log_and_print(
                        f"[val] step {optimizer_step} {runs_lib.split_label(val_name)}[:{len(validation_items)}] "
                        f"{format_eval_metrics(val_metrics)}{best_text}{ablation_text}"
                    )
                    for example in val_metrics["examples"]:
                        log_and_print(f"    [{example['id']}] REF: {example['ref']}")
                        log_and_print(f"    [{example['id']}] HYP: {example['hyp']}")
                    tracker.log_metrics(
                        "val", step=optimizer_step, split=val_name,
                        **{key: val for key, val in val_metrics.items() if key != "examples"}, **ablation_wers,
                    )
                tracker.update(
                    step=optimizer_step, total_steps=total_optimizer_steps,
                    val_wer=next(iter(val_wers.values())), val_wers=val_wers,
                    best_val_wer=round(best_val_wer * 100, 2), best_val_step=best_val_step,
                )
                runs_lib.refresh_run_views(run_dir, log_and_print)
                window = reset_window_stats()  # exclude eval time from training throughput

            if optimizer_step >= total_optimizer_steps:
                break

            data_wait_start = time.time()

        if optimizer_step >= total_optimizer_steps:
            break

    progress_bar.close()
    log_and_print(
        f"training done: {optimizer_step} steps in {format_duration(time.time() - train_start_time)}, "
        f"best val WER {best_val_wer * 100:.2f} @ step {best_val_step}"
    )

    # Save trained LoRA adapter and projector; runs/latest_model -> the newest (non-smoke) run that has weights
    model.save(run_dir)
    tracker.update(weights_saved_step=optimizer_step)
    if not args.max_steps:
        runs_lib.update_latest_link(run_dir, "latest_model")
    log_and_print("saved", run_dir)

    if args.max_steps:
        return model, featurizer  # Smoke test run: skip full evaluation on test splits

    test_results = run_test_evaluation(run_dir, featurizer, model, log_fn=log_and_print)
    wer_summary = {split_name: round(metrics["wer"] * 100, 2) for split_name, metrics in test_results.items()}
    for split_name, metrics in test_results.items():
        tracker.log_metrics("test", step=optimizer_step, split=split_name, **metrics)
    tracker.update(test_wer=wer_summary)
    log_and_print("TEST", json.dumps(wer_summary))
    return model, featurizer
