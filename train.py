"""
Baseline: encoder-free Mel -> LLM ASR, fixed-rate time-only chunking (Mel-LLM style, arXiv 2606.10231).

    wav -> 80-dim log-mel (10 ms hop) -> MVN (train stats) -> stack STACK frames (time-only chunk, full freq)
        -> Linear(STACK*80 -> d_model) -> [BOS][audio tokens]["Transcription:"][text][EOS] -> LFM2.5-2.6B-Base + LoRA

Everything that defines a run is a constant at the top of this file. Change it here, not via flags.

    python train.py                       # train on train-clean-100, val on dev-clean subset, test on test-clean/other
    python train.py --max_steps 50        # smoke test
    python train.py --eval_only runs/X    # re-run test eval from a saved run
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------- config
RUN_NAME = "baseline_fixed12.5hz"
SEED = 0
MODEL = "LiquidAI/LFM2.5-350M"
DATA = "data/LibriSpeech"
TRAIN_SPLIT = "train-clean-100"
VAL_SPLIT = "dev-clean"
VAL_N = 200                  # fixed dev subset used for in-training WER
TEST_SPLITS = ["test-clean", "test-other"]
MAX_TRAIN_SEC = 30.0         # drop the few very long utterances

# Audio feature extraction configuration
SR = 16000                   # Target audio sample rate (Hz)
N_FFT = 400                  # STFT window size (400 samples = 25 ms at 16 kHz)
HOP = 160                    # STFT hop size (160 samples = 10 ms at 16 kHz -> 100 frames/s)
N_MELS = 80                  # Number of mel filterbank channels
MVN_N = 2000                 # Number of utterances sampled to compute dataset Mean-Variance Normalization stats
STACK = 8                    # Number of consecutive mel frames stacked per audio token (100 Hz / 8 = 12.5 Hz token rate)

# Model and optimization hyperparameters
PROMPT = "Transcription:"
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LR = 1e-4                    # Peak learning rate: 1e-4 with linear warmup and decay
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 200
EPOCHS = 2
BATCH_SIZE = 8
GRAD_ACCUM = 4               # Gradient accumulation steps -> effective batch size = 8 * 4 = 32
GRAD_CLIP = 1.0
GRAD_CKPT = True             # Enable gradient checkpointing to reduce VRAM
LOG_EVERY = 20
EVAL_EVERY = 500             # Evaluate on validation split every N optimizer steps
EVAL_BATCH_SIZE = 16
MAX_NEW_TOKENS = 200
NUM_WORKERS = 8

DEVICE = "cuda"
DTYPE = torch.bfloat16


def get_config_dictionary():
    """Extract all uppercase global constants into a dictionary for configuration saving."""
    return {
        key: val
        for key, val in globals().items()
        if key.isupper() and isinstance(val, (int, float, str, list, bool))
    }


def seed_everything(seed):
    """Seed all pseudo-random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def load_librispeech_split(split_name):
    """
    Scan LibriSpeech directory for the given split and return a deterministic list of utterances.

    Returns:
        list[dict]: A list of dicts with keys:
            - 'id': utterance ID (e.g., '103-1240-0000')
            - 'path': absolute path to the .flac file
            - 'text': ground-truth transcript (lowercased)
            - 'sec': audio duration in seconds
    """
    utterance_records = []
    for transcript_file in sorted(Path(DATA, split_name).rglob("*.trans.txt")):
        for line in transcript_file.read_text().splitlines():
            utterance_id, text = line.split(" ", 1)
            audio_path = transcript_file.parent / f"{utterance_id}.flac"
            utterance_records.append({
                "id": utterance_id,
                "path": str(audio_path),
                "text": text.lower(),
                "sec": sf.info(str(audio_path)).duration,
            })
    return utterance_records


def build_htk_mel_filterbank():
    """
    Construct an HTK-style triangular mel filterbank matrix.

    Returns:
        torch.Tensor: Filterbank weights of shape (N_MELS, N_FFT // 2 + 1) = (80, 201).
    """
    hz_to_mel = lambda freq_hz: 2595.0 * np.log10(1.0 + freq_hz / 700.0)
    mel_to_hz = lambda mel_val: 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)

    # Linearly spaced points in mel scale converted back to Hz
    mel_bin_frequencies_hz = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2), N_MELS + 2))
    fft_frequencies_hz = np.fft.rfftfreq(N_FFT, 1.0 / SR)

    filterbank_weights = np.zeros((N_MELS, len(fft_frequencies_hz)))
    for mel_idx in range(N_MELS):
        left_hz = mel_bin_frequencies_hz[mel_idx]
        center_hz = mel_bin_frequencies_hz[mel_idx + 1]
        right_hz = mel_bin_frequencies_hz[mel_idx + 2]
        # Triangular filter response
        filterbank_weights[mel_idx] = np.maximum(
            0.0,
            np.minimum(
                (fft_frequencies_hz - left_hz) / (center_hz - left_hz),
                (right_hz - fft_frequencies_hz) / (right_hz - center_hz),
            ),
        )
    return torch.tensor(filterbank_weights, dtype=torch.float32)


MEL_FILTERBANK = build_htk_mel_filterbank()
STFT_WINDOW = torch.hann_window(N_FFT)


def extract_log_mel_spectrogram(audio_filepath):
    """
    Load an audio file and extract 80-channel log mel-spectrogram features.

    Args:
        audio_filepath (str or Path): Path to the .flac audio file.

    Returns:
        torch.Tensor: Log mel-spectrogram tensor of shape (num_frames, N_MELS).
    """
    waveform, sample_rate = sf.read(audio_filepath, dtype="float32")
    assert sample_rate == SR, f"Expected sample rate {SR}, but got {sample_rate} for {audio_filepath}"

    # Short-Time Fourier Transform (STFT) -> Power Spectrogram
    stft_complex = torch.stft(
        torch.from_numpy(waveform),
        n_fft=N_FFT,
        hop_length=HOP,
        window=STFT_WINDOW,
        return_complex=True,
    )
    power_spectrogram = stft_complex.abs() ** 2

    # Project power spectrogram onto mel filterbank bins and compute log
    mel_energies = MEL_FILTERBANK @ power_spectrogram
    log_mel = torch.log(mel_energies + 1e-6).T  # Transpose to shape (num_frames, N_MELS)
    return log_mel


def compute_or_load_mvn_stats(dataset_items):
    """
    Compute global Mean-Variance Normalization (MVN) statistics over a sample of train utterances.
    Caches the resulting mean and std vectors in data/mvn_<split>_n<sample_size>.pt.

    Returns:
        dict: {"mean": torch.Tensor(80,), "std": torch.Tensor(80,)}
    """
    cache_filepath = Path(DATA, f"mvn_{TRAIN_SPLIT}_n{MVN_N}.pt")
    if cache_filepath.exists():
        return torch.load(cache_filepath)

    deterministic_rng = random.Random(SEED)
    sampled_items = deterministic_rng.sample(dataset_items, min(MVN_N, len(dataset_items)))
    concatenated_mel_features = torch.cat(
        [extract_log_mel_spectrogram(item["path"]) for item in sampled_items]
    )
    mvn_stats = {
        "mean": concatenated_mel_features.mean(dim=0),
        "std": concatenated_mel_features.std(dim=0),
    }
    torch.save(mvn_stats, cache_filepath)
    return mvn_stats


class SpeechDataset(torch.utils.data.Dataset):
    """
    Dataset that loads audio waveforms, computes normalized mel-spectrograms,
    stacks STACK consecutive time frames into 1 token representation, and encodes text targets.
    """
    def __init__(self, dataset_items, mvn_stats, tokenizer):
        self.dataset_items = dataset_items
        self.mvn_stats = mvn_stats
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, index):
        item = self.dataset_items[index]

        # 1. Compute and normalize log mel-spectrogram: (num_frames, 80)
        normalized_mel = (extract_log_mel_spectrogram(item["path"]) - self.mvn_stats["mean"]) / self.mvn_stats["std"]

        # 2. Pad tail with zeros so the total frame count is divisible by STACK.
        # (Zero represents the mean value after MVN normalization)
        padding_frames_needed = (-normalized_mel.shape[0]) % STACK
        if padding_frames_needed > 0:
            zero_padding = normalized_mel.new_zeros(padding_frames_needed, N_MELS)
            normalized_mel = torch.cat([normalized_mel, zero_padding], dim=0)

        # 3. Stack consecutive STACK (8) frames across all 80 frequency bins -> (num_tokens, 8 * 80) = (num_tokens, 640)
        stacked_audio_features = normalized_mel.reshape(-1, STACK * N_MELS)

        # 4. Tokenize target transcript: append a leading space and trailing [EOS] token
        target_token_ids = (
            self.tokenizer(" " + item["text"], add_special_tokens=False).input_ids
            + [self.tokenizer.eos_token_id]
        )

        return {
            "audio": stacked_audio_features,  # Shape: (num_audio_tokens, 640)
            "target": target_token_ids,       # List of token IDs
            "text": item["text"],
            "id": item["id"],
        }


def collate_raw_batch(batch):
    """
    Pass raw list of items through; GPU handles dynamic projection and padding in build_multimodal_inputs.
    """
    return batch


def create_length_batched_sampler(dataset_items, batch_size, seed):
    """
    Bucket utterances by audio duration to minimize padding overhead:
    shuffle items, sort within chunks of 50 batches, then shuffle the resulting batches.
    """
    rng = random.Random(seed)
    item_indices = list(range(len(dataset_items)))
    rng.shuffle(item_indices)

    chunk_size = batch_size * 50
    batches = []
    for chunk_start in range(0, len(item_indices), chunk_size):
        chunk_indices = item_indices[chunk_start : chunk_start + chunk_size]
        sorted_chunk = sorted(chunk_indices, key=lambda idx: dataset_items[idx]["sec"])
        batches += [
            sorted_chunk[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(sorted_chunk), batch_size)
        ]
    rng.shuffle(batches)
    return batches


def init_dataloader_worker(_worker_id):
    """Set torch single-threaded inside each DataLoader worker process to prevent CPU oversubscription."""
    torch.set_num_threads(1)


def create_dataloader(speech_dataset, batch_sampler):
    """Create a PyTorch DataLoader configured with multi-worker prefetching."""
    return torch.utils.data.DataLoader(
        speech_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_raw_batch,
        num_workers=NUM_WORKERS,
        worker_init_fn=init_dataloader_worker,
        persistent_workers=False,
    )


# ---------------------------------------------------------------- model
def build_multimodal_inputs(batch, model, audio_projector, tokenizer, prompt_token_ids, is_train):
    """
    Project stacked mel features into token embedding space, concatenate with text prompt/target,
    and build padded input embeddings and attention masks.

    Sequence structure:
        [BOS] + [Audio Embeddings] + [Prompt: "Transcription:"] + [Target Text Tokens] + [EOS]

    Args:
        batch (list[dict]): Batch items containing 'audio' and 'target'.
        model (nn.Module): The base causal language model.
        audio_projector (nn.Linear): Linear layer mapping (STACK * N_MELS -> hidden_size).
        tokenizer: Hugging Face tokenizer.
        prompt_token_ids (list[int]): Token IDs for the prompt prefix (e.g., 'Transcription:').
        is_train (bool): If True, right-pads sequences and constructs training labels (with audio/prompt masked to -100).
                         If False, left-pads sequences for autoregressive generation.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - padded_inputs_embeds: Float tensor of shape (batch_size, max_seq_len, hidden_size).
            - attention_mask: Long tensor of shape (batch_size, max_seq_len) with 1 for tokens, 0 for pad.
            - target_labels: Long tensor of shape (batch_size, max_seq_len) with -100 on masked tokens.
    """
    text_embeddings_layer = model.get_input_embeddings()
    bos_token_tensor = torch.tensor([tokenizer.bos_token_id], device=DEVICE)
    bos_token_embed = text_embeddings_layer(bos_token_tensor)  # Shape: (1, hidden_size)

    sequence_embeddings_list = []
    token_labels_list = []

    for item in batch:
        # 1. Project stacked mel features (num_audio_tokens, 640) -> (num_audio_tokens, hidden_size)
        projected_audio_embeds = audio_projector(item["audio"].to(DEVICE)).to(DTYPE)

        # 2. Embed prompt and (if training) target text
        text_token_ids = prompt_token_ids + (item["target"] if is_train else [])
        text_token_embeds = text_embeddings_layer(torch.tensor(text_token_ids, device=DEVICE))

        # 3. Concatenate: [BOS] (1 token) + [Audio Tokens] + [Text Tokens]
        full_sequence_embeds = torch.cat([bos_token_embed, projected_audio_embeds, text_token_embeds], dim=0)
        sequence_embeddings_list.append(full_sequence_embeds)

        # 4. Construct labels: mask [BOS], all audio tokens, and prompt tokens with -100
        if is_train:
            masked_prefix_length = 1 + projected_audio_embeds.shape[0] + len(prompt_token_ids)
            utterance_labels = ([-100] * masked_prefix_length) + item["target"]
            token_labels_list.append(utterance_labels)

    # Maximum sequence length in this batch
    max_sequence_len = max(seq.shape[0] for seq in sequence_embeddings_list)
    batch_size = len(sequence_embeddings_list)
    hidden_dimension = sequence_embeddings_list[0].shape[1]

    pad_token_embed = text_embeddings_layer(torch.tensor([tokenizer.pad_token_id], device=DEVICE))

    # Initialize batch tensors
    padded_inputs_embeds = torch.zeros(batch_size, max_sequence_len, hidden_dimension, device=DEVICE, dtype=DTYPE)
    attention_mask = torch.zeros(batch_size, max_sequence_len, device=DEVICE, dtype=torch.long)
    target_labels = torch.full((batch_size, max_sequence_len), -100, device=DEVICE, dtype=torch.long)

    for batch_idx, seq_embed in enumerate(sequence_embeddings_list):
        seq_len = seq_embed.shape[0]
        # Right-padding for training; Left-padding for batched causal generation
        position_slice = slice(0, seq_len) if is_train else slice(max_sequence_len - seq_len, max_sequence_len)

        # Fill with pad token embeddings, then place valid sequence embeddings
        padded_inputs_embeds[batch_idx] = pad_token_embed.expand(max_sequence_len, -1)
        padded_inputs_embeds[batch_idx, position_slice] = seq_embed
        attention_mask[batch_idx, position_slice] = 1

        if is_train:
            target_labels[batch_idx, :seq_len] = torch.tensor(token_labels_list[batch_idx], device=DEVICE)

    return padded_inputs_embeds, attention_mask, target_labels


def load_model_and_projector(saved_run_dir=None):
    """
    Load base language model, tokenizer, and linear audio projector.
    If saved_run_dir is provided, loads trained LoRA weights and projector state dict from disk.
    Otherwise, initializes a fresh LoRA adapter and randomly initialized linear projector.

    Returns:
        tuple: (tokenizer, model, audio_projector)
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE)
    hidden_dimension = model.config.hidden_size

    # Linear projector: projects stacked mel frames (8 * 80 = 640) -> hidden_dimension (fp32)
    audio_projector = nn.Linear(STACK * N_MELS, hidden_dimension).to(DEVICE)

    if saved_run_dir is None:
        lora_configuration = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_configuration)
    else:
        lora_weights_path = os.path.join(saved_run_dir, "lora")
        projector_weights_path = os.path.join(saved_run_dir, "proj.pt")
        model = PeftModel.from_pretrained(model, lora_weights_path)
        audio_projector.load_state_dict(torch.load(projector_weights_path))

    return tokenizer, model, audio_projector


# ---------------------------------------------------------------- eval
def normalize_text_for_wer(text):
    """
    Standardize text for Word Error Rate (WER) computation:
    lowercase, retain only letters/apostrophes/spaces, and collapse multiple spaces.
    """
    text = re.sub(r"[^a-z' ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


@torch.no_grad()
def evaluate_dataset(dataset_items, mvn_stats, tokenizer, model, audio_projector, output_hypotheses_path=None):
    """
    Evaluate Word Error Rate (WER) and audio token throughput on a set of utterances using greedy decoding.

    Args:
        dataset_items (list[dict]): Utterances to evaluate.
        mvn_stats (dict): Normalization stats.
        tokenizer: Pretrained tokenizer.
        model (nn.Module): Language model.
        audio_projector (nn.Linear): Audio projector.
        output_hypotheses_path (str, optional): If specified, write {id, ref, hyp} JSONL to this file.

    Returns:
        dict: Evaluation metrics containing 'wer', 'n_utt', 'audio_tokens_per_sec', and 'audio_tokens'.
    """
    model.eval()
    prompt_token_ids = tokenizer(PROMPT, add_special_tokens=False).input_ids
    speech_dataset = SpeechDataset(dataset_items, mvn_stats, tokenizer)

    # Sort evaluation utterances by audio length to minimize left-padding
    sorted_indices = sorted(range(len(dataset_items)), key=lambda idx: dataset_items[idx]["sec"])
    evaluation_batches = [
        sorted_indices[i : i + EVAL_BATCH_SIZE]
        for i in range(0, len(sorted_indices), EVAL_BATCH_SIZE)
    ]

    reference_transcripts = []
    hypothesis_transcripts = []
    output_rows = []
    total_audio_tokens = 0

    for batch in create_dataloader(speech_dataset, evaluation_batches):
        inputs_embeds, attention_mask, _ = build_multimodal_inputs(
            batch,
            model=model,
            audio_projector=audio_projector,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            is_train=False,
        )

        generated_token_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

        for item, gen_tokens in zip(batch, generated_token_ids):
            decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            normalized_hyp = normalize_text_for_wer(decoded_text)
            normalized_ref = normalize_text_for_wer(item["text"])

            reference_transcripts.append(normalized_ref)
            hypothesis_transcripts.append(normalized_hyp)
            output_rows.append({"id": item["id"], "ref": normalized_ref, "hyp": normalized_hyp})
            total_audio_tokens += item["audio"].shape[0]

    total_audio_seconds = sum(item["sec"] for item in dataset_items)
    word_error_rate = jiwer.wer(reference_transcripts, hypothesis_transcripts)

    if output_hypotheses_path:
        with open(output_hypotheses_path, "w") as f:
            for row in output_rows:
                f.write(json.dumps(row) + "\n")

    model.train()
    return {
        "wer": word_error_rate,
        "n_utt": len(dataset_items),
        "audio_tokens_per_sec": total_audio_tokens / total_audio_seconds,
        "audio_tokens": total_audio_tokens,
    }


def run_test_evaluation(run_dir, mvn_stats, tokenizer, model, audio_projector):
    """
    Run evaluation across all configured test splits (e.g. test-clean, test-other) and save results.
    """
    test_results = {}
    for split_name in TEST_SPLITS:
        start_time = time.time()
        hypotheses_filepath = os.path.join(run_dir, f"hyp_{split_name}.jsonl")
        split_items = load_librispeech_split(split_name)
        test_results[split_name] = evaluate_dataset(
            split_items,
            mvn_stats=mvn_stats,
            tokenizer=tokenizer,
            model=model,
            audio_projector=audio_projector,
            output_hypotheses_path=hypotheses_filepath,
        )
        test_results[split_name]["eval_sec"] = time.time() - start_time
        print(split_name, test_results[split_name], flush=True)

    json.dump(test_results, open(os.path.join(run_dir, "test_results.json"), "w"), indent=2)
    return test_results


# ---------------------------------------------------------------- train
def main():
    arg_parser = argparse.ArgumentParser(description="Train or evaluate encoder-free Mel-to-LLM ASR model.")
    arg_parser.add_argument("--max_steps", type=int, default=None, help="Cap optimizer steps (for smoke tests)")
    arg_parser.add_argument("--eval_only", type=str, default=None, help="Run directory to evaluate without training")
    args = arg_parser.parse_args()

    seed_everything(SEED)

    # Fast path: evaluation only mode from existing checkpoint
    if args.eval_only:
        tokenizer, model, audio_projector = load_model_and_projector(args.eval_only)
        mvn_stats = torch.load(Path(DATA, f"mvn_{TRAIN_SPLIT}_n{MVN_N}.pt"))
        run_test_evaluation(args.eval_only, mvn_stats, tokenizer, model, audio_projector)
        return

    # Set up run output directory and logging
    run_directory_name = RUN_NAME + (f"_smoke{args.max_steps}" if args.max_steps else "")
    run_dir = os.path.join("runs", run_directory_name)
    os.makedirs(run_dir, exist_ok=True)

    config = get_config_dictionary()
    config["git"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    config["max_steps"] = args.max_steps
    json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    log_file = open(os.path.join(run_dir, "log.txt"), "a")

    def log_and_print(*print_args):
        message = " ".join(str(arg) for arg in print_args)
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    # Load dataset splits
    train_items = [item for item in load_librispeech_split(TRAIN_SPLIT) if item["sec"] <= MAX_TRAIN_SEC]
    validation_items = sorted(load_librispeech_split(VAL_SPLIT), key=lambda it: it["id"])[:VAL_N]
    mvn_stats = compute_or_load_mvn_stats(train_items)

    total_train_hours = sum(item["sec"] for item in train_items) / 3600.0
    log_and_print(f"train utts {len(train_items)} ({total_train_hours:.1f} h), val utts {len(validation_items)}")

    # Initialize model and linear projector
    tokenizer, model, audio_projector = load_model_and_projector()
    if GRAD_CKPT:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # Collect trainable parameters (LoRA adapter weights + linear projector)
    trainable_lora_params = [param for param in model.parameters() if param.requires_grad]
    all_trainable_params = trainable_lora_params + list(audio_projector.parameters())

    lora_param_count_m = sum(param.numel() for param in trainable_lora_params) / 1e6
    proj_param_count_m = sum(param.numel() for param in audio_projector.parameters()) / 1e6
    log_and_print(f"trainable: lora {lora_param_count_m:.1f}M, proj {proj_param_count_m:.2f}M")

    optimizer = torch.optim.AdamW(all_trainable_params, lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))

    # Calculate steps and linear learning rate schedule with warmup
    steps_per_epoch = math.ceil(len(train_items) / BATCH_SIZE) // GRAD_ACCUM
    total_optimizer_steps = args.max_steps or (steps_per_epoch * EPOCHS)

    def compute_lr_multiplier(step_idx):
        if step_idx < WARMUP_STEPS:
            return (step_idx + 1) / WARMUP_STEPS
        remaining_ratio = (total_optimizer_steps - step_idx) / max(1, total_optimizer_steps - WARMUP_STEPS)
        return max(0.0, remaining_ratio)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=compute_lr_multiplier)
    log_and_print(f"steps/epoch {steps_per_epoch}, total steps {total_optimizer_steps}")

    prompt_token_ids = tokenizer(PROMPT, add_special_tokens=False).input_ids
    train_dataset = SpeechDataset(train_items, mvn_stats, tokenizer)

    optimizer_step = 0
    micro_step = 0
    step_start_time = time.time()
    running_loss = 0.0
    running_tokens = 0

    for epoch in range(EPOCHS):
        epoch_sampler = create_length_batched_sampler(train_items, BATCH_SIZE, seed=SEED + epoch)
        dataloader = create_dataloader(train_dataset, epoch_sampler)

        for batch in dataloader:
            inputs_embeds, attention_mask, target_labels = build_multimodal_inputs(
                batch,
                model=model,
                audio_projector=audio_projector,
                tokenizer=tokenizer,
                prompt_token_ids=prompt_token_ids,
                is_train=True,
            )

            outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=target_labels)
            loss = outputs.loss
            (loss / GRAD_ACCUM).backward()

            running_loss += loss.item()
            running_tokens += int(attention_mask.sum())
            micro_step += 1

            if micro_step % GRAD_ACCUM != 0:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, GRAD_CLIP)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if optimizer_step % LOG_EVERY == 0:
                elapsed_time = time.time() - step_start_time
                average_loss = running_loss / (LOG_EVERY * GRAD_ACCUM)
                tokens_per_sec = running_tokens / elapsed_time
                current_lr = lr_scheduler.get_last_lr()[0]
                peak_vram_gb = torch.cuda.max_memory_allocated() / (2**30)

                log_and_print(
                    f"ep {epoch} step {optimizer_step}/{total_optimizer_steps} loss {average_loss:.4f} "
                    f"gnorm {grad_norm:.2f} lr {current_lr:.2e} tok/s {tokens_per_sec:.0f} "
                    f"mem {peak_vram_gb:.1f}G"
                )
                step_start_time, running_loss, running_tokens = time.time(), 0.0, 0

            if optimizer_step % EVAL_EVERY == 0 or optimizer_step == total_optimizer_steps:
                val_metrics = evaluate_dataset(
                    validation_items,
                    mvn_stats=mvn_stats,
                    tokenizer=tokenizer,
                    model=model,
                    audio_projector=audio_projector,
                )
                log_and_print(f"[val] step {optimizer_step} {VAL_SPLIT}[:{VAL_N}] WER {val_metrics['wer'] * 100:.2f}")
                step_start_time = time.time()

            if optimizer_step >= total_optimizer_steps:
                break

        if optimizer_step >= total_optimizer_steps:
            break

    # Save trained LoRA adapter and linear projector
    model.save_pretrained(os.path.join(run_dir, "lora"))
    torch.save(audio_projector.state_dict(), os.path.join(run_dir, "proj.pt"))
    log_and_print("saved", run_dir)

    if args.max_steps:
        return  # Smoke test run: skip full evaluation on test splits

    test_results = run_test_evaluation(run_dir, mvn_stats, tokenizer, model, audio_projector)
    wer_summary = {split_name: round(metrics["wer"] * 100, 2) for split_name, metrics in test_results.items()}
    log_and_print("TEST", json.dumps(wer_summary))


if __name__ == "__main__":
    main()
