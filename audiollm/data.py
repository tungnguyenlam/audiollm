"""
Corpora and batching: split specs -> utterance records, SpeechDataset (features + target ids), samplers, DataLoader.

A record is {"id", "path", "text", "sec"}. The dataset emits frame-level features from a featurizer
(audiollm/features.py); how frames become tokens is up to the projector, so the data side never depends on it.
"""
import json
import random
import re
import unicodedata
from pathlib import Path

import soundfile as sf
import torch

from audiollm import config as C


# ---------------------------------------------------------------- corpora
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
    split_dir = Path(C.DATA, split_name)
    manifest_path = split_dir / "manifest.jsonl"
    if manifest_path.exists():
        return [json.loads(line) for line in manifest_path.read_text().splitlines()]

    utterance_records = []
    for transcript_file in sorted(split_dir.rglob("*.trans.txt")):
        for line in transcript_file.read_text().splitlines():
            utterance_id, text = line.split(" ", 1)
            audio_path = transcript_file.parent / f"{utterance_id}.flac"
            utterance_records.append({
                "id": utterance_id,
                "path": str(audio_path),
                "text": text.lower(),
                "sec": sf.info(str(audio_path)).duration,
            })
    if utterance_records and (split_dir / ".done").exists():
        manifest_path.write_text("".join(json.dumps(r) + "\n" for r in utterance_records))
    return utterance_records


def normalize_training_text(text):
    """Training target / reference text: strip annotation tags, lowercase ASCII, keep letters/digits/apostrophes."""
    if text.strip() == "IGNORE_TIME_SEGMENT_IN_SCORING":
        return ""
    text = re.sub(r"<[^>]+>|\[[^\]]+\]", " ", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text)).strip()


def load_split(split_spec):
    """
    Utterances for a split spec: a LibriSpeech split name ("train-clean-100"), an unpacked corpus split
    ("ami_ihm/train", see unpack_parquet_asr_corpora.py), or several of them joined with "+". A "*K" suffix
    ("fleurs_en/train*6") repeats that split's utterances K times; a fractional K adds a SEED-fixed random subset.
    Returns the same dicts as load_librispeech_split(), with text normalised by normalize_training_text().
    """
    records = []
    for part in split_spec.split("+"):
        name, _, weight_text = part.partition("*")
        weight = float(weight_text) if weight_text else 1.0
        if "/" in name:
            manifest = Path(C.ASR_CORPORA, name, "manifest.jsonl")
            split_records = [json.loads(line) for line in manifest.read_text().splitlines()]
        else:
            split_records = load_librispeech_split(name)
        for record in split_records:
            record["text"] = normalize_training_text(record["text"])
        split_records = [record for record in split_records if record["text"]]
        repeats, fraction = int(weight), weight - int(weight)
        records += split_records * repeats
        if fraction > 0:
            records += random.Random(C.SEED).sample(split_records, round(fraction * len(split_records)))
    return records


def unweighted_split(split_spec):
    """Split spec without "*K" weights: "a*2+b" -> "a+b"."""
    return re.sub(r"\*[0-9.]+", "", split_spec)


def parse_val_splits(val_split):
    """VAL_SPLIT "dev-clean,ami_ihm/validation" -> ["dev-clean", "ami_ihm/validation"]."""
    return [name.strip() for name in val_split.split(",") if name.strip()]


def load_validation_items(split_name):
    """
    Fixed VAL_N-utterance subset of one dev split, sorted by id. LibriSpeech keeps the first VAL_N ids so dev-clean
    stays comparable with older runs; a corpus split ("ami_ihm/validation") is ordered by meeting/speaker, so it gets a
    SEED-fixed random sample of the utterances inside the training length range instead.
    """
    items = sorted(load_split(split_name), key=lambda it: it["id"])
    if "/" not in split_name:
        return items[:C.VAL_N]
    items = [item for item in items if C.MIN_TRAIN_SEC <= item["sec"] <= C.MAX_TRAIN_SEC]
    return sorted(random.Random(C.SEED).sample(items, min(C.VAL_N, len(items))), key=lambda it: it["id"])


def load_train_items():
    """TRAIN_SPLIT utterances inside [MIN_TRAIN_SEC, MAX_TRAIN_SEC]."""
    return [item for item in load_split(C.TRAIN_SPLIT) if C.MIN_TRAIN_SEC <= item["sec"] <= C.MAX_TRAIN_SEC]


# ---------------------------------------------------------------- dataset
class SpeechDataset(torch.utils.data.Dataset):
    """Utterance records -> {"features": (num_frames, feature_dim), "target": token ids + [EOS], "text", "id", "sec"}."""

    def __init__(self, dataset_items, featurizer, tokenizer):
        self.dataset_items = dataset_items
        self.featurizer = featurizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset_items)

    def __getitem__(self, index):
        item = self.dataset_items[index]
        # Target transcript: a leading space and a trailing [EOS] token
        target_token_ids = (
            self.tokenizer(" " + item["text"], add_special_tokens=False).input_ids
            + [self.tokenizer.eos_token_id]
        )
        return {
            "features": self.featurizer(item["path"]),  # (num_frames, feature_dim), full frame rate
            "target": target_token_ids,
            "text": item["text"],
            "id": item["id"],
            "sec": item["sec"],
        }


def collate_speech_batch(items):
    """
    List of SpeechDataset items -> one batch dict: "features" zero-padded to (batch, max_frames, feature_dim)
    (zero = the MVN mean), "feature_lengths" (batch,), and the per-utterance lists "target", "text", "id", "sec".
    """
    feature_lengths = torch.tensor([item["features"].shape[0] for item in items])
    features = items[0]["features"].new_zeros(len(items), int(feature_lengths.max()), items[0]["features"].shape[1])
    for row, item in enumerate(items):
        features[row, :item["features"].shape[0]] = item["features"]
    batch = {key: [item[key] for item in items] for key in ("target", "text", "id", "sec")}
    return dict(batch, features=features, feature_lengths=feature_lengths)


# ---------------------------------------------------------------- batching
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


def create_token_batched_sampler(dataset_items, sequence_lengths, max_tokens, chunk_size, seed):
    """
    Like create_length_batched_sampler, but each batch holds as many utterances as fit in max_tokens padded
    positions: len(batch) * max(sequence_lengths in batch) <= max_tokens (an utterance longer than max_tokens gets a
    batch of its own). Short utterances then come in big batches and long ones in small batches.
    """
    rng = random.Random(seed)
    item_indices = list(range(len(dataset_items)))
    rng.shuffle(item_indices)

    batches = []
    for chunk_start in range(0, len(item_indices), chunk_size):
        sorted_chunk = sorted(item_indices[chunk_start : chunk_start + chunk_size], key=lambda idx: sequence_lengths[idx])
        batch = []
        for idx in sorted_chunk:  # ascending, so the newest item is always the batch's longest
            if batch and (len(batch) + 1) * sequence_lengths[idx] > max_tokens:
                batches.append(batch)
                batch = []
            batch.append(idx)
        batches.append(batch)
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
        collate_fn=collate_speech_batch,
        num_workers=C.NUM_WORKERS,
        worker_init_fn=init_dataloader_worker,
        persistent_workers=False,
    )
