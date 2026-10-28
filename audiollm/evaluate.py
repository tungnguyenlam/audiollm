"""WER evaluation: greedy decoding over a list of utterances, val hypotheses, and the final test splits."""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import jiwer
from tqdm import tqdm

from audiollm import config as C
from audiollm.data import SpeechDataset, create_dataloader, load_split
from audiollm.utils import compute_context, make_progress_bar


def normalize_text_for_wer(text):
    """
    Standardize text for Word Error Rate (WER) computation:
    lowercase, retain only letters/digits/apostrophes/spaces, and collapse multiple spaces.
    """
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def val_hypotheses_path(run_dir, split_name, step):
    """runs/<id>/val_hyps/<split>/step_001000.jsonl: every val utt's {id, ref, hyp} for one val pass on one dev split."""
    split_dir = Path(run_dir, "val_hyps", re.sub(r"[^\w.-]+", "_", split_name.split("/")[0]))
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir / f"step_{step:06d}.jsonl"


def evaluate_dataset(dataset_items, featurizer, model, output_hypotheses_path=None, progress_desc="eval",
                     audio_ablation="none"):
    """
    Evaluate Word Error Rate (WER) and audio token throughput on a set of utterances using greedy decoding.

    Args:
        dataset_items (list[dict]): Utterances to evaluate.
        featurizer: audiollm.features featurizer (holds the MVN stats).
        model (SpeechLLM): the model to decode with.
        output_hypotheses_path (str, optional): If specified, write {id, ref, hyp} JSONL to this file.
        progress_desc (str): Label shown on the tqdm progress bar.
        audio_ablation (str): "none" | "shuffle" (each utt is decoded from its length-sorted batch neighbour's audio,
            scored against its own reference) | "zeros" (all-zero features). A model that uses the audio gets a
            much higher WER under either ablation; one that ignores it scores about the same.

    Returns:
        dict: Evaluation metrics containing 'wer', 'sub', 'del', 'ins' (rates), 'exact' (fraction of utterances whose
              normalised hypothesis equals the reference), 'n_utt', 'n_words',
              'audio_tokens_per_sec', 'audio_tokens', 'eval_sec', 'rtfx' and 'examples' (a few ref/hyp pairs).
    """
    model.eval()
    eval_start_time = time.time()
    speech_dataset = SpeechDataset(dataset_items, featurizer, model.tokenizer)

    # Sort evaluation utterances by audio length to minimize left-padding
    sorted_indices = sorted(range(len(dataset_items)), key=lambda idx: dataset_items[idx]["sec"])
    evaluation_batches = [
        sorted_indices[i : i + C.EVAL_BATCH_SIZE]
        for i in range(0, len(sorted_indices), C.EVAL_BATCH_SIZE)
    ]

    reference_transcripts = []
    hypothesis_transcripts = []
    output_rows = []
    total_audio_tokens = 0

    progress_bar = make_progress_bar(total=len(dataset_items), desc=progress_desc, unit="utt", leave=False)
    for batch in create_dataloader(speech_dataset, evaluation_batches):
        with compute_context():
            decoded_texts, projected = model.transcribe(batch, audio_ablation=audio_ablation)

        for utt_id, text, decoded_text in zip(batch["id"], batch["text"], decoded_texts):
            normalized_hyp = normalize_text_for_wer(decoded_text)
            normalized_ref = normalize_text_for_wer(text)
            reference_transcripts.append(normalized_ref)
            hypothesis_transcripts.append(normalized_hyp)
            output_rows.append({"id": utt_id, "ref": normalized_ref, "hyp": normalized_hyp})
        total_audio_tokens += int(projected.lengths.sum())

        progress_bar.update(len(batch["id"]))
    progress_bar.close()

    total_audio_seconds = sum(item["sec"] for item in dataset_items)
    word_alignment = jiwer.process_words(reference_transcripts, hypothesis_transcripts)
    n_reference_words = word_alignment.hits + word_alignment.substitutions + word_alignment.deletions
    eval_seconds = time.time() - eval_start_time

    if output_hypotheses_path:
        with open(output_hypotheses_path, "w") as f:
            for row in output_rows:
                f.write(json.dumps(row) + "\n")

    model.train()
    return {
        "wer": word_alignment.wer,
        "sub": word_alignment.substitutions / max(1, n_reference_words),
        "del": word_alignment.deletions / max(1, n_reference_words),
        "ins": word_alignment.insertions / max(1, n_reference_words),
        "exact": sum(ref == hyp for ref, hyp in zip(reference_transcripts, hypothesis_transcripts)) / max(1, len(dataset_items)),
        "n_utt": len(dataset_items),
        "n_words": n_reference_words,
        "audio_tokens_per_sec": total_audio_tokens / total_audio_seconds,
        "audio_tokens": total_audio_tokens,
        "eval_sec": eval_seconds,
        "rtfx": total_audio_seconds / eval_seconds,  # seconds of audio decoded per wall-clock second
        "examples": random.Random(C.SEED).sample(output_rows, min(C.LOG_N_EXAMPLES, len(output_rows))),
    }


def format_eval_metrics(metrics):
    """One-line human-readable summary of evaluate_dataset() output."""
    return (
        f"WER {metrics['wer'] * 100:.2f} (sub {metrics['sub'] * 100:.2f} del {metrics['del'] * 100:.2f} "
        f"ins {metrics['ins'] * 100:.2f}) | exact {metrics['exact'] * 100:.1f}% | {metrics['n_utt']} utts {metrics['n_words']} words | "
        f"{metrics['eval_sec']:.0f}s RTFx {metrics['rtfx']:.0f} | audio tok/s {metrics['audio_tokens_per_sec']:.2f}"
    )


def run_test_evaluation(run_dir, featurizer, model, log_fn=None):
    """
    Run evaluation across all configured test splits (e.g. test-clean, test-other) and save results.
    log_fn (callable, optional) receives one summary line per split; defaults to tqdm.write.
    """
    log_fn = log_fn or (lambda message: tqdm.write(message, file=sys.stdout))
    test_results = {}
    for split_name in C.TEST_SPLITS:
        hypotheses_filepath = os.path.join(run_dir, f"hyp_{split_name.replace('/', '-')}.jsonl")
        test_results[split_name] = evaluate_dataset(
            load_split(split_name),
            featurizer,
            model,
            output_hypotheses_path=hypotheses_filepath,
            progress_desc=f"test {split_name}",
        )
        test_results[split_name].pop("examples")
        log_fn(f"[test] {split_name} {format_eval_metrics(test_results[split_name])}")

    json.dump(test_results, open(os.path.join(run_dir, "test_results.json"), "w"), indent=2)
    return test_results
