"""Portable, step-addressable training backups on the Hugging Face Hub."""
import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audiollm import config as C

IGNORE_PATTERNS = ["*.tmp", "checkpoint.tmp", "checkpoint.old", "__pycache__", ".cache", "hub_status.json"]


def prepare_bundle(run_dir):
    """Keep the inference source and any local base/tokenizer alongside the adapters."""
    run_dir = Path(run_dir)
    source = Path(__file__).resolve().parent.parent
    target = run_dir / "quickstart"
    target.mkdir(exist_ok=True)
    if target.resolve() != source.resolve():
        shutil.copytree(source / "audiollm", target / "audiollm", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for name in (() if target.resolve() == source.resolve() else ("requirements.txt", "runs.py", "train.py", "benchmark.py")):
        shutil.copy2(source / name, target / name)
    for name in (() if target.resolve() == source.resolve() else ("transcribe.py", "hf_backup.py")):
        origin = source / "scripts" / name
        shutil.copy2(origin if origin.exists() else source / name, target / name)
    base = Path(C.MODEL)
    if base.is_dir() and base.resolve() != (run_dir / "base_model").resolve():
        shutil.copytree(base, run_dir / "base_model", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".git", ".cache", "__pycache__"))
    (run_dir / "README.md").write_text(
        f"# AudioLLM backup: {run_dir.name}\n\n"
        "Install PyTorch for your platform, then `pip install -r quickstart/requirements.txt`.\n\n"
        "From this directory: `python quickstart/transcribe.py --run . --audio speech.wav`.\n"
        "Use `--step 6000` to test `checkpoints/step-000006000/`; `--step latest` tests checkpoint/.\n"
        "Use `--device cpu` for CPU float32 inference; CUDA defaults to bfloat16.\n"
        "For correction-capable runs, add `--hypothesis 'your ASR hypothesis'`.\n\n"
        "Final weights: lora/, proj.pt, optional embed.pt. Retained resumable states: checkpoints/step-N/.\n"
        "mvn.pt contains training normalization. base_model/ contains a local base/tokenizer when used; "
        "otherwise config.json names the upstream base. This custom audio model needs the bundled Python code.\n"
        "To resume: restore a step using quickstart/hf_backup.py, then run quickstart/train.py --resume <restored-run>. "
        "Training datasets and augmentation assets must be provided separately.\n"
    )


class HubSync:
    def __init__(self, repo_id, run_dir, log):
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.run_dir, self.log = Path(run_dir), log
        prefix = f"{self.api.whoami()['name']}/audiollm" if repo_id == "auto" else repo_id
        self.repo_id = f"{prefix}-{self.run_dir.name}"
        self.future = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.api.create_repo(self.repo_id, repo_type="model", private=False, exist_ok=True)
        prepare_bundle(self.run_dir)
        self.log(f"hub: backing up to https://huggingface.co/{self.repo_id}/tree/main/{self.run_dir.name}")

    @classmethod
    def from_config(cls, run_dir, log):
        return cls(C.HUB_REPO, run_dir, log) if C.HUB_REPO else None

    def _status(self, **values):
        (self.run_dir / "hub_status.json").write_text(json.dumps(values, indent=2) + "\n")

    def push(self, message, final=False):
        """Snapshot locally before uploading; every checkpoint gets a retained remote step path."""
        self.wait()
        staging = tempfile.TemporaryDirectory(prefix="audiollm-hub-")
        snapshot = Path(staging.name) / self.run_dir.name
        try:
            shutil.copytree(self.run_dir, snapshot, ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))
            checkpoint = snapshot / "checkpoint"
            if (checkpoint / "trainer_state.pt").exists():
                import torch
                state = torch.load(checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
                step_dir = snapshot / "checkpoints" / f"step-{state['optimizer_step']:09d}"
                # Hardlinks within the immutable snapshot avoid another full optimizer-state copy.
                shutil.copytree(checkpoint, step_dir, copy_function=os.link, dirs_exist_ok=True)
            (Path(staging.name) / "README.md").write_text(self.model_card(message))
            self._status(status="uploading", message=message)
            self.future = self.executor.submit(self._upload, Path(staging.name), staging, message)
        except BaseException:
            staging.cleanup()
            raise

    def model_card(self, message):
        config = json.loads((self.run_dir / "config.json").read_text())
        summary_path = self.run_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        run = self.run_dir.name
        fields = ("MODEL", "LANG", "PROJECTOR", "STACK", "LORA_R", "LORA_ALPHA", "LR", "BATCH_SIZE",
                  "GRAD_ACCUM", "TRAIN_SPLIT", "TRAIN_STEPS", "EPOCHS", "CHECKPOINT_EVERY")
        rows = "\n".join(f"| {key} | `{config.get(key)}` |" for key in fields)
        progress = "\n".join(f"| {key} | {summary[key]} |" for key in
                             ("status", "step", "total_steps", "checkpoint_step", "weights_saved_step",
                              "best_val_wer", "best_val_step", "val_wers", "test_wer", "duration_sec") if key in summary)
        return (
            "---\n" + "tags:\n- automatic-speech-recognition\n- peft\n- audiollm\n"
            "pipeline_tag: automatic-speech-recognition\n---\n\n"
            f"# AudioLLM: {run}\n\nLatest backup: **{message}** (UTC {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}).\n\n"
            "Training progress updates at checkpoint uploads and completion. Metrics reflect this training run; "
            "best_val_wer/test_wer are percentages, val_wers are fractions.\n\n"
            "| Progress | Value |\n| --- | --- |\n" + progress + "\n\n"
            "| Configuration | Value |\n| --- | --- |\n" + rows + "\n\n"
            f"Full [config]({run}/config.json), [metrics]({run}/metrics.jsonl), "
            f"[log]({run}/log.txt), and [plots]({run}/plots.png).\n\n"
            "## Quickstart\n\nInstall PyTorch for your platform and clone the AudioLLM source repository. Then:\n\n"
            "```bash\npip install -r requirements.txt\n"
            f"python scripts/hf_backup.py pull {self.repo_id} {run} --output downloads\n"
            f"python scripts/transcribe.py --run downloads/{run} --audio speech.wav\n```\n\n"
            "For an in-progress run add `--step latest` to both commands. For a retained step, "
            "use e.g. `--step 6000` to both commands. Add `--device cpu` for CPU inference or "
            "`--hypothesis 'ASR hypothesis'` for correction-capable runs.\n\n"
            f"The [portable bundle]({run}/quickstart) also contains the inference code and requirements: "
            f"download this repository with `hf download {self.repo_id} --local-dir backup`, then "
            f"run `python backup/{run}/quickstart/transcribe.py --run backup/{run} --audio speech.wav`.\n\n"
            "Each retained checkpoint includes LoRA, projector, optional trained embeddings, optimizer, "
            "scheduler and RNG state. Local base model/tokenizer files are bundled; otherwise the upstream "
            "model in config.json is downloaded. No training audio is included.\n"
        )

    def _upload(self, snapshot, staging, message):
        try:
            last_error = None
            for attempt in range(1, 4):
                try:
                    commit = self.api.upload_folder(
                        repo_id=self.repo_id, repo_type="model", folder_path=snapshot,
                        path_in_repo="", commit_message=f"{self.run_dir.name}: {message}",
                    )
                    self._status(status="uploaded", message=message, commit=str(commit))
                    self.log(f"hub: uploaded {message}: {commit}")
                    return True
                except Exception as error:
                    last_error = repr(error)
                    self.log(f"hub: upload attempt {attempt}/3 failed: {error!r}")
                    if attempt < 3:
                        time.sleep(attempt * 2)
            self._status(status="failed", message=message, error=last_error)
            return False
        finally:
            staging.cleanup()

    def wait(self):
        if self.future is None:
            return True
        future, self.future = self.future, None
        try:
            return future.result()
        except Exception as error:
            self._status(status="failed", error=repr(error))
            self.log(f"hub: upload failed: {error!r}")
            return False
