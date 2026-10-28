"""
Mirror a run directory to a Hugging Face Hub model repo (HUB_REPO): runs/<id>/ -> <HUB_REPO>/<id>/.

Uploads run in a background thread, so training never waits on the network. A failed upload is logged and the run
goes on: the run dir on disk is the source of truth, the Hub copy is for persistence (Colab) and remote monitoring
(plots.png, metrics.jsonl and log.txt are pushed along with the weights). Pull a run back with `runs.py pull`.
"""
from pathlib import Path

from audiollm import config as C

IGNORE_PATTERNS = ["*.tmp", "checkpoint.tmp/*", "checkpoint.old/*"]  # half-written files of an ongoing save


class HubSync:
    def __init__(self, repo_id, run_dir, log):
        from huggingface_hub import HfApi

        self.api = HfApi()
        self.repo_id = repo_id
        self.run_dir = Path(run_dir)
        self.log = log
        self.future = None
        self.api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        log(f"hub: mirroring {self.run_dir} -> https://huggingface.co/{repo_id}/tree/main/{self.run_dir.name}")

    @classmethod
    def from_config(cls, run_dir, log):
        """HubSync for HUB_REPO, or None when HUB_REPO is empty."""
        return cls(C.HUB_REPO, run_dir, log) if C.HUB_REPO else None

    def push(self, message, final=False):
        """
        Start uploading the run dir (only changed files are sent). If the previous upload is still running, this one
        is skipped, unless final=True: then it waits for it, and also deletes checkpoint/ from the Hub copy.
        """
        if self.future is not None and not self.future.done():
            if not final:
                self.log(f"hub: previous upload still running, skipping '{message}'")
                return
            self.wait()
        self.future = self.api.upload_folder(
            repo_id=self.repo_id, folder_path=self.run_dir, path_in_repo=self.run_dir.name, commit_message=message,
            ignore_patterns=IGNORE_PATTERNS, delete_patterns=["checkpoint/*"] if final else None, run_as_future=True,
        )

    def wait(self):
        """Block until the current upload is done; an upload error is logged, not raised."""
        if self.future is None:
            return
        try:
            self.future.result()
        except Exception as error:  # noqa: BLE001
            self.log(f"hub: upload failed: {error!r}")
        self.future = None
