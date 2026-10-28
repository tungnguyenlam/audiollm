"""Run directories and their bookkeeping: runs/<id>/ creation, git state, metrics.jsonl and summary.json."""
import json
import subprocess
import time
from pathlib import Path


def get_git_info():
    """Commit hash, whether tracked files have uncommitted changes, and the diff against HEAD."""
    def git(*cmd):
        return subprocess.run(["git", *cmd], capture_output=True, text=True).stdout
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip()),
        "diff": git("diff", "HEAD"),
    }


def create_run_dir(run_name):
    """Create a fresh, never-reused directory runs/<YYYYmmdd-HHMMSS>_<run_name>[_N]."""
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{run_name}"
    run_dir, suffix = Path("runs", run_id), 1
    while run_dir.exists():
        suffix += 1
        run_dir = Path("runs", f"{run_id}_{suffix}")
    run_dir.mkdir(parents=True)
    return run_dir


class RunTracker:
    """Per-run bookkeeping: metrics.jsonl (append-only time series) and summary.json (status + headline numbers)."""

    def __init__(self, run_dir, **initial_summary):
        self.run_dir = Path(run_dir)
        summary_path = self.run_dir / "summary.json"
        self.summary = json.load(open(summary_path)) if summary_path.exists() else {}
        self.update(**initial_summary)

    def log_metrics(self, kind, **values):
        """Append one row, e.g. log_metrics("val", step=500, wer=0.12)."""
        row = {"type": kind, "time": round(time.time(), 1), **values}
        with open(self.run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def update(self, **values):
        """Merge values into summary.json (atomic rewrite, so a crash never leaves it half-written)."""
        self.summary.update(values)
        tmp_path = self.run_dir / "summary.json.tmp"
        tmp_path.write_text(json.dumps(self.summary, indent=2))
        tmp_path.replace(self.run_dir / "summary.json")
