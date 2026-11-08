"""Upload an existing run or restore a final model / numbered training checkpoint."""
import argparse
import shutil
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def restore(repo, run_id, output, step=None, revision="main"):
    from huggingface_hub import snapshot_download

    if PurePosixPath(run_id).name != run_id or run_id in (".", "..") or "\\" in run_id:
        raise ValueError("run_id must be a single directory name")
    output = Path(output)
    run = output / run_id
    if run.exists():
        raise FileExistsError(f"refusing to mix a backup with existing files: {run}")
    weights = "checkpoint" if step == "latest" else f"checkpoints/step-{int(step):09d}" if step else None
    patterns = [f"{run_id}/{name}" for name in ("config.json", "mvn.pt", "summary.json", "README.md")]
    patterns += [f"{run_id}/quickstart/*", f"{run_id}/base_model/*"]
    patterns += [f"{run_id}/{weights}/*"] if weights else [f"{run_id}/lora/*", f"{run_id}/proj.pt", f"{run_id}/embed.pt"]
    snapshot_download(repo, repo_type="model", revision=revision, allow_patterns=patterns, local_dir=output)
    if not (run / "config.json").exists():
        raise FileNotFoundError(f"no run {run_id} at {repo}@{revision}")
    if weights:
        source = run / weights
        if not (source / "trainer_state.pt").exists():
            raise FileNotFoundError(f"missing resumable state: {source}")
        if weights != "checkpoint":
            shutil.copytree(source, run / "checkpoint")
    if not (run / (weights or "") / "proj.pt").exists():
        raise FileNotFoundError("selected backup has no model weights")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    push = sub.add_parser("push")
    push.add_argument("repo")
    push.add_argument("run", type=Path)
    pull = sub.add_parser("pull")
    pull.add_argument("repo")
    pull.add_argument("run_id")
    pull.add_argument("--output", type=Path, default=Path("runs"))
    pull.add_argument("--step", help="step number or latest; omit for final weights")
    pull.add_argument("--revision", default="main", help="Hub commit SHA, tag or branch")
    args = parser.parse_args()
    if args.command == "pull":
        print(restore(args.repo, args.run_id, args.output, args.step, args.revision))
    else:
        import json
        from audiollm import config as C
        from audiollm.hub import HubSync
        C.apply_config(json.loads((args.run / "config.json").read_text()))
        hub = HubSync(args.repo, args.run, print)
        hub.push("manual backup", final=True)
        if not hub.wait():
            raise SystemExit("backup failed; see hub_status.json")


if __name__ == "__main__":
    main()
