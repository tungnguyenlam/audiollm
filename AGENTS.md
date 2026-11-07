# Coding-agent instructions

## Worklog

- Append decisions and results to the bottom of `WORKLOG.md` with `cat >> WORKLOG.md`
  (oldest entry first). Start each entry with `## YYYY-MM-DD - title`, keep it under
  ~20 lines, and put long detail in a report under the run directory (`runs/<id>/`) or
  `results/` that the entry links to.
- Log every decision that changes code, defaults, or the experiment plan, and every
  experiment result (run id, config delta, WER, what it rules in/out).
- Never read `WORKLOG.md` or any file over ~300 lines in full. Read the newest entries
  from the bottom up:

```bash
grep -n '^## ' WORKLOG.md | tail -15                                  # index of recent entries
tac WORKLOG.md | awk '/^## /{if(++n==2){print; exit}} {print}' | tac  # last 2 entries in full
sed -n '120,150p' WORKLOG.md                                         # one entry by line range
grep -n -i 'projector' WORKLOG.md docs/worklog_archive/*.md | tail   # older history by keyword
```

- At the start of a new month, move the previous month's entries to
  `docs/worklog_archive/YYYY-MM.md`.

## Long-running jobs

- Always fire training and long benchmark runs inside a detached `tmux` session, and start it with
  `systemd-run --user --scope` so the tmux server gets its own scope:
  ```bash
  systemd-run --user --scope --unit=<name> --collect \
    tmux new-session -d -s <session_name> 'cd <repo> && source .venv/bin/activate && python train.py ...'
  ```
- **Rationale:** agent tool processes run inside transient systemd scopes (`t3code-*.scope`), and the whole
  scope is killed when the agent session ends. A plain `tmux new-session` run from the agent shell starts the
  tmux server *inside* that scope, so the job still dies. This killed the Stage C merge sweep on 2026-11-07.
  Check with `cat /proc/<tmux-server-pid>/cgroup`: it must not show `t3code-*`.

## Repository

Encoder-free Mel -> LLM ASR baseline: CLI scripts `train.py`, `benchmark.py`, `runs.py` over the `audiollm/` package
(config, data, features, projectors, backbone, model, trainer). See `README.md` ("Code layout").
Runtime artifacts (`data/`, `runs/`, `results/`) are gitignored; never commit them.


