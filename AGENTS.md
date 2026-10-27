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

## Repository

Encoder-free Mel -> LLM ASR baseline (`train.py`, `benchmark.py`, `runs.py`). See `README.md`.
Runtime artifacts (`data/`, `runs/`, `results/`) are gitignored; never commit them.
