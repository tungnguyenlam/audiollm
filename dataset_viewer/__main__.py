"""python -m dataset_viewer [--port 8765] [--data data] [--db data/curation.sqlite3]"""
import argparse
import sys

try:
    import uvicorn

    from dataset_viewer.server import create_app
except ModuleNotFoundError as e:  # usually: started with the system python instead of the project venv
    sys.exit(f"dataset_viewer: missing module {e.name!r}. Run it from the project venv:\n"
             f"  .venv/bin/python -m dataset_viewer      (or: source .venv/bin/activate)")

p = argparse.ArgumentParser(description="Audio dataset viewer / curation tool")
p.add_argument("--data", default="data", help="data root (LibriSpeech/, corpora/, esb/)")
p.add_argument("--db", default="data/curation.sqlite3", help="SQLite file for reviews, metrics and decisions")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=8765)
p.add_argument("--scan-workers", type=int, default=4, help="threads used by background quality scans")
a = p.parse_args()
app = create_app(a.data, a.db, a.scan_workers)
print(f"Dataset viewer: http://{a.host}:{a.port}")
uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
