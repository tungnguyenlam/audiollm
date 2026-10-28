"""
Git clean filter: notebook JSON on stdin -> same notebook without outputs, execution counts and run-time
language metadata on stdout.
Only the committed copy is stripped; the working-tree notebook keeps its outputs. Stdlib only, Jupyter's own
JSON layout (indent 1, sorted keys), so diffs show source changes only. Enable once per clone:

    git config filter.strip-notebook-outputs.clean "python3 tools/strip_notebook_outputs.py"
    git config filter.strip-notebook-outputs.smudge cat
"""
import json
import sys

notebook = json.load(sys.stdin)
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cell.get("metadata", {}).pop("execution", None)
language_info = notebook.get("metadata", {}).get("language_info")
if language_info:  # Jupyter fills in interpreter version etc. on every run
    notebook["metadata"]["language_info"] = {"name": language_info.get("name", "python")}
sys.stdout.write(json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
