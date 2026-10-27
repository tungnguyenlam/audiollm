#!/usr/bin/env bash
# One-shot dataset setup: builds data/ exactly as the training / benchmark / dataset_viewer code expects.
#
#   ./setup_dataset.sh             # everything (~83 GB): LibriSpeech + extra corpora + ESB benchmark sets
#   ./setup_dataset.sh --minimal   # only what train.py needs: LibriSpeech train-clean-100, dev-clean, test-clean/other (~7 GB)
#   ./setup_dataset.sh --no-esb    # skip the ESB benchmark test sets (~19 GB)
#   ./setup_dataset.sh --no-corpora  # skip data/corpora (~33 GB, not used by train.py yet)
#   ./setup_dataset.sh --check     # don't download, just verify what is on disk
#
# Resulting layout:
#   data/LibriSpeech/<split>/<spk>/<chapter>/<id>.flac + <spk>-<chapter>.trans.txt   (prepare_data.py)
#       dev-clean dev-other test-clean test-other train-clean-100 train-clean-360
#   data/corpora/<name>/...parquet + .done                                          (download_corpora.py)
#       air_bench ami_ihm clotho_aqa cremad edacc esc50 fleurs_en mmau_mini speech_commands vocalsound
#   data/esb/<config>/<split>-XXXXX-of-XXXXX.parquet                                (benchmark.py's cache)
#       ami earnings22 gigaspeech librispeech spgispeech voxpopuli
#
# Every step is idempotent / resumable: re-run after an interruption and finished parts are skipped.
# The MVN stats (data/LibriSpeech/mvn_*.pt) are computed by train.py on first run; data/curation.sqlite3 is
# created by `python -m dataset_viewer`. Neither is produced here.
# Gated / rate-limited downloads: run `hf auth login` (or export HF_TOKEN) first.
set -euo pipefail
cd "$(dirname "$0")"

MINIMAL=0; ESB=1; CORPORA=1; CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --minimal) MINIMAL=1 ;;
        --no-esb) ESB=0 ;;
        --no-corpora) CORPORA=0 ;;
        --check) CHECK_ONLY=1 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown option: $arg (see --help)"; exit 2 ;;
    esac
done
if [ "$MINIMAL" = 1 ]; then ESB=0; CORPORA=0; fi

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
"$PY" -c "import huggingface_hub, pyarrow" 2>/dev/null || {
    echo "error: $PY lacks huggingface_hub/pyarrow. Run ./setup_rocm.sh first (or: uv pip install huggingface_hub pyarrow)."
    exit 1
}

LIBRI_MIN=(train-clean-100 dev-clean test-clean test-other)
LIBRI_ALL=(train-clean-100 dev-clean test-clean test-other dev-other train-clean-360)

download() {
    mkdir -p data
    local need_gb=7
    [ "$MINIMAL" = 0 ] && need_gb=$((need_gb + 24))
    [ "$CORPORA" = 1 ] && need_gb=$((need_gb + 33))
    [ "$ESB" = 1 ] && need_gb=$((need_gb + 19))
    local free_gb; free_gb=$(df -BG --output=avail data | tail -1 | tr -dc '0-9')
    echo "== target up to ~${need_gb} GB (already-downloaded parts are skipped); ${free_gb} GB free on data/"

    echo "== [1/3] LibriSpeech -> data/LibriSpeech"
    if [ "$MINIMAL" = 1 ]; then "$PY" prepare_data.py "${LIBRI_MIN[@]}"; else "$PY" prepare_data.py "${LIBRI_ALL[@]}"; fi
    # prepare_data.py deletes each split's parquets after unpacking; drop the leftover HF metadata dir too.
    rm -rf data/hf_librispeech

    if [ "$CORPORA" = 1 ]; then
        echo "== [2/3] extra corpora -> data/corpora + data/asr_corpora"
        "$PY" download_corpora.py
        "$PY" unpack_parquet_asr_corpora.py
    else
        echo "== [2/3] extra corpora: skipped"
    fi

    if [ "$ESB" = 1 ]; then
        echo "== [3/3] ESB benchmark test sets -> data/esb"
        # Same files benchmark.load_set() fetches lazily ({config}/{prefix}-*.parquet), prefetched up front.
        "$PY" - <<'EOF'
import ast
from huggingface_hub import snapshot_download
# Read SETS / ESB_REPO / ESB_DIR from benchmark.py without importing it (avoids loading torch/transformers).
consts = {}
for node in ast.parse(open("benchmark.py").read()).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
            and node.targets[0].id in ("ESB_REPO", "ESB_DIR", "SETS"):
        consts[node.targets[0].id] = ast.literal_eval(node.value)
patterns = sorted({f"{cfg}/{prefix}-*.parquet" for cfg, prefix in consts["SETS"].values()})
print("patterns:", patterns)
snapshot_download(consts["ESB_REPO"], repo_type="dataset", allow_patterns=patterns, local_dir=consts["ESB_DIR"])
EOF
    else
        echo "== [3/3] ESB benchmark sets: skipped (benchmark.py will download them on demand)"
    fi
}

# Expected contents, taken from the reference data/ this project was developed on.
check() {
    local fail=0
    ok()  { printf '  ok    %s\n' "$1"; }
    bad() { printf '  FAIL  %s\n' "$1"; fail=1; }
    count() { find "$1" -name "$2" 2>/dev/null | wc -l; }

    echo "== check data/LibriSpeech (flac files per split)"
    declare -A FLAC=([train-clean-100]=28539 [dev-clean]=2703 [test-clean]=2620 [test-other]=2939
                     [dev-other]=2864 [train-clean-360]=104014)
    local splits=("${LIBRI_ALL[@]}"); [ "$MINIMAL" = 1 ] && splits=("${LIBRI_MIN[@]}")
    for s in "${splits[@]}"; do
        local d=data/LibriSpeech/$s n t
        n=$(count "$d" '*.flac'); t=$(count "$d" '*.trans.txt')
        if [ -f "$d/.done" ] && [ "$n" = "${FLAC[$s]}" ] && [ "$t" -gt 0 ]; then ok "$s: $n utts, $t transcripts"
        else bad "$s: $n/${FLAC[$s]} flac, $t transcripts, .done=$([ -f "$d/.done" ] && echo y || echo n)"; fi
    done

    if [ "$CORPORA" = 1 ]; then
        echo "== check data/corpora (parquet files per corpus)"
        declare -A PQ=([air_bench]=4 [ami_ihm]=51 [clotho_aqa]=11 [cremad]=3 [edacc]=16 [esc50]=2
                       [fleurs_en]=3 [mmau_mini]=8 [speech_commands]=14 [vocalsound]=2)
        for c in air_bench ami_ihm clotho_aqa cremad edacc esc50 fleurs_en mmau_mini speech_commands vocalsound; do
            local d=data/corpora/$c n
            n=$(find "$d" -name '*.parquet' -not -path '*/.cache/*' 2>/dev/null | wc -l)
            if [ -f "$d/.done" ] && [ "$n" = "${PQ[$c]}" ]; then ok "$c: $n parquet"
            else bad "$c: $n/${PQ[$c]} parquet, .done=$([ -f "$d/.done" ] && echo y || echo n)"; fi
        done
    fi

    if [ "$ESB" = 1 ]; then
        echo "== check data/esb (parquet files per config)"
        declare -A ESBN=([ami]=15 [earnings22]=5 [gigaspeech]=19 [librispeech]=2 [spgispeech]=38 [voxpopuli]=4)
        for c in ami earnings22 gigaspeech librispeech spgispeech voxpopuli; do
            local n; n=$(count "data/esb/$c" '*.parquet')
            if [ "$n" = "${ESBN[$c]}" ]; then ok "$c: $n parquet"; else bad "$c: $n/${ESBN[$c]} parquet"; fi
        done
    fi

    echo
    du -sh data/LibriSpeech data/corpora data/esb 2>/dev/null || true
    if [ "$fail" = 0 ]; then echo "dataset OK"; else echo "dataset INCOMPLETE: re-run ./setup_dataset.sh to resume"; fi
    return "$fail"
}

[ "$CHECK_ONLY" = 1 ] || download
check
