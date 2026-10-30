"""
Download high-quality ASR and audio understanding corpora from Hugging Face into data/corpora/<name>/
or unpack LibriSpeech into data/LibriSpeech in OpenSLR layout.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

# name: (repo_id, allow_patterns, target_dir, approx_gb, kind 'asr'|'asr_vi'|'understanding')
REGISTRY = {
    # Default active English corpora (human-transcribed, high-quality; planned total ~58 GB <= 65 GB budget)
    "librispeech_clean360": ("openslr/librispeech_asr", ["all/train.clean.360/*", "all/validation.other/*"], "data/LibriSpeech", 23.5, "asr"),
    "ami_ihm": ("edinburghcstr/ami", ["ihm/*"], "data/corpora/ami_ihm", 15.2, "asr"),
    "fleurs_en": ("google/fleurs", ["parquet-data/en_us/*"], "data/corpora/fleurs_en", 2.4, "asr"),
    "edacc": ("edinburghcstr/edacc", ["data/*"], "data/corpora/edacc", 7.0, "asr"),
    "speech_commands": ("barto17/speech_commands", ["data/*"], "data/corpora/speech_commands", 4.9, "asr"),
    "clotho_aqa": ("lmms-lab-audio/ClothoAQA", ["clotho_aqa/*"], "data/corpora/clotho_aqa", 1.6, "understanding"),
    "esc50": ("ashraq/esc50", ["data/*"], "data/corpora/esc50", 0.8, "understanding"),
    "vocalsound": ("lmms-lab-audio/vocalsound", ["data/*"], "data/corpora/vocalsound", 0.6, "understanding"),
    "cremad": ("confit/cremad-parquet", ["data/*"], "data/corpora/cremad", 0.6, "understanding"),
    "mmau_mini": ("lmms-lab-audio/mmau", ["data/test_mini*"], "data/corpora/mmau_mini", 1.2, "understanding"),
    "air_bench": ("pbcong/AIR-bench", ["Chat/*"], "data/corpora/air_bench", 0.3, "understanding"),

    # Optional / off-by-default English corpora (large or lower SNR, runnable via --only <name>)
    "librispeech_other500": ("openslr/librispeech_asr", ["all/train.other.500/*"], "data/LibriSpeech", 30.8, "asr"),
    "tedlium3": ("sanchit-gandhi/tedlium-data", ["data/*"], "data/corpora/tedlium3", 52.7, "asr"),
    "voxpopuli_en": ("facebook/voxpopuli", ["en/*"], "data/corpora/voxpopuli_en", 98.8, "asr"),
    # gated (accept the terms on the HF dataset page first); parquet "m" is the cumulative M subset (1000 h, includes S).
    # Their test-* shards are the Open ASR benchmark sets, so only train + validation are fetched
    "gigaspeech_m": ("speechcolab/gigaspeech", ["parquet-data/m/train-*", "parquet-data/m/validation-*"], "data/corpora/gigaspeech", 116.0, "asr"),
    "spgispeech_s": ("kensho/spgispeech", ["S/train-*", "S/validation-00000-*"], "data/corpora/spgispeech", 34.0, "asr"),
    "earnings22": ("distil-whisper/earnings22", ["chunked/*"], "data/corpora/earnings22", 24.0, "asr"),
    "libritts_r": ("mythicinfinity/libritts_r", ["data/*"], "data/corpora/libritts_r", 100.6, "asr"),
    "peoples_speech_clean": ("MLCommons/peoples_speech", ["clean/*"], "data/corpora/peoples_speech_clean", 398.6, "asr"),
    "mls_eng": ("parler-tts/mls_eng", ["data/*"], "data/corpora/mls_eng", 705.2, "asr"),
    "wavcaps": ("cvssp/WavCaps", None, "data/corpora/wavcaps", 819.5, "understanding"),
    "vggsound": ("Loie/VGGSound", None, "data/corpora/vggsound", 338.2, "understanding"),
    "fsd50k": ("nateraw/fsd50k", ["data/*"], "data/corpora/fsd50k", 23.3, "understanding"),
    "meld": ("declare-lab/MELD", None, "data/corpora/meld", 11.8, "understanding"),

    # Vietnamese ASR corpora & benchmarks (see docs/DATASETS_VI.md; shortcut: --vi_stage1)
    "gigaspeech2_vi_eval": ("doof-ferb/gigaspeech2_vie", ["data/validation-*", "data/test-*"], "data/corpora/gigaspeech2_vi", 2.5, "asr_vi"),
    "fleurs_vi": ("google/fleurs", ["parquet-data/vi_vn/*"], "data/corpora/fleurs_vi", 1.5, "asr_vi"),
    "vlsp2020_100h": ("doof-ferb/vlsp2020_vinai_100h", ["data/*"], "data/corpora/vlsp2020_100h", 12.0, "asr_vi"),
    "lsvsc": ("doof-ferb/LSVSC", ["data/*"], "data/corpora/lsvsc", 12.0, "asr_vi"),
    "vimd": ("nguyendv02/ViMD_Dataset", ["data/*"], "data/corpora/vimd", 16.0, "asr_vi"),
    "fpt_fosd": ("doof-ferb/fpt_fosd", ["data/*"], "data/corpora/fpt_fosd", 2.6, "asr_vi"),
    # Stage 1 (~186 h slice = first 100/803 train shards + full val/test) vs full 1,494 h PhoAudiobook
    "phoaudiobook_200h": ("thivux/phoaudiobook", ["data/train-000*.parquet", "data/validation-*", "data/test-*"], "data/corpora/phoaudiobook_200h", 21.0, "asr_vi"),
    "phoaudiobook": ("thivux/phoaudiobook", ["data/*"], "data/corpora/phoaudiobook", 167.0, "asr_vi"),
}

DEFAULT_ENABLED = [
    "librispeech_clean360",
    "ami_ihm",
    "fleurs_en",
    "edacc",
    "speech_commands",
    "clotho_aqa",
    "esc50",
    "vocalsound",
    "cremad",
    "mmau_mini",
    "air_bench",
]

VI_STAGE1 = [
    "gigaspeech2_vi_eval",
    "fleurs_vi",
    "vlsp2020_100h",
    "lsvsc",
    "vimd",
    "fpt_fosd",
    "phoaudiobook_200h",
]


def download_corpus(name):
    if name not in REGISTRY:
        raise ValueError(f"Unknown corpus '{name}'. Available: {list(REGISTRY.keys())}")
    repo_id, allow_patterns, target_dir, approx_gb, kind = REGISTRY[name]
    target_path = Path(target_dir)

    if name == "librispeech_clean360":
        done_360 = (Path("data/LibriSpeech/train-clean-360/.done")).exists()
        done_dev = (Path("data/LibriSpeech/dev-other/.done")).exists()
        if done_360 and done_dev:
            print(f"skip {name} (already done)")
            return
        print(f"downloading {name} (~{approx_gb} GB) via prepare_data.py...")
        subprocess.run([sys.executable, "prepare_data.py", "validation.other", "train.clean.360"], check=True)
        print(f"done {name}")
        return

    if name == "librispeech_other500":
        if (Path("data/LibriSpeech/train-other-500/.done")).exists():
            print(f"skip {name} (already done)")
            return
        print(f"downloading {name} (~{approx_gb} GB) via prepare_data.py...")
        subprocess.run([sys.executable, "prepare_data.py", "train.other.500"], check=True)
        print(f"done {name}")
        return

    if (target_path / ".done").exists():
        print(f"skip {name} (already done)")
        return

    print(f"downloading {name} (~{approx_gb} GB) from {repo_id} into {target_path}...")
    target_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, repo_type="dataset", allow_patterns=allow_patterns, local_dir=str(target_path))
    (target_path / ".done").touch()
    print(f"done {name}")


def main():
    parser = argparse.ArgumentParser(description="Download ASR and audio understanding corpora from Hugging Face.")
    parser.add_argument("--list", action="store_true", help="List all corpora in the registry")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated corpus names to download")
    parser.add_argument("--vi_stage1", action="store_true",
                        help=f"Download Vietnamese Stage 1 + benchmark suite ({','.join(VI_STAGE1)})")
    args = parser.parse_args()

    if args.list:
        print(f"{'Name':22} {'Repo ID':32} {'GB':>6}  {'Kind':13} {'Default':8} {'Status':10}")
        print("-" * 98)
        for name, (repo_id, _, target_dir, gb, kind) in REGISTRY.items():
            is_def = "yes" if name in DEFAULT_ENABLED else ("vi_s1" if name in VI_STAGE1 else "off")
            if name.startswith("librispeech_"):
                split_name = "train-clean-360" if "clean" in name else "train-other-500"
                done = (Path("data/LibriSpeech") / split_name / ".done").exists()
            else:
                done = (Path(target_dir) / ".done").exists()
            status = "done" if done else "pending"
            print(f"{name:22} {repo_id:32} {gb:6.1f}  {kind:13} {is_def:8} {status:10}")
        return

    if args.only:
        targets = [x.strip() for x in args.only.split(",") if x.strip()]
    elif args.vi_stage1:
        targets = VI_STAGE1
    else:
        targets = DEFAULT_ENABLED
    total_planned = sum(REGISTRY[t][3] for t in targets if t in REGISTRY)
    print(f"Starting downloads for {len(targets)} corpora (~{total_planned:.1f} GB total planned): {targets}\n")

    for name in targets:
        download_corpus(name)
    print("\nAll requested downloads finished.")


if __name__ == "__main__":
    main()
