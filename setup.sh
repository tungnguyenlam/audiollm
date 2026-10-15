#!/usr/bin/env bash
# One-time environment setup (AMD ROCm, RX 9060 XT / gfx1200).
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.12 .venv
source .venv/bin/activate

# PyTorch ROCm wheel (bundles its own ROCm runtime; only the kernel driver is used from the system).
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

uv pip install "transformers>=5.9" peft accelerate safetensors soundfile numpy jiwer pyarrow huggingface_hub

python - <<'EOF'
import torch
print("torch", torch.__version__, "hip", torch.version.hip)
print("gpu ok:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
