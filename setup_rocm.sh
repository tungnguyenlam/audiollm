#!/usr/bin/env bash
# One-time environment setup (AMD ROCm, RX 9060 XT / gfx1200).
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.12 .venv
source .venv/bin/activate

# PyTorch ROCm wheel (bundles its own ROCm runtime; only the kernel driver is used from the system).
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

uv pip install "transformers>=5.9" peft accelerate safetensors soundfile numpy jiwer pyarrow huggingface_hub

# Inject ROCm env vars into the venv. A .pth file runs at every interpreter startup,
# so this applies to `uv run`, `.venv/bin/python` and activated shells alike.
# setdefault => an explicit value in your shell still wins.
#   TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: enable AOTriton flash / mem-efficient SDPA on gfx12xx
#     (otherwise torch falls back to the slow math attention kernel).
SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cat > "$SITE_PACKAGES/_rocm_env.pth" <<'EOF'
import os; os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
EOF

python - <<'EOF'
import os, torch
print("AOTriton experimental:", os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"))
print("torch", torch.__version__, "hip", torch.version.hip)
print("gpu ok:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
EOF
