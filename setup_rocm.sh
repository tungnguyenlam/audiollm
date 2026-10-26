#!/usr/bin/env bash
# One-time environment setup (AMD ROCm, RX 9060 XT / gfx1200).
set -euo pipefail
cd "$(dirname "$0")"

uv venv --python 3.12 .venv
source .venv/bin/activate

# PyTorch ROCm wheel (bundles its own ROCm runtime; only the kernel driver is used from the system).
uv pip install torch --index-url https://download.pytorch.org/whl/rocm7.1

uv pip install "transformers>=5.9" peft accelerate safetensors soundfile numpy jiwer pyarrow huggingface_hub matplotlib soxr

# causal-conv1d: optimized kernel for LFM2's short convs (otherwise transformers falls back to the
# PyTorch reference and warns). No prebuilt ROCm wheel exists, so build from upstream source with the
# system hipcc (/opt/rocm). Pinned commit for reproducibility; ~2-3 min build.
#   HIP_ARCHITECTURES: gfx1200 = RX 9060 XT (setup.py default is "native").
#   --no-build-isolation: compile against the torch already in this venv.
CC1D_COMMIT=cd81f0413cad2fc1e6f17e785ac39f59aae690cd
CC1D_SRC=.build/cc1d-src
if [ ! -d "$CC1D_SRC/.git" ]; then
    git clone https://github.com/Dao-AILab/causal-conv1d.git "$CC1D_SRC"
fi
git -C "$CC1D_SRC" fetch --depth 1 origin "$CC1D_COMMIT"
git -C "$CC1D_SRC" checkout -q "$CC1D_COMMIT"
uv pip install ninja packaging wheel setuptools
CAUSAL_CONV1D_FORCE_BUILD=TRUE HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-gfx1200}" MAX_JOBS="${MAX_JOBS:-8}" \
    uv pip install --no-build-isolation --reinstall-package causal-conv1d "./$CC1D_SRC"

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
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
x, w = torch.randn(2, 64, 128, device="cuda"), torch.randn(64, 4, device="cuda")
ref = F.conv1d(x, w.unsqueeze(1), padding=3, groups=64)[..., :128]
print("causal_conv1d ok:", torch.allclose(causal_conv1d_fn(x, w), ref, atol=1e-4))
EOF
