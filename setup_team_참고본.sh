#!/bin/bash
set -e

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface

echo "=== pip install ==="
pip install -q diffusers==0.31.0 timm av pandas pyarrow tensorboard accelerate transformers

echo "=== model cache warmup ==="
python - <<'PYEOF'
import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

from diffusers import AutoencoderKL
AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
print("[OK] SDXL VAE")

import timm
timm.create_model("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", pretrained=True, num_classes=0)
print("[OK] EVA02-B")

from torchvision.models.video import MC3_18_Weights, mc3_18
mc3_18(weights=MC3_18_Weights.DEFAULT)
print("[OK] MC3-18")
PYEOF

echo "=== setup done ==="