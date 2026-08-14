#!/bin/bash
# v3_ar 학습 환경 준비.
#
# 팀 안내문의 `bash /workspace/setup.sh` 가 저장소에 없어서 만든 것이다.
# 필요한 패키지는 v3_ar 의 train.py / loss.py / model.py / dataset.py 임포트에서
# 역산했다. 학습 코드는 건드리지 않는다.
#
# 사용:
#   bash setup.sh

set -u
echo "===== 1. 현재 환경 ====="
python -V
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU", p.name, "%.1f GB" % (p.total_memory / 1073741824))
PY

echo ""
echo "===== 2. 패키지 설치 ====="
# --break-system-packages: RunPod 기본 이미지는 PEP668 로 시스템 파이썬이 잠겨 있다
pip install -q --no-input --break-system-packages \
    timm diffusers av pandas tensorboard 2>&1 | tail -2

echo ""
echo "===== 3. 임포트 확인 ====="
python - <<'PY'
import importlib, sys
ok = True
need = ["torch", "torchvision", "timm", "diffusers", "av", "pandas", "numpy"]
for m in need:
    try:
        x = importlib.import_module(m)
        print("  %-12s %s" % (m, getattr(x, "__version__", "?")))
    except Exception as e:
        print("  %-12s 실패 %s" % (m, e)); ok = False
try:
    from torch.utils.tensorboard import SummaryWriter
    print("  %-12s OK" % "tensorboard")
except Exception as e:
    print("  %-12s 실패 %s" % ("tensorboard", e)); ok = False
sys.exit(0 if ok else 1)
PY
[ $? -ne 0 ] && { echo "[실패] 패키지 설치 확인 실패"; exit 1; }

echo ""
echo "===== 4. 경로 확인 ====="
for p in /workspace/v3_ar /workspace/data/train; do
    if [ -d "$p" ]; then
        echo "  있음   $p"
    else
        echo "  ★없음  $p"
    fi
done
N=$(find /workspace/data/train -name "*.mp4" 2>/dev/null | wc -l)
echo "  train mp4 $N 개"

echo ""
echo "===== 5. 디스크 ====="
df -h / /workspace 2>/dev/null | grep -vE "^Filesystem"

echo ""
echo "준비 완료. 다음 순서:"
echo "  export HF_HOME=/workspace/.cache/huggingface"
echo "  cd /workspace/v3_ar"
echo "  python verify_data.py        # 학습 전 점검 12가지"
echo "  python train.py              # 통과하면 학습 시작"
