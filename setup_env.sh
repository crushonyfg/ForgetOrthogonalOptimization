#!/usr/bin/env bash
# 在任意机器（含 fr1a）一键重建训练 venv。
# 用法:  bash setup_env.sh [venv路径]   # 默认在脚本同级建 .venv
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-$HERE/.venv}"

# 建 venv（优先 virtualenv，系统缺 ensurepip 时也能用）
if command -v virtualenv >/dev/null 2>&1; then
  virtualenv -p python3.11 "$VENV"
else
  python3.11 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip

# torch 2.9.1 需 cu126 wheel（内置 torch.optim.Muon，2.8 及以下没有）；
# --extra-index-url 让 requirements.txt 里的 torch==2.9.1+cu126 及 nvidia-* 能被解析到。
pip install -r "$HERE/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu126

echo "== venv ready: $VENV =="
python - <<'PY'
import torch, torch.optim as o
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| Muon", hasattr(o, "Muon"))
PY
