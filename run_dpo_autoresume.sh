#!/usr/bin/env bash
# Cluster DPO 自动重启脚本：遇到 core dump(139) 时，从最新 checkpoint 续跑，
# 并临时跳过崩溃前最后若干 accession（通过过滤 train csv）。
#
# 训练目标与对照：与 cluster GRPO（run_grpo_muon_autoresume.sh）共用同一份
#   - 数据集（dataset_cluster_split_train/test.csv）
#   - SFT 初始化 checkpoint
#   - reward function（refusal-aware cluster reward）
#   - mask 方案 / NUM_GENERATIONS / batch 组成 / 固定验证集 / optimizer
# 仅训练目标不同（DPO vs GRPO），便于在论文中作 apples-to-apples 对照。

set -u
set -o pipefail

# ROOT 可用环境变量覆盖（迁移到 NAS / fr1a 时无需改文件）；默认= 脚本所在目录
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$ROOT"

# ===== 本机环境（venv / ESMFold / foldseek），可被外部环境变量覆盖 =====
export ESMFOLD_MODEL="${ESMFOLD_MODEL:-$(cd "$ROOT/.." && pwd)/models/esmfold_v1}"
export FOLDSEEK_BIN="${FOLDSEEK_BIN:-$ROOT/bin/foldseek/bin/foldseek}"
export PATH="$ROOT/bin/foldseek/bin:$PATH"

# ===== 基础配置（可直接改 / 通过环境变量覆盖） =====
# 默认使用官方 ProteinMPNN vanilla 权重（v_48_020）作为 DPO 初始化；
# 若有 refusal SFT checkpoint，可通过环境变量 INIT_CKPT 覆盖。
INIT_CKPT="${INIT_CKPT:-$ROOT/vanilla_model_weights/v_48_020.pt}"
RUN_DIR="${RUN_DIR:-/home/tiger/yxu/ProtGPT/ProteinMPNN/outputs/dpo_muon_autoresume}"
REF_PDB_CACHE="${REF_PDB_CACHE:-/home/tiger/yxu/ProtGPT/ProteinMPNN/pdb_cache/reference}"
GEN_PDB_CACHE="${GEN_PDB_CACHE:-/home/tiger/yxu/ProtGPT/ProteinMPNN/pdb_cache/generated}"
TMPDIR="${TMPDIR:-/home/tiger/yxu/ProtGPT/ProteinMPNN/tmp}"
LOG_PATH="$RUN_DIR/train.log"

# 训练超参（DPO cluster 默认设定）
GRPO_BATCH_SIZE="${GRPO_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MUON_LEARNING_RATE="${MUON_LEARNING_RATE:-8e-7}"
GRPO_CLUSTER_ACCUM_STEPS="${GRPO_CLUSTER_ACCUM_STEPS:-1}"
GRPO_VALIDATION_SIZE_PER_CLASS="${GRPO_VALIDATION_SIZE_PER_CLASS:-50}"
GRPO_VALIDATION_MASK_RATIO="${GRPO_VALIDATION_MASK_RATIO:-0.3}"
GRPO_VALIDATION_SEED="${GRPO_VALIDATION_SEED:-2026}"
# 0 表示完全跳过逐步 validation（每个真实 optimizer.step 后都不评估）
GRPO_VALIDATION_EVERY="${GRPO_VALIDATION_EVERY:-0}"
SAVE_CHECKPOINT_EVERY="${SAVE_CHECKPOINT_EVERY:-10}"
GRPO_OPTIMIZER="${GRPO_OPTIMIZER:-muon}"  # adam 或 muon；与 GRPO 实验对应

# DPO 专属超参（标准设定，与 Rafailov et al. 2023 默认范围一致）
DPO_BETA="${DPO_BETA:-0.1}"
DPO_MARGIN="${DPO_MARGIN:-0.02}"
DPO_MAX_PAIRS_PER_PROTEIN="${DPO_MAX_PAIRS_PER_PROTEIN:-3}"
DPO_RUN_NAME="${DPO_RUN_NAME:-dpo_v260506}"

EVAL_MASK_RATIOS="${EVAL_MASK_RATIOS:-0.1,0.3}"
# 训练完所有 epoch 后自动在 test 集上跑评估（用 mpnn_model_final.pt + EVAL_MASK_RATIOS）
# autoresume 期间的中途 segfault 不会触发 eval；只有真正训完 5 个 epoch 才会跑
ENABLE_EVAL_AFTER="${ENABLE_EVAL_AFTER:-true}"
DEBUG_CUDA_BLOCKING="${DEBUG_CUDA_BLOCKING:-false}"

# 自动恢复相关
MAX_RESTARTS="${MAX_RESTARTS:-20}"
BLACKLIST_WINDOW="${BLACKLIST_WINDOW:-8}"   # 崩溃前最后 N 个 accession 加入黑名单
PRUNE_KEEP_EVERY="${PRUNE_KEEP_EVERY:-20}"  # step checkpoint 每隔多少保留一个
PRUNE_KEEP_LAST="${PRUNE_KEEP_LAST:-8}"      # 额外保留最新 N 个

ORIG_TRAIN_CSV="${ORIG_TRAIN_CSV:-$ROOT/dataset/dataset_cluster_split_train.csv}"
ORIG_TEST_CSV="${ORIG_TEST_CSV:-$ROOT/dataset/dataset_cluster_split_test.csv}"
WORK_TRAIN_CSV="$RUN_DIR/train_filtered.csv"
BLACKLIST_TXT="$RUN_DIR/skip_accessions.txt"

ensure_writable_dir_or_fallback() {
  # $1 var name, $2 fallback dir
  local var_name="$1"
  local fallback="$2"
  local current="${!var_name}"
  if mkdir -p "$current" 2>/dev/null; then
    if touch "$current/.writetest" 2>/dev/null; then
      rm -f "$current/.writetest" 2>/dev/null || true
      return 0
    fi
  fi
  echo "[warn] $var_name is not writable: $current ; fallback -> $fallback"
  printf -v "$var_name" '%s' "$fallback"
  if ! mkdir -p "${!var_name}" 2>/dev/null; then
    echo "ERROR: fallback directory still not writable for $var_name: ${!var_name}" >&2
    exit 1
  fi
  if ! touch "${!var_name}/.writetest" 2>/dev/null; then
    echo "ERROR: cannot write fallback directory for $var_name: ${!var_name}" >&2
    exit 1
  fi
  rm -f "${!var_name}/.writetest" 2>/dev/null || true
}

ensure_writable_dir_or_fallback RUN_DIR "/home/tiger/yxu/ProtGPT/ProteinMPNN/outputs/dpo_muon_autoresume"
ensure_writable_dir_or_fallback REF_PDB_CACHE "/home/tiger/yxu/ProtGPT/ProteinMPNN/pdb_cache/reference"
ensure_writable_dir_or_fallback GEN_PDB_CACHE "/home/tiger/yxu/ProtGPT/ProteinMPNN/pdb_cache/generated"
ensure_writable_dir_or_fallback TMPDIR "/home/tiger/yxu/ProtGPT/ProteinMPNN/tmp"

mkdir -p "$RUN_DIR" "$REF_PDB_CACHE" "$GEN_PDB_CACHE" "$TMPDIR" || {
  echo "ERROR: failed to create runtime directories" >&2
  exit 1
}
touch "$BLACKLIST_TXT" || {
  echo "ERROR: cannot create blacklist file: $BLACKLIST_TXT" >&2
  exit 1
}
ulimit -c 0 || true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TMPDIR

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: INIT_CKPT 不存在: $INIT_CKPT" >&2
  exit 1
fi
if [[ ! -f "$ORIG_TRAIN_CSV" ]]; then
  echo "ERROR: ORIG_TRAIN_CSV 不存在: $ORIG_TRAIN_CSV" >&2
  exit 1
fi

build_filtered_csv() {
  ORIG_TRAIN_CSV="$ORIG_TRAIN_CSV" WORK_TRAIN_CSV="$WORK_TRAIN_CSV" BLACKLIST_TXT="$BLACKLIST_TXT" \
  python - <<'PY'
import os
import pandas as pd
from pathlib import Path
orig = Path(os.environ['ORIG_TRAIN_CSV'])
out = Path(os.environ['WORK_TRAIN_CSV'])
bl = Path(os.environ['BLACKLIST_TXT'])

df = pd.read_csv(orig)
skips = {x.strip() for x in bl.read_text().splitlines() if x.strip()} if bl.exists() else set()
if skips and 'accession' in df.columns:
    before = len(df)
    df = df[~df['accession'].astype(str).str.strip().isin(skips)]
    print(f"[filter] removed={before-len(df)} remain={len(df)} skips={len(skips)}")
else:
    print(f"[filter] no skip applied, rows={len(df)}")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
PY
}

latest_ckpt() {
  ls -1 "$RUN_DIR"/rl_checkpoint/mpnn_model_step_*.pt 2>/dev/null | sort -V | tail -1
}

prune_checkpoints() {
  RUN_DIR="$RUN_DIR" PRUNE_KEEP_EVERY="$PRUNE_KEEP_EVERY" PRUNE_KEEP_LAST="$PRUNE_KEEP_LAST" \
  python - <<'PY'
import os, re
from pathlib import Path
run_dir = Path(os.environ['RUN_DIR'])
keep_every = int(os.environ['PRUNE_KEEP_EVERY'])
keep_last = int(os.environ['PRUNE_KEEP_LAST'])
ckpt_dir = run_dir / 'rl_checkpoint'
files = sorted(ckpt_dir.glob('mpnn_model_step_*.pt'), key=lambda p: int(re.search(r'step_(\d+)\.pt$', p.name).group(1)))
if len(files) <= keep_last:
    raise SystemExit(0)
steps = [int(re.search(r'step_(\d+)\.pt$', p.name).group(1)) for p in files]
last_steps = set(steps[-keep_last:])
for p, s in zip(files, steps):
    if s in last_steps:
        continue
    if keep_every > 0 and (s % keep_every == 0):
        continue
    try:
        p.unlink()
    except Exception:
        pass
print(f"[prune] checkpoints={len(files)} keep_last={keep_last} keep_every={keep_every}")
PY
}

append_recent_accessions_to_blacklist() {
  local tmp="$RUN_DIR/.recent_acc.txt"
  grep -E "\\[(DPO-Rollout|Rollout)\\].*accession=.* start" "$LOG_PATH" > "$tmp" 2>/dev/null || true
  RUN_DIR="$RUN_DIR" BLACKLIST_WINDOW="$BLACKLIST_WINDOW" python - <<'PY'
import os, re
from pathlib import Path
run_dir = Path(os.environ['RUN_DIR'])
window = int(os.environ['BLACKLIST_WINDOW'])
log_lines = run_dir / '.recent_acc.txt'
bl = run_dir / 'skip_accessions.txt'
accs = []
if log_lines.exists():
    for line in log_lines.read_text().splitlines():
        m = re.search(r"accession=([^\s]+)", line)
        if m:
            accs.append(m.group(1))
accs = accs[-window:]
old = {x.strip() for x in bl.read_text().splitlines() if x.strip()} if bl.exists() else set()
new = old | set(accs)
bl.write_text("\n".join(sorted(new)) + ("\n" if new else ""))
print(f"[blacklist] add={len(set(accs)-old)} total={len(new)} last={accs}")
PY
}

run_once() {
  local ckpt="$1"
  build_filtered_csv
  echo "[run] ckpt=$ckpt" | tee -a "$LOG_PATH"
  echo "[run] output_dir=$RUN_DIR" | tee -a "$LOG_PATH"
  echo "[run] ref_pdb_cache=$REF_PDB_CACHE gen_pdb_cache=$GEN_PDB_CACHE tmpdir=$TMPDIR" | tee -a "$LOG_PATH"
  echo "[run] dpo_beta=$DPO_BETA dpo_margin=$DPO_MARGIN dpo_max_pairs=$DPO_MAX_PAIRS_PER_PROTEIN optimizer=$GRPO_OPTIMIZER" | tee -a "$LOG_PATH"

  local -a env_cmd=(env "PYTHONFAULTHANDLER=1")
  if [[ "$DEBUG_CUDA_BLOCKING" == "true" ]]; then
    env_cmd+=("CUDA_LAUNCH_BLOCKING=1")
  fi

  local -a extra_args=()
  if [[ "$ENABLE_EVAL_AFTER" == "true" ]]; then
    extra_args+=("--eval-after")
  fi

  "${env_cmd[@]}" python unified_training_dpo.py --method dpo \
    --grpo-data-source cluster_csv \
    --grpo-optimizer "$GRPO_OPTIMIZER" \
    --path-to-model-weights "$ckpt" \
    --output-run-prefix "$RUN_DIR" \
    --grpo-train-csv "$WORK_TRAIN_CSV" \
    --grpo-test-csv "$ORIG_TEST_CSV" \
    --grpo-reference-pdb-dir "$REF_PDB_CACHE" \
    --grpo-generated-pdb-dir "$GEN_PDB_CACHE" \
    --grpo-batch-size "$GRPO_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --muon-learning-rate "$MUON_LEARNING_RATE" \
    --grpo-cluster-accum-steps "$GRPO_CLUSTER_ACCUM_STEPS" \
    --grpo-validation-size-per-class "$GRPO_VALIDATION_SIZE_PER_CLASS" \
    --grpo-validation-mask-ratio "$GRPO_VALIDATION_MASK_RATIO" \
    --grpo-validation-seed "$GRPO_VALIDATION_SEED" \
    --grpo-validation-every "$GRPO_VALIDATION_EVERY" \
    --save-checkpoint-every "$SAVE_CHECKPOINT_EVERY" \
    --dpo-beta "$DPO_BETA" \
    --dpo-margin "$DPO_MARGIN" \
    --dpo-max-pairs-per-protein "$DPO_MAX_PAIRS_PER_PROTEIN" \
    --dpo-run-name "$DPO_RUN_NAME" \
    --eval-mask-ratios "$EVAL_MASK_RATIOS" \
    "${extra_args[@]}" \
    2>&1 | tee -a "$LOG_PATH"
  local train_code=${PIPESTATUS[0]}

  return "$train_code"
}

attempt=0
while true; do
  ckpt="$(latest_ckpt)"
  if [[ -z "${ckpt:-}" ]]; then
    ckpt="$INIT_CKPT"
  fi

  run_once "$ckpt"
  code=$?

  if [[ $code -eq 0 ]]; then
    echo "[done] training finished successfully" | tee -a "$LOG_PATH"
    exit 0
  fi

  echo "[warn] run exited code=$code" | tee -a "$LOG_PATH"

  # 典型 core dump 返回码 139（128+SIGSEGV）
  if [[ $code -ne 139 ]]; then
    echo "[abort] non-core-dump failure, stop autorestart" | tee -a "$LOG_PATH"
    exit $code
  fi

  attempt=$((attempt+1))
  if [[ $attempt -gt $MAX_RESTARTS ]]; then
    echo "[abort] exceed MAX_RESTARTS=$MAX_RESTARTS" | tee -a "$LOG_PATH"
    exit 139
  fi

  append_recent_accessions_to_blacklist
  prune_checkpoints
  echo "[resume] restart attempt=$attempt" | tee -a "$LOG_PATH"
  sleep 3
done
