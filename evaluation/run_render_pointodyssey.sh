#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
用法: evaluation/run_render_pointodyssey.sh DATASET_ROOT TARGET_POSE_DIR CKPT_PATH [-- 其他 render_pointodyssey.py 选项]

示例:
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 evaluation/run_render_pointodyssey.sh \
      /nas/datasets/PointOdyssey \
      evaluation/target_traj \
      wandb/10-16-160559_Exp07j/checkpoints/step1100.ckpt \
      --timestamp 20251101_202538 --seed 42
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "$#" -lt 3 ]]; then
    usage
    exit 1
fi

DATASET_ROOT="$1"
TARGET_POSE_DIR="$2"
CKPT_PATH="$3"
shift 3

if [[ "${1:-}" == "--" ]]; then
    shift
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _visible_gpu_array <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_GPUS="${#_visible_gpu_array[@]}"
else
    NUM_GPUS="$("$PYTHON_BIN" - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)"
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -lt 1 ]]; then
    NUM_GPUS=1
fi

echo "[info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} -> 使用 GPU 数量: $NUM_GPUS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT="${SCRIPT_DIR}/render_pointodyssey.py"

if [[ "$NUM_GPUS" -gt 1 ]]; then
    MASTER_PORT="${MASTER_PORT:-29500}"
    LAUNCH=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" --master_port "$MASTER_PORT")
else
    LAUNCH=("$PYTHON_BIN")
fi

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    "${LAUNCH[@]}" "$ENTRYPOINT" \
    --dataset_root "$DATASET_ROOT" \
    --target_pose_dir "$TARGET_POSE_DIR" \
    --ckpt_path "$CKPT_PATH" \
    "$@"
