#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
用法: CUDA_VISIBLE_DEVICES=0,2 NPROC=2./scripts/inference_unified.sh <dataset_kind> <dataset_path>
    <target_pose_dir> <ckpt> [extra args]

示例:
   ./scripts/inference_unified.sh example example_test_data evaluation/target_traj wandb/10-16-160559_Exp07j/checkpoints/step1100.ckpt
EOF
}

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 DATASET_KIND DATASET_PATH TARGET_POSE_DIR CKPT_PATH [extra args...]" >&2
  exit 1
fi

DATASET_KIND=$1
DATASET_PATH=$2
TARGET_POSE_DIR=$3
CKPT_PATH=$4
shift 4

NPROC=${NPROC:-1}

BASE_ARGS=(
  --dataset_kind "${DATASET_KIND}"
  --dataset_path "${DATASET_PATH}"
  --target_pose_dir "${TARGET_POSE_DIR}"
  --ckpt_path "${CKPT_PATH}"
  --debug
)

if [ "${NPROC}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="${NPROC}" src/inference_unified.py "${BASE_ARGS[@]}" "$@"
else
  python src/inference_unified.py "${BASE_ARGS[@]}" "$@"
fi
