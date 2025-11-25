#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
用法: CUDA_VISIBLE_DEVICES=0,2 NPROC=2 ./scripts/inference_unified.sh \
    <dataset_kind> <dataset_path> <target_pose_dir> <ckpt> [pipeline_kind] [extra args]

说明:
  pipeline_kind 可选，默认 v2v，可指定为 i2v。

示例:
   ./scripts/inference_unified.sh example example_test_data evaluation/v2v_eval/target_traj \
       models/checkpoints/step6631.ckpt v2v
   ./scripts/inference_unified.sh example_i2v evaluation/i2v_eval/example_data evaluation/i2v_eval/example_data/target_traj \
       training_log/11-14-195326_exp13a/checkpoints/step10208.ckpt i2v 
EOF
}

if [ "$#" -lt 4 ]; then
  usage
  exit 1
fi

DATASET_KIND=$1
DATASET_PATH=$2
TARGET_POSE_DIR=$3
CKPT_PATH=$4
shift 4

PIPELINE_KIND=${PIPELINE_KIND:-v2v}
if [ "$#" -ge 1 ]; then
  case "$1" in
    v2v|i2v)
      PIPELINE_KIND=$1
      shift
      ;;
  esac
fi

NPROC=${NPROC:-1}

BASE_ARGS=(
  --dataset_kind "${DATASET_KIND}"
  --dataset_path "${DATASET_PATH}"
  --target_pose_dir "${TARGET_POSE_DIR}"
  # --ckpt_path "${CKPT_PATH}"
  --pipeline_kind "${PIPELINE_KIND}"
  --num_inference_steps 10
  # --debug
)

if [ "${NPROC}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="${NPROC}" src/inference_unified.py "${BASE_ARGS[@]}" "$@"
else
  python src/inference_unified.py "${BASE_ARGS[@]}" "$@"
fi
