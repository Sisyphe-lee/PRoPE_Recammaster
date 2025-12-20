#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
用法: CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 ./scripts/inference_unified.sh \
    <dataset_kind> <dataset_path> <target_pose_dir> <ckpt> [pipeline_kind] [extra args]

说明:
  pipeline_kind 可选，默认 v2v，可指定为 i2v。

示例:
   CUDA_VISIBLE_DEVICES=0,1 NPROC=2 ./scripts/inference_unified.sh example example_test_data evaluation/v2v_eval/target_traj \
       models/checkpoints/step6631.ckpt v2v

   CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 NPROC=6 ./scripts/inference_unified.sh example_i2v evaluation/i2v_eval/example_test evaluation/i2v_eval/example_test/target_traj \
       training_log/12-17-190950_exp13j/checkpoints/step3260.ckpt i2v

    DATASET_OPTION="metadata_path=/data1/lcy/projects/ReCamMaster/metadata/sdg_subset.csv pose_dir=/nas/datasets/vipe_wild_sdg_1m/pose" \
    CUDA_VISIBLE_DEVICES=0,1 NPROC=2 \
        ./scripts/inference_unified.sh sdg_v2v /nas/datasets/vipe_wild_sdg_1m evaluation/v2v_eval/target_traj models/checkpoints/step6631.ckpt v2v
 
    DATASET_OPTION="metadata_path=/data1/lcy/projects/ReCamMaster/metadata/sdg_subset_i2v.csv" \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 NPROC=6 \
        ./scripts/inference_unified.sh sdg_i2v /nas/datasets/vipe_wild_sdg_1m evaluation/i2v_eval/target_traj \
        training_log/12-17-190950_exp13j/checkpoints/step3260.ckpt i2v
    

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
I2V_CKPT_TYPE=${I2V_CKPT_TYPE:-wan22}

NPROC=${NPROC:-1}

# 若使用 sdg_v2v，可通过环境变量或附加参数传递 metadata_path / pose_dir，例如：
# DATASET_KIND=sdg_v2v DATASET_OPTION="metadata_path=metadata/sdg_subset.csv pose_dir=/nas/datasets/vipe_wild_sdg_1m/pose"
DATASET_OPTION=${DATASET_OPTION:-}
EXTRA_DATASET_OPTS=()
if [ -n "${DATASET_OPTION}" ]; then
  for kv in ${DATASET_OPTION}; do
    EXTRA_DATASET_OPTS+=(--dataset_option "${kv}")
  done
fi

BASE_ARGS=(
  --dataset_kind "${DATASET_KIND}"
  --dataset_path "${DATASET_PATH}"
  --target_pose_dir "${TARGET_POSE_DIR}"
  --ckpt_path "${CKPT_PATH}"
  --pipeline_kind "${PIPELINE_KIND}"
  --i2v_ckpt_type "${I2V_CKPT_TYPE}"
  --num_inference_steps 25
  --output_dir "evaluation/i2v_eval/sdg_eval"
  "${EXTRA_DATASET_OPTS[@]}"
)

if [ "${NPROC}" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="${NPROC}" src/inference_unified.py "${BASE_ARGS[@]}" "$@"
else
  python src/inference_unified.py "${BASE_ARGS[@]}" "$@"
fi
