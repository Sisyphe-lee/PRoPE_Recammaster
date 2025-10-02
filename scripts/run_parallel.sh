#!/bin/bash

 在脚本开头处，清空代理环境变量
#  unset http_proxy
#  unset https_proxy
#  unset all_proxy

# 1. 设置您想要使用的GPU列表
GPUS_TO_USE=(4 5 6 7)

# 2. 循环遍历GPU列表，为每个GPU启动一个后台进程
echo "Starting parallel processing..."
for GPU_ID in "${GPUS_TO_USE[@]}"
do
  echo "-> Launching process on GPU ${GPU_ID}"
  
  # 使用 CUDA_VISIBLE_DEVICES 来为当前进程指定可见的GPU
  # 同时，使用 --gpu_id 参数来告知Python脚本它正在使用哪个GPU，以便正确切分数据
  CUDA_VISIBLE_DEVICES=${GPU_ID} python ./generate_caption.py --gpu_id ${GPU_ID} &
done

echo ""
echo "All processes launched in the background."
echo "You can monitor their progress using 'nvidia-smi' or 'htop'."
echo ""
echo "Waiting for all background processes to complete..."

# 3. 等待所有后台启动的进程执行完毕
wait
