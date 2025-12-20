## fix cooridinate oriantation bug

bash scripts/train.sh \
  -y i2v \
  -u wan21 \
  -S "multicam,sdg" \
  -s "/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train,/nas/datasets/vipe_wild_sdg_1m" \
  -m "metadata/output_recam.csv,metadata/output_sdg_with_pose.csv" \
  --dataset-weights "0.7,0.3" \
  -w exp13h \
  -b 4 \
  -c "0" \
  -g 92 \
  -v 12 \
  -i 200 \
  -t 0.5 \
  -R training_log/11-28-021628_exp13e/checkpoints/step650.ckpt\
  -d 





    