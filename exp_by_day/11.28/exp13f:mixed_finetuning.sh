# bash scripts/train.sh \
#   -y i2v \
#   -u wan21 \
#   -S "multicam,sdg" \
#   -s "/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train,/nas/datasets/vipe_wild_sdg_1m" \
#   -m "metadata/output_recam.csv,metadata/output_sdg.csv" \
#   --dataset-weights "0.8,0.2" \
#   -w exp13f \
#   -b 4 \
#   -c "2,3,4,5,6,7" \
#   -g 92 \
#   -v 24 \
#   -i 200 \
#   -t 0.5 \
#   -R "training_log/11-28-021628_exp13e/checkpoints/step650.ckpt"


## resume
bash scripts/train.sh \
  -y i2v \
  -u wan21 \
  -S "multicam,sdg" \
  -s "/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train,/nas/datasets/vipe_wild_sdg_1m" \
  -m "metadata/output_recam.csv,metadata/output_sdg.csv" \
  --dataset-weights "0.8,0.2" \
  -w exp13f \
  -b 4 \
  -c "2,3,4,5,6,7" \
  -g 92 \
  -v 24 \
  -i 200 \
  -t 0.5 \
  -R "training_log/11-28-132705_exp13f/checkpoints/step200.ckpt"
   