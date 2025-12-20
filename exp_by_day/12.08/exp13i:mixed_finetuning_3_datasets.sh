## fix cooridinate oriantation bug

bash scripts/train.sh \
  -y i2v \
  -u wan22 \
  -S "multicam,sdg,rel10k" \
  -s "/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train,/nas/datasets/vipe_wild_sdg_1m,/nas/datasets/relestate10k/train" \
  -m "metadata/output_recam.csv,metadata/output_sdg_motion_small.csv,metadata/metadata_re10k.csv" \
  --dataset-weights "0.6,0.2,0.2" \
  -w exp13i \
  -b 8 \
  -c "3,4,5,6,7" \
  -g 92 \
  -v 24 \
  -i 200 \
  -t 0.5 \
  -R "training_log/12-11-183535_exp13i/checkpoints/step4136.ckpt"







    