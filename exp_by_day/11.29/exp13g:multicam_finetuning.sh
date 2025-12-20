## Resume "training_log/11-28-021628_exp13e/checkpoints/step200.ckpt" initially
## Resume training from "training_log/11-29-185758_exp13g/checkpoints/step3067.ckpt"
bash scripts/train.sh \
  -y i2v \
  -u wan21 \
  -S "multicam" \
  -s "/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train" \
  -m "metadata/output_recam.csv" \
  -w exp13g \
  -b 4 \
  -c "2,3,4,5,6,7" \
  -g 92 \
  -v 24 \
  -i 200 \
  -t 0.5 \
  -R "training_log/11-29-185758_exp13g/checkpoints/step3067.ckpt"
  
