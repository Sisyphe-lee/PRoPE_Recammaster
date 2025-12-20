## fix cooridinate oriantation bug

bash scripts/train.sh \
  -y i2v \
  -u wan22 \
  -S "multicam" \
  -s "/nas/datasets/relestate10k/train" \
  -m "metadata/debug_recam.csv" \
  -w debug \
  -b 4 \
  -c "2" \
  -g 92 \
  -v 1 \
  -i 200 \
  -t 0.5 \
  -R "training_log/12-11-183535_exp13i/checkpoints/step3936.ckpt" \
  -d





    