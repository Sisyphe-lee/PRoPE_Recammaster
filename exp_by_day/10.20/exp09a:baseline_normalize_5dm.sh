## 解决viewmats的bug
## 做了normalization
## 加入gradient clip
## 5帧降采样
## resume ckpt /data1/lcy/projects/ReCamMaster/wandb/10-07-210247_Exp07g/checkpoints/step1312.ckpt



bash ./scripts/train.sh -T -F 5 -b 8 -c "4,5,6,7" -g 42 -w Exp09a 