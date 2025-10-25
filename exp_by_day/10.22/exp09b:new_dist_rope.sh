## 解决viewmats的bug
## 做了normalization
## 加入gradient clip
## 5帧降采样
## resume ckpt /data1/lcy/projects/ReCamMaster/wandb/10-07-210247_Exp07g/checkpoints/step1312.ckpt



bash ./scripts/train.sh -F 5 -T  -b 4  -c "2,3,4,5,6,7 " -g 16 -w Exp09b