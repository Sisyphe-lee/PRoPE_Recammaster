## 解决viewmats的bug
## 做了normalization
## 加入gradient clip
## resume ckpt /data1/lcy/projects/ReCamMaster/wandb/09-27-130130_Exp07b/checkpoints/step1815.ckpt


bash ./scripts/train.sh -b -F 5 -c "2,3,4,5,6,7" -g 42 -w Exp07g -R /data1/lcy/projects/ReCamMaster/wandb/09-27-130130_Exp07b/checkpoints/step1815.ckpt
# bash ./scripts/train.sh -F 5 -c "2" -d -g 42 -w Exp07g -R /data1/lcy/projects/ReCamMaster/wandb/09-27-130130_Exp07b/checkpoints/step1815.ckpt