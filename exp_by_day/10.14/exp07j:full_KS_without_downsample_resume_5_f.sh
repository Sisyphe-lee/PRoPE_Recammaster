## 已经在5帧上验证了没问题
## 在全内参，不降采样的情况下完整train一版本
## trian from wan2.1 original ckpt


# bash ./scripts/train.sh -b 2 -c "2,3,4,5,6,7" -g 42 -w Exp07j -R /data1/lcy/projects/ReCamMaster/wandb/10-08-111048_Exp07g/checkpoints/step2722.ckpt

# ## 因为训练中断，接着上一次训练的继续
# bash ./scripts/train.sh -b 2 -c "2,3,4,5,6,7" -g 42 -w Exp07j -R /data1/lcy/projects/ReCamMaster/wandb/10-14-134124_Exp07j/checkpoints/step300.ckpt

# ## 因为训练中断，接着上一次训练的继续
# bash ./scripts/train.sh -b 2 -c "4,5,6,7" -g 42 -w Exp07j -R /data1/lcy/projects/ReCamMaster/wandb/10-15-095226_Exp07j/checkpoints/step100.ckpt
 ## 因为训练中断，接着上一次训练的继续
# bash ./scripts/train.sh -b 2 -c "3,4,5,6,7" -g 42 -w Exp07j -R  /data1/lcy/projects/ReCamMaster/wandb/10-15-145432_Exp07j/checkpoints/step200.ckpt
## 因为训练中断，接着上一次训练的继续
bash ./scripts/train.sh -b 1 -c "2,3,4,5,6,7" -g 42 -w Exp07j -R /data1/lcy/projects/ReCamMaster/wandb/10-16-160559_Exp07j/checkpoints/step1100.ckpt