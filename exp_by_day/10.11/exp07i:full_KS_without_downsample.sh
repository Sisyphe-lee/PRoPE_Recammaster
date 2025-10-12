## 已经在5帧上验证了没问题
## 在全内参，不降采样的情况下完整train一版本
## trian from wan2.1 original ckpt


bash ./scripts/train.sh -b 2 -c "2,3,4,5,6,7" -g 42 -w Exp07i 
