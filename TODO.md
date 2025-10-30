## 项目介绍
+ 这个项目原先是基于wan2.1（T2V）的相机控制模型。具体说明可以查看 README.md
+ 它的输入是一段已知pose的video和一段目标的trajectory，输出是模型预测的目标trajectory下的video。
+ 它把一个T2V的视频生成模型微调成了一个V2V的模型，具体的实施是把target latents和condition latents在sequence维度cat在一起。然后仅对前一半（target latents）加噪去噪和做loss。
+ 在dataset.py里面有对应latents和camera的处理方式。

## 需求
+ 目前我们在项目支持了wan2.2(I2V)的模型，通过参数-y控制是wan2.1还是wan2.2。
+ 但是 I2V 模型的输入发生了改变。首先condition不再是latents中的一段posed video，而是一张image。目前可以load wan2.2进行训练。但实际上的load latents和lightening trainer等中仍然只是适配之前wan2.1的setting
+ 现在我希望你可以实现wan2.2相关的训练流程，包括I2V专用的dataset、修改trainer中的相关功能
+ 可以补充我遗漏的，实现训练的关键步骤

### dataset
1. 希望为I2V新写一个dataset类，通过-y参数判断是使用v2v的dataset还是i2v的dataset。
2. 请先理解wan2.2是如何实现image condition的
3. 重点的get item中，现在不需要load两段video的latents分别作为target和condition。而是只需要load一段video，video的第一帧（的latents，请确认condition需要image还是image的latents）作为condition，整段video的latents上去噪，对应的pose作为target trajectory。
4. trajectory仍然需要normalization，先计算第一帧(condition image)的relative pose，并把translation 归一化。

### trainer
1. 修改已用的trainer：LightningModelForTrain。通过-y来适配不同的训练模式。请你补充选择i2v时training和validation的逻辑。
2. 这种情况下我们需要对整段video的latents做加噪去噪，计算loss
3. 在log video时，condition video表示为一张静态图片，可以把广播到对应帧数