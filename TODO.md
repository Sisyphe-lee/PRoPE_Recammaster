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

### 说明补充
1. **wan2.2 的 condition 机制**：I2V 模式下条件输入不再是整段条件视频潜变量，而是源图像（或其 VAE 潜变量）作为首帧，后续帧由模型在去噪过程中生成。DiffSynth 的 `WanVideoPipeline.encode_image` 会生成 `clip_feature` 与 `y` 两类嵌入：`clip_feature` 来自图像编码器的语义向量，`y` 则由 VAE 对首尾帧编码并拼接掩膜，用于向扩散模型注入固定的首帧潜变量。在推理或训练时，调度器会始终保持首帧潜变量不被加噪，其他帧则依据噪声预测逐步还原。
2. **官方 TI2V 时间步控制**：DiffSynth 的 Wan2.2 推理（`WanVideoPipeline`）在首帧图像编码后，会设置 `fuse_vae_embedding_in_latents=True` 并缓存 `first_frame_latents`。`model_fn_wan_video` 检测到该标记后，调用 `WanModel` 时会：
   - 在每次 `scheduler.step` 后把首帧潜变量重写为原图；
   - 生成补丁级 timestep 序列：首帧所有 patch 的时间步被强制设为 0，其余帧按常规 timestep 赋值；
   - 将该二维 timestep 传给 `WanModel.forward`，后者在 `seperated_timestep` 模式下把时间嵌入按 patch 展开（首帧恒为 0，其余帧跟随噪声 schedule）。这一流程确保模型明确知道首帧是参考图像，其余帧围绕它去噪。
   我们在 `validation_step` 中复刻了这条数据流：生成噪声后缓存 `first_frame_latents`，构造和官方一致的 patch-level timestep（首帧补丁全为 0，其余补丁为当前 timestep），再将 `fuse_vae_embedding_in_latents=True`、`first_frame_latents` 一并传给 DiT；`WanModel.forward` 支持接收二维 timestep 并按官方方式展开。这样 I2V 验证与 DiffSynth 推理保持一致，首帧条件可以稳定生效。

## 需求2
+ 目前已经实现i2v的dataset和train_step,val_step。但是在val_step的时候遇到了bug，下面我为你详细描述这个bug
+ 首先i2v时load的model是wan2.2 TI2V 5B。我的val_step是给首帧的latents，然后其余的F-1个latents全是高斯噪声，和推理的行为一致。并且当我的T_HIGHFREQ_RATIO=0时，即不进行PRoPE，理论上模型与原始模型一模一样，那么应该就和TI2V的正常推理结果一致，但是目前进行在训练前进行的第一次val_step效果不太好，似乎后续生成的video和第一个latents（image condition）没关系，并且质量不够高。请你排查两个地方，一个是我的validation_step的逻辑是否错误，另一个是模型文件有误。目前我已经对齐了ti2v默认的720p分辨率(在train.sh) 。这是我的启动脚本：exp_by_day/10.30/exp11:i2v.sh
+ 我现在基本已经确认原因所在，因为我没有在validation时对timestep做处理，Diffsynth官方是在对首帧timestep置0，其余的正常去噪。而我的首帧没有置0。请确认是不是这个问题，如果是我们可以进一步讨论该如何修改。
+ 我给你一个Diffsynth 推理wan2.2的代码供参考,应该有助于你排查bug。它的TI2V的推理启动命令：python /data1/lcy/projects/ReCamMaster/third_party/DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py