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

## 需求3
+ 关注我的推理脚本：src/inference_recammaster.py 和启动代码：scripts/inference.sh。目前它仅支持单卡推理，请帮我实现多卡推理。我可以用CUDA_VISIBLE_DIVICES来指定哪些卡来inference
+ 请关注在load ckpt时 WAN21_RESUME_CHECKPOINT_PATH="/data1/lcy/projects/ReCamMaster/wandb/10-16-160559_Exp07j/checkpoints/step1100.ckpt 会不会有shape的问题，这个ckpt是用v2v模式训练过程中存的检查点。

## 需求4
+ 我现在想做v2v的evaluation。我们先完成第一步，能够inference pointodyssey 数据集，你可以建立一个新的脚本 render_pointodyssey.py写在evaluation目录里，你可以参考推理脚本src/inference_recammaster.py来实现。
+ 大致设想：
    + 在PointOdyssey的test集上实现v2v的evaluation。路径在/nas/datasets/PointOdyssey。你可以用它test set中的数据来获取condition video，condition trajectory 和 intrinsic。
        + 内部数据结构，以test为例：里面有一系列.mp4文件，和与对应.mp4文件对应的文件夹。在同名文件夹里的anno.npz里面包含该video的内外参。
        + 跳过不到81帧的数据；超过81帧的数据我们选取它前81帧，拿到对应的video和相机信息。
    + 关于target pose，我会给你指定一个.json文件，里面有数条target trajectory。数据里每一条video需要在所有target trajectory上做一次inference
    + 关于文件的存储和命名。我希望模型inference的结果可以存在/nas/users/lcy/v2v_eval里面。每次运行建立一个时间戳的子目录，把运行结果存在时间戳子目录里面。然后不要分rank和分cam_type的子目录，所有video都存在时间戳目录里，因此命名上用 {source_video_name}_{cam_type}.mp4 以区分。同时把每个video对应的target pose也存一遍（虽然target pose会重复存多次，但方便后续evaluation），同样以{source_video_name}_{cam_type}.npz命名。 

+ 具体实施：
    + 在inference code的基础上，你修改一下dataset以支持Pointodyssey。
    + 对于缺失的metadata.csv，我们v2v模式实质是不需要caption，因此你可以用一段空字符串或者固定长度的随机字符串来做self.text
    + 最后的文件保存与命名也需要作出修改
    + render_pointodyssey.py同样需要支持多卡处理

+ 回答未决问题：
1. 我打算不给你.json，而是输入一个目录里面包含若干个.npz。每一个.npz都是一个trajectory的pose，代表着一个target trajectory。.npz的一个例子如下：eval_data/videos/1.npz
2. 暂时不需要PointOdyssey的内参，请查清楚原先的推理脚本在不输入内参的情况下，模型是是如何推理的，third_party/DiffSynth-Studio/diffsynth/models/wan_video_dit.py 在里面self_attn是输入的是无内参还是默认的内参
3. 需要多卡并行
4. 需要resize到当前模型训练的分辨率，或者先查清楚原先的inference的脚本是如何处理分辨率的，再决定要不要follow

## 需求5
1. 目前的evaluation/render_pointodyssey.py已经基本实现功能。请先理解这个推理脚本。
2. 目前这个推理脚本无法resume一个已经推理部分的目录，具体是说我某次推理已经render的train split的一半，现在我想接着上次的继续推理，而不要重新开始。
3. 请先确保完整理解代码逻辑后，尽量最小程度的改动代码，不要过度设置鲁棒保护。有不明确的细节可以我们讨论后再实施。
### 具体实施
1. 可是给脚本加一个timestamp参数，这个参数默认是没有，如果没有就新建一个当前时间戳的目录，然后重新推理。如果启动时输入了参数timestamp，则先找这个timestamp命名的子目录，在这个子目录下继续推理。
2. 然后在推理的循环中，先确认目标.mp4（video_path）是否存在，如果存在就continue

## 需求6
+ 这是我之前的推理脚本 src/inference_recammaster.py 。然后我目前实现了一版既能够推理 example data也能够推理pointodyssey 数据集的推理脚本 src/inference_unified.py ，已经能够成功运行。
+ 但是这个新的推理脚本在推理example data的时候的结果和之前推理脚本的结果不一样。请你检查有哪些地方的实施和原始脚本不同。
+ 这是双方的启动命令：
    1. bash src/inference.sh
    2. ./scripts/inference_unified.sh example example_test_data example_test_data/target_traj_json wandb/10-16-160559_Exp07j/checkpoints/step1100.ckpt

+ 我给你一点检查的思路提示:
    1. 首先需要确认input 是否一致。已知用的是同样的input video和condition pose。你需要确认target pose是否一致，原始的需要.json文件作为输入，新的推理脚本输入的是一个目录下的.npz文件，这里的.npz是我读取.json然后转换过来的，理论上是一样的，但还是建议你读取文件然后确认是否一致。
    2. 在输入模型前的数据预处理是否一样，包括对视频裁剪到模型分辨率。对位姿的处理，比如说对齐到原点，计算相对位姿，平移归一化，计算w2c等。
    3. 已知是load同一个checkpoint，模型因此大概率是一样的。但推理管线pipeline是否一样还需要确认。

## 需求7
+ 我的通用推理脚本 src/inference_unified.py 目前已经可以支持多个数据集（example data和pointodyssey）的V2V推理了。目前我想进一步实现 i2v 推理，该脚本有着不错的拓展性，再次基础上实现i2v不算难
+ 我们进一步讨论下实现i2v的细节
    1. 目前我仅需要你支持example data就行，不需要考虑pointodyssey 
    2. 这是i2v的训练脚本 src/train_recammaster.py src/dataset.py src/lightning_trainer.py。其中validation_step和inference的情况基本一致，或许能给你很好的参考。
    3. i2v inference的输入只需要condition image和target trajectory，不需要condition video和condition trajectory。为了进一步增加鲁棒性，如果此时的dataset是video时，你可以取第一帧做condition image。这是example data的路径evaluation/i2v_eval
    4. 对于推理需要的pipeline和model，你可以查看validation_step是如何load。我建议仔细理解i2v的pipeline：third_party/DiffSynth-Studio/diffsynth/pipelines/wan_video_new.py。重新封装main中如何load 模型的部分，如果是v2v则调用WanVideoReCamMasterPipeline并load wan2.1,如果是i2v则调用WanVideoPipeline并load wan2.2
    5. 最后如何存推理结果依然复用v2v的逻辑存inference video和pose
+ 实施纲领：
    1. 尽可能保持代码的可拓展性。比如说要留好以后用i2v来推理其他数据集的接口
    2. 尽可能保持代码的复用性，如果能利用已有的函数就不要重复造轮子
    3. 不要过度封装和鲁棒检测，一定要兼顾代码的优雅与可读性
+ 对于实施的细节有什么不清楚的地方，我们先进行讨论。明确后给出一个详细的可执行的方案，经我确认后再修改代码

+ 确认细节：
    1. 仍然是笛卡尔积，一个video需要inference所有pose
    2. 如果有数据集中有metadata.csv则输入文本，如果没有直接用空字符串。沿用现有NEGATIVE_PROMPT
    3. 分辨率固定


## 需求8
+ 目前的训练脚本src/train_recammaster.py src/lightning_trainer.py src/dataset.py中已经支持v2v和i2v的camera control训练。
+ 目前我想进一步扩展训练时支持的数据集。目前仅能在MultiCam dataset上进行训练，/nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset.但是我现在希望能够支持在relestate10k：/nas/datasets/re10k  上训练。
+ 实现原则：
    1. 尽可能保证少的修改代码，在重复理解已有框架的情况下修改，不要大段造轮子。
    2. 代码要做到简洁优雅，有着高度可读性。
    3. 要实现一定的拓展性，以后还会加入其他的数据集
### 提取vae feature
+ 在训练前的第一步是提取并保存re10k的vae feature。这是我原先的提取vae feature的脚本和启动命令: src/vae_feature.py scripts/extract_vae.sh. 它目前只支持提取MultiCamVideo-Dataset的feature.
+ 具体实施建议:
    1. 你可以在 src/vae_feature.py 里面新建立一个dataset来适配rel10k数据集。需要你先完整的理解原始数据的文件格式。
    2. re10k的数据存储比较麻烦，是一系列.torch文件，每个里面有若干段数据；还有一个.json文件，标记哪个数据是属于哪个.torch。因此我希望在提取并存储vae feature的同时，重新整理一下数据集的格式。我新建立了一个文件夹/nas/datasets/relestate10k,里面有train和test子目录。
        + 在trian或者test的split子目录下我希望这样存每个数据，可以在存vae feature的时候顺便实现：
            + 以每段数据的序列名建立子目录。然后序列名子目录下有一段.mp4文件，一个intrinsics.npz，一个extrinsics.npz，一个metadata.json来这个数据的其他元信息。最后还有一个.wan22.tensors.pth来存vae feature    
    3. 加入resume功能（如果当前就有，可以不加），当在提取vae feature被中断时，我们不需要再次重新提取全部feature，而是跳过已经提取的。

+ 有什么不清楚的细节可以先问我，我们先讨论出一个合理的方案，我确认后再实施修改。我在一些我认为方便你修改的地方加了##TODO 注释来提醒。

### 实现训练脚本的rel10k支持
+ 目前重新整理过并提取的rel10k数据集路径在这里：/nas/datasets/relestate10k。现在我想实现训练脚本中对rel10k数据集的支持，训练脚本相关的文件：src/dataset.py，src/train_recammaster.py，src/lightning_trainer.py。
+ 目前训练脚本已经支持在i2v和v2v两种模式训练。这两种模式只能支持multicam数据集。我现在希望能够支持rel10k，但实际上v2v不会在rel10k上训练。

+ 再次重申实现原则：
    1. 尽可能保证少的修改代码，在重复理解已有框架的情况下修改，不要大段造轮子。
    2. 代码要做到简洁优雅，有着高度可读性。
    3. 要实现高度的拓展性，可以方便的接入未来可能用到的其他数据集

+ 具体实现建议，我认为需要修改的地方已经用##TODO:标记出来：
    1. 我想实现一个wraper，能够在train i2v model的时候，可以在两个数据集里面随机sample，而不是每次训练只能在某一个数据集上进行，具体来说的实现可能需要你给我一些启发。我建议可以调整成有一个i2v的基类dataset，然后给每个数据集都有一个dataset。
    2. 因为理论上rel10k不会在v2v上训练，因此我觉得 ImageConditionTensorDataset 可以不用继承TensorDataset，而是单独是一个基类或者在init中额外加入处理多数据sample的逻辑。但我不确认有没有必要给每个数据集(multicam,rel10k)都设置一个dataset class来继承ImageConditionTensorDataset（用TODO标出），这个你可以自己把握
    3. rel 10k的内参是每个场景都不一样的，在每个场景里面的intrinsics.npz
    4. rel 10k的metadata.csv（里面包含caption）的格式和mulcam的一样，因此理论上只需要在两个metadata.csv里面sample。
    5. 为了提高代码拓展性，我希望修改尽量限制在 dataset上，训练流程的其他地方尽量不要改变

+ 有什么不清楚的细节可以先问我，我们先讨论出一个合理的方案，我确认后再实施修改。


## 需求9
### setting
+ 我在multi recam上进行测试。
+ 先用训练脚本进行测试，查看了在开始训练前第一次validate的结果：exp_by_day/11.8/exp13a:new_caption_i2v.sh。
+ 用在同一个数据上用官方的推理管线测试
### 结果与疑问
+ 在用我用wan2.2TI2V-5B官方的推理管线时，对图片进行了resize，实现了居中最大面积裁剪的效果，保持图片比例的同时裁剪了最大的面积。third_party/DiffSynth-Studio/examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py
+ 但是我在运行训练脚本的src/lightning_trainer.py：validation_step时，发现decode出来的结果没有实现最大面积的裁剪，只是裁剪了中心区域的一部分，这样很容易无法裁剪到完整的人物动态。
+ 请帮我分析是validation_step是如何resize的，导致只裁剪了中心区域的一部分。
+ 这里有一些可能的地方需要你排查:
    1. 在提取vae feature时的是如何resize的 src/vae_feature.py，尤其是latents的分辨率。。提示：在multicam上提取vae feature时没有保存latents，并在在validation_step时也只是传递了空的image_emb，用第一帧的latents做img condition
    2. 在validation_step之前，dataset对latents是否做了处理
    3. 在validation_step最后decode and combine video是否做了处理