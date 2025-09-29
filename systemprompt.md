## PRoPE on ReCamMaster
+ 这是一个基于视频生成模型（wan2.1）的相机控制模型
+ wan2.1是一个使用flow matching技术的视频生成模型
+ codebase是ReCamMaster，一个相机控制模型，这是他的模型pipeline：diffsynth/pipelines/wan_video_recammaster.py。在diffsynth/models/wan_video_dit.py中可以看见ReCamMaster的核心模块，是如何condition target video和如何来注入位姿的。
+ 我在它上面进行了改进，修改了模型注入相机位姿的方式，通过修改旋转位置编码来注入位姿。移除了原来通过MLP映射后直接将位姿加入self_attn前的feature的方式。而是在模型进行旋转位置编码的时候，在三维旋转位置编码的负责时间的低频通道中注入PRoPE提出的projection term
+ 模型的Dit block在这里diffsynth/models/wan_video_dit.py，也是我对模型进行修改的地方。
+ 这是训练脚本train_recammaster.py和启动脚本train.sh