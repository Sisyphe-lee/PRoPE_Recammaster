1. render 
conda acitvate recammaster && CUDA_VISIBLE_DEVICES=0,1 evaluation/run_render_pointodyssey.sh \
    /nas/datasets/PointOdyssey \
    evaluation/target_traj \
    wandb/10-16-160559_Exp07j/checkpoints/step1100.ckpt \
    --seed 42

2. VIPE 估计pose
conda activate vipe && python /data1/lcy/projects/vipe/run.py \
pipeline=default \
streams=raw_mp4_stream \
streams.base_path=evaluation/v2v_eval/20251101_183816 \
pipeline.init.instance=null \
pipeline.post.depth_align_model=null \
pipeline.slam.keyframe_depth=null \
pipeline.slam.optimize_intrinsics=true \
pipeline.output.save_artifacts=true \
pipeline.output.save_viz=false \
pipeline.output.save_slam_map=false \
pipeline.output.path=evaluation/v2v_eval/20251101_183816


3. 比较两组trajectory
python3 /data1/lcy/projects/ReCamMaster/evaluate_with_evo.py evaluation/v2v_eval/20251101_183816 evaluation/v2v_eval/20251101_183816/pose --work-dir .evaluation/evo_outputs --per-file | cat


