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
streams.base_path=evaluation/example_eval/20251103_165044 \
pipeline.init.instance=null \
pipeline.post.depth_align_model=null \
pipeline.slam.keyframe_depth=null \
pipeline.slam.optimize_intrinsics=true \
pipeline.output.save_artifacts=true \
pipeline.output.save_viz=false \
pipeline.output.save_slam_map=false \
pipeline.output.path=evaluation/example_eval/20251103_165044


3. 比较两组trajectory
python3 evaluation/evaluate_with_evo.py evaluation/example_eval/20251103_165044 evaluation/example_eval/20251103_205753/pose 


