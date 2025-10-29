1. VIPE 估计pose

python /data1/lcy/projects/vipe/run.py \
pipeline=default \
streams=raw_mp4_stream \
streams.base_path=/data1/lcy/projects/ReCamMaster/eval_data/videos \
pipeline.init.instance=null \
pipeline.post.depth_align_model=null \
pipeline.slam.keyframe_depth=null \
pipeline.slam.optimize_intrinsics=true \
pipeline.output.save_artifacts=true \
pipeline.output.save_viz=false \
pipeline.output.save_slam_map=false

2. 把.json转换成 ./npz
python /data1/lcy/projects/ReCamMaster/convert_pose_formats.py /data1/lcy/projects/ReCamMaster/eval_data2/cameras/camera_extrinsics.json -o /data1/lcy/projects/ReCamMaster/eval_data2/videos

3. 比较两组trajectory
python3 /data1/lcy/projects/ReCamMaster/evaluate_with_evo.py /data1/lcy/projects/ReCamMaster/eval_data2/videos /data1/lcy/projects/vipe/vipe_results/pose --work-dir ./evo_outputs --per-file | cat