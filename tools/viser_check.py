#!/usr/bin/env python3

"""Gen3R + SpaTracker-style teaser visualizer built on top of viser.
Example usage:
    python tools/viser_check.py \
    --pose /nas/datasets/relestate10k/train/0a013a7ad5bdecd7/extrinsics.npz \
    --pose-b debug_coord/scene3295/cam07.npz \
    --video-path /nas/datasets/relestate10k/train/0a013a7ad5bdecd7/video.mp4 \
    --video-path-b /nas/datasets/MultiCamVideo-Dataset/MultiCamVideo-Dataset/train/f18_aperture10/scene3295/videos/cam07.mp4 \
    --translation-scale 10.0 --translation-scale-b 0.05 \
    --reorder-axes-b --transpose-pose-b

    python tools/viser_check.py \
    --pose evaluation/example_eval/20251103_165044/pose/1_cam05.npz \
    --video-path /data1/lcy/projects/ReCamMaster/evaluation/example_eval/20251103_165044/1_cam05.mp4 \
    --translation-scale 10 \
    --reorder-axes-a 

    python tools/viser_check.py --pose /nas/datasets/relestate10k/train/0a1dd3e1f6524f08/extrinsics.npz \
        --video-path /nas/datasets/relestate10k/train/0a1dd3e1f6524f08/video.mp4 --translation-scale 10 
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import trimesh

import viser
import viser.transforms as vtf

try:
    from matplotlib import cm as _mpl_cm
except ImportError:  # pragma: no cover - optional dependency
    _mpl_cm = None


_FALLBACK_SPECTRAL: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.0, (158, 1, 66)),
    (0.1, (213, 62, 79)),
    (0.2, (244, 109, 67)),
    (0.3, (253, 174, 97)),
    (0.4, (254, 224, 139)),
    (0.5, (255, 255, 191)),
    (0.6, (230, 245, 152)),
    (0.7, (171, 221, 164)),
    (0.8, (102, 194, 165)),
    (0.9, (50, 136, 189)),
    (1.0, (94, 79, 162)),
]


def spectral_color(value: float) -> Tuple[int, int, int]:
    """Map [0, 1] -> RGB using matplotlib's Spectral colormap or a fallback."""
    value = float(np.clip(value, 0.0, 1.0))
    if _mpl_cm is not None:
        rgb = _mpl_cm.get_cmap("Spectral_r")(value)[:3]
        return tuple(int(round(255.0 * c)) for c in rgb)

    for (p0, c0), (p1, c1) in zip(_FALLBACK_SPECTRAL[:-1], _FALLBACK_SPECTRAL[1:]):
        if value <= p1:
            denom = (p1 - p0) if p1 > p0 else 1.0
            local_t = (value - p0) / denom
            return tuple(
                int(round(c0[channel] + (c1[channel] - c0[channel]) * local_t))
                for channel in range(3)
            )
    return _FALLBACK_SPECTRAL[-1][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Gen3R point clouds with SpaTracker-style camera poses and images."
        )
    )
    parser.add_argument(
        "--pose",
        type=Path,
        required=True,
        help="Path to .npz pose file containing c2w matrices (key: data) and frame indices (key: inds).",
    )
    parser.add_argument(
        "--pose-b",
        type=Path,
        default=None,
        help="Optional second .npz pose file for side-by-side visualization.",
    )
    parser.add_argument(
        "--transpose-pose",
        action="store_true",
        help="Transpose the primary pose matrices before visualization.",
    )
    parser.add_argument(
        "--transpose-pose-b",
        action="store_true",
        help="Transpose the secondary pose matrices before visualization.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to .mp4 video whose frames align with inds stored in the primary pose file.",
    )
    parser.add_argument(
        "--video-path-b",
        type=Path,
        default=None,
        help="Optional .mp4 video aligned with the secondary pose file; required when --pose-b is set.",
    )
    parser.add_argument(
        "--pcd-path",
        type=Path,
        default=None,
        help=(
            "Optional PLY point cloud path; omit to skip loading point clouds."
        ),
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=0.10,
        help=(
            "Scale multiplier applied to camera translations (and point clouds) "
            "after recentering the trajectory to the first pose."
        ),
    )
    parser.add_argument(
        "--translation-scale-b",
        type=float,
        default=None,
        help="Optional scale for the second trajectory; defaults to --translation-scale.",
    )
    parser.add_argument(
        "--reorder-axes-a",
        action="store_true",
        help=(
            "Apply axis mapping X_new=Y_old, Y_new=-Z_old, Z_new=X_old to primary poses "
            "before visualization."
        ),
    )
    parser.add_argument(
        "--reorder-axes-b",
        action="store_true",
        help=(
            "Apply axis mapping X_new=Y_old, Y_new=-Z_old, Z_new=X_old to secondary poses "
            "before visualization."
        ),
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=60.0,
        help="Field of view in degrees used for camera frustums (intrinsics absent in npz).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=4,
        help="Sample every N-th frame when visualizing camera poses.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard cap on the number of frames after downsampling.",
    )
    parser.add_argument(
        "--image-short-edge",
        type=int,
        default=320,
        help="Downsample images so their short edge matches this size (<=0 disables).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=-1,
        help="Randomly subsample the point cloud to this many points (<=0 keeps all).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.05,
        help="Point size (meters) for the point cloud visualization.",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=0.5,
        help="Scale factor passed to viser.scene.add_camera_frustum (0.15 mirrors docs).",
    )
    parser.add_argument(
        "--axes-length",
        type=float,
        default=0.15,
        help="Length of the camera frame axes in meters.",
    )
    parser.add_argument(
        "--axes-radius",
        type=float,
        default=0.005,
        help="Radius of the camera frame axes in meters.",
    )
    parser.add_argument(
        "--scene-scale",
        type=float,
        default=1.0,
        help="Global scale multiplier applied to both points and camera translations.",
    )
    parser.add_argument(
        "--camera-pos-scale",
        type=float,
        default=1.0,
        help="Additional multiplier applied only to camera translations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for point cloud subsampling.",
    )
    return parser.parse_args()


def load_c2w_poses(
    pose_npz: Path,
    t:bool
) -> Tuple[List[np.ndarray], List[int]]:
    data = np.load(pose_npz)
    pose_keys = ("data", "c2w", "poses", "extrinsics", "arr_0")
    pose_key = next((k for k in pose_keys if k in data), None)
    if pose_key is None:
        raise ValueError(
            f"{pose_npz} must contain one of {pose_keys} with c2w matrices."
        )

    c2w_mats = np.asarray(data[pose_key])
    if(t):
        c2w_mats = c2w_mats.transpose(0,2,1)

    if c2w_mats.ndim != 3 or c2w_mats.shape[1] not in (3, 4):
        raise ValueError(
            f"Pose array must be (N, 3/4, 4); got {c2w_mats.shape} in {pose_npz}."
        )
    if c2w_mats.shape[1:] == (3, 4):
        mats_full = np.tile(np.eye(4, dtype=c2w_mats.dtype), (c2w_mats.shape[0], 1, 1))
        mats_full[:, :3, :4] = c2w_mats
        c2w_mats = mats_full
    if c2w_mats.shape[1:] != (4, 4):
        raise ValueError(
            f"Could not coerce poses to (N, 4, 4); got {c2w_mats.shape} in {pose_npz}."
        )

    index_keys = ("inds", "frame_ids", "frames", "ids", "indices")
    idx_key = next((k for k in index_keys if k in data), None)
    if idx_key is not None:
        inds_arr = np.asarray(data[idx_key]).astype(int)
        if inds_arr.shape[0] != c2w_mats.shape[0]:
            raise ValueError(
                f"Length mismatch: {pose_npz} has {c2w_mats.shape[0]} poses but "
                f"{inds_arr.shape[0]} {idx_key}."
            )
        frame_indices = inds_arr.tolist()
    else:
        frame_indices = list(range(c2w_mats.shape[0]))

    poses = [c2w_mats[i] for i in range(c2w_mats.shape[0])]
    return poses, frame_indices


def _geometry_to_arrays(
    geometry: trimesh.base.Trimesh,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(geometry, trimesh.Scene):
        vertices: List[np.ndarray] = []
        colors: List[np.ndarray] = []
        for mesh in geometry.geometry.values():
            pts, cols = _geometry_to_arrays(mesh)
            vertices.append(pts)
            colors.append(cols)
        return np.concatenate(vertices, axis=0), np.concatenate(colors, axis=0)

    if isinstance(geometry, trimesh.Trimesh):
        pts = np.asarray(geometry.vertices)
        if (
            geometry.visual is not None
            and geometry.visual.vertex_colors is not None
            and len(geometry.visual.vertex_colors) == len(pts)
        ):
            cols = np.asarray(geometry.visual.vertex_colors)[:, :3]
        else:
            cols = np.full((len(pts), 3), 255, dtype=np.uint8)
        return pts, cols

    if isinstance(geometry, trimesh.points.PointCloud):
        pts = np.asarray(geometry.vertices)
        cols = (
            np.asarray(geometry.colors)[:, :3]
            if geometry.colors is not None
            else np.full((len(pts), 3), 255, dtype=np.uint8)
        )
        return pts, cols

    raise TypeError(f"Unsupported geometry type: {type(geometry)}")


def load_point_cloud(
    pcd_path: Path,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    geometry = trimesh.load(pcd_path, process=False)
    points, colors = _geometry_to_arrays(geometry)
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(points), size=max_points, replace=False)
        points = points[indices]
        colors = colors[indices]
    return points.astype(np.float32), colors.astype(np.uint8)


def resize_frame_image(
    frame: np.ndarray,
    target_short_edge: int,
) -> Tuple[np.ndarray, float]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - convenience error path.
        raise ImportError(
            "This script depends on Pillow. Install it with `pip install pillow`."
        ) from exc

    rgb_img = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    width, height = rgb_img.size
    scale = 1.0
    if target_short_edge > 0:
        short_edge = min(height, width)
        if short_edge > target_short_edge:
            scale = target_short_edge / short_edge
            resampling = getattr(Image, "Resampling", Image)
            resample_filter = getattr(resampling, "LANCZOS", Image.BICUBIC)
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            rgb_img = rgb_img.resize(new_size, resample=resample_filter)
            width, height = rgb_img.size

    rgb = np.asarray(rgb_img, dtype=np.uint8)
    return np.ascontiguousarray(rgb), scale


def c2w_to_pose(c2w: np.ndarray) -> vtf.SE3:
    rotation_world_from_cam = vtf.SO3.from_matrix(c2w[:3, :3])
    translation_world_from_cam = translation_from_c2w(c2w)
    return vtf.SE3.from_rotation_and_translation(
        rotation_world_from_cam, translation_world_from_cam
    )


def translation_from_c2w(c2w: np.ndarray) -> np.ndarray:
    translation_world_from_cam = c2w[:3, 3]
    return translation_world_from_cam


def reorder_c2w_axes(c2w: np.ndarray) -> np.ndarray:
    """Remap axes: X_new=Y_old, Y_new=Z_old, Z_new=X_old; keep translation aligned."""
    c2w = c2w[:, [1, 2, 0, 3]]
    c2w = c2w[ [1, 2, 0, 3], :]
    c2w[:3, 1] *= -1.0
    c2w[1, :3] *= -1.0
    c2w[1, 3] *= -1.0
    return c2w


def relativize_to_first_pose(poses: List[np.ndarray]) -> List[np.ndarray]:
    """Left-multiply by the inverse of the first pose so the first frame becomes identity."""
    if not poses:
        return poses
    base_inv = np.linalg.inv(poses[0])
    return [base_inv @ pose for pose in poses]


def get_video_reader(video_path: Path) -> imageio.Reader:
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open video {video_path}. Ensure imageio[ffmpeg] is installed."
        ) from exc
    return reader


def get_video_length(reader: imageio.Reader) -> Optional[int]:
    for attr in ("count_frames", "__len__"):
        func = getattr(reader, attr, None)
        if func is None:
            continue
        try:
            return int(func())
        except Exception:
            continue
    return None


def load_video_frame(
    reader: imageio.Reader,
    frame_idx: int,
    target_short_edge: int,
) -> Tuple[np.ndarray, float]:
    frame = reader.get_data(frame_idx)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    frame = frame.astype(np.uint8)
    return resize_frame_image(frame, target_short_edge)


def main() -> None:
    args = parse_args()
    if False:
        print("Debug mode is enabled.")
        import debugpy  # type: ignore

        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        print("Attached, continue...")

    cameras_path = args.pose
    cameras_path_b = args.pose_b
    video_path = args.video_path
    video_path_b = args.video_path_b
    pcd_path: Optional[Path] = (
        None if args.pcd_path is None or str(args.pcd_path).strip() == "" else args.pcd_path
    )

    if args.frame_step <= 0:
        raise ValueError("--frame-step must be >= 1.")
    if not cameras_path.exists():
        raise FileNotFoundError(cameras_path)
    if not video_path.exists() or not video_path.is_file():
        raise FileNotFoundError(
            f"--video-path must be an .mp4 file: {video_path}"
        )
    if pcd_path is not None and not pcd_path.exists():
        raise FileNotFoundError(pcd_path)
    if cameras_path_b is not None and video_path_b is None:
        raise ValueError("--pose-b provided but --video-path-b is missing.")
    if video_path_b is not None and not video_path_b.exists():
        raise FileNotFoundError(video_path_b)
    if video_path_b is not None and not video_path_b.is_file():
        raise FileNotFoundError(f"--video-path-b must be an .mp4 file: {video_path_b}")

    poses_c2w, frame_indices = load_c2w_poses(cameras_path, args.transpose_pose)
    if args.reorder_axes_a:
        poses_c2w = [reorder_c2w_axes(mat) for mat in poses_c2w]
        poses_c2w = relativize_to_first_pose(poses_c2w)
    poses_c2w_b: Optional[List[np.ndarray]] = None
    frame_indices_b: Optional[List[int]] = None
    if cameras_path_b is not None:
        if not cameras_path_b.exists():
            raise FileNotFoundError(cameras_path_b)
        poses_c2w_b, frame_indices_b = load_c2w_poses(cameras_path_b, args.transpose_pose_b)
        if args.reorder_axes_b:
            poses_c2w_b = [reorder_c2w_axes(mat) for mat in poses_c2w_b]
            poses_c2w_b = relativize_to_first_pose(poses_c2w_b)

    translations = [translation_from_c2w(mat) for mat in poses_c2w]
    origin_translation = translations[0] if translations else np.zeros(3, dtype=np.float32)
    translation_offsets = [t - origin_translation for t in translations]
    base_scale = float(args.translation_scale)

    translations_b: Optional[List[np.ndarray]] = None
    origin_translation_b: Optional[np.ndarray] = None
    translation_offsets_b: Optional[List[np.ndarray]] = None
    base_scale_b: float = float(
        args.translation_scale_b if args.translation_scale_b is not None else args.translation_scale
    )
    if poses_c2w_b is not None:
        translations_b = [translation_from_c2w(mat) for mat in poses_c2w_b]
        origin_translation_b = translations_b[0] if translations_b else np.zeros(3, dtype=np.float32)
        translation_offsets_b = [t - origin_translation_b for t in translations_b]

    points_raw: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    if pcd_path is not None:
        points_raw, colors = load_point_cloud(pcd_path, args.max_points, args.seed)
        points_raw = (points_raw - origin_translation) * base_scale

    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.gui.add_markdown(
        "### Gen3R teaser\n"
        "* Click any camera frustum to teleport the viewer.\n"
        "* Adjust sampling via --frame-step/--max-points.\n"
        f"* Translations are recentered to the first pose and scaled by {base_scale:.4g}.\n"
    )

    gui_scene_scale = server.gui.add_slider(
        "Scene scale",
        min=0.01,
        max=30.0,
        step=0.1,
        initial_value=args.scene_scale,
    )
    gui_camera_pos_scale = server.gui.add_slider(
        "Camera pos scale",
        min=0.1,
        max=10.0,
        step=0.1,
        initial_value=args.camera_pos_scale,
    )
    gui_point_size = server.gui.add_slider(
        "Point size",
        min=0.002,
        max=1.0,
        step=0.01,
        initial_value=args.point_size,
    )
    gui_camera_scale = server.gui.add_slider(
        "Camera scale",
        min=0.1,
        max=3,
        step=0.005,
        initial_value=args.frustum_scale,
    )
    gui_show_pose_a = server.gui.add_checkbox(
        "Show pose A",
        initial_value=True,
    )
    gui_show_pose_b = server.gui.add_checkbox(
        "Show pose B",
        initial_value=True,
    )

    point_cloud = (
        server.scene.add_point_cloud(
            name="/gen3r/pcd",
            points=points_raw * gui_scene_scale.value,
            colors=colors,
            point_size=gui_point_size.value,
            point_shape="rounded",
        )
        if points_raw is not None
        else None
    )

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []
    frame_positions: List[np.ndarray] = []
    axes_radius_ratio = (
        args.axes_radius / args.axes_length if args.axes_length > 0 else 0.0333
    )

    def set_pose_visibility(
        target_frames: List[viser.FrameHandle],
        target_frustums: List[viser.CameraFrustumHandle],
        visible: bool,
    ) -> None:
        for frame in target_frames:
            frame.visible = visible
        for frustum in target_frustums:
            frustum.visible = visible

    def update_positions() -> None:
        scene_scale = gui_scene_scale.value
        cam_scale = gui_camera_pos_scale.value
        with server.atomic():
            if point_cloud is not None and points_raw is not None:
                point_cloud.points = points_raw * scene_scale
            for frame, base_pos in zip(frames, frame_positions):
                frame.position = base_pos * scene_scale * cam_scale

    @gui_point_size.on_update
    def _(_) -> None:
        if point_cloud is not None:
            point_cloud.point_size = gui_point_size.value

    @gui_scene_scale.on_update
    def _(_) -> None:
        update_positions()

    @gui_camera_pos_scale.on_update
    def _(_) -> None:
        update_positions()

    @gui_camera_scale.on_update
    def _(_) -> None:
        radius = gui_camera_scale.value * axes_radius_ratio
        for frame in frames:
            frame.axes_length = gui_camera_scale.value
            frame.axes_radius = radius
        for frustum in frustums:
            frustum.scale = gui_camera_scale.value

    if not poses_c2w:
        raise ValueError(f"No poses found in {cameras_path}.")
    if poses_c2w_b is not None and not poses_c2w_b:
        raise ValueError(f"No poses found in {cameras_path_b}.")

    video_reader_a = get_video_reader(video_path)
    video_length_a = get_video_length(video_reader_a)
    if video_length_a is not None and max(frame_indices) >= video_length_a:
        raise ValueError(
            f"Pose indices exceed video length: max ind {max(frame_indices)} vs {video_length_a} frames."
        )
    video_reader_b: Optional[imageio.Reader] = None
    video_length_b: Optional[int] = None
    if poses_c2w_b is not None and frame_indices_b is not None:
        if video_path_b is None:
            raise ValueError("--pose-b provided but no video for secondary stream.")
        video_reader_b = get_video_reader(video_path_b)
        video_length_b = get_video_length(video_reader_b)
        if video_length_b is not None and max(frame_indices_b) >= video_length_b:
            raise ValueError(
                f"Pose-B indices exceed video length: max ind {max(frame_indices_b)} vs {video_length_b} frames."
            )

    selected_frames: List[Tuple[int, np.ndarray, np.ndarray]] = []
    for idx, (pose_mat, frame_idx) in enumerate(zip(poses_c2w, frame_indices)):
        if idx % args.frame_step != 0:
            continue
        selected_frames.append((frame_idx, pose_mat, translation_offsets[idx]))
        if args.max_frames is not None and len(selected_frames) >= args.max_frames:
            break

    if not selected_frames:
        raise ValueError("No frames selected. Try reducing --frame-step or max filters.")

    total_selected = len(selected_frames)
    denom = max(1, total_selected - 1)

    for order, (frame_idx, c2w_mat, trans_offset) in enumerate(selected_frames):
        color = spectral_color(order / denom)
        pose = c2w_to_pose(c2w_mat)
        rotation = pose.rotation()
        frame_name = f"/gen3r/frame_{frame_idx:04d}"
        base_translation = trans_offset * base_scale
        frame = server.scene.add_frame(
            frame_name,
            wxyz=rotation.wxyz,
            position=base_translation
            * gui_scene_scale.value
            * gui_camera_pos_scale.value,
            axes_length=gui_camera_scale.value,
            axes_radius=gui_camera_scale.value * axes_radius_ratio,
            # origin_color=color,
        )
        frames.append(frame)
        frame_positions.append(base_translation)

        image, _ = load_video_frame(video_reader_a, frame_idx, args.image_short_edge)
        height, width = image.shape[:2]
        fov = np.deg2rad(args.fov_deg)
        aspect = width / height

        frustum = server.scene.add_camera_frustum(
            f"{frame_name}/frustum",
            fov=fov,
            aspect=aspect,
            scale=gui_camera_scale.value,
            image=image,
            color=color,
        )
        frustums.append(frustum)

        @frustum.on_click
        def _(event: viser.SceneNodePointerEvent, target_frame=frame) -> None:
            for client in server.get_clients().values():
                client.camera.wxyz = target_frame.wxyz
                client.camera.position = target_frame.position

    # Secondary trajectory rendering (optional).
    frames_b: List[viser.FrameHandle] = []
    frustums_b: List[viser.CameraFrustumHandle] = []
    frame_positions_b: List[np.ndarray] = []

    if poses_c2w_b is not None and frame_indices_b is not None and translation_offsets_b is not None:
        selected_frames_b: List[Tuple[int, np.ndarray, np.ndarray]] = []
        for idx, (pose_mat, frame_idx) in enumerate(zip(poses_c2w_b, frame_indices_b)):
            if idx % args.frame_step != 0:
                continue
            selected_frames_b.append((frame_idx, pose_mat, translation_offsets_b[idx]))
            if args.max_frames is not None and len(selected_frames_b) >= args.max_frames:
                break

        denom_b = max(1, len(selected_frames_b) - 1) if selected_frames_b else 1
        for order, (frame_idx, c2w_mat, trans_offset) in enumerate(selected_frames_b):
            color = spectral_color(0.5 * (order / denom_b) + 0.25)
            pose = c2w_to_pose(c2w_mat)
            rotation = pose.rotation()
            frame_name = f"/gen3r_b/frame_{frame_idx:04d}"
            base_translation = trans_offset * base_scale_b
            frame = server.scene.add_frame(
                frame_name,
                wxyz=rotation.wxyz,
                position=base_translation
                * gui_scene_scale.value
                * gui_camera_pos_scale.value,
                axes_length=gui_camera_scale.value,
                axes_radius=gui_camera_scale.value * axes_radius_ratio,
            )
            frames_b.append(frame)
            frame_positions_b.append(base_translation)

            if video_reader_b is None:
                raise ValueError("Secondary video reader is not initialized.")
            image, _ = load_video_frame(video_reader_b, frame_idx, args.image_short_edge)
            height, width = image.shape[:2]
            fov = np.deg2rad(args.fov_deg)
            aspect = width / height

            frustum = server.scene.add_camera_frustum(
                f"{frame_name}/frustum",
                fov=fov,
                aspect=aspect,
                scale=gui_camera_scale.value,
                image=image,
                color=color,
            )
            frustums_b.append(frustum)

            @frustum.on_click
            def _(event: viser.SceneNodePointerEvent, target_frame=frame) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = target_frame.wxyz
                    client.camera.position = target_frame.position

    def update_positions_all() -> None:
        update_positions()
        scene_scale = gui_scene_scale.value
        cam_scale = gui_camera_pos_scale.value
        with server.atomic():
            for frame, base_pos in zip(frames_b, frame_positions_b):
                frame.position = base_pos * scene_scale * cam_scale

    @gui_scene_scale.on_update
    def _(_) -> None:
        update_positions_all()

    @gui_camera_pos_scale.on_update
    def _(_) -> None:
        update_positions_all()

    @gui_show_pose_a.on_update
    def _(_) -> None:
        set_pose_visibility(frames, frustums, gui_show_pose_a.value)

    @gui_show_pose_b.on_update
    def _(_) -> None:
        set_pose_visibility(frames_b, frustums_b, gui_show_pose_b.value)

    update_positions_all()

    url = f"http://{server.get_host()}:{server.get_port()}"
    print(f"Viser server is live at {url}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viser server.")


if __name__ == "__main__":
    main()
