#!/usr/bin/env python3
"""在 viser 中对比两组同名轨迹，视锥贴上对应视频帧。"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import viser
import viser.transforms as vtf

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


@dataclass
class TrajectoryHandles:
    frames: List[viser.FrameHandle]
    frustums: List[viser.CameraFrustumHandle]
    base_positions: np.ndarray


@dataclass
class PairHandles:
    a: TrajectoryHandles
    b: TrajectoryHandles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="加载两个相机目录 + 视频目录，按同名条目对比轨迹并贴图对应帧。"
    )
    parser.add_argument("--poses-a", type=Path, required=True, help="第一组相机 extrinsics 所在目录（npz）。")
    parser.add_argument("--poses-b", type=Path, required=True, help="第二组相机 extrinsics 所在目录（npz）。")
    parser.add_argument("--videos", type=Path, required=True, help="包含同名视频的目录。")
    parser.add_argument("--frame-step", type=int, default=4, help="下采样步长，默认每 4 帧采样一次。")
    parser.add_argument("--max-frames", type=int, default=None, help="单条轨迹最多可视化的帧数。")
    parser.add_argument("--image-short-edge", type=int, default=320, help="帧图下采样的短边长度。")
    parser.add_argument("--fov-deg", type=float, default=60.0, help="视锥 FOV（度）。")
    parser.add_argument("--frustum-scale", type=float, default=0.5, help="视锥大小。")
    parser.add_argument("--axes-length", type=float, default=0.15, help="坐标轴长度。")
    parser.add_argument("--axes-radius", type=float, default=0.005, help="坐标轴粗细。")
    parser.add_argument("--translation-scale", type=float, default=1.0, help="A 轨迹平移缩放初值。")
    parser.add_argument("--translation-scale-b", type=float, default=None, help="B 轨迹平移缩放初值，默认与 A 相同。")
    parser.add_argument("--reorder-axes-a", action="store_true", help="对 A 组姿态应用 viser_check 的轴重排。")
    parser.add_argument("--reorder-axes-b", action="store_true", help="对 B 组姿态应用 viser_check 的轴重排。")
    parser.add_argument("--transpose-poses-a", action="store_true", help="对 A 组姿态矩阵转置。")
    parser.add_argument("--transpose-poses-b", action="store_true", help="对 B 组姿态矩阵转置。")
    parser.add_argument("--sidebar-width", type=int, default=5200, help="viser 右侧控制栏宽度（像素）。")
    parser.add_argument("--host", default="127.0.0.1", help="viser host。")
    parser.add_argument("--port", type=int, default=8080, help="viser port。")
    return parser.parse_args()


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
    try:
        from matplotlib import cm as _mpl_cm
    except ImportError:
        _mpl_cm = None

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


def reorder_c2w_axes(c2w: np.ndarray) -> np.ndarray:
    """X_new=Y_old, Y_new=-Z_old, Z_new=X_old。"""
    # c2w = c2w[:, [1, 2, 0, 3]]
    # c2w = c2w[[1, 2, 0, 3], :]
    # c2w[:3, 1] *= -1.0
    # c2w[1, :3] *= -1.0
    # c2w[1, 3] *= -1.0
    # c2w[:3, 0] *= -1.0
    # c2w[0, :3] *= -1.0
    # c2w[0, 3] *= -1.0
    return c2w


def load_c2w_poses(pose_npz: Path, transpose: bool) -> Tuple[List[np.ndarray], List[int]]:
    data = np.load(pose_npz)
    pose_keys = ("data", "c2w", "poses", "extrinsics", "arr_0")
    pose_key = next((k for k in pose_keys if k in data), None)
    if pose_key is None:
        raise ValueError(f"{pose_npz} must contain one of {pose_keys} with c2w matrices.")

    c2w_mats = np.asarray(data[pose_key])
    if transpose:
        c2w_mats = c2w_mats.transpose(0, 2, 1)

    if c2w_mats.ndim != 3 or c2w_mats.shape[1] not in (3, 4):
        raise ValueError(f"Pose array must be (N, 3/4, 4); got {c2w_mats.shape} in {pose_npz}.")
    if c2w_mats.shape[1:] == (3, 4):
        mats_full = np.tile(np.eye(4, dtype=c2w_mats.dtype), (c2w_mats.shape[0], 1, 1))
        mats_full[:, :3, :4] = c2w_mats
        c2w_mats = mats_full
    if c2w_mats.shape[1:] != (4, 4):
        raise ValueError(f"Could not coerce poses to (N, 4, 4); got {c2w_mats.shape} in {pose_npz}.")

    index_keys = ("inds", "frame_ids", "frames", "ids", "indices")
    idx_key = next((k for k in index_keys if k in data), None)
    if idx_key is not None:
        inds_arr = np.asarray(data[idx_key]).astype(int)
        if inds_arr.shape[0] != c2w_mats.shape[0]:
            raise ValueError(
                f"Length mismatch: {pose_npz} has {c2w_mats.shape[0]} poses but {inds_arr.shape[0]} {idx_key}."
            )
        frame_indices = inds_arr.tolist()
    else:
        frame_indices = list(range(c2w_mats.shape[0]))

    poses = [c2w_mats[i] for i in range(c2w_mats.shape[0])]
    return poses, frame_indices


def relativize_to_first_pose(poses: List[np.ndarray]) -> List[np.ndarray]:
    if not poses:
        return poses
    base_inv = np.linalg.inv(poses[0])
    return [base_inv @ pose for pose in poses]


def translation_from_c2w(c2w: np.ndarray) -> np.ndarray:
    return c2w[:3, 3]


def c2w_to_pose(c2w: np.ndarray) -> vtf.SE3:
    rotation_world_from_cam = vtf.SO3.from_matrix(c2w[:3, :3])
    translation_world_from_cam = translation_from_c2w(c2w)
    return vtf.SE3.from_rotation_and_translation(rotation_world_from_cam, translation_world_from_cam)


def resize_frame_image(frame: np.ndarray, target_short_edge: int) -> Tuple[np.ndarray, float]:
    from PIL import Image

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
    rgb = np.asarray(rgb_img, dtype=np.uint8)
    return np.ascontiguousarray(rgb), scale


def get_video_reader(video_path: Path) -> imageio.Reader:
    try:
        reader = imageio.get_reader(str(video_path), "ffmpeg")
    except Exception as exc:
        raise RuntimeError(f"Failed to open video {video_path}. Ensure imageio[ffmpeg] is installed.") from exc
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


def load_video_frame(reader: imageio.Reader, frame_idx: int, target_short_edge: int) -> Tuple[np.ndarray, float]:
    frame = reader.get_data(frame_idx)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    frame = frame.astype(np.uint8)
    return resize_frame_image(frame, target_short_edge)


def list_files_by_stem(directory: Path, suffixes: Iterable[str]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    allow = {s.lower() for s in suffixes}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in allow:
            mapping[path.stem] = path
    return mapping


def select_frames(
    poses: List[np.ndarray],
    frame_indices: List[int],
    frame_step: int,
    max_frames: Optional[int],
) -> List[Tuple[np.ndarray, int]]:
    selected: List[Tuple[np.ndarray, int]] = []
    for idx, (pose_mat, frame_idx) in enumerate(zip(poses, frame_indices)):
        if idx % frame_step != 0:
            continue
        selected.append((pose_mat, frame_idx))
        if max_frames is not None and len(selected) >= max_frames:
            break
    return selected


def cache_frames(
    reader: imageio.Reader,
    frame_ids: List[int],
    target_short_edge: int,
) -> Dict[int, np.ndarray]:
    cache: Dict[int, np.ndarray] = {}
    for frame_idx in sorted(set(frame_ids)):
        try:
            img, _ = load_video_frame(reader, frame_idx, target_short_edge)
            cache[frame_idx] = img
        except Exception as exc:
            print(f"[warn] 读取视频帧 {frame_idx} 失败: {exc}")
    return cache


def set_visibility(traj: TrajectoryHandles, visible: bool) -> None:
    for frame in traj.frames:
        frame.visible = visible
    for frustum in traj.frustums:
        frustum.visible = visible


def apply_translation_scale(traj: TrajectoryHandles, scale: float) -> None:
    if traj.base_positions.size == 0:
        return
    scaled = traj.base_positions * float(scale)
    for frame, frustum, pos in zip(traj.frames, traj.frustums, scaled):
        frame.position = pos
        frustum.position = pos


def apply_frustum_scale(traj: TrajectoryHandles, scale: float, axes_length: float, axes_radius: float) -> None:
    for frame in traj.frames:
        frame.axes_length = axes_length
        frame.axes_radius = axes_radius
    for frustum in traj.frustums:
        frustum.scale = scale


def add_traj_to_scene(
    server: viser.ViserServer,
    name_prefix: str,
    selected_frames: List[Tuple[np.ndarray, int]],
    frame_images: Dict[int, np.ndarray],
    translation_scale: float,
    frustum_scale: float,
    axes_length: float,
    axes_radius: float,
    fov_rad: float,
    color_offset: float,
) -> TrajectoryHandles:
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []
    base_positions: List[np.ndarray] = []

    total = len(selected_frames)
    denom = max(1, total - 1)
    first_translation = (
        translation_from_c2w(selected_frames[0][0]) if selected_frames else np.zeros(3, dtype=float)
    )

    for order, (c2w_mat, frame_idx) in enumerate(selected_frames):
        pose = c2w_to_pose(c2w_mat)
        color_val = np.clip(color_offset + 0.45 * (order / denom), 0.0, 1.0)
        color = spectral_color(color_val)

        base_translation = translation_from_c2w(c2w_mat) - first_translation
        base_positions.append(base_translation)

        frame_name = f"/{name_prefix}/frame_{frame_idx:04d}"
        frame = server.scene.add_frame(
            frame_name,
            wxyz=pose.rotation().wxyz,
            position=base_translation * translation_scale,
            axes_length=axes_length,
            axes_radius=axes_radius,
        )
        frames.append(frame)

        img = frame_images.get(frame_idx)
        aspect = (img.shape[1] / img.shape[0]) if img is not None and img.shape[0] > 0 else 16.0 / 9.0
        frustum_kwargs = dict(
            name=f"{frame_name}/frustum",
            fov=fov_rad,
            aspect=aspect,
            scale=frustum_scale,
            color=color,
            position=base_translation * translation_scale,
            wxyz=pose.rotation().wxyz,
        )
        if img is not None:
            frustum_kwargs["image"] = img
        frustum = server.scene.add_camera_frustum(**frustum_kwargs)
        frustums.append(frustum)

        @frustum.on_click
        def _(_: viser.SceneNodePointerEvent, target_frame=frame) -> None:
            for client in server.get_clients().values():
                client.camera.wxyz = target_frame.wxyz
                client.camera.position = target_frame.position

    return TrajectoryHandles(
        frames=frames,
        frustums=frustums,
        base_positions=np.asarray(base_positions, dtype=float),
    )


def main() -> None:
    args = parse_args()
    if args.frame_step <= 0:
        raise ValueError("--frame-step 必须 >=1")
    if not args.poses_a.is_dir() or not args.poses_b.is_dir():
        raise FileNotFoundError("poses-a / poses-b 需要是包含 npz 的目录")
    if not args.videos.is_dir():
        raise FileNotFoundError("--videos 需要是包含视频的目录")

    pose_map_a = list_files_by_stem(args.poses_a, (".npz",))
    pose_map_b = list_files_by_stem(args.poses_b, (".npz",))
    video_map = list_files_by_stem(args.videos, _VIDEO_EXTS)

    shared_names = sorted(set(pose_map_a).intersection(pose_map_b).intersection(video_map))
    if not shared_names:
        raise ValueError("在两个相机目录和视频目录中没有找到同名条目。")

    scale_b0 = args.translation_scale_b if args.translation_scale_b is not None else args.translation_scale
    axes_radius = args.axes_radius
    axes_length = args.axes_length

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    style_css = (
        "<style>"
        f".viser-app .ant-layout-sider, "
        f".viser-app .ant-layout-sider-light, "
        f".viser-app .viser-sidebar, "
        f".viser-app .viser-controls, "
        f".viser-app .ant-layout-sider .ant-layout-sider-children "
        f"{{ width: {int(args.sidebar_width)}px !important;"
        f" max-width: {int(args.sidebar_width)}px !important;"
        f" min-width: {int(args.sidebar_width)}px !important;"
        f" flex: 0 0 {int(args.sidebar_width)}px !important; }}"
        "</style>"
    )
    server.gui.add_markdown(style_css)

    server.gui.add_markdown(
        "- 左侧卡片：全局显示/缩放控制；右侧栏：勾选渲染任意多条同名轨迹对\n"
        "- 点击视锥可跳转视角，轨迹起点均已对齐到原点"
    )

    server.gui.add_markdown("### 全局控制")
    gui_show_a = server.gui.add_checkbox("显示 A", initial_value=True)
    gui_show_b = server.gui.add_checkbox("显示 B", initial_value=True)

    def _safe_log10(val: float) -> float:
        return math.log10(max(val, 1e-9))

    gui_scale_a = server.gui.add_slider(
        "A translation log10", min=-2.0, max=2.0, step=0.01, initial_value=_safe_log10(float(args.translation_scale))
    )
    gui_scale_b = server.gui.add_slider(
        "B translation log10", min=-2.0, max=2.0, step=0.01, initial_value=_safe_log10(float(scale_b0))
    )
    gui_frustum_scale = server.gui.add_slider(
        "视锥大小", min=0.05, max=3.0, step=0.01, initial_value=float(args.frustum_scale)
    )

    pair_handles: Dict[str, PairHandles] = {}
    pair_controls: Dict[str, viser.CheckboxHandle] = {}

    btn_hide_a = server.gui.add_button("隐藏 pose1")
    btn_hide_b = server.gui.add_button("隐藏 pose2")

    def active_pairs() -> List[str]:
        return [n for n, ctrl in pair_controls.items() if ctrl.value]

    def _slider_to_scale(val: float) -> float:
        return 10.0 ** float(val)

    def apply_scales_for_active() -> None:
        for name in active_pairs():
            handles = pair_handles.get(name)
            if handles is None:
                continue
            apply_translation_scale(handles.a, _slider_to_scale(gui_scale_a.value))
            apply_translation_scale(handles.b, _slider_to_scale(gui_scale_b.value))

    def apply_frustum_scale_active() -> None:
        for name in active_pairs():
            handles = pair_handles.get(name)
            if handles is None:
                continue
            apply_frustum_scale(
                handles.a,
                gui_frustum_scale.value,
                axes_length=axes_length,
                axes_radius=axes_radius,
            )
            apply_frustum_scale(
                handles.b,
                gui_frustum_scale.value,
                axes_length=axes_length,
                axes_radius=axes_radius,
            )

    def set_pair_visibility(name: str, visible: bool) -> None:
        handles = pair_handles.get(name)
        if handles is None:
            return
        if visible:
            set_visibility(handles.a, gui_show_a.value)
            set_visibility(handles.b, gui_show_b.value)
        else:
            set_visibility(handles.a, False)
            set_visibility(handles.b, False)

    def load_pair(name: str) -> PairHandles:
        pose_path_a = pose_map_a[name]
        pose_path_b = pose_map_b[name]
        video_path = video_map[name]

        poses_a, frame_ids_a = load_c2w_poses(pose_path_a, args.transpose_poses_a)
        poses_b, frame_ids_b = load_c2w_poses(pose_path_b, args.transpose_poses_b)

        if args.reorder_axes_a:
            poses_a = [reorder_c2w_axes(mat) for mat in poses_a]
        if args.reorder_axes_b:
            poses_b = [reorder_c2w_axes(mat) for mat in poses_b]

        poses_a = relativize_to_first_pose(poses_a)
        poses_b = relativize_to_first_pose(poses_b)

        selected_a = select_frames(poses_a, frame_ids_a, args.frame_step, args.max_frames)
        selected_b = select_frames(poses_b, frame_ids_b, args.frame_step, args.max_frames)
        if not selected_a and not selected_b:
            raise ValueError(f"{name} 没有可视化的帧，检查 --frame-step / --max-frames")

        reader = get_video_reader(video_path)
        video_len = get_video_length(reader)
        max_needed = max([fid for _, fid in selected_a + selected_b]) if (selected_a or selected_b) else -1
        if video_len is not None and max_needed >= video_len:
            reader.close()
            raise ValueError(f"{video_path} 长度不足，最大需要帧 {max_needed}，视频仅 {video_len} 帧")
        frame_cache = cache_frames(
            reader,
            [fid for _, fid in selected_a] + [fid for _, fid in selected_b],
            args.image_short_edge,
        )
        reader.close()

        fov_rad = np.deg2rad(args.fov_deg)
        handles_a = add_traj_to_scene(
            server,
            name_prefix=f"pairs/{name}/A",
            selected_frames=selected_a,
            frame_images=frame_cache,
            translation_scale=_slider_to_scale(gui_scale_a.value),
            frustum_scale=gui_frustum_scale.value,
            axes_length=axes_length,
            axes_radius=axes_radius,
            fov_rad=fov_rad,
            color_offset=0.05,
        )
        handles_b = add_traj_to_scene(
            server,
            name_prefix=f"pairs/{name}/B",
            selected_frames=selected_b,
            frame_images=frame_cache,
            translation_scale=_slider_to_scale(gui_scale_b.value),
            frustum_scale=gui_frustum_scale.value,
            axes_length=axes_length,
            axes_radius=axes_radius,
            fov_rad=fov_rad,
            color_offset=0.55,
        )
        apply_frustum_scale(handles_a, gui_frustum_scale.value, axes_length, axes_radius)
        apply_frustum_scale(handles_b, gui_frustum_scale.value, axes_length, axes_radius)
        return PairHandles(a=handles_a, b=handles_b)

    def on_toggle(name: str) -> None:
        ctrl = pair_controls.get(name)
        if ctrl is None:
            return
        if ctrl.value:
            if name not in pair_handles:
                print(f"[info] 加载 {name}")
                pair_handles[name] = load_pair(name)
            set_pair_visibility(name, True)
            apply_frustum_scale_active()
            apply_scales_for_active()
        else:
            set_pair_visibility(name, False)

    server.gui.add_markdown("### 轨迹对渲染")
    for name in shared_names:
        checkbox = server.gui.add_checkbox(f"渲染 {name}", initial_value=False)
        pair_controls[name] = checkbox

        @checkbox.on_update
        def _(_: viser.GuiEvent, target_name=name) -> None:
            on_toggle(target_name)

    @gui_scale_a.on_update
    def _(_: viser.GuiEvent) -> None:
        apply_scales_for_active()

    @gui_scale_b.on_update
    def _(_: viser.GuiEvent) -> None:
        apply_scales_for_active()

    @gui_frustum_scale.on_update
    def _(_: viser.GuiEvent) -> None:
        apply_frustum_scale_active()

    @gui_show_a.on_update
    def _(_: viser.GuiEvent) -> None:
        for name in active_pairs():
            if name in pair_handles:
                set_visibility(pair_handles[name].a, gui_show_a.value)

    @gui_show_b.on_update
    def _(_: viser.GuiEvent) -> None:
        for name in active_pairs():
            if name in pair_handles:
                set_visibility(pair_handles[name].b, gui_show_b.value)

    @btn_hide_a.on_click
    def _(_: viser.GuiEvent) -> None:
        gui_show_a.value = False
        for name in active_pairs():
            if name in pair_handles:
                set_visibility(pair_handles[name].a, False)

    @btn_hide_b.on_click
    def _(_: viser.GuiEvent) -> None:
        gui_show_b.value = False
        for name in active_pairs():
            if name in pair_handles:
                set_visibility(pair_handles[name].b, False)

    url = f"http://{server.get_host()}:{server.get_port()}"
    print(f"viser 已启动：{url}")
    print(f"可用轨迹对：{', '.join(shared_names)}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping viser server.")


if __name__ == "__main__":
    main()
