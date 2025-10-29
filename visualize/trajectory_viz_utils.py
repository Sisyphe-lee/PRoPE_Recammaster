"""
Shared utilities for RayDiffusion trajectory visualizations.
"""

import json
import os
from typing import Dict, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.vis.plotly_vis import plot_scene


def _normalize_c2w_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Convert RayDiffusion-style camera extrinsics to the PyTorch3D convention.

    The source data encodes c2w transforms with axes ordered as (Z, X, Y) and
    translations in centimetres. We reorder axes to (X, Y, Z), flip the Y axis
    to match the expected handedness, and convert translations to metres.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {mat.shape}")

    converted = mat[:, [1, 2, 0, 3]].copy()
    converted[:3, 1] *= -1.0
    converted[:3, 3] /= 100.0
    return converted


def _recompute_T_from_positions(traj_entry: Dict[str, np.ndarray]) -> None:
    R = traj_entry.get('R_p3d')
    pos = traj_entry.get('pos')
    if R is None or pos is None or R.size == 0 or pos.size == 0:
        return
    traj_entry['T_p3d'] = -np.einsum('nij,nj->ni', R, pos)


def recompute_translations(trajectories: Dict) -> None:
    for traj in trajectories.values():
        _recompute_T_from_positions(traj)


def apply_translation_offset(trajectories: Dict, offset: np.ndarray) -> None:
    offset = np.asarray(offset, dtype=float)
    if offset.shape != (3,):
        raise ValueError(f"Expected offset shape (3,), got {offset.shape}")
    if np.allclose(offset, 0.0):
        return
    for traj in trajectories.values():
        pos = traj.get('pos')
        if pos is None or pos.size == 0:
            continue
        traj['pos'] = pos - offset
        _recompute_T_from_positions(traj)


def anchor_trajectories_at_origin(trajectories: Dict) -> None:
    for traj in trajectories.values():
        pos = traj.get('pos')
        if pos is None or pos.size == 0:
            continue
        start = pos[0]
        traj['pos'] = pos - start
        _recompute_T_from_positions(traj)


def parse_transformation_matrix(matrix_str: str) -> np.ndarray:
    parts = matrix_str.strip().split('] ')
    cols = []
    for chunk in parts:
        chunk = chunk.replace('[', '').replace(']', '').strip()
        if not chunk:
            continue
        values = [float(x) for x in chunk.split() if x]
        cols.append(values)
    mat = np.array(cols, dtype=float).T
    return _normalize_c2w_matrix(mat)


def extract_camera_pose(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Rcw = c2w[:3, :3]
    t_world = c2w[:3, 3]
    R_p3d = Rcw.T
    T_p3d = -R_p3d @ t_world
    pos = t_world.copy()
    return R_p3d, T_p3d, pos, Rcw


def load_trajectory_data(path: str) -> Dict:
    if os.path.isdir(path):
        return _load_npz_directory(path)

    ext = os.path.splitext(path)[1].lower()
    if ext == '.json':
        return _load_json_trajectories(path)
    if ext == '.npz':
        return _load_npz_trajectory(path)
    raise ValueError(f"Unsupported trajectory format: {path}")


def _load_json_trajectories(json_file: str) -> Dict:
    with open(json_file, 'r') as f:
        data = json.load(f)
    trajectories: Dict[str, Dict[str, np.ndarray]] = {}
    first_frame = list(data.keys())[0]
    camera_names = list(data[first_frame].keys())
    for cam_name in camera_names:
        trajectories[cam_name] = {'R_p3d': [], 'T_p3d': [], 'pos': [], 'Rcw': [], 'frames': []}
    for frame_name, frame_data in data.items():
        frame_idx = int(frame_name.replace('frame', ''))
        for cam_name, matrix_str in frame_data.items():
            c2w = parse_transformation_matrix(matrix_str)
            R_p3d, T_p3d, pos, Rcw = extract_camera_pose(c2w)
            trajectories[cam_name]['R_p3d'].append(R_p3d)
            trajectories[cam_name]['T_p3d'].append(T_p3d)
            trajectories[cam_name]['pos'].append(pos)
            trajectories[cam_name]['Rcw'].append(Rcw)
            trajectories[cam_name]['frames'].append(frame_idx)
    for cam_name in trajectories:
        for key in ['R_p3d', 'T_p3d', 'pos', 'Rcw', 'frames']:
            trajectories[cam_name][key] = np.array(trajectories[cam_name][key])
    return trajectories


def _load_npz_trajectory(npz_file: str) -> Dict:
    with np.load(npz_file) as data:
        matrices = data['data']
        frame_indices = data['inds'] if 'inds' in data else np.arange(len(matrices))

    traj_name = os.path.splitext(os.path.basename(npz_file))[0]
    trajectory = {'R_p3d': [], 'T_p3d': [], 'pos': [], 'Rcw': [], 'frames': []}

    for mat, frame_idx in zip(matrices, frame_indices):
        c2w = _normalize_c2w_matrix(mat)
        R_p3d, T_p3d, pos, Rcw = extract_camera_pose(c2w)
        trajectory['R_p3d'].append(R_p3d)
        trajectory['T_p3d'].append(T_p3d)
        trajectory['pos'].append(pos)
        trajectory['Rcw'].append(Rcw)
        trajectory['frames'].append(int(frame_idx))

    if not trajectory['R_p3d']:
        raise ValueError(f"No pose data found in {npz_file}")

    trajectory['R_p3d'] = np.stack(trajectory['R_p3d'], axis=0)
    trajectory['T_p3d'] = np.stack(trajectory['T_p3d'], axis=0)
    trajectory['pos'] = np.stack(trajectory['pos'], axis=0)
    trajectory['Rcw'] = np.stack(trajectory['Rcw'], axis=0)
    trajectory['frames'] = np.array(trajectory['frames'], dtype=int)

    return {traj_name: trajectory}


def _load_npz_directory(npz_dir: str) -> Dict:
    trajectories: Dict[str, Dict] = {}
    npz_files = sorted(
        [
            os.path.join(npz_dir, fname)
            for fname in os.listdir(npz_dir)
            if fname.lower().endswith('.npz')
        ]
    )

    if not npz_files:
        raise ValueError(f"No .npz files found in directory: {npz_dir}")

    for npz_path in npz_files:
        traj_dict = _load_npz_trajectory(npz_path)
        overlap = set(traj_dict.keys()).intersection(trajectories.keys())
        if overlap:
            raise ValueError(f"Duplicate trajectory names detected: {overlap}")
        trajectories.update(traj_dict)

    return trajectories


def trajectory_centroid(trajectories: Dict) -> np.ndarray:
    positions = [d['pos'] for d in trajectories.values() if d['pos'].size > 0]
    if not positions:
        return np.zeros(3)
    return np.concatenate(positions, axis=0).mean(axis=0)


def auto_scale_trajectories(
    trajectories: Dict,
    min_span: float = 0.5,
    target_span: float = 2.0,
) -> float:
    positions = [d['pos'] for d in trajectories.values() if d['pos'].size > 0]
    if not positions:
        return 1.0

    stacked = np.concatenate(positions, axis=0)
    span = float(np.max(np.max(stacked, axis=0) - np.min(stacked, axis=0)))
    if span >= min_span or span <= 0.0:
        return 1.0

    scale = target_span / span
    for traj in trajectories.values():
        traj['pos'] *= scale
        _recompute_T_from_positions(traj)
    return scale


def scale_trajectories(trajectories: Dict, scale: float) -> None:
    if scale <= 0.0:
        raise ValueError(f"Scale must be positive, got {scale}")
    if np.isclose(scale, 1.0):
        return
    for traj in trajectories.values():
        pos = traj.get('pos')
        if pos is None or pos.size == 0:
            continue
        traj['pos'] = pos * scale
        _recompute_T_from_positions(traj)


def trajectory_span(trajectories: Dict) -> float:
    positions = [d['pos'] for d in trajectories.values() if d['pos'].size > 0]
    if not positions:
        return 0.0
    stacked = np.concatenate(positions, axis=0)
    return float(np.max(np.max(stacked, axis=0) - np.min(stacked, axis=0)))


def principal_axes(trajectories: Dict) -> np.ndarray:
    positions = [d['pos'] for d in trajectories.values() if d['pos'].size > 0]
    if not positions:
        return np.eye(3)
    stacked = np.concatenate(positions, axis=0)
    if np.allclose(stacked, stacked[0]):
        return np.eye(3)
    cov = np.cov(stacked.T)
    U, _, _ = np.linalg.svd(cov)
    axes = U
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    return axes


def rotate_trajectories(trajectories: Dict, rotation: np.ndarray) -> None:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix of shape (3,3), got {rotation.shape}")
    for traj in trajectories.values():
        pos = traj.get('pos')
        if pos is not None and pos.size > 0:
            traj['pos'] = (rotation @ pos.T).T
        Rcw = traj.get('Rcw')
        if Rcw is not None and Rcw.size > 0:
            traj['Rcw'] = rotation @ Rcw
        Rp3d = traj.get('R_p3d')
        if Rp3d is not None and Rp3d.size > 0:
            traj['R_p3d'] = Rp3d @ rotation.T
        _recompute_T_from_positions(traj)


def add_origin_and_arcs(fig: go.Figure, radius: float = 5.0) -> None:
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0],
        mode='markers',
        marker=dict(size=4, color='#000000'),
        name='origin'
    ))
    t = np.linspace(0, 2 * np.pi, 181)
    r = radius
    fig.add_trace(go.Scatter3d(
        x=r * np.cos(t), y=r * np.sin(t), z=np.zeros_like(t),
        mode='lines',
        line=dict(color='#888888', width=2),
        name='arc_XY'
    ))
    fig.add_trace(go.Scatter3d(
        x=r * np.cos(t), y=np.zeros_like(t), z=r * np.sin(t),
        mode='lines',
        line=dict(color='#888888', width=2),
        name='arc_XZ'
    ))
    fig.add_trace(go.Scatter3d(
        x=np.zeros_like(t), y=r * np.cos(t), z=r * np.sin(t),
        mode='lines',
        line=dict(color='#888888', width=2),
        name='arc_YZ'
    ))


def create_trajectory_visualization(
    trajectories: Dict,
    positions_offset: np.ndarray,
    marker_size: int,
    mark_interval: int,
    cmap_name: str,
    name_prefix: str,
) -> Tuple[go.Figure, np.ndarray]:
    fig = go.Figure()
    all_positions_for_zoom = []
    cmap = plt.cm.get_cmap(cmap_name.lower())

    num_cams = len(trajectories)
    for idx, (cam_name, traj_data) in enumerate(trajectories.items()):
        positions = traj_data['pos'] - positions_offset
        if positions.size == 0:
            continue

        color_val = idx / max(1, num_cams - 1) if num_cams > 1 else 0.5
        color_hex = mcolors.to_hex(cmap(color_val))
        base_name = f'{name_prefix}-{cam_name}'

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='lines',
            line=dict(color=color_hex, width=3),
            name=f'{base_name}-path',
            showlegend=False
        ))

        if mark_interval > 0 and marker_size > 0:
            marker_idxs = np.arange(0, positions.shape[0], mark_interval)
            fig.add_trace(go.Scatter3d(
                x=positions[marker_idxs, 0],
                y=positions[marker_idxs, 1],
                z=positions[marker_idxs, 2],
                mode='markers',
                marker=dict(size=marker_size, color=color_hex),
                name=f'{base_name}-markers',
                showlegend=False
            ))

        all_positions_for_zoom.append(positions)
    all_positions_np = np.concatenate(all_positions_for_zoom, axis=0) if all_positions_for_zoom else np.zeros((0, 3))
    return fig, all_positions_np


def overlay_p3d_frustums(
    fig: go.Figure,
    trajectories: Dict,
    subsample_interval: int,
    camera_scale: float,
    offset: np.ndarray,
    cmap_name: str,
    name_prefix: str,
) -> None:
    cmap = plt.cm.get_cmap(cmap_name.lower())

    for cam_name, traj_data in trajectories.items():
        idxs = np.arange(0, traj_data['R_p3d'].shape[0], subsample_interval)
        if idxs.size == 0:
            continue
        R_p3d_sub = traj_data['R_p3d'][idxs]
        pos_sub = traj_data['pos'][idxs]
        frames_sub = traj_data['frames'][idxs]

        pos_offset = pos_sub - offset
        T_p3d_offset = -np.einsum('nji,nj->ni', R_p3d_sub, pos_offset)

        all_frames = traj_data['frames']
        frame_min, frame_max = all_frames.min(), all_frames.max()
        frame_range = max(1, frame_max - frame_min)

        for R_single, T_single, frame_idx in zip(R_p3d_sub, T_p3d_offset, frames_sub):
            norm_val = (frame_idx - frame_min) / frame_range
            color_hex = mcolors.to_hex(cmap(norm_val))

            R_tensor = torch.from_numpy(R_single[None]).float()
            T_tensor = torch.from_numpy(T_single[None]).float()
            single_camera = PerspectiveCameras(R=R_tensor, T=T_tensor, device='cpu')
            single_fig = plot_scene({"frustums": {f"{name_prefix}_{cam_name}_f{frame_idx}": single_camera}},
                                    camera_scale=camera_scale)

            if single_fig.data:
                trace = single_fig.data[0]
                trace.update(
                    line=dict(color=color_hex, width=1),
                    name=f'{name_prefix}-{cam_name}-f{frame_idx}',
                    legendgroup=f'{name_prefix}-{cam_name}',
                    showlegend=False
                )
                fig.add_trace(trace)


def configure_layout(fig: go.Figure, title: str) -> None:
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'font': {'size': 20}},
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.8),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0)
            )
        ),
        width=1920,
        height=1080,
        showlegend=False,
        margin=dict(l=200, r=50, t=80, b=50)
    )


def apply_zoom(fig: go.Figure, all_positions: np.ndarray, zoom_in: float) -> None:
    if all_positions.size == 0 or zoom_in <= 1.0:
        return
    min_pos, max_pos = all_positions.min(axis=0), all_positions.max(axis=0)
    center = 0.5 * (min_pos + max_pos)
    span = (max_pos - min_pos) / zoom_in
    fig.update_layout(scene=dict(
        xaxis=dict(range=[center[0] - 0.5 * span[0], center[0] + 0.5 * span[0]]),
        yaxis=dict(range=[center[1] - 0.5 * span[1], center[1] + 0.5 * span[1]]),
        zaxis=dict(range=[center[2] - 0.5 * span[2], center[2] + 0.5 * span[2]])
    ))


def add_single_controls(fig: go.Figure) -> None:
    buttons = [
        dict(
            label="Toggle All",
            method="restyle",
            args=[{"visible": "legendonly"}],
            args2=[{"visible": True}]
        )
    ]

    cam_names = set()
    for trace in fig.data:
        name = getattr(trace, 'name', '')
        if name.startswith('cams-'):
            parts = name.split('-')
            if len(parts) >= 2:
                cam_names.add(parts[1])

    for cam_name in sorted(cam_names):
        indices = [i for i, trace in enumerate(fig.data) if getattr(trace, 'name', '').startswith(f'cams-{cam_name}-')]
        buttons.append(dict(
            label=f"Toggle {cam_name}",
            method="restyle",
            args=[{"visible": "legendonly"}, indices],
            args2=[{"visible": True}, indices]
        ))

    fig.update_layout(updatemenus=[dict(
        type="buttons",
        direction="down",
        x=0.02,
        y=0.98,
        showactive=True,
        buttons=buttons,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.3)",
        font=dict(size=11)
    )])


def add_compare_controls(fig: go.Figure) -> None:
    buttons = [
        dict(
            label="Toggle All",
            method="restyle",
            args=[{"visible": "legendonly"}],
            args2=[{"visible": True}]
        ),
        dict(
            label="Toggle Group A",
            method="restyle",
            args=[{"visible": "legendonly"}, [i for i, t in enumerate(fig.data) if "Group A" in getattr(t, 'name', '')]],
            args2=[{"visible": True}, [i for i, t in enumerate(fig.data) if "Group A" in getattr(t, 'name', '')]]
        ),
        dict(
            label="Toggle Group B",
            method="restyle",
            args=[{"visible": "legendonly"}, [i for i, t in enumerate(fig.data) if "Group B" in getattr(t, 'name', '')]],
            args2=[{"visible": True}, [i for i, t in enumerate(fig.data) if "Group B" in getattr(t, 'name', '')]]
        )
    ]

    cam_names = set()
    for trace in fig.data:
        name = getattr(trace, 'name', '')
        if '-' in name:
            parts = name.split('-')
            if len(parts) >= 2 and parts[0] in {'Group A', 'Group B'}:
                cam_names.add(parts[1])

    for cam_name in sorted(cam_names):
        indices = [i for i, trace in enumerate(fig.data) if f'-{cam_name}-' in getattr(trace, 'name', '')]
        buttons.append(dict(
            label=f"Toggle {cam_name}",
            method="restyle",
            args=[{"visible": "legendonly"}, indices],
            args2=[{"visible": True}, indices]
        ))

    fig.update_layout(updatemenus=[dict(
        type="buttons",
        direction="down",
        x=0.02,
        y=0.98,
        showactive=True,
        buttons=buttons,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.3)",
        font=dict(size=11)
    )])
