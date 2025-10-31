#!/usr/bin/env python3
"""
CLI for comparing two trajectories in RayDiffusion.
"""

import argparse
import os

import numpy as np
import plotly.graph_objects as go

from trajectory_viz_utils import (
    add_compare_controls,
    add_origin_and_arcs,
    apply_zoom,
    configure_layout,
    create_trajectory_visualization,
    load_trajectory_data,
    overlay_p3d_frustums,
    anchor_trajectories_at_origin,
    apply_translation_offset,
    recompute_translations,
    scale_trajectories,
    trajectory_span,
    trajectory_centroid,
)


ARC_RADIUS = 3.0


def _dataset_label(path: str) -> str:
    norm = os.path.normpath(path)
    ext = os.path.splitext(norm)[1].lower()
    is_file_like = os.path.isfile(norm) or ext in {'.json', '.npz'}
    search_path = os.path.dirname(norm) if is_file_like else norm
    components = [comp for comp in search_path.split(os.sep) if comp]
    for comp in reversed(components):
        if 'data' in comp.lower():
            return comp.replace(' ', '_')
    if is_file_like:
        base = os.path.splitext(os.path.basename(norm))[0]
    else:
        base = components[-1] if components else os.path.basename(norm)
    return (base or 'input').replace(' ', '_')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize two trajectories side by side.')
    parser.add_argument(
        '--input', '-i', required=True,
        help='JSON file, NPZ file, or directory of NPZ files for Group A trajectories.'
    )
    parser.add_argument(
        '--input2', '-j', required=True,
        help='JSON file, NPZ file, or directory of NPZ files for Group B trajectories.'
    )
    parser.add_argument('--output-dir', default='trajectory_vis', help='Directory to save visualization outputs.')
    parser.add_argument('--title', type=str, default=None, help='Title shown on the Plotly figure.')
    parser.add_argument('--zoom-in', type=float, default=1.0, help='Zoom factor (>1 zooms in).')
    parser.add_argument('--image-scale', type=float, default=1.0, help='Scale factor for static image export.')
    parser.add_argument('--save-image', default=None, help='Optional path to also write a static image.')

    group_align = parser.add_mutually_exclusive_group()
    group_align.add_argument('--align-centers', dest='align_centers', action='store_true',
                             help='Align both trajectories by their centroids.')
    group_align.add_argument('--no-align-centers', dest='align_centers', action='store_false',
                             help='Keep original coordinates for each group.')
    parser.set_defaults(align_centers=True)

    parser.add_argument('--cam-every', type=int, default=8, help='Subsample interval for frustums.')
    parser.add_argument('--cam-scale', type=float, default=0.1, help='Scale factor for frustums.')

    parser.add_argument('--marker-size', type=int, default=3, help='Marker size for sampled points.')
    parser.add_argument('--mark-interval', type=int, default=1, help='Interval for placing markers along the path.')
    parser.add_argument('--cmap-groupA', type=str, default='Viridis', help='Colormap for Group A.')
    parser.add_argument('--cmap-groupB', type=str, default='Plasma', help='Colormap for Group B.')
    parser.add_argument('--translation-scale', type=float, default=5.0,
                        help='Uniform scale factor applied to translations before plotting (default: 5.0).')
    parser.add_argument('--match', action='store_true',
                        help='Align Group B to Group A per trajectory name (scale + orientation).')

    # Normalize translations of both groups jointly into [0, 1]
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument('--normalize-01', dest='normalize_01', action='store_true',
                            help='Normalize translations jointly to [0,1] across both groups.')
    norm_group.add_argument('--no-normalize-01', dest='normalize_01', action='store_false',
                            help='Do not normalize translations to [0,1].')
    parser.set_defaults(normalize_01=True)

    return parser.parse_args()


def _match_named_trajectories(traj_a, traj_b) -> None:
    shared = set(traj_a.keys()).intersection(traj_b.keys())
    for key in shared:
        entry_a = traj_a[key]
        entry_b = traj_b[key]
        frames_a = entry_a.get('frames')
        frames_b = entry_b.get('frames')
        rot_a = entry_a.get('Rcw')
        rot_b = entry_b.get('Rcw')
        if frames_a is None or frames_b is None or rot_a is None or rot_b is None:
            continue
        frame_map_b = {int(frame): idx for idx, frame in enumerate(frames_b)}
        paired = []
        for idx_a, frame in enumerate(frames_a):
            idx_b = frame_map_b.get(int(frame))
            if idx_b is not None:
                paired.append((idx_a, idx_b))
        if not paired:
            continue
        for idx_a, idx_b in paired:
            corr = rot_a[idx_a] @ rot_b[idx_b].T
            entry_b['pos'][idx_b] = (corr @ entry_b['pos'][idx_b])
            entry_b['Rcw'][idx_b] = corr @ entry_b['Rcw'][idx_b]
        entry_b['R_p3d'] = np.transpose(entry_b['Rcw'], (0, 2, 1))
        recompute_translations({key: entry_b})

        span_a = trajectory_span({key: entry_a})
        span_b = trajectory_span({key: entry_b})
        if span_a > 0.0 and span_b > 0.0:
            scale = span_a / span_b
            if not np.isclose(scale, 1.0):
                scale_trajectories({key: entry_b}, scale)


def main():
    args = parse_args()

    traj_a = load_trajectory_data(args.input)
    traj_b = load_trajectory_data(args.input2)

    if args.match:
        _match_named_trajectories(traj_a, traj_b)

    offset_a = trajectory_centroid(traj_a) if args.align_centers else np.zeros(3)
    offset_b = trajectory_centroid(traj_b) if args.align_centers else np.zeros(3)

    if args.normalize_01:
        def collect_positions(traj_dict, base_offset):
            pos_list = []
            for d in traj_dict.values():
                if d['pos'].size > 0:
                    base = d['pos'] - base_offset if args.align_centers else d['pos']
                    pos_list.append(base)
            return np.concatenate(pos_list, axis=0) if pos_list else np.zeros((0, 3))

        all_pos_a_for_norm = collect_positions(traj_a, offset_a)
        all_pos_b_for_norm = collect_positions(traj_b, offset_b)
        total = all_pos_a_for_norm.size + all_pos_b_for_norm.size
        all_pos = np.concatenate([all_pos_a_for_norm, all_pos_b_for_norm], axis=0) if total > 0 else np.zeros((0, 3))

        if all_pos.size > 0:
            min_xyz = all_pos.min(axis=0)
            max_xyz = all_pos.max(axis=0)
            span = np.clip(max_xyz - min_xyz, 1e-9, None)

            for d in traj_a.values():
                if d['pos'].size > 0:
                    base = d['pos'] - offset_a if args.align_centers else d['pos']
                    d['pos'] = (base - min_xyz) / span
            for d in traj_b.values():
                if d['pos'].size > 0:
                    base = d['pos'] - offset_b if args.align_centers else d['pos']
                    d['pos'] = (base - min_xyz) / span

            recompute_translations(traj_a)
            recompute_translations(traj_b)

        offset_a = np.zeros(3)
        offset_b = np.zeros(3)
    else:
        if args.align_centers:
            apply_translation_offset(traj_a, offset_a)
            apply_translation_offset(traj_b, offset_b)
        offset_a = np.zeros(3)
        offset_b = np.zeros(3)

    if args.translation_scale != 1.0:
        scale_trajectories(traj_a, args.translation_scale)
        scale_trajectories(traj_b, args.translation_scale)

    anchor_trajectories_at_origin(traj_a)
    anchor_trajectories_at_origin(traj_b)
    offset_a = np.zeros(3)
    offset_b = np.zeros(3)

    fig_a, all_pos_a = create_trajectory_visualization(
        traj_a,
        offset_a,
        args.marker_size,
        args.mark_interval,
        args.cmap_groupA,
        'Group A'
    )
    fig_b, all_pos_b = create_trajectory_visualization(
        traj_b,
        offset_b,
        args.marker_size,
        args.mark_interval,
        args.cmap_groupB,
        'Group B'
    )

    fig = go.Figure(data=fig_a.data + fig_b.data)

    overlay_p3d_frustums(fig, traj_a, args.cam_every, args.cam_scale, offset_a, args.cmap_groupA, 'Group A')
    overlay_p3d_frustums(fig, traj_b, args.cam_every, args.cam_scale, offset_b, args.cmap_groupB, 'Group B')

    add_origin_and_arcs(fig, ARC_RADIUS)

    configure_layout(fig, args.title or 'Trajectory Comparison')
    apply_zoom(fig, np.concatenate([all_pos_a, all_pos_b], axis=0), args.zoom_in)
    add_compare_controls(fig)

    os.makedirs(args.output_dir, exist_ok=True)
    base1 = _dataset_label(args.input)
    base2 = _dataset_label(args.input2)
    out_html = os.path.join(args.output_dir, f'compare_{base1}_vs_{base2}.html')
    fig.write_html(out_html)
    print(f"Visualization saved to: {out_html}")

    if args.save_image:
        try:
            fig.write_image(args.save_image, scale=args.image_scale, width=1920, height=1080)
            print(f"Saved static image to: {args.save_image}")
        except Exception as exc:
            print(f"Warning: Could not save image ({exc}). Install kaleido: pip install kaleido")


if __name__ == '__main__':
    main()
