#!/usr/bin/env python3
"""
CLI for visualizing a single trajectory or trajectory set in RayDiffusion.
"""

import argparse
import os

import numpy as np
import plotly.graph_objects as go

from trajectory_viz_utils import (
    add_origin_and_arcs,
    add_single_controls,
    anchor_trajectories_at_origin,
    apply_zoom,
    configure_layout,
    create_trajectory_visualization,
    load_trajectory_data,
    overlay_p3d_frustums,
    apply_translation_offset,
    recompute_translations,
    scale_trajectories,
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
    parser = argparse.ArgumentParser(description='Visualize a single trajectory dataset.')
    parser.add_argument(
        '--input', '-i', required=True,
        help='JSON file, NPZ file, or directory of NPZ files containing trajectory poses.'
    )
    parser.add_argument('--output-dir', default='trajectory_vis', help='Directory to save visualization outputs.')
    parser.add_argument('--title', type=str, default=None, help='Title shown on the Plotly figure.')
    parser.add_argument('--zoom-in', type=float, default=1.0, help='Zoom factor (>1 zooms in).')
    parser.add_argument('--image-scale', type=float, default=1.0, help='Scale factor for static image export.')
    parser.add_argument('--save-image', default=None, help='Optional path to also write a static image.')

    parser.add_argument('--cam-every', type=int, default=8, help='Subsample interval for frustums.')
    parser.add_argument('--cam-scale', type=float, default=0.1, help='Scale factor for frustums.')

    parser.add_argument('--marker-size', type=int, default=3, help='Marker size for sampled points.')
    parser.add_argument('--mark-interval', type=int, default=1, help='Interval for placing markers along the path.')
    parser.add_argument('--cmap', type=str, default='Viridis', help='Matplotlib colormap name for trajectory lines.')
    parser.add_argument('--translation-scale', type=float, default=5.0,
                        help='Uniform scale factor applied to translations before plotting (default: 5.0).')

    group_align = parser.add_mutually_exclusive_group()
    group_align.add_argument('--align-centroid', dest='align_centroid', action='store_true',
                             help='Subtract the global centroid before visualization.')
    group_align.add_argument('--no-align-centroid', dest='align_centroid', action='store_false',
                             help='Keep raw coordinates without centroid alignment.')
    parser.set_defaults(align_centroid=True)

    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument('--normalize-01', dest='normalize_01', action='store_true',
                            help='Normalize translations to [0,1] after alignment.')
    norm_group.add_argument('--no-normalize-01', dest='normalize_01', action='store_false',
                            help='Do not normalize translations to [0,1].')
    parser.set_defaults(normalize_01=True)

    return parser.parse_args()


def main():
    args = parse_args()

    trajectories = load_trajectory_data(args.input)

    if args.align_centroid:
        centroid = trajectory_centroid(trajectories)
        apply_translation_offset(trajectories, centroid)

    if args.normalize_01:
        positions = [d['pos'] for d in trajectories.values() if d['pos'].size > 0]
        if positions:
            stacked = np.concatenate(positions, axis=0)
            min_xyz = stacked.min(axis=0)
            max_xyz = stacked.max(axis=0)
            span = np.clip(max_xyz - min_xyz, 1e-9, None)
            for d in trajectories.values():
                if d['pos'].size > 0:
                    d['pos'] = (d['pos'] - min_xyz) / span
            recompute_translations(trajectories)

    if args.translation_scale != 1.0:
        scale_trajectories(trajectories, args.translation_scale)

    anchor_trajectories_at_origin(trajectories)

    fig, all_positions = create_trajectory_visualization(
        trajectories,
        positions_offset=np.zeros(3),
        marker_size=args.marker_size,
        mark_interval=args.mark_interval,
        cmap_name=args.cmap,
        name_prefix='cams'
    )

    overlay_p3d_frustums(
        fig,
        trajectories,
        subsample_interval=args.cam_every,
        camera_scale=args.cam_scale,
        offset=np.zeros(3),
        cmap_name=args.cmap,
        name_prefix='cams'
    )

    add_origin_and_arcs(fig, ARC_RADIUS)
    configure_layout(fig, args.title or 'Trajectory Visualization')
    apply_zoom(fig, all_positions, args.zoom_in)
    add_single_controls(fig)

    os.makedirs(args.output_dir, exist_ok=True)
    base = _dataset_label(args.input)
    out_html = os.path.join(args.output_dir, f'view_{base}.html')
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

