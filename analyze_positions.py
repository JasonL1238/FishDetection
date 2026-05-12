#!/usr/bin/env python3
"""
Analyze fish movement from a pipeline `*_positions.csv` export.

Outputs:
  - distance_summary_binned.csv (legacy, all per-bin columns)
  - bins_locomotion.csv, bins_social_large_cells.csv, bins_geometry.csv,
    bins_pairwise_middle.csv
  - per_frame_<stat>.csv (fish_id rows, frame columns, one file per stat)
  - analysis_manifest.json
  - fish01_xy_over_time.png ... fish28_xy_over_time.png (x(t), y(t) in mm)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis import geometry as geom
from analysis.positions import (
    build_trajectories,
    compute_bin_summary,
    compute_geometry_bin_metrics,
    compute_pairwise_middle_bin_metrics,
    find_positions_csv,
    load_positions_csv,
    slice_trajectory_into_bins,
    write_analysis_manifest,
    write_bins_geometry_csv,
    write_bins_locomotion_csv,
    write_bins_pairwise_middle_csv,
    write_bins_social_large_cells_csv,
    write_distance_summary_csv,
)
from analysis.per_frame_matrices import write_all_per_frame_stat_csvs
from analysis.plots import plot_xy_over_time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze distance traveled and x/y over time from positions CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to a *_positions.csv (alternative to --csv / --run-dir).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run output directory (e.g. data/output/<video_stem>/).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Explicit path to a *_positions.csv (overrides --run-dir discovery).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir for analysis artifacts (default: <run-dir>/analysis).",
    )
    parser.add_argument(
        "--fish-count",
        type=int,
        default=28,
        help="Number of fish/cells (default: 28).",
    )
    parser.add_argument(
        "--movement-threshold-mm",
        type=float,
        default=0.25,
        help="Distance threshold (mm) below which a frame-to-frame step "
             "counts as stationary (default: 0.25).",
    )
    parser.add_argument(
        "--bin-size-sec",
        type=float,
        default=0.05,
        help="Time bin size in seconds (default: 0.05, i.e. per-frame at 20 FPS). "
             "Larger bins can be reconstructed by aggregating smaller ones.",
    )
    parser.add_argument(
        "--outer-edge-alpha",
        type=float,
        default=0.2,
        help="Outer-edge fraction: count samples where wall_dist < alpha * "
             "min(half_w, half_h) (default: 0.2).",
    )
    args = parser.parse_args()

    if args.csv_path is None and args.csv is None and args.run_dir is None:
        parser.error("Provide a CSV path (positional or --csv) or a --run-dir")

    if args.csv is not None and args.csv_path is not None and args.csv != args.csv_path:
        parser.error("Provide CSV via positional OR --csv, not both")

    if args.csv is not None or args.csv_path is not None:
        csv_path = args.csv if args.csv is not None else args.csv_path
        run_dir = csv_path.parent
    else:
        run_dir = Path(args.run_dir)
        csv_path = find_positions_csv(run_dir)

    out_dir = args.out_dir or (Path(run_dir) / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Positions CSV: {csv_path}")
    print(f"Analysis out:  {out_dir}")
    print(f"Movement threshold: {args.movement_threshold_mm} mm")
    print(f"Bin size: {args.bin_size_sec} sec")
    print(f"Outer-edge alpha: {args.outer_edge_alpha}")

    time_sec, frame, fish_xy, units = load_positions_csv(
        csv_path, fish_count=args.fish_count
    )
    print(f"Units: {units}")

    if units != "mm":
        print(
            "Warning: per-frame geometry uses mm cell model; "
            "positions are not in mm — per_frame CSVs may be meaningless."
        )

    per_frame_files = write_all_per_frame_stat_csvs(
        out_dir,
        time_sec,
        frame,
        fish_xy,
        args.movement_threshold_mm,
        args.outer_edge_alpha,
    )
    print(f"Wrote {len(per_frame_files)} per-frame matrix CSVs")

    trajs = build_trajectories(time_sec, fish_xy)
    traj_by_id = {tr.fish_id: tr for tr in trajs}

    all_bin_summaries = []
    all_geometry_metrics = []
    all_pairwise_metrics = []
    for tr in trajs:
        bins = slice_trajectory_into_bins(tr, args.bin_size_sec)
        for bin_idx, bin_start, bin_end, sub_traj in bins:
            summary = compute_bin_summary(
                sub_traj, bin_idx, bin_start, bin_end,
                args.movement_threshold_mm,
            )
            all_bin_summaries.append(summary)
            all_geometry_metrics.append(
                compute_geometry_bin_metrics(
                    sub_traj,
                    bin_idx,
                    bin_start,
                    bin_end,
                    args.outer_edge_alpha,
                )
            )
            mid_id = geom.middle_fish_id(tr.fish_id)
            mid_tr = traj_by_id[mid_id] if mid_id is not None else None
            all_pairwise_metrics.append(
                compute_pairwise_middle_bin_metrics(
                    tr, mid_tr, bin_idx, bin_start, bin_end,
                )
            )

    write_distance_summary_csv(
        all_bin_summaries, out_dir / "distance_summary_binned.csv"
    )
    write_bins_locomotion_csv(
        all_bin_summaries, out_dir / "bins_locomotion.csv"
    )
    write_bins_social_large_cells_csv(
        all_bin_summaries, out_dir / "bins_social_large_cells.csv"
    )
    write_bins_geometry_csv(
        all_geometry_metrics, out_dir / "bins_geometry.csv"
    )
    write_bins_pairwise_middle_csv(
        all_pairwise_metrics, out_dir / "bins_pairwise_middle.csv"
    )

    out_names = [
        "distance_summary_binned.csv",
        "bins_locomotion.csv",
        "bins_social_large_cells.csv",
        "bins_geometry.csv",
        "bins_pairwise_middle.csv",
        *per_frame_files,
    ]
    middle_rule = (
        "row0_large: middle_fish_id = fish_id + 7 (row1 same column); "
        "row3_large: middle_fish_id = fish_id - 7 (row2 same column); "
        "else null"
    )
    pairwise_def = (
        "mean Euclidean distance in column plane: hypot(x_focal - x_middle, "
        "s_focal - s_middle) with s from geometry.column_axis_s_mm_batch per fish"
    )
    frac_def = (
        "fraction of valid samples with wall_dist < outer_edge_alpha * "
        "min(half_w, half_h)"
    )
    per_frame_layout = (
        "per_frame_*.csv: rows are fish_id (1–28); columns are frame indices "
        "from positions CSV (duplicate frame values get suffix __colIndex). "
        "Empty cells are missing or not applicable. "
        "d_middle_euclid_mm matches bins: hypot(x_focal-x_middle, s_focal-s_middle)."
    )
    write_analysis_manifest(
        out_dir / "analysis_manifest.json",
        positions_csv=str(csv_path),
        bin_size_sec=args.bin_size_sec,
        movement_threshold_mm=args.movement_threshold_mm,
        outer_edge_alpha=args.outer_edge_alpha,
        output_files=out_names,
        pairwise_middle_definition=pairwise_def,
        middle_fish_rule=middle_rule,
        frac_outer_edge_definition=frac_def,
        per_frame_csv_layout=per_frame_layout,
    )

    print(f"Wrote {len(all_bin_summaries)} bin summaries "
          f"({len(trajs)} fish x {args.bin_size_sec}s bins)")

    for tr in trajs:
        out_path = out_dir / f"fish{tr.fish_id:02d}_xy_over_time.png"
        plot_xy_over_time(tr, out_path, units=units)

    print("Done.")


if __name__ == "__main__":
    main()
