#!/usr/bin/env python3
"""
Analyze fish movement from a pipeline `*_positions.csv` export.

Outputs:
  - distance_summary.csv (per-fish distance in mm, time moving/stationary)
  - fish01_xy_over_time.png ... fish28_xy_over_time.png (x(t), y(t) in mm)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analysis.positions import (
    build_trajectories,
    compute_fish_summary,
    find_positions_csv,
    load_positions_csv,
    write_distance_summary_csv,
)
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
             "counts as stationary (default: 0.5).",
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

    time_sec, frame, fish_xy, units = load_positions_csv(
        csv_path, fish_count=args.fish_count
    )
    print(f"Units: {units}")

    trajs = build_trajectories(time_sec, fish_xy)

    summaries = [
        compute_fish_summary(tr, args.movement_threshold_mm) for tr in trajs
    ]
    write_distance_summary_csv(summaries, out_dir / "distance_summary.csv")

    for tr in trajs:
        out_path = out_dir / f"fish{tr.fish_id:02d}_xy_over_time.png"
        plot_xy_over_time(tr, out_path, units=units)

    print("Done.")


if __name__ == "__main__":
    main()
