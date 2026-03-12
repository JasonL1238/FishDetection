from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import math

import numpy as np


@dataclass(frozen=True)
class FishTrajectory:
    fish_id: int
    time_sec: np.ndarray  # shape (T,)
    x: np.ndarray  # shape (T,), float, NaN for missing
    y: np.ndarray  # shape (T,), float, NaN for missing


@dataclass(frozen=True)
class FishDistanceSummary:
    fish_id: int
    distance_total_mm: float
    n_samples_total: int
    n_samples_valid: int
    coverage_pct: float
    n_gaps: int
    time_moving_sec: float
    time_stationary_sec: float
    pct_moving: float


def find_positions_csv(run_dir: Path) -> Path:
    """Find the newest `*_positions.csv` in a run directory."""
    run_dir = Path(run_dir)
    matches = list(run_dir.glob("*_positions.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No '*_positions.csv' found in run dir: {run_dir}"
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def _parse_float(cell: str) -> float:
    cell = cell.strip()
    if cell == "":
        return float("nan")
    return float(cell)


def _detect_units(fieldnames: list[str]) -> str:
    """Detect whether CSV uses _mm, _norm, or raw column naming."""
    for f in fieldnames:
        if f.endswith("_x_mm"):
            return "mm"
        if f.endswith("_x_norm"):
            return "norm"
    return "raw"


COLS = 7
CELL_WIDTH_MM = 30.0
ROW_HEIGHT_MM = {0: 80.0, 1: 18.0, 2: 18.0, 3: 80.0}


def _norm_to_mm(fish_id: int, x_norm: float, y_norm: float) -> Tuple[float, float]:
    """Convert normalized (center-origin) coords to mm for a given fish."""
    row = (fish_id - 1) // COLS
    return x_norm * CELL_WIDTH_MM, y_norm * ROW_HEIGHT_MM[row]


def load_positions_csv(
    csv_path: Path,
    fish_count: int = 28,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, Tuple[np.ndarray, np.ndarray]], str]:
    """
    Load wide-format positions CSV.

    Returns:
      time_sec (T,), frame (T,), fish_xy dict:
        fish_id -> (x(T,), y(T,)) with NaNs for missing,
      units: "mm" or "raw".

    If the CSV uses _norm columns, values are automatically converted to mm.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        required = {"time_sec", "frame"}
        missing_required = required - set(reader.fieldnames)
        if missing_required:
            raise ValueError(
                f"CSV missing required columns {sorted(missing_required)}: {csv_path}"
            )

        units = _detect_units(list(reader.fieldnames))

        def colnames_for_fish(fid: int) -> Tuple[str, str]:
            candidates = [
                (f"fish{fid}_x_mm", f"fish{fid}_y_mm"),
                (f"fish{fid}_x_norm", f"fish{fid}_y_norm"),
                (f"fish{fid}_x", f"fish{fid}_y"),
            ]
            fields = set(reader.fieldnames or [])
            for xcol, ycol in candidates:
                if xcol in fields and ycol in fields:
                    return xcol, ycol
            raise ValueError(
                f"CSV missing fish columns for fish {fid} in {csv_path}"
            )

        fish_cols: Dict[int, Tuple[str, str]] = {
            fid: colnames_for_fish(fid) for fid in range(1, fish_count + 1)
        }

        times: List[float] = []
        frames: List[int] = []
        xs: Dict[int, List[float]] = {fid: [] for fid in fish_cols}
        ys: Dict[int, List[float]] = {fid: [] for fid in fish_cols}

        for row in reader:
            times.append(float(row["time_sec"]))
            frames.append(int(float(row["frame"])))
            for fid, (xcol, ycol) in fish_cols.items():
                xval = _parse_float(row.get(xcol, ""))
                yval = _parse_float(row.get(ycol, ""))
                if units == "norm" and math.isfinite(xval) and math.isfinite(yval):
                    xval, yval = _norm_to_mm(fid, xval, yval)
                xs[fid].append(xval)
                ys[fid].append(yval)

    t = np.asarray(times, dtype=float)
    fr = np.asarray(frames, dtype=int)
    fish_xy = {
        fid: (np.asarray(xs[fid], dtype=float), np.asarray(ys[fid], dtype=float))
        for fid in fish_cols
    }
    effective_units = "mm" if units in ("mm", "norm") else "raw"
    return t, fr, fish_xy, effective_units


def build_trajectories(
    time_sec: np.ndarray,
    fish_xy: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> List[FishTrajectory]:
    trajs: List[FishTrajectory] = []
    for fid in sorted(fish_xy.keys()):
        x, y = fish_xy[fid]
        trajs.append(FishTrajectory(fish_id=fid, time_sec=time_sec, x=x, y=y))
    return trajs


def count_missing_gaps(valid_mask: np.ndarray) -> int:
    """
    Count contiguous missing stretches that occur after at least one valid sample.

    Example:
      valid: T T F F T F T -> gaps = 2 (the F F and the single F)
    """
    if valid_mask.size == 0:
        return 0

    gaps = 0
    in_missing = False
    seen_valid = False
    for v in valid_mask:
        if v:
            seen_valid = True
            in_missing = False
        else:
            if seen_valid and not in_missing:
                gaps += 1
                in_missing = True
    return gaps


def _bridged_distance(traj: FishTrajectory) -> Tuple[float, int, int, float, int]:
    """
    Core distance computation: sum Euclidean distances between successive
    valid samples (bridging gaps).

    Returns (distance_mm, n_total, n_valid, coverage_pct, n_gaps).
    """
    x, y = traj.x, traj.y
    valid = np.isfinite(x) & np.isfinite(y)

    n_total = int(valid.size)
    n_valid = int(valid.sum())
    coverage = (n_valid / n_total * 100.0) if n_total > 0 else 0.0
    n_gaps = count_missing_gaps(valid)

    if n_valid < 2:
        return 0.0, n_total, n_valid, coverage, n_gaps

    xv = x[valid]
    yv = y[valid]
    dx = np.diff(xv)
    dy = np.diff(yv)
    dist = float(np.sum(np.sqrt(dx * dx + dy * dy)))

    return dist, n_total, n_valid, coverage, n_gaps


def _time_moving(
    traj: FishTrajectory,
    epsilon_mm: float,
) -> Tuple[float, float, float]:
    """
    Option A: a frame counts as "moving" if the step distance from the
    previous valid sample exceeds epsilon_mm.

    Returns (time_moving_sec, time_stationary_sec, pct_moving).
    """
    x, y, t = traj.x, traj.y, traj.time_sec
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        return 0.0, 0.0, 0.0

    xv = x[valid]
    yv = y[valid]
    tv = t[valid]

    dx = np.diff(xv)
    dy = np.diff(yv)
    d = np.sqrt(dx * dx + dy * dy)
    dt = np.diff(tv)

    moving = d > epsilon_mm
    time_mov = float(np.sum(dt[moving]))
    time_stat = float(np.sum(dt[~moving]))
    total = time_mov + time_stat
    pct = (time_mov / total * 100.0) if total > 0 else 0.0

    return time_mov, time_stat, pct


def compute_fish_summary(
    traj: FishTrajectory,
    epsilon_mm: float,
) -> FishDistanceSummary:
    """Build a full per-fish summary: distance + time moving."""
    dist, n_total, n_valid, coverage, n_gaps = _bridged_distance(traj)
    time_mov, time_stat, pct_mov = _time_moving(traj, epsilon_mm)

    return FishDistanceSummary(
        fish_id=traj.fish_id,
        distance_total_mm=dist,
        n_samples_total=n_total,
        n_samples_valid=n_valid,
        coverage_pct=coverage,
        n_gaps=n_gaps,
        time_moving_sec=time_mov,
        time_stationary_sec=time_stat,
        pct_moving=pct_mov,
    )


def write_distance_summary_csv(
    summaries: Iterable[FishDistanceSummary],
    out_csv_path: Path,
) -> None:
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(summaries)

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fish_id",
                "distance_total_mm",
                "n_samples_total",
                "n_samples_valid",
                "coverage_pct",
                "n_gaps",
                "time_moving_sec",
                "time_stationary_sec",
                "pct_moving",
            ]
        )
        for s in rows:
            writer.writerow(
                [
                    s.fish_id,
                    round(s.distance_total_mm, 4),
                    s.n_samples_total,
                    s.n_samples_valid,
                    round(s.coverage_pct, 3),
                    s.n_gaps,
                    round(s.time_moving_sec, 4),
                    round(s.time_stationary_sec, 4),
                    round(s.pct_moving, 2),
                ]
            )
