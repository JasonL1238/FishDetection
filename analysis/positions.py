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
class FishBinSummary:
    fish_id: int
    bin_index: int
    bin_start_sec: float
    bin_end_sec: float
    bin_duration_sec: float
    distance_total_mm: float
    speed_mm_per_sec: float
    n_samples_total: int
    n_samples_valid: int
    coverage_sec: float
    n_gaps: int
    time_moving_sec: float
    time_stationary_sec: float
    time_close_half_sec: Optional[float]
    dist_close_half_mm: Optional[float]
    time_near_social_sec: Optional[float]
    dist_near_social_mm: Optional[float]


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
SOCIAL_BAND_MM = 5.0


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


def slice_trajectory_into_bins(
    traj: FishTrajectory,
    bin_size_sec: float,
) -> List[Tuple[int, float, float, FishTrajectory]]:
    """
    Slice a trajectory into fixed-duration, non-overlapping time bins.

    Returns list of (bin_index, bin_start_sec, bin_end_sec, sub_trajectory).
    The last bin may be shorter than bin_size_sec.
    """
    t = traj.time_sec
    t_start = float(t[0])
    t_end = float(t[-1])

    bins: List[Tuple[int, float, float, FishTrajectory]] = []
    bin_idx = 0
    current_start = t_start

    while current_start < t_end:
        current_end = current_start + bin_size_sec
        mask = (t >= current_start) & (t < current_end)
        # Include the very last sample in the final bin
        if current_end >= t_end:
            mask = mask | (t == t_end)

        if mask.any():
            sub_traj = FishTrajectory(
                fish_id=traj.fish_id,
                time_sec=t[mask],
                x=traj.x[mask],
                y=traj.y[mask],
            )
            bins.append((bin_idx, current_start, current_end, sub_traj))

        bin_idx += 1
        current_start = current_end

    return bins


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

    Returns (distance_mm, n_total, n_valid, coverage_sec, n_gaps).
    """
    x, y, t = traj.x, traj.y, traj.time_sec
    valid = np.isfinite(x) & np.isfinite(y)

    n_total = int(valid.size)
    n_valid = int(valid.sum())
    n_gaps = count_missing_gaps(valid)

    dt = np.diff(t)
    coverage_sec = float(np.sum(dt[valid[:-1]])) if len(dt) > 0 else 0.0

    if n_valid < 2:
        return 0.0, n_total, n_valid, coverage_sec, n_gaps

    xv = x[valid]
    yv = y[valid]
    dx = np.diff(xv)
    dy = np.diff(yv)
    dist = float(np.sum(np.sqrt(dx * dx + dy * dy)))

    return dist, n_total, n_valid, coverage_sec, n_gaps


def _time_moving(
    traj: FishTrajectory,
    epsilon_mm: float,
) -> Tuple[float, float]:
    """
    A frame counts as "moving" if the step distance from the
    previous valid sample exceeds epsilon_mm.

    Returns (time_moving_sec, time_stationary_sec).
    """
    x, y, t = traj.x, traj.y, traj.time_sec
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        return 0.0, 0.0

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

    return time_mov, time_stat


def _close_half_metrics(
    traj: FishTrajectory,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Absolute time (seconds) and distance (mm) spent in the half of the cell
    closest to the middle fish (rows 1/2).

    Row 0 (top large): close half = y < 0 (below center, toward middle rows)
    Row 3 (bottom large): close half = y > 0 (above center, toward middle rows)
    Rows 1/2 (middle): not applicable, returns (None, None).
    """
    row = (traj.fish_id - 1) // COLS

    if row not in (0, 3):
        return None, None

    x, y, t = traj.x, traj.y, traj.time_sec
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        return None, None

    yv = y[valid]
    xv = x[valid]
    tv = t[valid]

    if row == 0:
        in_close = yv < 0
    else:
        in_close = yv > 0

    dt = np.diff(tv)
    in_close_steps = in_close[:-1]
    time_close_sec = float(np.sum(dt[in_close_steps]))

    dx = np.diff(xv)
    dy = np.diff(yv)
    step_dist = np.sqrt(dx * dx + dy * dy)
    dist_close_mm = float(np.sum(step_dist[in_close_steps]))

    return time_close_sec, dist_close_mm


def _near_social_window_metrics(
    traj: FishTrajectory,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Absolute time (seconds) and distance (mm) spent within SOCIAL_BAND_MM
    of the cell edge closest to the social window (middle rows).

    Row 0: social edge is y = -half_height (bottom). Band: y in [-40, -35].
    Row 3: social edge is y = +half_height (top).    Band: y in [35, 40].
    Rows 1/2: not applicable, returns (None, None).
    """
    row = (traj.fish_id - 1) // COLS

    if row not in (0, 3):
        return None, None

    x, y, t = traj.x, traj.y, traj.time_sec
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        return None, None

    yv = y[valid]
    xv = x[valid]
    tv = t[valid]

    half_h = ROW_HEIGHT_MM[row] / 2.0

    if row == 0:
        in_band = (yv >= -half_h) & (yv <= -half_h + SOCIAL_BAND_MM)
    else:
        in_band = (yv >= half_h - SOCIAL_BAND_MM) & (yv <= half_h)

    dt = np.diff(tv)
    in_band_steps = in_band[:-1]
    time_near_sec = float(np.sum(dt[in_band_steps]))

    dx = np.diff(xv)
    dy = np.diff(yv)
    step_dist = np.sqrt(dx * dx + dy * dy)
    dist_near_mm = float(np.sum(step_dist[in_band_steps]))

    return time_near_sec, dist_near_mm


def compute_bin_summary(
    traj: FishTrajectory,
    bin_index: int,
    bin_start: float,
    bin_end: float,
    epsilon_mm: float,
) -> FishBinSummary:
    """Build a per-fish, per-bin summary with all metrics in absolute units."""
    dist, n_total, n_valid, coverage_sec, n_gaps = _bridged_distance(traj)
    time_mov, time_stat = _time_moving(traj, epsilon_mm)
    time_close_sec, dist_close_mm = _close_half_metrics(traj)
    time_near_social, dist_near_social = _near_social_window_metrics(traj)

    elapsed = bin_end - bin_start
    speed = dist / elapsed if elapsed > 0 else 0.0

    return FishBinSummary(
        fish_id=traj.fish_id,
        bin_index=bin_index,
        bin_start_sec=bin_start,
        bin_end_sec=bin_end,
        bin_duration_sec=elapsed,
        distance_total_mm=dist,
        speed_mm_per_sec=speed,
        n_samples_total=n_total,
        n_samples_valid=n_valid,
        coverage_sec=coverage_sec,
        n_gaps=n_gaps,
        time_moving_sec=time_mov,
        time_stationary_sec=time_stat,
        time_close_half_sec=time_close_sec,
        dist_close_half_mm=dist_close_mm,
        time_near_social_sec=time_near_social,
        dist_near_social_mm=dist_near_social,
    )


def aggregate_bins(bins: List[FishBinSummary]) -> FishBinSummary:
    """
    Aggregate consecutive FishBinSummary records into a single summary.

    All additive fields are summed; speed is recomputed from the totals.
    All bins must belong to the same fish_id.
    """
    if not bins:
        raise ValueError("Cannot aggregate an empty list of bins")

    fish_id = bins[0].fish_id
    if any(b.fish_id != fish_id for b in bins):
        raise ValueError("All bins must belong to the same fish_id")

    total_dist = sum(b.distance_total_mm for b in bins)
    total_duration = bins[-1].bin_end_sec - bins[0].bin_start_sec
    speed = total_dist / total_duration if total_duration > 0 else 0.0

    has_close = bins[0].time_close_half_sec is not None
    time_close = (
        sum(b.time_close_half_sec for b in bins if b.time_close_half_sec is not None)
        if has_close else None
    )
    dist_close = (
        sum(b.dist_close_half_mm for b in bins if b.dist_close_half_mm is not None)
        if has_close else None
    )

    has_social = bins[0].time_near_social_sec is not None
    time_social = (
        sum(b.time_near_social_sec for b in bins if b.time_near_social_sec is not None)
        if has_social else None
    )
    dist_social = (
        sum(b.dist_near_social_mm for b in bins if b.dist_near_social_mm is not None)
        if has_social else None
    )

    return FishBinSummary(
        fish_id=fish_id,
        bin_index=bins[0].bin_index,
        bin_start_sec=bins[0].bin_start_sec,
        bin_end_sec=bins[-1].bin_end_sec,
        bin_duration_sec=total_duration,
        distance_total_mm=total_dist,
        speed_mm_per_sec=speed,
        n_samples_total=sum(b.n_samples_total for b in bins),
        n_samples_valid=sum(b.n_samples_valid for b in bins),
        coverage_sec=sum(b.coverage_sec for b in bins),
        n_gaps=sum(b.n_gaps for b in bins),
        time_moving_sec=sum(b.time_moving_sec for b in bins),
        time_stationary_sec=sum(b.time_stationary_sec for b in bins),
        time_close_half_sec=time_close,
        dist_close_half_mm=dist_close,
        time_near_social_sec=time_social,
        dist_near_social_mm=dist_social,
    )


def write_distance_summary_csv(
    summaries: Iterable[FishBinSummary],
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
                "bin_index",
                "bin_start_sec",
                "bin_end_sec",
                "bin_duration_sec",
                "distance_total_mm",
                "speed_mm_per_sec",
                "n_samples_total",
                "n_samples_valid",
                "coverage_sec",
                "n_gaps",
                "time_moving_sec",
                "time_stationary_sec",
                "time_close_half_sec",
                "dist_close_half_mm",
                "time_near_social_sec",
                "dist_near_social_mm",
            ]
        )
        for s in rows:
            writer.writerow(
                [
                    s.fish_id,
                    s.bin_index,
                    round(s.bin_start_sec, 4),
                    round(s.bin_end_sec, 4),
                    round(s.bin_duration_sec, 4),
                    round(s.distance_total_mm, 4),
                    round(s.speed_mm_per_sec, 4),
                    s.n_samples_total,
                    s.n_samples_valid,
                    round(s.coverage_sec, 4),
                    s.n_gaps,
                    round(s.time_moving_sec, 4),
                    round(s.time_stationary_sec, 4),
                    round(s.time_close_half_sec, 4) if s.time_close_half_sec is not None else "",
                    round(s.dist_close_half_mm, 4) if s.dist_close_half_mm is not None else "",
                    round(s.time_near_social_sec, 4) if s.time_near_social_sec is not None else "",
                    round(s.dist_near_social_mm, 4) if s.dist_near_social_mm is not None else "",
                ]
            )
