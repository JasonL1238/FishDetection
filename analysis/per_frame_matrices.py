"""
Fish-by-frame matrices: each row is fish_id, each column is one positions row
(frame index from CSV). One CSV per statistic.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import geometry as geom

SOCIAL_BAND_MM = 5.0
ROW_HEIGHT_MM = geom.ROW_HEIGHT_MM


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return ""
    return str(round(float(v), 6))


def _in_close_half_at_sample(fish_id: int, y: float) -> float:
    row = geom.fish_row(fish_id)
    if row == 0:
        return 1.0 if y < 0 else 0.0
    if row == 3:
        return 1.0 if y > 0 else 0.0
    return float("nan")


def _in_near_social_at_sample(fish_id: int, y: float) -> float:
    row = geom.fish_row(fish_id)
    if row not in (0, 3):
        return float("nan")
    half_h = ROW_HEIGHT_MM[row] / 2.0
    if row == 0:
        ok = (-half_h <= y <= -half_h + SOCIAL_BAND_MM)
    else:
        ok = (half_h - SOCIAL_BAND_MM <= y <= half_h)
    return 1.0 if ok else 0.0


def compute_per_frame_matrices(
    time_sec: np.ndarray,
    frames: np.ndarray,
    fish_xy: Dict[int, Tuple[np.ndarray, np.ndarray]],
    epsilon_mm: float,
    outer_edge_alpha: float,
) -> Tuple[List[int], np.ndarray, Dict[str, np.ndarray]]:
    """
    Build (n_fish, n_frames) matrices. Pairwise middle matches binned metric:
    hypot(x_focal - x_middle, s_focal - s_middle) with s from column_axis.
    """
    fish_ids = sorted(fish_xy.keys())
    n_fish = len(fish_ids)
    n = len(time_sec)
    frame_headers = np.asarray(frames, dtype=int)

    dist_center = np.full((n_fish, n), np.nan)
    wall_d = np.full((n_fish, n), np.nan)
    in_outer = np.full((n_fish, n), np.nan)
    in_close = np.full((n_fish, n), np.nan)
    in_social = np.full((n_fish, n), np.nan)
    step_dist = np.full((n_fish, n), np.nan)
    dt_sec = np.full((n_fish, n), np.nan)
    inst_speed = np.full((n_fish, n), np.nan)
    is_moving = np.full((n_fish, n), np.nan)
    d_middle_euclid = np.full((n_fish, n), np.nan)

    xs = [fish_xy[fid][0] for fid in fish_ids]
    ys = [fish_xy[fid][1] for fid in fish_ids]

    for fi, fid in enumerate(fish_ids):
        x = xs[fi]
        y = ys[fi]
        valid = np.isfinite(x) & np.isfinite(y)
        hw, hh = geom.cell_half_dims_mm(fid)
        thresh = outer_edge_alpha * min(hw, hh)

        dist_center[fi, valid] = np.hypot(x[valid], y[valid])
        wdist = np.minimum(hw - np.abs(x), hh - np.abs(y))
        wall_d[fi, valid] = wdist[valid]
        in_outer[fi, valid] = (wdist[valid] < thresh).astype(float)

        for i in range(n):
            if not valid[i]:
                continue
            in_close[fi, i] = _in_close_half_at_sample(fid, float(y[i]))
            in_social[fi, i] = _in_near_social_at_sample(fid, float(y[i]))

        for i in range(1, n):
            dt = float(time_sec[i] - time_sec[i - 1])
            dt_sec[fi, i] = dt
            if valid[i] and valid[i - 1]:
                sd = float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
                step_dist[fi, i] = sd
                if dt > 0:
                    inst_speed[fi, i] = sd / dt
                is_moving[fi, i] = 1.0 if sd > epsilon_mm else 0.0

        mid = geom.middle_fish_id(fid)
        if mid is not None:
            mid_idx = fish_ids.index(mid)
            xm, ym = xs[mid_idx], ys[mid_idx]
            vm = np.isfinite(xm) & np.isfinite(ym)
            both = valid & vm
            if np.any(both):
                sf = geom.column_axis_s_mm_batch(fid, y)
                sm = geom.column_axis_s_mm_batch(mid, ym)
                d_middle_euclid[fi, both] = np.hypot(
                    (x - xm)[both], (sf - sm)[both]
                )

    matrices = {
        "dist_cell_center_mm": dist_center,
        "wall_distance_mm": wall_d,
        "in_outer_edge": in_outer,
        "in_close_half": in_close,
        "in_near_social_band": in_social,
        "step_distance_mm": step_dist,
        "dt_sec": dt_sec,
        "instant_speed_mm_per_s": inst_speed,
        "is_moving": is_moving,
        "d_middle_euclid_mm": d_middle_euclid,
    }
    return fish_ids, frame_headers, matrices


def write_fish_by_frame_matrix_csv(
    out_path: Path,
    fish_ids: List[int],
    frame_headers: np.ndarray,
    matrix: np.ndarray,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_fish, n = matrix.shape
    if n_fish != len(fish_ids):
        raise ValueError("fish_ids length must match matrix rows")
    if n != len(frame_headers):
        raise ValueError("frame_headers length must match matrix columns")

    seen: Dict[int, int] = {}
    col_names: List[str] = []
    for j in range(n):
        fr = int(frame_headers[j])
        seen[fr] = seen.get(fr, 0) + 1
        col_names.append(f"{fr}__{j}" if seen[fr] > 1 else str(fr))

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fish_id"] + col_names)
        for fi, fid in enumerate(fish_ids):
            w.writerow([fid] + [_fmt(float(matrix[fi, j])) for j in range(n)])


def write_all_per_frame_stat_csvs(
    out_dir: Path,
    time_sec: np.ndarray,
    frames: np.ndarray,
    fish_xy: Dict[int, Tuple[np.ndarray, np.ndarray]],
    epsilon_mm: float,
    outer_edge_alpha: float,
) -> List[str]:
    fish_ids, frame_headers, matrices = compute_per_frame_matrices(
        time_sec, frames, fish_xy, epsilon_mm, outer_edge_alpha
    )
    out_dir = Path(out_dir)
    names: List[str] = []
    for stat_name, mat in matrices.items():
        fname = f"per_frame_{stat_name}.csv"
        write_fish_by_frame_matrix_csv(
            out_dir / fname, fish_ids, frame_headers, mat
        )
        names.append(fname)
    return names
