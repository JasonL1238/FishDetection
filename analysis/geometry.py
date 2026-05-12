"""
Cell layout and column-axis geometry for position analysis.

Fish IDs 1–28 are row-major: row 0 (ids 1–7), row 1 (8–14), row 2 (15–21),
row 3 (22–28). Physical sizes match grid.py (large rows 0 and 3).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

COLS = 7
CELL_WIDTH_MM = 30.0
ROW_HEIGHT_MM = {0: 80.0, 1: 18.0, 2: 18.0, 3: 80.0}

CellType = Literal["large", "small"]


def fish_row(fish_id: int) -> int:
    return (fish_id - 1) // COLS


def fish_col(fish_id: int) -> int:
    return (fish_id - 1) % COLS


def cell_type(fish_id: int) -> CellType:
    return "large" if fish_row(fish_id) in (0, 3) else "small"


def middle_fish_id(fish_id: int) -> Optional[int]:
    """
    Adjacent small-compartment fish in the same column: row 0 -> +7 (row 1),
    row 3 -> -7 (row 2). Middle rows have no reference for this metric.
    """
    row = fish_row(fish_id)
    if row == 0:
        return fish_id + 7
    if row == 3:
        return fish_id - 7
    return None


def row_stack_offset_mm(row: int) -> float:
    """Cumulative mm from the top of the column down to the top of this row."""
    return float(sum(ROW_HEIGHT_MM[r] for r in range(row)))


def column_axis_s_mm(fish_id: int, y_mm: float) -> float:
    """
    Scalar position (mm) along the column from the top of row 0 downward.

    Cell-local y is positive toward the top of the frame (center origin).
    s increases monotonically toward the bottom of the plate.
    """
    row = fish_row(fish_id)
    half_h = ROW_HEIGHT_MM[row] / 2.0
    base = row_stack_offset_mm(row)
    return base + (half_h - y_mm)


def column_axis_s_mm_batch(fish_id: int, y_mm: np.ndarray) -> np.ndarray:
    """Vectorized `column_axis_s_mm` for arrays of y (mm)."""
    row = fish_row(fish_id)
    half_h = ROW_HEIGHT_MM[row] / 2.0
    base = row_stack_offset_mm(row)
    return base + half_h - y_mm


def cell_half_dims_mm(fish_id: int) -> Tuple[float, float]:
    """Half-width and half-height in mm for the fish's cell."""
    row = fish_row(fish_id)
    w = CELL_WIDTH_MM / 2.0
    h = ROW_HEIGHT_MM[row] / 2.0
    return w, h


def wall_distance_mm(x_mm: float, y_mm: float, half_w: float, half_h: float) -> float:
    """Axis-aligned distance from (x,y) to the nearest cell wall (mm)."""
    return min(half_w - abs(x_mm), half_h - abs(y_mm))


def centroid_distance_mm(x_mm: float, y_mm: float) -> float:
    return float(np.hypot(x_mm, y_mm))
