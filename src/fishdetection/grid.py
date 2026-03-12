"""
Grid utilities for dividing video frames into cells.

Provides functions to compute the 7x4 grid layout and to detect
the largest fish blob within a single cell.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from .hsv_masker import HSVMasker


def get_grid_cells(width: int, height: int) -> Tuple[List[tuple], List[int]]:
    """
    Compute 28-cell grid boundaries (7 columns x 4 rows).

    The horizontal row boundaries are adjusted so the middle two rows
    are slightly compressed toward the frame center.

    Returns:
        (cells, row_boundaries) where cells is a list of
        (x_start, y_start, x_end, y_end) tuples in row-major order
        (row 0 left-to-right, then row 1, etc.) and row_boundaries
        is the list of y-coordinates separating rows.
    """
    cols = 7
    cell_width = width / cols

    rows_for_calc = 7
    cell_height_base = height / rows_for_calc

    line_3_y = int(3 * cell_height_base)
    line_4_y = int(4 * cell_height_base)

    adjustment = int(cell_height_base * 0.15)
    top_line_y = line_3_y - adjustment
    center_line_y = height // 2
    bottom_line_y = line_4_y + adjustment

    row_boundaries = [0, top_line_y, center_line_y, bottom_line_y, height]

    cells = []
    for row_idx in range(len(row_boundaries) - 1):
        y_start = row_boundaries[row_idx]
        y_end = row_boundaries[row_idx + 1]
        for col_idx in range(cols):
            x_start = int(col_idx * cell_width)
            x_end = int((col_idx + 1) * cell_width)
            cells.append((x_start, y_start, x_end, y_end))

    return cells, row_boundaries


def find_largest_blob_in_cell(
    hsv_masker: HSVMasker,
    bg_subtracted_frame: np.ndarray,
    x_start: int, y_start: int,
    x_end: int, y_end: int,
) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    """
    Find the largest blob/contour in a single grid cell.

    Returns:
        (contour, centroid) in full-frame coordinates,
        or (None, None) if no blob is found.
    """
    cell_region = bg_subtracted_frame[y_start:y_end, x_start:x_end]

    if cell_region.size == 0:
        return None, None

    original_min_size = hsv_masker.min_object_size
    hsv_masker.min_object_size = 1

    contours, centroids = (
        hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(cell_region)
    )

    hsv_masker.min_object_size = original_min_size

    if len(contours) == 0:
        return None, None

    largest_contour = None
    largest_area = 0
    largest_centroid = None

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
            largest_centroid = centroids[i]

    if largest_contour is None:
        return None, None

    full_frame_contour = largest_contour.copy()
    full_frame_contour[:, :, 0] += x_start
    full_frame_contour[:, :, 1] += y_start

    full_frame_centroid = (largest_centroid[0] + y_start,
                           largest_centroid[1] + x_start)

    return full_frame_contour, full_frame_centroid
