"""Tests for analysis geometry and middle-fish mapping."""

from __future__ import annotations

import unittest

import numpy as np

from analysis import geometry as geom
from analysis.positions import FishTrajectory, compute_pairwise_middle_bin_metrics


class TestMiddleFishId(unittest.TestCase):
    def test_top_row_plus_seven(self) -> None:
        self.assertEqual(geom.middle_fish_id(1), 8)
        self.assertEqual(geom.middle_fish_id(7), 14)

    def test_bottom_row_minus_seven(self) -> None:
        self.assertEqual(geom.middle_fish_id(22), 15)
        self.assertEqual(geom.middle_fish_id(28), 21)

    def test_small_rows_none(self) -> None:
        for fid in range(8, 22):
            self.assertIsNone(geom.middle_fish_id(fid))


class TestColumnAxis(unittest.TestCase):
    def test_row0_top_bottom(self) -> None:
        # Row 0 half height 40 mm; s = 40 - y
        self.assertAlmostEqual(geom.column_axis_s_mm(1, 40.0), 0.0, places=5)
        self.assertAlmostEqual(geom.column_axis_s_mm(1, 0.0), 40.0, places=5)
        self.assertAlmostEqual(geom.column_axis_s_mm(1, -40.0), 80.0, places=5)

    def test_batch_matches_scalar(self) -> None:
        y = np.array([0.0, -40.0])
        b = geom.column_axis_s_mm_batch(1, y)
        self.assertAlmostEqual(b[0], geom.column_axis_s_mm(1, 0.0), places=5)
        self.assertAlmostEqual(b[1], geom.column_axis_s_mm(1, -40.0), places=5)


class TestWallDistance(unittest.TestCase):
    def test_center_large_cell(self) -> None:
        hw, hh = geom.cell_half_dims_mm(1)
        d = geom.wall_distance_mm(0.0, 0.0, hw, hh)
        self.assertAlmostEqual(d, min(hw, hh), places=5)


class TestPairwiseMiddle(unittest.TestCase):
    def test_euclid_zero_when_stacked_s_aligns(self) -> None:
        # s_mm equal when fish1 at bottom of large (y=-40) and fish8 at top
        # of small (y=+9): both map to s=80 along the column.
        t = np.array([0.0, 0.05, 0.1], dtype=float)
        y1 = np.full(3, -40.0)
        y8 = np.full(3, 9.0)
        x = np.zeros(3, dtype=float)
        focal = FishTrajectory(fish_id=1, time_sec=t, x=x.copy(), y=y1)
        middle = FishTrajectory(fish_id=8, time_sec=t, x=x.copy(), y=y8)
        m = compute_pairwise_middle_bin_metrics(
            focal, middle, bin_index=0, bin_start=0.0, bin_end=0.15,
        )
        self.assertEqual(m.middle_fish_id, 8)
        self.assertEqual(m.n_valid_pair, 3)
        self.assertAlmostEqual(m.mean_d_middle_euclid_mm, 0.0, places=4)

    def test_euclid_at_cell_centers(self) -> None:
        # Both at local origin: vertical separation in column plane is 49 mm.
        t = np.array([0.0, 0.05], dtype=float)
        focal = FishTrajectory(fish_id=1, time_sec=t, x=np.zeros(2), y=np.zeros(2))
        middle = FishTrajectory(fish_id=8, time_sec=t, x=np.zeros(2), y=np.zeros(2))
        m = compute_pairwise_middle_bin_metrics(
            focal, middle, bin_index=0, bin_start=0.0, bin_end=0.1,
        )
        self.assertAlmostEqual(m.mean_d_middle_euclid_mm, 49.0, places=4)


if __name__ == "__main__":
    unittest.main()
