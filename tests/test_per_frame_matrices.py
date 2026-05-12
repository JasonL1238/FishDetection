"""Tests for per-frame fish×frame matrix exports."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.per_frame_matrices import compute_per_frame_matrices


class TestPerFrameMatrices(unittest.TestCase):
    def test_d_middle_euclid_aligns_with_binned_hypot(self) -> None:
        t = np.array([0.0, 0.05, 0.1], dtype=float)
        fr = np.array([0, 1, 2], dtype=int)
        y1 = np.full(3, -40.0)
        y8 = np.full(3, 9.0)
        x = np.zeros(3, dtype=float)
        fish_xy = {
            1: (x.copy(), y1.copy()),
            8: (x.copy(), y8.copy()),
        }
        _, _, mats = compute_per_frame_matrices(
            t, fr, fish_xy, epsilon_mm=0.25, outer_edge_alpha=0.2
        )
        d = mats["d_middle_euclid_mm"]
        self.assertTrue(np.all(np.isfinite(d[0])))
        self.assertTrue(np.all(np.isnan(d[1])))
        self.assertAlmostEqual(float(np.mean(d[0])), 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
