# analysis -- Post-Processing Analysis

This package takes the positions CSV produced by the detection pipeline and computes behavioral metrics: how much each fish moved, where it spent its time, how close it stayed to the walls or to neighboring fish, and more.

All analysis is run via `analyze_positions.py` at the project root. The modules here do the actual computation.

## How Time Bins Work

The positions CSV has one row per video frame (typically 20 frames per second). Rather than reporting a single number for the entire video, the analysis divides the recording into short **time bins** (default: 0.05 seconds, which equals one frame at 20 FPS) and computes metrics for each bin independently.

This means you get a granular time series: for every fish, for every time bin, you know how far it moved, how fast it was going, where it was positioned, and so on. You can aggregate these bins into larger windows (1-second, 5-second, etc.) downstream in your own analysis.

## Module Guide

### `positions.py` -- Trajectories, Binned Metrics, and CSV Export

The largest module. Handles the full analysis pipeline:

1. **Load** the positions CSV and parse x/y coordinates for all 28 fish.
2. **Build trajectories** -- a time series of (x, y) positions for each fish.
3. **Slice into time bins** -- divide each trajectory into fixed-duration, non-overlapping windows.
4. **Compute metrics** per bin per fish:
   - **Locomotion**: distance, speed, moving/stationary time.
   - **Social proximity** (large cells only): close-half time, social band time.
   - **Geometry**: wall distance, centroid distance, outer-edge fraction.
   - **Pairwise middle**: distance to the adjacent small-compartment fish.
5. **Write CSVs** -- one file per metric category, plus a combined legacy file.

### `geometry.py` -- Cell Layout Math

Provides functions that map a fish ID to its grid position and physical cell dimensions. Key concepts:

- **`fish_row()` / `fish_col()`** -- given a fish ID (1--28), returns its row (0--3) and column (0--6).
- **`cell_type()`** -- returns "large" (rows 0, 3) or "small" (rows 1, 2).
- **`middle_fish_id()`** -- for a fish in a large compartment, returns the ID of the fish in the adjacent small compartment in the same column. For example, Fish 3 (row 0, col 2) is paired with Fish 10 (row 1, col 2). Fish in small compartments return no pairing.
- **`wall_distance_mm()`** -- how far the fish is from the nearest wall of its cell.
- **`column_axis_s_mm()`** -- a coordinate that stacks all four rows in a column into a single vertical axis, so you can compute the true physical distance between a large-row fish and its paired middle-row fish even though they are in different cells.

### `per_frame_matrices.py` -- Fish-by-Frame Matrices

Computes ten statistics at every single frame for every fish, organized as matrices with 28 rows (one per fish) and one column per frame. Each statistic is saved as its own CSV file.

### `plots.py` -- Trajectory Plots

Generates a PNG image for each fish showing its x position and y position plotted over time. These are simple line charts useful for visual inspection.

## Metric Definitions

### Locomotion metrics (in `bins_locomotion.csv`)

| Column | What it measures |
|--------|-----------------|
| `distance_total_mm` | Total distance the fish traveled during this bin, in millimeters. Computed as the sum of straight-line distances between consecutive detected positions. |
| `speed_mm_per_sec` | Average speed: total distance divided by the bin duration. |
| `time_moving_sec` | Seconds spent moving (frame-to-frame distance > 0.25 mm by default). |
| `time_stationary_sec` | Seconds spent stationary (frame-to-frame distance <= 0.25 mm). |
| `n_samples_total` | Total number of frames in this bin. |
| `n_samples_valid` | Number of frames where the fish was successfully detected. |
| `coverage_sec` | Total time span covered by valid detections. |
| `n_gaps` | Number of separate stretches where the fish was not detected. |

### Geometry metrics (in `bins_geometry.csv`)

| Column | What it measures |
|--------|-----------------|
| `row` | Grid row of this fish (0--3). |
| `col` | Grid column of this fish (0--6). |
| `cell_type` | "large" (rows 0, 3) or "small" (rows 1, 2). |
| `mean_centroid_dist_mm` | Average distance from the fish to the center of its cell. Higher values mean the fish spent more time away from the middle. |
| `mean_wall_dist_mm` | Average distance from the fish to the nearest wall. Lower values mean the fish tended to stay close to the edges. |
| `frac_outer_edge` | Fraction of time the fish was in the outer edge zone of its cell. The outer edge is defined as the region where wall distance is less than 20% (by default) of the cell's smaller half-dimension. A value of 1.0 means the fish was always near a wall; 0.0 means it was never near a wall. |
| `n_geom_samples` | Number of valid position samples used for these calculations. |

### Social metrics (in `bins_social_large_cells.csv`)

These metrics only apply to fish in **large compartments** (rows 0 and 3). Fish in small compartments (rows 1 and 2) have blank values.

| Column | What it measures |
|--------|-----------------|
| `time_close_half_sec` | Seconds spent in the half of the cell that is closer to the middle rows. For row 0 fish, this is the bottom half (toward row 1). For row 3 fish, this is the top half (toward row 2). Measures whether the fish approaches the boundary shared with the small-compartment fish. |
| `dist_close_half_mm` | Distance traveled while in the close half. |
| `time_near_social_sec` | Seconds spent within 5 mm of the wall facing the middle rows. This is a narrow band right at the shared boundary -- the fish is essentially as close as it can get to its neighbor in the small compartment. |
| `dist_near_social_mm` | Distance traveled while in the social band. |

### Pairwise middle metrics (in `bins_pairwise_middle.csv`)

| Column | What it measures |
|--------|-----------------|
| `middle_fish_id` | The ID of the paired fish in the adjacent small compartment (same column). Row 0 fish are paired with row 1; row 3 fish are paired with row 2. Fish in small compartments have no pairing (blank). |
| `mean_d_middle_euclid_mm` | Average Euclidean distance (mm) between the focal fish and its paired middle-row fish, computed in a shared column coordinate system. This accounts for both horizontal (x) and vertical (cross-cell) separation. |
| `n_valid_pair` | Number of frames where both the focal fish and the paired fish were detected. |

### Per-frame matrix statistics (in `per_frame_*.csv`)

Each file has 28 rows (fish 1--28) and one column per frame. The ten statistics are:

| File | What each cell contains |
|------|------------------------|
| `per_frame_dist_cell_center_mm.csv` | Distance (mm) from the fish to the center of its cell. |
| `per_frame_wall_distance_mm.csv` | Distance (mm) from the fish to the nearest cell wall. |
| `per_frame_in_outer_edge.csv` | 1 if the fish is in the outer edge zone, 0 if not. |
| `per_frame_in_close_half.csv` | 1 if the fish is in the close half (large cells only), 0 if not, blank for small cells. |
| `per_frame_in_near_social_band.csv` | 1 if the fish is within 5 mm of the social wall (large cells only), 0 if not, blank for small cells. |
| `per_frame_step_distance_mm.csv` | Distance (mm) the fish moved since the previous frame. |
| `per_frame_dt_sec.csv` | Time elapsed (seconds) since the previous frame. |
| `per_frame_instant_speed_mm_per_s.csv` | Instantaneous speed: step distance divided by time elapsed. |
| `per_frame_is_moving.csv` | 1 if the fish moved more than the movement threshold (0.25 mm), 0 if stationary. |
| `per_frame_d_middle_euclid_mm.csv` | Euclidean distance (mm) to the paired middle-row fish (large cells only), blank for small cells and unpaired fish. |

## Large vs. Small Cell Distinction

The physical plate has two types of compartments:

- **Large cells** (rows 0 and 3, 80 mm tall): these are the main experimental compartments. All metrics are computed for these fish, including the social proximity metrics that measure approach behavior toward the adjacent small-compartment fish.

- **Small cells** (rows 1 and 2, 18 mm tall): these are narrow end compartments. Locomotion and basic geometry metrics are computed, but social metrics (close half, social band, pairwise middle distance) are left blank because these cells are too small for those measurements to be meaningful, and these fish do not have a "middle" neighbor to measure distance to.
