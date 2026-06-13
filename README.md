# Fish Detection

Automated fish tracking for multi-compartment lab videos. Given a video of fish housed in a 7-column by 4-row grid (28 individual compartments), this software finds each fish in every frame and produces data files describing their movement, position, and social proximity.

## The Physical Setup

The tracking grid matches the physical plate layout used in the lab:

- **28 compartments** arranged in 7 columns and 4 rows, each holding one fish.
- Every column is **30 mm wide**.
- The **outer rows** (rows 0 and 3) are **large compartments**: 80 mm tall.
- The **middle rows** (rows 1 and 2) are **small compartments**: 18 mm tall.

Fish are numbered 1--28 in row-major order (left to right, top to bottom):

```
         Col 0   Col 1   Col 2   Col 3   Col 4   Col 5   Col 6
        ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┐
Row 0   │ Fish 1│ Fish 2│ Fish 3│ Fish 4│ Fish 5│ Fish 6│ Fish 7│  80 mm tall
(large) │       │       │       │       │       │       │       │  (large)
        ├───────┼───────┼───────┼───────┼───────┼───────┼───────┤
Row 1   │Fish 8 │Fish 9 │Fish10 │Fish11 │Fish12 │Fish13 │Fish14 │  18 mm tall
(small) ├───────┼───────┼───────┼───────┼───────┼───────┼───────┤  (small)
Row 2   │Fish15 │Fish16 │Fish17 │Fish18 │Fish19 │Fish20 │Fish21 │  18 mm tall
(small) ├───────┼───────┼───────┼───────┼───────┼───────┼───────┤  (small)
Row 3   │Fish22 │Fish23 │Fish24 │Fish25 │Fish26 │Fish27 │Fish28 │  80 mm tall
(large) │       │       │       │       │       │       │       │  (large)
        └───────┴───────┴───────┴───────┴───────┴───────┴───────┘
                          all columns 30 mm wide
```

Positions are recorded in **millimeters** relative to the center of each compartment. An x value of 0, y value of 0 means the fish is exactly in the center of its cell.

## How Detection Works

The pipeline processes every frame of the video through four steps:

1. **Build a background image.** The software samples frames spread across the video and takes the median pixel value at each location. This produces a "clean" image of the empty tank -- what the scene looks like with no fish movement.

2. **Subtract the background.** Each video frame is compared against the background image. Pixels that differ significantly are marked as foreground (likely a fish). Small noise is cleaned up by filling tiny gaps in the detected shapes.

3. **Filter by brightness.** The foreground image is converted to HSV color space and filtered to keep only bright objects (value >= 100). This removes shadows and artifacts that are not fish.

4. **Detect one fish per compartment.** The frame is divided into the 28-cell grid. Within each cell, the software finds the single largest bright region (blob) and records its center point as that fish's position. The pixel position is converted to millimeters using the known physical dimensions of each compartment.

## How to Run It

### Installation

Requires Python 3.9 or newer.

```bash
pip install -e .
```

### Step 1: Run the detection pipeline

Place your video file(s) in `data/input/videos/`, then run:

```bash
python run.py data/input/videos/YourVideo.mp4
```

Optional arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `--output <dir>` | Where to save results | `data/output/<video_name>/` |
| `--fps <number>` | Frames per second to process | 20 |

This processes the entire video and writes results to the output directory.

### Step 2: Analyze the position data

After running the pipeline, analyze the positions CSV to compute movement and behavior metrics:

```bash
python analyze_positions.py --run-dir data/output/YourVideo
```

Optional arguments:

| Flag | Description | Default |
|------|-------------|---------|
| `--csv <path>` | Path to a specific positions CSV | Auto-detected from run dir |
| `--out-dir <dir>` | Where to write analysis files | `<run-dir>/analysis/` |
| `--bin-size-sec <sec>` | Time window for binned summaries | 0.05 (one frame at 20 FPS) |
| `--movement-threshold-mm <mm>` | Minimum distance to count as "moving" | 0.25 |
| `--outer-edge-alpha <frac>` | Fraction defining the outer edge of a cell | 0.2 |

## What You Get (Outputs)

After both steps, the output directory contains:

### From the detection pipeline (`run.py`)

| File | What it is |
|------|------------|
| `*_positions.csv` | The core data file. One row per video frame, with columns for time, frame number, and x/y position (in mm) for each of the 28 fish. Empty cells mean the fish was not detected in that frame. |
| `*.mp4` | An annotated copy of the video with grid lines drawn on top, green outlines around detected fish, red dots at fish centers, and fish numbers labeled. Useful for visually verifying that tracking is working. |
| `*_summary.txt` | A text report with detection statistics: how many fish were found per frame on average, what percentage of frames had all 28 fish detected, and the pipeline settings used. |
| `background_model.npy` | The computed background image (saved for reuse). |
| `frames/` | PNG snapshots of annotated frames taken every 5 frames, for quick visual review without opening the video. |

### From the analysis step (`analyze_positions.py`)

All analysis files are written to the `analysis/` subdirectory:

| File | What it contains |
|------|------------------|
| **`bins_locomotion.csv`** | Movement data for every fish in every time bin: total distance traveled (mm), speed (mm/sec), time spent moving vs. stationary, and data quality indicators (valid sample count, gap count). |
| **`bins_geometry.csv`** | Position data for every fish in every time bin: how far the fish is from the center of its cell, how far from the nearest wall, and what fraction of time it spent near the outer edges. Includes the fish's row, column, and cell type (large or small). |
| **`bins_social_large_cells.csv`** | Social proximity data (large compartments only, rows 0 and 3). Measures how much time each fish spent in the half of its cell closest to the middle rows ("close half") and within a 5 mm band right at the shared wall ("social band"). This captures approach behavior toward neighboring fish in the small compartments. |
| **`bins_pairwise_middle.csv`** | For each fish in a large compartment, the average distance (mm) to its paired fish in the adjacent small compartment (same column). Fish 1 (row 0, col 0) is paired with Fish 8 (row 1, col 0), Fish 22 (row 3, col 0) is paired with Fish 15 (row 2, col 0), etc. |
| **`per_frame_*.csv`** | Ten detailed matrices (one file per statistic), with one row per fish and one column per frame. Statistics include: distance from cell center, wall distance, whether the fish is in the outer edge, close half, or social band, step distance between frames, speed, and pairwise distance to the middle-row fish. |
| **`fish01_xy_over_time.png` ... `fish28_xy_over_time.png`** | Line plots showing each fish's x and y position over the entire video duration. Useful for spotting periods of high activity or stillness. |
| **`distance_summary_binned.csv`** | A combined file with all locomotion and social columns in one CSV (kept for backward compatibility). |
| **`analysis_manifest.json`** | Machine-readable metadata documenting which settings were used for the analysis (bin size, movement threshold, outer-edge fraction) and definitions of computed metrics. |

### Understanding "large" vs. "small" cell metrics

Some metrics only apply to fish in **large compartments** (rows 0 and 3). These are the outer rows where the cells are tall enough (80 mm) to measure meaningful spatial behavior like wall proximity and social approach. In the social CSVs, fish from the small middle rows (rows 1 and 2) will have blank values for social metrics since those cells are only 18 mm tall.

## Project Structure

```
FishDetection/
├── run.py                     # Main entry point: run detection on a video
├── analyze_positions.py       # Analyze movement/behavior from positions CSV
├── analyze.py                 # Per-fish tracking quality report (legacy)
├── pyproject.toml             # Package configuration and dependencies
├── src/
│   └── fishdetection/         # Core detection library
│       ├── pipeline.py        # Main detection loop (CustomGridPipeline)
│       ├── background_subtractor.py  # Background model and subtraction
│       ├── hsv_masker.py      # Brightness filtering and blob detection
│       ├── grid.py            # Grid layout and per-cell detection
│       ├── base_pipeline.py   # Shared pipeline configuration
│       └── base_masker.py     # Masker interface
├── analysis/                  # Post-processing analysis modules
│   ├── positions.py           # Trajectory building, binned metrics, CSV export
│   ├── geometry.py            # Cell layout math (wall distance, column axis)
│   ├── per_frame_matrices.py  # Fish-by-frame statistic matrices
│   └── plots.py               # x/y trajectory plots
├── tests/                     # Unit tests
│   ├── test_analysis_geometry.py
│   └── test_per_frame_matrices.py
└── data/
    ├── input/videos/          # Place input videos here (not tracked in git)
    └── output/                # Pipeline results go here (not tracked in git)
```

## Dependencies

Managed via `pyproject.toml`. Installed automatically by `pip install -e .`:

- **opencv-python** -- video reading, image processing, annotated video writing
- **numpy** -- numerical computation
- **scikit-image** -- connected-component labeling for blob detection
- **matplotlib** -- trajectory plots
- **pims** -- indexed video frame access
