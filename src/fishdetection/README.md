# fishdetection -- Core Detection Library

This package contains the detection engine that processes a video and finds each fish in every frame. It does **not** use deep learning or neural networks -- it uses classical image processing techniques (background subtraction, color filtering, and blob detection).

## How the Modules Fit Together

The detection pipeline chains four components in order:

```
Video frame
    │
    ▼
BackgroundSubtractor ──► Produces a binary mask (white = foreground, black = background)
    │
    ▼
HSVMasker ──► Filters the mask to keep only bright objects (fish)
    │
    ▼
grid.py ──► Divides the frame into 28 cells, finds the biggest blob in each
    │
    ▼
CustomGridPipeline ──► Orchestrates the above, converts to mm, writes outputs
```

## Module Guide

### `pipeline.py` -- CustomGridPipeline

The main orchestrator. When you call `pipeline.process(video_path, output_dir, ...)`:

1. Creates a background model from sampled frames.
2. Loops through every frame of the video.
3. For each frame, subtracts the background, splits into 28 grid cells, and finds the largest bright blob in each cell.
4. Converts the detected pixel position to millimeters using known physical cell dimensions.
5. Draws an annotated version of the frame (grid lines, contours, centroids, fish IDs).
6. Writes the annotated video, a positions CSV, a summary text file, and periodic PNG snapshots.

### `background_subtractor.py` -- BackgroundSubtractor

Builds a "background image" of the empty tank by taking the median of 12 frames sampled across the video (frames 0, 2000, 4000, ..., 22000). Because fish move around, the median captures the static background at each pixel while the fish average out.

For each video frame, the subtractor:
1. Applies a slight Gaussian blur to reduce noise.
2. Computes the absolute pixel-by-pixel difference from the background.
3. Marks pixels with a difference above the threshold (default: 15) as foreground.
4. Applies morphological closing (fills small gaps in the detected shapes so fish blobs stay connected).

The result is a black-and-white image where white regions are things that differ from the background -- mostly fish.

### `hsv_masker.py` -- HSVMasker

Takes the foreground image from background subtraction and filters it by brightness using the HSV (Hue, Saturation, Value) color space. Fish appear as bright objects, so the filter keeps pixels with a brightness value of 100 or higher (out of 255). This removes dark shadows and artifacts.

After filtering, it:
1. Applies morphological open/close to clean up noise.
2. Labels connected groups of white pixels as distinct objects.
3. Removes objects smaller than a minimum size.
4. Extracts the outline (contour) and center point (centroid) of each remaining object.

### `grid.py` -- Grid Layout and Per-Cell Detection

Defines the 7-column by 4-row grid that maps to the physical plate layout. The grid boundaries are computed from the video frame dimensions, with a small adjustment that compresses the middle two rows slightly toward the frame center (to match how they appear on camera).

Key functions:
- **`get_grid_cells()`** -- returns the pixel boundaries (x_start, y_start, x_end, y_end) for all 28 cells.
- **`get_cell_mm()`** -- returns the physical dimensions (width_mm, height_mm) of a given cell.
- **`find_largest_blob_in_cell()`** -- crops the frame to a single cell, runs HSV masking on it, and returns the largest detected blob and its center point.

### `base_pipeline.py` -- BasePipeline

An abstract base class that stores shared configuration. All parameters have defaults:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `fps` | 20 | Video frame rate |
| `threshold` | 15 | Minimum pixel difference to count as foreground |
| `morph_kernel_size` | (5, 5) | Size of the filter used to fill gaps in detected shapes |
| `hsv_lower` | (0, 0, 100) | Minimum HSV values for brightness filter |
| `hsv_upper` | (180, 255, 255) | Maximum HSV values for brightness filter |
| `min_object_size` | 10 | Smallest blob (in pixels) to keep |

### `base_masker.py` -- BaseMasker

An abstract interface that defines the methods any masker must implement (`process_frame`, `process_video`). Currently only HSVMasker implements this.

## How Coordinates Work

Positions go through three stages:

1. **Pixel coordinates** -- the raw (row, column) location of a fish's center in the video frame.
2. **Normalized coordinates** -- the pixel position is converted to a fraction relative to the cell center. A value of 0 means dead center; positive x is right of center, positive y is above center.
3. **Millimeter coordinates** -- the normalized value is multiplied by the physical cell dimensions (30 mm wide; 80 mm or 18 mm tall depending on the row). This is what gets written to the positions CSV.

The origin (0, 0) for each fish is the **center of its own compartment**. Positive x points right, positive y points up (toward the top of the frame).
