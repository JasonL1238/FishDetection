# fishdetect

A simple computer vision package for fish tracking and background subtraction, designed for basic fish detection in video frames.

## Features

- **Simple Fish Tracking**: Basic tracking of fish in video frames using background subtraction
- **Background Subtraction**: Create background models and apply background subtraction
- **Frame Processing**: Process video frames and save results
- **Visualization**: Generate annotated images showing tracked fish with shapes and centroids

## Quick Start

```bash
# Install dependencies
make setup

# Run simple tracking test on first 30 frames
python src/fishdetect/simple_tracking_test.py

# Apply background subtraction to 30 frames
python apply_bg_subtraction_30_frames.py
```

## Usage

### Simple Tracking Test

The `simple_tracking_test.py` script:
- Processes the first 30 frames of a video
- Creates a background model from the first 10 frames
- Tracks fish using background subtraction
- Draws shape outlines and centroids
- Saves tracking data and visualization images

### Background Subtraction

The `apply_bg_subtraction_30_frames.py` script:
- Creates a background model using specified frame indices
- Applies background subtraction to the first 30 frames
- Saves original and background-subtracted frames

## Configuration

Both scripts use hardcoded paths that can be modified:
- Video path: `input/Clutch1_20250804_122715.mp4`
- Output directory: `Results/SimpleTracking/` or `output/bg_subtracted_30_frames/`
- Number of fish: 28 (4x7 layout)
- Frame dimensions: 544x512 pixels

## Algorithm

The simple tracking algorithm uses:

1. **Background Model**: Create background from median of first 10 frames
2. **Background Subtraction**: Calculate absolute difference between current frame and background
3. **Thresholding**: Apply binary threshold to create mask
4. **Morphological Operations**: Clean up the mask
5. **Contour Detection**: Find connected components
6. **Shape Analysis**: Filter by area and aspect ratio
7. **Tracking**: Assign fish numbers based on position in 4x7 grid

## Output

- **Tracking Images**: Overlay images with fish shapes and centroids
- **Tracking Data**: NumPy arrays containing centroid coordinates
- **Background Subtracted Frames**: Binary masks showing detected motion
