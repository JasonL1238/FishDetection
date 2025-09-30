# Tracking Methods

This directory contains all the different fish tracking algorithms organized by method type.

## Directory Structure

```
tracking_methods/
├── background_subtraction/          # Background subtraction tracking
│   ├── __init__.py
│   ├── background_subtraction_masker.py
│   └── tracking_program_28fish_bg_subtraction.py
├── canny_edge_detection/            # Canny edge detection tracking
│   ├── __init__.py
│   └── canny_masker.py
├── optical_flow/                    # Optical flow tracking
│   ├── __init__.py
│   └── optical_flow_masker.py
├── traditional_tracking/            # Traditional tracking methods
│   ├── __init__.py
│   └── tracking_program_28fish_traditional.py
├── yolo_tracking/                   # YOLO-based tracking
│   ├── __init__.py
│   ├── fish_tracker.py
│   └── yolo_tracker.py
├── common/                          # Shared utilities
│   ├── __init__.py
│   ├── base_masker.py
│   └── utils.py
├── config.py                        # Configuration settings
├── masker_factory.py               # Factory for creating maskers
└── README.md                       # This file
```

## Available Tracking Methods

### 1. Background Subtraction (NEW)
- **Location**: `background_subtraction/`
- **Description**: Uses custom background subtraction for fish detection
- **Fish Count**: 28 fish (4 rows × 7 columns)
- **Key Features**:
  - Integrates with existing `BackgroundSubtractor` class
  - Configurable threshold and blur parameters
  - Automatic background model creation
  - Full analysis pipeline included

### 2. Traditional Tracking
- **Location**: `traditional_tracking/`
- **Description**: Original tracking method updated for 28 fish
- **Fish Count**: 28 fish (4 rows × 7 columns)
- **Key Features**:
  - Updated from original 14-fish version
  - Same analysis pipeline as original
  - Compatible with existing data formats

### 3. Optical Flow
- **Location**: `optical_flow/`
- **Description**: Uses optical flow for motion detection
- **Fish Count**: Configurable
- **Key Features**:
  - Motion-based detection
  - Good for tracking moving objects

### 4. Canny Edge Detection
- **Location**: `canny_edge_detection/`
- **Description**: Uses Canny edge detection for fish boundaries
- **Fish Count**: Configurable
- **Key Features**:
  - Edge-based detection
  - Good for well-defined fish shapes

### 5. YOLO Tracking
- **Location**: `yolo_tracking/`
- **Description**: Uses YOLO object detection for fish identification
- **Fish Count**: Configurable
- **Key Features**:
  - Deep learning-based detection
  - High accuracy for complex scenarios

## Usage

### Using the Factory Pattern

```python
from tracking_methods.masker_factory import MaskerFactory

# Create a background subtraction masker
bg_masker = MaskerFactory.create_masker('background_subtraction')

# Create a traditional tracking masker
traditional_masker = MaskerFactory.create_masker('traditional_tracking')

# Get available methods
methods = MaskerFactory.get_available_methods()
print(methods)  # ['optical_flow', 'canny', 'yolo', 'background_subtraction']
```

### Running Tracking Programs

#### Background Subtraction Tracking (28 fish)
```python
from tracking_methods.background_subtraction.tracking_program_28fish_bg_subtraction import main

# Run the tracking program
main()
```

#### Traditional Tracking (28 fish)
```python
from tracking_methods.traditional_tracking.tracking_program_28fish_traditional import main

# Run the tracking program
main()
```

### Using Individual Maskers

```python
from tracking_methods.background_subtraction import BackgroundSubtractionMasker

# Create masker
masker = BackgroundSubtractionMasker(
    threshold=25,
    min_object_size=25
)

# Create background model
masker.create_background_model('path/to/video.mp4')

# Process a single frame
processed_frame = masker.process_frame(frame)

# Detect fish centroids
centroids = masker.detect_fish_centroids(frame)
```

## Configuration

### Fish Layout for 28 Fish
- **Rows**: 4
- **Columns**: 7
- **Total Wells**: 28
- **Layout**: 4×7 grid

### Glass Row Definitions (for 28 fish)
- `glassrow1`: 42
- `glassrow2`: 136 (544/4 = 136)
- `glassrow3`: 230 (136 + 94)
- `glassrow4`: 324 (230 + 94)
- `glassrow5`: 418 (324 + 94)
- `glassrow6`: 512 (418 + 94)

## Demo Script

A demo script is available to test the different tracking methods:

```bash
python examples/demo_background_subtraction_tracking.py
```

This script allows you to choose between:
1. Background Subtraction tracking (NEW)
2. Traditional tracking (Updated for 28 fish)

## Output Files

All tracking methods save results to:
- `data/output/tracking_results/` - Raw tracking data
- `data/output/gene_analysis/` - Analysis results
- `data/output/tracked_images/` - Visualization images

## Dependencies

- OpenCV (`cv2`)
- NumPy
- scikit-image
- pandas
- matplotlib
- seaborn
- scipy
- pims (for video handling)
- tqdm (for progress bars)

