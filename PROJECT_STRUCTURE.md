# Project Structure Overview

This document provides a quick reference for the organized project structure.

## Directory Organization

### Core Code (`src/`)
- `fishdetect/` - Main fish detection modules
- `image_pre/` - Image preprocessing utilities  
- `tracking_methods/` - Different tracking algorithms
  - `canny_edge_detection/` - Edge-based detection
  - `common/` - Shared utilities and base classes
  - `optical_flow/` - Motion-based detection
  - `yolo_tracking/` - Deep learning detection

### Scripts (`scripts/`)
- `tests/` - All test scripts for different methods
- `analysis/` - Analysis and visualization scripts

### Examples (`examples/`)
- Demo scripts and pipeline examples
- Run these to see the system in action

### Data (`data/`)
- `input/videos/` - Input video files
- `input/frames/` - Preprocessed frame data
- `output/` - Results organized by method
  - `simple_tracking/` - Background subtraction results
  - `optical_flow/` - Optical flow results
  - `yolo_tracking/` - YOLO detection results
  - `fish_analysis/` - Analysis visualizations

### Models (`models/`)
- Pre-trained YOLO model files (.pt)

### Documentation (`docs/`)
- README files and documentation

## Key Files

### Test Scripts
- `scripts/tests/test_fish_tracker.py` - Background subtraction test
- `scripts/tests/test_yolo_fish_detection.py` - YOLO detection test
- `scripts/tests/test_optical_flow.py` - Optical flow test
- `scripts/tests/test_yolo_tracking.py` - YOLO tracking test

### Example Scripts
- `examples/run_optical_flow_pipeline.py` - Full optical flow pipeline
- `examples/demo_optical_flow.py` - Optical flow demo
- `examples/optical_flow_video_processor.py` - Video processing

### Analysis Scripts
- `scripts/analysis/analyze_fish_detection.py` - Detection analysis
- `scripts/analysis/visualize_yolo_results.py` - YOLO visualization
- `scripts/analysis/process_preprocessed_canny.py` - Canny processing

## Quick Commands

```bash
# Run tests
python scripts/tests/test_fish_tracker.py
python scripts/tests/test_yolo_fish_detection.py
python scripts/tests/test_optical_flow.py

# Run examples
python examples/run_optical_flow_pipeline.py
python examples/demo_optical_flow.py

# Run analysis
python scripts/analysis/analyze_fish_detection.py
```

## Path Updates

All scripts have been updated to use the new organized structure:
- Input videos: `data/input/videos/`
- Output results: `data/output/{method}/`
- Model files: `models/`
- Source code: `src/`
