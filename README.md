# Fish Detection Project

A comprehensive computer vision package for fish tracking and detection using multiple approaches including background subtraction, optical flow, and YOLO-based detection.

## Project Structure

```
FishDetection/
├── src/                          # Core source code
│   ├── fishdetect/              # Main fish detection modules
│   ├── image_pre/               # Image preprocessing utilities
│   └── tracking_methods/        # Different tracking algorithms
│       ├── canny_edge_detection/
│       ├── common/              # Shared utilities
│       ├── optical_flow/
│       └── yolo_tracking/
├── scripts/                     # Executable scripts
│   ├── tests/                   # Test scripts for all methods
│   └── analysis/                # Analysis and visualization scripts
├── examples/                    # Demo and example scripts
├── models/                      # Pre-trained model files
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   └── yolov8m.pt
├── data/                        # Data directories
│   ├── input/                   # Input data
│   │   ├── videos/              # Video files
│   │   └── frames/              # Preprocessed frames
│   └── output/                  # Output results
│       ├── simple_tracking/     # Background subtraction results
│       ├── optical_flow/        # Optical flow results
│       ├── yolo_tracking/       # YOLO detection results
│       └── fish_analysis/       # Analysis visualizations
├── docs/                        # Documentation
├── notebooks/                   # Jupyter notebooks (if needed)
├── pyproject.toml              # Project configuration
└── Makefile                    # Build commands
```

## Features

### Detection Methods
- **Background Subtraction**: Traditional computer vision approach using frame differencing
- **Optical Flow**: Motion-based detection using Lucas-Kanade optical flow
- **YOLO Detection**: Deep learning-based object detection using YOLOv8
- **Hybrid Approaches**: Combining multiple methods for improved accuracy

### Core Capabilities
- **Fish Tracking**: Track individual fish across video frames
- **Multi-fish Detection**: Handle multiple fish in the same frame
- **Background Modeling**: Create and update background models
- **Visualization**: Generate annotated images with detection results
- **Performance Analysis**: Comprehensive testing and benchmarking

## Quick Start

### Installation
```bash
# Install dependencies
make setup

# Or manually
pip install -e .
```

### Running Tests
```bash
# Test background subtraction approach
python scripts/tests/test_fish_tracker.py

# Test YOLO detection
python scripts/tests/test_yolo_fish_detection.py

# Test optical flow
python scripts/tests/test_optical_flow.py
```

### Running Examples
```bash
# Run optical flow pipeline
python examples/run_optical_flow_pipeline.py

# Demo optical flow
python examples/demo_optical_flow.py
```

## Usage

### Background Subtraction Method
The simplest approach using traditional computer vision:
- Creates background model from initial frames
- Detects motion using frame differencing
- Applies morphological operations to clean up detections
- Tracks fish based on position in expected grid layout

### YOLO Detection Method
Deep learning approach using pre-trained models:
- Uses YOLOv8 models (nano, small, medium)
- Detects fish as objects with bounding boxes
- Provides confidence scores for detections
- Can be fine-tuned for specific fish types

### Optical Flow Method
Motion-based detection:
- Tracks pixel movement between frames
- Identifies regions with significant motion
- Good for detecting moving fish
- Less sensitive to lighting changes

## Configuration

### Video Input
Place your video files in `data/input/videos/`:
- Supported formats: MP4, AVI, MOV
- Default test video: `Clutch1_20250804_122715.mp4`

### Output Results
Results are saved to `data/output/` organized by method:
- `simple_tracking/`: Background subtraction results
- `yolo_tracking/`: YOLO detection results  
- `optical_flow/`: Optical flow results
- `fish_analysis/`: Analysis visualizations

### Model Files
Pre-trained YOLO models are stored in `models/`:
- `yolov8n.pt`: Nano model (fastest, least accurate)
- `yolov8s.pt`: Small model (balanced)
- `yolov8m.pt`: Medium model (most accurate, slower)

## Algorithm Details

### Background Subtraction
1. **Background Model**: Create median background from first 10 frames
2. **Frame Differencing**: Calculate absolute difference with current frame
3. **Thresholding**: Apply binary threshold to create motion mask
4. **Morphological Operations**: Clean up noise and fill gaps
5. **Contour Detection**: Find connected components
6. **Shape Analysis**: Filter by area and aspect ratio
7. **Grid Assignment**: Assign fish numbers based on 4x7 grid position

### YOLO Detection
1. **Model Loading**: Load pre-trained YOLOv8 model
2. **Frame Processing**: Resize and normalize input frames
3. **Object Detection**: Run inference to get bounding boxes
4. **Confidence Filtering**: Filter detections by confidence threshold
5. **Non-Maximum Suppression**: Remove overlapping detections
6. **Tracking**: Associate detections across frames

### Optical Flow
1. **Feature Detection**: Find good features to track
2. **Flow Calculation**: Compute optical flow between frames
3. **Motion Analysis**: Identify regions with significant movement
4. **Mask Generation**: Create binary mask of moving regions
5. **Contour Detection**: Find connected motion regions

## Performance

### Expected Results
- **Fish Count**: 28 fish per frame (4x7 grid layout)
- **Frame Size**: 544x512 pixels
- **Processing Speed**: Varies by method
  - Background Subtraction: ~30 FPS
  - YOLO Nano: ~15 FPS
  - YOLO Medium: ~5 FPS
  - Optical Flow: ~10 FPS

### Detection Accuracy
- **Background Subtraction**: Good for static backgrounds
- **YOLO**: Best for general object detection
- **Optical Flow**: Good for moving objects
- **Hybrid**: Combines strengths of multiple methods

## Development

### Adding New Methods
1. Create new module in `src/tracking_methods/`
2. Implement base interface from `common/base_masker.py`
3. Add to factory in `masker_factory.py`
4. Create test script in `scripts/tests/`

### Testing
All test scripts are in `scripts/tests/`:
- `test_fish_tracker.py`: Background subtraction tests
- `test_yolo_fish_detection.py`: YOLO detection tests
- `test_optical_flow.py`: Optical flow tests
- `test_yolo_tracking.py`: YOLO tracking tests

### Analysis
Analysis scripts in `scripts/analysis/`:
- `analyze_fish_detection.py`: Detection analysis
- `visualize_yolo_results.py`: YOLO result visualization
- `process_preprocessed_canny.py`: Canny edge processing

## Dependencies

- **OpenCV**: Computer vision operations
- **NumPy**: Numerical computations
- **Ultralytics**: YOLO model support
- **Matplotlib**: Visualization
- **scikit-image**: Image processing
- **tqdm**: Progress bars

## License

This project is for research and educational purposes.
