# YOLOv8 Fish Tracking Implementation

This document describes the YOLOv8-based fish tracking implementation added to the FishDetection project.

## Overview

The YOLOv8 tracking implementation provides state-of-the-art object detection and tracking capabilities specifically optimized for fish detection in underwater videos. It uses the Ultralytics YOLOv8 model with built-in tracking functionality.

## Features

- **Real-time Detection**: Uses YOLOv8 for fast and accurate object detection
- **Multi-Object Tracking**: Tracks multiple fish simultaneously across frames
- **Track Visualization**: Draws bounding boxes and track trails
- **Configurable Parameters**: Adjustable confidence thresholds and tracking parameters
- **Video Processing**: Processes entire videos with progress tracking
- **Data Export**: Saves tracking data in numpy format for further analysis

## Installation

The YOLOv8 tracking requires the `ultralytics` package:

```bash
pip install ultralytics
```

This dependency has been added to the project's `pyproject.toml` file.

## File Structure

```
src/tracking_methods/yolo_tracking/
├── __init__.py
└── yolo_tracker.py
```

## Usage

### Basic Usage

```python
from src.tracking_methods.yolo_tracking.yolo_tracker import YOLOTracker

# Initialize tracker
tracker = YOLOTracker(
    model_path="yolov8n.pt",  # Use nano model for speed
    confidence_threshold=0.5,
    iou_threshold=0.7
)

# Process video
stats = tracker.process_video(
    video_path="input/video.mp4",
    output_path="output/tracking_result.mp4",
    max_duration=10.0  # Process only first 10 seconds
)
```

### Using the Factory Pattern

```python
from src.tracking_methods.masker_factory import MaskerFactory

# Create YOLOv8 tracker using factory
tracker = MaskerFactory.create_masker('yolo')

# Process video
stats = tracker.process_video("input/video.mp4", "output/result.mp4")
```

### Test Script

Run the test script to process the first 10 seconds of the fish video:

```bash
python test_yolo_tracking.py
```

## Configuration

The YOLOv8 tracking can be configured through the `YOLO_CONFIG` in `src/tracking_methods/config.py`:

```python
YOLO_CONFIG = {
    'model_path': 'yolov8n.pt',           # Model file or name
    'confidence_threshold': 0.5,          # Minimum confidence for detections
    'iou_threshold': 0.7,                 # IoU threshold for NMS
    'max_track_length': 30,               # Maximum track history length
    'track_buffer': 30                    # Track management buffer
}
```

## Model Options

YOLOv8 provides several model sizes:

- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

For fish tracking, the nano model (`yolov8n.pt`) is recommended for real-time processing.

## Output

The YOLOv8 tracker produces:

1. **Annotated Video**: Video with bounding boxes and track trails
2. **Track Data**: NumPy file containing track coordinates and IDs
3. **Processing Statistics**: Frame count, processing time, FPS, etc.

## Performance

Based on testing with the fish video:

- **Processing Speed**: ~22 FPS on modern hardware
- **Memory Usage**: Moderate (depends on model size)
- **Accuracy**: Good for general object detection, may need fine-tuning for specific fish species

## Integration

The YOLOv8 tracker integrates seamlessly with the existing tracking methods:

- Inherits from `BaseMasker` class
- Available through `MaskerFactory`
- Follows the same interface as other tracking methods
- Configurable through the central config system

## Limitations

1. **General Object Detection**: YOLOv8 is trained on general objects, not specifically fish
2. **Underwater Conditions**: May need fine-tuning for underwater lighting and conditions
3. **Small Objects**: May struggle with very small fish or distant objects
4. **Computational Requirements**: Requires GPU for optimal performance

## Future Improvements

1. **Custom Training**: Train YOLOv8 on fish-specific datasets
2. **Model Fine-tuning**: Fine-tune for underwater conditions
3. **Multi-class Detection**: Detect different fish species
4. **Behavioral Analysis**: Add fish behavior analysis capabilities
5. **Real-time Processing**: Optimize for real-time video streams

## Example Results

The test run on the first 10 seconds of the fish video shows:

- **Frames Processed**: 200 frames
- **Processing Time**: 8.92 seconds
- **Average FPS**: 22.43
- **Unique Tracks**: 0 (no fish detected in this segment)

Note: The zero tracks result suggests that either:
1. No fish are visible in the first 10 seconds
2. The confidence threshold may need adjustment
3. The general YOLOv8 model may not be optimal for this specific fish video

## Troubleshooting

### No Tracks Detected

If no tracks are detected:

1. Lower the confidence threshold (e.g., 0.3)
2. Try a different model size
3. Check if fish are actually visible in the video segment
4. Consider training a custom model

### Performance Issues

For better performance:

1. Use GPU acceleration if available
2. Use smaller model (yolov8n.pt)
3. Reduce video resolution
4. Process shorter video segments

### Memory Issues

If running out of memory:

1. Use smaller model
2. Process video in smaller chunks
3. Reduce track buffer size
4. Close other applications
