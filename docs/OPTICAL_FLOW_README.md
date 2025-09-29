# Optical Flow Fish Detection

This implementation provides optical flow-based motion detection for fish tracking, based on the tutorial by Isaac Berrios on "Introduction to Motion Detection: Part 2 - Using Optical Flow to Detect Motion".

## Overview

The optical flow implementation detects fish movement by:

1. **Computing Dense Optical Flow** between consecutive frames using Farneback's method
2. **Thresholding Optical Flow** to get motion masks with variable thresholds
3. **Finding Contours** in the motion mask and extracting bounding boxes
4. **Applying Non-Maximal Suppression** to remove overlapping detections

## Key Features

- **Variable Thresholding**: Uses linear thresholding from top to bottom of image to account for perspective
- **Morphological Operations**: Applies erosion, opening, and closing to clean up motion masks
- **Flow Angle Filtering**: Optional filtering based on flow angle consistency
- **Non-Maximum Suppression**: Removes overlapping detections using IoU thresholding
- **Integration with Background Subtraction**: Works with preprocessed background subtracted frames

## Files

### Core Implementation
- `src/masking_methods/optical_flow/optical_flow_masker.py` - Main optical flow masker class
- `optical_flow_video_processor.py` - Complete video processing pipeline
- `test_optical_flow.py` - Test script for optical flow on preprocessed frames
- `run_optical_flow_pipeline.py` - Main pipeline runner with all steps

### Integration
- Works with existing background subtraction from `apply_bg_subtraction_30_frames.py`
- Uses image preprocessing from `src/image_pre/preprocessor.py`
- Compatible with existing fish tracking infrastructure

## Usage

### Quick Test
```bash
python run_optical_flow_pipeline.py --quick-test
```

### Full Pipeline
```bash
python run_optical_flow_pipeline.py --frames 30 --motion-thresh 1.0 --bbox-thresh 400
```

### Custom Parameters
```bash
python run_optical_flow_pipeline.py \
    --frames 50 \
    --motion-thresh 0.8 \
    --bbox-thresh 300 \
    --bg-threshold 20
```

## Parameters

### Optical Flow Parameters
- `motion_thresh`: Minimum flow threshold for motion detection (default: 1.0)
- `bbox_thresh`: Minimum bounding box area threshold (default: 400)
- `nms_thresh`: IoU threshold for non-maximum suppression (default: 0.1)
- `use_variable_threshold`: Use variable thresholding based on image position (default: True)
- `min_thresh`: Minimum threshold for variable thresholding (default: 0.3)
- `max_thresh`: Maximum threshold for variable thresholding (default: 1.0)

### Farneback Parameters
- `pyr_scale`: Image scale for pyramid building (default: 0.75)
- `levels`: Number of pyramid levels (default: 3)
- `winsize`: Average window size (default: 5)
- `iterations`: Number of iterations per pyramid level (default: 3)
- `poly_n`: Size of pixel neighborhood for polynomial expansion (default: 10)
- `poly_sigma`: Standard deviation for Gaussian smoothing (default: 1.2)

## Algorithm Details

### 1. Dense Optical Flow Computation
```python
flow = cv2.calcOpticalFlowFarneback(
    gray1, gray2, None,
    pyr_scale=0.75,
    levels=3,
    winsize=5,
    iterations=3,
    poly_n=10,
    poly_sigma=1.2,
    flags=0
)
```

### 2. Variable Thresholding
The implementation uses a linear threshold that increases from top to bottom of the image:
```python
motion_thresh_array = np.c_[np.linspace(min_thresh, max_thresh, height)].repeat(width, axis=-1)
```

This accounts for the fact that objects at the top of the image (further away) have smaller pixel displacements than objects at the bottom (closer).

### 3. Motion Mask Creation
```python
motion_mask = np.uint8(flow_mag > motion_thresh) * 255
motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
```

### 4. Detection and NMS
- Find contours in the motion mask
- Extract bounding boxes with area filtering
- Apply non-maximum suppression to remove overlaps

## Output

The pipeline generates:

1. **Visualization Images**: Side-by-side comparisons showing:
   - Original frame with detections
   - Background subtracted frame
   - Preprocessed frame
   - Motion mask

2. **Detection Data**: Text files with detection coordinates and areas

3. **Summary Report**: Processing statistics and detection counts

## Integration with Existing System

The optical flow implementation integrates seamlessly with the existing fish detection pipeline:

1. **Background Subtraction**: Uses the same background model creation as the original tracking program
2. **Preprocessing**: Applies Gaussian blur and sharpening to background subtracted frames
3. **Detection Format**: Returns detections in the same format as other maskers
4. **Visualization**: Compatible with existing visualization tools

## Performance Considerations

- **Computational Cost**: Optical flow is more computationally expensive than frame differencing
- **Memory Usage**: Stores previous frame for flow computation
- **Parameter Tuning**: May require adjustment of thresholds for different video conditions
- **Real-time**: Can be optimized for real-time processing with appropriate parameter tuning

## Troubleshooting

### Common Issues

1. **No Detections**: Try lowering `motion_thresh` or `bbox_thresh`
2. **Too Many Detections**: Increase `motion_thresh` or `bbox_thresh`
3. **Poor Quality Masks**: Adjust morphological operation parameters
4. **Memory Issues**: Process fewer frames at a time

### Parameter Tuning

- **Motion Threshold**: Start with 1.0, adjust based on fish movement speed
- **Bbox Threshold**: Start with 400, adjust based on expected fish size
- **Variable Threshold**: Enable for better performance with perspective effects
- **NMS Threshold**: Lower values (0.1) for stricter overlap removal

## References

- Isaac Berrios, "Introduction to Motion Detection: Part 2 - Using Optical Flow to Detect Motion", Medium, 2023
- OpenCV Documentation: Optical Flow
- Farneback, G., "Two-Frame Motion Estimation Based on Polynomial Expansion", 2003
