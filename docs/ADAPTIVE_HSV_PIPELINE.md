# Adaptive HSV Pipeline Flow

This document traces the data flow through the adaptive HSV analysis pipeline with no tolerance.

## Pipeline Overview

```
Input Video → Background Model Creation → Frame Processing → HSV Masking → Adaptive Adjustment → Output
```

## File Flow

### 1. Entry Point
**File**: `examples/adaptive_hsv_analysis_no_tolerance.py`
- **Function**: `main()`
- **Purpose**: Entry point that sets up paths and calls the main processing function
- **Output**: Calls `process_adaptive_hsv_analysis()`

### 2. Video Loading & Background Model
**Files**: 
- `examples/adaptive_hsv_analysis_no_tolerance.py` (lines 80-120)
- `src/processing/tracking_program_background_subtractor.py`

**Flow**:
```
Video Path → pims.PyAVReaderIndexed (load video)
           ↓
Select frame indices: [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]
           ↓
Extract grayscale frames from each index
           ↓
Calculate median across all frames → Background Model
           ↓
Apply Gaussian blur (kernel: 3x3, sigma: 0)
           ↓
Store background_model
```

### 3. Frame-by-Frame Processing Loop
**File**: `examples/adaptive_hsv_analysis_no_tolerance.py` (lines 140-250)

For each frame:
```
Frame[i] → Convert to grayscale (gray_frame)
           ↓
Apply background subtraction (TrackingProgramBackgroundSubtractor)
           ↓
bg_subtracted_frame (binary mask of moving objects)
           ↓
Binary search for optimal min_size to get exactly 28 fish
           ↓
Apply HSV masking with optimal min_size
           ↓
Detect contours and centroids
```

### 4. Background Subtraction
**File**: `src/processing/tracking_program_background_subtractor.py` (lines 95-127)

```
frame (grayscale) → Convert to float64
                    ↓
Apply Gaussian blur
                    ↓
Calculate absolute difference: |background - frame_blurred|
                    ↓
Apply threshold (threshold=25)
                    ↓
Binary mask (uint8)
```

**Class**: `TrackingProgramBackgroundSubtractor`
- **Methods Used**:
  - `create_background_model()` - Creates background model from 12 frames
  - `apply_background_subtraction()` - Subtracts background from a frame

### 5. HSV Masking
**File**: `src/tracking_methods/hsv_masking/hsv_masker.py`

**Flow**:
```
bg_subtracted_frame (grayscale binary mask)
           ↓
Convert grayscale → 3-channel BGR → HSV
           ↓
Create HSV mask: cv2.inRange(hsv_frame, lower_bright, upper_bright)
  lower_bright = [0, 0, 100] (low hue, low sat, high value)
  upper_bright = [180, 255, 255] (any hue, any sat, high value)
           ↓
Apply morphological operations (Open + Close)
           ↓
Label connected components (skimage.measure.label)
           ↓
Remove small objects (skimage.morphology.remove_small_objects)
  min_size = dynamic (from binary search)
           ↓
Re-label cleaned components
           ↓
Return labeled_final
```

**Class**: `HSVMasker`
- **Methods Used**:
  - `process_background_subtracted_frame()` - Process BG-subtracted frame with HSV
  - `detect_fish_contours_and_centroids_bg_subtracted()` - Extract contours and centroids

### 6. Adaptive Binary Search
**File**: `examples/adaptive_hsv_analysis_no_tolerance.py` (lines 23-77)

**Algorithm**:
```
target_count = 28 fish
min_val = 4, max_val = 25
max_iterations = 10

For each iteration:
  test_size = (min_val + max_val) // 2
  Set hsv_masker.min_object_size = test_size
  Get count of detected fish
  If count == 28:
    RETURN test_size, 28, iterations
  If count > 28:
    min_val = test_size + 1 (increase min_size)
  Else:
    max_val = test_size - 1 (decrease min_size)

Return best approximation
```

**Purpose**: Dynamically adjust `min_object_size` to get exactly 28 fish per frame

### 7. Visualization & Output
**File**: `examples/adaptive_hsv_analysis_no_tolerance.py` (lines 165-280)

**Outputs**:
1. **Frame-by-frame visualization** (every 2 frames):
   - Original frame
   - Background subtracted frame
   - HSV masked result
   - Overlay with contours and centroids
   - Saved as PNG: `frame_XXX_adaptive_hsv_analysis.png`

2. **Video output**:
   - Original frame with contours drawn (yellow)
   - Centroids marked (red with white outline)
   - Frame info overlay (frame #, time, fish count, min_size)
   - Saved as MP4: `adaptive_hsv_analysis_28_fish.mp4`

3. **Summary text file**:
   - Detection statistics
   - Frame-by-frame results (fish count, min_size, iterations)
   - Saved as TXT: `adaptive_hsv_analysis_28_fish_summary.txt`

## Key Classes and Their Responsibilities

### TrackingProgramBackgroundSubtractor
**File**: `src/processing/tracking_program_background_subtractor.py`
- Creates background model from multiple frames
- Applies Gaussian blur to frames
- Calculates absolute difference
- Applies threshold to create binary mask

### HSVMasker
**File**: `src/tracking_methods/hsv_masking/hsv_masker.py`
- Converts frames to HSV color space
- Creates HSV mask for bright objects
- Applies morphological operations (open, close)
- Labels connected components
- Removes small objects based on dynamic min_size
- Detects contours and centroids

### Adaptive Script
**File**: `examples/adaptive_hsv_analysis_no_tolerance.py`
- Orchestrates the entire pipeline
- Implements binary search algorithm
- Creates visualizations
- Writes output files

## Data Types at Each Stage

1. **Input**: Video file (MP4)
2. **Video loading**: NumPy array (height, width, channels)
3. **Grayscale frame**: NumPy array (height, width)
4. **Background model**: NumPy array (float64) (height, width)
5. **Background subtracted**: NumPy array (uint8 binary mask)
6. **HSV frame**: NumPy array (HSV color space)
7. **HSV mask**: NumPy array (binary mask)
8. **Labeled components**: NumPy array (uint8 with region labels)
9. **Contours**: List of numpy arrays (contour points)
10. **Centroids**: List of tuples (y, x coordinates)

## Parameters

### Background Subtraction
- `threshold`: 25
- `blur_kernel_size`: (3, 3)
- `blur_sigma`: 0

### HSV Masking
- `lower_hsv`: (0, 0, 100)
- `upper_hsv`: (180, 255, 255)
- `min_object_size`: Dynamic (4-25 via binary search)
- `morph_kernel_size`: 3
- `apply_morphology`: True

### Binary Search
- `target_count`: 28 fish
- `min_val`: 4
- `max_val`: 25
- `max_iterations`: 10

## Output Files

All outputs are saved to: `data/output/adaptive_hsv_analysis_no_tolerance/`

1. **Video**: `adaptive_hsv_analysis_28_fish.mp4`
   - 10 seconds of video (200 frames at 20 FPS)
   - Shows detected fish with contours and centroids

2. **Summary**: `adaptive_hsv_analysis_28_fish_summary.txt`
   - Statistics and frame-by-frame results
   - Fish counts, min_sizes, iterations

3. **Frame images**: `frames/frame_XXX_adaptive_hsv_analysis.png`
   - Every 2nd frame (100 images total)
   - 4-panel visualization

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entry Point                                   │
│         examples/adaptive_hsv_analysis_no_tolerance.py          │
│                           main()                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│            Initialize Components                                │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ TrackingProgramBackgroundSubtractor(threshold=25)          │ │
│  └──────────────────────────────────────────────────────┘   │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ HSVMasker(lower_hsv=(0,0,100), upper_hsv=(180,255,255))  │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│          Create Background Model                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Load video with pims.PyAVReaderIndexed                    │ │
│  │ Select frames: [0,2k,4k,6k,8k,10k,12k,14k,16k,18k,20k,22k]│ │
│  │ Extract grayscale                                           │ │
│  │ Calculate median → background_model                         │ │
│  │ Apply Gaussian blur                                         │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────┴───────────┐
         │    For each frame     │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│         Apply Background Subtraction                            │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ frame → blur → |frame - background| → threshold → mask  │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│        Binary Search for Optimal min_size                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ For 1 to max_iterations:                                 │ │
│  │   test_size = (min_val + max_val) // 2                   │ │
│  │   Set min_object_size = test_size                        │ │
│  │   Apply HSV masking and count fish                        │ │
│  │   If count == 28: return test_size                        │ │
│  │   Else: adjust min_val/max_val                           │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│           Apply HSV Masking with Optimal min_size                │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ bg_subtracted → BGR → HSV                                │ │
│  │ Create HSV mask (bright objects only)                    │ │
│  │ Morphology: Open → Close                                  │ │
│  │ Label connected components                                │ │
│  │ Remove small objects (min_object_size)                    │ │
│  │ Detect contours and centroids                            │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│            Generate Outputs                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 1. Frame visualization (every 2 frames)                   │ │
│  │    - Original, BG-subtracted, HSV-masked, Overlay       │ │
│  │ 2. Video frame with contours and centroids               │ │
│  │ 3. Statistics and results                                │ │
│  └───────────────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────┴───────────┐
         │   Write to disk       │
         └───────────┬───────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output Files                                  │
│  • adaptive_hsv_analysis_28_fish.mp4                            │
│  • adaptive_hsv_analysis_28_fish_summary.txt                    │
│  • frames/frame_XXX_adaptive_hsv_analysis.png                  │
└─────────────────────────────────────────────────────────────────┘
```

## Summary

The pipeline uses a two-stage approach:
1. **Background Subtraction**: Isolates moving objects (fish) from static background
2. **HSV Masking**: Further isolates fish based on brightness characteristics

The adaptive component uses binary search to dynamically adjust the `min_object_size` parameter to achieve exactly 28 fish detections per frame with no tolerance, ensuring high precision in fish counting.









