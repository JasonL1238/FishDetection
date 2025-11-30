# Fish Detection Project - Complete Explanation

## 🎯 Project Purpose

This project detects and tracks **28 fish** in video frames arranged in a **7 columns × 4 rows** grid. The system processes video frames to:
1. Identify fish locations
2. Count fish per column (target: 4 fish per column)
3. Generate annotated videos showing detections
4. Produce statistics and summaries

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ENTRY POINTS                              │
│  ┌──────────────────┐  ┌──────────────────────────────┐   │
│  │ complete_pipeline│  │ complete_pipeline_v2_columns │   │
│  │      _v2.py      │  │           .py                │   │
│  └────────┬─────────┘  └──────────────┬───────────────┘   │
│           │                           │                    │
│           │                           │                    │
└───────────┼───────────────────────────┼────────────────────┘
            │                           │
            │                           │
    ┌───────▼───────────────────────────▼────────┐
    │         PIPELINE LAYER                     │
    │  ┌──────────────────────────────────────┐  │
    │  │      BasePipeline (Abstract)         │  │
    │  │  - Common processing logic            │  │
    │  │  - Frame iteration                    │  │
    │  │  - Statistics collection              │  │
    │  └──────────────┬───────────────────────┘  │
    │                 │                           │
    │    ┌────────────┴────────────┐            │
    │    │                         │            │
    │  ┌─▼────────┐  ┌─────────────▼──┐         │
    │  │ Default  │  │ Shared/        │         │
    │  │ Segment  │  │ Adaptive       │         │
    │  └──────────┘  └────────────────┘         │
    └───────┬───────────────────────────────────┘
            │
    ┌───────▼───────────────────────────────────┐
    │         CORE COMPONENTS                    │
    │  ┌──────────────────────────────────────┐ │
    │  │ Background Subtractor V2             │ │
    │  │  - Creates background model           │ │
    │  │  - Subtracts background from frames   │ │
    │  └──────────────┬───────────────────────┘ │
    │                 │                          │
    │  ┌──────────────▼───────────────────────┐ │
    │  │ HSV Masker                            │ │
    │  │  - Filters by color/brightness        │ │
    │  │  - Detects contours & centroids       │ │
    │  └──────────────┬───────────────────────┘ │
    │                 │                          │
    │  ┌──────────────▼───────────────────────┐ │
    │  │ Column-Based Detection               │ │
    │  │  - Divides frame into 7 columns       │ │
    │  │  - Binary search for optimal params   │ │
    │  └──────────────────────────────────────┘ │
    └───────────────────────────────────────────┘
```

---

## 📦 Component Breakdown

### 1. **Entry Point Scripts** (`examples/`)

#### `complete_pipeline_v2.py`
- **Purpose**: Original pipeline that targets exactly 28 fish per frame (not column-based)
- **Flow**: 
  1. Creates background model
  2. Processes each frame
  3. Uses binary search to find optimal `min_object_size` to get exactly 28 fish
  4. Outputs annotated video

#### `complete_pipeline_v2_columns.py`
- **Purpose**: Wrapper for column-based detection (backward compatibility)
- **Flow**: 
  1. Imports from new modular structure
  2. Calls `process_complete_pipeline_v2_columns()` from wrapper
  3. Maintains same interface as old code

#### `process_4_segments.py`
- **Purpose**: Processes 4 five-minute segments of video
- **Flow**: 
  1. Loops through 4 segments (0-5min, 5-10min, 10-15min, 15-20min)
  2. Calls pipeline for each segment
  3. Supports different variants (default/shared/segment/adaptive)

---

### 2. **Pipeline Layer** (`examples/pipelines/`)

#### `base/base_pipeline.py` - **The Foundation**
This is the **abstract base class** that contains all the common logic:

**What it does:**
- Defines the processing workflow (frame loading, iteration, output generation)
- Handles video I/O (reading frames, writing output video)
- Manages statistics collection
- Generates summary reports
- Draws visualizations (contours, centroids, column boundaries)

**Key Method:**
```python
def process(video_path, output_dir, duration_seconds, start_frame):
    # 1. Initialize background subtractor (via strategy)
    # 2. Initialize HSV masker
    # 3. Load video
    # 4. For each frame:
    #    - Apply background subtraction
    #    - Detect fish (column-based)
    #    - Draw results
    #    - Save frame
    # 5. Generate summary
```

**Abstract Method (must be implemented by subclasses):**
```python
@abstractmethod
def create_background_model_strategy(...):
    # Each variant implements this differently
    # Returns configured background subtractor
```

#### `v2_columns/` - **Pipeline Variants**

All variants inherit from `BasePipeline` and only differ in **how they create the background model**:

##### `default_bg.py` - DefaultBackgroundPipeline
- **Strategy**: Uses default frame indices from entire video
- **Frame indices**: [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]
- **Use case**: Baseline, works well for first segment
- **Problem**: Accuracy degrades for later segments (background drift)

##### `shared_bg.py` - SharedBackgroundPipeline  
- **Strategy**: Uses single background model from beginning of video
- **Frame indices**: Always uses frames from start (e.g., [0, 2000, 4000, ...])
- **Use case**: When background is stable over time
- **Benefit**: Consistent accuracy across all segments

##### `segment_bg.py` - SegmentBackgroundPipeline
- **Strategy**: Uses frame indices within each segment
- **Frame indices**: Calculated relative to `start_frame` (e.g., if start_frame=6000, uses [6000, 6200, 6400, ...])
- **Use case**: When background changes significantly over time
- **Benefit**: Adapts to background changes per segment

##### `adaptive_column_limit.py` - AdaptiveColumnLimitPipeline
- **Strategy**: Uses default background, but different detection approach
- **Detection**: Instead of adjusting global `min_size`, it:
  1. Detects many candidates with permissive `min_size`
  2. Groups by column
  3. Selects best 4 fish per column based on size/quality
- **Use case**: More adaptive selection when fish overlap or vary in size

##### `wrapper.py` - Backward Compatibility
- **Purpose**: Maintains old function interface
- **Function**: `process_complete_pipeline_v2_columns()`
- **Parameters**: Same as old code, but routes to appropriate variant

---

### 3. **Core Components** (`src/`)

#### `processing/tracking_program_background_subtractor_v2.py`

**Purpose**: Separates moving fish from static background

**How it works:**
1. **Background Model Creation**:
   ```python
   create_background_model(video_path, frame_indices)
   # - Loads specified frames
   # - Computes median across frames → background model
   # - Applies Gaussian blur
   ```

2. **Background Subtraction**:
   ```python
   apply_background_subtraction(frame)
   # - Computes |frame - background|
   # - Applies threshold (default: 15)
   # - Applies morphological closing (5x5) to reconnect split blobs
   # - Returns binary mask (white = fish, black = background)
   ```

**Key Parameters:**
- `threshold=15`: Lower than V1 (25) to capture faint fish parts
- `morph_kernel_size=(5,5)`: Prevents blob splitting

#### `tracking_methods/hsv_masking/hsv_masker.py`

**Purpose**: Filters detections by color/brightness and finds fish contours

**How it works:**
1. **HSV Masking** (if applied to color frames):
   ```python
   # Converts frame to HSV color space
   # Creates mask: lower_hsv <= pixel <= upper_hsv
   # Default: (0,0,100) to (180,255,255) → filters bright objects
   ```

2. **Contour Detection**:
   ```python
   detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_frame)
   # - Finds connected components in binary mask
   # - Filters by min_object_size
   # - Returns contours and centroids (y, x)
   ```

**Key Parameters:**
- `min_object_size`: Minimum pixel area for valid fish (adjusted adaptively)
- `lower_hsv=(0,0,100)`: Filters for bright objects (value > 100)
- `apply_morphology=True`: Cleans up noise

#### `pipelines/base/utils.py`

**Purpose**: Shared utility functions for column-based detection

**Functions:**
1. `count_fish_per_column(centroids, width, num_columns)`:
   - Divides frame width into `num_columns` equal columns
   - Counts how many centroids fall in each column
   - Returns list: `[count_col0, count_col1, ..., count_col6]`

2. `binary_search_min_size_columns(...)`:
   - Binary search to find optimal `min_object_size`
   - Goal: Get exactly `target_per_column` fish in each column
   - Returns: `(best_min_size, column_counts, iterations)`

---

## 🔄 Data Flow Through the System

### Complete Processing Flow:

```
1. ENTRY POINT
   └─> User runs: python examples/complete_pipeline_v2_columns.py
       │
       └─> Calls: process_complete_pipeline_v2_columns(variant="default")
           │
           └─> Creates: DefaultBackgroundPipeline instance
               │
               └─> Calls: pipeline.process()

2. INITIALIZATION (in BasePipeline.process())
   │
   ├─> Create background subtractor (via strategy)
   │   └─> DefaultBackgroundPipeline.create_background_model_strategy()
   │       │
   │       └─> TrackingProgramBackgroundSubtractorV2.create_background_model()
   │           │
   │           ├─> Loads video: pims.PyAVReaderIndexed()
   │           ├─> Extracts frames at indices: [0, 2000, 4000, ...]
   │           ├─> Computes median: np.median(frames, axis=2)
   │           └─> Applies blur: cv2.GaussianBlur()
   │
   ├─> Initialize HSV masker
   │   └─> HSVMasker(lower_hsv=(0,0,100), upper_hsv=(180,255,255), ...)
   │
   └─> Load video for processing
       └─> pims.PyAVReaderIndexed(video_path)

3. FRAME PROCESSING LOOP (for each frame)
   │
   ├─> Load frame: video[frame_idx]
   │   └─> Convert to grayscale if needed
   │
   ├─> Background Subtraction
   │   └─> bg_subtractor.apply_background_subtraction(gray_frame)
   │       │
   │       ├─> Compute difference: |frame - background_model|
   │       ├─> Threshold: diff > 15 → white, else → black
   │       ├─> Morphological closing: cv2.morphologyEx(MORPH_CLOSE)
   │       └─> Returns: binary_mask (uint8, 0 or 255)
   │
   ├─> Column-Based Detection
   │   └─> binary_search_min_size_columns(...)
   │       │
   │       ├─> Binary search loop (up to 15 iterations):
   │       │   │
   │       │   ├─> Set test min_size: (min_val + max_val) // 2
   │       │   ├─> Detect fish: hsv_masker.detect_fish_contours_and_centroids_bg_subtracted()
   │       │   │   │
   │       │   │   ├─> Find contours: cv2.findContours()
   │       │   │   ├─> Filter by area: area >= min_object_size
   │       │   │   └─> Compute centroids: cv2.moments()
   │       │   │
   │       │   ├─> Count per column: count_fish_per_column(centroids, width, 7)
   │       │   ├─> Calculate score: sum(|count - target| for each column)
   │       │   │
   │       │   └─> Adjust search range:
   │       │       ├─> If total_fish > target_total: min_val = test_size + 1
   │       │       └─> Else: max_val = test_size - 1
   │       │
   │       └─> Returns: (best_min_size, column_counts, iterations)
   │
   ├─> Final Detection
   │   └─> hsv_masker.min_object_size = best_min_size
   │   └─> contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted()
   │
   ├─> Visualization
   │   ├─> Draw column boundaries: cv2.line() (yellow vertical lines)
   │   ├─> Draw contours: cv2.drawContours() (green)
   │   ├─> Draw centroids: cv2.circle() (red dots)
   │   └─> Add text stats: cv2.putText()
   │
   └─> Save Output
       ├─> Write frame to video: out.write(output_frame)
       └─> Save frame image (every 5th frame): cv2.imwrite()

4. SUMMARY GENERATION
   │
   └─> BasePipeline._generate_summary()
       │
       ├─> Calculate statistics:
       │   ├─> Average fish per column
       │   ├─> Standard deviation
       │   ├─> Column accuracy (how many frames have perfect columns)
       │   └─> Average min_size and iterations
       │
       └─> Write to file: complete_pipeline_v2_columns_7x4_summary.txt
```

---

## 🎨 Visual Example

### Input Frame:
```
┌─────────────────────────────────────────┐
│                                         │
│  [Fish]  [Fish]  [Fish]  [Fish]  ...   │  Row 1
│  [Fish]  [Fish]  [Fish]  [Fish]  ...   │  Row 2
│  [Fish]  [Fish]  [Fish]  [Fish]  ...   │  Row 3
│  [Fish]  [Fish]  [Fish]  [Fish]  ...   │  Row 4
│                                         │
│  Col1  Col2  Col3  Col4  Col5  Col6  Col7 │
└─────────────────────────────────────────┘
```

### After Background Subtraction:
```
┌─────────────────────────────────────────┐
│                                         │
│  ████    ████    ████    ████    ...   │  White = fish
│  ████    ████    ████    ████    ...   │  Black = background
│  ████    ████    ████    ████    ...   │
│  ████    ████    ████    ████    ...   │
│                                         │
└─────────────────────────────────────────┘
```

### After Column Detection:
```
┌─────────────────────────────────────────┐
│  |      |      |      |      |  ...    │  Yellow lines = column boundaries
│  |  ●   |  ●   |  ●   |  ●   |  ...    │  Red dots = centroids
│  | ═══  | ═══  | ═══  | ═══  |  ...    │  Green = contours
│  |  ●   |  ●   |  ●   |  ●   |  ...    │
│  |      |      |      |      |  ...    │
│  Col1:4 Col2:4 Col3:4 Col4:4 ...      │  Stats: C1:4 = Column 1 has 4 fish
└─────────────────────────────────────────┘
```

---

## 🔑 Key Concepts

### 1. **Background Subtraction**
- **Why**: Fish move, background doesn't → difference highlights fish
- **How**: Median of multiple frames = background model
- **V2 Improvement**: Lower threshold (15 vs 25) + morphological closing

### 2. **HSV Masking**
- **Why**: Filters out noise, keeps only bright objects (fish are bright)
- **How**: Converts to HSV, masks pixels with value > 100

### 3. **Column-Based Detection**
- **Why**: Ensures even distribution (4 fish per column = 28 total)
- **How**: Divides frame into 7 equal columns, counts fish per column

### 4. **Adaptive Binary Search**
- **Why**: Different frames need different `min_object_size` (lighting, fish size varies)
- **How**: Binary search finds `min_size` that gives exactly 4 fish per column

### 5. **Pipeline Variants**
- **Why**: Different background strategies for different scenarios
- **How**: All share same processing logic, only differ in background model creation

---

## 📊 Output Files

After processing, you get:

1. **Video**: `complete_pipeline_v2_columns_7x4.mp4`
   - Annotated video with detections drawn

2. **Summary**: `complete_pipeline_v2_columns_7x4_summary.txt`
   - Statistics: average fish per column, accuracy, etc.

3. **Background Model**: `background_model.npy`
   - Saved numpy array of background

4. **Frames**: `frames/frame_XXXXX_columns.png`
   - Individual annotated frames (every 5th frame)

---

## 🚀 How to Use

### Run Original Pipeline (28 fish total):
```bash
python examples/complete_pipeline_v2.py
```

### Run Column-Based Pipeline (4 fish per column):
```bash
python examples/complete_pipeline_v2_columns.py
```

### Process 4 Segments with Different Variants:
```bash
# Default (original)
python examples/process_4_segments.py --variant default

# Shared background (consistent across segments)
python examples/process_4_segments.py --variant shared

# Segment-specific background (adapts per segment)
python examples/process_4_segments.py --variant segment

# Adaptive column limit (different detection strategy)
python examples/process_4_segments.py --variant adaptive
```

---

## 🔧 Configuration

Key parameters you can adjust:

- **`num_columns`**: Number of columns (default: 7)
- **`target_per_column`**: Target fish per column (default: 4)
- **`threshold`**: Background subtraction threshold (default: 15)
- **`hsv_lower`**: Lower HSV bound (default: (0,0,100))
- **`hsv_upper`**: Upper HSV bound (default: (180,255,255))
- **`min_object_size`**: Initial minimum object size (default: 10)

---

## 🎓 Summary

**The project is organized in layers:**

1. **Entry Points** → User-facing scripts
2. **Pipeline Layer** → Abstract base + concrete variants
3. **Core Components** → Background subtraction + HSV masking
4. **Utilities** → Column counting + binary search

**The magic happens in:**
- Background subtraction separates fish from background
- HSV masking filters noise
- Column-based detection ensures even distribution
- Binary search adapts parameters per frame

**All variants share the same processing logic** - they only differ in **how the background model is created**, which affects accuracy across different video segments.

