# Pipeline Scripts Guide

This document explains what each script does and when to use it.

## 📁 New Organization (Updated)

Scripts are now organized into folders for better clarity:

```
examples/
├── scripts/
│   ├── original_pipeline/          # Original standalone pipeline
│   ├── column_pipelines/           # All column/section-based pipelines
│   └── batch_processing/           # Batch processing scripts
├── shared/                         # Shared utilities (wrapper.py)
└── pipelines/                      # Pipeline implementations
```

**All scripts can be run from project root:**
```bash
python examples/scripts/[folder]/[script].py
```

## Main Pipeline Scripts

### 1. `scripts/original_pipeline/complete_pipeline_v2.py`
**What it does:**
- Original pipeline that processes the entire frame at once
- Uses binary search to find a single threshold that gives exactly 28 fish total
- **No column/section division** - treats the whole frame as one unit
- **Target:** 28 fish total per frame

**When to use:**
- Simple baseline comparison
- When you don't need column-based detection

**Output:** Single video with 28 fish total

---

### 2. `scripts/column_pipelines/complete_pipeline_v2_columns.py`
**What it does:**
- Divides frame into **7 vertical columns**
- Uses binary search to find a single threshold that gives **4 fish per column** (28 total)
- All columns use the **same threshold** (global search)
- Uses the "default" variant (standard background model)

**When to use:**
- Standard column-based detection
- Want even distribution across columns
- Baseline for comparing with segmented version

**Output:** Video with 7 columns, 4 fish per column, yellow column boundaries

---

### 3. `complete_pipeline_v2_columns_segmented.py`
**What it does:**
- Divides frame into **7 vertical columns**
- Breaks video into **7 temporal segments**
- For each segment, processes each column **independently**
- Each column uses **exhaustive threshold search** (tries all 1-30) to get exactly 4 fish
- **Each column can have a different threshold** - they're independent!

**When to use:**
- Want maximum accuracy per column
- Need independent threshold adjustment per column
- Best accuracy (86.33% overall)

**Output:** Video with 7 columns, 4 fish per column, independent thresholds per column

---

### 4. `scripts/column_pipelines/complete_pipeline_v2_half_sectioned.py`
**What it does:**
- Splits frame into **top and bottom halves** (horizontal split)
- Each half divided into **7 vertical sections** (like columns)
- Total: **14 sections** (7 top + 7 bottom)
- Breaks video into **7 temporal segments**
- For each segment, processes each section **independently**
- Each section uses **exhaustive threshold search** to get exactly **2 fish per section**
- **Each section can have a different threshold** - they're independent!

**When to use:**
- Want even finer control (14 sections vs 7 columns)
- Need to handle top/bottom differences separately
- Want exactly 2 fish per section (28 total)

**Output:** Video with 14 sections (7 top + 7 bottom), 2 fish per section, yellow boundaries

---

## Batch Processing Scripts

### 5. `scripts/batch_processing/run_segmented_all_segments.py`
**What it does:**
- Runs the **segmented columns pipeline** on all 4 segments of a 20-minute video
- Segment 1: 0-5 minutes (frames 0-5999)
- Segment 2: 5-10 minutes (frames 6000-11999)
- Segment 3: 10-15 minutes (frames 12000-17999)
- Segment 4: 15-20 minutes (frames 18000-23999)
- Each segment outputs to its own folder in `SegmentedOutputs/`

**When to use:**
- Process entire 20-minute video in one go
- Want separate outputs for each 5-minute segment
- Automated batch processing

**Output:** 4 folders in `SegmentedOutputs/`, one per segment

---

### 6. `scripts/batch_processing/run_segmented_segment4.py`
**What it does:**
- Runs the **segmented columns pipeline** on **only segment 4** (15-20 minutes)
- Useful for re-running a specific segment

**When to use:**
- Need to re-process just segment 4
- Testing/debugging a specific time period

**Output:** Single folder for segment 4

---

## Shared Utilities

### `shared/wrapper.py` ⭐
- **Purpose**: Shared wrapper function used by all column-based and batch processing scripts
- **Function**: `process_complete_pipeline_v2_columns()`
- **Used by**: All scripts in `scripts/column_pipelines/` and `scripts/batch_processing/`
- **See**: `shared/README.md` for detailed documentation

## Pipeline Variants (in `examples/pipelines/v2_columns/`)

These are the underlying implementations that the scripts above use:

### `default_bg.py` - Default Background Pipeline
- Uses default frame indices for background model
- Standard approach

### `shared_bg.py` - Shared Background Pipeline
- Uses a single background model from the beginning of video
- More consistent across segments

### `segment_bg.py` - Segment Background Pipeline
- Creates a new background model for each segment
- Better adaptation to changing conditions

### `adaptive_column_limit.py` - Adaptive Column Limit Pipeline
- Adaptively selects best 4 fish per column based on size/quality
- More sophisticated selection than binary search

### `segmented_columns.py` - Segmented Columns Pipeline ⭐
- **7 temporal segments**
- **Independent per-column exhaustive threshold search**
- **4 fish per column**
- **Best accuracy: 86.33%**

### `half_sectioned.py` - Half-Sectioned Pipeline ⭐ NEW
- **7 temporal segments**
- **14 sections** (7 top + 7 bottom)
- **Independent per-section exhaustive threshold search**
- **2 fish per section**
- **99% accuracy on test**

---

## Quick Reference Table

| Script | Columns/Sections | Fish per Unit | Threshold Method | Accuracy |
|--------|------------------|---------------|-------------------|----------|
| `scripts/original_pipeline/complete_pipeline_v2.py` | None (whole frame) | 28 total | Binary search (global) | Baseline |
| `scripts/column_pipelines/complete_pipeline_v2_columns.py` | 7 columns | 4 per column | Binary search (global) | 72.09% |
| `scripts/column_pipelines/complete_pipeline_v2_columns_segmented.py` | 7 columns | 4 per column | Exhaustive (per column) | **86.33%** |
| `scripts/column_pipelines/complete_pipeline_v2_half_sectioned.py` | 14 sections (7+7) | 2 per section | Exhaustive (per section) | **99%** (test) |

---

## Which Script Should I Use?

**For best accuracy:**
- Use `scripts/column_pipelines/complete_pipeline_v2_half_sectioned.py` (14 sections, 2 fish each)
- Or `scripts/column_pipelines/complete_pipeline_v2_columns_segmented.py` (7 columns, 4 fish each)

**For processing full 20-minute video:**
- Use `scripts/batch_processing/run_segmented_all_segments.py` (runs segmented columns on all 4 segments)

**For simple baseline:**
- Use `scripts/column_pipelines/complete_pipeline_v2_columns.py` (standard approach)

---

## Key Differences Summary

1. **Frame Division:**
   - `scripts/original_pipeline/complete_pipeline_v2.py`: No division (whole frame)
   - `scripts/column_pipelines/complete_pipeline_v2_columns.py`: 7 vertical columns
   - `scripts/column_pipelines/complete_pipeline_v2_columns_segmented.py`: 7 vertical columns
   - `scripts/column_pipelines/complete_pipeline_v2_half_sectioned.py`: 14 sections (top/bottom halves, then 7 each)

2. **Threshold Search:**
   - `scripts/original_pipeline/complete_pipeline_v2.py`: Binary search (global)
   - `scripts/column_pipelines/complete_pipeline_v2_columns.py`: Binary search (global, all columns same)
   - `scripts/column_pipelines/complete_pipeline_v2_columns_segmented.py`: **Exhaustive search (per column, independent)**
   - `scripts/column_pipelines/complete_pipeline_v2_half_sectioned.py`: **Exhaustive search (per section, independent)**

3. **Temporal Segmentation:**
   - `scripts/original_pipeline/complete_pipeline_v2.py`: None
   - `scripts/column_pipelines/complete_pipeline_v2_columns.py`: None
   - `scripts/column_pipelines/complete_pipeline_v2_columns_segmented.py`: **7 temporal segments**
   - `scripts/column_pipelines/complete_pipeline_v2_half_sectioned.py`: **7 temporal segments**

---

## Output Locations

- `scripts/original_pipeline/complete_pipeline_v2.py` → `data/output/complete_pipeline_v2_...`
- `scripts/column_pipelines/complete_pipeline_v2_columns.py` → `data/output/StandardPipeline/CompleteVideos/...`
- `scripts/column_pipelines/complete_pipeline_v2_columns_segmented.py` → `data/output/segmented_columns_...`
- `scripts/column_pipelines/complete_pipeline_v2_half_sectioned.py` → `data/output/half_sectioned_...`
- `scripts/batch_processing/run_segmented_all_segments.py` → `data/output/SegmentedOutputs/columns_segmented_seg{1-4}_5min/`

