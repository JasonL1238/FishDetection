# Project Structure Overview

This document provides a quick reference for the organized project structure.

## Directory Organization

### Core Code (`src/`)
- `processing/` - Background subtraction methods (V2 is used by all pipelines)
- `tracking_methods/` - Fish detection algorithms
  - `hsv_masking/` - HSV-based fish detection (used by all pipelines)
  - `common/` - Shared utilities and base classes
- `image_pre/` - Image preprocessing utilities

### Active Pipelines (`scripts/`)
- `segmented_columns/` - **ACTIVE** - Segmented columns pipeline (86.33% accuracy)
  - 7 columns, 4 fish per column, exhaustive per-column threshold search
- `half_sectioned/` - **ACTIVE** - Half-sectioned pipeline (84.97% accuracy)
  - 14 sections (7 top + 7 bottom), 2 fish per section, exhaustive per-section threshold search
- `default_columns/` - Baseline reference pipeline
  - 7 columns, 4 fish per column, global binary search

Each pipeline folder is self-contained with:
- `run.py` - Main execution script
- `pipeline.py` - Pipeline implementation
- `base_pipeline.py` - BasePipeline class (local copy)
- `utils.py` - Utility functions (local copy)
- `README.md` - Pipeline-specific documentation

### Data (`data/`)
- `input/videos/` - Input video files
- `output/` - Results organized by pipeline
  - `SegmentedOutputs/` - Segmented columns pipeline results
  - `HalfSectioned/` - Half-sectioned pipeline results
  - `StandardPipeline/` - Baseline pipeline results

## Key Features

- **Self-contained pipelines:** Each pipeline has all its dependencies
- **Independent operation:** No shared code between pipelines
- **Modular design:** Easy to add new pipeline variants
- **Comprehensive documentation:** Each pipeline has its own README

## Running Pipelines

```bash
# Segmented columns - single run
python -m scripts.segmented_columns.run

# Segmented columns - all 4 segments (20 minutes)
python -m scripts.segmented_columns.run_all_segments

# Half-sectioned - full video (20 minutes)
python -m scripts.half_sectioned.run

# Default columns - baseline
python -m scripts.default_columns.run
```

## Core Library Dependencies

All pipelines use:
- `src/processing/tracking_program_background_subtractor_v2.py` - V2 background subtraction
- `src/tracking_methods/hsv_masking/hsv_masker.py` - HSV-based fish detection
