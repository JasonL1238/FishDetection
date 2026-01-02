# Fish Detection Pipelines

This directory contains self-contained fish detection pipelines. Each pipeline is completely independent with its own implementation and dependencies.

## Structure

```
scripts/
├── segmented_columns/      # Segmented columns pipeline (ACTIVE)
│   ├── run.py              # Single run script (10 seconds)
│   ├── run_all_segments.py # Batch processing (all 4 segments of 20-min video)
│   ├── run_segment4.py     # Single segment script (segment 4 only)
│   ├── pipeline.py         # SegmentedColumnsPipeline implementation
│   ├── base_pipeline.py    # BasePipeline class (local copy)
│   ├── utils.py            # Utility functions (local copy)
│   └── README.md
│
├── half_sectioned/         # Half-sectioned pipeline (ACTIVE)
│   ├── run.py              # Script to run the pipeline
│   ├── pipeline.py         # HalfSectionedPipeline implementation
│   ├── base_pipeline.py    # BasePipeline class (local copy)
│   ├── utils.py            # Utility functions (local copy)
│   └── README.md
│
└── default_columns/        # Default columns pipeline (baseline reference)
    ├── run.py              # Script to run the pipeline
    ├── pipeline.py         # DefaultBackgroundPipeline implementation
    ├── base_pipeline.py    # BasePipeline class (local copy)
    ├── utils.py            # Utility functions (local copy)
    └── README.md
```

## Active Pipelines

### Segmented Columns Pipeline
- **Accuracy:** 86.33% (20,719/24,000 perfect frames)
- **Method:** 7 vertical columns, 4 fish per column, exhaustive per-column threshold search
- **Temporal Segments:** 7 segments per video segment
- **Best for:** Column-based detection with independent per-column adaptation

**Usage:**
```bash
# Single run (10 seconds)
python -m scripts.segmented_columns.run

# All 4 segments (20 minutes total)
python -m scripts.segmented_columns.run_all_segments

# Segment 4 only (15-20 minutes)
python -m scripts.segmented_columns.run_segment4
```

### Half-Sectioned Pipeline
- **Accuracy:** 84.97% (20,394/24,000 perfect frames)
- **Method:** 14 sections (7 top + 7 bottom), 2 fish per section, exhaustive per-section threshold search
- **Temporal Segments:** 7 segments
- **Best for:** More granular detection with top/bottom spatial division

**Usage:**
```bash
# Full video (20 minutes)
python -m scripts.half_sectioned.run
```

### Default Columns Pipeline
- **Method:** 7 vertical columns, 4 fish per column, global binary search
- **Baseline reference:** Standard column-based approach without per-column adaptation

**Usage:**
```bash
python -m scripts.default_columns.run
```

## Key Features

- **Self-contained:** Each pipeline folder has all its dependencies
- **Independent:** No shared code between pipelines (each has its own BasePipeline copy)
- **Modular:** Easy to add new pipeline variants
- **Documented:** Each folder has its own README with specific details

## Core Library

All pipelines use the core library in `src/`:
- `src/tracking_methods/` - HSV masking and detection methods
- `src/processing/` - Background subtraction (V2)
- `src/image_pre/` - Image preprocessing utilities

## Output Locations

- **Segmented Columns:** `data/output/SegmentedOutputs/`
- **Half-Sectioned:** `data/output/HalfSectioned/`
- **Default Columns:** `data/output/` (configurable in run.py)
