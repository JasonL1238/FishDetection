# Segmented Columns Pipeline

Self-contained pipeline for segmented column-based detection with independent per-column threshold search.

## Files

- `run.py` - Single run script (10 seconds)
- `run_all_segments.py` - Batch processing script (all 4 segments of 20-min video)
- `run_segment4.py` - Single segment script (segment 4 only)
- `pipeline.py` - SegmentedColumnsPipeline implementation
- `base_pipeline.py` - BasePipeline class (local copy)
- `utils.py` - Utility functions (local copy)

## Usage

```bash
# Single run (10 seconds)
python examples/segmented_columns/run.py

# All 4 segments (20 minutes total)
python examples/segmented_columns/run_all_segments.py

# Segment 4 only (15-20 minutes)
python examples/segmented_columns/run_segment4.py
```

## What it does

- Divides frame into 7 vertical columns
- Breaks video into 7 temporal segments
- For each segment, processes each column independently
- Each column uses exhaustive threshold search (tries all 1-30) to get exactly 4 fish
- Each column can have a different threshold - they're independent!

