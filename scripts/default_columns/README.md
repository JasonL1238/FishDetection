# Default Columns Pipeline

Self-contained pipeline for default column-based detection.

## Files

- `run.py` - Script to run the pipeline
- `pipeline.py` - DefaultBackgroundPipeline implementation
- `base_pipeline.py` - BasePipeline class (local copy)
- `utils.py` - Utility functions (local copy)

## Usage

```bash
python examples/default_columns/run.py
```

## What it does

- Divides frame into 7 vertical columns
- Uses binary search to find a single threshold that gives 4 fish per column (28 total)
- All columns use the same threshold (global search)

