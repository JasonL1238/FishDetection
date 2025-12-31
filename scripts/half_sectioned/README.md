# Half-Sectioned Pipeline

Self-contained pipeline for half-sectioned detection with independent per-section threshold search.

## Files

- `run.py` - Script to run the pipeline
- `pipeline.py` - HalfSectionedPipeline implementation
- `base_pipeline.py` - BasePipeline class (local copy)
- `utils.py` - Utility functions (local copy, minimal)

## Usage

```bash
python examples/half_sectioned/run.py
```

## What it does

- Splits frame into top and bottom halves
- Each half is divided into 7 sections (like columns)
- Each of the 14 sections is processed independently
- Each section uses exhaustive threshold search (tries all 1-30) to get exactly 2 fish
- Each section can have a different threshold - they're independent!
- 7 temporal segments

