# Custom Grid Pipeline

Self-contained pipeline for custom grid-based detection with largest blob selection per cell.

## Files

- `run.py` - Script to run the pipeline
- `pipeline.py` - CustomGridPipeline implementation
- `base_pipeline.py` - BasePipeline class (local copy)

## Usage

```bash
python -m scripts.custom_grid.run
```

## What it does

- Divides frame into custom 7x7 grid with specific horizontal lines:
  - Top line: moved up from standard position
  - Center line: through middle of frame
  - Bottom line: moved down from standard position
- 7 vertical columns (evenly spaced)
- Total: 28 cells (7 columns × 4 rows)
- For each cell, finds 1 fish by selecting the largest blob/contour
- Uses V2 background subtraction + HSV masking
- 7 temporal segments

## Detection Method

Instead of exhaustive threshold search, this pipeline:
1. Applies background subtraction and HSV masking
2. Finds all blobs in each cell
3. Selects the largest blob (by area) as the fish for that cell
4. Target: 1 fish per cell

