# Original Pipeline

Standalone original pipeline that processes the whole frame at once.

## Files

- `complete_pipeline_v2.py` - Standalone script (no dependencies)

## Usage

```bash
python examples/original_pipeline/complete_pipeline_v2.py
```

## What it does

- Processes the entire frame at once (no column/section division)
- Uses binary search to find a single threshold that gives exactly 28 fish total
- No dependencies on other pipeline code

