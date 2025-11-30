# Pipeline Reorganization

## Overview

The pipeline code has been reorganized into a modular structure where each pipeline variant is independent. This allows you to implement different solutions (like different background model strategies) without affecting previous code.

## New Structure

```
examples/
├── pipelines/                          # New modular pipeline structure
│   ├── __init__.py
│   ├── README.md                      # Detailed documentation
│   ├── base/                          # Base classes and shared utilities
│   │   ├── __init__.py
│   │   ├── base_pipeline.py          # BasePipeline class with common logic
│   │   └── utils.py                  # Shared utility functions
│   └── v2_columns/                    # V2 column-based pipeline variants
│       ├── __init__.py
│       ├── wrapper.py                 # Backward compatibility wrapper
│       ├── default_bg.py             # Default background (original)
│       ├── shared_bg.py               # Shared background (Solution 1)
│       └── segment_bg.py             # Segment-specific background (Solution 2)
├── complete_pipeline_v2_columns.py    # Backward compatible wrapper
└── process_4_segments.py             # Updated to support variants
```

## Key Benefits

1. **Isolation**: Each variant is completely independent
2. **Reusability**: Common logic in `BasePipeline` is shared
3. **Extensibility**: Easy to add new variants
4. **Testability**: Each variant can be tested independently
5. **Backward Compatibility**: Original interface still works

## Pipeline Variants

### 1. Default Background (`default_bg.py`)
- **Current implementation** - Uses default frame indices
- **Issue**: Accuracy degrades for later segments
- **Use**: Baseline comparison

### 2. Shared Background (`shared_bg.py`)
- **Solution 1** - Uses single background from beginning
- **Benefit**: Consistent accuracy across segments
- **Use**: When background is stable

### 3. Segment Background (`segment_bg.py`)
- **Solution 2** - Uses segment-specific frame indices
- **Benefit**: Adapts to background changes
- **Use**: When background changes over time

## Usage Examples

### Using the Wrapper (Backward Compatible)

```python
from pipelines.v2_columns.wrapper import process_complete_pipeline_v2_columns

# Default variant
process_complete_pipeline_v2_columns(
    video_path="video.mp4",
    output_dir="output",
    variant="default"
)

# Shared background variant
process_complete_pipeline_v2_columns(
    video_path="video.mp4",
    output_dir="output",
    variant="shared"
)
```

### Using Pipeline Classes Directly

```python
from pipelines.v2_columns.shared_bg import SharedBackgroundPipeline

pipeline = SharedBackgroundPipeline(
    fps=20,
    num_columns=7,
    target_per_column=4
)

pipeline.process(
    video_path=Path("video.mp4"),
    output_dir=Path("output"),
    duration_seconds=300,
    start_frame=0
)
```

### Processing 4 Segments

```bash
# Default variant
python examples/process_4_segments.py --variant default

# Shared background variant
python examples/process_4_segments.py --variant shared

# Segment-specific background variant
python examples/process_4_segments.py --variant segment
```

## Adding New Variants

To add a new pipeline variant:

1. Create a new file in `v2_columns/` (e.g., `adaptive_bg.py`)
2. Inherit from `BasePipeline`:
   ```python
   from ..base.base_pipeline import BasePipeline
   
   class AdaptiveBackgroundPipeline(BasePipeline):
       def create_background_model_strategy(self, video_path, start_frame, num_frames, output_dir):
           # Implement your strategy
           pass
   ```
3. Add to `v2_columns/__init__.py`
4. Add to `wrapper.py` for backward compatibility

## Migration Guide

### Old Code
```python
from complete_pipeline_v2_columns import process_complete_pipeline_v2_columns
```

### New Code (Same Interface)
```python
from complete_pipeline_v2_columns import process_complete_pipeline_v2_columns
# Still works! Now uses wrapper internally
```

### New Code (Using Variants)
```python
from pipelines.v2_columns.wrapper import process_complete_pipeline_v2_columns

process_complete_pipeline_v2_columns(
    ...,
    variant="shared"  # Choose your variant
)
```

## Files Changed

- ✅ Created `examples/pipelines/` structure
- ✅ Moved common logic to `BasePipeline`
- ✅ Created 3 pipeline variants
- ✅ Updated `process_4_segments.py` to support variants
- ✅ Maintained backward compatibility in `complete_pipeline_v2_columns.py`

## Next Steps

1. Test each variant to compare accuracy
2. Implement additional variants as needed
3. Use the variant that works best for your use case

