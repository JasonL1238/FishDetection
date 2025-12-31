# Pipeline Implementations

This directory contains modular pipeline implementations for fish detection. Each pipeline variant is independent and can be developed/tested without affecting others.

## Structure

```
pipelines/
├── __init__.py
├── README.md
├── base/                          # Base classes and shared utilities
│   ├── __init__.py
│   ├── base_pipeline.py          # Base pipeline class with common logic
│   └── utils.py                  # Shared utility functions
└── v2_columns/                    # V2 column-based pipeline variants
    ├── __init__.py
    ├── wrapper.py                 # Backward compatibility wrapper
    ├── default_bg.py             # Default background (current implementation)
    ├── shared_bg.py              # Shared background (Solution 1)
    └── segment_bg.py             # Segment-specific background (Solution 2)
```

## Pipeline Variants

### 1. Default Background (`default_bg.py`)
**Current implementation** - Uses default frame indices from entire video.

- **Pros**: Simple, works well for first segment
- **Cons**: Accuracy degrades for later segments due to background drift
- **Use case**: Baseline comparison

### 2. Shared Background (`shared_bg.py`)
**Solution 1** - Uses single background model from beginning of video.

- **Pros**: Consistent accuracy across all segments
- **Cons**: Assumes background doesn't change significantly
- **Use case**: When background is stable over time

### 3. Segment Background (`segment_bg.py`)
**Solution 2** - Uses frame indices within each segment.

- **Pros**: Adapts to background changes over time
- **Cons**: More complex, requires segment-specific processing
- **Use case**: When background changes significantly over time

## Usage

### Using the Wrapper (Backward Compatible)

```python
from pipelines.v2_columns.wrapper import process_complete_pipeline_v2_columns

# Default variant (original implementation)
process_complete_pipeline_v2_columns(
    video_path="data/input/videos/video.mp4",
    output_dir="data/output/default",
    variant="default"
)

# Shared background variant
process_complete_pipeline_v2_columns(
    video_path="data/input/videos/video.mp4",
    output_dir="data/output/shared",
    variant="shared"
)

# Segment-specific background variant
process_complete_pipeline_v2_columns(
    video_path="data/input/videos/video.mp4",
    output_dir="data/output/segment",
    variant="segment"
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
    video_path=Path("data/input/videos/video.mp4"),
    output_dir=Path("data/output/shared"),
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
           # Implement your background model creation strategy
           pass
   ```
3. Add to `v2_columns/__init__.py`
4. Add to `wrapper.py` if you want backward compatibility

## Benefits of This Structure

1. **Isolation**: Each variant is independent - changes don't affect others
2. **Reusability**: Common logic in `BasePipeline` is shared
3. **Extensibility**: Easy to add new variants
4. **Testability**: Each variant can be tested independently
5. **Backward Compatibility**: Wrapper maintains original interface



