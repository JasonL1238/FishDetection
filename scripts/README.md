# Pipeline Organization

Each pipeline is completely self-contained in its own folder. Scripts and pipeline implementations that belong together are grouped together.

## Structure

```
examples/
├── original_pipeline/          # Standalone original pipeline (no dependencies)
│   ├── complete_pipeline_v2.py
│   └── README.md
│
├── default_columns/            # Default columns pipeline (self-contained)
│   ├── run.py                  # Script to run the pipeline
│   ├── pipeline.py             # DefaultBackgroundPipeline
│   ├── base_pipeline.py        # BasePipeline (local copy)
│   ├── utils.py                # Utils (local copy)
│   └── README.md
│
├── segmented_columns/          # Segmented columns pipeline (self-contained)
│   ├── run.py                  # Single run script
│   ├── run_all_segments.py     # Batch processing script
│   ├── run_segment4.py         # Single segment script
│   ├── pipeline.py             # SegmentedColumnsPipeline
│   ├── base_pipeline.py        # BasePipeline (local copy)
│   ├── utils.py                # Utils (local copy)
│   └── README.md
│
└── half_sectioned/            # Half-sectioned pipeline (self-contained)
    ├── run.py                  # Script to run the pipeline
    ├── pipeline.py             # HalfSectionedPipeline
    ├── base_pipeline.py        # BasePipeline (local copy)
    ├── utils.py                # Utils (local copy)
    └── README.md
```

## Running Pipelines

Each pipeline can be run independently:

```bash
# Original pipeline (standalone, no dependencies)
python examples/original_pipeline/complete_pipeline_v2.py

# Default columns (self-contained)
python examples/default_columns/run.py

# Segmented columns - single run (self-contained)
python examples/segmented_columns/run.py

# Segmented columns - all 4 segments (self-contained)
python examples/segmented_columns/run_all_segments.py

# Segmented columns - segment 4 only (self-contained)
python examples/segmented_columns/run_segment4.py

# Half-sectioned (self-contained)
python examples/half_sectioned/run.py
```

## Key Points

- **Each folder is completely independent** - no shared dependencies between pipelines
- **All imports are local** - each folder has its own copies of BasePipeline and utils
- **Scripts that use the same pipeline are together** - e.g., all segmented_columns scripts are in one folder
- **Pipeline implementations are with their scripts** - pipeline.py is in the same folder as run.py

See each folder's README.md for more details about that specific pipeline.

