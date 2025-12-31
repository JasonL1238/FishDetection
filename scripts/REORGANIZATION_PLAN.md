# Reorganization Plan

Each pipeline will be completely self-contained in its own folder:

1. **original_pipeline/** - Standalone script (no dependencies)
   - complete_pipeline_v2.py (standalone, no BasePipeline needed)

2. **default_columns/** - Default columns pipeline
   - run.py (script)
   - pipeline.py (DefaultBackgroundPipeline)
   - base_pipeline.py (BasePipeline - copied)
   - utils.py (utils - copied)

3. **segmented_columns/** - Segmented columns pipeline
   - run.py (single run script)
   - run_all_segments.py (batch processing)
   - run_segment4.py (single segment)
   - pipeline.py (SegmentedColumnsPipeline)
   - base_pipeline.py (BasePipeline - copied)
   - utils.py (utils - copied, with count_fish_per_column)

4. **half_sectioned/** - Half-sectioned pipeline
   - run.py (script)
   - pipeline.py (HalfSectionedPipeline)
   - base_pipeline.py (BasePipeline - copied)
   - utils.py (utils - copied, but may not need count_fish_per_column)

Each folder is completely independent - no shared dependencies.
