#!/usr/bin/env python3
"""
Run Default Columns Pipeline

7 columns, 4 fish per column, global threshold search.
"""

from pathlib import Path
from .pipeline import DefaultBackgroundPipeline


def main():
    """Main execution."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/complete_pipeline_v2_columns_5min")
    
    pipeline = DefaultBackgroundPipeline(
        fps=20,
        num_columns=7,
        target_per_column=4
    )
    
    pipeline.process(
        video_path=video_path,
        output_dir=output_dir,
        duration_seconds=5 * 60
    )


if __name__ == "__main__":
    main()

