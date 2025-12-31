#!/usr/bin/env python3
"""
Run Segmented Columns Pipeline (single run)

7 columns, 4 fish per column, independent per-column threshold search, 7 temporal segments.
"""

from pathlib import Path
from .pipeline import SegmentedColumnsPipeline


def main():
    """Main execution."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/segmented_columns_10sec")
    
    pipeline = SegmentedColumnsPipeline(
        fps=20,
        num_columns=7,
        target_per_column=4,
        num_segments=7
    )
    
    pipeline.process(
        video_path=video_path,
        output_dir=output_dir,
        duration_seconds=10
    )


if __name__ == "__main__":
    main()

