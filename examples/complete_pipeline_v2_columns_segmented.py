#!/usr/bin/env python3
"""
Complete Pipeline V2 with Segmented Column-Based Detection

This pipeline:
- Breaks the video into 4 temporal segments
- For each segment, processes each column independently
- For each column, adaptively adjusts min_size threshold to get 4 fish independently
- Each column's threshold is adjusted independently of other columns

This is similar to complete_pipeline_v2_columns but with:
1. Temporal segmentation (4 pieces)
2. Independent per-column processing with adaptive thresholds
"""

import sys
from pathlib import Path

# Import from modular structure
sys.path.append(str(Path(__file__).parent))
from pipelines.v2_columns.wrapper import process_complete_pipeline_v2_columns


def main():
    """Main execution."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/segmented_columns_10sec")
    
    process_complete_pipeline_v2_columns(
        video_path=video_path,
        output_dir=output_dir,
        fps=20,
        num_columns=7,
        target_per_column=4,
        duration_seconds=10,  # 10 seconds
        variant="segmented"  # Use the new segmented variant
    )


if __name__ == "__main__":
    main()

