#!/usr/bin/env python3
"""
Complete Pipeline V2 with Column-Based Detection: 7 columns, 4 fish per column

This pipeline uses:
- V2 Background Subtraction (threshold=15 + morphological closing)
- HSV Masking for bright fish detection
- Column-based adaptive detection: divides frame into 7 columns, targets 4 fish per column
- Draws column boundaries on output video

NOTE: This file is maintained for backward compatibility.
For new code, use the modular pipeline structure in examples/pipelines/
The original implementation has been moved to examples/pipelines/v2_columns/default_bg.py
"""

import sys
from pathlib import Path

# Import from new modular structure
sys.path.append(str(Path(__file__).parent))
from pipelines.v2_columns.wrapper import process_complete_pipeline_v2_columns

# Re-export for backward compatibility
__all__ = ['process_complete_pipeline_v2_columns']


def main():
    """Main execution."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/complete_pipeline_v2_columns_5min")
    
    process_complete_pipeline_v2_columns(
        video_path=video_path,
        output_dir=output_dir,
        fps=20,
        num_columns=7,
        target_per_column=4,
        duration_seconds=5 * 60,
        variant="default"
    )


if __name__ == "__main__":
    main()
