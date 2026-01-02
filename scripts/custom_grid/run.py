#!/usr/bin/env python3
"""
Run Custom Grid Pipeline

Custom 7x7 grid with specific horizontal lines, 1 fish per cell (largest blob).
"""

from pathlib import Path
from .pipeline import CustomGridPipeline


def main():
    """Main execution - run on full 20-minute video."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/custom_grid_full_video")
    
    # Full video: 20 minutes = 1200 seconds = 24000 frames at 20 fps
    fps = 20
    duration_seconds = 20 * 60  # 20 minutes
    
    print(f"\n{'='*80}")
    print(f"Running Custom Grid Pipeline on Full Video")
    print(f"Duration: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}\n")
    
    pipeline = CustomGridPipeline(
        fps=fps,
        num_columns=7,  # Not used but required by BasePipeline
        target_per_column=4,  # Not used but required by BasePipeline
        num_segments=7
    )
    
    pipeline.process(
        video_path=video_path,
        output_dir=output_dir,
        duration_seconds=duration_seconds
    )
    
    print(f"\n{'='*80}")
    print("Full video processing completed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

