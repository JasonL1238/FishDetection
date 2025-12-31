#!/usr/bin/env python3
"""
Run Half-Sectioned Pipeline

14 sections (7 top + 7 bottom), 2 fish per section, independent per-section threshold search, 7 temporal segments.
"""

from pathlib import Path
from .pipeline import HalfSectionedPipeline


def main():
    """Main execution - run on full 20-minute video."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/HalfSectioned/half_sectioned_full_video")
    
    # 20 minutes = 1200 seconds = 24000 frames at 20 fps
    fps = 20
    duration_seconds = 20 * 60  # 20 minutes
    
    print(f"\n{'='*80}")
    print(f"Running Half-Sectioned Pipeline on Full Video")
    print(f"Duration: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}\n")
    
    pipeline = HalfSectionedPipeline(
        fps=fps,
        num_columns=7,  # Not used but required by BasePipeline
        target_per_column=4,  # Not used but required by BasePipeline
        num_segments=7,
        num_sections_per_half=7
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

