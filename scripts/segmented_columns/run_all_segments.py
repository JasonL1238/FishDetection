#!/usr/bin/env python3
"""
Run segmented columns pipeline on all 4 segments of a 20-minute video.

Each segment is 5 minutes, and outputs to a separate folder in SegmentedOutputs.
"""

from pathlib import Path
from .pipeline import SegmentedColumnsPipeline


def main():
    """Main execution - run on all 4 segments."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    base_output_dir = Path("data/output/SegmentedOutputs")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 20 minutes = 1200 seconds = 24000 frames at 20 fps
    # Each segment is 5 minutes = 300 seconds = 6000 frames
    fps = 20
    segment_duration = 5 * 60  # 5 minutes in seconds
    num_segments = 4
    
    for seg_idx in range(num_segments):
        start_frame = seg_idx * 6000  # 6000 frames per segment
        output_dir = base_output_dir / f"columns_segmented_seg{seg_idx+1}_5min"
        
        print(f"\n{'='*80}")
        print(f"Processing Segment {seg_idx + 1}/{num_segments}")
        print(f"Start Frame: {start_frame}, Duration: {segment_duration} seconds")
        print(f"Output Directory: {output_dir}")
        print(f"{'='*80}\n")
        
        pipeline = SegmentedColumnsPipeline(
            fps=fps,
            num_columns=7,
            target_per_column=4,
            num_segments=7  # 7 temporal segments per 5-minute segment
        )
        
        pipeline.process(
            video_path=video_path,
            output_dir=output_dir,
            duration_seconds=segment_duration,
            start_frame=start_frame
        )
        
        print(f"\n✓ Completed Segment {seg_idx + 1}/{num_segments}\n")
    
    print(f"\n{'='*80}")
    print("All segments completed!")
    print(f"Results saved in: {base_output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

