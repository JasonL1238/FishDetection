#!/usr/bin/env python3
"""
Run segmented columns pipeline on segment 4 only (15-20 minutes).
"""

from pathlib import Path
from .pipeline import SegmentedColumnsPipeline


def main():
    """Main execution - run segment 4 only."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    base_output_dir = Path("data/output/SegmentedOutputs")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Segment 4: 15-20 minutes (frames 18000-23999)
    fps = 20
    segment_duration = 5 * 60  # 5 minutes in seconds
    seg_idx = 3  # 0-indexed, so segment 4 is index 3
    start_frame = seg_idx * 6000  # 18000 frames
    output_dir = base_output_dir / f"columns_segmented_seg{seg_idx+1}_5min"
    
    print(f"\n{'='*80}")
    print(f"Processing Segment 4/4")
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
    
    print(f"\n{'='*80}")
    print("Segment 4 completed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

