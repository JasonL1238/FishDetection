#!/usr/bin/env python3
"""
Run Custom Grid Pipeline

Custom 7x7 grid with specific horizontal lines, 1 fish per cell (largest blob).

Processes both the 22-minute video and 3-minute video with organized outputs.
"""

from pathlib import Path
import pims
from .pipeline import CustomGridPipeline


def get_video_duration(video_path: Path, fps: int = 20) -> float:
    """Get video duration in seconds."""
    video = pims.PyAVReaderIndexed(str(video_path))
    num_frames = len(video)
    duration_seconds = num_frames / fps
    return duration_seconds


def main():
    """Main execution - run on 22-minute and 3-minute videos."""
    # Match existing folder structure (note: CustonGrid has a typo but matches existing structure)
    base_output_dir = Path("data/output/CustonGrid")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Video paths
    video_3min = Path("data/input/videos/20250603_1553_tlf-inx_S5374_DOB250425_3minutetest.avi")
    video_22min = Path("data/input/videos/20250618_1127_S5403_DOB_06.18.25.avi")
    
    fps = 20
    
    # Videos to process: (video_path, expected_duration_minutes, description, folder_name)
    videos_to_process = [
        (video_3min, 3, "3-minute test video", "custom_grid_3min"),
        (video_22min, 22, "22-minute video", "custom_grid_22min"),
    ]
    
    for video_path, expected_duration_min, description, folder_name in videos_to_process:
        if not video_path.exists():
            print(f"⚠️  Warning: Video not found: {video_path}")
            continue
        
        # Get actual video duration
        try:
            actual_duration_seconds = get_video_duration(video_path, fps)
            actual_duration_minutes = actual_duration_seconds / 60
            print(f"\n📹 Video: {video_path.name}")
            print(f"   Expected: ~{expected_duration_min} minutes")
            print(f"   Actual: {actual_duration_minutes:.2f} minutes ({actual_duration_seconds:.1f} seconds)")
        except Exception as e:
            print(f"⚠️  Error reading video {video_path.name}: {e}")
            # Use expected duration as fallback
            actual_duration_seconds = expected_duration_min * 60
        
        # Create output directory matching the Clutch1 pattern (custom_grid_full_video style)
        output_dir = base_output_dir / folder_name
        
        print(f"\n{'='*80}")
        print(f"Running Custom Grid Pipeline")
        print(f"Video: {description}")
        print(f"File: {video_path.name}")
        print(f"Duration: {actual_duration_seconds:.1f} seconds ({actual_duration_seconds/60:.1f} minutes)")
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
            duration_seconds=int(actual_duration_seconds)
        )
        
        print(f"\n{'='*80}")
        print(f"✓ Completed: {description}")
        print(f"Results saved in: {output_dir}")
        print(f"{'='*80}\n")
    
    print(f"\n{'='*80}")
    print("🎉 All videos processed!")
    print(f"All results saved in: {base_output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

