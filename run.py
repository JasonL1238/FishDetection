#!/usr/bin/env python3
"""
Run the custom grid fish detection pipeline.

Processes videos from data/input/videos/ and saves results to
data/output/.
"""

from pathlib import Path
import pims
from fishdetection import CustomGridPipeline


def get_video_duration(video_path: Path, fps: int = 20) -> float:
    """Get video duration in seconds."""
    video = pims.PyAVReaderIndexed(str(video_path))
    return len(video) / fps


def main():
    base_output_dir = Path("data/output")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    video_3min = Path(
        "data/input/videos/"
        "20250603_1553_tlf-inx_S5374_DOB250425_3minutetest.avi"
    )
    video_22min = Path(
        "data/input/videos/"
        "20250618_1127_S5403_DOB_06.18.25.avi"
    )

    fps = 20

    videos = [
        (video_3min, 3, "3-minute test video", "grid_3min"),
        (video_22min, 22, "22-minute video", "grid_22min"),
    ]

    for video_path, expected_min, description, folder in videos:
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            continue

        try:
            duration_sec = get_video_duration(video_path, fps)
            print(f"\nVideo: {video_path.name}")
            print(f"  Expected: ~{expected_min} min")
            print(f"  Actual:   {duration_sec / 60:.2f} min "
                  f"({duration_sec:.1f}s)")
        except Exception as e:
            print(f"Error reading video {video_path.name}: {e}")
            duration_sec = expected_min * 60

        output_dir = base_output_dir / folder

        print(f"\n{'=' * 60}")
        print(f"Running pipeline: {description}")
        print(f"Output: {output_dir}")
        print(f"{'=' * 60}\n")

        pipeline = CustomGridPipeline(
            fps=fps,
            num_columns=7,
            target_per_column=4,
            num_segments=7,
        )

        pipeline.process(
            video_path=video_path,
            output_dir=output_dir,
            duration_seconds=int(duration_sec),
        )

        print(f"\nCompleted: {description}")
        print(f"Results: {output_dir}\n")


if __name__ == "__main__":
    main()
