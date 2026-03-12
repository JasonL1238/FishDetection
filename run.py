#!/usr/bin/env python3
"""
Run the custom grid fish detection pipeline.

Usage:
    python run.py <video_path> [--output <output_dir>] [--fps <fps>]

Examples:
    python run.py data/input/videos/Clutch1_20250804_122715.mp4
    python run.py data/input/videos/my_video.avi --output data/output/my_run
    python run.py data/input/videos/my_video.avi --fps 30
"""

import argparse
from pathlib import Path
import pims
from fishdetection import CustomGridPipeline


def get_video_duration(video_path: Path, fps: int = 20) -> float:
    """Get video duration in seconds."""
    video = pims.PyAVReaderIndexed(str(video_path))
    return len(video) / fps


def main():
    parser = argparse.ArgumentParser(
        description="Run the custom grid fish detection pipeline."
    )
    parser.add_argument(
        "video", type=Path, help="Path to the input video file"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (default: data/output/<video_stem>)",
    )
    parser.add_argument(
        "--fps", type=int, default=20, help="Frames per second (default: 20)"
    )
    args = parser.parse_args()

    video_path = args.video
    if not video_path.exists():
        parser.error(f"Video not found: {video_path}")

    output_dir = args.output or Path("data/output") / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        duration_sec = get_video_duration(video_path, args.fps)
        print(f"\nVideo: {video_path.name}")
        print(f"  Duration: {duration_sec / 60:.2f} min "
              f"({duration_sec:.1f}s)")
    except Exception as e:
        print(f"Error reading video {video_path.name}: {e}")
        return

    print(f"\n{'=' * 60}")
    print(f"Running pipeline: {video_path.name}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")

    pipeline = CustomGridPipeline(
        fps=args.fps,
        num_columns=7,
        target_per_column=4,
        num_segments=7,
    )

    pipeline.process(
        video_path=video_path,
        output_dir=output_dir,
        duration_seconds=int(duration_sec),
    )

    print(f"\nCompleted: {video_path.name}")
    print(f"Results: {output_dir}\n")


if __name__ == "__main__":
    main()
