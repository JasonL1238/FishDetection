#!/usr/bin/env python3
"""
Test script for YOLOv8 fish tracking.

This script tests the YOLOv8 tracking implementation on the fish video
for the first 10 seconds.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tracking_methods.yolo_tracking.yolo_tracker import test_yolo_tracking


def main():
    """Main test function."""
    # Video path
    video_path = "/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Output directory
    output_dir = "/Users/jasonli/FishDetection/output/yolo_tracking_test"
    
    print("Starting YOLOv8 fish tracking test...")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Duration: 10 seconds")
    print("-" * 50)
    
    try:
        # Run the test
        stats = test_yolo_tracking(
            video_path=video_path,
            output_dir=output_dir,
            max_duration=10.0
        )
        
        print("\n" + "=" * 50)
        print("TEST RESULTS:")
        print("=" * 50)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        print(f"Average FPS: {stats['fps']:.2f}")
        print(f"Unique tracks detected: {stats['unique_tracks']}")
        print(f"Output video: {output_dir}/yolo_tracking_result.mp4")
        print(f"Track data: {output_dir}/track_data.npz")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
