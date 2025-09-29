#!/usr/bin/env python3
"""
Test script for the specialized fish tracker.

This script tests the fish tracker that combines background subtraction
with YOLOv8 for better fish detection.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tracking_methods.yolo_tracking.fish_tracker import test_fish_tracker


def main():
    """Main test function."""
    # Video path
    video_path = "/Users/jasonli/FishDetection/data/input/videos/Clutch1_20250804_122715.mp4"
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Output directory
    output_dir = "/Users/jasonli/FishDetection/data/output/simple_tracking"
    
    print("Starting specialized fish tracker test...")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Duration: 10 seconds")
    print(f"Expected: 28 fish per frame")
    print("-" * 60)
    
    try:
        # Run the test
        stats, summary = test_fish_tracker(
            video_path=video_path,
            output_dir=output_dir,
            max_duration=10.0
        )
        
        print("\n" + "=" * 60)
        print("FISH TRACKER TEST RESULTS:")
        print("=" * 60)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Processing time: {stats['processing_time']:.2f} seconds")
        print(f"Average FPS: {stats['fps']:.2f}")
        print(f"Total fish detected: {stats['total_fish_detected']}")
        print(f"Average fish per frame: {stats['avg_fish_per_frame']:.1f}")
        print(f"Max fish in single frame: {stats['max_fish_in_frame']}")
        print(f"Output video: {output_dir}/fish_tracking_result.mp4")
        print(f"Detection data: {output_dir}/fish_detection_data.npz")
        
        # Compare with expected
        expected_fish = 28
        detected_fish = stats['avg_fish_per_frame']
        detection_rate = (detected_fish / expected_fish) * 100 if expected_fish > 0 else 0
        
        print(f"\nDetection Analysis:")
        print(f"Expected fish per frame: {expected_fish}")
        print(f"Detected fish per frame: {detected_fish:.1f}")
        print(f"Detection rate: {detection_rate:.1f}%")
        
        if detection_rate < 50:
            print(f"\nNote: Detection rate is low. This could be due to:")
            print(f"1. Fish are very small or have low contrast")
            print(f"2. Fish are moving too fast for background subtraction")
            print(f"3. Need to adjust detection parameters")
            print(f"4. May need custom training for this specific fish type")
        elif detection_rate < 80:
            print(f"\nNote: Detection rate is moderate. Consider:")
            print(f"1. Adjusting min_fish_area and max_fish_area parameters")
            print(f"2. Fine-tuning background subtraction parameters")
            print(f"3. Using a different YOLO model size")
        else:
            print(f"\nGreat! Detection rate is good.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
