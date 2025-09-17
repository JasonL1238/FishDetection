#!/usr/bin/env python3
"""Test script for optimized frame differencing detector targeting ~20 fish per frame."""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from fishdetect.balanced_frame_diff_detector import BalancedFrameDifferencingDetector


def test_optimized_frame_differencing(video_path: str, max_frames: int = 50):
    """Test optimized frame differencing with parameters tuned for ~20 fish per frame."""
    print(f"=== OPTIMIZED FRAME DIFFERENCING TEST ===")
    print(f"Video: {video_path}")
    print(f"Testing first {max_frames} frames")
    
    # Create optimized detector with parameters tuned for ~20 fish per frame
    detector = BalancedFrameDifferencingDetector(
        min_area=3,  # Very small minimum area
        max_area=100,  # Small maximum area
        min_aspect_ratio=0.8,  # Very flexible aspect ratio
        max_aspect_ratio=12.0,
        diff_threshold=12,  # Lower threshold for more sensitivity
        expected_fish_count=20,
        temporal_window=5,
        confidence_threshold=0.25  # Lower confidence threshold
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS")
    
    # Setup output
    output_dir = Path("optimized_frame_diff_output")
    output_dir.mkdir(exist_ok=True)
    
    frame_count = 0
    detection_counts = []
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detections = detector.process_frame(frame)
            detection_counts.append(len(detections))
            
            # Debug output for first few frames
            if frame_count < 10:
                print(f"\nFrame {frame_count}:")
                print(f"  Detections: {len(detections)}")
                
                if len(detections) > 0:
                    for i, det in enumerate(detections[:10]):  # Show first 10
                        print(f"    Detection {i}: area={det.area}, conf={det.confidence:.3f}, "
                              f"motion={det.motion_magnitude:.1f}, diff={det.frame_diff:.1f}, "
                              f"temp={det.temporal_consistency:.2f}")
                
                # Save debug visualization
                vis_frame = detector.visualize_detections(frame, detections)
                debug_path = output_dir / f"frame_{frame_count:03d}_optimized.png"
                cv2.imwrite(str(debug_path), vis_frame)
                print(f"  Debug image saved: {debug_path}")
            
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{max_frames} frames")
    
    finally:
        cap.release()
    
    # Analysis
    print(f"\n=== ANALYSIS ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Total detections: {sum(detection_counts)}")
    print(f"Average detections per frame: {np.mean(detection_counts):.2f}")
    print(f"Detection range: {min(detection_counts)} - {max(detection_counts)}")
    print(f"Frames with detections: {sum(1 for c in detection_counts if c > 0)}/{frame_count}")
    
    # Show detection distribution
    print(f"\nDetection distribution:")
    for i in range(0, max(detection_counts) + 1, 2):
        count = sum(1 for c in detection_counts if i <= c < i + 2)
        if count > 0:
            print(f"  {i}-{i+1} detections: {count} frames")
    
    # Get detector statistics
    stats = detector.get_detection_statistics()
    if stats:
        print(f"\nDetector Statistics:")
        print(f"  Average confidence: {stats['avg_confidence']:.3f}")
        print(f"  Average area: {stats['avg_area']:.1f} pixels")
        print(f"  Area std: {stats['area_std']:.1f} pixels")
    
    # Target analysis
    target_fish = 20
    avg_detections = np.mean(detection_counts)
    accuracy = (avg_detections / target_fish) * 100
    print(f"\nTarget Analysis:")
    print(f"  Target fish per frame: {target_fish}")
    print(f"  Actual average: {avg_detections:.2f}")
    print(f"  Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80 and accuracy <= 120:
        print(f"  ✓ Good accuracy for target!")
    elif accuracy < 80:
        print(f"  ⚠ Under-detecting (need more sensitive parameters)")
    else:
        print(f"  ⚠ Over-detecting (need more selective parameters)")
    
    # Show frames with closest to target
    target_frames = []
    for i, count in enumerate(detection_counts):
        if 15 <= count <= 25:  # Close to target
            target_frames.append((i, count))
    
    if target_frames:
        print(f"\nFrames closest to target (15-25 detections):")
        for frame_num, count in target_frames[:10]:  # Show first 10
            print(f"  Frame {frame_num}: {count} detections")
    
    print(f"\nDebug images saved to: {output_dir}")


if __name__ == "__main__":
    video_path = "input/Clutch1_20250804_122715.mp4"
    
    if not Path(video_path).exists():
        print(f"Error: Video file {video_path} not found")
        sys.exit(1)
    
    test_optimized_frame_differencing(video_path, max_frames=50)
