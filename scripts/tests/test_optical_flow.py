#!/usr/bin/env python3
"""
Test script for optical flow motion detection on background subtracted frames.

This script tests the optical flow implementation on the preprocessed frames
from the background subtraction step.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from masking_methods.optical_flow.optical_flow_masker import OpticalFlowMasker


def test_optical_flow_on_frames(input_dir: str, output_dir: str, num_frames: int = 10):
    """
    Test optical flow detection on background subtracted frames.
    
    Args:
        input_dir: Directory containing background subtracted frames
        output_dir: Directory to save results
        num_frames: Number of frames to process
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize optical flow masker
    masker = OpticalFlowMasker(
        motion_thresh=1.0,
        bbox_thresh=400,
        nms_thresh=0.1,
        use_variable_threshold=True,
        min_thresh=0.3,
        max_thresh=1.0
    )
    
    print(f"Testing optical flow on {num_frames} frames from {input_dir}")
    
    # Process frames
    for i in range(num_frames):
        # Load background subtracted frame
        frame_path = os.path.join(input_dir, f"frame_{i:03d}_bg_subtracted.png")
        
        if not os.path.exists(frame_path):
            print(f"Frame {i} not found, skipping...")
            continue
        
        # Load frame
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Could not load frame {i}")
            continue
        
        # Process with optical flow
        motion_mask, detections = masker.process_frame(frame)
        
        # Create visualization
        vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2, area = detection
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Area: {area:.0f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Create side-by-side comparison
        motion_mask_colored = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        comparison = np.hstack([vis_frame, motion_mask_colored])
        
        # Add labels
        cv2.putText(comparison, "Original + Detections", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(comparison, "Motion Mask", (frame.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save results
        output_path = os.path.join(output_dir, f"frame_{i:03d}_optical_flow.png")
        cv2.imwrite(output_path, comparison)
        
        print(f"Frame {i}: {len(detections)} detections")
        for j, detection in enumerate(detections):
            x1, y1, x2, y2, area = detection
            print(f"  Detection {j+1}: ({x1}, {y1}) to ({x2}, {y2}), area={area:.2f}")
    
    print(f"Test complete! Results saved to: {output_dir}")


def main():
    """Main function."""
    # Configuration
    input_dir = "/Users/jasonli/FishDetection/output/bg_subtracted_30_frames"
    output_dir = "/Users/jasonli/FishDetection/output/optical_flow_test"
    num_frames = 10
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        print("Please run the background subtraction script first.")
        return
    
    # Run test
    test_optical_flow_on_frames(input_dir, output_dir, num_frames)


if __name__ == "__main__":
    main()
