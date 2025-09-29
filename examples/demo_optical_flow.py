#!/usr/bin/env python3
"""
Demo script for Optical Flow Fish Detection

This script demonstrates the complete optical flow pipeline:
1. Background subtraction
2. Image preprocessing  
3. Optical flow motion detection
4. Results visualization

Run this script to see the optical flow implementation in action.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from masking_methods.optical_flow.optical_flow_masker import OpticalFlowMasker
from optical_flow_video_processor import OpticalFlowVideoProcessor


def demo_optical_flow():
    """Demonstrate the optical flow fish detection pipeline."""
    
    print("=" * 60)
    print("OPTICAL FLOW FISH DETECTION DEMO")
    print("=" * 60)
    
    # Configuration
    video_path = "/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4"
    output_dir = "/Users/jasonli/FishDetection/output/optical_flow_demo"
    
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize optical flow processor
    print("\nInitializing optical flow processor...")
    processor = OpticalFlowVideoProcessor(
        bg_subtraction_threshold=25,
        optical_flow_params={
            'motion_thresh': 0.8,
            'bbox_thresh': 300,
            'nms_thresh': 0.1,
            'use_variable_threshold': True,
            'min_thresh': 0.3,
            'max_thresh': 1.0
        }
    )
    
    # Process video
    print("\nProcessing video with optical flow...")
    results = processor.process_video(
        video_path=video_path,
        output_dir=output_dir,
        num_frames=10,
        save_visualizations=True,
        save_detections=True
    )
    
    # Print results summary
    total_frames = len(results['frame_detections'])
    total_detections = sum(len(detections) for detections in results['frame_detections'])
    avg_detections = total_detections / total_frames if total_frames > 0 else 0
    
    print(f"\n" + "=" * 60)
    print("DEMO RESULTS")
    print("=" * 60)
    print(f"Total frames processed: {total_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per frame: {avg_detections:.2f}")
    
    # Show detection details
    print(f"\nDetection details:")
    for i, detections in enumerate(results['frame_detections']):
        if len(detections) > 0:
            print(f"  Frame {i:03d}: {len(detections)} detections")
            for j, detection in enumerate(detections):
                x1, y1, x2, y2, area = detection
                print(f"    Detection {j+1}: ({x1}, {y1}) to ({x2}, {y2}), area={area:.2f}")
    
    print(f"\nDemo complete! Check {output_dir} for results.")
    print("\nFiles generated:")
    print(f"  - visualizations/: Frame-by-frame visualizations")
    print(f"  - detections/: Detection data files")
    print(f"  - processing_summary.txt: Summary report")


def demo_optical_flow_parameters():
    """Demonstrate different optical flow parameters."""
    
    print("\n" + "=" * 60)
    print("OPTICAL FLOW PARAMETER DEMO")
    print("=" * 60)
    
    # Test different parameter combinations
    parameter_sets = [
        {"motion_thresh": 0.5, "bbox_thresh": 200, "name": "Sensitive"},
        {"motion_thresh": 1.0, "bbox_thresh": 400, "name": "Default"},
        {"motion_thresh": 1.5, "bbox_thresh": 600, "name": "Conservative"}
    ]
    
    video_path = "/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4"
    
    for params in parameter_sets:
        print(f"\nTesting {params['name']} parameters:")
        print(f"  Motion threshold: {params['motion_thresh']}")
        print(f"  Bbox threshold: {params['bbox_thresh']}")
        
        # Create processor with these parameters
        processor = OpticalFlowVideoProcessor(
            bg_subtraction_threshold=25,
            optical_flow_params={
                'motion_thresh': params['motion_thresh'],
                'bbox_thresh': params['bbox_thresh'],
                'nms_thresh': 0.1,
                'use_variable_threshold': True,
                'min_thresh': 0.3,
                'max_thresh': 1.0
            }
        )
        
        # Process a few frames
        try:
            results = processor.process_video(
                video_path=video_path,
                output_dir=f"/Users/jasonli/FishDetection/output/optical_flow_demo_{params['name'].lower()}",
                num_frames=5,
                save_visualizations=False,
                save_detections=False
            )
            
            total_detections = sum(len(detections) for detections in results['frame_detections'])
            avg_detections = total_detections / len(results['frame_detections'])
            
            print(f"  Results: {total_detections} total detections, {avg_detections:.2f} avg per frame")
            
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main demo function."""
    print("Optical Flow Fish Detection Demo")
    print("This demo shows the complete pipeline in action.\n")
    
    # Run main demo
    demo_optical_flow()
    
    # Run parameter comparison
    demo_optical_flow_parameters()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("The optical flow implementation successfully detects fish movement")
    print("using dense optical flow analysis combined with background subtraction.")
    print("\nKey features demonstrated:")
    print("  - Variable thresholding for perspective correction")
    print("  - Morphological operations for noise reduction")
    print("  - Non-maximum suppression for overlapping detections")
    print("  - Integration with existing background subtraction pipeline")
    print("\nCheck the output directories for detailed results!")


if __name__ == "__main__":
    main()
