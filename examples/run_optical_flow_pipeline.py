#!/usr/bin/env python3
"""
Complete Optical Flow Pipeline for Fish Detection

This script runs the complete pipeline:
1. Background subtraction (if not already done)
2. Image preprocessing
3. Optical flow motion detection
4. Results visualization and analysis

Based on the tutorial by Isaac Berrios on optical flow motion detection.
"""

import os
import sys
import argparse
from pathlib import Path
import subprocess

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from optical_flow_video_processor import OpticalFlowVideoProcessor


def check_prerequisites():
    """Check if all required files and directories exist."""
    print("Checking prerequisites...")
    
    # Check if video exists
    video_path = "/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    # Check if background subtracted frames exist
    bg_frames_dir = "/Users/jasonli/FishDetection/output/bg_subtracted_30_frames"
    if not os.path.exists(bg_frames_dir):
        print(f"Background subtracted frames not found at {bg_frames_dir}")
        print("Running background subtraction first...")
        
        # Run background subtraction script
        try:
            subprocess.run([sys.executable, "apply_bg_subtraction_30_frames.py"], check=True)
            print("Background subtraction completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error running background subtraction: {e}")
            return False
    
    print("All prerequisites met!")
    return True


def run_optical_flow_pipeline(num_frames: int = 30, 
                             motion_thresh: float = 1.0,
                             bbox_thresh: int = 400,
                             bg_threshold: int = 25,
                             save_visualizations: bool = True,
                             save_detections: bool = True):
    """
    Run the complete optical flow pipeline.
    
    Args:
        num_frames: Number of frames to process
        motion_thresh: Optical flow motion threshold
        bbox_thresh: Minimum bounding box area threshold
        bg_threshold: Background subtraction threshold
        save_visualizations: Whether to save visualization images
        save_detections: Whether to save detection data
    """
    print("=" * 60)
    print("OPTICAL FLOW FISH DETECTION PIPELINE")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("Prerequisites check failed. Exiting.")
        return
    
    # Configuration
    video_path = "/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4"
    output_dir = "/Users/jasonli/FishDetection/output/optical_flow_pipeline_results"
    
    print(f"\nProcessing Configuration:")
    print(f"  Video: {video_path}")
    print(f"  Output: {output_dir}")
    print(f"  Frames: {num_frames}")
    print(f"  Motion Threshold: {motion_thresh}")
    print(f"  Bbox Threshold: {bbox_thresh}")
    print(f"  BG Threshold: {bg_threshold}")
    print(f"  Save Visualizations: {save_visualizations}")
    print(f"  Save Detections: {save_detections}")
    
    # Configure optical flow parameters
    optical_flow_params = {
        'motion_thresh': motion_thresh,
        'bbox_thresh': bbox_thresh,
        'nms_thresh': 0.1,
        'use_variable_threshold': True,
        'min_thresh': 0.3,
        'max_thresh': 1.0
    }
    
    # Create processor
    print(f"\nInitializing optical flow processor...")
    processor = OpticalFlowVideoProcessor(
        bg_subtraction_threshold=bg_threshold,
        optical_flow_params=optical_flow_params
    )
    
    # Process video
    print(f"\nProcessing video with optical flow...")
    try:
        results = processor.process_video(
            video_path=video_path,
            output_dir=output_dir,
            num_frames=num_frames,
            save_visualizations=save_visualizations,
            save_detections=save_detections
        )
        
        # Print summary
        total_frames = len(results['frame_detections'])
        total_detections = sum(len(detections) for detections in results['frame_detections'])
        avg_detections = total_detections / total_frames if total_frames > 0 else 0
        
        print(f"\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Total frames processed: {total_frames}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {avg_detections:.2f}")
        print(f"Results saved to: {output_dir}")
        
        # Show detection summary by frame
        print(f"\nDetection summary by frame:")
        for i, detections in enumerate(results['frame_detections']):
            if len(detections) > 0:
                print(f"  Frame {i:03d}: {len(detections)} detections")
                for j, detection in enumerate(detections):
                    x1, y1, x2, y2, area = detection
                    print(f"    Detection {j+1}: ({x1}, {y1}) to ({x2}, {y2}), area={area:.2f}")
        
        print(f"\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return


def run_quick_test():
    """Run a quick test on a few frames."""
    print("Running quick test on 5 frames...")
    
    # Check if background subtracted frames exist
    bg_frames_dir = "/Users/jasonli/FishDetection/output/bg_subtracted_30_frames"
    if not os.path.exists(bg_frames_dir):
        print("Background subtracted frames not found. Please run background subtraction first.")
        return
    
    # Run test
    try:
        subprocess.run([sys.executable, "test_optical_flow.py"], check=True)
        print("Quick test completed! Check output/optical_flow_test/ for results.")
    except subprocess.CalledProcessError as e:
        print(f"Error running quick test: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run optical flow fish detection pipeline')
    parser.add_argument('--frames', type=int, default=30,
                       help='Number of frames to process (default: 30)')
    parser.add_argument('--motion-thresh', type=float, default=1.0,
                       help='Optical flow motion threshold (default: 1.0)')
    parser.add_argument('--bbox-thresh', type=int, default=400,
                       help='Minimum bounding box area threshold (default: 400)')
    parser.add_argument('--bg-threshold', type=int, default=25,
                       help='Background subtraction threshold (default: 25)')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip saving visualization images')
    parser.add_argument('--no-detections', action='store_true',
                       help='Skip saving detection data')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test on 5 frames only')
    
    args = parser.parse_args()
    
    if args.quick_test:
        run_quick_test()
    else:
        run_optical_flow_pipeline(
            num_frames=args.frames,
            motion_thresh=args.motion_thresh,
            bbox_thresh=args.bbox_thresh,
            bg_threshold=args.bg_threshold,
            save_visualizations=not args.no_visualizations,
            save_detections=not args.no_detections
        )


if __name__ == "__main__":
    main()
