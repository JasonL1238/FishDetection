#!/usr/bin/env python3
"""
Optical Flow Video Processor for Fish Detection

This script processes videos using optical flow motion detection combined with
background subtraction preprocessing. It breaks down videos into frames,
applies background subtraction, and then uses optical flow to detect fish movement.

Based on the tutorial by Isaac Berrios on optical flow motion detection.
"""

import cv2
import numpy as np
import os
import pims
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from masking_methods.optical_flow.optical_flow_masker import OpticalFlowMasker
from image_pre.preprocessor import ImagePreprocessor


class OpticalFlowVideoProcessor:
    """
    Video processor that combines background subtraction with optical flow detection.
    """
    
    def __init__(self, 
                 bg_subtraction_threshold: int = 25,
                 optical_flow_params: dict = None,
                 preprocess_params: dict = None):
        """
        Initialize the video processor.
        
        Args:
            bg_subtraction_threshold: Threshold for background subtraction
            optical_flow_params: Parameters for optical flow detection
            preprocess_params: Parameters for image preprocessing
        """
        self.bg_subtraction_threshold = bg_subtraction_threshold
        
        # Default optical flow parameters
        if optical_flow_params is None:
            optical_flow_params = {
                'motion_thresh': 1.0,
                'bbox_thresh': 400,
                'nms_thresh': 0.1,
                'use_variable_threshold': True,
                'min_thresh': 0.3,
                'max_thresh': 1.0
            }
        
        # Default preprocessing parameters
        if preprocess_params is None:
            preprocess_params = {
                'gaussian_kernel_size': (5, 5),
                'gaussian_sigma': 1.0,
                'sharpen_kernel_strength': 1.0
            }
        
        # Initialize components
        self.optical_flow_masker = OpticalFlowMasker(**optical_flow_params)
        self.preprocessor = ImagePreprocessor(**preprocess_params)
        
        # Background model
        self.background_model = None
        
    def create_background_model(self, video_path: str, frame_indices: list = None) -> np.ndarray:
        """
        Create a background model using specified frame indices.
        
        Args:
            video_path: Path to the video file
            frame_indices: List of frame indices to use for background model
            
        Returns:
            Background model as numpy array
        """
        if frame_indices is None:
            # Default background frame indices (from the original tracking program)
            frame_indices = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]
        
        print("Loading video for background model creation...")
        video = pims.PyAVReaderIndexed(video_path)
        
        # Extract frames for background model
        frames = []
        for idx in frame_indices:
            if idx < len(video):
                frame = video[idx][:, :, 0]  # Take only the first channel (grayscale)
                frames.append(frame)
        
        # Create background model using median
        print(f"Creating background model from {len(frames)} frames...")
        bg = np.median(np.stack(frames, axis=2), axis=2)
        
        # Apply Gaussian blur to smooth the background
        bg_smoothed = cv2.GaussianBlur(bg.astype(np.float64), (3, 3), 0)
        
        return bg_smoothed
    
    def apply_background_subtraction(self, frame: np.ndarray, background: np.ndarray) -> np.ndarray:
        """
        Apply background subtraction to a single frame.
        
        Args:
            frame: Input frame (grayscale)
            background: Background model
            
        Returns:
            Background subtracted frame
        """
        # Convert frame to float64 for processing
        frame_float = frame.astype(np.float64)
        
        # Apply Gaussian blur to the frame
        frame_blurred = cv2.GaussianBlur(frame_float, (3, 3), 0)
        
        # Calculate absolute difference
        mask = np.abs(background - frame_blurred)
        
        # Apply threshold to create binary mask
        _, thresh = cv2.threshold(mask, self.bg_subtraction_threshold, 255, cv2.THRESH_BINARY)
        
        return thresh.astype(np.uint8)
    
    def process_video(self, 
                     video_path: str, 
                     output_dir: str, 
                     num_frames: int = 30,
                     save_visualizations: bool = True,
                     save_detections: bool = True) -> dict:
        """
        Process a video with optical flow motion detection.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save output
            num_frames: Number of frames to process
            save_visualizations: Whether to save visualization images
            save_detections: Whether to save detection data
            
        Returns:
            Dictionary with processing results
        """
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        
        if save_visualizations:
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)
        
        if save_detections:
            det_dir = os.path.join(output_dir, "detections")
            os.makedirs(det_dir, exist_ok=True)
        
        # Create background model
        self.background_model = self.create_background_model(video_path)
        
        # Load video for processing
        print("Loading video for frame processing...")
        video = pims.PyAVReaderIndexed(video_path)
        
        print(f"Processing first {num_frames} frames...")
        
        # Storage for results
        results = {
            'frame_detections': [],
            'optical_flow_visualizations': [],
            'motion_masks': [],
            'background_subtracted_frames': []
        }
        
        # Process each frame
        for i in tqdm(range(min(num_frames, len(video)))):
            # Get frame and convert to grayscale
            frame = video[i][:, :, 0]
            
            # Apply background subtraction
            bg_subtracted = self.apply_background_subtraction(frame, self.background_model)
            
            # Apply preprocessing to background subtracted frame
            preprocessed = self.preprocessor.preprocess_image(bg_subtracted)
            
            # Apply optical flow detection
            motion_mask, detections = self.optical_flow_masker.process_frame(preprocessed)
            
            # Store results
            results['frame_detections'].append(detections)
            results['motion_masks'].append(motion_mask)
            results['background_subtracted_frames'].append(bg_subtracted)
            
            # Save visualizations
            if save_visualizations:
                self._save_visualizations(i, frame, bg_subtracted, preprocessed, 
                                        motion_mask, detections, vis_dir)
            
            # Save detection data
            if save_detections:
                self._save_detection_data(i, detections, det_dir)
        
        # Save summary results
        self._save_summary_results(results, output_dir)
        
        print(f"Processing complete! Results saved to: {output_dir}")
        return results
    
    def _save_visualizations(self, frame_idx: int, original_frame: np.ndarray, 
                           bg_subtracted: np.ndarray, preprocessed: np.ndarray,
                           motion_mask: np.ndarray, detections: list, vis_dir: str):
        """Save visualization images for a frame."""
        
        # Create composite visualization
        # Resize all images to same size for stacking
        height, width = original_frame.shape
        
        # Create visualization with detections
        vis_frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2, area = detection
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Area: {area:.0f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Create 2x2 grid visualization
        # Top row: Original and Background Subtracted
        # Bottom row: Preprocessed and Motion Mask
        
        # Resize images to same size
        vis_size = (width//2, height//2)
        
        orig_resized = cv2.resize(original_frame, vis_size)
        bg_sub_resized = cv2.resize(bg_subtracted, vis_size)
        preprocessed_resized = cv2.resize(preprocessed, vis_size)
        motion_mask_resized = cv2.resize(motion_mask, vis_size)
        
        # Convert to BGR for consistent display
        orig_bgr = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)
        bg_sub_bgr = cv2.cvtColor(bg_sub_resized, cv2.COLOR_GRAY2BGR)
        preprocessed_bgr = cv2.cvtColor(preprocessed_resized, cv2.COLOR_GRAY2BGR)
        motion_mask_bgr = cv2.cvtColor(motion_mask_resized, cv2.COLOR_GRAY2BGR)
        
        # Create grid
        top_row = np.hstack([orig_bgr, bg_sub_bgr])
        bottom_row = np.hstack([preprocessed_bgr, motion_mask_bgr])
        grid = np.vstack([top_row, bottom_row])
        
        # Add labels
        cv2.putText(grid, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(grid, "BG Subtracted", (width//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(grid, "Preprocessed", (10, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(grid, "Motion Mask", (width//2 + 10, height//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        vis_path = os.path.join(vis_dir, f"frame_{frame_idx:03d}_visualization.png")
        cv2.imwrite(vis_path, grid)
        
        # Save individual frames
        cv2.imwrite(os.path.join(vis_dir, f"frame_{frame_idx:03d}_original.png"), original_frame)
        cv2.imwrite(os.path.join(vis_dir, f"frame_{frame_idx:03d}_bg_subtracted.png"), bg_subtracted)
        cv2.imwrite(os.path.join(vis_dir, f"frame_{frame_idx:03d}_preprocessed.png"), preprocessed)
        cv2.imwrite(os.path.join(vis_dir, f"frame_{frame_idx:03d}_motion_mask.png"), motion_mask)
    
    def _save_detection_data(self, frame_idx: int, detections: list, det_dir: str):
        """Save detection data for a frame."""
        det_path = os.path.join(det_dir, f"frame_{frame_idx:03d}_detections.txt")
        
        with open(det_path, 'w') as f:
            f.write(f"Frame {frame_idx} Detections:\n")
            f.write("Format: x1, y1, x2, y2, area\n")
            for i, detection in enumerate(detections):
                x1, y1, x2, y2, area = detection
                f.write(f"Detection {i+1}: {x1}, {y1}, {x2}, {y2}, {area:.2f}\n")
    
    def _save_summary_results(self, results: dict, output_dir: str):
        """Save summary results."""
        summary_path = os.path.join(output_dir, "processing_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("Optical Flow Video Processing Summary\n")
            f.write("=" * 40 + "\n\n")
            
            total_frames = len(results['frame_detections'])
            total_detections = sum(len(detections) for detections in results['frame_detections'])
            
            f.write(f"Total frames processed: {total_frames}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Average detections per frame: {total_detections/total_frames:.2f}\n\n")
            
            f.write("Detection summary by frame:\n")
            for i, detections in enumerate(results['frame_detections']):
                f.write(f"Frame {i:03d}: {len(detections)} detections\n")
                for j, detection in enumerate(detections):
                    x1, y1, x2, y2, area = detection
                    f.write(f"  Detection {j+1}: ({x1}, {y1}) to ({x2}, {y2}), area={area:.2f}\n")


def main():
    """Main function to run the optical flow video processor."""
    parser = argparse.ArgumentParser(description='Process video with optical flow motion detection')
    parser.add_argument('--video', 
                       default='/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4',
                       help='Path to input video')
    parser.add_argument('--output', 
                       default='/Users/jasonli/FishDetection/output/optical_flow_results',
                       help='Output directory')
    parser.add_argument('--frames', type=int, default=30,
                       help='Number of frames to process')
    parser.add_argument('--bg-threshold', type=int, default=25,
                       help='Background subtraction threshold')
    parser.add_argument('--motion-thresh', type=float, default=1.0,
                       help='Optical flow motion threshold')
    parser.add_argument('--bbox-thresh', type=int, default=400,
                       help='Minimum bounding box area threshold')
    parser.add_argument('--no-visualizations', action='store_true',
                       help='Skip saving visualization images')
    parser.add_argument('--no-detections', action='store_true',
                       help='Skip saving detection data')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found at {args.video}")
        return
    
    print(f"Processing video: {args.video}")
    print(f"Output directory: {args.output}")
    print(f"Number of frames to process: {args.frames}")
    
    # Configure optical flow parameters
    optical_flow_params = {
        'motion_thresh': args.motion_thresh,
        'bbox_thresh': args.bbox_thresh,
        'nms_thresh': 0.1,
        'use_variable_threshold': True,
        'min_thresh': 0.3,
        'max_thresh': 1.0
    }
    
    # Create processor
    processor = OpticalFlowVideoProcessor(
        bg_subtraction_threshold=args.bg_threshold,
        optical_flow_params=optical_flow_params
    )
    
    # Process video
    results = processor.process_video(
        video_path=args.video,
        output_dir=args.output,
        num_frames=args.frames,
        save_visualizations=not args.no_visualizations,
        save_detections=not args.no_detections
    )
    
    print("Optical flow video processing complete!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
