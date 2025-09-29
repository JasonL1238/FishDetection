#!/usr/bin/env python3
"""
Apply background subtraction to the first 30 frames of a video
and save the results to a specified output folder.
"""

import numpy as np
import cv2
import os
import pims
from tqdm import tqdm

def create_background_model(video_path, frame_indices):
    """
    Create a background model using specified frame indices.
    
    Args:
        video_path: Path to the video file
        frame_indices: List of frame indices to use for background model
    
    Returns:
        Background model as numpy array
    """
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

def apply_background_subtraction(frame, background, threshold=25):
    """
    Apply background subtraction to a single frame.
    
    Args:
        frame: Input frame (grayscale)
        background: Background model
        threshold: Threshold for binary mask
    
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
    _, thresh = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    return thresh.astype(np.uint8)

def process_video_frames(video_path, output_dir, num_frames=30, background_frames=None):
    """
    Process the first num_frames of a video with background subtraction.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save output frames
        num_frames: Number of frames to process (default: 30)
        background_frames: Frame indices to use for background model
    """
    # Default background frame indices (from the original tracking program)
    if background_frames is None:
        background_frames = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create background model
    background = create_background_model(video_path, background_frames)
    
    # Load video for processing
    print("Loading video for frame processing...")
    video = pims.PyAVReaderIndexed(video_path)
    
    print(f"Processing first {num_frames} frames...")
    
    # Process each frame
    for i in tqdm(range(min(num_frames, len(video)))):
        # Get frame and convert to grayscale
        frame = video[i][:, :, 0]
        
        # Apply background subtraction
        bg_subtracted = apply_background_subtraction(frame, background)
        
        # Save original frame
        original_path = os.path.join(output_dir, f"frame_{i:03d}_original.png")
        cv2.imwrite(original_path, frame)
        
        # Save background subtracted frame
        bg_subtracted_path = os.path.join(output_dir, f"frame_{i:03d}_bg_subtracted.png")
        cv2.imwrite(bg_subtracted_path, bg_subtracted)
    
    print(f"Processing complete! Results saved to: {output_dir}")

def main():
    """Main function to run the background subtraction process."""
    # Configuration
    video_path = "/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4"
    output_dir = "/Users/jasonli/FishDetection/output/bg_subtracted_30_frames"
    num_frames = 30
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    print(f"Processing video: {video_path}")
    print(f"Output directory: {output_dir}")
    print(f"Number of frames to process: {num_frames}")
    
    # Process the video
    process_video_frames(video_path, output_dir, num_frames)
    
    print("Background subtraction complete!")

if __name__ == "__main__":
    main()
