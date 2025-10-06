#!/usr/bin/env python3
"""
HSV Analysis for First 10 Seconds of Video

This script performs HSV analysis on the first 10 seconds of the video,
generating both individual frame analysis images and a complete video output
with HSV masking applied.
"""

import cv2
import numpy as np
import pims
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from tracking_methods.background_subtraction.background_subtraction_masker import BackgroundSubtractionMasker


def create_background_model(video_path, frame_indices=None):
    """
    Create background model from video using median of selected frames.
    
    Args:
        video_path: Path to video file
        frame_indices: List of frame indices to use for background model
        
    Returns:
        Background model as numpy array
    """
    print(f"Loading video: {video_path}")
    video = pims.PyAVReaderIndexed(str(video_path))
    
    if frame_indices is None:
        # Use frames distributed throughout the video
        total_frames = len(video)
        frame_indices = [
            int(total_frames * 0.1),
            int(total_frames * 0.2),
            int(total_frames * 0.3),
            int(total_frames * 0.4),
            int(total_frames * 0.5),
            int(total_frames * 0.6),
            int(total_frames * 0.7),
            int(total_frames * 0.8),
            int(total_frames * 0.9)
        ]
    
    print(f"Using frame indices for background: {frame_indices}")
    
    # Extract frames
    frames = []
    for idx in frame_indices:
        if idx < len(video):
            frame = video[idx]
            if len(frame.shape) == 3:
                gray_frame = frame[:, :, 0]  # Take first channel
            else:
                gray_frame = frame
            frames.append(gray_frame)
    
    # Create background model using median
    background_model = np.median(np.stack(frames, axis=2), axis=2)
    
    # Apply Gaussian blur
    background_model = cv2.GaussianBlur(background_model.astype(np.float64), (3, 3), 0)
    
    print(f"Background model created with shape: {background_model.shape}")
    return background_model


def apply_background_subtraction(frame, background_model, threshold=25):
    """
    Apply background subtraction to a frame.
    
    Args:
        frame: Input frame (grayscale)
        background_model: Background model
        threshold: Threshold for binary mask creation
        
    Returns:
        Background subtracted frame
    """
    # Calculate absolute difference
    diff = np.abs(frame.astype(np.float64) - background_model)
    
    # Apply threshold
    mask = (diff > threshold).astype(np.uint8) * 255
    
    return mask


def process_10_seconds_hsv_analysis(video_path, output_dir, fps=20):
    """
    Process first 10 seconds of video with HSV analysis.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save output
        fps: Video frame rate
    """
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Calculate number of frames for 10 seconds
    num_frames = fps * 10  # 10 seconds
    
    # Create background model
    print("Creating background model...")
    background_model = create_background_model(video_path)
    
    # Initialize HSV masker
    hsv_masker = HSVMasker(
        lower_hsv=(0, 0, 100),  # Low hue, low sat, high value for bright objects
        upper_hsv=(180, 255, 255),  # Any hue, any sat, high value
        min_object_size=10,  # Reduced to catch smaller fish
        apply_morphology=True
    )
    
    # Load video
    print(f"Loading video: {video_path}")
    video = pims.PyAVReaderIndexed(str(video_path))
    
    print(f"Processing first {num_frames} frames (10 seconds at {fps} FPS)...")
    
    # Prepare video writer for HSV analysis output
    height, width = video[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output_path = output_dir / "hsv_analysis_10_seconds.mp4"
    out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height), isColor=True)
    
    # Process frames
    fish_counts = []
    
    try:
        for i in range(min(num_frames, len(video))):
            frame = video[i]
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray_frame = frame[:, :, 0]
            else:
                gray_frame = frame
            
            # Apply background subtraction
            bg_subtracted = apply_background_subtraction(gray_frame, background_model)
            
            # Apply HSV masking to background subtracted frame
            hsv_masked = hsv_masker.process_background_subtracted_frame(bg_subtracted)
            
            # Detect fish contours and centroids
            contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted)
            fish_counts.append(len(centroids))
            
            # Create visualization for individual frames (every 2nd frame to save space)
            if i % 2 == 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Original frame
                axes[0, 0].imshow(gray_frame, cmap='gray')
                axes[0, 0].set_title(f'Original Frame {i} (t={i/fps:.1f}s)')
                axes[0, 0].axis('off')
                
                # Background subtracted
                axes[0, 1].imshow(bg_subtracted, cmap='gray')
                axes[0, 1].set_title(f'Background Subtracted Frame {i}')
                axes[0, 1].axis('off')
                
                # HSV masked with contours
                axes[1, 0].imshow(hsv_masked, cmap='viridis')
                # Draw contours
                for contour in contours:
                    contour_points = contour.reshape(-1, 2)
                    axes[1, 0].plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
                axes[1, 0].set_title(f'HSV Masked with Outlines Frame {i}')
                axes[1, 0].axis('off')
                
                # Overlay contours and centroids on original
                axes[1, 1].imshow(gray_frame, cmap='gray')
                
                # Draw contours
                for contour in contours:
                    contour_points = contour.reshape(-1, 2)
                    axes[1, 1].plot(contour_points[:, 0], contour_points[:, 1], 'yellow', linewidth=2)
                
                # Draw centroids
                if centroids:
                    y_coords, x_coords = zip(*centroids)
                    axes[1, 1].scatter(x_coords, y_coords, c='red', s=100, marker='o', 
                                     edgecolors='white', linewidths=2, alpha=0.8)
                    axes[1, 1].scatter(x_coords, y_coords, c='red', s=20, marker='+', 
                                     linewidths=3, alpha=1.0)
                
                axes[1, 1].set_title(f'Detected Fish (Count: {len(centroids)})')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                
                # Save visualization
                frame_output_path = frames_dir / f'frame_{i:03d}_hsv_analysis.png'
                plt.savefig(frame_output_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # Create video frame with HSV analysis overlay
            video_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            
            # Draw contours on video frame
            for contour in contours:
                cv2.drawContours(video_frame, [contour], -1, (0, 255, 255), 2)  # Yellow contours
            
            # Draw centroids on video frame
            for centroid in centroids:
                y, x = int(centroid[0]), int(centroid[1])
                cv2.circle(video_frame, (x, y), 5, (0, 0, 255), -1)  # Red centroids
                cv2.circle(video_frame, (x, y), 8, (255, 255, 255), 2)  # White outline
            
            # Add frame info text
            cv2.putText(video_frame, f'Frame: {i} | Time: {i/fps:.1f}s | Fish: {len(centroids)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Write frame to video
            out.write(video_frame)
            
            if i % 20 == 0:
                print(f"Processed frame {i+1}/{num_frames} - Found {len(centroids)} fish")
    
    finally:
        out.release()
    
    # Create summary
    summary_path = output_dir / 'hsv_analysis_10_seconds_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("HSV Analysis - First 10 Seconds Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames processed: {num_frames}\n")
        f.write(f"Frame rate: {fps} FPS\n")
        f.write(f"Duration: 10.0 seconds\n")
        f.write(f"HSV range: {hsv_masker.get_hsv_range()}\n")
        f.write(f"Min object size: {hsv_masker.min_object_size}\n")
        f.write(f"Morphology applied: {hsv_masker.apply_morphology}\n\n")
        
        f.write("Fish Detection Statistics:\n")
        f.write(f"Total fish detected: {sum(fish_counts)}\n")
        f.write(f"Average fish per frame: {np.mean(fish_counts):.2f}\n")
        f.write(f"Max fish in single frame: {max(fish_counts)}\n")
        f.write(f"Min fish in single frame: {min(fish_counts)}\n\n")
        
        f.write("Output Files:\n")
        f.write(f"- Video: hsv_analysis_10_seconds.mp4\n")
        f.write(f"- Frame analysis images: frames/ directory\n")
        f.write(f"- Summary: hsv_analysis_10_seconds_summary.txt\n\n")
        
        f.write("Analysis shows:\n")
        f.write("1. Original frame\n")
        f.write("2. Background subtracted frame\n")
        f.write("3. HSV masked result with contours\n")
        f.write("4. Detected fish centroids overlaid on original\n")
    
    print(f"HSV analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Video output: {video_output_path}")
    print(f"Frame analysis images: {frames_dir}")
    print(f"Summary: {summary_path}")
    print(f"Total fish detected: {sum(fish_counts)}")
    print(f"Average fish per frame: {np.mean(fish_counts):.2f}")


def main():
    """Main function to run the HSV analysis for 10 seconds."""
    # Set paths
    video_path = "/Users/jasonli/FishDetection/data/input/videos/Clutch1_20250804_122715.mp4"
    output_dir = "/Users/jasonli/FishDetection/data/output/hsv_analysis"
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("Starting HSV analysis for first 10 seconds of video...")
    print(f"Video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Run the analysis
    process_10_seconds_hsv_analysis(
        video_path=video_path,
        output_dir=output_dir,
        fps=20  # Adjust based on your video's actual frame rate
    )


if __name__ == "__main__":
    main()
