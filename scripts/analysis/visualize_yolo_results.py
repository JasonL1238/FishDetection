#!/usr/bin/env python3
"""
Visualize YOLOv8 tracking results.

This script extracts and displays some frames from the YOLOv8 tracking results.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def extract_frames_from_video(video_path: str, output_dir: str, num_frames: int = 5):
    """
    Extract frames from the tracking result video.
    
    Args:
        video_path: Path to the tracking result video
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS")
    
    # Extract frames at regular intervals
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame
            frame_path = output_dir / f"frame_{frame_idx:03d}.png"
            cv2.imwrite(str(frame_path), frame)
            
            print(f"Extracted frame {frame_idx} -> {frame_path}")
        else:
            print(f"Failed to read frame {frame_idx}")
    
    cap.release()
    print(f"Extracted {len(frame_indices)} frames to {output_dir}")


def create_comparison_grid(video_path: str, output_path: str, num_frames: int = 6):
    """
    Create a comparison grid showing tracking results.
    
    Args:
        video_path: Path to the tracking result video
        output_path: Path to save the comparison image
        num_frames: Number of frames to show in the grid
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    if not frames:
        print("No frames extracted")
        return
    
    # Create subplot grid
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle('YOLOv8 Fish Tracking Results', fontsize=16)
    
    for i, frame in enumerate(frames[:num_frames]):
        row = i // cols
        col = i % cols
        
        if row < rows and col < cols:
            axes[row, col].imshow(frame)
            axes[row, col].set_title(f'Frame {frame_indices[i]}')
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(frames), rows * cols):
        row = i // cols
        col = i % cols
        if row < rows and col < cols:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison grid saved to: {output_path}")


def main():
    """Main function."""
    video_path = "/Users/jasonli/FishDetection/output/yolo_tracking_test/yolo_tracking_result.mp4"
    output_dir = "/Users/jasonli/FishDetection/output/yolo_tracking_test/frames"
    
    if not Path(video_path).exists():
        print(f"Video not found: {video_path}")
        return
    
    print("Extracting frames from YOLOv8 tracking results...")
    extract_frames_from_video(video_path, output_dir, num_frames=8)
    
    print("\nCreating comparison grid...")
    comparison_path = "/Users/jasonli/FishDetection/output/yolo_tracking_test/tracking_comparison.png"
    create_comparison_grid(video_path, comparison_path, num_frames=6)


if __name__ == "__main__":
    main()
