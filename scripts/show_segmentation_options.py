#!/usr/bin/env python3
"""
Show individual horizontal lines on frame 1, each labeled with a number.
"""

import cv2
import numpy as np
import pims
from pathlib import Path


def main():
    """Show individual horizontal lines with numbers on frame 1."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/segmentation_options")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading video: {video_path}")
    video = pims.PyAVReaderIndexed(str(video_path))
    
    # Get frame 1
    frame = video[1]
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray_frame = frame[:, :, 0]
    else:
        gray_frame = frame
    
    # Convert to BGR for drawing
    frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    height, width = frame_bgr.shape[:2]
    
    print(f"\nCreating image with individual horizontal lines...")
    print(f"Frame dimensions: {width}x{height}")
    
    # Start with original frame
    result = frame_bgr.copy()
    
    # Draw individual horizontal lines at different positions
    # Spread them evenly across the frame
    num_lines = 12  # Reasonable number of lines
    line_positions = []
    
    # Calculate positions (avoiding top and bottom edges)
    margin = 20
    available_height = height - 2 * margin
    step = available_height / (num_lines + 1)
    
    for i in range(1, num_lines + 1):
        y = int(margin + i * step)
        line_positions.append((y, i))
    
    # Draw each line with a number
    for line_num, y in line_positions:
        # Draw the horizontal line
        cv2.line(result, (0, y), (width, y), (0, 255, 0), 2)
        
        # Add number label on the left side
        label = str(line_num)
        cv2.putText(result, label, (10, y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Also add number on the right side for visibility
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(result, label, (width - text_size[0] - 10, y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add title
    cv2.putText(result, "Horizontal Segmentation Lines - Frame 1", 
               (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save the image
    output_path = output_dir / "frame_0001_individual_horizontal_lines.png"
    cv2.imwrite(str(output_path), result)
    
    print(f"\n✓ Created image with individual horizontal lines: {output_path}")
    print(f"  Image size: {result.shape[1]}x{result.shape[0]}")
    print(f"  Number of lines: {num_lines}")
    print(f"  Each line is labeled with a number (1-{num_lines})")
    print(f"\nPick which line numbers you want to use for segmentation!")


if __name__ == "__main__":
    main()
