#!/usr/bin/env python3
"""
Show frame 1 divided into a grid of cells (boxes).
"""

import cv2
import numpy as np
import pims
from pathlib import Path


def main():
    """Show frame 1 divided into a grid of cells."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/cell_grid_examples")
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
    
    print(f"\nFrame dimensions: {width}x{height}")
    
    # Different grid options to show
    grid_options = [
        (2, 2),   # 2x2 = 4 cells
        (3, 3),   # 3x3 = 9 cells
        (4, 4),   # 4x4 = 16 cells
        (5, 5),   # 5x5 = 25 cells
        (7, 7),   # 7x7 = 49 cells
        (2, 7),   # 2 rows x 7 cols = 14 cells (like half-sectioned)
        (4, 7),   # 4 rows x 7 cols = 28 cells
        (7, 4),   # 7 rows x 4 cols = 28 cells
    ]
    
    print(f"\nCreating grid examples...")
    
    for rows, cols in grid_options:
        result = frame_bgr.copy()
        
        # Calculate cell dimensions
        cell_width = width / cols
        cell_height = height / rows
        
        # Draw vertical lines (columns)
        for i in range(1, cols):
            x = int(i * cell_width)
            cv2.line(result, (x, 0), (x, height), (0, 255, 0), 2)
        
        # Draw horizontal lines (rows)
        for i in range(1, rows):
            y = int(i * cell_height)
            cv2.line(result, (0, y), (width, y), (0, 255, 0), 2)
        
        # Draw outer border
        cv2.rectangle(result, (0, 0), (width-1, height-1), (0, 255, 0), 2)
        
        # Add label
        label = f"{rows}x{cols} = {rows*cols} cells"
        cv2.putText(result, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save
        output_path = output_dir / f"frame_0001_grid_{rows}x{cols}.png"
        cv2.imwrite(str(output_path), result)
        print(f"  Created: {rows}x{cols} grid ({rows*cols} cells) -> {output_path.name}")
    
    # Create a single image showing multiple grid options
    print(f"\nCreating comparison image...")
    
    comparison_images = []
    for rows, cols in [(2, 2), (3, 3), (4, 4), (7, 7)]:
        result = frame_bgr.copy()
        
        cell_width = width / cols
        cell_height = height / rows
        
        # Draw grid
        for i in range(1, cols):
            x = int(i * cell_width)
            cv2.line(result, (x, 0), (x, height), (0, 255, 0), 2)
        for i in range(1, rows):
            y = int(i * cell_height)
            cv2.line(result, (0, y), (width, y), (0, 255, 0), 2)
        cv2.rectangle(result, (0, 0), (width-1, height-1), (0, 255, 0), 2)
        
        label = f"{rows}x{cols}"
        cv2.putText(result, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Resize for comparison
        resized = cv2.resize(result, (400, 300))
        comparison_images.append(resized)
    
    # Create 2x2 grid
    top_row = np.hstack(comparison_images[:2])
    bottom_row = np.hstack(comparison_images[2:])
    comparison = np.vstack([top_row, bottom_row])
    
    comparison_path = output_dir / "frame_0001_grid_comparison.png"
    cv2.imwrite(str(comparison_path), comparison)
    print(f"  Created comparison: {comparison_path.name}")
    
    print(f"\n✓ All grid examples saved to: {output_dir}")
    print(f"\nEach image shows the frame divided into cells (boxes).")
    print(f"Pick which grid size you want to use!")


if __name__ == "__main__":
    main()

