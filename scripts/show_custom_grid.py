#!/usr/bin/env python3
"""
Show frame 1 with 7x7 grid but custom horizontal lines.
"""

import cv2
import numpy as np
import pims
from pathlib import Path


def main():
    """Show frame 1 with 7x7 grid, custom horizontal lines."""
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
    print(f"Creating 7x7 grid with custom horizontal lines...")
    
    result = frame_bgr.copy()
    
    rows = 7
    cols = 7
    
    # Calculate cell dimensions for reference
    cell_width = width / cols
    cell_height = height / rows
    
    # Draw ALL vertical lines (columns) - 6 lines for 7 columns
    for i in range(1, cols):
        x = int(i * cell_width)
        cv2.line(result, (x, 0), (x, height), (0, 255, 0), 2)
    
    # Calculate original middle 3 horizontal line positions
    all_horizontal_positions = []
    for i in range(1, rows):
        y = int(i * cell_height)
        all_horizontal_positions.append((i, y))
    
    # Original middle 3 were lines 3, 4, 5
    # Now we want:
    # - Remove line 5 (lowest)
    # - Keep lines 3 and 4, but adjust them
    # - Add center line
    
    line_3_y = all_horizontal_positions[2][1]  # Original line 3 position
    line_4_y = all_horizontal_positions[3][1]  # Original line 4 position
    center_y = height // 2  # Center of frame
    
    # Move line 3 up a little (reduce y by ~15% of cell height)
    adjustment = int(cell_height * 0.15)
    line_3_y_adjusted = line_3_y - adjustment
    
    # Move line 4 down a little (increase y by ~15% of cell height)
    line_4_y_adjusted = line_4_y + adjustment
    
    print(f"  Original line 3 position: y={line_3_y}")
    print(f"  Adjusted line 3 (moved up): y={line_3_y_adjusted}")
    print(f"  Original line 4 position: y={line_4_y}")
    print(f"  Adjusted line 4 (moved down): y={line_4_y_adjusted}")
    print(f"  Center line: y={center_y}")
    
    # Draw the adjusted horizontal lines
    cv2.line(result, (0, line_3_y_adjusted), (width, line_3_y_adjusted), (0, 255, 0), 2)
    cv2.line(result, (0, center_y), (width, center_y), (0, 255, 0), 2)
    cv2.line(result, (0, line_4_y_adjusted), (width, line_4_y_adjusted), (0, 255, 0), 2)
    
    # Draw outer border
    cv2.rectangle(result, (0, 0), (width-1, height-1), (0, 255, 0), 2)
    
    # Add label
    label = "7x7 grid - custom horizontal lines"
    cv2.putText(result, label, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save
    output_path = output_dir / "frame_0001_grid_7x7_custom_horizontal.png"
    cv2.imwrite(str(output_path), result)
    
    print(f"\n✓ Created: {output_path.name}")
    print(f"  Shows 7x7 grid with all vertical lines")
    print(f"  Custom horizontal lines:")
    print(f"    - Top line (moved up from original line 3)")
    print(f"    - Center line (through middle)")
    print(f"    - Bottom line (moved down from original line 4)")
    print(f"  (Removed lowest line, added center line)")


if __name__ == "__main__":
    main()
