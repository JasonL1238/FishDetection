#!/usr/bin/env python3
"""
Check which cell missed the fish in frame 120.
"""

import cv2
import numpy as np
import pims
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2


def get_custom_grid_cells(width: int, height: int):
    """Get custom grid cell boundaries."""
    cols = 7
    cell_width = width / cols
    
    # Calculate horizontal line positions
    rows = 7
    cell_height_base = height / rows
    
    # Original positions (for reference)
    line_3_y_original = int(3 * cell_height_base)  # ~219
    line_4_y_original = int(4 * cell_height_base)  # ~292
    
    # Adjusted positions
    adjustment = int(cell_height_base * 0.15)
    top_line_y = line_3_y_original - adjustment      # ~209
    center_line_y = height // 2                       # 256
    bottom_line_y = line_4_y_original + adjustment    # ~302
    
    # Define row boundaries
    row_boundaries = [0, top_line_y, center_line_y, bottom_line_y, height]
    
    cells = []
    for row_idx in range(len(row_boundaries) - 1):
        y_start = row_boundaries[row_idx]
        y_end = row_boundaries[row_idx + 1]
        
        for col_idx in range(cols):
            x_start = int(col_idx * cell_width)
            x_end = int((col_idx + 1) * cell_width)
            
            cells.append((x_start, y_start, x_end, y_end))
    
    return cells, row_boundaries


def find_largest_blob_in_cell(hsv_masker, bg_subtracted_frame, x_start, y_start, x_end, y_end):
    """Find the largest blob/contour in a cell region."""
    # Extract cell region
    cell_region = bg_subtracted_frame[y_start:y_end, x_start:x_end]
    
    if cell_region.size == 0:
        return None, None
    
    # Use HSV masker to find contours in this cell
    original_min_size = hsv_masker.min_object_size
    hsv_masker.min_object_size = 1  # Very low to catch all blobs
    
    contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(cell_region)
    
    hsv_masker.min_object_size = original_min_size
    
    if len(contours) == 0:
        return None, None
    
    # Find largest contour by area
    largest_contour = None
    largest_area = 0
    largest_centroid = None
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
            largest_centroid = centroids[i]
    
    if largest_contour is None:
        return None, None
    
    # Convert contour and centroid back to full-frame coordinates
    full_frame_contour = largest_contour.copy()
    full_frame_contour[:, :, 0] += x_start  # x coordinate
    full_frame_contour[:, :, 1] += y_start  # y coordinate
    
    full_frame_centroid = (largest_centroid[0] + y_start, largest_centroid[1] + x_start)
    
    return full_frame_contour, full_frame_centroid


def main():
    """Check frame 120 to see which cell has no fish."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/frame120_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading video...")
    video = pims.PyAVReaderIndexed(str(video_path))
    
    # Get frame 120
    frame = video[120]
    
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray_frame = frame[:, :, 0]
    else:
        gray_frame = frame
    
    height, width = gray_frame.shape[:2]
    
    # Initialize background subtractor and load model
    bg_subtractor = TrackingProgramBackgroundSubtractorV2(threshold=15, morph_kernel_size=(5, 5))
    bg_model_path = Path("data/output/custom_grid_10sec/background_model.npy")
    if bg_model_path.exists():
        bg_subtractor.load_background_model(bg_model_path)
    else:
        bg_subtractor.create_background_model(video_path)
    
    # Apply background subtraction
    bg_subtracted = bg_subtractor.apply_background_subtraction(gray_frame)
    
    # Initialize HSV masker
    hsv_masker = HSVMasker(
        lower_hsv=(0, 0, 100),
        upper_hsv=(180, 255, 255),
        min_object_size=10,
        apply_morphology=True
    )
    
    # Get custom grid cells
    cells, row_boundaries = get_custom_grid_cells(width, height)
    num_cells = len(cells)
    
    print(f"\nAnalyzing frame 120...")
    print(f"Grid: {num_cells} cells (7 columns × 4 rows)")
    
    # Check each cell
    cell_results = []
    for cell_idx, (x_start, y_start, x_end, y_end) in enumerate(cells):
        contour, centroid = find_largest_blob_in_cell(
            hsv_masker,
            bg_subtracted,
            x_start, y_start, x_end, y_end
        )
        
        has_fish = contour is not None and centroid is not None
        cell_results.append((cell_idx, has_fish, x_start, y_start, x_end, y_end))
        
        if not has_fish:
            row = cell_idx // 7
            col = cell_idx % 7
            print(f"  Cell {cell_idx} (Row {row+1}, Col {col+1}): NO FISH FOUND")
            print(f"    Region: x:{x_start}-{x_end}, y:{y_start}-{y_end}")
    
    # Create visualization
    output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    # Draw grid
    for i in range(1, 7):
        x = int(i * width / 7)
        cv2.line(output_frame, (x, 0), (x, height), (255, 255, 0), 2)
    
    for y in row_boundaries[1:-1]:
        cv2.line(output_frame, (0, y), (width, y), (255, 255, 0), 2)
    
    cv2.rectangle(output_frame, (0, 0), (width-1, height-1), (255, 255, 0), 2)
    
    # Highlight cells with no fish in red
    for cell_idx, has_fish, x_start, y_start, x_end, y_end in cell_results:
        if not has_fish:
            cv2.rectangle(output_frame, (x_start, y_start), (x_end-1, y_end-1), (0, 0, 255), 3)
            # Add cell number
            row = cell_idx // 7
            col = cell_idx % 7
            cv2.putText(output_frame, f"Cell {cell_idx}", (x_start + 5, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(output_frame, f"R{row+1}C{col+1}", (x_start + 5, y_start + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Draw detected fish
    all_contours = []
    all_centroids = []
    for cell_idx, (x_start, y_start, x_end, y_end) in enumerate(cells):
        contour, centroid = find_largest_blob_in_cell(
            hsv_masker,
            bg_subtracted,
            x_start, y_start, x_end, y_end
        )
        if contour is not None and centroid is not None:
            all_contours.append(contour)
            all_centroids.append(centroid)
    
    if len(all_contours) > 0:
        cv2.drawContours(output_frame, all_contours, -1, (0, 255, 0), 2)
    
    for y, x in all_centroids:
        cv2.circle(output_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    # Add text
    missing_cells = [idx for idx, has_fish, _, _, _, _ in cell_results if not has_fish]
    cv2.putText(output_frame, f"Frame 120 - Missing fish in cells: {missing_cells}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(output_frame, f"Total detected: {len(all_contours)}/28", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save
    output_path = output_dir / "frame_120_analysis.png"
    cv2.imwrite(str(output_path), output_frame)
    
    # Also save background subtracted version
    bg_output = cv2.cvtColor(bg_subtracted, cv2.COLOR_GRAY2BGR)
    bg_path = output_dir / "frame_120_bg_subtracted.png"
    cv2.imwrite(str(bg_path), bg_output)
    
    print(f"\n✓ Analysis saved to: {output_dir}")
    print(f"  - frame_120_analysis.png - Shows which cell(s) are missing fish (red boxes)")
    print(f"  - frame_120_bg_subtracted.png - Background subtracted frame")
    print(f"\nMissing fish in {len(missing_cells)} cell(s): {missing_cells}")


if __name__ == "__main__":
    main()
