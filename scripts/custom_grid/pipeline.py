"""
Custom Grid Pipeline

This pipeline:
1. Divides frame into custom 7x7 grid with specific horizontal lines
2. For each cell, finds 1 fish by selecting the largest blob/contour
3. Uses V2 background subtraction + HSV masking
"""

import cv2
import numpy as np
import pims
from pathlib import Path
from typing import List, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2
from .base_pipeline import BasePipeline


def get_custom_grid_cells(width: int, height: int):
    """
    Get custom grid cell boundaries.
    
    Custom grid: 7 columns, 3 horizontal lines (top moved up, center, bottom moved down)
    
    Returns:
        List of (x_start, y_start, x_end, y_end) for each cell
    """
    cols = 7
    cell_width = width / cols
    
    # Calculate horizontal line positions
    # Based on 7 rows, but with custom adjustments
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
    """
    Find the largest blob/contour in a cell region.
    
    Args:
        hsv_masker: HSVMasker instance
        bg_subtracted_frame: Full background subtracted frame
        x_start, y_start, x_end, y_end: Cell boundaries
    
    Returns:
        (contour, centroid) or (None, None) if no blob found
        Contour and centroid are in full-frame coordinates
    """
    # Extract cell region
    cell_region = bg_subtracted_frame[y_start:y_end, x_start:x_end]
    
    if cell_region.size == 0:
        return None, None
    
    # Use HSV masker to find contours in this cell
    # Set a low min_object_size to catch all blobs, then pick largest
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
    # Contour points need to be shifted
    full_frame_contour = largest_contour.copy()
    full_frame_contour[:, :, 0] += x_start  # x coordinate
    full_frame_contour[:, :, 1] += y_start  # y coordinate
    
    # Centroid needs to be shifted
    full_frame_centroid = (largest_centroid[0] + y_start, largest_centroid[1] + x_start)
    
    return full_frame_contour, full_frame_centroid


class CustomGridPipeline(BasePipeline):
    """
    Custom Grid Pipeline.
    
    This pipeline:
    1. Divides frame into custom 7x7 grid with specific horizontal lines
    2. For each cell, finds 1 fish by selecting the largest blob
    3. Uses V2 background subtraction + HSV masking
    """
    
    def __init__(self, *args, num_segments: int = 7, **kwargs):
        """
        Initialize custom grid pipeline.
        
        Args:
            num_segments: Number of temporal segments to break video into (default 7)
            *args, **kwargs: Passed to BasePipeline
        """
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
        self.target_per_cell = 1  # 1 fish per cell
    
    def create_background_model_strategy(self,
                                       video_path: Path,
                                       start_frame: int,
                                       num_frames: int,
                                       output_dir: Path) -> TrackingProgramBackgroundSubtractorV2:
        """
        Create background model using default strategy.
        """
        bg_subtractor = TrackingProgramBackgroundSubtractorV2(
            threshold=self.threshold,
            morph_kernel_size=self.morph_kernel_size
        )
        
        print("Creating background model using default frame indices...")
        bg_subtractor.create_background_model(video_path)
        bg_subtractor.save_background_model(output_dir / "background_model.npy")
        
        return bg_subtractor
    
    def process(self,
                video_path: Path,
                output_dir: Path,
                duration_seconds: int = 10,
                start_frame: int = 0) -> None:
        """
        Process video with custom grid detection.
        
        Divides frame into custom grid cells, finds 1 fish per cell (largest blob).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        num_frames = int(self.fps * duration_seconds)
        
        # Initialize background subtractor
        print("Initializing V2 background subtractor (threshold=15 + morphology)...")
        self.bg_subtractor = self.create_background_model_strategy(
            video_path, start_frame, num_frames, output_dir
        )
        
        # Initialize HSV masker
        print("Initializing HSV masker...")
        self.hsv_masker = HSVMasker(
            lower_hsv=self.hsv_lower,
            upper_hsv=self.hsv_upper,
            min_object_size=self.min_object_size,
            apply_morphology=True
        )
        
        # Load video
        print(f"Loading video: {video_path}")
        self.video = pims.PyAVReaderIndexed(str(video_path))
        
        # Check if we have enough frames
        end_frame = min(start_frame + num_frames, len(self.video))
        actual_num_frames = end_frame - start_frame
        
        height, width = self.video[0].shape[:2]
        
        # Get custom grid cells
        cells, row_boundaries = get_custom_grid_cells(width, height)
        num_cells = len(cells)
        
        print(f"\nProcessing frames {start_frame} to {end_frame-1} ({actual_num_frames} frames)")
        print(f"Custom grid: {num_cells} cells (7 columns, 4 rows with custom horizontal lines)")
        print(f"Target: 1 fish per cell (largest blob)")
        print(f"Temporal segments: {self.num_segments}")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output_path = output_dir / f"custom_grid_{num_cells}cells_{self.num_segments}seg.mp4"
        out = cv2.VideoWriter(str(video_output_path), fourcc, self.fps, (width, height), isColor=True)
        
        # Calculate segment boundaries
        frames_per_segment = actual_num_frames // self.num_segments
        segment_boundaries = [start_frame + i * frames_per_segment for i in range(self.num_segments + 1)]
        segment_boundaries[-1] = end_frame
        
        # Statistics
        all_cell_counts = []  # List of lists: [cell_0_count, cell_1_count, ...] per frame
        total_fish_counts = []
        frames_with_missing_fish = []  # Track which frames have missing fish
        
        # Process each segment
        for seg_idx in range(self.num_segments):
            seg_start = segment_boundaries[seg_idx]
            seg_end = segment_boundaries[seg_idx + 1]
            seg_num_frames = seg_end - seg_start
            
            print(f"\n--- Processing Segment {seg_idx + 1}/{self.num_segments} "
                  f"(frames {seg_start} to {seg_end - 1}, {seg_num_frames} frames) ---")
            
            # Process frames in this segment
            for local_seg_idx in range(seg_num_frames):
                global_idx = seg_start + local_seg_idx
                frame = self.video[global_idx]
                
                # Convert to grayscale
                if len(frame.shape) == 3:
                    gray_frame = frame[:, :, 0]
                else:
                    gray_frame = frame
                
                # Apply V2 background subtraction
                bg_subtracted = self.bg_subtractor.apply_background_subtraction(gray_frame)
                
                # Process each cell
                all_contours = []
                all_centroids = []
                cell_counts = [0] * num_cells
                
                for cell_idx, (x_start, y_start, x_end, y_end) in enumerate(cells):
                    contour, centroid = find_largest_blob_in_cell(
                        self.hsv_masker,
                        bg_subtracted,
                        x_start, y_start, x_end, y_end
                    )
                    
                    if contour is not None and centroid is not None:
                        all_contours.append(contour)
                        all_centroids.append(centroid)
                        cell_counts[cell_idx] = 1
                
                all_cell_counts.append(cell_counts)
                total_fish_counts.append(len(all_centroids))
                
                # Track frames with missing fish
                if len(all_centroids) < num_cells:
                    frames_with_missing_fish.append(global_idx)
                
                # Create output frame
                output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                
                # Draw grid lines
                # Vertical lines (7 columns)
                for i in range(1, 7):
                    x = int(i * width / 7)
                    cv2.line(output_frame, (x, 0), (x, height), (255, 255, 0), 2)
                
                # Horizontal lines (custom positions)
                for y in row_boundaries[1:-1]:  # Skip first (0) and last (height)
                    cv2.line(output_frame, (0, y), (width, y), (255, 255, 0), 2)
                
                # Draw outer border
                cv2.rectangle(output_frame, (0, 0), (width-1, height-1), (255, 255, 0), 2)
                
                # Draw segment boundary indicator
                if local_seg_idx == 0:
                    cv2.line(output_frame, (0, 0), (width, 0), (0, 255, 255), 5)
                
                # Draw contours
                if len(all_contours) > 0:
                    cv2.drawContours(output_frame, all_contours, -1, (0, 255, 0), 2)
                
                # Draw centroids
                for y, x in all_centroids:
                    cv2.circle(output_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                # Add stats text
                total_fish = len(all_centroids)
                stats_text = [
                    f"Segment: {seg_idx + 1}/{self.num_segments} | Frame: {local_seg_idx:3d}/{seg_num_frames-1} (Global: {global_idx})",
                    f"Total Fish: {total_fish}/{num_cells} cells",
                    f"Target: 1 fish per cell (largest blob)"
                ]
                for j, text in enumerate(stats_text):
                    cv2.putText(output_frame, text, (10, 30 + j * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                out.write(output_frame)
                
                # Save individual frame (every 5th frame)
                if local_seg_idx % 5 == 0:
                    frame_path = frames_dir / f"frame_{global_idx:04d}_seg{seg_idx+1}_cells.png"
                    cv2.imwrite(str(frame_path), output_frame)
                
                if local_seg_idx % 20 == 0:
                    avg_total = np.mean(total_fish_counts[-20:]) if len(total_fish_counts) >= 20 else np.mean(total_fish_counts)
                    print(f"  Segment {seg_idx + 1}, Frame {local_seg_idx:3d}/{seg_num_frames-1} (Global: {global_idx}): "
                          f"avg_total={avg_total:.1f} fish")
        
        out.release()
        
        # Generate summary
        self._generate_summary(
            output_dir, all_cell_counts, total_fish_counts, duration_seconds, start_frame, num_cells, frames_with_missing_fish
        )
        
        print(f"\n✓ Results saved to: {output_dir}")
    
    def _generate_summary(self,
                         output_dir: Path,
                         all_cell_counts: List[List[int]],
                         total_fish_counts: List[int],
                         duration_seconds: int,
                         start_frame: int,
                         num_cells: int,
                         frames_with_missing_fish: List[int]) -> None:
        """Generate summary statistics and save to file."""
        # Calculate statistics
        avg_total = np.mean(total_fish_counts)
        std_total = np.std(total_fish_counts)
        avg_per_cell = [np.mean([counts[j] for counts in all_cell_counts]) for j in range(num_cells)]
        
        # Calculate cell accuracy
        frames_with_all_cells = sum(1 for counts in all_cell_counts if sum(counts) == num_cells)
        frames_with_missing_cells = len(all_cell_counts) - frames_with_all_cells
        
        summary = f"""
Custom Grid Pipeline - Results
==================================
Variant: CustomGridPipeline

Configuration:
- Background Subtraction: V2 (Threshold={self.threshold}, Morphology={self.morph_kernel_size} kernel)
- HSV Masking: Lower HSV={self.hsv_lower}, Upper HSV={self.hsv_upper}
- Custom Grid: {num_cells} cells (7 columns, 4 rows with custom horizontal lines)
- Detection Method: Largest blob per cell (1 fish per cell)
- Temporal Segments: {self.num_segments} segments
- Frames Processed: {len(total_fish_counts)} ({duration_seconds} seconds @ {self.fps} FPS)
- Start Frame: {start_frame}, End Frame: {start_frame + len(total_fish_counts) - 1}

Fish Detection Statistics:
- Average Total Fish per Frame: {avg_total:.2f}
- Target Total: {num_cells}
- Standard Deviation: {std_total:.2f}
- Min Total: {np.min(total_fish_counts)}
- Max Total: {np.max(total_fish_counts)}

Cell Accuracy Analysis:
- Frames with ALL cells having exactly 1 fish: {frames_with_all_cells} ({frames_with_all_cells/len(total_fish_counts)*100:.2f}%)
- Frames with at least ONE cell missing fish: {frames_with_missing_cells} ({frames_with_missing_cells/len(total_fish_counts)*100:.2f}%)
{f'- Frames with missing fish: {frames_with_missing_fish}' if frames_with_missing_fish else '- All frames had all cells with fish'}

Method Details:
1. Custom Grid Layout:
   - 7 vertical columns (evenly spaced)
   - 4 horizontal rows with custom line positions:
     * Top line: moved up from standard position
     * Center line: through middle of frame
     * Bottom line: moved down from standard position
   - Total: {num_cells} cells
   
2. Detection Method:
   - For each cell, finds the largest blob/contour
   - Uses HSV masking to filter for bright objects
   - Selects single largest detection per cell
   - Target: 1 fish per cell
   
3. Temporal Segmentation:
   - Video divided into {self.num_segments} equal temporal segments
   - Each segment processed independently

Output Files:
- Video: custom_grid_{num_cells}cells_{self.num_segments}seg.mp4
- Background model: background_model.npy
- Summary: custom_grid_{num_cells}cells_{self.num_segments}seg_summary.txt
    """
        
        with open(output_dir / f"custom_grid_{num_cells}cells_{self.num_segments}seg_summary.txt", 'w') as f:
            f.write(summary)
        
        print("\n" + summary)

