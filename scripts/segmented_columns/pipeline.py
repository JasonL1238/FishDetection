"""
Segmented Columns Pipeline

This pipeline:
1. Breaks the video into 7 temporal segments
2. For each segment, processes each column independently
3. For each column in each segment, exhaustively tries all min_size thresholds
   until it gets 4 fish per column or exhausts all possible thresholds
"""

import cv2
import numpy as np
import pims
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2
from .base_pipeline import BasePipeline
from .utils import count_fish_per_column


def exhaustive_search_min_size_for_column(hsv_masker, bg_subtracted_column, target_count=4,
                                          min_val=1, max_val=30):
    """
    Exhaustively try all min_object_size values to find one that gives target_count fish.
    Stops early if target_count is found, otherwise tries all values in range.
    
    Args:
        hsv_masker: HSVMasker instance
        bg_subtracted_column: Background subtracted frame for a single column (width x height)
        target_count: Target number of fish (default 4)
        min_val: Minimum min_size to try
        max_val: Maximum min_size to try
        
    Returns:
        (best_min_size, actual_count, iterations)
    """
    best_min_size = min_val
    best_count = 0
    best_diff = float('inf')
    iterations = 0
    
    # Try all values from min_val to max_val
    for test_size in range(min_val, max_val + 1):
        iterations += 1
        
        original_min_size = hsv_masker.min_object_size
        hsv_masker.min_object_size = test_size
        
        _, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_column)
        count = len(centroids)
        
        hsv_masker.min_object_size = original_min_size
        
        diff = abs(count - target_count)
        
        # Perfect match - stop early
        if count == target_count:
            return test_size, count, iterations
        
        # Track best result
        if diff < best_diff:
            best_min_size = test_size
            best_count = count
            best_diff = diff
    
    # Return best result found
    return best_min_size, best_count, iterations


def process_column_independently(hsv_masker, bg_subtracted_frame, column_idx, num_columns,
                                width, height, target_per_column=4):
    """
    Process a single column independently to get target_per_column fish.
    Exhaustively tries all thresholds until it gets target_per_column or exhausts all options.
    
    Args:
        hsv_masker: HSVMasker instance
        bg_subtracted_frame: Full background subtracted frame
        column_idx: Index of column to process (0-based)
        num_columns: Total number of columns
        width: Full frame width
        height: Full frame height
        target_per_column: Target fish count for this column
        
    Returns:
        (contours, centroids, min_size_used, iterations)
        Note: contours and centroids are in full-frame coordinates
    """
    # Extract column region
    column_width = width / num_columns
    start_x = int(column_idx * column_width)
    end_x = int((column_idx + 1) * column_width)
    bg_subtracted_column = bg_subtracted_frame[:, start_x:end_x]
    
    # Exhaustive search for optimal min_size for this column
    best_min_size, actual_count, iterations = exhaustive_search_min_size_for_column(
        hsv_masker, bg_subtracted_column, target_count=target_per_column,
        min_val=1, max_val=30
    )
    
    # Get final detection with optimal min_size
    original_min_size = hsv_masker.min_object_size
    hsv_masker.min_object_size = best_min_size
    contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_column)
    hsv_masker.min_object_size = original_min_size
    
    # Convert centroids back to full-frame coordinates
    full_frame_centroids = [(y, x + start_x) for y, x in centroids]
    
    # Convert contours back to full-frame coordinates
    full_frame_contours = []
    for contour in contours:
        # Shift contour points by start_x
        shifted_contour = contour.copy()
        shifted_contour[:, :, 0] += start_x
        full_frame_contours.append(shifted_contour)
    
    return full_frame_contours, full_frame_centroids, best_min_size, iterations


class SegmentedColumnsPipeline(BasePipeline):
    """
    Segmented Columns Pipeline.
    
    This pipeline:
    1. Breaks the video into 7 temporal segments
    2. For each segment, processes each column independently
    3. For each column, exhaustively tries all min_size thresholds (1-30)
       until it gets 4 fish or exhausts all possible thresholds
    """
    
    def __init__(self, *args, num_segments: int = 7, **kwargs):
        """
        Initialize segmented columns pipeline.
        
        Args:
            num_segments: Number of temporal segments to break video into (default 7)
            *args, **kwargs: Passed to BasePipeline
        """
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
    
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
        Process video with segmented column-based detection.
        
        Breaks video into num_segments temporal pieces, then processes
        each column independently within each segment.
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
        
        print(f"\nProcessing frames {start_frame} to {end_frame-1} ({actual_num_frames} frames) "
              f"with {self.num_columns} columns, {self.num_segments} temporal segments, "
              f"targeting {self.target_per_column} fish per column (exhaustive threshold search per column)...")
        
        # Video writer
        height, width = self.video[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output_path = output_dir / f"segmented_columns_{self.num_columns}x{self.target_per_column}_{self.num_segments}seg.mp4"
        out = cv2.VideoWriter(str(video_output_path), fourcc, self.fps, (width, height), isColor=True)
        
        # Calculate column boundaries
        column_width = width / self.num_columns
        column_boundaries = [int(i * column_width) for i in range(self.num_columns + 1)]
        
        # Calculate segment boundaries
        frames_per_segment = actual_num_frames // self.num_segments
        segment_boundaries = [start_frame + i * frames_per_segment for i in range(self.num_segments + 1)]
        segment_boundaries[-1] = end_frame  # Ensure last segment goes to end
        
        # Statistics
        all_column_counts = []
        min_sizes_used_per_column = []  # List of lists: [column_0_sizes, column_1_sizes, ...]
        iterations_per_column = []  # List of lists: [column_0_iters, column_1_iters, ...]
        total_fish_counts = []
        
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
                
                # Process each column independently
                all_contours = []
                all_centroids = []
                column_min_sizes = []
                column_iterations = []
                
                for col_idx in range(self.num_columns):
                    # Process this column independently
                    contours, centroids, min_size, iterations = process_column_independently(
                        self.hsv_masker,
                        bg_subtracted,
                        col_idx,
                        self.num_columns,
                        width,
                        height,
                        target_per_column=self.target_per_column
                    )
                    
                    all_contours.extend(contours)
                    all_centroids.extend(centroids)
                    column_min_sizes.append(min_size)
                    column_iterations.append(iterations)
                
                # Calculate column counts
                final_column_counts = count_fish_per_column(all_centroids, width, self.num_columns)
                
                all_column_counts.append(final_column_counts)
                min_sizes_used_per_column.append(column_min_sizes)
                iterations_per_column.append(column_iterations)
                total_fish_counts.append(len(all_centroids))
                
                # Create output frame
                output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                
                # Draw column boundaries (vertical lines)
                for boundary in column_boundaries:
                    cv2.line(output_frame, (boundary, 0), (boundary, height), (255, 255, 0), 2)
                
                # Draw segment boundary indicator (horizontal line at top)
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
                column_str = " ".join([f"C{i+1}:{c}" for i, c in enumerate(final_column_counts)])
                min_size_str = " ".join([f"C{i+1}:{s:2d}" for i, s in enumerate(column_min_sizes)])
                stats_text = [
                    f"Segment: {seg_idx + 1}/{self.num_segments} | Frame: {local_seg_idx:3d}/{seg_num_frames-1} (Global: {global_idx}) | Total Fish: {total_fish}",
                    f"Columns: {column_str}",
                    f"MinSizes: {min_size_str}",
                    f"Target: {self.target_per_column} per column (Independent per Column)"
                ]
                for j, text in enumerate(stats_text):
                    cv2.putText(output_frame, text, (10, 30 + j * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                out.write(output_frame)
                
                # Save individual frame (every 5th frame to save space)
                if local_seg_idx % 5 == 0:
                    frame_path = frames_dir / f"frame_{global_idx:04d}_seg{seg_idx+1}_columns.png"
                    cv2.imwrite(str(frame_path), output_frame)
                
                if local_seg_idx % 20 == 0:
                    avg_total = np.mean(total_fish_counts[-20:]) if len(total_fish_counts) >= 20 else np.mean(total_fish_counts)
                    avg_per_col = [np.mean([counts[j] for counts in all_column_counts[-20:]]) 
                                  for j in range(self.num_columns)] if len(all_column_counts) >= 20 else [np.mean([counts[j] for counts in all_column_counts]) for j in range(self.num_columns)]
                    print(f"  Segment {seg_idx + 1}, Frame {local_seg_idx:3d}/{seg_num_frames-1} (Global: {global_idx}): "
                          f"avg_total={avg_total:.1f} fish, avg_per_col={[f'{c:.1f}' for c in avg_per_col]}")
        
        out.release()
        
        # Generate summary
        self._generate_summary(
            output_dir, all_column_counts, min_sizes_used_per_column, iterations_per_column,
            total_fish_counts, duration_seconds, start_frame
        )
        
        print(f"\n✓ Results saved to: {output_dir}")
    
    def _generate_summary(self,
                         output_dir: Path,
                         all_column_counts: List[List[int]],
                         min_sizes_used_per_column: List[List[int]],
                         iterations_per_column: List[List[int]],
                         total_fish_counts: List[int],
                         duration_seconds: int,
                         start_frame: int) -> None:
        """Generate summary statistics and save to file."""
        # Analyze column accuracy
        frames_with_imperfect_columns = 0
        frames_with_all_perfect = 0
        columns_off_count = {i: 0 for i in range(self.num_columns + 1)}
        
        for column_counts in all_column_counts:
            columns_off = sum(1 for count in column_counts if count != self.target_per_column)
            columns_off_count[columns_off] += 1
            
            if columns_off > 0:
                frames_with_imperfect_columns += 1
            else:
                frames_with_all_perfect += 1
        
        # Calculate statistics
        avg_total = np.mean(total_fish_counts)
        std_total = np.std(total_fish_counts)
        avg_per_column = [np.mean([counts[j] for counts in all_column_counts]) for j in range(self.num_columns)]
        std_per_column = [np.std([counts[j] for counts in all_column_counts]) for j in range(self.num_columns)]
        
        # Calculate per-column min_size statistics
        avg_min_size_per_column = [np.mean([sizes[j] for sizes in min_sizes_used_per_column]) 
                                   for j in range(self.num_columns)]
        std_min_size_per_column = [np.std([sizes[j] for sizes in min_sizes_used_per_column]) 
                                   for j in range(self.num_columns)]
        
        # Calculate per-column iteration statistics
        avg_iterations_per_column = [np.mean([iters[j] for iters in iterations_per_column]) 
                                     for j in range(self.num_columns)]
        
        # Get variant name for summary
        variant_name = self.__class__.__name__
        
        summary = f"""
Segmented Columns Pipeline - Results
=====================================
Variant: {variant_name}

Configuration:
- Background Subtraction: V2 (Threshold={self.threshold}, Morphology={self.morph_kernel_size} kernel)
- HSV Masking: Lower HSV={self.hsv_lower}, Upper HSV={self.hsv_upper}
- Column-Based Detection: {self.num_columns} columns, {self.target_per_column} fish per column
- Temporal Segments: {self.num_segments} segments
- Independent Column Processing: Each column processed separately with its own adaptive threshold
- Frames Processed: {len(total_fish_counts)} ({duration_seconds} seconds @ {self.fps} FPS)
- Start Frame: {start_frame}, End Frame: {start_frame + len(total_fish_counts) - 1}

Fish Detection Statistics:
- Average Total Fish per Frame: {avg_total:.2f}
- Target Total: {self.num_columns * self.target_per_column}
- Standard Deviation: {std_total:.2f}
- Min Total: {np.min(total_fish_counts)}
- Max Total: {np.max(total_fish_counts)}

Per-Column Statistics:
"""
        for j in range(self.num_columns):
            summary += f"- Column {j+1}: avg={avg_per_column[j]:.2f}, std={std_per_column[j]:.2f}, target={self.target_per_column}\n"
            summary += f"  MinSize: avg={avg_min_size_per_column[j]:.2f}, std={std_min_size_per_column[j]:.2f}\n"
            summary += f"  Iterations: avg={avg_iterations_per_column[j]:.2f}\n"
        
        summary += f"""

Column Accuracy Analysis:
- Frames with ALL columns having exactly {self.target_per_column} fish: {frames_with_all_perfect} ({frames_with_all_perfect/len(total_fish_counts)*100:.2f}%)
- Frames with at least ONE column not having {self.target_per_column} fish: {frames_with_imperfect_columns} ({frames_with_imperfect_columns/len(total_fish_counts)*100:.2f}%)

Breakdown by number of columns off target:
"""
        for num_off in range(self.num_columns + 1):
            if columns_off_count[num_off] > 0:
                summary += f"- {num_off} column(s) off: {columns_off_count[num_off]} frames ({columns_off_count[num_off]/len(total_fish_counts)*100:.2f}%)\n"
        
        summary += f"""

Method Details:
1. Temporal Segmentation:
   - Video divided into {self.num_segments} equal temporal segments
   - Each segment processed independently
   
2. Independent Column Processing:
   - Each column processed separately within each segment
   - Each column uses its own exhaustive min_size threshold search
   - Tries all thresholds from 1 to 30 until it gets {self.target_per_column} fish or exhausts all options
   - Target: {self.target_per_column} fish per column
   
3. V2 Background Subtraction:
   - Lower threshold ({self.threshold} vs 25) captures more faint fish parts
   - Morphological closing ({self.morph_kernel_size}) prevents blob splitting
   
4. HSV Masking:
   - Filters for bright objects (value > {self.hsv_lower[2]})
   - Reduces false detections significantly

Output Files:
- Video: segmented_columns_{self.num_columns}x{self.target_per_column}_{self.num_segments}seg.mp4
- Background model: background_model.npy
- Summary: segmented_columns_{self.num_columns}x{self.target_per_column}_{self.num_segments}seg_summary.txt
    """
        
        with open(output_dir / f"segmented_columns_{self.num_columns}x{self.target_per_column}_{self.num_segments}seg_summary.txt", 'w') as f:
            f.write(summary)
        
        print("\n" + summary)

