"""
Adaptive Column Limit Pipeline

This variant adaptively limits to 4 fish per column by:
1. Detecting all potential fish with a permissive min_size
2. Grouping detections by column
3. Selecting the best 4 fish per column based on size/quality
4. Adapting selection criteria per column independently

This is more adaptive than global min_size adjustment.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2
from ..base.base_pipeline import BasePipeline
from ..base.utils import count_fish_per_column


def select_best_fish_per_column(contours: List, centroids: List[Tuple[float, float]], 
                                width: int, num_columns: int, target_per_column: int,
                                min_object_size: int) -> Tuple[List, List[Tuple[float, float]]]:
    """
    Select the best N fish per column based on contour area (size).
    
    Args:
        contours: List of contours
        centroids: List of (y, x) centroids
        width: Frame width
        num_columns: Number of columns
        target_per_column: Target fish per column
        min_object_size: Minimum object size for filtering
        
    Returns:
        Selected contours and centroids
    """
    column_width = width / num_columns
    
    # Group fish by column with their sizes
    column_fish = [[] for _ in range(num_columns)]
    
    for i, (centroid, contour) in enumerate(zip(centroids, contours)):
        y, x = centroid
        column_idx = min(int(x / column_width), num_columns - 1)
        
        # Calculate contour area as quality metric
        area = cv2.contourArea(contour)
        
        # Filter by minimum size
        if area >= min_object_size:
            column_fish[column_idx].append({
                'index': i,
                'centroid': centroid,
                'contour': contour,
                'area': area
            })
    
    # Select best N fish per column (sorted by area, largest first)
    selected_contours = []
    selected_centroids = []
    
    for col_idx in range(num_columns):
        fish_in_column = column_fish[col_idx]
        
        # Sort by area (largest first) - larger fish are typically more reliable
        fish_in_column.sort(key=lambda f: f['area'], reverse=True)
        
        # Take top N fish
        selected = fish_in_column[:target_per_column]
        
        for fish in selected:
            selected_contours.append(fish['contour'])
            selected_centroids.append(fish['centroid'])
    
    return selected_contours, selected_centroids


def adaptive_column_limit_detection(hsv_masker, bg_subtracted_frame, width, 
                                    num_columns=7, target_per_column=4,
                                    base_min_size=5, max_min_size=20):
    """
    Adaptively detect and limit to target_per_column fish per column.
    
    Strategy:
    1. Start with a permissive min_size to get many candidates
    2. Group by column
    3. Select best N per column based on size
    4. If a column has too few, gradually lower min_size for that column
    5. If a column has too many, select the largest N
    
    Returns:
        (selected_contours, selected_centroids, min_size_used, iterations)
    """
    # Start with a permissive min_size to get many candidates
    test_min_size = base_min_size
    iterations = 0
    max_iterations = 10
    
    while iterations < max_iterations:
        hsv_masker.min_object_size = test_min_size
        contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_frame)
        
        if len(contours) == 0:
            # No detections, try smaller min_size
            test_min_size = max(1, test_min_size - 2)
            iterations += 1
            continue
        
        # Group by column and check distribution
        column_counts = count_fish_per_column(centroids, width, num_columns)
        
        # Check if we have enough candidates in each column
        min_candidates = min(column_counts)
        max_candidates = max(column_counts)
        
        # If we have at least target_per_column candidates in most columns, proceed
        columns_with_enough = sum(1 for c in column_counts if c >= target_per_column)
        
        if columns_with_enough >= num_columns * 0.7:  # At least 70% of columns have enough
            # We have enough candidates, now select best per column
            selected_contours, selected_centroids = select_best_fish_per_column(
                contours, centroids, width, num_columns, target_per_column, test_min_size
            )
            
            # Check final distribution
            final_counts = count_fish_per_column(selected_centroids, width, num_columns)
            
            # If we're close to target, return
            total_off = sum(abs(c - target_per_column) for c in final_counts)
            if total_off <= num_columns * 0.5:  # Allow some tolerance
                return selected_contours, selected_centroids, test_min_size, iterations + 1
        
        # Adjust min_size based on total detections
        total_detections = len(contours)
        target_total = num_columns * target_per_column
        
        if total_detections < target_total * 0.8:
            # Too few detections, lower min_size
            test_min_size = max(1, test_min_size - 1)
        elif total_detections > target_total * 1.5:
            # Too many detections, raise min_size
            test_min_size = min(max_min_size, test_min_size + 1)
        else:
            # We're in a good range, proceed with selection
            selected_contours, selected_centroids = select_best_fish_per_column(
                contours, centroids, width, num_columns, target_per_column, test_min_size
            )
            return selected_contours, selected_centroids, test_min_size, iterations + 1
        
        iterations += 1
    
    # Fallback: use last attempt
    hsv_masker.min_object_size = test_min_size
    contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_frame)
    selected_contours, selected_centroids = select_best_fish_per_column(
        contours, centroids, width, num_columns, target_per_column, test_min_size
    )
    return selected_contours, selected_centroids, test_min_size, iterations


class AdaptiveColumnLimitPipeline(BasePipeline):
    """
    Adaptive column limit pipeline.
    
    Instead of adjusting global min_size, this variant:
    1. Detects fish with a permissive min_size
    2. Groups by column
    3. Selects the best 4 fish per column adaptively
    4. Adapts per column independently
    """
    
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
        Process video with adaptive column limiting.
        Overrides base process to use adaptive detection.
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
              f"with {self.num_columns} columns, adaptively limiting to {self.target_per_column} fish per column...")
        
        # Video writer
        height, width = self.video[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output_path = output_dir / f"complete_pipeline_v2_columns_{self.num_columns}x{self.target_per_column}.mp4"
        out = cv2.VideoWriter(str(video_output_path), fourcc, self.fps, (width, height), isColor=True)
        
        # Calculate column boundaries
        column_width = width / self.num_columns
        column_boundaries = [int(i * column_width) for i in range(self.num_columns + 1)]
        
        # Statistics
        all_column_counts = []
        min_sizes_used = []
        iterations_per_frame = []
        total_fish_counts = []
        
        # Process frames
        for local_idx in range(actual_num_frames):
            global_idx = start_frame + local_idx
            frame = self.video[global_idx]
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray_frame = frame[:, :, 0]
            else:
                gray_frame = frame
            
            # Apply V2 background subtraction
            bg_subtracted = self.bg_subtractor.apply_background_subtraction(gray_frame)
            
            # Adaptive column limit detection
            contours, centroids, min_size_used, iterations = adaptive_column_limit_detection(
                self.hsv_masker,
                bg_subtracted,
                width,
                num_columns=self.num_columns,
                target_per_column=self.target_per_column,
                base_min_size=5,
                max_min_size=20
            )
            
            # Calculate column counts
            final_column_counts = count_fish_per_column(centroids, width, self.num_columns)
            
            all_column_counts.append(final_column_counts)
            min_sizes_used.append(min_size_used)
            iterations_per_frame.append(iterations)
            total_fish_counts.append(len(centroids))
            
            # Create output frame
            output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            
            # Draw column boundaries (vertical lines)
            for boundary in column_boundaries:
                cv2.line(output_frame, (boundary, 0), (boundary, height), (255, 255, 0), 2)
            
            # Draw contours
            if len(contours) > 0:
                cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 2)
            
            # Draw centroids
            for y, x in centroids:
                cv2.circle(output_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # Add stats text
            total_fish = len(centroids)
            column_str = " ".join([f"C{i+1}:{c}" for i, c in enumerate(final_column_counts)])
            stats_text = [
                f"Frame: {local_idx:3d}/{actual_num_frames-1} (Global: {global_idx}) | Total Fish: {total_fish}",
                f"Columns: {column_str}",
                f"MinSize: {min_size_used:2d} | Iterations: {iterations}",
                f"Target: {self.target_per_column} per column (Adaptive Selection)"
            ]
            for j, text in enumerate(stats_text):
                cv2.putText(output_frame, text, (10, 30 + j * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(output_frame)
            
            # Save individual frame (every 5th frame to save space)
            if local_idx % 5 == 0:
                frame_path = frames_dir / f"frame_{global_idx:04d}_columns.png"
                cv2.imwrite(str(frame_path), output_frame)
            
            if local_idx % 20 == 0:
                avg_total = np.mean(total_fish_counts)
                avg_min_size = np.mean(min_sizes_used)
                avg_per_col = [np.mean([counts[j] for counts in all_column_counts]) for j in range(self.num_columns)]
                print(f"  Frame {local_idx:3d}/{actual_num_frames-1} (Global: {global_idx}): avg_total={avg_total:.1f} fish, "
                      f"avg_per_col={[f'{c:.1f}' for c in avg_per_col]}, "
                      f"avg_min_size={avg_min_size:.1f}")
        
        out.release()
        
        # Generate summary
        self._generate_summary(
            output_dir, all_column_counts, min_sizes_used, iterations_per_frame,
            total_fish_counts, duration_seconds, start_frame
        )
        
        print(f"\n✓ Results saved to: {output_dir}")



