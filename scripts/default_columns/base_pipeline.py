"""
Base pipeline class for column-based fish detection.

This class provides the common processing logic that all pipeline variants share.
Each variant only needs to implement the background model creation strategy.
"""

import cv2
import numpy as np
import pims
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2
from .utils import count_fish_per_column, binary_search_min_size_columns


class BasePipeline(ABC):
    """
    Base class for column-based fish detection pipelines.
    
    Subclasses should implement:
    - create_background_model_strategy() - How to create the background model
    """
    
    def __init__(self, 
                 fps: int = 20,
                 num_columns: int = 7,
                 target_per_column: int = 4,
                 threshold: int = 15,
                 morph_kernel_size: Tuple[int, int] = (5, 5),
                 hsv_lower: Tuple[int, int, int] = (0, 0, 100),
                 hsv_upper: Tuple[int, int, int] = (180, 255, 255),
                 min_object_size: int = 10):
        """Initialize the base pipeline."""
        self.fps = fps
        self.num_columns = num_columns
        self.target_per_column = target_per_column
        self.threshold = threshold
        self.morph_kernel_size = morph_kernel_size
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.min_object_size = min_object_size
        
        # Will be initialized in process()
        self.bg_subtractor = None
        self.hsv_masker = None
        self.video = None
    
    @abstractmethod
    def create_background_model_strategy(self, 
                                       video_path: Path,
                                       start_frame: int,
                                       num_frames: int,
                                       output_dir: Path) -> TrackingProgramBackgroundSubtractorV2:
        """
        Create and configure the background subtractor with appropriate background model.
        
        Args:
            video_path: Path to input video
            start_frame: Starting frame number
            num_frames: Number of frames to process
            output_dir: Output directory
            
        Returns:
            Configured TrackingProgramBackgroundSubtractorV2 instance with background model
        """
        pass
    
    def process(self,
                video_path: Path,
                output_dir: Path,
                duration_seconds: int = 10,
                start_frame: int = 0) -> None:
        """
        Process video with column-based detection.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save output
            duration_seconds: Duration in seconds to process
            start_frame: Starting frame number (default 0)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        num_frames = int(self.fps * duration_seconds)
        
        # Initialize background subtractor using strategy
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
              f"with {self.num_columns} columns, targeting {self.target_per_column} fish per column...")
        
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
            
            # Binary search for optimal min_size based on column distribution
            best_min_size, column_counts, iterations = binary_search_min_size_columns(
                self.hsv_masker,
                bg_subtracted,
                width,
                num_columns=self.num_columns,
                target_per_column=self.target_per_column,
                min_val=4,
                max_val=25,
                max_iterations=15
            )
            
            # Set optimal min_size and get final detection
            self.hsv_masker.min_object_size = best_min_size
            contours, centroids = self.hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted)
            
            # Recalculate column counts with final detection
            final_column_counts = count_fish_per_column(centroids, width, self.num_columns)
            
            all_column_counts.append(final_column_counts)
            min_sizes_used.append(best_min_size)
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
                f"MinSize: {best_min_size:2d} | Iterations: {iterations}",
                f"Target: {self.target_per_column} per column"
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
    
    def _generate_summary(self,
                         output_dir: Path,
                         all_column_counts: List[List[int]],
                         min_sizes_used: List[int],
                         iterations_per_frame: List[int],
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
        avg_min_size = np.mean(min_sizes_used)
        avg_iterations = np.mean(iterations_per_frame)
        
        # Get variant name for summary
        variant_name = self.__class__.__name__
        
        summary = f"""
Complete Pipeline V2 - Column-Based Detection Results
=====================================================
Variant: {variant_name}

Configuration:
- Background Subtraction: V2 (Threshold={self.threshold}, Morphology={self.morph_kernel_size} kernel)
- HSV Masking: Lower HSV={self.hsv_lower}, Upper HSV={self.hsv_upper}
- Column-Based Detection: {self.num_columns} columns, {self.target_per_column} fish per column
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

Adaptive Parameters:
- Average Min Size Used: {avg_min_size:.2f}
- Average Binary Search Iterations: {avg_iterations:.2f}

Method Details:
1. V2 Background Subtraction:
   - Lower threshold ({self.threshold} vs 25) captures more faint fish parts
   - Morphological closing ({self.morph_kernel_size}) prevents blob splitting
   
2. HSV Masking:
   - Filters for bright objects (value > {self.hsv_lower[2]})
   - Reduces false detections significantly
   
3. Column-Based Adaptive Detection:
   - Divides frame into {self.num_columns} equal columns
   - Binary search adjusts min_size to achieve {self.target_per_column} fish per column
   - Ensures even distribution across the frame
   - Column boundaries drawn as yellow vertical lines

Output Files:
- Video: complete_pipeline_v2_columns_{self.num_columns}x{self.target_per_column}.mp4
- Background model: background_model.npy
- Summary: complete_pipeline_v2_columns_{self.num_columns}x{self.target_per_column}_summary.txt
    """
        
        with open(output_dir / f"complete_pipeline_v2_columns_{self.num_columns}x{self.target_per_column}_summary.txt", 'w') as f:
            f.write(summary)
        
        print("\n" + summary)

