"""
Half-Sectioned Pipeline

This pipeline:
1. Splits the frame into top and bottom halves
2. Each half is divided into 7 sections (like columns)
3. Each of the 14 sections is processed independently
4. For each section, exhaustively tries all min_size thresholds (1-30)
   until it gets exactly 2 fish or exhausts all possible thresholds
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


def exhaustive_search_min_size_for_section(hsv_masker, bg_subtracted_section, target_count=2,
                                          min_val=1, max_val=30):
    """
    Exhaustively try all min_object_size values to find one that gives target_count fish.
    Stops early if target_count is found, otherwise tries all values in range.
    
    Args:
        hsv_masker: HSVMasker instance
        bg_subtracted_section: Background subtracted frame for a single section (width x height)
        target_count: Target number of fish (default 2)
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
        
        _, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_section)
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


def process_section_independently(hsv_masker, bg_subtracted_frame, section_idx, num_sections_per_half,
                                  width, height, target_per_section=2):
    """
    Process a single section independently to get target_per_section fish.
    Exhaustively tries all thresholds until it gets target_per_section or exhausts all options.
    
    Args:
        hsv_masker: HSVMasker instance
        bg_subtracted_frame: Full background subtracted frame
        section_idx: Index of section to process (0-13, where 0-6 are top half, 7-13 are bottom half)
        num_sections_per_half: Number of sections per half (7)
        width: Full frame width
        height: Full frame height
        target_per_section: Target fish count for this section (default 2)
        
    Returns:
        (contours, centroids, min_size_used, iterations)
        Note: contours and centroids are in full-frame coordinates
    """
    # Determine which half (top or bottom)
    is_bottom_half = section_idx >= num_sections_per_half
    half_section_idx = section_idx % num_sections_per_half
    
    # Calculate section boundaries
    section_width = width / num_sections_per_half
    start_x = int(half_section_idx * section_width)
    end_x = int((half_section_idx + 1) * section_width)
    
    half_height = height // 2
    if is_bottom_half:
        start_y = half_height
        end_y = height
    else:
        start_y = 0
        end_y = half_height
    
    # Extract section region
    bg_subtracted_section = bg_subtracted_frame[start_y:end_y, start_x:end_x]
    
    # Exhaustive search for optimal min_size for this section
    best_min_size, actual_count, iterations = exhaustive_search_min_size_for_section(
        hsv_masker, bg_subtracted_section, target_count=target_per_section,
        min_val=1, max_val=30
    )
    
    # Get final detection with optimal min_size
    original_min_size = hsv_masker.min_object_size
    hsv_masker.min_object_size = best_min_size
    contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_section)
    hsv_masker.min_object_size = original_min_size
    
    # Convert centroids back to full-frame coordinates
    full_frame_centroids = [(y + start_y, x + start_x) for y, x in centroids]
    
    # Convert contours back to full-frame coordinates
    full_frame_contours = []
    for contour in contours:
        # Shift contour points by start_x and start_y
        shifted_contour = contour.copy()
        shifted_contour[:, :, 0] += start_x  # x coordinate
        shifted_contour[:, :, 1] += start_y  # y coordinate
        full_frame_contours.append(shifted_contour)
    
    return full_frame_contours, full_frame_centroids, best_min_size, iterations


def count_fish_per_section(centroids, width, height, num_sections_per_half=7):
    """Count fish in each section (14 total: 7 top, 7 bottom)."""
    section_width = width / num_sections_per_half
    half_height = height // 2
    section_counts = [0] * (num_sections_per_half * 2)  # 14 sections total
    
    for y, x in centroids:
        # Determine which half
        is_bottom = y >= half_height
        
        # Determine which section within the half
        section_in_half = min(int(x / section_width), num_sections_per_half - 1)
        
        # Calculate global section index
        if is_bottom:
            section_idx = num_sections_per_half + section_in_half
        else:
            section_idx = section_in_half
        
        section_counts[section_idx] += 1
    
    return section_counts


class HalfSectionedPipeline(BasePipeline):
    """
    Half-Sectioned Pipeline.
    
    This pipeline:
    1. Splits the frame into top and bottom halves
    2. Each half is divided into 7 sections (like columns)
    3. Each of the 14 sections is processed independently
    4. For each section, exhaustively tries all min_size thresholds (1-30)
       until it gets exactly 2 fish or exhausts all possible thresholds
    """
    
    def __init__(self, *args, num_segments: int = 7, num_sections_per_half: int = 7, **kwargs):
        """
        Initialize half-sectioned pipeline.
        
        Args:
            num_segments: Number of temporal segments to break video into (default 7)
            num_sections_per_half: Number of sections per half (default 7, so 14 total)
            *args, **kwargs: Passed to BasePipeline
        """
        super().__init__(*args, **kwargs)
        self.num_segments = num_segments
        self.num_sections_per_half = num_sections_per_half
        self.target_per_section = 2  # Fixed at 2 fish per section
    
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
        Process video with half-sectioned detection.
        
        Breaks video into temporal segments, then processes each of the 14 sections
        (7 top, 7 bottom) independently within each segment.
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
        
        total_sections = self.num_sections_per_half * 2  # 14 sections total
        
        print(f"\nProcessing frames {start_frame} to {end_frame-1} ({actual_num_frames} frames) "
              f"with {total_sections} sections ({self.num_sections_per_half} top + {self.num_sections_per_half} bottom), "
              f"{self.num_segments} temporal segments, "
              f"targeting {self.target_per_section} fish per section (exhaustive threshold search per section)...")
        
        # Video writer
        height, width = self.video[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output_path = output_dir / f"half_sectioned_{total_sections}sections_{self.num_segments}seg.mp4"
        out = cv2.VideoWriter(str(video_output_path), fourcc, self.fps, (width, height), isColor=True)
        
        # Calculate section boundaries
        section_width = width / self.num_sections_per_half
        half_height = height // 2
        section_boundaries_x = [int(i * section_width) for i in range(self.num_sections_per_half + 1)]
        
        # Calculate segment boundaries
        frames_per_segment = actual_num_frames // self.num_segments
        segment_boundaries = [start_frame + i * frames_per_segment for i in range(self.num_segments + 1)]
        segment_boundaries[-1] = end_frame  # Ensure last segment goes to end
        
        # Statistics
        all_section_counts = []
        min_sizes_used_per_section = []  # List of lists: [section_0_sizes, section_1_sizes, ...]
        iterations_per_section = []  # List of lists: [section_0_iters, section_1_iters, ...]
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
                
                # Process each section independently
                all_contours = []
                all_centroids = []
                section_min_sizes = []
                section_iterations = []
                
                for section_idx in range(total_sections):
                    # Process this section independently
                    contours, centroids, min_size, iterations = process_section_independently(
                        self.hsv_masker,
                        bg_subtracted,
                        section_idx,
                        self.num_sections_per_half,
                        width,
                        height,
                        target_per_section=self.target_per_section
                    )
                    
                    all_contours.extend(contours)
                    all_centroids.extend(centroids)
                    section_min_sizes.append(min_size)
                    section_iterations.append(iterations)
                
                # Calculate section counts
                final_section_counts = count_fish_per_section(all_centroids, width, height, self.num_sections_per_half)
                
                all_section_counts.append(final_section_counts)
                min_sizes_used_per_section.append(section_min_sizes)
                iterations_per_section.append(section_iterations)
                total_fish_counts.append(len(all_centroids))
                
                # Create output frame
                output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
                
                # Draw horizontal line dividing top and bottom halves
                cv2.line(output_frame, (0, half_height), (width, half_height), (255, 255, 0), 3)
                
                # Draw vertical section boundaries
                for boundary in section_boundaries_x:
                    # Top half
                    cv2.line(output_frame, (boundary, 0), (boundary, half_height), (255, 255, 0), 2)
                    # Bottom half
                    cv2.line(output_frame, (boundary, half_height), (boundary, height), (255, 255, 0), 2)
                
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
                # Format section counts: top row then bottom row
                top_sections = final_section_counts[:self.num_sections_per_half]
                bottom_sections = final_section_counts[self.num_sections_per_half:]
                section_str = " ".join([f"S{i+1}:{c}" for i, c in enumerate(top_sections)]) + " | " + \
                             " ".join([f"S{i+8}:{c}" for i, c in enumerate(bottom_sections)])
                
                min_size_str = " ".join([f"S{i+1}:{s:2d}" for i, s in enumerate(section_min_sizes[:7])]) + " | " + \
                              " ".join([f"S{i+8}:{s:2d}" for i, s in enumerate(section_min_sizes[7:])])
                
                stats_text = [
                    f"Segment: {seg_idx + 1}/{self.num_segments} | Frame: {local_seg_idx:3d}/{seg_num_frames-1} (Global: {global_idx}) | Total Fish: {total_fish}",
                    f"Sections: {section_str}",
                    f"MinSizes: {min_size_str}",
                    f"Target: {self.target_per_section} per section (14 sections: 7 top + 7 bottom)"
                ]
                for j, text in enumerate(stats_text):
                    cv2.putText(output_frame, text, (10, 30 + j * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                out.write(output_frame)
                
                # Save individual frame (every 5th frame to save space)
                if local_seg_idx % 5 == 0:
                    frame_path = frames_dir / f"frame_{global_idx:04d}_seg{seg_idx+1}_sections.png"
                    cv2.imwrite(str(frame_path), output_frame)
                
                if local_seg_idx % 20 == 0:
                    avg_total = np.mean(total_fish_counts[-20:]) if len(total_fish_counts) >= 20 else np.mean(total_fish_counts)
                    avg_per_section = [np.mean([counts[j] for counts in all_section_counts[-20:]]) 
                                      for j in range(total_sections)] if len(all_section_counts) >= 20 else [np.mean([counts[j] for counts in all_section_counts]) for j in range(total_sections)]
                    print(f"  Segment {seg_idx + 1}, Frame {local_seg_idx:3d}/{seg_num_frames-1} (Global: {global_idx}): "
                          f"avg_total={avg_total:.1f} fish, avg_per_section={[f'{c:.1f}' for c in avg_per_section]}")
        
        out.release()
        
        # Generate summary
        self._generate_summary(
            output_dir, all_section_counts, min_sizes_used_per_section, iterations_per_section,
            total_fish_counts, duration_seconds, start_frame
        )
        
        print(f"\n✓ Results saved to: {output_dir}")
    
    def _generate_summary(self,
                         output_dir: Path,
                         all_section_counts: List[List[int]],
                         min_sizes_used_per_section: List[List[int]],
                         iterations_per_section: List[List[int]],
                         total_fish_counts: List[int],
                         duration_seconds: int,
                         start_frame: int) -> None:
        """Generate summary statistics and save to file."""
        total_sections = self.num_sections_per_half * 2
        
        # Analyze section accuracy
        frames_with_imperfect_sections = 0
        frames_with_all_perfect = 0
        sections_off_count = {i: 0 for i in range(total_sections + 1)}
        
        for section_counts in all_section_counts:
            sections_off = sum(1 for count in section_counts if count != self.target_per_section)
            sections_off_count[sections_off] += 1
            
            if sections_off > 0:
                frames_with_imperfect_sections += 1
            else:
                frames_with_all_perfect += 1
        
        # Calculate statistics
        avg_total = np.mean(total_fish_counts)
        std_total = np.std(total_fish_counts)
        avg_per_section = [np.mean([counts[j] for counts in all_section_counts]) for j in range(total_sections)]
        std_per_section = [np.std([counts[j] for counts in all_section_counts]) for j in range(total_sections)]
        
        # Calculate per-section min_size statistics
        avg_min_size_per_section = [np.mean([sizes[j] for sizes in min_sizes_used_per_section]) 
                                   for j in range(total_sections)]
        std_min_size_per_section = [np.std([sizes[j] for sizes in min_sizes_used_per_section]) 
                                   for j in range(total_sections)]
        
        # Calculate per-section iteration statistics
        avg_iterations_per_section = [np.mean([iters[j] for iters in iterations_per_section]) 
                                     for j in range(total_sections)]
        
        # Get variant name for summary
        variant_name = self.__class__.__name__
        
        summary = f"""
Half-Sectioned Pipeline - Results
==================================
Variant: {variant_name}

Configuration:
- Background Subtraction: V2 (Threshold={self.threshold}, Morphology={self.morph_kernel_size} kernel)
- HSV Masking: Lower HSV={self.hsv_lower}, Upper HSV={self.hsv_upper}
- Section-Based Detection: {total_sections} sections ({self.num_sections_per_half} top + {self.num_sections_per_half} bottom), {self.target_per_section} fish per section
- Temporal Segments: {self.num_segments} segments
- Independent Section Processing: Each section processed separately with its own adaptive threshold
- Frames Processed: {len(total_fish_counts)} ({duration_seconds} seconds @ {self.fps} FPS)
- Start Frame: {start_frame}, End Frame: {start_frame + len(total_fish_counts) - 1}

Fish Detection Statistics:
- Average Total Fish per Frame: {avg_total:.2f}
- Target Total: {total_sections * self.target_per_section}
- Standard Deviation: {std_total:.2f}
- Min Total: {np.min(total_fish_counts)}
- Max Total: {np.max(total_fish_counts)}

Per-Section Statistics (Top Half - Sections 1-7):
"""
        for j in range(self.num_sections_per_half):
            summary += f"- Section {j+1} (Top): avg={avg_per_section[j]:.2f}, std={std_per_section[j]:.2f}, target={self.target_per_section}\n"
            summary += f"  MinSize: avg={avg_min_size_per_section[j]:.2f}, std={std_min_size_per_section[j]:.2f}\n"
            summary += f"  Iterations: avg={avg_iterations_per_section[j]:.2f}\n"
        
        summary += f"\nPer-Section Statistics (Bottom Half - Sections 8-14):\n"
        for j in range(self.num_sections_per_half, total_sections):
            summary += f"- Section {j+1} (Bottom): avg={avg_per_section[j]:.2f}, std={std_per_section[j]:.2f}, target={self.target_per_section}\n"
            summary += f"  MinSize: avg={avg_min_size_per_section[j]:.2f}, std={std_min_size_per_section[j]:.2f}\n"
            summary += f"  Iterations: avg={avg_iterations_per_section[j]:.2f}\n"
        
        summary += f"""

Section Accuracy Analysis:
- Frames with ALL sections having exactly {self.target_per_section} fish: {frames_with_all_perfect} ({frames_with_all_perfect/len(total_fish_counts)*100:.2f}%)
- Frames with at least ONE section not having {self.target_per_section} fish: {frames_with_imperfect_sections} ({frames_with_imperfect_sections/len(total_fish_counts)*100:.2f}%)

Breakdown by number of sections off target:
"""
        for num_off in range(total_sections + 1):
            if sections_off_count[num_off] > 0:
                summary += f"- {num_off} section(s) off: {sections_off_count[num_off]} frames ({sections_off_count[num_off]/len(total_fish_counts)*100:.2f}%)\n"
        
        summary += f"""

Method Details:
1. Frame Segmentation:
   - Frame divided into top and bottom halves (horizontal split)
   - Each half divided into {self.num_sections_per_half} vertical sections
   - Total: {total_sections} sections ({self.num_sections_per_half} top + {self.num_sections_per_half} bottom)
   
2. Temporal Segmentation:
   - Video divided into {self.num_segments} equal temporal segments
   - Each segment processed independently
   
3. Independent Section Processing:
   - Each section processed separately within each segment
   - Each section uses its own exhaustive min_size threshold search
   - Tries all thresholds from 1 to 30 until it gets {self.target_per_section} fish or exhausts all options
   - Target: {self.target_per_section} fish per section
   
4. V2 Background Subtraction:
   - Lower threshold ({self.threshold} vs 25) captures more faint fish parts
   - Morphological closing ({self.morph_kernel_size}) prevents blob splitting
   
5. HSV Masking:
   - Filters for bright objects (value > {self.hsv_lower[2]})
   - Reduces false detections significantly

Output Files:
- Video: half_sectioned_{total_sections}sections_{self.num_segments}seg.mp4
- Background model: background_model.npy
- Summary: half_sectioned_{total_sections}sections_{self.num_segments}seg_summary.txt
    """
        
        with open(output_dir / f"half_sectioned_{total_sections}sections_{self.num_segments}seg_summary.txt", 'w') as f:
            f.write(summary)
        
        print("\n" + summary)

