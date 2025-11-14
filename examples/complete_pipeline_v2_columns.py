#!/usr/bin/env python3
"""
Complete Pipeline V2 with Column-Based Detection: 7 columns, 4 fish per column

This pipeline uses:
- V2 Background Subtraction (threshold=15 + morphological closing)
- HSV Masking for bright fish detection
- Column-based adaptive detection: divides frame into 7 columns, targets 4 fish per column
- Draws column boundaries on output video
"""

import cv2
import numpy as np
import pims
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2


def count_fish_per_column(centroids, width, num_columns=7):
    """Count fish in each column."""
    column_width = width / num_columns
    column_counts = [0] * num_columns
    
    for y, x in centroids:
        column_idx = min(int(x / column_width), num_columns - 1)
        column_counts[column_idx] += 1
    
    return column_counts


def binary_search_min_size_columns(hsv_masker, bg_subtracted_frame, width, num_columns=7, 
                                    target_per_column=4, min_val=1, max_val=30, max_iterations=15):
    """Binary search to find min_object_size that gives target_per_column fish in each column."""
    best_min_size = min_val
    best_score = float('inf')
    best_column_counts = None
    
    for iteration in range(max_iterations):
        test_size = (min_val + max_val) // 2
        
        original_min_size = hsv_masker.min_object_size
        hsv_masker.min_object_size = test_size
        
        _, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_frame)
        column_counts = count_fish_per_column(centroids, width, num_columns)
        
        hsv_masker.min_object_size = original_min_size
        
        # Calculate score: sum of absolute differences from target
        score = sum(abs(count - target_per_column) for count in column_counts)
        
        # Check if all columns have exactly target_per_column
        if all(count == target_per_column for count in column_counts):
            return test_size, column_counts, iteration + 1
        
        # Track best result
        if score < best_score:
            best_min_size = test_size
            best_score = score
            best_column_counts = column_counts.copy()
        
        # Adjust search range
        total_fish = sum(column_counts)
        target_total = num_columns * target_per_column
        
        if total_fish > target_total:
            min_val = test_size + 1
        else:
            max_val = test_size - 1
        
        if min_val > max_val:
            break
    
    return best_min_size, best_column_counts, iteration + 1


def process_complete_pipeline_v2_columns(video_path, output_dir, fps=20, num_columns=7, 
                                         target_per_column=4, duration_seconds=10):
    """Process video with column-based detection: 7 columns, 4 fish per column."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    num_frames = int(fps * duration_seconds)
    
    # Initialize V2 background subtractor
    print("Initializing V2 background subtractor (threshold=15 + morphology)...")
    bg_subtractor = TrackingProgramBackgroundSubtractorV2(
        threshold=15,
        morph_kernel_size=(5, 5)
    )
    
    # Create background model
    print("Creating background model...")
    bg_subtractor.create_background_model(video_path)
    bg_subtractor.save_background_model(output_dir / "background_model.npy")
    
    # Initialize HSV masker
    print("Initializing HSV masker...")
    hsv_masker = HSVMasker(
        lower_hsv=(0, 0, 100),
        upper_hsv=(180, 255, 255),
        min_object_size=10,
        apply_morphology=True
    )
    
    # Load video
    print(f"Loading video: {video_path}")
    video = pims.PyAVReaderIndexed(str(video_path))
    
    print(f"\nProcessing {num_frames} frames with {num_columns} columns, targeting {target_per_column} fish per column...")
    
    # Video writer
    height, width = video[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output_path = output_dir / f"complete_pipeline_v2_columns_{num_columns}x{target_per_column}.mp4"
    out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height), isColor=True)
    
    # Calculate column boundaries
    column_width = width / num_columns
    column_boundaries = [int(i * column_width) for i in range(num_columns + 1)]
    
    # Statistics
    all_column_counts = []
    min_sizes_used = []
    iterations_per_frame = []
    total_fish_counts = []
    
    for i in range(min(num_frames, len(video))):
        frame = video[i]
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray_frame = frame[:, :, 0]
        else:
            gray_frame = frame
        
        # Apply V2 background subtraction
        bg_subtracted = bg_subtractor.apply_background_subtraction(gray_frame)
        
        # Binary search for optimal min_size based on column distribution
        best_min_size, column_counts, iterations = binary_search_min_size_columns(
            hsv_masker,
            bg_subtracted,
            width,
            num_columns=num_columns,
            target_per_column=target_per_column,
            min_val=4,
            max_val=25,
            max_iterations=15
        )
        
        # Set optimal min_size and get final detection
        hsv_masker.min_object_size = best_min_size
        contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted)
        
        # Recalculate column counts with final detection
        final_column_counts = count_fish_per_column(centroids, width, num_columns)
        
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
            f"Frame: {i:3d}/{num_frames-1} | Total Fish: {total_fish}",
            f"Columns: {column_str}",
            f"MinSize: {best_min_size:2d} | Iterations: {iterations}",
            f"Target: {target_per_column} per column"
        ]
        for j, text in enumerate(stats_text):
            cv2.putText(output_frame, text, (10, 30 + j * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(output_frame)
        
        # Save individual frame (every 5th frame to save space)
        if i % 5 == 0:
            frame_path = frames_dir / f"frame_{i:04d}_columns.png"
            cv2.imwrite(str(frame_path), output_frame)
        
        if i % 20 == 0:
            avg_total = np.mean(total_fish_counts)
            avg_min_size = np.mean(min_sizes_used)
            avg_per_col = [np.mean([counts[j] for counts in all_column_counts]) for j in range(num_columns)]
            print(f"  Frame {i:3d}/{num_frames-1}: avg_total={avg_total:.1f} fish, "
                  f"avg_per_col={[f'{c:.1f}' for c in avg_per_col]}, "
                  f"avg_min_size={avg_min_size:.1f}")
    
    out.release()
    
    # Analyze column accuracy
    frames_with_imperfect_columns = 0
    frames_with_all_perfect = 0
    columns_off_count = {i: 0 for i in range(num_columns + 1)}  # Track how many columns are off per frame
    
    for column_counts in all_column_counts:
        columns_off = sum(1 for count in column_counts if count != target_per_column)
        columns_off_count[columns_off] += 1
        
        if columns_off > 0:
            frames_with_imperfect_columns += 1
        else:
            frames_with_all_perfect += 1
    
    # Create summary
    avg_total = np.mean(total_fish_counts)
    std_total = np.std(total_fish_counts)
    avg_per_column = [np.mean([counts[j] for counts in all_column_counts]) for j in range(num_columns)]
    std_per_column = [np.std([counts[j] for counts in all_column_counts]) for j in range(num_columns)]
    avg_min_size = np.mean(min_sizes_used)
    avg_iterations = np.mean(iterations_per_frame)
    
    summary = f"""
Complete Pipeline V2 - Column-Based Detection Results
=====================================================

Configuration:
- Background Subtraction: V2 (Threshold=15, Morphology=5x5 kernel)
- HSV Masking: Lower HSV=(0,0,100), Upper HSV=(180,255,255)
- Column-Based Detection: {num_columns} columns, {target_per_column} fish per column
- Frames Processed: {len(total_fish_counts)} ({duration_seconds} seconds @ {fps} FPS)

Fish Detection Statistics:
- Average Total Fish per Frame: {avg_total:.2f}
- Target Total: {num_columns * target_per_column}
- Standard Deviation: {std_total:.2f}
- Min Total: {np.min(total_fish_counts)}
- Max Total: {np.max(total_fish_counts)}

Per-Column Statistics:
"""
    for j in range(num_columns):
        summary += f"- Column {j+1}: avg={avg_per_column[j]:.2f}, std={std_per_column[j]:.2f}, target={target_per_column}\n"
    
    summary += f"""

Column Accuracy Analysis:
- Frames with ALL columns having exactly {target_per_column} fish: {frames_with_all_perfect} ({frames_with_all_perfect/len(total_fish_counts)*100:.2f}%)
- Frames with at least ONE column not having {target_per_column} fish: {frames_with_imperfect_columns} ({frames_with_imperfect_columns/len(total_fish_counts)*100:.2f}%)

Breakdown by number of columns off target:
"""
    for num_off in range(num_columns + 1):
        if columns_off_count[num_off] > 0:
            summary += f"- {num_off} column(s) off: {columns_off_count[num_off]} frames ({columns_off_count[num_off]/len(total_fish_counts)*100:.2f}%)\n"
    
    summary += f"""
Adaptive Parameters:
- Average Min Size Used: {avg_min_size:.2f}
- Average Binary Search Iterations: {avg_iterations:.2f}

Method Details:
1. V2 Background Subtraction:
   - Lower threshold (15 vs 25) captures more faint fish parts
   - Morphological closing (5x5) prevents blob splitting
   
2. HSV Masking:
   - Filters for bright objects (value > 100)
   - Reduces false detections significantly
   
3. Column-Based Adaptive Detection:
   - Divides frame into {num_columns} equal columns
   - Binary search adjusts min_size to achieve {target_per_column} fish per column
   - Ensures even distribution across the frame
   - Column boundaries drawn as yellow vertical lines

Output Files:
- Video: complete_pipeline_v2_columns_{num_columns}x{target_per_column}.mp4
- Background model: background_model.npy
- Summary: complete_pipeline_v2_columns_{num_columns}x{target_per_column}_summary.txt
    """
    
    with open(output_dir / f"complete_pipeline_v2_columns_{num_columns}x{target_per_column}_summary.txt", 'w') as f:
        f.write(summary)
    
    print("\n" + summary)
    print(f"\n✓ Results saved to: {output_dir}")


def main():
    """Main execution."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/complete_pipeline_v2_columns_5min")
    
    process_complete_pipeline_v2_columns(
        video_path=video_path,
        output_dir=output_dir,
        fps=20,
        num_columns=7,
        target_per_column=4,
        duration_seconds=5 * 60
    )


if __name__ == "__main__":
    main()

