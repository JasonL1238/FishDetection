#!/usr/bin/env python3
"""
Complete Pipeline V2: Improved Background Subtraction + HSV Masking + Adaptive Detection

This pipeline uses:
- V2 Background Subtraction (threshold=15 + morphological closing)
- HSV Masking for bright fish detection
- Adaptive min_object_size to target exactly 28 fish per frame (no tolerance)
"""

import cv2
import numpy as np
import pims
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2


def binary_search_min_size(hsv_masker, bg_subtracted_frame, target_count, min_val=1, max_val=30, max_iterations=10):
    """Binary search to find min_object_size that gives exactly target_count fish."""
    best_min_size = min_val
    best_count = 0
    
    for iteration in range(max_iterations):
        test_size = (min_val + max_val) // 2
        
        original_min_size = hsv_masker.min_object_size
        hsv_masker.min_object_size = test_size
        
        _, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_frame)
        count = len(centroids)
        
        hsv_masker.min_object_size = original_min_size
        
        if count == target_count:
            return test_size, count, iteration + 1
        
        if abs(count - target_count) < abs(best_count - target_count):
            best_min_size = test_size
            best_count = count
        
        if count > target_count:
            min_val = test_size + 1
        else:
            max_val = test_size - 1
        
        if min_val > max_val:
            break
    
    return best_min_size, best_count, iteration + 1


def process_complete_pipeline_v2(video_path, output_dir, fps=20, target_fish=28, duration_seconds=10):
    """Process a specified duration with complete V2 pipeline."""
    
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
    
    print(f"\nProcessing {num_frames} frames targeting exactly {target_fish} fish per frame...")
    
    # Video writer
    height, width = video[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output_path = output_dir / f"complete_pipeline_v2_{target_fish}_fish.mp4"
    out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height), isColor=True)
    
    # Statistics
    fish_counts = []
    min_sizes_used = []
    iterations_per_frame = []
    
    for i in range(min(num_frames, len(video))):
        frame = video[i]
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray_frame = frame[:, :, 0]
        else:
            gray_frame = frame
        
        # Apply V2 background subtraction
        bg_subtracted = bg_subtractor.apply_background_subtraction(gray_frame)
        
        # Binary search for optimal min_size
        best_min_size, actual_count, iterations = binary_search_min_size(
            hsv_masker,
            bg_subtracted,
            target_count=target_fish,
            min_val=4,
            max_val=25,
            max_iterations=10
        )
        
        # Set optimal min_size and get final detection
        hsv_masker.min_object_size = best_min_size
        contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted)
        
        fish_counts.append(len(centroids))
        min_sizes_used.append(best_min_size)
        iterations_per_frame.append(iterations)
        
        # Create output frame
        output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Draw contours
        if len(contours) > 0:
            cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 2)
        
        # Draw centroids
        for y, x in centroids:
            cv2.circle(output_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # Add stats text
        stats_text = [
            f"Frame: {i:3d}/{num_frames-1} | Fish: {actual_count}",
            f"MinSize: {best_min_size:2d} | Iterations: {iterations}",
            f"V2 BG Sub (T=15+Morph) + HSV Masking"
        ]
        for j, text in enumerate(stats_text):
            cv2.putText(output_frame, text, (10, 30 + j * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(output_frame)
        
        # Save individual frame (every 5th frame to save space)
        if i % 5 == 0:
            frame_path = frames_dir / f"frame_{i:04d}_complete_v2.png"
            cv2.imwrite(str(frame_path), output_frame)
        
        if i % 20 == 0:
            print(f"  Frame {i:3d}/{num_frames-1}: avg={np.mean(fish_counts):.1f} fish, "
                  f"avg_min_size={np.mean(min_sizes_used):.1f}")
    
    out.release()
    
    # Create summary
    avg_count = np.mean(fish_counts)
    std_count = np.std(fish_counts)
    min_count = np.min(fish_counts)
    max_count = np.max(fish_counts)
    avg_min_size = np.mean(min_sizes_used)
    avg_iterations = np.mean(iterations_per_frame)
    
    summary = f"""
Complete Pipeline V2 - Results Summary
=======================================

Configuration:
- Background Subtraction: V2 (Threshold=15, Morphology=5x5 kernel)
- HSV Masking: Lower HSV=(0,0,100), Upper HSV=(180,255,255)
- Adaptive Detection: Binary search for exactly {target_fish} fish
- Frames Processed: {len(fish_counts)} ({duration_seconds} seconds @ {fps} FPS)

Fish Detection Statistics:
- Average Fish per Frame: {avg_count:.2f}
- Target: {target_fish}
- Accuracy: {(1 - abs(avg_count - target_fish) / target_fish) * 100:.1f}%
- Standard Deviation: {std_count:.2f}
- Min Fish: {min_count}
- Max Fish: {max_count}

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
   
3. Adaptive min_object_size:
   - Binary search adjusts min_size per frame
   - Targets exactly {target_fish} fish per frame
   - No tolerance approach

Output Files:
- Video: complete_pipeline_v2_{target_fish}_fish.mp4
- Background model: background_model.npy
- Summary: complete_pipeline_v2_{target_fish}_fish_summary.txt
    """
    
    with open(output_dir / f"complete_pipeline_v2_{target_fish}_fish_summary.txt", 'w') as f:
        f.write(summary)
    
    print("\n" + summary)
    print(f"\n✓ Results saved to: {output_dir}")


def main():
    """Main execution."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/complete_pipeline_v2_5min")
    
    process_complete_pipeline_v2(
        video_path=video_path,
        output_dir=output_dir,
        fps=20,
        target_fish=28,
        duration_seconds=5 * 60
    )


if __name__ == "__main__":
    main()

