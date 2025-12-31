#!/usr/bin/env python3
"""
Complete Pipeline V2: Improved Background Subtraction + HSV Masking + Adaptive Detection

This pipeline uses:
- V2 Background Subtraction (threshold=15 + morphological closing)
- HSV Masking for bright fish detection
- Adaptive min_object_size to target exactly 28 fish per frame (no tolerance)

This is the ORIGINAL standalone pipeline that processes the whole frame at once.
No column/section division - treats entire frame as one unit.
"""

import cv2
import numpy as np
import pims
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

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
    
    return best_min_size, best_count, max_iterations


def process_complete_pipeline_v2(video_path, output_dir, fps=20, target_fish=28, duration_seconds=10):
    """
    Process video with V2 background subtraction and adaptive detection.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save output
        fps: Frames per second
        target_fish: Target number of fish per frame (default 28)
        duration_seconds: Duration in seconds to process
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    num_frames = int(fps * duration_seconds)
    
    # Initialize V2 background subtractor
    print("Initializing V2 background subtractor (threshold=15 + morphology)...")
    bg_subtractor = TrackingProgramBackgroundSubtractorV2(threshold=15, morph_kernel_size=(5, 5))
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
    
    # Check if we have enough frames
    actual_num_frames = min(num_frames, len(video))
    
    print(f"\nProcessing {actual_num_frames} frames (target: {target_fish} fish per frame)...")
    
    # Video writer
    height, width = video[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output_path = output_dir / "complete_pipeline_v2.mp4"
    out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height), isColor=True)
    
    # Statistics
    fish_counts = []
    min_sizes_used = []
    iterations_per_frame = []
    
    for i in range(actual_num_frames):
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
            f"Frame: {i:3d}/{actual_num_frames-1} | Fish: {actual_count}",
            f"MinSize: {best_min_size:2d} | Iterations: {iterations}",
            f"V2 BG Sub (T=15+Morph) + HSV Masking"
        ]
        for j, text in enumerate(stats_text):
            cv2.putText(output_frame, text, (10, 30 + j * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(output_frame)
        
        # Save individual frame (every 5th frame to save space)
        if i % 5 == 0:
            frame_path = frames_dir / f"frame_{i:04d}.png"
            cv2.imwrite(str(frame_path), output_frame)
        
        if i % 20 == 0:
            avg_fish = np.mean(fish_counts[-20:]) if len(fish_counts) >= 20 else np.mean(fish_counts)
            print(f"  Frame {i:3d}/{actual_num_frames-1}: avg_fish={avg_fish:.1f}, "
                  f"min_size={best_min_size:2d}, iterations={iterations}")
    
    out.release()
    
    # Generate summary
    avg_fish = np.mean(fish_counts)
    std_fish = np.std(fish_counts)
    avg_min_size = np.mean(min_sizes_used)
    avg_iterations = np.mean(iterations_per_frame)
    
    summary = f"""
Complete Pipeline V2 - Results
==============================
Configuration:
- Background Subtraction: V2 (Threshold=15, Morphology=(5, 5) kernel)
- HSV Masking: Lower HSV=(0, 0, 100), Upper HSV=(180, 255, 255)
- Target Fish: {target_fish} per frame
- Frames Processed: {actual_num_frames} ({duration_seconds} seconds @ {fps} FPS)

Fish Detection Statistics:
- Average Fish per Frame: {avg_fish:.2f}
- Target: {target_fish}
- Standard Deviation: {std_fish:.2f}
- Min: {np.min(fish_counts)}
- Max: {np.max(fish_counts)}

Threshold Statistics:
- Average min_size: {avg_min_size:.2f}
- Average iterations: {avg_iterations:.2f}

Method Details:
1. V2 Background Subtraction:
   - Lower threshold (15 vs 25) captures more faint fish parts
   - Morphological closing ((5, 5)) prevents blob splitting
   
2. HSV Masking:
   - Filters for bright objects (value > 100)
   - Reduces false detections significantly
   
3. Adaptive Detection:
   - Binary search for optimal min_object_size
   - Targets exactly {target_fish} fish per frame

Output Files:
- Video: complete_pipeline_v2.mp4
- Background model: background_model.npy
- Summary: complete_pipeline_v2_summary.txt
    """
    
    with open(output_dir / "complete_pipeline_v2_summary.txt", 'w') as f:
        f.write(summary)
    
    print("\n" + summary)
    print(f"\n✓ Results saved to: {output_dir}")


def main():
    """Main execution."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/complete_pipeline_v2_10sec")
    
    process_complete_pipeline_v2(
        video_path=video_path,
        output_dir=output_dir,
        fps=20,
        target_fish=28,
        duration_seconds=10
    )


if __name__ == "__main__":
    main()

