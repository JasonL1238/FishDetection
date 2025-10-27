#!/usr/bin/env python3
"""
Adaptive HSV Analysis with No Tolerance

This script performs HSV analysis with adaptive min_object_size adjustment
to find exactly 28 fish per frame using binary search algorithm.
"""

import cv2
import numpy as np
import pims
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor import TrackingProgramBackgroundSubtractor


def binary_search_min_size(hsv_masker, bg_subtracted_frame, target_count, min_val=1, max_val=30, max_iterations=10):
    """
    Use binary search to find min_object_size that gives exactly target_count fish.
    
    Args:
        hsv_masker: HSVMasker instance
        bg_subtracted_frame: Background subtracted frame
        target_count: Target number of fish (28)
        min_val: Minimum min_object_size to try
        max_val: Maximum min_object_size to try
        max_iterations: Maximum iterations for binary search
        
    Returns:
        Tuple of (best_min_size, actual_count)
    """
    best_min_size = min_val
    best_count = 0
    
    for iteration in range(max_iterations):
        # Try the midpoint
        test_size = (min_val + max_val) // 2
        
        # Set the min_object_size
        original_min_size = hsv_masker.min_object_size
        hsv_masker.min_object_size = test_size
        
        # Get count
        _, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted_frame)
        count = len(centroids)
        
        # Restore original min_size
        hsv_masker.min_object_size = original_min_size
        
        # Check if we found the target
        if count == target_count:
            return test_size, count, iteration + 1
        
        # Update best approximation
        if abs(count - target_count) < abs(best_count - target_count):
            best_min_size = test_size
            best_count = count
        
        # Adjust search range
        if count > target_count:
            # Too many fish, increase min_size
            min_val = test_size + 1
        else:
            # Too few fish, decrease min_size
            max_val = test_size - 1
        
        # Check if search space exhausted
        if min_val > max_val:
            break
    
    return best_min_size, best_count, iteration + 1


def process_adaptive_hsv_analysis(video_path, output_dir, fps=20, target_fish=28):
    """
    Process video with adaptive HSV analysis targeting exactly target_fish per frame.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save output
        fps: Video frame rate
        target_fish: Target number of fish per frame
    """
    # Create output directories
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Calculate number of frames for 10 seconds
    num_frames = fps * 10  # 10 seconds
    
    # Initialize tracking program background subtractor
    print("Initializing tracking program background subtractor...")
    bg_subtractor = TrackingProgramBackgroundSubtractor(
        threshold=25,
        blur_kernel_size=(3, 3),
        blur_sigma=0
    )
    
    # Create background model using tracking program method
    print("Creating background model using tracking program method...")
    background_model = bg_subtractor.create_background_model(video_path)
    
    # Initialize HSV masker
    hsv_masker = HSVMasker(
        lower_hsv=(0, 0, 100),
        upper_hsv=(180, 255, 255),
        min_object_size=10,  # Initial value, will be adjusted
        apply_morphology=True
    )
    
    # Load video
    print(f"Loading video: {video_path}")
    video = pims.PyAVReaderIndexed(str(video_path))
    
    print(f"Processing first {num_frames} frames with adaptive min_size targeting exactly {target_fish} fish per frame...")
    
    # Prepare video writer for HSV analysis output
    height, width = video[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output_path = output_dir / f"adaptive_hsv_analysis_{target_fish}_fish.mp4"
    out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height), isColor=True)
    
    # Process frames
    fish_counts = []
    min_sizes_used = []
    iterations_per_frame = []
    
    try:
        for i in range(min(num_frames, len(video))):
            frame = video[i]
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray_frame = frame[:, :, 0]
            else:
                gray_frame = frame
            
            # Apply tracking program background subtraction
            bg_subtracted = bg_subtractor.apply_background_subtraction(gray_frame)
            
            # Use binary search to find optimal min_size for exactly target_fish fish
            best_min_size, actual_count, iterations = binary_search_min_size(
                hsv_masker, 
                bg_subtracted, 
                target_count=target_fish,
                min_val=4,
                max_val=25,
                max_iterations=10
            )
            
            # Set the optimal min_size and get final detection
            hsv_masker.min_object_size = best_min_size
            _, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted)
            
            fish_counts.append(len(centroids))
            min_sizes_used.append(best_min_size)
            iterations_per_frame.append(iterations)
            
            # Create visualization for individual frames (every 2nd frame to save space)
            if i % 2 == 0:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                
                # Original frame
                axes[0, 0].imshow(gray_frame, cmap='gray')
                axes[0, 0].set_title(f'Original Frame {i} (t={i/fps:.1f}s)')
                axes[0, 0].axis('off')
                
                # Tracking program background subtracted
                axes[0, 1].imshow(bg_subtracted, cmap='gray')
                axes[0, 1].set_title(f'Tracking BG Subtracted Frame {i}')
                axes[0, 1].axis('off')
                
                # HSV masked result
                hsv_masked = hsv_masker.process_background_subtracted_frame(bg_subtracted)
                axes[1, 0].imshow(hsv_masked, cmap='viridis')
                axes[1, 0].set_title(f'HSV Masked (min_size={best_min_size})')
                axes[1, 0].axis('off')
                
                # Overlay contours and centroids on original
                axes[1, 1].imshow(gray_frame, cmap='gray')
                
                # Convert HSV masked to binary for contour detection
                mask_binary = (hsv_masked > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours
                for contour in contours:
                    contour_points = contour.reshape(-1, 2)
                    axes[1, 1].plot(contour_points[:, 0], contour_points[:, 1], 'yellow', linewidth=2)
                
                # Draw centroids
                if centroids:
                    y_coords, x_coords = zip(*centroids)
                    axes[1, 1].scatter(x_coords, y_coords, c='red', s=100, marker='o', 
                                     edgecolors='white', linewidths=2, alpha=0.8)
                    axes[1, 1].scatter(x_coords, y_coords, c='red', s=20, marker='+', 
                                     linewidths=3, alpha=1.0)
                
                axes[1, 1].set_title(f'Detected Fish: {len(centroids)} (target: {target_fish})')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                
                # Save visualization
                frame_output_path = frames_dir / f'frame_{i:03d}_adaptive_hsv_analysis.png'
                plt.savefig(frame_output_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            # Create video frame with HSV analysis overlay
            video_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            
            # Get HSV masked result for video
            hsv_masked = hsv_masker.process_background_subtracted_frame(bg_subtracted)
            
            # Convert HSV masked to binary for contour detection
            mask_binary = (hsv_masked > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on video frame
            for contour in contours:
                cv2.drawContours(video_frame, [contour], -1, (0, 255, 255), 2)  # Yellow contours
            
            # Draw centroids on video frame
            for centroid in centroids:
                y, x = int(centroid[0]), int(centroid[1])
                cv2.circle(video_frame, (x, y), 5, (0, 0, 255), -1)  # Red centroids
                cv2.circle(video_frame, (x, y), 8, (255, 255, 255), 2)  # White outline
            
            # Add frame info text
            cv2.putText(video_frame, f'Frame: {i} | Time: {i/fps:.1f}s | Fish: {len(centroids)} | min_size: {best_min_size}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame to video
            out.write(video_frame)
            
            if i % 10 == 0:
                print(f"Processed frame {i+1}/{num_frames} - Found {len(centroids)} fish (min_size: {best_min_size}, iterations: {iterations})")
    
    finally:
        out.release()
    
    # Create summary
    summary_path = output_dir / f'adaptive_hsv_analysis_{target_fish}_fish_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Adaptive HSV Analysis - Target {target_fish} Fish per Frame\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames processed: {num_frames}\n")
        f.write(f"Frame rate: {fps} FPS\n")
        f.write(f"Duration: 10.0 seconds\n")
        f.write(f"Target fish count per frame: {target_fish}\n")
        f.write(f"Background subtractor: TrackingProgramBackgroundSubtractor\n")
        f.write(f"Background threshold: {bg_subtractor.threshold}\n")
        f.write(f"Blur kernel size: {bg_subtractor.blur_kernel_size}\n")
        f.write(f"Blur sigma: {bg_subtractor.blur_sigma}\n")
        f.write(f"HSV range: {hsv_masker.get_hsv_range()}\n")
        f.write(f"Initial min object size: 10\n")
        f.write(f"Morphology applied: {hsv_masker.apply_morphology}\n\n")
        
        f.write("Adaptive Detection Statistics:\n")
        f.write(f"Average fish per frame: {np.mean(fish_counts):.2f}\n")
        f.write(f"Max fish in single frame: {max(fish_counts)}\n")
        f.write(f"Min fish in single frame: {min(fish_counts)}\n")
        f.write(f"Standard deviation: {np.std(fish_counts):.2f}\n")
        f.write(f"Frames with exact target count: {sum(1 for c in fish_counts if c == target_fish)}\n")
        f.write(f"Frames that hit max iterations: {sum(1 for i in iterations_per_frame if i >= 10)}\n\n")
        
        f.write("Min Size Adjustment Statistics:\n")
        f.write(f"Average min_size used: {np.mean(min_sizes_used):.2f}\n")
        f.write(f"Max min_size used: {max(min_sizes_used)}\n")
        f.write(f"Min min_size used: {min(min_sizes_used)}\n")
        f.write(f"Min_size standard deviation: {np.std(min_sizes_used):.2f}\n")
        f.write(f"Average iterations per frame: {np.mean(iterations_per_frame):.2f}\n\n")
        
        f.write("Frame-by-frame results:\n")
        f.write("Frame | Fish Count | Min Size | Iterations\n")
        f.write("-" * 40 + "\n")
        for i, (count, size, iters) in enumerate(zip(fish_counts, min_sizes_used, iterations_per_frame)):
            f.write(f"{i:5d} | {count:10d} | {size:8d} | {iters:10d}\n")
        
        f.write("\nOutput Files:\n")
        f.write(f"- Video: adaptive_hsv_analysis_{target_fish}_fish.mp4\n")
        f.write(f"- Frame analysis images: frames/ directory\n")
        f.write(f"- Summary: adaptive_hsv_analysis_{target_fish}_fish_summary.txt\n\n")
        
        f.write("Analysis shows:\n")
        f.write("1. Original frame\n")
        f.write("2. Tracking program background subtracted frame\n")
        f.write("3. HSV masked result with contours (adaptive min_size)\n")
        f.write("4. Detected fish centroids overlaid on original\n\n")
        
        f.write("Key features:\n")
        f.write("- Dynamic adjustment of min_object_size per frame\n")
        f.write("- Binary search algorithm for optimal size finding\n")
        f.write(f"- Strict target-driven detection (exactly {target_fish} fish per frame, no tolerance)\n")
        f.write("- Falls back to best approximation only when max iterations reached\n")
        f.write("- Frame-by-frame min_size tracking\n")
    
    print(f"\nAdaptive HSV analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Video output: {video_output_path}")
    print(f"Frame analysis images: {frames_dir}")
    print(f"Summary: {summary_path}")
    print(f"Average fish per frame: {np.mean(fish_counts):.2f}")
    print(f"Max fish in single frame: {max(fish_counts)}")
    print(f"Min fish in single frame: {min(fish_counts)}")
    print(f"Frames with exact target count: {sum(1 for c in fish_counts if c == target_fish)}/{len(fish_counts)}")


def main():
    """Main function to run the adaptive HSV analysis."""
    # Set paths
    video_path = "/Users/jasonli/FishDetection/data/input/videos/Clutch1_20250804_122715.mp4"
    output_dir = "/Users/jasonli/FishDetection/data/output/adaptive_hsv_analysis_no_tolerance"
    
    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print("Starting adaptive HSV analysis with no tolerance targeting exactly 28 fish per frame...")
    print(f"Video: {video_path}")
    print(f"Output directory: {output_dir}")
    
    # Run the analysis
    process_adaptive_hsv_analysis(
        video_path=video_path,
        output_dir=output_dir,
        fps=20,
        target_fish=28  # Exactly 28 fish with no tolerance
    )


if __name__ == "__main__":
    main()

