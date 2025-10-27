#!/usr/bin/env python3
"""
HSV Analysis with V2 Background Subtraction

Simple pipeline combining:
- V2 Background Subtraction (threshold=15 + morphology)
- Existing HSV Masker
"""

import cv2
import numpy as np
import pims
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2


def create_v2_bg_subtractor():
    """Create V2 background subtractor with improved settings."""
    return TrackingProgramBackgroundSubtractorV2(
        threshold=15,
        morph_kernel_size=(5, 5)
    )


def process_video_v2(video_path, output_dir, num_frames=200, fps=20, target_fish=28):
    """Process video with V2 background subtraction + HSV masking."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    print(f"Initializing V2 background subtractor...")
    bg_subtractor = create_v2_bg_subtractor()
    
    print(f"Creating background model...")
    bg_subtractor.create_background_model(video_path)
    bg_subtractor.save_background_model(output_dir / "background_model.npy")
    
    print(f"Initializing HSV masker...")
    hsv_masker = HSVMasker(
        lower_hsv=(0, 0, 100),
        upper_hsv=(180, 255, 255),
        min_object_size=10,
        apply_morphology=True
    )
    
    print(f"Loading video: {video_path}")
    video = pims.PyAVReaderIndexed(str(video_path))
    
    # Video writer
    height, width = video[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path_out = output_dir / f"hsv_v2_bg_subtraction_{target_fish}fish.mp4"
    out = cv2.VideoWriter(str(video_path_out), fourcc, fps, (width, height), isColor=True)
    
    print(f"Processing {num_frames} frames...")
    
    fish_counts = []
    
    for i in range(min(num_frames, len(video))):
        frame = video[i]
        gray_frame = frame[:, :, 0] if len(frame.shape) == 3 else frame
        
        # V2 background subtraction
        bg_subtracted = bg_subtractor.apply_background_subtraction(gray_frame)
        
        # HSV masking
        hsv_result = hsv_masker.process_background_subtracted_frame(bg_subtracted)
        
        # Get fish count
        contours, centroids = hsv_masker.detect_fish_contours_and_centroids_bg_subtracted(bg_subtracted)
        fish_count = len(centroids)
        fish_counts.append(fish_count)
        
        # Create output frame
        output_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Draw contours
        if len(contours) > 0:
            cv2.drawContours(output_frame, contours, -1, (0, 255, 0), 2)
        
        # Draw centroids
        for y, x in centroids:
            cv2.circle(output_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        # Add text
        cv2.putText(output_frame, f'Frame: {i} | Fish: {fish_count} | Avg: {np.mean(fish_counts):.1f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(output_frame)
        
        if (i + 1) % 20 == 0:
            print(f"  Frame {i+1}/{num_frames} - Fish: {fish_count}, Avg: {np.mean(fish_counts):.1f}")
    
    out.release()
    
    # Summary
    avg_count = np.mean(fish_counts)
    std_count = np.std(fish_counts)
    
    summary = f"""
HSV Analysis with V2 Background Subtraction
============================================

Configuration:
- Background Subtraction: V2 (Threshold=15, Morphology=5x5)
- HSV Masking: (0,0,100) to (180,255,255)
- Min Object Size: 10
- Target Fish: {target_fish}

Results:
- Frames Processed: {len(fish_counts)}
- Average Fish: {avg_count:.2f}
- Std Deviation: {std_count:.2f}
- Min/Max: {np.min(fish_counts)}/{np.max(fish_counts)}
- Target: {target_fish}
- Difference: {avg_count - target_fish:.2f}

Output:
- Video: {video_path_out.name}
- Background Model: background_model.npy
    """
    
    with open(output_dir / "summary.txt", 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"\n✓ Results saved to: {output_dir}")


def main():
    """Main execution."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/hsv_v2_bg_subtraction")
    
    process_video_v2(video_path, output_dir, num_frames=200, fps=20, target_fish=28)


if __name__ == "__main__":
    main()

