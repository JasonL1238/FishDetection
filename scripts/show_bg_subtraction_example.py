#!/usr/bin/env python3
"""
Show example of V2 background subtraction on first 2 frames.
"""

import cv2
import numpy as np
import pims
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2


def main():
    """Show background subtraction results for first 2 frames."""
    video_path = Path("data/input/videos/Clutch1_20250804_122715.mp4")
    output_dir = Path("data/output/bg_subtraction_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading video: {video_path}")
    video = pims.PyAVReaderIndexed(str(video_path))
    
    # Initialize V2 background subtractor
    print("Initializing V2 background subtractor (threshold=15 + morphology)...")
    bg_subtractor = TrackingProgramBackgroundSubtractorV2(
        threshold=15,
        morph_kernel_size=(5, 5)
    )
    
    # Create background model
    print("Creating background model...")
    bg_subtractor.create_background_model(video_path)
    
    # Process first 2 frames
    print("\nProcessing first 2 frames...")
    for frame_idx in range(2):
        frame = video[frame_idx]
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray_frame = frame[:, :, 0]  # Take first channel
        else:
            gray_frame = frame
        
        # Apply background subtraction
        bg_subtracted = bg_subtractor.apply_background_subtraction(gray_frame)
        
        # Save original frame
        original_path = output_dir / f"frame_{frame_idx:04d}_original.png"
        cv2.imwrite(str(original_path), gray_frame)
        
        # Save background subtracted result
        bg_path = output_dir / f"frame_{frame_idx:04d}_bg_subtracted.png"
        cv2.imwrite(str(bg_path), bg_subtracted)
        
        # Create side-by-side comparison
        # Normalize bg_subtracted for display (0-255)
        bg_display = (bg_subtracted * 255).astype(np.uint8)
        
        # Create comparison image
        comparison = np.hstack([gray_frame, bg_display])
        comparison_path = output_dir / f"frame_{frame_idx:04d}_comparison.png"
        cv2.imwrite(str(comparison_path), comparison)
        
        # Add text labels
        labeled = comparison.copy()
        if len(labeled.shape) == 2:
            labeled = cv2.cvtColor(labeled, cv2.COLOR_GRAY2BGR)
        
        cv2.putText(labeled, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(labeled, "Background Subtracted", (gray_frame.shape[1] + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        labeled_path = output_dir / f"frame_{frame_idx:04d}_labeled_comparison.png"
        cv2.imwrite(str(labeled_path), labeled)
        
        print(f"  Frame {frame_idx}:")
        print(f"    - Original: {original_path}")
        print(f"    - BG Subtracted: {bg_path}")
        print(f"    - Comparison: {labeled_path}")
        print(f"    - BG Subtracted stats: min={bg_subtracted.min():.2f}, max={bg_subtracted.max():.2f}, mean={bg_subtracted.mean():.2f}")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - frame_0000_original.png - Original frame 0")
    print(f"  - frame_0000_bg_subtracted.png - Background subtracted frame 0")
    print(f"  - frame_0000_labeled_comparison.png - Side-by-side comparison frame 0")
    print(f"  - frame_0001_original.png - Original frame 1")
    print(f"  - frame_0001_bg_subtracted.png - Background subtracted frame 1")
    print(f"  - frame_0001_labeled_comparison.png - Side-by-side comparison frame 1")


if __name__ == "__main__":
    main()

