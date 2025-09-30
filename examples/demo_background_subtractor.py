#!/usr/bin/env python3
"""
Demo script for the BackgroundSubtractor module.

This script demonstrates how to use the BackgroundSubtractor class
independently for background subtraction tasks.
"""

import os
import sys
from pathlib import Path

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from processing.background_subtractor import BackgroundSubtractor


def demo_background_subtractor():
    """Demonstrate the BackgroundSubtractor functionality."""
    
    print("=" * 60)
    print("BACKGROUND SUBTRACTOR DEMO")
    print("=" * 60)
    
    # Configuration
    video_path = "/Users/jasonli/FishDetection/data/input/videos/Clutch1_20250804_122715.mp4"
    output_dir = "/Users/jasonli/FishDetection/data/output/background_subtractor_demo"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    
    # Initialize background subtractor with custom parameters
    print("\nInitializing background subtractor...")
    bg_subtractor = BackgroundSubtractor(
        threshold=30,  # Higher threshold for more selective detection
        blur_kernel_size=(5, 5),  # Larger blur kernel
        blur_sigma=1.0  # Some Gaussian blur
    )
    
    print(f"Background subtractor: {bg_subtractor}")
    
    # Create background model
    print("\nCreating background model...")
    try:
        background_model = bg_subtractor.create_background_model(video_path)
        print(f"✓ Background model created successfully")
        print(f"  Shape: {background_model.shape}")
        print(f"  Data type: {background_model.dtype}")
        
        # Save background model for later use
        bg_model_path = os.path.join(output_dir, "background_model.npy")
        bg_subtractor.save_background_model(bg_model_path)
        print(f"  Saved to: {bg_model_path}")
        
    except Exception as e:
        print(f"✗ Failed to create background model: {e}")
        return
    
    # Process frames and save results
    print("\nProcessing frames...")
    try:
        # Process first 10 frames
        bg_subtracted_frames = bg_subtractor.process_video_frames(
            video_path, 
            num_frames=10, 
            start_frame=0
        )
        
        print(f"✓ Processed {len(bg_subtracted_frames)} frames successfully")
        
        # Save individual results
        for i, frame in enumerate(bg_subtracted_frames):
            output_path = os.path.join(output_dir, f"frame_{i:03d}_bg_subtracted.png")
            import cv2
            cv2.imwrite(output_path, frame)
            print(f"  Saved: frame_{i:03d}_bg_subtracted.png")
        
    except Exception as e:
        print(f"✗ Frame processing failed: {e}")
        return
    
    # Demonstrate loading a saved background model
    print("\nDemonstrating background model loading...")
    try:
        # Create new instance
        bg_subtractor2 = BackgroundSubtractor(threshold=20)
        
        # Load the saved model
        bg_subtractor2.load_background_model(bg_model_path)
        
        print("✓ Background model loaded successfully")
        print(f"  New instance: {bg_subtractor2}")
        
        # Process one frame with the loaded model
        import pims
        video = pims.PyAVReaderIndexed(video_path)
        frame = video[0][:, :, 0]
        bg_subtracted = bg_subtractor2.apply_background_subtraction(frame)
        
        # Save result
        import cv2
        cv2.imwrite(os.path.join(output_dir, "loaded_model_test.png"), bg_subtracted)
        print("  Saved: loaded_model_test.png")
        
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
    
    print(f"\nDemo complete! Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    demo_background_subtractor()
