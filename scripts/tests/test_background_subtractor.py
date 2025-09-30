#!/usr/bin/env python3
"""
Test script for the new BackgroundSubtractor module.

This script tests the extracted background subtraction functionality
to ensure it works correctly as a standalone module.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from processing.background_subtractor import BackgroundSubtractor


def test_background_subtractor():
    """Test the BackgroundSubtractor class functionality."""
    
    print("=" * 60)
    print("TESTING BACKGROUND SUBTRACTOR MODULE")
    print("=" * 60)
    
    # Configuration
    video_path = "/Users/jasonli/FishDetection/data/input/videos/Clutch1_20250804_122715.mp4"
    output_dir = "/Users/jasonli/FishDetection/data/output/background_subtractor_test"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    
    # Initialize background subtractor
    print("\nInitializing background subtractor...")
    bg_subtractor = BackgroundSubtractor(
        threshold=25,
        blur_kernel_size=(3, 3),
        blur_sigma=0
    )
    
    # Test 1: Create background model
    print("\nTest 1: Creating background model...")
    try:
        background_model = bg_subtractor.create_background_model(video_path)
        print(f"✓ Background model created successfully")
        print(f"  Shape: {background_model.shape}")
        print(f"  Data type: {background_model.dtype}")
        print(f"  Min value: {background_model.min():.2f}")
        print(f"  Max value: {background_model.max():.2f}")
    except Exception as e:
        print(f"✗ Failed to create background model: {e}")
        return
    
    # Test 2: Save and load background model
    print("\nTest 2: Save and load background model...")
    try:
        bg_model_path = os.path.join(output_dir, "background_model.npy")
        bg_subtractor.save_background_model(bg_model_path)
        
        # Create new instance and load model
        bg_subtractor2 = BackgroundSubtractor()
        bg_subtractor2.load_background_model(bg_model_path)
        
        # Compare models
        if np.allclose(background_model, bg_subtractor2.get_background_model()):
            print("✓ Background model save/load successful")
        else:
            print("✗ Background model save/load failed - models don't match")
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")
    
    # Test 3: Process individual frames
    print("\nTest 3: Processing individual frames...")
    try:
        import pims
        video = pims.PyAVReaderIndexed(video_path)
        
        # Process first 5 frames
        for i in range(min(5, len(video))):
            frame = video[i][:, :, 0]  # Grayscale
            bg_subtracted = bg_subtractor.apply_background_subtraction(frame)
            
            # Save results
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}_original.png"), frame)
            cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}_bg_subtracted.png"), bg_subtracted)
            
            print(f"  Frame {i}: processed successfully")
        
        print("✓ Individual frame processing successful")
    except Exception as e:
        print(f"✗ Individual frame processing failed: {e}")
    
    # Test 4: Process multiple frames
    print("\nTest 4: Processing multiple frames...")
    try:
        bg_subtracted_frames = bg_subtractor.process_video_frames(
            video_path, 
            num_frames=10, 
            start_frame=0
        )
        
        print(f"✓ Processed {len(bg_subtracted_frames)} frames successfully")
        
        # Save some results
        for i, frame in enumerate(bg_subtracted_frames[:5]):
            cv2.imwrite(os.path.join(output_dir, f"batch_frame_{i:03d}_bg_subtracted.png"), frame)
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
    
    # Test 5: Test reset functionality
    print("\nTest 5: Testing reset functionality...")
    try:
        bg_subtractor.reset()
        if bg_subtractor.get_background_model() is None:
            print("✓ Reset functionality works correctly")
        else:
            print("✗ Reset functionality failed")
    except Exception as e:
        print(f"✗ Reset test failed: {e}")
    
    print(f"\nTest complete! Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    test_background_subtractor()
