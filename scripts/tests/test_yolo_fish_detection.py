#!/usr/bin/env python3
"""
Enhanced test script for YOLOv8 fish detection with optimized parameters.

This script tests different configurations to improve fish detection
when we expect 28 fish per frame.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tracking_methods.yolo_tracking.yolo_tracker import YOLOTracker


def test_detection_parameters(video_path: str, output_dir: str = "data/output/yolo_tracking"):
    """
    Test different detection parameters to improve fish detection.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test configurations
    configs = [
        {
            'name': 'low_confidence',
            'confidence_threshold': 0.1,
            'iou_threshold': 0.5,
            'model': 'yolov8n.pt'
        },
        {
            'name': 'very_low_confidence',
            'confidence_threshold': 0.05,
            'iou_threshold': 0.3,
            'model': 'yolov8n.pt'
        },
        {
            'name': 'small_model_low_conf',
            'confidence_threshold': 0.1,
            'iou_threshold': 0.4,
            'model': 'yolov8s.pt'
        },
        {
            'name': 'medium_model_low_conf',
            'confidence_threshold': 0.15,
            'iou_threshold': 0.5,
            'model': 'yolov8m.pt'
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing configuration: {config['name']}")
        print(f"Model: {config['model']}")
        print(f"Confidence: {config['confidence_threshold']}")
        print(f"IoU: {config['iou_threshold']}")
        print(f"{'='*60}")
        
        try:
            # Initialize tracker with current config
            tracker = YOLOTracker(
                model_path=config['model'],
                confidence_threshold=config['confidence_threshold'],
                iou_threshold=config['iou_threshold']
            )
            
            # Process video
            output_video = output_dir / f"yolo_{config['name']}_result.mp4"
            stats = tracker.process_video(
                video_path=video_path,
                output_path=str(output_video),
                max_duration=10.0
            )
            
            # Get track data
            track_data = tracker.get_track_data()
            unique_tracks = len(track_data)
            
            results[config['name']] = {
                'stats': stats,
                'unique_tracks': unique_tracks,
                'track_data': track_data,
                'output_video': str(output_video)
            }
            
            print(f"Results for {config['name']}:")
            print(f"  Unique tracks: {unique_tracks}")
            print(f"  Processing time: {stats['processing_time']:.2f}s")
            print(f"  FPS: {stats['fps']:.2f}")
            
            # Save track data
            track_data_path = output_dir / f"track_data_{config['name']}.npz"
            tracker.save_track_data(str(track_data_path))
            
        except Exception as e:
            print(f"Error with config {config['name']}: {e}")
            results[config['name']] = {'error': str(e)}
    
    return results


def analyze_detection_results(video_path: str, output_dir: str = "output/yolo_fish_test"):
    """
    Analyze what objects are being detected in the video.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ANALYZING DETECTION RESULTS")
    print(f"{'='*60}")
    
    # Use very low confidence to see all detections
    tracker = YOLOTracker(
        model_path="yolov8n.pt",
        confidence_threshold=0.01,  # Very low to catch everything
        iou_threshold=0.3
    )
    
    # Process just a few frames to analyze
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    max_frames = 5  # Analyze first 5 frames
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection on this frame
        results = tracker.model(frame, conf=0.01, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            print(f"\nFrame {frame_count}:")
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                print(f"  Detected {len(boxes)} objects:")
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    print(f"    Object {i+1}: class={int(cls)}, conf={conf:.3f}, size={width:.1f}x{height:.1f}")
            else:
                print("  No objects detected")
        
        frame_count += 1
    
    cap.release()


def create_detection_visualization(video_path: str, output_dir: str = "output/yolo_fish_test"):
    """
    Create visualization showing all detections with different confidence levels.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for visualization")
        return
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CREATING DETECTION VISUALIZATION")
    print(f"{'='*60}")
    
    # Test different confidence levels
    confidences = [0.01, 0.1, 0.3, 0.5]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return
    
    # Create visualization for different confidence levels
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLOv8 Detection Results with Different Confidence Thresholds', fontsize=16)
    
    for i, conf in enumerate(confidences):
        row = i // 2
        col = i % 2
        
        # Run detection with this confidence
        tracker = YOLOTracker(
            model_path="yolov8n.pt",
            confidence_threshold=conf,
            iou_threshold=0.5
        )
        
        results = tracker.model(frame, conf=conf, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            annotated_frame = result.plot()
        else:
            annotated_frame = frame.copy()
        
        # Convert to RGB for matplotlib
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        axes[row, col].imshow(annotated_frame_rgb)
        axes[row, col].set_title(f'Confidence: {conf}')
        axes[row, col].axis('off')
        
        # Count detections
        if results and len(results) > 0 and results[0].boxes is not None:
            num_detections = len(results[0].boxes)
            axes[row, col].text(10, 30, f'Detections: {num_detections}', 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                              fontsize=12, color='black')
    
    plt.tight_layout()
    visualization_path = output_dir / "detection_comparison.png"
    plt.savefig(str(visualization_path), dpi=150, bbox_inches='tight')
    plt.show()
    
    cap.release()
    print(f"Detection visualization saved to: {visualization_path}")


def main():
    """Main test function."""
    video_path = "/Users/jasonli/FishDetection/data/input/videos/Clutch1_20250804_122715.mp4"
    output_dir = "/Users/jasonli/FishDetection/data/output/yolo_tracking"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    print("Starting enhanced YOLOv8 fish detection test...")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Expected: 28 fish per frame")
    print("-" * 60)
    
    try:
        # Test different parameters
        results = test_detection_parameters(video_path, output_dir)
        
        # Analyze what's being detected
        analyze_detection_results(video_path, output_dir)
        
        # Create visualization
        try:
            import matplotlib.pyplot as plt
            create_detection_visualization(video_path, output_dir)
        except ImportError:
            print("Matplotlib not available for visualization")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY OF RESULTS")
        print(f"{'='*60}")
        
        for config_name, result in results.items():
            if 'error' in result:
                print(f"{config_name}: ERROR - {result['error']}")
            else:
                tracks = result['unique_tracks']
                print(f"{config_name}: {tracks} unique tracks detected")
        
        print(f"\nNote: We expected 28 fish per frame.")
        print("If detection is still low, consider:")
        print("1. Training a custom YOLOv8 model on fish data")
        print("2. Using a different detection approach")
        print("3. Preprocessing the video (background subtraction, etc.)")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
