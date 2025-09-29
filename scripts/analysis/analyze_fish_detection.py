#!/usr/bin/env python3
"""
Analyze fish detection issues and create a specialized approach.

This script analyzes why YOLOv8 isn't detecting the expected 28 fish per frame
and creates a specialized detection approach.
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tracking_methods.yolo_tracking.yolo_tracker import YOLOTracker


def analyze_video_frames(video_path: str, num_frames: int = 10):
    """
    Analyze video frames to understand the fish characteristics.
    """
    print(f"\n{'='*60}")
    print("ANALYZING VIDEO FRAMES FOR FISH CHARACTERISTICS")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    
    # Analyze frames
    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\nFrame {frame_count}:")
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Basic statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        print(f"  Mean brightness: {mean_brightness:.1f}")
        print(f"  Std brightness: {std_brightness:.1f}")
        
        # Look for small moving objects (potential fish)
        if frame_count > 0:
            # Calculate frame difference
            frame_diff = cv2.absdiff(prev_gray, gray)
            diff_threshold = np.mean(frame_diff)
            print(f"  Motion level: {diff_threshold:.1f}")
            
            # Find contours in frame difference
            _, thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter small contours (potential fish)
            small_contours = [c for c in contours if 20 < cv2.contourArea(c) < 500]
            print(f"  Small moving objects: {len(small_contours)}")
            
            # Save frame with contours for inspection
            if len(small_contours) > 10:  # If we find many small objects
                debug_frame = frame.copy()
                cv2.drawContours(debug_frame, small_contours, -1, (0, 255, 0), 1)
                debug_path = f"output/fish_analysis/frame_{frame_count:03d}_contours.png"
                os.makedirs(os.path.dirname(debug_path), exist_ok=True)
                cv2.imwrite(debug_path, debug_frame)
                print(f"  Saved debug frame: {debug_path}")
        
        prev_gray = gray.copy()
        frame_count += 1
    
    cap.release()


def test_yolo_with_small_objects(video_path: str, output_dir: str = "output/fish_analysis"):
    """
    Test YOLOv8 specifically for small object detection.
    """
    print(f"\n{'='*60}")
    print("TESTING YOLO FOR SMALL OBJECT DETECTION")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with very aggressive parameters for small objects
    tracker = YOLOTracker(
        model_path="yolov8n.pt",
        confidence_threshold=0.01,  # Very low confidence
        iou_threshold=0.1,          # Very low IoU to avoid merging small objects
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    max_frames = 5
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\nAnalyzing frame {frame_count}:")
        
        # Run YOLO detection
        results = tracker.model(frame, conf=0.01, iou=0.1, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                print(f"  Detected {len(boxes)} objects:")
                
                # Filter for small objects (potential fish)
                small_objects = []
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Consider objects smaller than 1000 pixels as potential fish
                    if area < 1000:
                        small_objects.append((box, conf, cls, area))
                        print(f"    Small object {i+1}: class={int(cls)}, conf={conf:.3f}, size={width:.1f}x{height:.1f}, area={area:.1f}")
                
                print(f"  Small objects (potential fish): {len(small_objects)}")
                
                # Create visualization
                if len(small_objects) > 0:
                    vis_frame = frame.copy()
                    for box, conf, cls, area in small_objects:
                        x1, y1, x2, y2 = box.astype(int)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis_frame, f"{conf:.2f}", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    vis_path = output_dir / f"frame_{frame_count:03d}_small_objects.png"
                    cv2.imwrite(str(vis_path), vis_frame)
                    print(f"  Saved visualization: {vis_path}")
            else:
                print("  No objects detected")
        else:
            print("  No detection results")
        
        frame_count += 1
    
    cap.release()


def create_fish_detection_approach(video_path: str, output_dir: str = "output/fish_analysis"):
    """
    Create a specialized approach for fish detection.
    """
    print(f"\n{'='*60}")
    print("CREATING SPECIALIZED FISH DETECTION APPROACH")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # This approach combines:
    # 1. Background subtraction to find moving objects
    # 2. YOLOv8 for object classification
    # 3. Size filtering for fish-like objects
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    # Initialize YOLO
    tracker = YOLOTracker(
        model_path="yolov8n.pt",
        confidence_threshold=0.1,
        iou_threshold=0.3,
    )
    
    frame_count = 0
    max_frames = 10
    fish_detections = []
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        print(f"\nProcessing frame {frame_count}:")
        
        # Step 1: Background subtraction
        fg_mask = bg_subtractor.apply(frame)
        
        # Step 2: Find contours in foreground
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 3: Filter contours by size (fish-like)
        fish_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 2000:  # Fish-like size range
                x, y, w, h = cv2.boundingRect(contour)
                fish_candidates.append((x, y, w, h, area))
        
        print(f"  Background subtraction found {len(fish_candidates)} fish candidates")
        
        # Step 4: Use YOLO to classify candidates
        results = None
        boxes = []
        confidences = []
        classes = []
        
        if len(fish_candidates) > 0:
            # Run YOLO on the entire frame
            results = tracker.model(frame, conf=0.1, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    # Match YOLO detections with fish candidates
                    matched_detections = 0
                    for box, conf, cls in zip(boxes, confidences, classes):
                        x1, y1, x2, y2 = box
                        yolo_area = (x2 - x1) * (y2 - y1)
                        
                        # Check if this YOLO detection overlaps with any fish candidate
                        for fx, fy, fw, fh, farea in fish_candidates:
                            # Calculate overlap
                            overlap_x1 = max(x1, fx)
                            overlap_y1 = max(y1, fy)
                            overlap_x2 = min(x2, fx + fw)
                            overlap_y2 = min(y2, fy + fh)
                            
                            if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                                overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                                if overlap_area > 0.1 * min(yolo_area, farea):  # 10% overlap threshold
                                    matched_detections += 1
                                    fish_detections.append((frame_count, box, conf, cls))
                                    break
                    
                    print(f"  YOLO found {len(boxes)} objects, {matched_detections} matched with fish candidates")
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Draw fish candidates from background subtraction
        for fx, fy, fw, fh, farea in fish_candidates:
            cv2.rectangle(vis_frame, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 1)  # Blue for candidates
        
        # Draw YOLO detections
        if results and len(results) > 0 and len(boxes) > 0:
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for YOLO
                cv2.putText(vis_frame, f"{conf:.2f}", (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save visualization
        vis_path = output_dir / f"frame_{frame_count:03d}_combined.png"
        cv2.imwrite(str(vis_path), vis_frame)
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nTotal fish detections across {max_frames} frames: {len(fish_detections)}")
    print(f"Average per frame: {len(fish_detections)/max_frames:.1f}")
    
    return fish_detections


def main():
    """Main analysis function."""
    video_path = "/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4"
    output_dir = "/Users/jasonli/FishDetection/output/fish_analysis"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
    
    print("Starting comprehensive fish detection analysis...")
    print(f"Video: {video_path}")
    print(f"Expected: 28 fish per frame")
    print("-" * 60)
    
    try:
        # Analyze video characteristics
        analyze_video_frames(video_path, num_frames=5)
        
        # Test YOLO for small objects
        test_yolo_with_small_objects(video_path, output_dir)
        
        # Create specialized approach
        fish_detections = create_fish_detection_approach(video_path, output_dir)
        
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print("The issue with detecting 28 fish per frame is likely due to:")
        print("1. Fish are very small in the video")
        print("2. YOLOv8 is trained on general objects, not fish")
        print("3. Fish may be moving too fast or have low contrast")
        print("4. The general YOLOv8 model may not recognize fish as distinct objects")
        print("\nRecommendations:")
        print("1. Train a custom YOLOv8 model on fish data")
        print("2. Use background subtraction + YOLO combination")
        print("3. Preprocess video to enhance fish visibility")
        print("4. Use a different detection approach (e.g., template matching)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
