"""
Specialized fish tracker using background subtraction and YOLOv8.

This module provides a fish-specific tracking approach that combines
background subtraction for detection with YOLOv8 for validation.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

from ..common.base_masker import BaseMasker


class FishTracker(BaseMasker):
    """
    Specialized fish tracker that combines background subtraction with YOLOv8.
    
    This tracker is specifically designed for detecting small fish in underwater videos
    where traditional object detection methods may fail.
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.1,
                 min_fish_area: int = 20,
                 max_fish_area: int = 2000,
                 bg_learning_rate: float = 0.001,
                 morph_kernel_size: int = 3):
        """
        Initialize fish tracker.
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for YOLO detections
            min_fish_area: Minimum area for fish candidates
            max_fish_area: Maximum area for fish candidates
            bg_learning_rate: Background subtractor learning rate
            morph_kernel_size: Morphological operations kernel size
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.min_fish_area = min_fish_area
        self.max_fish_area = max_fish_area
        self.bg_learning_rate = bg_learning_rate
        self.morph_kernel_size = morph_kernel_size
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        
        # Morphological kernel for noise reduction
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        # Tracking state
        self.track_history = {}
        self.frame_count = 0
        self.fish_detections = []
        
    def detect_fish_candidates(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect fish candidates using background subtraction.
        
        Args:
            frame: Input frame
            
        Returns:
            List of (x, y, w, h, area) tuples for fish candidates
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.bg_learning_rate)
        
        # Apply morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        fish_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_fish_area < area < self.max_fish_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (fish are typically longer than wide)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio for fish
                    fish_candidates.append((x, y, w, h, area))
        
        return fish_candidates
    
    def validate_with_yolo(self, frame: np.ndarray, fish_candidates: List[Tuple[int, int, int, int, float]]) -> List[Tuple[int, int, int, int, float, float]]:
        """
        Validate fish candidates using YOLOv8.
        
        Args:
            frame: Input frame
            fish_candidates: List of fish candidates from background subtraction
            
        Returns:
            List of validated fish detections with confidence scores
        """
        if not fish_candidates:
            return []
        
        # Run YOLO detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        validated_fish = []
        
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                # Match YOLO detections with fish candidates
                for box, conf, cls in zip(boxes, confidences, classes):
                    x1, y1, x2, y2 = box
                    yolo_area = (x2 - x1) * (y2 - y1)
                    
                    # Find best matching fish candidate
                    best_match = None
                    best_overlap = 0
                    
                    for i, (fx, fy, fw, fh, farea) in enumerate(fish_candidates):
                        # Calculate overlap
                        overlap_x1 = max(x1, fx)
                        overlap_y1 = max(y1, fy)
                        overlap_x2 = min(x2, fx + fw)
                        overlap_y2 = min(y2, fy + fh)
                        
                        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                            overlap_ratio = overlap_area / min(yolo_area, farea)
                            
                            if overlap_ratio > best_overlap and overlap_ratio > 0.1:  # 10% overlap threshold
                                best_overlap = overlap_ratio
                                best_match = (fx, fy, fw, fh, farea, conf)
                    
                    if best_match:
                        validated_fish.append(best_match)
        
        # If no YOLO matches, use all fish candidates with low confidence
        if not validated_fish and fish_candidates:
            for fx, fy, fw, fh, farea in fish_candidates:
                validated_fish.append((fx, fy, fw, fh, farea, 0.1))  # Low confidence for unvalidated
        
        return validated_fish
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for fish detection and tracking.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Frame with fish tracking annotations
        """
        if frame is None:
            return frame
        
        # Detect fish candidates using background subtraction
        fish_candidates = self.detect_fish_candidates(frame)
        
        # Validate with YOLO
        validated_fish = self.validate_with_yolo(frame, fish_candidates)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw fish detections
        for i, (x, y, w, h, area, conf) in enumerate(validated_fish):
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence score
            cv2.putText(annotated_frame, f"Fish {i+1}: {conf:.2f}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Update track history
            center_x = x + w // 2
            center_y = y + h // 2
            
            if i not in self.track_history:
                self.track_history[i] = []
            
            self.track_history[i].append((center_x, center_y))
            
            # Limit track history
            if len(self.track_history[i]) > 30:
                self.track_history[i].pop(0)
            
            # Draw track trail
            if len(self.track_history[i]) > 1:
                points = np.array(self.track_history[i], dtype=np.int32)
                cv2.polylines(annotated_frame, [points], False, (0, 255, 0), 1)
        
        # Store detection data
        self.fish_detections.append({
            'frame': self.frame_count,
            'fish_count': len(validated_fish),
            'detections': validated_fish
        })
        
        self.frame_count += 1
        return annotated_frame
    
    def process_video(self, video_path: str, output_path: str, max_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Process video for fish tracking.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
            max_duration: Maximum duration to process in seconds
            
        Returns:
            Dictionary with processing statistics
        """
        video_path = Path(video_path)
        output_path = Path(output_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate max frames if duration is specified
        max_frames = total_frames
        if max_duration is not None:
            max_frames = min(total_frames, int(max_duration * fps))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Processing statistics
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_fish_detected': 0,
            'avg_fish_per_frame': 0,
            'max_fish_in_frame': 0,
            'processing_time': 0,
            'fps': 0
        }
        
        start_time = time.time()
        frame_count = 0
        
        print(f"Processing video: {video_path.name}")
        print(f"Max frames to process: {max_frames}")
        
        try:
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Write frame
                out.write(processed_frame)
                
                frame_count += 1
                stats['total_frames'] = frame_count
                
                # Update statistics
                if self.fish_detections:
                    last_detection = self.fish_detections[-1]
                    stats['total_fish_detected'] += last_detection['fish_count']
                    stats['max_fish_in_frame'] = max(stats['max_fish_in_frame'], last_detection['fish_count'])
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    avg_fish = stats['total_fish_detected'] / frame_count if frame_count > 0 else 0
                    print(f"Processed {frame_count}/{max_frames} frames ({current_fps:.1f} FPS, {avg_fish:.1f} fish/frame)")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            
            # Calculate final statistics
            end_time = time.time()
            stats['processing_time'] = end_time - start_time
            stats['fps'] = frame_count / stats['processing_time'] if stats['processing_time'] > 0 else 0
            stats['avg_fish_per_frame'] = stats['total_fish_detected'] / frame_count if frame_count > 0 else 0
            
            print(f"\nProcessing complete!")
            print(f"Total frames processed: {frame_count}")
            print(f"Total fish detected: {stats['total_fish_detected']}")
            print(f"Average fish per frame: {stats['avg_fish_per_frame']:.1f}")
            print(f"Max fish in single frame: {stats['max_fish_in_frame']}")
            print(f"Processing time: {stats['processing_time']:.2f} seconds")
            print(f"Average FPS: {stats['fps']:.2f}")
            print(f"Output saved to: {output_path}")
        
        return stats
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """
        Get summary of fish detections.
        
        Returns:
            Dictionary with detection summary
        """
        if not self.fish_detections:
            return {'total_frames': 0, 'avg_fish_per_frame': 0, 'total_fish': 0}
        
        total_fish = sum(d['fish_count'] for d in self.fish_detections)
        avg_fish = total_fish / len(self.fish_detections)
        
        return {
            'total_frames': len(self.fish_detections),
            'total_fish': total_fish,
            'avg_fish_per_frame': avg_fish,
            'max_fish_in_frame': max(d['fish_count'] for d in self.fish_detections),
            'min_fish_in_frame': min(d['fish_count'] for d in self.fish_detections)
        }
    
    def save_detection_data(self, output_path: str):
        """
        Save detection data to file.
        
        Args:
            output_path: Path to save detection data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        detection_data = {
            'detections': self.fish_detections,
            'summary': self.get_detection_summary(),
            'parameters': {
                'min_fish_area': self.min_fish_area,
                'max_fish_area': self.max_fish_area,
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        # Save as numpy file
        np.savez(str(output_path), **detection_data)
        print(f"Detection data saved to: {output_path}")


def test_fish_tracker(video_path: str, output_dir: str = "output/fish_tracker_test", max_duration: float = 10.0):
    """
    Test function for fish tracker.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for results
        max_duration: Maximum duration to process in seconds
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize fish tracker
    tracker = FishTracker(
        model_path="yolov8n.pt",
        confidence_threshold=0.1,
        min_fish_area=20,
        max_fish_area=2000
    )
    
    # Process video
    output_video = output_dir / "fish_tracking_result.mp4"
    stats = tracker.process_video(
        video_path=video_path,
        output_path=str(output_video),
        max_duration=max_duration
    )
    
    # Save detection data
    detection_data_path = output_dir / "fish_detection_data.npz"
    tracker.save_detection_data(str(detection_data_path))
    
    # Print summary
    summary = tracker.get_detection_summary()
    print(f"\nFish Detection Summary:")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Total fish detected: {summary['total_fish']}")
    print(f"  Average fish per frame: {summary['avg_fish_per_frame']:.1f}")
    print(f"  Max fish in frame: {summary['max_fish_in_frame']}")
    print(f"  Min fish in frame: {summary['min_fish_in_frame']}")
    
    return stats, summary


if __name__ == "__main__":
    # Test with the fish video
    video_path = "/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4"
    test_fish_tracker(video_path, max_duration=10.0)
