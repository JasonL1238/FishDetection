"""
YOLOv8-based object tracking implementation.

This module provides fish detection and tracking using YOLOv8
with built-in tracking capabilities.
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


class YOLOTracker(BaseMasker):
    """
    YOLOv8-based object tracker for fish detection and tracking.
    
    This class uses YOLOv8 for object detection and tracking,
    specifically optimized for fish detection in underwater videos.
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.7,
                 max_track_length: int = 30,
                 track_buffer: int = 30):
        """
        Initialize YOLOv8 tracker.
        
        Args:
            model_path: Path to YOLOv8 model file or model name
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            max_track_length: Maximum length of track history
            track_buffer: Buffer for track management
        """
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_track_length = max_track_length
        self.track_buffer = track_buffer
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Tracking state
        self.track_history = {}
        self.frame_count = 0
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for fish detection and tracking.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Frame with tracking annotations
        """
        if frame is None:
            return frame
            
        # Run YOLOv8 tracking on the frame
        results = self.model.track(
            frame, 
            persist=True,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Get the first result (single frame)
        if results and len(results) > 0:
            result = results[0]
            
            # Draw tracking results
            annotated_frame = result.plot()
            
            # Update track history
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    # Calculate center point
                    x1, y1, x2, y2 = box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Update track history
                    if track_id not in self.track_history:
                        self.track_history[track_id] = []
                    
                    self.track_history[track_id].append((center_x, center_y))
                    
                    # Limit track history length
                    if len(self.track_history[track_id]) > self.max_track_length:
                        self.track_history[track_id].pop(0)
                    
                    # Draw track trail
                    if len(self.track_history[track_id]) > 1:
                        points = np.array(self.track_history[track_id], dtype=np.int32)
                        cv2.polylines(annotated_frame, [points], False, (0, 255, 0), 2)
            
            return annotated_frame
        else:
            return frame
    
    def process_video(self, video_path: str, output_path: str, max_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Process video for fish tracking.
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
            max_duration: Maximum duration to process in seconds (None for full video)
            
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
            'detection_count': 0,
            'unique_tracks': 0,
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
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {frame_count}/{max_frames} frames ({current_fps:.1f} FPS)")
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            
            # Calculate final statistics
            end_time = time.time()
            stats['processing_time'] = end_time - start_time
            stats['fps'] = frame_count / stats['processing_time'] if stats['processing_time'] > 0 else 0
            stats['unique_tracks'] = len(self.track_history)
            
            print(f"\nProcessing complete!")
            print(f"Total frames processed: {frame_count}")
            print(f"Processing time: {stats['processing_time']:.2f} seconds")
            print(f"Average FPS: {stats['fps']:.2f}")
            print(f"Unique tracks detected: {stats['unique_tracks']}")
            print(f"Output saved to: {output_path}")
        
        return stats
    
    def get_track_data(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Get current track data.
        
        Returns:
            Dictionary mapping track IDs to lists of (x, y) coordinates
        """
        return self.track_history.copy()
    
    def reset_tracking(self):
        """Reset tracking state."""
        self.track_history.clear()
        self.frame_count = 0
    
    def save_track_data(self, output_path: str):
        """
        Save track data to file.
        
        Args:
            output_path: Path to save track data
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        track_data = {}
        for track_id, points in self.track_history.items():
            track_data[str(track_id)] = points
        
        # Save as numpy file
        np.savez(str(output_path), **track_data)
        print(f"Track data saved to: {output_path}")


def test_yolo_tracking(video_path: str, output_dir: str = "output/yolo_test", max_duration: float = 10.0):
    """
    Test function for YOLOv8 tracking.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for results
        max_duration: Maximum duration to process in seconds
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tracker
    tracker = YOLOTracker(
        model_path="yolov8n.pt",  # Use nano model for speed
        confidence_threshold=0.5,
        iou_threshold=0.7
    )
    
    # Process video
    output_video = output_dir / "yolo_tracking_result.mp4"
    stats = tracker.process_video(
        video_path=video_path,
        output_path=str(output_video),
        max_duration=max_duration
    )
    
    # Save track data
    track_data_path = output_dir / "track_data.npz"
    tracker.save_track_data(str(track_data_path))
    
    return stats


if __name__ == "__main__":
    # Test with the fish video
    video_path = "/Users/jasonli/FishDetection/input/Clutch1_20250804_122715.mp4"
    test_yolo_tracking(video_path, max_duration=10.0)
