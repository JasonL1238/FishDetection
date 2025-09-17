"""Motion-based fish detection for tiny fish in challenging environments."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class FishTrack:
    """Represents a tracked fish with position history and properties."""
    id: int
    x: int
    y: int
    width: int
    height: int
    area: int
    confidence: float
    age: int
    last_seen: int
    velocity: Tuple[float, float]
    position_history: List[Tuple[int, int]]


class BarMaskDetector:
    """Detects and creates masks for vertical bars/occluders in the tank."""
    
    def __init__(self, bar_width_threshold: int = 20, min_bar_length: int = 100):
        """
        Initialize bar mask detector.
        
        Args:
            bar_width_threshold: Maximum width to consider as a bar (pixels)
            min_bar_length: Minimum length to consider as a bar (pixels)
        """
        self.bar_width_threshold = bar_width_threshold
        self.min_bar_length = min_bar_length
        self.static_mask = None
        self.bar_regions = []
    
    def detect_bars(self, image: np.ndarray) -> np.ndarray:
        """Detect vertical bars and create a mask."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Use edge detection to find vertical lines
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                               minLineLength=self.min_bar_length, maxLineGap=10)
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        bar_regions = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Check if line is roughly vertical
                if abs(x2 - x1) < self.bar_width_threshold:
                    # Create a thicker line for the mask
                    thickness = max(3, abs(x2 - x1) + 10)
                    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
                    
                    # Store bar region for tracking
                    bar_regions.append({
                        'x': min(x1, x2),
                        'y': min(y1, y2),
                        'width': abs(x2 - x1) + thickness,
                        'height': abs(y2 - y1)
                    })
        
        self.bar_regions = bar_regions
        self.static_mask = mask
        return mask
    
    def update_mask(self, image: np.ndarray) -> np.ndarray:
        """Update bar mask, using static mask if available."""
        if self.static_mask is not None:
            return self.static_mask
        return self.detect_bars(image)


class MotionFishDetector:
    """Motion-based fish detector optimized for tiny fish in challenging environments."""
    
    def __init__(self, 
                 var_threshold: int = 16,
                 history: int = 500,
                 detect_shadows: bool = False,
                 min_area: int = 20,
                 max_area: int = 800,
                 min_aspect_ratio: float = 2.2,
                 max_aspect_ratio: float = 8.0,
                 tracking_threshold: float = 0.3):
        """
        Initialize motion-based fish detector.
        
        Args:
            var_threshold: MOG2 variance threshold for background subtraction
            history: Number of frames for background model
            detect_shadows: Whether to detect shadows
            min_area: Minimum area for fish detection
            max_area: Maximum area for fish detection
            min_aspect_ratio: Minimum aspect ratio for fish (long and thin)
            max_aspect_ratio: Maximum aspect ratio for fish
            tracking_threshold: IoU threshold for tracking
        """
        self.var_threshold = var_threshold
        self.history = history
        self.detect_shadows = detect_shadows
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.tracking_threshold = tracking_threshold
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
            history=history
        )
        
        # Initialize bar mask detector
        self.bar_detector = BarMaskDetector()
        
        # Tracking variables
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
        
        # Morphological kernels
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Preprocessing parameters
        self.blur_kernel = (5, 5)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better detection."""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply CLAHE for better contrast
        enhanced = self.clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, self.blur_kernel, 0)
        
        return blurred
    
    def detect_motion_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect moving objects using background subtraction."""
        # Preprocess frame
        processed = self.preprocess_frame(frame)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(processed)
        
        # Get bar mask and apply it
        bar_mask = self.bar_detector.update_mask(frame)
        fg_mask[bar_mask == 255] = 0  # Ignore areas with bars
        
        # Morphological operations to clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel_close, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = max(w, h) / max(1, min(w, h))
            
            # Filter by aspect ratio (fish are long and thin)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Calculate confidence based on shape and area
            confidence = self._calculate_confidence(contour, area, aspect_ratio)
            
            detections.append((x, y, w, h, confidence))
        
        return detections
    
    def _calculate_confidence(self, contour: np.ndarray, area: int, aspect_ratio: float) -> float:
        """Calculate confidence score for a detection."""
        # Calculate contour properties
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        
        # Circularity (4π*area/perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Fish should be somewhat elongated but not too irregular
        circularity_score = max(0, 1.0 - abs(circularity - 0.3) / 0.3)
        
        # Aspect ratio score (prefer elongated shapes)
        aspect_score = min(1.0, aspect_ratio / 3.0)
        
        # Area score (prefer medium-sized objects)
        ideal_area = (self.min_area + self.max_area) / 2
        area_score = 1.0 - abs(area - ideal_area) / ideal_area
        
        # Combine scores
        confidence = (circularity_score * 0.3 + aspect_score * 0.4 + area_score * 0.3)
        return min(max(confidence, 0.0), 1.0)
    
    def update_tracks(self, detections: List[Tuple[int, int, int, int, float]]) -> List[FishTrack]:
        """Update fish tracks using simple Hungarian algorithm approach."""
        self.frame_count += 1
        
        # Convert detections to current format
        current_detections = [(x, y, w, h, conf) for x, y, w, h, conf in detections]
        
        # Update existing tracks
        updated_tracks = []
        used_detections = set()
        
        for track_id, track in self.tracks.items():
            if track.last_seen < self.frame_count - 10:  # Remove old tracks
                continue
            
            # Find best matching detection
            best_match = None
            best_iou = 0
            best_idx = -1
            
            for i, (x, y, w, h, conf) in enumerate(current_detections):
                if i in used_detections:
                    continue
                
                iou = self._calculate_iou(
                    (track.x, track.y, track.width, track.height),
                    (x, y, w, h)
                )
                
                if iou > self.tracking_threshold and iou > best_iou:
                    best_iou = iou
                    best_match = (x, y, w, h, conf)
                    best_idx = i
            
            if best_match is not None:
                # Update existing track
                x, y, w, h, conf = best_match
                area = w * h
                
                # Calculate velocity
                velocity = (
                    (x - track.x) / max(1, self.frame_count - track.last_seen),
                    (y - track.y) / max(1, self.frame_count - track.last_seen)
                )
                
                # Update track
                track.x = x
                track.y = y
                track.width = w
                track.height = h
                track.area = area
                track.confidence = conf
                track.age += 1
                track.last_seen = self.frame_count
                track.velocity = velocity
                track.position_history.append((x, y))
                
                # Keep only recent history
                if len(track.position_history) > 10:
                    track.position_history.pop(0)
                
                updated_tracks.append(track)
                used_detections.add(best_idx)
            else:
                # Track not seen, but keep it for a few frames
                track.last_seen = self.frame_count
                if track.last_seen < self.frame_count - 5:
                    updated_tracks.append(track)
        
        # Create new tracks for unmatched detections
        for i, (x, y, w, h, conf) in enumerate(current_detections):
            if i not in used_detections:
                area = w * h
                new_track = FishTrack(
                    id=self.next_id,
                    x=x, y=y, width=w, height=h,
                    area=area, confidence=conf,
                    age=1, last_seen=self.frame_count,
                    velocity=(0, 0),
                    position_history=[(x, y)]
                )
                updated_tracks.append(new_track)
                self.tracks[self.next_id] = new_track
                self.next_id += 1
        
        # Update tracks dictionary
        self.tracks = {track.id: track for track in updated_tracks}
        
        return updated_tracks
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_frame(self, frame: np.ndarray) -> List[FishTrack]:
        """Process a single frame and return active fish tracks."""
        # Detect moving objects
        detections = self.detect_motion_objects(frame)
        
        # Update tracks
        active_tracks = self.update_tracks(detections)
        
        return active_tracks
    
    def visualize_tracks(self, frame: np.ndarray, tracks: List[FishTrack]) -> np.ndarray:
        """Visualize fish tracks on the frame."""
        result = frame.copy()
        
        for track in tracks:
            # Draw bounding box
            color = (0, 255, 0) if track.age > 3 else (0, 255, 255)  # Green for mature, yellow for new
            cv2.rectangle(result, 
                         (track.x, track.y), 
                         (track.x + track.width, track.y + track.height),
                         color, 2)
            
            # Draw track ID and confidence
            label = f"ID:{track.id} ({track.confidence:.2f})"
            cv2.putText(result, label, 
                       (track.x, track.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw velocity vector
            if track.velocity[0] != 0 or track.velocity[1] != 0:
                end_x = int(track.x + track.velocity[0] * 10)
                end_y = int(track.y + track.velocity[1] * 10)
                cv2.arrowedLine(result, 
                               (track.x + track.width//2, track.y + track.height//2),
                               (end_x, end_y), color, 2)
        
        return result
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[Dict]:
        """Process a video file and return detection results."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                tracks = self.process_frame(frame)
                
                # Store results
                frame_results = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'tracks': [
                        {
                            'id': track.id,
                            'x': track.x, 'y': track.y,
                            'width': track.width, 'height': track.height,
                            'area': track.area, 'confidence': track.confidence,
                            'age': track.age, 'velocity': track.velocity
                        }
                        for track in tracks
                    ]
                }
                results.append(frame_results)
                
                # Visualize and write frame
                if writer:
                    vis_frame = self.visualize_tracks(frame, tracks)
                    writer.write(vis_frame)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames, {len(tracks)} active tracks")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        return results
