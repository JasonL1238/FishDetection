"""Balanced frame differencing detector optimized for ~20 fish per frame."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class BalancedDetection:
    """Represents a detection from balanced frame differencing."""
    x: int
    y: int
    width: int
    height: int
    area: int
    confidence: float
    motion_magnitude: float
    frame_diff: float
    temporal_consistency: float


class BalancedFrameDifferencingDetector:
    """Balanced frame differencing detector optimized for ~20 fish per frame."""
    
    def __init__(self, 
                 min_area: int = 5,
                 max_area: int = 120,
                 min_aspect_ratio: float = 1.0,
                 max_aspect_ratio: float = 10.0,
                 diff_threshold: int = 15,
                 gaussian_kernel: Tuple[int, int] = (3, 3),
                 expected_fish_count: int = 20,
                 temporal_window: int = 5,
                 confidence_threshold: float = 0.4):
        """
        Initialize balanced frame differencing detector.
        
        Args:
            min_area: Small minimum area for tiny fish
            max_area: Small maximum area
            min_aspect_ratio: Flexible aspect ratio
            max_aspect_ratio: Flexible aspect ratio
            diff_threshold: Low threshold for sensitivity
            gaussian_kernel: Small Gaussian blur kernel
            expected_fish_count: Expected number of fish per frame
            temporal_window: Number of frames to consider for temporal consistency
            confidence_threshold: Minimum confidence threshold for detections
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.diff_threshold = diff_threshold
        self.gaussian_kernel = gaussian_kernel
        self.expected_fish_count = expected_fish_count
        self.temporal_window = temporal_window
        self.confidence_threshold = confidence_threshold
        
        # Frame buffer for differencing
        self.frame_buffer = deque(maxlen=5)  # Keep last 5 frames
        self.background_model = None
        self.background_alpha = 0.03  # Moderate learning rate
        
        # Detection history for temporal consistency
        self.detection_history = deque(maxlen=temporal_window)
        
        # CLAHE for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        
        self.frame_count = 0
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better frame differencing."""
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        
        # Apply CLAHE for better contrast
        enhanced = self.clahe.apply(gray)
        
        # Apply minimal Gaussian blur to reduce noise while preserving small details
        blurred = cv2.GaussianBlur(enhanced, self.gaussian_kernel, 0)
        
        return blurred
    
    def update_background_model(self, frame: np.ndarray) -> np.ndarray:
        """Update background model using running average."""
        if self.background_model is None:
            self.background_model = frame.astype(np.float32)
        else:
            # Running average background model
            cv2.accumulateWeighted(frame, self.background_model, self.background_alpha)
        
        return self.background_model.astype(np.uint8)
    
    def compute_balanced_frame_difference(self, current_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute balanced frame difference using multiple methods."""
        # Add current frame to buffer
        self.frame_buffer.append(current_frame)
        
        if len(self.frame_buffer) < 2:
            return np.zeros_like(current_frame), np.zeros_like(current_frame)
        
        # Method 1: Consecutive frame difference
        prev_frame = self.frame_buffer[-2]
        consecutive_diff = cv2.absdiff(current_frame, prev_frame)
        
        # Method 2: Background subtraction
        background = self.update_background_model(current_frame)
        background_diff = cv2.absdiff(current_frame, background)
        
        # Method 3: Multi-frame difference
        multi_frame_diff = np.zeros_like(current_frame)
        if len(self.frame_buffer) >= 3:
            # Use recent frames for more robust detection
            for i in range(max(0, len(self.frame_buffer) - 3), len(self.frame_buffer) - 1):
                diff = cv2.absdiff(self.frame_buffer[i], self.frame_buffer[-1])
                multi_frame_diff = cv2.bitwise_or(multi_frame_diff, diff)
        
        # Method 4: Temporal gradient
        temporal_gradient = np.zeros_like(current_frame)
        if len(self.frame_buffer) >= 3:
            # Compute gradient across recent frames
            for i in range(max(0, len(self.frame_buffer) - 3), len(self.frame_buffer) - 1):
                grad = cv2.absdiff(self.frame_buffer[i], self.frame_buffer[i + 1])
                temporal_gradient = cv2.bitwise_or(temporal_gradient, grad)
        
        # Combine differences with balanced weights
        combined_diff = cv2.addWeighted(consecutive_diff, 0.4, background_diff, 0.4, 0)
        combined_diff = cv2.addWeighted(combined_diff, 0.7, multi_frame_diff, 0.3, 0)
        combined_diff = cv2.addWeighted(combined_diff, 0.8, temporal_gradient, 0.2, 0)
        
        return combined_diff, consecutive_diff
    
    def detect_motion_regions(self, frame: np.ndarray) -> List[BalancedDetection]:
        """Detect motion regions using balanced frame differencing."""
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Compute frame differences
        combined_diff, consecutive_diff = self.compute_balanced_frame_difference(processed_frame)
        
        # Apply threshold
        _, thresh = cv2.threshold(combined_diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Gentle morphological operations to preserve small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter by aspect ratio
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            
            # Calculate motion magnitude in this region
            roi_diff = consecutive_diff[y:y+h, x:x+w]
            motion_magnitude = np.mean(roi_diff) if roi_diff.size > 0 else 0
            
            # Calculate frame difference strength
            roi_combined = combined_diff[y:y+h, x:x+w]
            frame_diff_strength = np.mean(roi_combined) if roi_combined.size > 0 else 0
            
            # Calculate temporal consistency
            temporal_consistency = self._calculate_temporal_consistency(x, y, w, h)
            
            # Calculate confidence
            confidence = self._calculate_balanced_confidence(contour, area, aspect_ratio, 
                                                           motion_magnitude, frame_diff_strength,
                                                           temporal_consistency)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                continue
            
            detection = BalancedDetection(
                x=x, y=y, width=w, height=h,
                area=int(area), confidence=confidence,
                motion_magnitude=motion_magnitude,
                frame_diff=frame_diff_strength,
                temporal_consistency=temporal_consistency
            )
            detections.append(detection)
        
        # Update detection history
        self.detection_history.append(detections)
        
        # Apply temporal filtering
        detections = self._apply_temporal_filtering(detections)
        
        # Apply non-maximum suppression
        detections = self._non_maximum_suppression(detections)
        
        # Limit to expected fish count if we have too many
        if len(detections) > self.expected_fish_count * 2:
            detections.sort(key=lambda x: x.confidence, reverse=True)
            detections = detections[:self.expected_fish_count * 2]
        
        return detections
    
    def _calculate_temporal_consistency(self, x: int, y: int, w: int, h: int) -> float:
        """Calculate temporal consistency score."""
        if len(self.detection_history) < 2:
            return 0.5  # Neutral score if not enough history
        
        # Check if similar detections appeared in recent frames
        consistency_score = 0.0
        total_frames = 0
        
        for frame_detections in list(self.detection_history)[:-1]:  # Exclude current frame
            for det in frame_detections:
                # Calculate overlap with current detection
                overlap = self._calculate_overlap_boxes(
                    (x, y, w, h), (det.x, det.y, det.width, det.height)
                )
                if overlap > 0.3:  # Significant overlap
                    consistency_score += 1.0
                total_frames += 1
        
        return consistency_score / max(1, total_frames) if total_frames > 0 else 0.0
    
    def _calculate_overlap_boxes(self, box1: Tuple[int, int, int, int], 
                                box2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
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
    
    def _calculate_balanced_confidence(self, contour: np.ndarray, area: int, aspect_ratio: float,
                                     motion_magnitude: float, frame_diff: float,
                                     temporal_consistency: float) -> float:
        """Calculate balanced confidence score for fish detection."""
        # Shape-based confidence
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        
        # Circularity (prefer somewhat oval shapes)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        circularity_score = max(0.0, 1.0 - abs(circularity - 0.6) / 0.6)
        
        # Aspect ratio score (prefer elongated shapes)
        ideal_aspect_ratio = 3.0
        aspect_score = max(0.0, 1.0 - abs(aspect_ratio - ideal_aspect_ratio) / ideal_aspect_ratio)
        
        # Area score (prefer small to medium objects)
        ideal_area = (self.min_area + self.max_area) / 2
        area_score = max(0.0, 1.0 - abs(area - ideal_area) / ideal_area)
        
        # Motion score
        motion_score = min(1.0, motion_magnitude / 25.0)
        
        # Frame difference score
        diff_score = min(1.0, frame_diff / 50.0)
        
        # Temporal consistency score
        temporal_score = temporal_consistency
        
        # Size-based confidence adjustment
        size_factor = 1.0
        if area < 30:  # Very small fish get a small boost
            size_factor = 1.1
        elif area > 80:  # Large fish get a small penalty
            size_factor = 0.9
        
        # Combine scores with balanced weights
        confidence = (circularity_score * 0.15 + 
                     aspect_score * 0.15 + 
                     area_score * 0.15 + 
                     motion_score * 0.25 + 
                     diff_score * 0.25 + 
                     temporal_score * 0.05) * size_factor
        
        return min(max(confidence, 0.0), 1.0)
    
    def _apply_temporal_filtering(self, detections: List[BalancedDetection]) -> List[BalancedDetection]:
        """Apply temporal filtering to improve detection quality."""
        if len(self.detection_history) < 2:
            return detections
        
        # Boost confidence for detections that appear consistently
        for detection in detections:
            if detection.temporal_consistency > 0.4:
                detection.confidence *= 1.1  # Small boost for consistent detections
        
        return detections
    
    def _non_maximum_suppression(self, detections: List[BalancedDetection], 
                                overlap_threshold: float = 0.3) -> List[BalancedDetection]:
        """Remove overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            # Take the detection with highest confidence
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            remaining = []
            for detection in detections:
                if self._calculate_overlap_boxes(
                    (current.x, current.y, current.width, current.height),
                    (detection.x, detection.y, detection.width, detection.height)
                ) < overlap_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return keep
    
    def process_frame(self, frame: np.ndarray) -> List[BalancedDetection]:
        """Process a single frame and return detections."""
        self.frame_count += 1
        return self.detect_motion_regions(frame)
    
    def visualize_detections(self, frame: np.ndarray, detections: List[BalancedDetection]) -> np.ndarray:
        """Visualize detections on the frame."""
        result = frame.copy()
        
        for i, detection in enumerate(detections):
            # Color based on confidence
            if detection.confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif detection.confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Draw bounding box
            cv2.rectangle(result, 
                         (detection.x, detection.y), 
                         (detection.x + detection.width, detection.y + detection.height),
                         color, 2)
            
            # Draw confidence and motion info (only for high confidence detections)
            if detection.confidence > 0.6:
                label = f"{detection.confidence:.2f}"
                cv2.putText(result, label, 
                           (detection.x, detection.y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add frame info
        info_text = f"Frame: {self.frame_count}, Detections: {len(detections)}"
        cv2.putText(result, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    def get_detection_statistics(self) -> Dict:
        """Get statistics about detections."""
        if not self.detection_history:
            return {}
        
        all_detections = []
        for frame_detections in self.detection_history:
            all_detections.extend(frame_detections)
        
        if not all_detections:
            return {}
        
        detection_counts = [len(frame_detections) for frame_detections in self.detection_history]
        confidences = [det.confidence for det in all_detections]
        areas = [det.area for det in all_detections]
        
        stats = {
            'total_frames': len(self.detection_history),
            'total_detections': len(all_detections),
            'avg_detections_per_frame': np.mean(detection_counts),
            'detection_std': np.std(detection_counts),
            'min_detections': min(detection_counts),
            'max_detections': max(detection_counts),
            'avg_confidence': np.mean(confidences),
            'avg_area': np.mean(areas),
            'area_std': np.std(areas)
        }
        
        return stats
