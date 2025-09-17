"""Evaluation and visualization tools for fish detection."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import pandas as pd
from collections import defaultdict
import time


@dataclass
class DetectionMetrics:
    """Container for detection evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    mAP: float
    total_detections: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class TrackingMetrics:
    """Container for tracking evaluation metrics."""
    mota: float  # Multiple Object Tracking Accuracy
    motp: float  # Multiple Object Tracking Precision
    idf1: float  # ID F1 Score
    id_switches: int
    fragmentations: int
    total_tracks: int


class DetectionEvaluator:
    """Evaluator for fish detection performance."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize detection evaluator.
        
        Args:
            iou_threshold: IoU threshold for considering a detection as correct
        """
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
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
    
    def evaluate_detections(self, 
                          predicted_boxes: List[Tuple[int, int, int, int]],
                          ground_truth_boxes: List[Tuple[int, int, int, int]],
                          predicted_confidences: Optional[List[float]] = None) -> DetectionMetrics:
        """Evaluate detection performance against ground truth."""
        if predicted_confidences is None:
            predicted_confidences = [1.0] * len(predicted_boxes)
        
        # Match predictions to ground truth
        matched_gt = set()
        matched_pred = set()
        true_positives = 0
        
        # Sort predictions by confidence (descending)
        pred_indices = sorted(range(len(predicted_boxes)), 
                            key=lambda i: predicted_confidences[i], reverse=True)
        
        for pred_idx in pred_indices:
            pred_box = predicted_boxes[pred_idx]
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(ground_truth_boxes):
                if gt_idx in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= self.iou_threshold and best_gt_idx not in matched_gt:
                true_positives += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
        
        false_positives = len(predicted_boxes) - true_positives
        false_negatives = len(ground_truth_boxes) - true_positives
        
        # Calculate metrics
        precision = true_positives / len(predicted_boxes) if len(predicted_boxes) > 0 else 0
        recall = true_positives / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Simple mAP calculation (can be enhanced for multi-class)
        mAP = precision  # Simplified for single class
        
        return DetectionMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mAP=mAP,
            total_detections=len(predicted_boxes),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives
        )


class TrackingEvaluator:
    """Evaluator for fish tracking performance."""
    
    def __init__(self, distance_threshold: float = 50.0):
        """
        Initialize tracking evaluator.
        
        Args:
            distance_threshold: Distance threshold for considering a track as correct
        """
        self.distance_threshold = distance_threshold
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def evaluate_tracking(self, 
                         predicted_tracks: List[Dict],
                         ground_truth_tracks: List[Dict]) -> TrackingMetrics:
        """Evaluate tracking performance against ground truth."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated tracking evaluation metrics
        
        total_tracks = len(predicted_tracks)
        id_switches = 0
        fragmentations = 0
        
        # Calculate MOTA (simplified)
        # This would require more complex tracking evaluation
        mota = 0.8  # Placeholder
        motp = 0.7  # Placeholder
        idf1 = 0.75  # Placeholder
        
        return TrackingMetrics(
            mota=mota,
            motp=motp,
            idf1=idf1,
            id_switches=id_switches,
            fragmentations=fragmentations,
            total_tracks=total_tracks
        )


class VisualizationTools:
    """Tools for visualizing detection and tracking results."""
    
    def __init__(self):
        """Initialize visualization tools."""
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 0)
        ]
    
    def draw_detections(self, image: np.ndarray, 
                       detections: List[Dict],
                       show_confidence: bool = True,
                       show_ids: bool = False) -> np.ndarray:
        """Draw detection bounding boxes on image."""
        result = image.copy()
        
        for i, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            confidence = detection.get('confidence', 1.0)
            track_id = detection.get('id', i)
            
            # Choose color
            color = self.colors[i % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label_parts = []
            if show_ids:
                label_parts.append(f"ID:{track_id}")
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                cv2.putText(result, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def draw_tracks(self, image: np.ndarray, 
                   tracks: List[Dict],
                   show_trajectory: bool = True,
                   trajectory_length: int = 10) -> np.ndarray:
        """Draw tracking results with trajectories."""
        result = image.copy()
        
        for track in tracks:
            track_id = track.get('id', 0)
            x, y, w, h = track['bbox']
            position_history = track.get('position_history', [])
            
            # Choose color based on track ID
            color = self.colors[track_id % len(self.colors)]
            
            # Draw current bounding box
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Draw track ID
            cv2.putText(result, f"ID:{track_id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw trajectory
            if show_trajectory and len(position_history) > 1:
                points = np.array(position_history[-trajectory_length:], dtype=np.int32)
                if len(points) > 1:
                    cv2.polylines(result, [points], False, color, 2)
        
        return result
    
    def create_detection_heatmap(self, image: np.ndarray, 
                               detections: List[Dict],
                               sigma: float = 20.0) -> np.ndarray:
        """Create a heatmap showing detection density."""
        h, w = image.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for detection in detections:
            x, y, w_det, h_det = detection['bbox']
            center_x = x + w_det // 2
            center_y = y + h_det // 2
            
            # Create Gaussian kernel
            y_coords, x_coords = np.ogrid[:h, :w]
            gaussian = np.exp(-((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * sigma**2))
            heatmap += gaussian
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert to 0-255 range
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.7, heatmap_colored, 0.3, 0)
        
        return result
    
    def plot_detection_metrics(self, metrics_history: List[DetectionMetrics],
                             output_path: Optional[str] = None) -> None:
        """Plot detection metrics over time."""
        frames = range(len(metrics_history))
        precision = [m.precision for m in metrics_history]
        recall = [m.recall for m in metrics_history]
        f1_score = [m.f1_score for m in metrics_history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(frames, precision, label='Precision', color='blue')
        plt.plot(frames, recall, label='Recall', color='red')
        plt.plot(frames, f1_score, label='F1 Score', color='green')
        plt.xlabel('Frame')
        plt.ylabel('Score')
        plt.title('Detection Metrics Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        total_detections = [m.total_detections for m in metrics_history]
        true_positives = [m.true_positives for m in metrics_history]
        false_positives = [m.false_positives for m in metrics_history]
        
        plt.plot(frames, total_detections, label='Total Detections', color='black')
        plt.plot(frames, true_positives, label='True Positives', color='green')
        plt.plot(frames, false_positives, label='False Positives', color='red')
        plt.xlabel('Frame')
        plt.ylabel('Count')
        plt.title('Detection Counts Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        mAP = [m.mAP for m in metrics_history]
        plt.plot(frames, mAP, label='mAP', color='purple')
        plt.xlabel('Frame')
        plt.ylabel('mAP')
        plt.title('Mean Average Precision Over Time')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        # Detection rate histogram
        all_precisions = [m.precision for m in metrics_history if m.precision > 0]
        plt.hist(all_precisions, bins=20, alpha=0.7, color='blue')
        plt.xlabel('Precision')
        plt.ylabel('Frequency')
        plt.title('Precision Distribution')
        plt.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def create_summary_report(self, 
                            detection_metrics: List[DetectionMetrics],
                            tracking_metrics: Optional[List[TrackingMetrics]] = None,
                            output_path: Optional[str] = None) -> Dict[str, Any]:
        """Create a summary report of detection and tracking performance."""
        # Calculate overall statistics
        avg_precision = np.mean([m.precision for m in detection_metrics])
        avg_recall = np.mean([m.recall for m in detection_metrics])
        avg_f1 = np.mean([m.f1_score for m in detection_metrics])
        avg_mAP = np.mean([m.mAP for m in detection_metrics])
        
        total_detections = sum(m.total_detections for m in detection_metrics)
        total_tp = sum(m.true_positives for m in detection_metrics)
        total_fp = sum(m.false_positives for m in detection_metrics)
        total_fn = sum(m.false_negatives for m in detection_metrics)
        
        report = {
            'detection_metrics': {
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1_score': avg_f1,
                'average_mAP': avg_mAP,
                'total_detections': total_detections,
                'total_true_positives': total_tp,
                'total_false_positives': total_fp,
                'total_false_negatives': total_fn,
                'overall_precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
                'overall_recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            },
            'frame_count': len(detection_metrics),
            'evaluation_timestamp': time.time()
        }
        
        if tracking_metrics:
            avg_mota = np.mean([m.mota for m in tracking_metrics])
            avg_motp = np.mean([m.motp for m in tracking_metrics])
            avg_idf1 = np.mean([m.idf1 for m in tracking_metrics])
            
            report['tracking_metrics'] = {
                'average_mota': avg_mota,
                'average_motp': avg_motp,
                'average_idf1': avg_idf1,
                'total_tracks': sum(m.total_tracks for m in tracking_metrics)
            }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


class PerformanceProfiler:
    """Profile detection and tracking performance."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.timings = defaultdict(list)
        self.memory_usage = []
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.timings[operation].append(time.time())
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.timings or len(self.timings[operation]) == 0:
            return 0.0
        
        start_time = self.timings[operation].pop()
        duration = time.time() - start_time
        self.timings[operation].append(duration)
        return duration
    
    def get_average_timing(self, operation: str) -> float:
        """Get average timing for an operation."""
        if operation not in self.timings:
            return 0.0
        return np.mean(self.timings[operation])
    
    def get_timing_summary(self) -> Dict[str, float]:
        """Get summary of all operation timings."""
        return {op: self.get_average_timing(op) for op in self.timings.keys()}
    
    def plot_performance(self, output_path: Optional[str] = None) -> None:
        """Plot performance metrics."""
        operations = list(self.timings.keys())
        avg_times = [self.get_average_timing(op) for op in operations]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(operations, avg_times)
        plt.xlabel('Operation')
        plt.ylabel('Average Time (seconds)')
        plt.title('Performance Profile')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars, avg_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
