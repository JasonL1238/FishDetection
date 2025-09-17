"""Video processing module for real-time fish detection and tracking."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
import time
import json
from dataclasses import dataclass, asdict
from .motion_detector import MotionFishDetector
from .enhanced_preprocessing import EnhancedPreprocessor
from .evaluation import DetectionEvaluator, VisualizationTools, PerformanceProfiler


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    # Motion detection parameters
    var_threshold: int = 16
    history: int = 500
    min_area: int = 20
    max_area: int = 800
    min_aspect_ratio: float = 2.2
    max_aspect_ratio: float = 8.0
    
    # Preprocessing parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    preprocessing_method: str = 'combined'
    
    # Tracking parameters
    tracking_threshold: float = 0.3
    max_track_age: int = 10
    
    # Output parameters
    save_video: bool = True
    save_annotations: bool = True
    save_metrics: bool = True
    output_fps: int = 30
    visualization_scale: float = 1.0


class VideoProcessor:
    """Main video processing class for fish detection and tracking."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize video processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        
        # Initialize components
        self.motion_detector = MotionFishDetector(
            var_threshold=self.config.var_threshold,
            history=self.config.history,
            min_area=self.config.min_area,
            max_area=self.config.max_area,
            min_aspect_ratio=self.config.min_aspect_ratio,
            max_aspect_ratio=self.config.max_aspect_ratio,
            tracking_threshold=self.config.tracking_threshold
        )
        
        self.preprocessor = EnhancedPreprocessor(
            clahe_clip_limit=self.config.clahe_clip_limit,
            clahe_tile_size=self.config.clahe_tile_size
        )
        
        self.evaluator = DetectionEvaluator()
        self.visualizer = VisualizationTools()
        self.profiler = PerformanceProfiler()
        
        # Processing state
        self.frame_count = 0
        self.detection_history = []
        self.tracking_history = []
        self.metrics_history = []
        
    def process_video(self, 
                     input_path: str,
                     output_dir: Optional[str] = None,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict:
        """
        Process a video file for fish detection and tracking.
        
        Args:
            input_path: Path to input video
            output_dir: Directory to save outputs
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary containing processing results
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_processed"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {input_path.name}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if self.config.save_video:
            output_video_path = output_dir / f"{input_path.stem}_detected.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_video_path), fourcc, 
                self.config.output_fps, (width, height)
            )
        
        # Process frames
        start_time = time.time()
        self.frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                self.profiler.start_timer('frame_processing')
                result = self._process_frame(frame)
                self.profiler.end_timer('frame_processing')
                
                # Store results
                self.detection_history.append(result['detections'])
                self.tracking_history.append(result['tracks'])
                
                # Calculate metrics if ground truth available
                if 'ground_truth' in result:
                    metrics = self.evaluator.evaluate_detections(
                        result['detections'], result['ground_truth']
                    )
                    self.metrics_history.append(metrics)
                
                # Write frame
                if writer:
                    vis_frame = self._create_visualization(frame, result)
                    if self.config.visualization_scale != 1.0:
                        h, w = vis_frame.shape[:2]
                        new_h = int(h * self.config.visualization_scale)
                        new_w = int(w * self.config.visualization_scale)
                        vis_frame = cv2.resize(vis_frame, (new_w, new_h))
                    writer.write(vis_frame)
                
                self.frame_count += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(self.frame_count, total_frames)
                
                # Print progress
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_current = self.frame_count / elapsed
                    print(f"Processed {self.frame_count}/{total_frames} frames "
                          f"({fps_current:.1f} FPS)")
        
        finally:
            cap.release()
            if writer:
                writer.release()
        
        # Save results
        results = self._save_results(output_dir, input_path.stem)
        
        # Print summary
        total_time = time.time() - start_time
        avg_fps = self.frame_count / total_time
        print(f"\nProcessing complete!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Total detections: {sum(len(d) for d in self.detection_history)}")
        print(f"Results saved to: {output_dir}")
        
        return results
    
    def _process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame."""
        # Detect fish using motion detection
        self.profiler.start_timer('motion_detection')
        tracks = self.motion_detector.process_frame(frame)
        self.profiler.end_timer('motion_detection')
        
        # Convert tracks to detection format
        detections = []
        for track in tracks:
            detections.append({
                'bbox': (track.x, track.y, track.width, track.height),
                'confidence': track.confidence,
                'id': track.id,
                'area': track.area,
                'age': track.age
            })
        
        return {
            'detections': detections,
            'tracks': [asdict(track) for track in tracks],
            'frame_number': self.frame_count
        }
    
    def _create_visualization(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Create visualization of detection results."""
        # Draw detections
        vis_frame = self.visualizer.draw_detections(
            frame, result['detections'], 
            show_confidence=True, show_ids=True
        )
        
        # Add frame info
        info_text = f"Frame: {self.frame_count}, Detections: {len(result['detections'])}"
        cv2.putText(vis_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def _save_results(self, output_dir: Path, video_name: str) -> Dict:
        """Save processing results."""
        results = {
            'video_name': video_name,
            'total_frames': self.frame_count,
            'total_detections': sum(len(d) for d in self.detection_history),
            'processing_time': time.time(),
            'config': asdict(self.config)
        }
        
        # Save annotations
        if self.config.save_annotations:
            annotations_path = output_dir / f"{video_name}_annotations.json"
            with open(annotations_path, 'w') as f:
                json.dump({
                    'detections': self.detection_history,
                    'tracks': self.tracking_history
                }, f, indent=2)
            results['annotations_path'] = str(annotations_path)
        
        # Save metrics
        if self.config.save_metrics and self.metrics_history:
            metrics_path = output_dir / f"{video_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump([asdict(m) for m in self.metrics_history], f, indent=2)
            results['metrics_path'] = str(metrics_path)
            
            # Create metrics visualization
            metrics_plot_path = output_dir / f"{video_name}_metrics_plot.png"
            self.visualizer.plot_detection_metrics(self.metrics_history, str(metrics_plot_path))
            results['metrics_plot_path'] = str(metrics_plot_path)
        
        # Save performance profile
        perf_path = output_dir / f"{video_name}_performance.json"
        with open(perf_path, 'w') as f:
            json.dump(self.profiler.get_timing_summary(), f, indent=2)
        results['performance_path'] = str(perf_path)
        
        # Create performance plot
        perf_plot_path = output_dir / f"{video_name}_performance_plot.png"
        self.visualizer.plot_performance(str(perf_plot_path))
        results['performance_plot_path'] = str(perf_plot_path)
        
        return results
    
    def process_realtime(self, 
                        camera_id: int = 0,
                        window_name: str = "Fish Detection",
                        exit_key: str = 'q') -> None:
        """
        Process video from camera in real-time.
        
        Args:
            camera_id: Camera ID (usually 0 for default camera)
            window_name: Name of the display window
            exit_key: Key to press to exit (default 'q')
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        print(f"Starting real-time detection. Press '{exit_key}' to exit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self._process_frame(frame)
                
                # Create visualization
                vis_frame = self._create_visualization(frame, result)
                
                # Display frame
                cv2.imshow(window_name, vis_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord(exit_key):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def batch_process(self, 
                     input_dir: str,
                     output_dir: str,
                     video_extensions: List[str] = None) -> Dict:
        """
        Process multiple video files in batch.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save outputs
            video_extensions: List of video file extensions to process
            
        Returns:
            Dictionary containing batch processing results
        """
        if video_extensions is None:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
            video_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not video_files:
            raise ValueError(f"No video files found in {input_dir}")
        
        print(f"Found {len(video_files)} video files to process")
        
        # Process each video
        batch_results = {}
        for i, video_file in enumerate(video_files):
            print(f"\nProcessing {i+1}/{len(video_files)}: {video_file.name}")
            
            try:
                video_output_dir = output_path / video_file.stem
                result = self.process_video(
                    str(video_file), 
                    str(video_output_dir)
                )
                batch_results[video_file.name] = result
                
            except Exception as e:
                print(f"Error processing {video_file.name}: {e}")
                batch_results[video_file.name] = {'error': str(e)}
        
        # Save batch summary
        summary_path = output_path / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        print(f"\nBatch processing complete! Summary saved to {summary_path}")
        return batch_results


def create_default_config() -> ProcessingConfig:
    """Create a default processing configuration."""
    return ProcessingConfig()


def create_high_accuracy_config() -> ProcessingConfig:
    """Create a high accuracy configuration (slower but more accurate)."""
    return ProcessingConfig(
        var_threshold=12,
        history=1000,
        min_area=15,
        max_area=1000,
        clahe_clip_limit=3.0,
        preprocessing_method='combined'
    )


def create_fast_config() -> ProcessingConfig:
    """Create a fast configuration (faster but less accurate)."""
    return ProcessingConfig(
        var_threshold=20,
        history=200,
        min_area=30,
        max_area=600,
        clahe_clip_limit=1.5,
        preprocessing_method='clahe_adaptive'
    )
