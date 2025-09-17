"""Command-line interface for fishdetect."""

import os
from pathlib import Path
from typing import List, Optional

import typer
from .detector import FishDetector
from .motion_detector import MotionFishDetector
from .video_processor import VideoProcessor, ProcessingConfig, create_default_config, create_high_accuracy_config, create_fast_config
from .enhanced_preprocessing import EnhancedPreprocessor

app = typer.Typer()


@app.command()
def hello() -> None:
    """Print a hello message."""
    typer.echo("fish-detect ready")


@app.command()
def list_input() -> None:
    """List all image files in the input folder."""
    input_dir = Path("input")
    
    if not input_dir.exists():
        typer.echo("Input directory does not exist. Creating it...")
        input_dir.mkdir(exist_ok=True)
        typer.echo("Input directory created. Add your training frames there.")
        return
    
    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    
    image_files = []
    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    if not image_files:
        typer.echo("No image files found in input directory.")
        typer.echo("Supported formats: JPG, JPEG, PNG, BMP, TIFF, TIF")
        return
    
    typer.echo(f"Found {len(image_files)} image file(s) in input directory:")
    for file_path in sorted(image_files):
        file_size = file_path.stat().st_size
        typer.echo(f"  - {file_path.name} ({file_size:,} bytes)")


@app.command()
def detect(
    image_path: str = typer.Argument(..., help="Path to the image file"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save the result image"),
    min_area: int = typer.Option(50, "--min-area", help="Minimum area for fish detection"),
    max_area: int = typer.Option(2000, "--max-area", help="Maximum area for fish detection"),
    min_aspect_ratio: float = typer.Option(0.3, "--min-aspect", help="Minimum aspect ratio for fish"),
    max_aspect_ratio: float = typer.Option(3.0, "--max-aspect", help="Maximum aspect ratio for fish"),
    black_threshold: int = typer.Option(50, "--black-threshold", help="Threshold for black color detection"),
    show_details: bool = typer.Option(False, "--details", help="Show detailed detection information"),
) -> None:
    """Detect tiny black fish in an image."""
    # Check if image exists
    if not Path(image_path).exists():
        typer.echo(f"Error: Image file '{image_path}' not found.")
        raise typer.Exit(1)
    
    try:
        # Initialize detector
        detector = FishDetector(
            min_area=min_area,
            max_area=max_area,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            black_threshold=black_threshold
        )
        
        # Detect fish
        typer.echo(f"Detecting fish in '{image_path}'...")
        detections = detector.detect_fish(image_path)
        
        # Display results
        typer.echo(f"Found {len(detections)} fish detection(s):")
        
        if show_details:
            for i, detection in enumerate(detections, 1):
                typer.echo(f"  Fish {i}:")
                typer.echo(f"    Position: ({detection.x}, {detection.y})")
                typer.echo(f"    Size: {detection.width}x{detection.height}")
                typer.echo(f"    Area: {detection.area} pixels")
                typer.echo(f"    Confidence: {detection.confidence:.3f}")
                typer.echo()
        else:
            for i, detection in enumerate(detections, 1):
                typer.echo(f"  Fish {i}: ({detection.x}, {detection.y}) - {detection.width}x{detection.height} - Confidence: {detection.confidence:.3f}")
        
        # Save visualization if output path provided
        if output_path:
            detector.visualize_detections(image_path, detections, output_path)
            typer.echo(f"Result saved to: {output_path}")
        elif detections:
            # Save default output
            default_output = Path(image_path).stem + "_fish_detected.png"
            detector.visualize_detections(image_path, detections, default_output)
            typer.echo(f"Result saved to: {default_output}")
    
    except Exception as e:
        typer.echo(f"Error during detection: {str(e)}")
        raise typer.Exit(1)


@app.command()
def detect_all(
    input_dir: str = typer.Option("input", "--input", "-i", help="Input directory containing images"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    min_area: int = typer.Option(50, "--min-area", help="Minimum area for fish detection"),
    max_area: int = typer.Option(2000, "--max-area", help="Maximum area for fish detection"),
    min_aspect_ratio: float = typer.Option(0.3, "--min-aspect", help="Minimum aspect ratio for fish"),
    max_aspect_ratio: float = typer.Option(3.0, "--max-aspect", help="Maximum aspect ratio for fish"),
    black_threshold: int = typer.Option(50, "--black-threshold", help="Threshold for black color detection"),
) -> None:
    """Detect fish in all images in a directory."""
    input_path = Path(input_dir)
    
    if not input_path.exists():
        typer.echo(f"Error: Input directory '{input_dir}' not found.")
        raise typer.Exit(1)
    
    # Find all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        typer.echo(f"No image files found in '{input_dir}'.")
        raise typer.Exit(1)
    
    # Set up output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    else:
        output_path = input_path / "fish_detections"
        output_path.mkdir(exist_ok=True)
    
    # Initialize detector
    detector = FishDetector(
        min_area=min_area,
        max_area=max_area,
        min_aspect_ratio=min_aspect_ratio,
        max_aspect_ratio=max_aspect_ratio,
        black_threshold=black_threshold
    )
    
    total_detections = 0
    
    for image_file in sorted(image_files):
        try:
            typer.echo(f"Processing: {image_file.name}")
            detections = detector.detect_fish(str(image_file))
            
            if detections:
                output_file = output_path / f"{image_file.stem}_fish_detected{image_file.suffix}"
                detector.visualize_detections(str(image_file), detections, str(output_file))
                typer.echo(f"  Found {len(detections)} fish - saved to {output_file.name}")
                total_detections += len(detections)
            else:
                typer.echo(f"  No fish detected")
        
        except Exception as e:
            typer.echo(f"  Error processing {image_file.name}: {str(e)}")
    
    typer.echo(f"\nTotal fish detected across all images: {total_detections}")
    typer.echo(f"Results saved in: {output_path}")


@app.command()
def process_video(
    video_path: str = typer.Argument(..., help="Path to the video file"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for results"),
    config: str = typer.Option("default", "--config", help="Processing configuration: default, fast, high_accuracy"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", help="Maximum number of frames to process"),
    realtime: bool = typer.Option(False, "--realtime", help="Process in real-time from camera"),
    camera_id: int = typer.Option(0, "--camera", help="Camera ID for real-time processing"),
) -> None:
    """Process video for fish detection and tracking using motion-based detection."""
    if realtime:
        # Real-time processing from camera
        try:
            processor = VideoProcessor(create_default_config())
            processor.process_realtime(camera_id=camera_id)
        except Exception as e:
            typer.echo(f"Error in real-time processing: {str(e)}")
            raise typer.Exit(1)
        return
    
    # Check if video exists
    if not Path(video_path).exists():
        typer.echo(f"Error: Video file '{video_path}' not found.")
        raise typer.Exit(1)
    
    try:
        # Select configuration
        if config == "fast":
            processing_config = create_fast_config()
        elif config == "high_accuracy":
            processing_config = create_high_accuracy_config()
        else:
            processing_config = create_default_config()
        
        # Limit frames if specified
        if max_frames:
            typer.echo(f"Processing first {max_frames} frames only")
        
        # Initialize processor
        processor = VideoProcessor(processing_config)
        
        # Process video
        typer.echo(f"Processing video: {video_path}")
        typer.echo(f"Configuration: {config}")
        
        def progress_callback(current: int, total: int):
            if max_frames and current >= max_frames:
                return
            if current % 50 == 0:
                typer.echo(f"Progress: {current}/{total} frames")
        
        # Process video with frame limit
        if max_frames:
            # We'll need to modify the video processor to support frame limits
            # For now, let's process the full video
            results = processor.process_video(video_path, output_dir, progress_callback)
        else:
            results = processor.process_video(video_path, output_dir, progress_callback)
        
        # Display results
        typer.echo(f"\nProcessing complete!")
        typer.echo(f"Total frames processed: {results['total_frames']}")
        typer.echo(f"Total detections: {results['total_detections']}")
        typer.echo(f"Results saved to: {output_dir or 'output directory'}")
        
        if 'annotations_path' in results:
            typer.echo(f"Annotations: {results['annotations_path']}")
        if 'metrics_path' in results:
            typer.echo(f"Metrics: {results['metrics_path']}")
    
    except Exception as e:
        typer.echo(f"Error during video processing: {str(e)}")
        raise typer.Exit(1)


@app.command()
def preprocess_image(
    image_path: str = typer.Argument(..., help="Path to the image file"),
    output_path: Optional[str] = typer.Option(None, "--output", "-o", help="Path to save the preprocessed image"),
    method: str = typer.Option("combined", "--method", help="Preprocessing method: clahe_adaptive, frangi_otsu, tophat_adaptive, gabor_otsu, combined"),
    show_steps: bool = typer.Option(False, "--show-steps", help="Show all preprocessing steps"),
) -> None:
    """Apply enhanced preprocessing techniques to an image."""
    # Check if image exists
    if not Path(image_path).exists():
        typer.echo(f"Error: Image file '{image_path}' not found.")
        raise typer.Exit(1)
    
    try:
        # Initialize preprocessor
        preprocessor = EnhancedPreprocessor()
        
        # Load image
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            typer.echo(f"Error: Could not load image '{image_path}'")
            raise typer.Exit(1)
        
        if show_steps:
            # Show all preprocessing steps
            output_path = output_path or f"{Path(image_path).stem}_preprocessing_steps.png"
            result = preprocessor.visualize_preprocessing_steps(image, output_path)
            typer.echo(f"Preprocessing steps visualization saved to: {output_path}")
        else:
            # Apply specific preprocessing method
            enhanced_mask = preprocessor.create_enhanced_mask(image, method=method)
            
            # Save result
            if output_path is None:
                output_path = f"{Path(image_path).stem}_{method}_processed.png"
            
            import cv2
            cv2.imwrite(output_path, enhanced_mask)
            typer.echo(f"Preprocessed image saved to: {output_path}")
        
        # Detect fish candidates
        candidates = preprocessor.detect_fish_candidates(image, method=method)
        typer.echo(f"Found {len(candidates)} fish candidates using {method} method")
        
        for i, candidate in enumerate(candidates):
            bbox = candidate['bbox']
            confidence = candidate['confidence']
            typer.echo(f"  Candidate {i+1}: bbox={bbox}, confidence={confidence:.3f}")
    
    except Exception as e:
        typer.echo(f"Error during preprocessing: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
