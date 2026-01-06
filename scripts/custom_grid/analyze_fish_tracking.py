#!/usr/bin/env python3
"""
Analyze per-fish tracking statistics from custom grid pipeline outputs.

For each fish (cell), calculates:
- % of frames correctly tracked
- Longest stretch of frames where it is incorrectly tracked
"""

import cv2
import numpy as np
import pims
from pathlib import Path
from typing import List, Tuple, Dict

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from tracking_methods.hsv_masking.hsv_masker import HSVMasker
from processing.tracking_program_background_subtractor_v2 import TrackingProgramBackgroundSubtractorV2
from .pipeline import get_custom_grid_cells, find_largest_blob_in_cell


def analyze_fish_tracking(output_dir: Path, video_path: Path, fps: int = 20) -> None:
    """
    Analyze per-fish tracking statistics from processed video.
    
    Args:
        output_dir: Directory containing the pipeline output
        video_path: Path to the original input video
        fps: Frames per second
    """
    output_dir = Path(output_dir)
    
    # Load background model
    bg_model_path = output_dir / "background_model.npy"
    if not bg_model_path.exists():
        print(f"Error: Background model not found at {bg_model_path}")
        return
    
    # Initialize background subtractor
    bg_subtractor = TrackingProgramBackgroundSubtractorV2(
        threshold=15,
        morph_kernel_size=(5, 5)
    )
    bg_subtractor.load_background_model(bg_model_path)
    
    # Initialize HSV masker
    hsv_masker = HSVMasker(
        lower_hsv=(0, 0, 100),
        upper_hsv=(180, 255, 255),
        min_object_size=10,
        apply_morphology=True
    )
    
    # Load video
    print(f"Loading video: {video_path}")
    video = pims.PyAVReaderIndexed(str(video_path))
    
    height, width = video[0].shape[:2]
    num_frames = len(video)
    
    # Get custom grid cells
    cells, row_boundaries = get_custom_grid_cells(width, height)
    num_cells = len(cells)
    
    print(f"Analyzing {num_frames} frames with {num_cells} cells...")
    
    # Track per-cell detection status for each frame
    # all_cell_detections[frame_idx][cell_idx] = 1 if detected, 0 if not
    all_cell_detections = []
    
    # Process each frame
    for frame_idx in range(num_frames):
        frame = video[frame_idx]
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray_frame = frame[:, :, 0]
        else:
            gray_frame = frame
        
        # Apply background subtraction
        bg_subtracted = bg_subtractor.apply_background_subtraction(gray_frame)
        
        # Process each cell
        cell_detections = [0] * num_cells
        
        for cell_idx, (x_start, y_start, x_end, y_end) in enumerate(cells):
            contour, centroid = find_largest_blob_in_cell(
                hsv_masker,
                bg_subtracted,
                x_start, y_start, x_end, y_end
            )
            
            if contour is not None and centroid is not None:
                cell_detections[cell_idx] = 1
        
        all_cell_detections.append(cell_detections)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames...")
    
    # Calculate per-fish statistics
    fish_stats = []
    
    for cell_idx in range(num_cells):
        # Get detection status for this cell across all frames
        cell_detection_sequence = [detections[cell_idx] for detections in all_cell_detections]
        
        # Calculate percentage correctly tracked
        total_frames = len(cell_detection_sequence)
        correctly_tracked = sum(cell_detection_sequence)
        percent_correct = (correctly_tracked / total_frames) * 100 if total_frames > 0 else 0
        
        # Find longest stretch of incorrect tracking (consecutive 0s)
        longest_incorrect_stretch = 0
        current_stretch = 0
        
        for detection in cell_detection_sequence:
            if detection == 0:  # Not detected (incorrect)
                current_stretch += 1
                longest_incorrect_stretch = max(longest_incorrect_stretch, current_stretch)
            else:  # Detected (correct)
                current_stretch = 0
        
        # Also find frame ranges where tracking failed
        incorrect_ranges = []
        start_frame = None
        for frame_idx, detection in enumerate(cell_detection_sequence):
            if detection == 0:
                if start_frame is None:
                    start_frame = frame_idx
            else:
                if start_frame is not None:
                    incorrect_ranges.append((start_frame, frame_idx - 1))
                    start_frame = None
        # Handle case where sequence ends with incorrect tracking
        if start_frame is not None:
            incorrect_ranges.append((start_frame, len(cell_detection_sequence) - 1))
        
        # Calculate row and column for this cell
        row_idx = cell_idx // 7
        col_idx = cell_idx % 7
        
        fish_stats.append({
            'cell_idx': cell_idx,
            'row': row_idx + 1,  # 1-indexed
            'col': col_idx + 1,  # 1-indexed
            'total_frames': total_frames,
            'correctly_tracked': correctly_tracked,
            'percent_correct': percent_correct,
            'longest_incorrect_stretch': longest_incorrect_stretch,
            'incorrect_ranges': incorrect_ranges,
            'num_incorrect_stretches': len(incorrect_ranges)
        })
    
    # Generate detailed report
    report_path = output_dir / "per_fish_tracking_analysis.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Per-Fish Tracking Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Video: {video_path.name}\n")
        f.write(f"Total Frames: {num_frames}\n")
        f.write(f"Total Cells (Fish): {num_cells}\n")
        f.write(f"Grid Layout: 7 columns × 4 rows\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Summary Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        avg_percent_correct = np.mean([stat['percent_correct'] for stat in fish_stats])
        avg_longest_stretch = np.mean([stat['longest_incorrect_stretch'] for stat in fish_stats])
        max_longest_stretch = max([stat['longest_incorrect_stretch'] for stat in fish_stats])
        
        f.write(f"Average % Correctly Tracked: {avg_percent_correct:.2f}%\n")
        f.write(f"Average Longest Incorrect Stretch: {avg_longest_stretch:.1f} frames\n")
        f.write(f"Maximum Longest Incorrect Stretch: {max_longest_stretch} frames\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Per-Fish Detailed Statistics\n")
        f.write("=" * 80 + "\n\n")
        
        # Sort by cell index (row, then column)
        sorted_stats = sorted(fish_stats, key=lambda x: (x['row'], x['col']))
        
        for stat in sorted_stats:
            f.write(f"Fish {stat['cell_idx']:2d} (Row {stat['row']}, Col {stat['col']})\n")
            f.write(f"  {'-' * 76}\n")
            f.write(f"  Total Frames: {stat['total_frames']}\n")
            f.write(f"  Correctly Tracked: {stat['correctly_tracked']} frames ({stat['percent_correct']:.2f}%)\n")
            f.write(f"  Incorrectly Tracked: {stat['total_frames'] - stat['correctly_tracked']} frames ({100 - stat['percent_correct']:.2f}%)\n")
            f.write(f"  Longest Incorrect Stretch: {stat['longest_incorrect_stretch']} frames\n")
            f.write(f"  Number of Incorrect Stretches: {stat['num_incorrect_stretches']}\n")
            
            if stat['incorrect_ranges']:
                f.write(f"  Incorrect Tracking Ranges:\n")
                for start, end in stat['incorrect_ranges']:
                    if start == end:
                        f.write(f"    - Frame {start}\n")
                    else:
                        f.write(f"    - Frames {start} to {end} ({end - start + 1} frames)\n")
            else:
                f.write(f"  Incorrect Tracking Ranges: None (perfect tracking!)\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Tracking Quality Ranking\n")
        f.write("=" * 80 + "\n\n")
        
        # Rank by percent correct (best first)
        ranked_by_percent = sorted(fish_stats, key=lambda x: x['percent_correct'], reverse=True)
        f.write("Best Tracked (by % correct):\n")
        for i, stat in enumerate(ranked_by_percent[:5], 1):
            f.write(f"  {i}. Fish {stat['cell_idx']:2d} (R{stat['row']}, C{stat['col']}): {stat['percent_correct']:.2f}%\n")
        
        f.write("\nWorst Tracked (by % correct):\n")
        for i, stat in enumerate(ranked_by_percent[-5:], 1):
            f.write(f"  {i}. Fish {stat['cell_idx']:2d} (R{stat['row']}, C{stat['col']}): {stat['percent_correct']:.2f}%\n")
        
        f.write("\n")
        
        # Rank by longest incorrect stretch (worst first)
        ranked_by_stretch = sorted(fish_stats, key=lambda x: x['longest_incorrect_stretch'], reverse=True)
        f.write("Longest Incorrect Stretches:\n")
        for i, stat in enumerate(ranked_by_stretch[:5], 1):
            f.write(f"  {i}. Fish {stat['cell_idx']:2d} (R{stat['row']}, C{stat['col']}): {stat['longest_incorrect_stretch']} frames\n")
    
    print(f"\n✓ Per-fish tracking analysis saved to: {report_path}")


def main():
    """Analyze tracking for specified videos."""
    base_output_dir = Path("data/output/CustonGrid")
    
    # Videos to analyze: (folder_name, video_path, description)
    videos_to_analyze = [
        ("custom_grid_full_clutch", Path("data/input/videos/Clutch1_20250804_122715.mp4"), "20-minute video"),
        ("custom_grid_3min", Path("data/input/videos/20250603_1553_tlf-inx_S5374_DOB250425_3minutetest.avi"), "3-minute video"),
        ("custom_grid_22min", Path("data/input/videos/20250618_1127_S5403_DOB_06.18.25.avi"), "22-minute video"),
    ]
    
    for folder_name, video_path, description in videos_to_analyze:
        output_dir = base_output_dir / folder_name
        
        if not output_dir.exists():
            print(f"⚠️  Warning: Output directory not found: {output_dir}")
            continue
        
        if not video_path.exists():
            print(f"⚠️  Warning: Video not found: {video_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Analyzing: {description}")
        print(f"Output Directory: {output_dir}")
        print(f"{'='*80}\n")
        
        try:
            analyze_fish_tracking(output_dir, video_path)
        except Exception as e:
            print(f"❌ Error analyzing {description}: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n✓ Completed analysis for {description}\n")
    
    print(f"\n{'='*80}")
    print("🎉 All analyses completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
