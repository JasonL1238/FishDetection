#!/usr/bin/env python3
"""
Analyze per-fish tracking statistics from pipeline output.

For each fish (grid cell) this calculates:
  - % of frames where the fish was correctly detected
  - Longest consecutive stretch of missed detections
"""

import numpy as np
import pims
from pathlib import Path
from typing import List

from fishdetection.background_subtractor import BackgroundSubtractor
from fishdetection.hsv_masker import HSVMasker
from fishdetection.grid import get_grid_cells, find_largest_blob_in_cell


def analyze_fish_tracking(output_dir: Path, video_path: Path,
                          fps: int = 20) -> None:
    """Run per-fish tracking analysis on a previously processed video."""
    output_dir = Path(output_dir)

    bg_model_path = output_dir / "background_model.npy"
    if not bg_model_path.exists():
        print(f"Error: Background model not found at {bg_model_path}")
        return

    bg_subtractor = BackgroundSubtractor(
        threshold=15, morph_kernel_size=(5, 5)
    )
    bg_subtractor.load_background_model(bg_model_path)

    hsv_masker = HSVMasker(
        lower_hsv=(0, 0, 100),
        upper_hsv=(180, 255, 255),
        min_object_size=10,
        apply_morphology=True,
    )

    print(f"Loading video: {video_path}")
    video = pims.PyAVReaderIndexed(str(video_path))

    height, width = video[0].shape[:2]
    num_frames = len(video)

    cells, _ = get_grid_cells(width, height)
    num_cells = len(cells)

    print(f"Analyzing {num_frames} frames with {num_cells} cells...")

    all_cell_detections: List[List[int]] = []

    for frame_idx in range(num_frames):
        frame = video[frame_idx]

        if len(frame.shape) == 3:
            gray = frame[:, :, 0]
        else:
            gray = frame

        bg_sub = bg_subtractor.apply(gray)

        cell_detections = [0] * num_cells
        for cell_idx, (x0, y0, x1, y1) in enumerate(cells):
            contour, centroid = find_largest_blob_in_cell(
                hsv_masker, bg_sub, x0, y0, x1, y1
            )
            if contour is not None and centroid is not None:
                cell_detections[cell_idx] = 1

        all_cell_detections.append(cell_detections)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames} frames...")

    # --- Per-fish statistics ---
    fish_stats = []
    for cell_idx in range(num_cells):
        seq = [d[cell_idx] for d in all_cell_detections]
        total = len(seq)
        correct = sum(seq)
        pct = (correct / total) * 100 if total > 0 else 0

        longest_miss = 0
        current_miss = 0
        miss_ranges = []
        miss_start = None

        for fi, det in enumerate(seq):
            if det == 0:
                current_miss += 1
                longest_miss = max(longest_miss, current_miss)
                if miss_start is None:
                    miss_start = fi
            else:
                if miss_start is not None:
                    miss_ranges.append((miss_start, fi - 1))
                    miss_start = None
                current_miss = 0

        if miss_start is not None:
            miss_ranges.append((miss_start, total - 1))

        fish_stats.append({
            'cell': cell_idx,
            'row': cell_idx // 7 + 1,
            'col': cell_idx % 7 + 1,
            'total': total,
            'correct': correct,
            'pct': pct,
            'longest_miss': longest_miss,
            'miss_ranges': miss_ranges,
        })

    # --- Write report ---
    report_path = output_dir / "per_fish_tracking_analysis.txt"

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Per-Fish Tracking Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Video: {video_path.name}\n")
        f.write(f"Total Frames: {num_frames}\n")
        f.write(f"Grid: 7 columns x 4 rows = {num_cells} cells\n\n")

        avg_pct = np.mean([s['pct'] for s in fish_stats])
        max_miss = max(s['longest_miss'] for s in fish_stats)
        f.write(f"Average % Correctly Tracked: {avg_pct:.2f}%\n")
        f.write(f"Max Longest Miss Streak: {max_miss} frames\n\n")

        f.write("-" * 70 + "\n")
        f.write("Per-Fish Detail\n")
        f.write("-" * 70 + "\n\n")

        for s in sorted(fish_stats, key=lambda x: (x['row'], x['col'])):
            f.write(f"Fish {s['cell']:2d} (Row {s['row']}, Col {s['col']})\n")
            f.write(f"  Detected: {s['correct']}/{s['total']} "
                    f"({s['pct']:.2f}%)\n")
            f.write(f"  Longest miss: {s['longest_miss']} frames\n")
            if s['miss_ranges']:
                for a, b in s['miss_ranges'][:5]:
                    f.write(f"    Frames {a}-{b} ({b - a + 1} frames)\n")
                if len(s['miss_ranges']) > 5:
                    f.write(f"    ... and {len(s['miss_ranges']) - 5} more\n")
            f.write("\n")

        f.write("-" * 70 + "\n")
        f.write("Ranking (best first)\n")
        f.write("-" * 70 + "\n\n")
        ranked = sorted(fish_stats, key=lambda x: x['pct'], reverse=True)
        for i, s in enumerate(ranked[:5], 1):
            f.write(f"  {i}. Fish {s['cell']:2d} "
                    f"(R{s['row']},C{s['col']}): {s['pct']:.2f}%\n")
        f.write("\nWorst:\n")
        for i, s in enumerate(ranked[-5:], 1):
            f.write(f"  {i}. Fish {s['cell']:2d} "
                    f"(R{s['row']},C{s['col']}): {s['pct']:.2f}%\n")

    print(f"\nAnalysis saved to: {report_path}")


def main():
    base_output_dir = Path("data/output")

    videos = [
        ("grid_3min",
         Path("data/input/videos/"
              "20250603_1553_tlf-inx_S5374_DOB250425_3minutetest.avi"),
         "3-minute video"),
        ("grid_22min",
         Path("data/input/videos/"
              "20250618_1127_S5403_DOB_06.18.25.avi"),
         "22-minute video"),
    ]

    for folder, video_path, description in videos:
        output_dir = base_output_dir / folder

        if not output_dir.exists():
            print(f"Warning: Output dir not found: {output_dir}")
            continue
        if not video_path.exists():
            print(f"Warning: Video not found: {video_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Analyzing: {description}")
        print(f"{'=' * 60}\n")

        try:
            analyze_fish_tracking(output_dir, video_path)
        except Exception as e:
            print(f"Error analyzing {description}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
