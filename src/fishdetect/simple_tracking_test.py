#!/usr/bin/env python3
"""
Simplified fish tracking program for 28 fish in 4x7 layout
Tests on first 30 frames with shape outlining and centroid labeling
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pims
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import remove_small_objects

# Configuration
inDir = "/Users/jasonli/FishDetection/"
Name = 'input/Clutch1_20250804_122715.mp4'
nFish = 28
nRow = 4
nCol = 7
nPixely = 544
nPixelx = 512

# Glass row boundaries for 4 rows
glassrow1 = 42
glassrow2 = 136  # 544/4 = 136
glassrow3 = 230  # 136 + 94 (middle gap)
glassrow4 = 324  # 230 + 94
glassrow5 = 418  # 324 + 94
glassrow6 = 478

# Create output directories
output_dir = inDir + 'Results/SimpleTracking/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir + 'trackedimages/', exist_ok=True)

def tracking(bg, img, index):
    """Track fish and create overlay with shapes and centroids"""
    
    # Background subtraction
    mask = abs(bg - img)
    mask_smooth = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # Threshold
    th_img = cv2.threshold(mask_smooth, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Label connected components
    lb = label(th_img)
    lb_sorem = remove_small_objects(lb, 25)
    lb2 = label(lb_sorem)
    
    # Get region properties
    props = regionprops(lb2)
    totobj = np.amax(lb2)
    print(f"Frame {index+1}: Found {totobj} objects")
    
    # Initialize arrays
    y_cent = np.zeros(nFish)
    y_cent[:] = np.nan
    x_cent = np.zeros(nFish)
    x_cent[:] = np.nan
    
    # Create overlay image
    im_height, im_width = np.shape(mask_smooth)
    overlay = np.zeros((im_height, im_width, 3), dtype=np.uint8)
    overlay[:,:,0] = img  # Red channel
    overlay[:,:,1] = img  # Green channel  
    overlay[:,:,2] = img  # Blue channel
    
    # Process each detected object
    for ind in range(totobj):
        cent = props[ind].centroid
        
        # Check if centroid is in valid region (between glass rows)
        if (cent[0] < glassrow1 or 
            (glassrow2 < cent[0] < glassrow3) or
            (glassrow4 < cent[0] < glassrow5) or
            cent[0] > glassrow6):
            continue
        
        # Determine fish row and column
        fishrow = 0
        fishcol = 0
        
        for row in range(1, nRow + 1):
            if cent[0] < row * nPixely / nRow:
                fishrow = row
                break
        
        for col in range(1, nCol + 1):
            if cent[1] < col * nPixelx / nCol:
                fishcol = col
                break
        
        # Calculate fish number (1-28)
        fishnum = ((fishrow - 1) * nCol) + fishcol
        
        # Skip if fish number is out of range
        if fishnum < 1 or fishnum > nFish:
            continue
            
        # Skip if this fish already has a centroid
        if np.isfinite(y_cent[fishnum - 1]):
            continue
        
        # Store centroid
        y_cent[fishnum - 1] = cent[0]
        x_cent[fishnum - 1] = cent[1]
        
        # Draw shape outline and centroid
        if not np.isnan(y_cent[fishnum - 1]) and not np.isnan(x_cent[fishnum - 1]):
            # Draw contour around the fish shape (green)
            contours = find_contours(lb2 == (ind + 1), 0.5)
            if len(contours) > 0:
                contour = contours[0]
                contour_int = contour.astype(int)
                for i in range(len(contour_int) - 1):
                    cv2.line(overlay, 
                            (contour_int[i, 1], contour_int[i, 0]), 
                            (contour_int[i+1, 1], contour_int[i+1, 0]), 
                            (0, 255, 0), 2)
            
            # Draw centroid as a circle (red)
            cv2.circle(overlay, (int(x_cent[fishnum - 1]), int(y_cent[fishnum - 1])), 5, (0, 0, 255), -1)
            
            # Add fish number label (white text)
            cv2.putText(overlay, str(fishnum), 
                       (int(x_cent[fishnum - 1]) + 10, int(y_cent[fishnum - 1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return overlay, y_cent, x_cent

def main():
    print("Loading video...")
    video = pims.PyAVReaderIndexed(inDir + Name)
    nFrames = min(len(video), 30)  # Limit to first 30 frames
    
    print(f"Processing {nFrames} frames...")
    
    # Create background from first few frames
    print("Creating background...")
    bg_frames = []
    for i in range(min(10, nFrames)):
        bg_frames.append(video[i][:,:,0])
    bg = np.median(np.stack(bg_frames, axis=2), axis=2)
    bg = cv2.GaussianBlur(bg.astype(np.float64), (3, 3), 0)
    
    # Initialize tracking data
    cents = np.zeros((nFish, 2, nFrames))
    
    # Process each frame
    for index in tqdm(range(nFrames), desc="Tracking"):
        img = video[index][:,:,0]
        img = cv2.GaussianBlur(img.astype(np.float64), (3, 3), 0)
        
        # Track fish
        overlay, y_cent, x_cent = tracking(bg, img, index)
        
        # Store centroids
        cents[:, 0, index] = y_cent
        cents[:, 1, index] = x_cent
        
        # Save overlay image
        cv2.imwrite(f"{output_dir}trackedimages/frame_{index:03d}_overlay.png", overlay)
        
        # Save original frame for comparison
        cv2.imwrite(f"{output_dir}trackedimages/frame_{index:03d}_original.png", img)
    
    # Save tracking data
    np.savez(f"{output_dir}tracking_data.npz", cents=cents)
    
    # Print summary
    print(f"\nTracking complete!")
    print(f"Processed {nFrames} frames")
    print(f"Tracked up to {nFish} fish in {nRow}x{nCol} layout")
    print(f"Results saved to: {output_dir}")
    
    # Print detection statistics
    valid_detections = np.sum(~np.isnan(cents[:, 0, :]))
    print(f"Total valid detections: {valid_detections}")
    print(f"Average detections per frame: {valid_detections / nFrames:.1f}")

if __name__ == "__main__":
    main()
