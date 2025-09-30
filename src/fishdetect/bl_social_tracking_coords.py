# -*- coding: utf-8 -*-
"""
BL Social Fish Tracking with Coordinate Output
Modified from TrackingProgram-Adult-BL-Social-BL-Yellow---Newrig (1).py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import re
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import remove_small_objects
from skimage import img_as_ubyte
import scipy.misc
import scipy.interpolate
import scipy.io
import pims
from tqdm import tqdm
import pandas as pd
import math
import imageio

# Set directory containing image files
inDir = "/Users/jasonli/FishDetection/"

# Names of videos to track
Name = 'data/input/videos/Clutch1_20250804_122715.mp4'

# Gene name
Gene = "BL_Social"

# Set directory containing background files
inDirbg = "/Users/jasonli/FishDetection/"

# Names of videos for background
Namebg = 'data/input/videos/Clutch1_20250804_122715.mp4'

# Set the directory in which to save the files
SaveTrackingIm = True

desttracked = inDir + 'data/output/tracked_images/'
desttracking = inDir + 'data/output/tracking_results/'
destgene = inDir + 'data/output/gene_analysis/'

# Number of fish to track
nFish = 14

# Number of rows and columns of wells
nRow = 2
nCol = 7

# Number of pixels in x and y direction
nPixely = 544
nPixelx = 512

# Frame rate in Hz
framerate = 20

# Time between frames in milliseconds
framelength = 1/framerate * 1000

# Define block time in seconds
blocksec = 60

# This define middle of each lane on top and bottom columns
ycutofftop1 = 130
ycutofftop2 = 388

yvaluemiddletop = 210
yvaluemiddlebottom = 308

# Define pixels
glassrow1 = 42
glassrow2 = 214
glassrow3 = 304
glassrow4 = 478

# Define number of total blocks
nBlocks = 20

# Define number of total expts
nExpts = 2

# This is to say if there is no folder, can make one
os.makedirs(os.path.dirname(desttracked), exist_ok=True)
os.makedirs(os.path.dirname(desttracking), exist_ok=True)
os.makedirs(os.path.dirname(destgene), exist_ok=True)

# Load video
video = pims.PyAVReaderIndexed(inDir + Name)
nFrames = len(video)

# Number of frames per block
nFramesblock = math.floor(nFrames/nBlocks)

# Load background video
videobg = pims.PyAVReaderIndexed(inDirbg + Namebg)
nFramesbg = len(videobg)

# Background subtraction
print("Creating background...")
fr1 = videobg[np.round(0)][:,:,0]
fr2 = videobg[np.round(2000)][:,:,0]
fr3 = videobg[np.round(4000)][:,:,0]
fr4 = videobg[np.round(6000)][:,:,0]
fr5 = videobg[np.round(8000)][:,:,0]
fr6 = videobg[np.round(10000)][:,:,0]
fr7 = videobg[np.round(12000)][:,:,0]
fr8 = videobg[np.round(14000)][:,:,0]
fr9 = videobg[np.round(16000)][:,:,0]
fr10 = videobg[np.round(18000)][:,:,0]
fr11 = videobg[np.round(20000)][:,:,0]
fr12 = videobg[np.round(22000)][:,:,0]

bg = np.median(np.stack((fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10, fr11, fr12), axis=2), axis=2)
bg1 = cv2.GaussianBlur(bg.astype(np.float64), (3,3),0)

def tracking(bg, img, index):
    """Track fish in a single frame"""
    
    if index <= nFrames:
        # Do a background subtraction, and blur the background
        mask = abs(bg1 - img)
        
        mask_smooth = cv2.GaussianBlur(mask, (3,3),0)
        
        # Threshold the mask
        th_img = cv2.threshold(mask_smooth, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Label the connected components in the thresholded mask
        lb = label(th_img)
        
        # Remove small objects from thresholded image
        lb_sorem = remove_small_objects(lb, 25)
        
        # Re-label since small objects removes so get 1 to nFish
        lb2 = label(lb_sorem)
        
        # Get statistics on these components
        props = regionprops(lb2)
        
        totobj = np.amax(lb2)
        print(f"Found {totobj} objects in frame {index+1}")
        
        # Make an overlay image so we can save them to inspect where the tracked points
        # are getting placed on the image
        im_height, im_width = np.shape(mask_smooth)
        
        # Copy the smoothed mask into a temporary image
        im_tmp = np.copy(mask_smooth)
        
        # Make single dimensional arrays the size of nFish to fill with x,y coordinates 
        # of the tracked points. Fill their initial entries with nans
        y_cent = np.zeros(nFish)
        y_cent[:] = np.nan
        x_cent = np.zeros(nFish)
        x_cent[:] = np.nan
        
        for ind in range(0, totobj):
            # Find the centroid of        
            cent = props[ind].centroid
            
            # Filter out objects in glass areas
            if cent[0] < glassrow1:
                continue  
            if cent[0] > glassrow2 and cent[0] < glassrow3:
                continue   
            if cent[0] > glassrow4:
                continue        
            
            # Find the centroid of        
            cent = props[ind].centroid
            fishrow = 0
            fishcol = 0
            
            # Go through and get row and column of this fish so can use it to index array
            for row in range(1, nRow+1):
                if cent[0] < row*nPixely/nRow:
                    fishrow = row
                    break
            
            for col in range(1, nCol+1):
                if cent[1] < col*nPixely/nCol:
                    fishcol = col
                    break
            
            # This will return number of fish from 1 to nFish starting in top left corner
            fishnum = ((fishrow-1) * nCol) + fishcol
            
            # If there is already data in arrays for this fish number, skip
            if np.isfinite(y_cent[fishnum-1]):
                y_cent[fishnum-1] = np.nan
                x_cent[fishnum-1] = np.nan
                continue
            
            # Find the centroid of the largest component
            y_cent[fishnum-1], x_cent[fishnum-1] = cent
            
            # Set the cent point to 255 in this temporary image
            im_tmp[y_cent[fishnum-1].astype(int), x_cent[fishnum-1].astype(int)] = 255

    return im_tmp, y_cent, x_cent

# Initialize an array to hold the coordinates of the tracked points
cents = np.zeros((nFish, 2, nFrames))

# Tracking
print("Tracking fish...")

# Track each frame
for index, img in enumerate(tqdm(video)):
    if (index < nFrames):
        ind = index
        img = img[:,:,0]  # Convert to grayscale
        
        # Apply Gaussian blur
        img = cv2.GaussianBlur(img.astype(np.float64), (3,3),0)
        
        # Call the tracking function
        timg, y_cent, x_cent = tracking(bg, img, ind)
        
        # Store coordinates (y, x format)
        cents[:, 0, ind] = y_cent
        cents[:, 1, ind] = x_cent
        
        # Save the tracked image files
        if SaveTrackingIm:
            cv2.imwrite(inDir + 'data/input/frames/frame_' + str(index).zfill(3) + '_tracked.png', np.vstack((timg)))        

# Save tracking data arrays
print("Saving tracking data arrays....")
np.savez(inDir + 'data/output/tracking_results/TrackData_BL_Social.npz', cents=cents)

# Create coordinate output files
print("Creating coordinate output files...")

# Create a CSV file with all coordinates
coords_data = []
for fish_id in range(nFish):
    for frame in range(nFrames):
        y_coord = cents[fish_id, 0, frame]
        x_coord = cents[fish_id, 1, frame]
        
        # Only include valid coordinates (not NaN)
        if not np.isnan(y_coord) and not np.isnan(x_coord):
            coords_data.append({
                'fish_id': fish_id + 1,
                'frame': frame,
                'x_coordinate': x_coord,
                'y_coordinate': y_coord,
                'time_seconds': frame / framerate
            })

# Convert to DataFrame and save
coords_df = pd.DataFrame(coords_data)
coords_df.to_csv(inDir + 'data/output/tracking_results/BL_Social_Coordinates.csv', index=False)

# Create individual fish coordinate files
for fish_id in range(nFish):
    fish_coords = coords_df[coords_df['fish_id'] == fish_id + 1]
    if not fish_coords.empty:
        fish_coords.to_csv(inDir + f'data/output/tracking_results/Fish_{fish_id+1}_Coordinates.csv', index=False)

# Create summary statistics
print("Creating summary statistics...")
summary_stats = []
for fish_id in range(nFish):
    fish_coords = coords_df[coords_df['fish_id'] == fish_id + 1]
    if not fish_coords.empty:
        summary_stats.append({
            'fish_id': fish_id + 1,
            'total_frames_tracked': len(fish_coords),
            'tracking_percentage': (len(fish_coords) / nFrames) * 100,
            'mean_x': fish_coords['x_coordinate'].mean(),
            'mean_y': fish_coords['y_coordinate'].mean(),
            'std_x': fish_coords['x_coordinate'].std(),
            'std_y': fish_coords['y_coordinate'].std(),
            'min_x': fish_coords['x_coordinate'].min(),
            'max_x': fish_coords['x_coordinate'].max(),
            'min_y': fish_coords['y_coordinate'].min(),
            'max_y': fish_coords['y_coordinate'].max()
        })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(inDir + 'data/output/tracking_results/BL_Social_Summary_Statistics.csv', index=False)

# Create a simple visualization of fish trajectories
print("Creating trajectory visualization...")
plt.figure(figsize=(15, 10))

for fish_id in range(min(6, nFish)):  # Show first 6 fish
    fish_coords = coords_df[coords_df['fish_id'] == fish_id + 1]
    if not fish_coords.empty:
        plt.subplot(2, 3, fish_id + 1)
        plt.plot(fish_coords['x_coordinate'], fish_coords['y_coordinate'], 'b-', alpha=0.7, linewidth=1)
        plt.scatter(fish_coords['x_coordinate'].iloc[0], fish_coords['y_coordinate'].iloc[0], 
                   color='green', s=50, label='Start', zorder=5)
        plt.scatter(fish_coords['x_coordinate'].iloc[-1], fish_coords['y_coordinate'].iloc[-1], 
                   color='red', s=50, label='End', zorder=5)
        plt.title(f'Fish {fish_id + 1} Trajectory')
        plt.xlabel('X Coordinate (pixels)')
        plt.ylabel('Y Coordinate (pixels)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates

plt.tight_layout()
plt.savefig(inDir + 'data/output/tracking_results/BL_Social_Trajectories.png', dpi=300, bbox_inches='tight')
plt.close()

print("BL Social tracking completed!")
print(f"Total frames processed: {nFrames}")
print(f"Total fish tracked: {nFish}")
print(f"Coordinates saved to: {inDir}data/output/tracking_results/")
print(f"Main coordinate file: BL_Social_Coordinates.csv")
print(f"Summary statistics: BL_Social_Summary_Statistics.csv")
print(f"Trajectory plot: BL_Social_Trajectories.png")
