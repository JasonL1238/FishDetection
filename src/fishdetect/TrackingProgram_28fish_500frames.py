# -*- coding: utf-8 -*-
"""
Modified tracking program for 28 fish, first 500 frames only
Based on TrackingProgram-Adult-BL-Social-BL-Yellow---Newrig (1).py
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
Gene = "test"

# Set directory containing background files
inDirbg = "/Users/jasonli/FishDetection/"

# Names of videos for background
Namebg = 'data/input/videos/Clutch1_20250804_122715.mp4'

# Set the directory in which to save the files
SaveTrackingIm = True

desttracked = inDir + 'data/output/tracked_images/'
desttracking = inDir + 'data/output/tracking_results/'
destgene = inDir + 'data/output/gene_analysis/'

# Number of fish to track - MODIFIED FOR 28 FISH
nFish = 28

# Number of rows and columns of wells - MODIFIED FOR 28 FISH (4 rows x 7 columns)
nRow = 4
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

# Define middle of each lane on top and bottom columns
ycutofftop1 = 130
ycutofftop2 = 388

yvaluemiddletop = 210
yvaluemiddlebottom = 308

# Define pixels for 4 rows
glassrow1 = 42
glassrow2 = 214
glassrow3 = 304
glassrow4 = 478
glassrow5 = 544  # Additional row for 4-row setup

# Define number of total blocks
nBlocks = 20

# Define number of total expts
nExpts = 2

# Create directories if they don't exist
os.makedirs(os.path.dirname(desttracked), exist_ok=True)
os.makedirs(os.path.dirname(desttracking), exist_ok=True)
os.makedirs(os.path.dirname(destgene), exist_ok=True)

# Load video
video = pims.PyAVReaderIndexed(inDir + Name)
nFrames = len(video)

# Limit to first 500 frames
nFrames = min(nFrames, 500)

# Number of frames per block
nFramesblock = math.floor(nFrames/nBlocks)

# Load background video
videobg = pims.PyAVReaderIndexed(inDirbg + Namebg)
nFramesbg = len(videobg)

# Background subtraction
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
    """Modified tracking function for 28 fish"""
    
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
        print(str(totobj) + " objects found in frame " + str(index+1))
        
        # Make an overlay image
        im_height, im_width = np.shape(mask_smooth)
        im_tmp = np.copy(mask_smooth)
        
        # Create a 3-channel overlay image for red dots
        overlay_img = np.zeros((im_height, im_width, 3), dtype=np.uint8)
        overlay_img[:, :, 0] = img  # Red channel
        overlay_img[:, :, 1] = img  # Green channel  
        overlay_img[:, :, 2] = img  # Blue channel
        
        # Make single dimensional arrays the size of nFish to fill with x,y coordinates
        y_cent = np.zeros(nFish)
        y_cent[:] = np.nan
        x_cent = np.zeros(nFish)
        x_cent[:] = np.nan
        
        for ind in range(0, totobj):
            # Find the centroid
            cent = props[ind].centroid
            
            # Skip objects outside the well areas
            if cent[0] < glassrow1:
                continue  
            if cent[0] > glassrow2 and cent[0] < glassrow3:
                continue   
            if cent[0] > glassrow4:
                continue        
            
            # Find the centroid
            cent = props[ind].centroid
            fishrow = 0
            fishcol = 0
            
            # Go through and get row and column of this fish
            for row in range(1, nRow+1):
                if cent[0] < row*nPixely/nRow:
                    fishrow = row
                    break
            
            for col in range(1, nCol+1):
                if cent[1] < col*nPixelx/nCol:
                    fishcol = col
                    break
            
            # This will return number of fish from 1 to nFish starting in top left corner
            fishnum = ((fishrow-1) * nCol) + fishcol
            
            # If there is already data in arrays for this fish number, skip
            if fishnum > nFish or fishnum < 1:
                continue
                
            if np.isfinite(y_cent[fishnum-1]):
                y_cent[fishnum-1] = np.nan
                x_cent[fishnum-1] = np.nan
                continue
            
            # Find the centroid of the largest component
            y_cent[fishnum-1], x_cent[fishnum-1] = cent
            
            # Set the cent point to 255 in this temporary image
            im_tmp[y_cent[fishnum-1].astype(int), x_cent[fishnum-1].astype(int)] = 255
            
            # Draw a big red dot at the fish coordinate on the overlay image
            center_x = int(x_cent[fishnum-1])
            center_y = int(y_cent[fishnum-1])
            # Draw a red circle with radius 8 pixels
            cv2.circle(overlay_img, (center_x, center_y), 8, (0, 0, 255), -1)  # (0, 0, 255) is red in BGR

    return im_tmp, y_cent, x_cent, overlay_img

# Initialize an array to hold the coordinates of the tracked points
cents = np.zeros((nFish, 2, nFrames))

# Tracking
print("Tracking 28 fish for first 500 frames...")

# Track each frame
for index, img in enumerate(tqdm(video)):
    if (index < nFrames):
        img = img[:,:,0]  # Convert to grayscale
        img = cv2.GaussianBlur(img.astype(np.float64), (3,3),0)
        
        # Call the tracking function
        timg, y_cent, x_cent, overlay_img = tracking(bg, img, index)
        
        cents[:, 0, index] = y_cent
        cents[:, 1, index] = x_cent

        # Save the tracked coordinates and the tracked image files
        if SaveTrackingIm:
            cv2.imwrite(inDir + 'data/input/tracked_frames/frame_' + str(index).zfill(3) + '_tracked.png', np.vstack((timg)))
            cv2.imwrite(inDir + 'data/input/tracked_frames/frame_' + str(index).zfill(3) + '_overlay.png', overlay_img)        

# Save tracking data arrays
print("Saving tracking data arrays....")
np.savez(inDir + 'data/output/tracking_results/TrackData_28fish_500frames.npz', cents=cents)

# Create CSV with fish locations for each frame
print("Creating CSV with fish locations...")

# Create a list to store all the data
data_rows = []

for frame in range(nFrames):
    row = {'frame': frame}
    
    # Add coordinates for each fish
    for fish_id in range(1, nFish + 1):
        if not np.isnan(cents[fish_id-1, 0, frame]) and not np.isnan(cents[fish_id-1, 1, frame]):
            row[f'fish_{fish_id}_x'] = cents[fish_id-1, 1, frame]  # x coordinate
            row[f'fish_{fish_id}_y'] = cents[fish_id-1, 0, frame]  # y coordinate
        else:
            row[f'fish_{fish_id}_x'] = np.nan
            row[f'fish_{fish_id}_y'] = np.nan
    
    data_rows.append(row)

# Create DataFrame and save to CSV
df = pd.DataFrame(data_rows)
csv_filename = inDir + 'data/output/tracking_results/Fish_Locations_28fish_500frames.csv'
df.to_csv(csv_filename, index=False)

print(f"CSV file saved as: {csv_filename}")
print(f"Shape: {df.shape} (rows: frames, columns: frame + 28 fish x 2 coordinates = 57 columns)")

# Also create a simplified version with just coordinates (28 columns for x, 28 for y)
print("Creating simplified CSV with 28 fish columns...")

# Create simplified data
simplified_data = []
for frame in range(nFrames):
    row = {'frame': frame}
    
    # Add x coordinates for all fish
    for fish_id in range(1, nFish + 1):
        if not np.isnan(cents[fish_id-1, 1, frame]):
            row[f'fish_{fish_id}'] = f"{cents[fish_id-1, 1, frame]:.2f},{cents[fish_id-1, 0, frame]:.2f}"
        else:
            row[f'fish_{fish_id}'] = "nan,nan"
    
    simplified_data.append(row)

# Create simplified DataFrame and save
df_simplified = pd.DataFrame(simplified_data)
csv_simplified_filename = inDir + 'data/output/tracking_results/Fish_Locations_28fish_500frames_simplified.csv'
df_simplified.to_csv(csv_simplified_filename, index=False)

print(f"Simplified CSV file saved as: {csv_simplified_filename}")
print(f"Shape: {df_simplified.shape} (rows: frames, columns: frame + 28 fish = 29 columns)")
print("Each fish column contains 'x,y' coordinates or 'nan,nan' if not detected")

print("Tracking complete!")
