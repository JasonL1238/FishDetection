# -*- coding: utf-8 -*-
"""
Background Subtraction Tracking Program for 28 Fish

This program uses background subtraction for fish tracking and is designed
to track 28 fish in a 4x7 grid layout.
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
import scipy.stats as stats
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn import metrics 
import statistics

# Import our background subtraction masker
from .background_subtraction_masker import BackgroundSubtractionMasker


def main():
    """Main tracking program for 28 fish using background subtraction."""
    
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
    
    # Number of fish to track - UPDATED TO 28
    nFish = 28
    
    # Number of rows and columns of wells - UPDATED FOR 28 FISH
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
    
    # Optional explicit edges for rows and columns.
    # If set to None, equal divisions are used based on nRow/nCol.
    # If set to a list, they must be monotonically increasing with length nRow+1 / nCol+1.
    ROW_EDGES_Y = None  # example: [0, 120, 270, 390, 544]
    COL_EDGES_X = None  # example: [0, 70, 150, 230, 305, 380, 445, 512]

    # Legacy glass row hints (kept for reference but not used when ROW_EDGES_Y is provided)
    glassrow1 = 42
    glassrow2 = 136  # 544/4 = 136
    glassrow3 = 230  # 136 + 94
    glassrow4 = 324  # 230 + 94
    glassrow5 = 418  # 324 + 94
    glassrow6 = 512  # 418 + 94

    # Build default equal-spacing edges if not provided
    if ROW_EDGES_Y is None:
        ROW_EDGES_Y = [int(round(i * (nPixely / nRow))) for i in range(nRow + 1)]
    if COL_EDGES_X is None:
        COL_EDGES_X = [int(round(i * (nPixelx / nCol))) for i in range(nCol + 1)]

    def get_row_index(y_val):
        for r in range(1, len(ROW_EDGES_Y)):
            if y_val < ROW_EDGES_Y[r]:
                return r
        return len(ROW_EDGES_Y) - 1

    def get_col_index(x_val):
        for c in range(1, len(COL_EDGES_X)):
            if x_val < COL_EDGES_X[c]:
                return c
        return len(COL_EDGES_X) - 1

    def get_row_center(row_idx):
        return 0.5 * (ROW_EDGES_Y[row_idx - 1] + ROW_EDGES_Y[row_idx])

    def get_col_center(col_idx):
        return 0.5 * (COL_EDGES_X[col_idx - 1] + COL_EDGES_X[col_idx])
    
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
    
    # Number of frames per block
    nFramesblock = math.floor(nFrames/nBlocks)
    
    # Load background video
    videobg = pims.PyAVReaderIndexed(inDirbg + Namebg)
    nFramesbg = len(videobg)
    
    # Initialize background subtraction masker
    bg_masker = BackgroundSubtractionMasker(
        threshold=25,
        blur_kernel_size=(3, 3),
        blur_sigma=0,
        min_object_size=25
    )
    
    # Create background model
    print("Creating background model...")
    bg_masker.create_background_model(inDirbg + Namebg)
    
    def tracking(bg_masker, img, index):
        """
        Track fish in a single frame using background subtraction.
        
        Args:
            bg_masker: Background subtraction masker instance
            img: Input frame
            index: Frame index
            
        Returns:
            tuple: (processed_image, y_centroids, x_centroids, overlay_image)
        """
        if index <= nFrames:
            # Apply background subtraction
            mask = bg_masker.apply_background_subtraction(img, bg_masker.background_model)
            
            # Smooth the mask
            mask_smooth = cv2.GaussianBlur(mask, (3, 3), 0)
            
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
            print(f"{totobj} objects detected in frame {index+1}")
            
            # Make an overlay image
            im_height, im_width = np.shape(mask_smooth)
            
            # Copy the smoothed mask into a temporary image
            im_tmp = np.copy(mask_smooth)
            
            # Create a 3-channel overlay image for red dots
            overlay_img = np.zeros((im_height, im_width, 3), dtype=np.uint8)
            # Convert grayscale to 3-channel by copying the grayscale to all channels
            overlay_img[:, :, 0] = img  # Red channel
            overlay_img[:, :, 1] = img  # Green channel  
            overlay_img[:, :, 2] = img  # Blue channel
            
            # Make single dimensional arrays the size of nFish to fill with x,y coordinates 
            # of the tracked points. Fill their initial entries with nans
            y_cent = np.zeros(nFish)
            y_cent[:] = np.nan
            x_cent = np.zeros(nFish)
            x_cent[:] = np.nan
            
            for ind in range(0, totobj):
                # Find the centroid
                cent = props[ind].centroid
                
                # Discard centroids outside configured bounds
                if cent[0] < ROW_EDGES_Y[0] or cent[0] > ROW_EDGES_Y[-1]:
                    continue
                if cent[1] < COL_EDGES_X[0] or cent[1] > COL_EDGES_X[-1]:
                    continue
                
                # Determine fish row and column
                fishrow = 0
                fishcol = 0
                
                # Map centroid to row/column using edges
                fishrow = get_row_index(cent[0])
                fishcol = get_col_index(cent[1])
                
                # This will return number of fish from 1 to nFish starting in top left corner
                fishnum = ((fishrow - 1) * nCol) + fishcol
                
                # Skip if fish number is out of range
                if fishnum > nFish or fishnum < 1:
                    continue
                
                # If there is already data in arrays for this fish number, skip
                if np.isfinite(y_cent[fishnum - 1]):
                    y_cent[fishnum - 1] = np.nan
                    x_cent[fishnum - 1] = np.nan
                    continue
                
                # Find the centroid of the largest component
                y_cent[fishnum - 1], x_cent[fishnum - 1] = cent
                
                # Set the cent point to 255 in this temporary image
                im_tmp[y_cent[fishnum - 1].astype(int), x_cent[fishnum - 1].astype(int)] = 255
                
                # Draw a big red dot at the fish coordinate on the overlay image
                center_x = int(x_cent[fishnum - 1])
                center_y = int(y_cent[fishnum - 1])
                # Draw a red circle with radius 8 pixels
                cv2.circle(overlay_img, (center_x, center_y), 8, (0, 0, 255), -1)  # (0, 0, 255) is red in BGR
            
            return im_tmp, y_cent, x_cent, overlay_img
    
    # Initialize an array to hold the coordinates of the tracked points
    cents = np.zeros((nFish, 2, nFrames))
    
    # Tracking
    print("Tracking....")
    
    # Process each frame
    for index, img in enumerate(tqdm(video)):
        if index < nFrames:
            img = img[:, :, 0]  # Convert to grayscale
            
            # Apply Gaussian blur
            img = cv2.GaussianBlur(img.astype(np.float64), (3, 3), 0)
            
            # Call the tracking function
            timg, y_cent, x_cent, overlay_img = tracking(bg_masker, img, index)
            
            # Store coordinates
            cents[:, 0, index] = y_cent
            cents[:, 1, index] = x_cent
            
            # Save the tracked coordinates and the tracked image files
            if SaveTrackingIm:
                cv2.imwrite(inDir + f'data/input/frames/frame_{index:03d}_tracked.png', np.vstack((timg)))
                cv2.imwrite(inDir + f'data/input/frames/frame_{index:03d}_overlay.png', overlay_img)
    
    # Save tracking data arrays
    print("Saving tracking data arrays....")
    np.savez(inDir + 'data/output/tracking_results/TrackData_28fish_bg_subtraction.npz', cents=cents)
    
    # Continue with analysis (same as original program but for 28 fish)
    print("Starting analysis...")
    
    # Import analysis functions from the original program
    from scipy.signal import medfilt
    
    # Setup med-filtered arrays
    medfiltcentY = np.zeros((nFish, nFrames))
    medfiltcentY[:, :] = np.nan
    medfiltcentX = np.zeros((nFish, nFrames))
    medfiltcentX[:, :] = np.nan
    
    for number in range(nFish):
        medfiltcentY[number, :] = medfilt(cents[number, 0, :], 3)
        medfiltcentX[number, :] = medfilt(cents[number, 1, :], 3)
    
    # Initialize analysis arrays (same structure as original but for 28 fish)
    disp = np.zeros((nFish, nFrames - 1))
    disp[:, :] = np.nan
    filtdisp = np.zeros((nFish, nFrames - 1))
    filtdisp[:, :] = np.nan
    
    thigmo = np.zeros((nFish, nFrames))
    thigmo[:, :] = np.nan
    filtthigmo = np.zeros((nFish, nFrames))
    filtthigmo[:, :] = np.nan
    
    centeryesno = np.zeros((nFish, nFrames))
    centeryesno[:, :] = np.nan
    topyesno = np.zeros((nFish, nFrames))
    topyesno[:, :] = np.nan
    distfrommiddle = np.zeros((nFish, nFrames))
    distfrommiddle[:, :] = np.nan
    
    movementyesno = np.zeros((nFish, nFrames - 1))
    movementyesno[:, :] = np.nan
    movementyesnofill = np.zeros((nFish, nFrames - 1))
    movementyesnofill[:, :] = np.nan
    
    # Block and period analysis arrays
    boutsperblock = np.zeros((nFish, nBlocks))
    boutsperblock[:, :] = np.nan
    totdistperblock = np.zeros((nFish, nBlocks))
    totdistperblock[:, :] = np.nan
    totdispperblock = np.zeros((nFish, nBlocks))
    totdispperblock[:, :] = np.nan
    tottimemovingperblock = np.zeros((nFish, nBlocks))
    tottimemovingperblock[:, :] = np.nan
    totavgspeedperblock = np.zeros((nFish, nBlocks))
    totavgspeedperblock[:, :] = np.nan
    avgthigmoperblock = np.zeros((nFish, nBlocks))
    avgthigmoperblock[:, :] = np.nan
    medthigmoperblock = np.zeros((nFish, nBlocks))
    medthigmoperblock[:, :] = np.nan
    fractionouterperblock = np.zeros((nFish, nBlocks))
    fractionouterperblock[:, :] = np.nan
    
    # Period analysis arrays
    boutsperperiod = np.zeros((nFish, nExpts))
    boutsperperiod[:, :] = np.nan
    totdistperperiod = np.zeros((nFish, nExpts))
    totdistperperiod[:, :] = np.nan
    totdispperperiod = np.zeros((nFish, nExpts))
    totdispperperiod[:, :] = np.nan
    tottimemovingperperiod = np.zeros((nFish, nExpts))
    tottimemovingperperiod[:, :] = np.nan
    totavgspeedperperiod = np.zeros((nFish, nExpts))
    totavgspeedperperiod[:, :] = np.nan
    avgthigmoperperiod = np.zeros((nFish, nExpts))
    avgthigmoperperiod[:, :] = np.nan
    medthigmoperperiod = np.zeros((nFish, nExpts))
    medthigmoperperiod[:, :] = np.nan
    fractionouterperperiod = np.zeros((nFish, nExpts))
    fractionouterperperiod[:, :] = np.nan
    
    print("Calculating displacement and thigmotaxis....")
    
    # Calculate displacement and thigmotaxis for each fish
    for number in range(nFish):
        i = 0
        while i < nFrames:
            block = math.floor(i / nFramesblock)
            
            # Fill in missing centroids with median of neighbors
            if np.isnan(cents[number, 0, i]):
                cents[number, 0, i] = np.nanmedian(cents[number, 0, i-1:i+2])
                cents[number, 1, i] = np.nanmedian(cents[number, 1, i-1:i+2])
            
            # Calculate thigmotaxis for each frame
            row = math.ceil((number + 1) / nCol)
            col = (number + 1) - (nCol * (row - 1))
            ycenter = get_row_center(row)
            xcenter = get_col_center(col)
            thigmo[number, i] = np.sqrt((cents[number, 0, i] - ycenter)**2 + (cents[number, 1, i] - xcenter)**2)
            filtthigmo[number, i] = np.sqrt((medfiltcentY[number, i] - ycenter)**2 + (medfiltcentX[number, i] - xcenter)**2)
            
            # Check if fish is near edges using configured edges
            xwall1 = COL_EDGES_X[col - 1]
            xwall2 = COL_EDGES_X[col]
            ywall1 = ROW_EDGES_Y[row - 1] + 43
            ywall2 = ROW_EDGES_Y[row] - 43
            
            if (cents[number, 0, i] < (ywall1 + 25) or cents[number, 0, i] > (ywall2 - 25) or 
                cents[number, 1, i] < (xwall1 + 25) or cents[number, 1, i] > (xwall2 - 25)):
                centeryesno[number, i] = 1
            else:
                centeryesno[number, i] = 0
            
            # Calculate displacement between frames
            if i > 0:
                disp[number, i - 1] = np.sqrt((cents[number, 0, i] - cents[number, 0, i - 1])**2 + 
                                            (cents[number, 1, i] - cents[number, 1, i - 1])**2)
                filtdisp[number, i - 1] = np.sqrt((medfiltcentY[number, i] - medfiltcentY[number, i - 1])**2 + 
                                                 (medfiltcentX[number, i] - medfiltcentX[number, i - 1])**2)
                
                # Define movement as displacement > 0.5 pixels
                if disp[number, i - 1] > 0.5:
                    movementyesno[number, i - 1] = 1
                else:
                    movementyesno[number, i - 1] = 0
            
            i += 1
    
    print("Analysis completed!")
    print(f"Successfully tracked {nFish} fish using background subtraction method")
    
    # Save analyzed data
    print("Saving analyzed data...")
    np.savez(inDir + 'data/output/tracking_results/AnalyzedData_28fish_bg_subtraction.npz',
             medfiltcentY=medfiltcentY, medfiltcentX=medfiltcentX, disp=disp, filtdisp=filtdisp,
             thigmo=thigmo, filtthigmo=filtthigmo, centeryesno=centeryesno,
             movementyesno=movementyesno, movementyesnofill=movementyesnofill,
             boutsperblock=boutsperblock, totdistperblock=totdistperblock,
             totdispperblock=totdispperblock, tottimemovingperblock=tottimemovingperblock,
             totavgspeedperblock=totavgspeedperblock, avgthigmoperblock=avgthigmoperblock,
             medthigmoperblock=medthigmoperblock, fractionouterperblock=fractionouterperblock,
             boutsperperiod=boutsperperiod, totdistperperiod=totdistperperiod,
             totdispperperiod=totdispperperiod, tottimemovingperperiod=tottimemovingperperiod,
             totavgspeedperperiod=totavgspeedperperiod, avgthigmoperperiod=avgthigmoperperiod,
             medthigmoperperiod=medthigmoperperiod, fractionouterperperiod=fractionouterperperiod)


if __name__ == "__main__":
    main()
