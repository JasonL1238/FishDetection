# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 21:23:34 2019

@author: phili_000
"""

import numpy as np
import cv2
#from TrackedObject import TrackedObject as TO
import matplotlib.pyplot as plt
import os
import re #regular expressions library
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

#import cv2

#image1 = cv2.imread('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/background.tif', 0)
#image2 = cv2.imread('F://Granato Lab/Test video/slice.tif', 0)
#bs = abs(image2 - image1)

#print(image1)



#set directory containing image files
inDir = "/Users/philcampbell/Downloads/"

# names of videos to track
Name = 'Clutch1_20250804_122715.mp4'

#Gene name
Gene = "test"

#set directory containing background files
inDirbg = "/Users/philcampbell/Downloads/"

# names of videos for background
Namebg = 'Clutch1_20250804_122715.mp4'


#set the directory in which to save the files
SaveTrackingIm = True; # or False

desttracked = inDir + 'Results/TrackedVideos/' + Name + '/trackedimages/'
desttracking = inDir + 'Results/TrackedVideos/' + Name + '/'
destgene = inDir + 'Results/' + Gene + '/' + Name + '/'

# Number of fish to track
nFish = 14

#Number of rows and columns of wells
nRow = 2
nCol = 7

#Number of pixels in x and y direction
nPixely = 544
nPixelx = 512

#frame rate in Hz
framerate = 20

# time between frames in milliseconds
framelength = 1/framerate * 1000

# Define block time in seconds
blocksec = 60

#this define middle of each lane on top and bottom columns
ycutofftop1=130
ycutofftop2=388
#ycutoffbottom=301

yvaluemiddletop=210
yvaluemiddlebottom=308

#Define pixels
glassrow1=42
glassrow2=214
glassrow3=304
glassrow4=478



# Define number of total blocks
nBlocks = 20

# Define number of total expts
nExpts = 2

#this is to say if there is no folder, can make one
os.makedirs(os.path.dirname(desttracked), exist_ok=True)
os.makedirs(os.path.dirname(desttracking), exist_ok=True)
os.makedirs(os.path.dirname(destgene), exist_ok=True)


#pims.open('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/CaSR22_09A_.avi')
video = pims.PyAVReaderIndexed(inDir + Name)
#video2 = pims.Video('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/CaSR22_09A_.avi')

nFrames = len(video)


# number of frames per block
nFramesblock = math.floor(nFrames/nBlocks)


#pims.open('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/CaSR22_09A_.avi')
videobg = pims.PyAVReaderIndexed(inDirbg + Namebg)
#video2 = pims.Video('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/CaSR22_09A_.avi')

nFramesbg = len(videobg)

#background subtraction


# #first 5 frames should be used to create background
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

# fr1 = videobg[np.round(23000)][:,:,0]
# fr2 = videobg[np.round(12000)][:,:,0]
# fr3 = videobg[np.round(13000)][:,:,0]
# fr4 = videobg[np.round(14000)][:,:,0]
# fr5 = videobg[np.round(15000)][:,:,0]
# fr6 = videobg[np.round(16000)][:,:,0]
# fr7 = videobg[np.round(17000)][:,:,0]
# fr8 = videobg[np.round(18000)][:,:,0]
# fr9 = videobg[np.round(19000)][:,:,0]
# fr10 = videobg[np.round(20000)][:,:,0]
# fr11 = videobg[np.round(21000)][:,:,0]
# fr12 = videobg[np.round(22000)][:,:,0]

# bg = np.median(np.stack((fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10, fr11, fr12), axis=2), axis=2)

# bg2 = cv2.GaussianBlur(bg.astype(np.float64), (3,3),0)

# fr1 = videobg[np.round(34000)][:,:,0]
# fr2 = videobg[np.round(24000)][:,:,0]
# fr3 = videobg[np.round(33000)][:,:,0]
# fr4 = videobg[np.round(24000)][:,:,0]
# fr5 = videobg[np.round(25000)][:,:,0]
# fr6 = videobg[np.round(26000)][:,:,0]
# fr7 = videobg[np.round(27000)][:,:,0]
# fr8 = videobg[np.round(28000)][:,:,0]
# fr9 = videobg[np.round(29000)][:,:,0]
# fr10 = videobg[np.round(30000)][:,:,0]
# fr11 = videobg[np.round(31000)][:,:,0]
# fr12 = videobg[np.round(32000)][:,:,0]

# bg = np.median(np.stack((fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10, fr11, fr12), axis=2), axis=2)

# bg3 = cv2.GaussianBlur(bg.astype(np.float64), (3,3),0)

# fr1 = videobg[np.round(46000)][:,:,0]
# fr2 = videobg[np.round(44000)][:,:,0]
# fr3 = videobg[np.round(45000)][:,:,0]
# fr4 = videobg[np.round(43000)][:,:,0]
# fr5 = videobg[np.round(35000)][:,:,0]
# fr6 = videobg[np.round(36000)][:,:,0]
# fr7 = videobg[np.round(37000)][:,:,0]
# fr8 = videobg[np.round(38000)][:,:,0]
# fr9 = videobg[np.round(39000)][:,:,0]
# fr10 = videobg[np.round(40000)][:,:,0]
# fr11 = videobg[np.round(41000)][:,:,0]
# fr12 = videobg[np.round(42000)][:,:,0]

# bg = np.median(np.stack((fr1, fr2, fr3, fr4, fr5, fr6, fr7, fr8, fr9, fr10, fr11, fr12), axis=2), axis=2)

# bg4 = cv2.GaussianBlur(bg.astype(np.float64), (3,3),0)


def tracking(bg, img, index):

#Define some variables that are used to track the points along the back
    
    if index<=nFrames:
    #Do a background subtraction, and blur the background
        mask = abs(bg1 - img)
    #mask[mask < 0] = 0


    # if index>=36000:
    # #Do a background subtraction, and blur the background
    #     mask = abs(bg4 - img)    

    # if index>=12000 and index<24000:
    # #Do a background subtraction, and blur the background
    #     mask = abs(bg2 - img)    

    # if index>=24000 and index<36000:
    # #Do a background subtraction, and blur the background
    #     mask = abs(bg3 - img)    
    
    
    # if index>=60000 and index<72000:
    # #Do a background subtraction, and blur the background
    #     mask = abs(bg6 - img)    

    
    # if index>=72000 and index<84000:
    # #Do a background subtraction, and blur the background
    #     mask = abs(bg7 - img)       

    # if index>=84000 and index<=96000:
    # #Do a background subtraction, and blur the background
    #     mask = abs(bg8 - img)   

    # if index>96000:
    # #Do a background subtraction, and blur the background
    #     mask = abs(bg8 - img)   

    # if index>=96000 and index<109000:
    # #Do a background subtraction, and blur the background
    #     mask = abs(bg9 - img)       
        
    # if index>=84000:
    # #Do a background subtraction, and blur the background
    #     mask = abs(bg7 - img)    
    # #Do I need to do a gaussian blur?

    mask_smooth = cv2.GaussianBlur(mask, (3,3),0)

    #min_val = -50.0
    #img_crop = np.where(mask < min_val, img_crop, np.zeros_like(img_crop))
    
    #threshold the mask
    #th_img = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
    th_img = cv2.threshold(mask_smooth, 25, 255, cv2.THRESH_BINARY)[1]
    
    #label the connected components in the thresholded mask
    lb = label(th_img)
    
    #remove small objects from thresholded image
    lb_sorem = remove_small_objects(lb, 25)
    
    #re-label since small objects removes so get 1 to nFish
    lb2 = label(lb_sorem)



    #get statistics on these components - takes a labeled input image and returns
    # a list of RegionProperties - area, centroid among others
    props = regionprops(lb2)
    
    totobj = np.amax(lb2)
    print(str(totobj) + "in" + str(index+1))
    #if np.max(props[:].label)!=nFish-1:
        #print("number of fish does not equal number of objects identified")
        #print("image number " + str(index))
        #quit()
    
    #Make an overlay image so we can save them to inspect where the tracked points
    #are getting placed on the image
    #overlay = img
    im_height, im_width = np.shape(mask_smooth)
    
    #copy the smoothed mask into a temporary image
    im_tmp = np.copy(mask_smooth)
    
    #make single dimensional arrays the size of nSeg to fill with x,y coordinates 
        #of the tracked points. Fill their initial entries with nans
        #medInt = np.median(mask_smooth)
    
    y_cent = np.zeros(nFish)
    y_cent[:] = np.nan
    x_cent = np.zeros(nFish)
    x_cent[:] = np.nan
    
    for ind in range(0, totobj):
        #if totobj > nFish:
        #    break
        #if max(max_row - min_row, max_col - min_col) >30:
        #    continue
      
#find the centroid of        
        cent = props[ind].centroid

        if cent[0] < glassrow1:
            continue  

        if cent[0] > glassrow2 and cent[0]<glassrow3:
            continue   

        if cent[0] > glassrow4:
            continue        

        if cent[0] > glassrow2 and cent[0]<glassrow3:
            continue
        
#find the centroid of        
        cent = props[ind].centroid
        fishrow=0
        fishcol=0
        #go through and get row and column of this fish so can use it to index array; fish numbered from 1 in top left to 36 in bottorm right
        for row in range(1, nRow+1):

            if cent[0] < row*nPixely/nRow:
                fishrow = row
                # print("x" + str(cent[0])+"y" + str(cent[1])+"test" + str(row*nPixely/nRow))
                break
        
        for col in range(1, nCol+1):
            if cent[1] < col*nPixely/nCol:
                fishcol = col
                # print("x" + str(cent[0])+"y" + str(cent[1])+"test" + str(col*nPixely/nCol))
                break
        
        #this will return number of fish from 1 to nFish starting in top left corner with 1 and going L to R
        fishnum = ((fishrow-1) * nCol) + fishcol
        
        # print(str(fishnum) + "row" + str(fishrow) + "col" + str(fishcol) + "x" + str(cent[0])+"y" + str(cent[1]))
        
        #if there is already data in arrays for this fish number, go to top of for loop for next object, put 3 if statements so will remove if up to 4 objects      
        if np.isfinite(y_cent[fishnum-1]):
            y_cent[fishnum-1] = np.nan
            x_cent[fishnum-1] = np.nan
            continue
        
        #find the centroid of the largest component
        y_cent[fishnum-1], x_cent[fishnum-1] = cent

      
        #set the cent point to 0 in this temporary image
         #   im_tmp[y_0.astype(int), x_0.astype(int)] = 0;
        im_tmp[y_cent[fishnum-1].astype(int), x_cent[fishnum-1].astype(int)] = 255;



    return im_tmp, y_cent, x_cent





#Initialize an array to hold the coordinates of the 8 tracked points (x and y coords)
    #for each tracked image. We want the array to be size 8, 2, page_num, where nFrames
    #is the number of images we are tracking. We will assign y coordinates as the first column
    #index i.e. top_coords[:,0,:] and x coordinates as the second index i.e. top_coords[:,1,:]
    #This is to be consistent with all of the other code used for this project
cents = np.zeros((nFish, 2, nFrames))

#tracking

print("Tracking....")
    
#tqdm shows a progress bar around any iterable
for index, img in enumerate(tqdm(video)):
       
        if (index < nFrames):
    #.copy() will maintain the origical list unchanged so that if you change the new list it
    #won't affect old one
            #img = plt.imread(inDir + "\\" + filename).copy()
            ind = index
            
            img = img[:,:,0] # its actually a black and white image, but gets read in as three channel. This is probably inefficient at some point
            
            # do I need to do gaussian blur?
            img = cv2.GaussianBlur(img.astype(np.float64), (3,3),0)
            
             # img = img[0:800,0:1075]
            
            #calling the tracking image with bg and individual img
            timg, y_cent, x_cent = tracking(bg, img, ind)
            
            cents[:, 0, ind] = y_cent
            cents[:, 1, ind] = x_cent

            #save the tracked coordinates and the tracked image files
            #scipy.io.savemat(dest)
            if SaveTrackingIm:
                
                #scipy.misc.imsave - (name of output file, array containing image values,format=image format unless
                #specified in name of file)
                #np.vstack - Stack arrays in sequence vertically (row wise)
                #timg here is the im_tmp object returned from tracking loop
                cv2.imwrite(inDir + 'Results/TrackedVideos/' + Name + '/trackedimages/' + Name + '_' + str(index) +'.png', np.vstack((timg)))        
        
    #np. savez - Save several arrays into a single file in uncompressed .npz format.
    # (file name, designed array names)
print("Saving tracking data arrays....")
np.savez(inDir + 'Results/TrackedVideos/' + Name + '/TrackData.npz', cents=cents)

        
import seaborn as sns
from scipy.signal import medfilt

# START CALCULATIONS HERE

#setup med-filtered arrays
medfiltcentY = np.zeros((nFish, nFrames))
medfiltcentY[:,:] = np.nan
medfiltcentX = np.zeros((nFish, nFrames))
medfiltcentX[:,:] = np.nan


for number in range(nFish):
        medfiltcentY[number,:] = medfilt(cents[number,0,:],3)
        medfiltcentX[number,:] = medfilt(cents[number,1,:],3)
      
        
# per block: #bouts, totdistance, totaldisplacement, tottimemoving, average speed, average from center, median from center, fraction time in outer
# per total period (light/dark): same as above
        # per bout, for each block: average and median distance, time moving, displacement, speed
        # per bout, for each period: average and median distance, time moving, displacement, speed

disp = np.zeros((nFish, nFrames-1))
disp[:,:] = np.nan
filtdisp = np.zeros((nFish, nFrames-1))
filtdisp[:,:] = np.nan


thigmo = np.zeros((nFish, nFrames))
thigmo[:,:] = np.nan
filtthigmo = np.zeros((nFish, nFrames))
filtthigmo[:,:] = np.nan

centeryesno = np.zeros((nFish, nFrames))
centeryesno[:,:] = np.nan
topyesno = np.zeros((nFish, nFrames))
topyesno[:,:] = np.nan
distfrommiddle = np.zeros((nFish, nFrames))
distfrommiddle[:,:] = np.nan


movementyesno = np.zeros((nFish, nFrames-1))
movementyesno[:,:] = np.nan
movementyesnofill = np.zeros((nFish, nFrames-1))
movementyesnofill[:,:] = np.nan


boutsperblock= np.zeros((nFish, nBlocks))
boutsperblock[:,:] = np.nan
totdistperblock = np.zeros((nFish, nBlocks))
totdistperblock[:,:] = np.nan

totdisttopperblock = np.zeros((nFish, nBlocks))
totdisttopperblock[:,:] = np.nan

totdispperblock = np.zeros((nFish, nBlocks))
totdispperblock[:,:] = np.nan
tottimemovingperblock = np.zeros((nFish, nBlocks))
tottimemovingperblock[:,:] = np.nan
totavgspeedperblock = np.zeros((nFish, nBlocks))
totavgspeedperblock[:,:] = np.nan
avgthigmoperblock = np.zeros((nFish, nBlocks))
avgthigmoperblock[:,:] = np.nan
medthigmoperblock = np.zeros((nFish, nBlocks))
medthigmoperblock[:,:] = np.nan
fractionouterperblock = np.zeros((nFish, nBlocks))
fractionouterperblock[:,:] = np.nan
fractiondisttopperblock = np.zeros((nFish, nBlocks))
fractiondisttopperblock[:,:] = np.nan
fractiontopperblock = np.zeros((nFish, nBlocks))
fractiontopperblock[:,:] = np.nan
avgdistfrommiddleperblock = np.zeros((nFish, nBlocks))
avgdistfrommiddleperblock[:,:] = np.nan
meddistfrommiddleperblock = np.zeros((nFish, nBlocks))
meddistfrommiddleperblock[:,:] = np.nan
transitionstotopperblock = np.zeros((nFish, nBlocks))
transitionstotopperblock[:,:] = np.nan

boutsperperiod= np.zeros((nFish, nExpts))
boutsperperiod[:,:] = np.nan
totdistperperiod = np.zeros((nFish, nExpts))
totdistperperiod[:,:] = np.nan
totdispperperiod = np.zeros((nFish, nExpts))
totdispperperiod[:,:] = np.nan
tottimemovingperperiod = np.zeros((nFish, nExpts))
tottimemovingperperiod[:,:] = np.nan
totavgspeedperperiod = np.zeros((nFish, nExpts))
totavgspeedperperiod[:,:] = np.nan
avgthigmoperperiod = np.zeros((nFish, nExpts))
avgthigmoperperiod[:,:] = np.nan
medthigmoperperiod = np.zeros((nFish, nExpts))
medthigmoperperiod[:,:] = np.nan
fractionouterperperiod = np.zeros((nFish, nExpts))
fractionouterperperiod[:,:] = np.nan
fractiontopperperiod = np.zeros((nFish, nExpts))
fractiontopperperiod[:,:] = np.nan
fractiondisttopperperiod = np.zeros((nFish, nExpts))
fractiondisttopperperiod[:,:] = np.nan

avgdistperboutperblock = np.zeros((nFish, nBlocks))
avgdistperboutperblock[:,:] = np.nan
avgdispperboutperblock = np.zeros((nFish, nBlocks))
avgdispperboutperblock[:,:] = np.nan
avgtimemovingperboutperblock = np.zeros((nFish, nBlocks))
avgtimemovingperboutperblock[:,:] = np.nan
avgspeedperboutperblock = np.zeros((nFish, nBlocks))
avgspeedperboutperblock[:,:] = np.nan

meddistperboutperblock = np.zeros((nFish, nBlocks))
meddistperboutperblock[:,:] = np.nan
meddispperboutperblock = np.zeros((nFish, nBlocks))
meddispperboutperblock[:,:] = np.nan
medtimemovingperboutperblock = np.zeros((nFish, nBlocks))
medtimemovingperboutperblock[:,:] = np.nan
medspeedperboutperblock = np.zeros((nFish, nBlocks))
medspeedperboutperblock[:,:] = np.nan

avgdistperboutperperiod = np.zeros((nFish, nExpts))
avgdistperboutperperiod[:,:] = np.nan
avgdispperboutperperiod = np.zeros((nFish, nExpts))
avgdispperboutperperiod[:,:] = np.nan
avgtimemovingperboutperperiod = np.zeros((nFish, nExpts))
avgtimemovingperboutperperiod[:,:] = np.nan
avgspeedperboutperperiod = np.zeros((nFish, nExpts))
avgspeedperboutperperiod[:,:] = np.nan

meddistperboutperperiod = np.zeros((nFish, nExpts))
meddistperboutperperiod[:,:] = np.nan
meddispperboutperperiod = np.zeros((nFish, nExpts))
meddispperboutperperiod[:,:] = np.nan
medtimemovingperboutperperiod = np.zeros((nFish, nExpts))
medtimemovingperboutperperiod[:,:] = np.nan
medspeedperboutperperiod = np.zeros((nFish, nExpts))
medspeedperboutperperiod[:,:] = np.nan

avgdistfrommiddleperperiod = np.zeros((nFish, nExpts))
avgdistfrommiddleperperiod[:,:] = np.nan
meddistfrommiddleperperiod = np.zeros((nFish, nExpts))
meddistfrommiddleperperiod[:,:] = np.nan
transitionstotopperperiod = np.zeros((nFish, nExpts))
transitionstotopperperiod[:,:] = np.nan

distbouts = np.zeros((nFish, nBlocks,1000))
distbouts[:,:,:] = np.nan
dispbouts = np.zeros((nFish, nBlocks,1000))
dispbouts[:,:,:] = np.nan
timebouts = np.zeros((nFish, nBlocks,1000))
timebouts[:,:,:] = np.nan
speedbouts = np.zeros((nFish, nBlocks,1000))
speedbouts[:,:,:] = np.nan

centsorig = np.copy(cents)

print("Calculating displacement and thigmotaxis....")
# calculating displacement and thigmotaxis
for number in range(nFish):
    i = 0
    while i < nFrames:
        block = math.floor(i/nFramesblock)
        if np.isnan(cents[number,0,i]):
            cents[number,0,i] = np.nanmedian(cents[number,0,i-1:i+2])
            cents[number,1,i] = np.nanmedian(cents[number,1,i-1:i+2])

        
        # if  np.isnan(cents[number, 0, (block*nFramesblock):((block+1)*nFramesblock)]).any():
        #     i=(block+1)*nFramesblock
        #     #if no centroid is defined in any frame in this block, skip to next block
        #     continue

           
            #calculate thigmotaxis for each frame
            
            
        row = math.ceil((number+1)/nCol)
        col = (number+1) - (nCol*(row-1))
        ycenter = (nPixely/nRow)/2 + ((row - 1)*(nPixely/nRow))
        xcenter = (nPixelx/nCol)/2 + ((col - 1)*(nPixelx/nCol))
        thigmo[number,i] = np.sqrt((cents[number, 0, i]-ycenter)**2 + (cents[number, 1, i]-xcenter)**2)
        filtthigmo[number,i] = np.sqrt((medfiltcentY[number, i]-ycenter)**2 + (medfiltcentX[number, i]-xcenter)**2)
            
        xwall1 = ((col - 1)*(nPixelx/nCol))
        xwall2 = ((col)*(nPixelx/nCol))
        ywall1 = ((row - 1)*(nPixely/nRow)) + 43
        ywall2 = ((row)*(nPixely/nRow)) - 43
            
        if cents[number, 0, i] < (ywall1+25) or cents[number, 0, i] > (ywall2-25) or cents[number, 1, i] < (xwall1+25) or cents[number, 1, i]> (xwall2-25):
            centeryesno[number,i] = 1
        else:
            centeryesno[number,i] = 0
            
            #if thigmo[number,i]>30:
            #    centeryesno[number,i] = 1
            #if thigmo[number,i]<=30:
            #    centeryesno[number,i] = 0

        if number<7:
            distfrommiddle[number,i] = yvaluemiddletop - cents[number, 0, i]
            if cents[number, 0, i]<=ycutofftop1:
                topyesno[number,i] = 0
            if cents[number, 0, i]>ycutofftop1:
                topyesno[number,i] = 1
        if number>=7:
            distfrommiddle[number,i] = cents[number, 0, i] - yvaluemiddlebottom     
            if cents[number, 0, i]<=ycutofftop2:
                topyesno[number,i] = 1
            if cents[number, 0, i]>ycutofftop2:
                topyesno[number,i] = 0        

                
            #calculate displacement between each frame
        if i>0:
            disp[number,i-1] = np.sqrt((cents[number, 0, i]-cents[number, 0, i-1])**2 + (cents[number, 1, i]-cents[number, 1, i-1])**2) 
            filtdisp[number,i-1] = np.sqrt((medfiltcentY[number, i]-medfiltcentY[number, i-1])**2 + (medfiltcentX[number, i]-medfiltcentX[number, i-1])**2) 
                   # define movement as displacement >0.2 and populate array with 1 for >0.2 and 0 for <0.2; nan if displ nan
            if disp[number,i-1] > 0.5:
                movementyesno[number,i-1] = 1
            else:
                movementyesno[number,i-1] = 0
            
        i=i+1


print("Defining movement....")
#Attempt to fill holes of movement array. Will eliminate singleton 1s or 0s so that for movement will need >1 frame with value of
                       #1 and to stop movement will need >1 frame with value of 0
# fill initial frame with values of initial array
movementyesnofill[:, 0] = movementyesno[:, 0]

for number in range(nFish):
    for i in range(1,nFrames-1):
        if  np.isnan(movementyesno[number, i]):
            continue
        #if nan either before or after value i, just keep value because cannot tell
        if  np.isnan(movementyesnofill[number, i-1]):
            movementyesnofill[number, i] = movementyesno[number, i]
            continue
        if i < nFrames-2:
            if np.isnan(movementyesno[number, i+1]):
                movementyesnofill[number, i] = movementyesno[number, i]
                continue


# here is where we define movement and stopping as >1 frame of either 1 or 0
        if movementyesno[number, i] == 0:
            if movementyesnofill[number, i-1] == 0:
                movementyesnofill[number, i] = 0
            if movementyesnofill[number, i-1] == 1:
                if i < nFrames-2:
                    if movementyesno[number, i+1] == 1:
                        movementyesnofill[number, i] = 1
                    if movementyesno[number, i+1] == 0:
                        movementyesnofill[number, i] = 0
                else:
                    movementyesnofill[number, i] = 0
        
        if movementyesno[number, i] == 1:
            if movementyesnofill[number, i-1] == 1:
                movementyesnofill[number, i] = 1
            if movementyesnofill[number, i-1] == 0:
                if i < nFrames-2:
                    if movementyesno[number, i+1] == 0:
                        movementyesnofill[number, i] = 0
                    if movementyesno[number, i+1] == 1:
                        movementyesnofill[number, i] = 1
                else:
                    movementyesnofill[number, i] = 1


print("Calculating values per block and per period")
for number in range(nFish):
    i=0
    while i < nBlocks:
        #check if no cent values anywhere, if so, skip this block
        if  np.isnan(cents[number, 0, (i*nFramesblock):((i+1)*nFramesblock)]).any():
            i=i+1
            #if no centroid is defined in any frame in this block, skip to next block
            continue        

        else:
            #label bouts in block
            lb = label(movementyesnofill[number, (i*nFramesblock):(((i+1)*nFramesblock)-1)])
            
            #remove bouts less than 2 - though shouldn't be any anyways
            lb_sorem = remove_small_objects(lb, 2)
            
            #re-label since small objects removed
            lb2 = label(lb_sorem)
            
            # Number of bouts for this block
            boutsperblock[number,i] = np.amax(lb2)
            
            #define array of distances in this block
            x = disp[number, (i*nFramesblock):(((i+1)*nFramesblock)-1)]
            
            #total distance traveled during periods defined as movements
            totdistperblock[number, i] = np.sum(x[lb2!=0])
            
            #define array of topyesno for this block
            y = topyesno[number, (i*nFramesblock):(((i+1)*nFramesblock)-1)]
            
            #total distance traveled in top during this block
            totdisttopperblock[number, i] = np.sum(x[(lb2!=0)&(y!=0)])

            #fraction of distance traveled in top
            fractiondisttopperblock[number, i] = totdisttopperblock[number, i]/totdistperblock[number, i]
   
            #total displacement for this block
            totdispperblock[number,i] = np.sqrt((cents[number, 0, (i*nFramesblock)]-cents[number, 0, (((i+1)*nFramesblock)-1)])**2 + (cents[number, 1, (i*nFramesblock)]-cents[number, 1, (((i+1)*nFramesblock)-1)])**2)
          
            #time moving for this block in ms
            tottimemovingperblock[number,i] = np.sum(movementyesnofill[number,(i*nFramesblock):(((i+1)*nFramesblock)-1)])*framelength
            
            #average speed for entire block pix/s
            totavgspeedperblock[number,i] = (totdistperblock[number, i]/tottimemovingperblock[number,i])*1000
         
            #average and median distance from center for entire block
            avgthigmoperblock[number,i] = np.nanmean(thigmo[number, (i*nFramesblock):((i+1)*nFramesblock)])
            medthigmoperblock[number,i] = np.nanmedian(thigmo[number, (i*nFramesblock):((i+1)*nFramesblock)])
            
            # fraction of time in outer part of well vs center
            fractionouterperblock[number,i] = np.sum(centeryesno[number, (i*nFramesblock):((i+1)*nFramesblock)])/nFramesblock
           
            # fraction of time in upper part
            fractiontopperblock[number,i] = np.sum(topyesno[number, (i*nFramesblock):((i+1)*nFramesblock)])/nFramesblock

            # Number of transitions to top for this block
            lbentry = label(topyesno[number, (i*nFramesblock):((i+1)*nFramesblock)])
            transitionstotopperblock[number,i] = np.amax(lbentry)

            #average
            avgdistfrommiddleperblock[number,i] = np.mean(distfrommiddle[number, (i*nFramesblock):((i+1)*nFramesblock)])
            meddistfrommiddleperblock[number,i] = np.median(distfrommiddle[number, (i*nFramesblock):((i+1)*nFramesblock)])


            #calulating values per bout
            
            index = 0

            while index < int(boutsperblock[number,i]):
                #distance travelled of this bout
                distbouts[number,i,index] = np.sum(x[lb2==(index+1)])
                
                #define first and last frame of this bout
                first = int(np.where(lb2==(index+1))[0][0])
                last = int(np.where(lb2==(index+1))[0][-1]+1)
                
                # displacement of this bout
                dispbouts[number,i,index] = np.sqrt((cents[number, 0, ((i*nFramesblock)+first)]-cents[number, 0, ((i*nFramesblock)+last)])**2 + (cents[number, 1, ((i*nFramesblock)+first)]-cents[number, 1, ((i*nFramesblock)+last)])**2)
                
                # time moving of this bout
                timebouts[number,i,index] = np.count_nonzero(lb2 == (index+1))*framelength
                
                #average speed of this bout pix/s
                speedbouts[number,i,index] = (distbouts[number,i,index]/timebouts[number,i,index])*1000
                
                index = index+1         
            
            
            #calculating perbout, per block values
                        
            avgdistperboutperblock[number, i] = np.nanmean(distbouts[number,i,:])
            meddistperboutperblock[number, i] = np.nanmedian(distbouts[number,i,:])
            
            avgdispperboutperblock[number, i] = np.nanmean(dispbouts[number,i,:])
            meddispperboutperblock[number, i] = np.nanmedian(dispbouts[number,i,:])
            
            avgtimemovingperboutperblock[number, i] = np.nanmean(timebouts[number,i,:])
            medtimemovingperboutperblock[number, i] = np.nanmedian(timebouts[number,i,:])
            
            avgspeedperboutperblock[number, i] = np.nanmean(speedbouts[number,i,:])
            medspeedperboutperblock[number, i] = np.nanmedian(speedbouts[number,i,:])
                     
            i=i+1

#calulating total for entire period (light is index 0, dark is index 1)    
    x=0
    while x < nExpts:
        boutsperperiod[number,x] = np.sum(boutsperblock[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts))])
        transitionstotopperperiod[number,x] = np.sum(transitionstotopperblock[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts))])
    
        totdistperperiod[number,x] = np.sum(totdistperblock[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts))])


        totdispperperiod[number,x] = np.sum(totdispperblock[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts))])


        tottimemovingperperiod[number,x] = np.sum(tottimemovingperblock[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts))])


        totavgspeedperperiod[number,x] = (totdistperperiod[number,x]/tottimemovingperperiod[number,x])*1000


        avgthigmoperperiod[number,x] = np.nanmean(thigmo[number, (x*int(nBlocks/nExpts)*nFramesblock):(((x+1)*int(nBlocks/nExpts))*nFramesblock)])                                               

        medthigmoperperiod[number,x] = np.nanmedian(thigmo[number, (x*int(nBlocks/nExpts)*nFramesblock):(((x+1)*int(nBlocks/nExpts))*nFramesblock)])
    
        fractionouterperperiod[number,x] = np.sum(centeryesno[number, (x*int(nBlocks/nExpts)*nFramesblock):(((x+1)*int(nBlocks/nExpts))*nFramesblock)])/(nFramesblock*(int(nBlocks/nExpts)))
    
        fractiontopperperiod[number,x] = np.sum(topyesno[number, (x*int(nBlocks/nExpts)*nFramesblock):(((x+1)*int(nBlocks/nExpts))*nFramesblock)])/(nFramesblock*(int(nBlocks/nExpts)))

        fractiondisttopperperiod[number,x] = np.sum(totdisttopperblock[number, x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts))])/np.sum(totdistperblock[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts))])

        avgdistfrommiddleperperiod[number,x] = np.nanmean(distfrommiddle[number, (x*int(nBlocks/nExpts)*nFramesblock):(((x+1)*int(nBlocks/nExpts))*nFramesblock)])
    
        meddistfrommiddleperperiod[number,x] = np.nanmedian(distfrommiddle[number, (x*int(nBlocks/nExpts)*nFramesblock):(((x+1)*int(nBlocks/nExpts))*nFramesblock)])


#calculating per bout, per period values
        avgdistperboutperperiod[number,x] = np.nanmean(distbouts[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts)),:])


        meddistperboutperperiod[number,x] = np.nanmedian(distbouts[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts)),:])


        avgdispperboutperperiod[number,x] = np.nanmean(dispbouts[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts)),:])


        meddispperboutperperiod[number,x] = np.nanmedian(dispbouts[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts)),:])


        avgtimemovingperboutperperiod[number,x] = np.nanmean(timebouts[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts)),:])


        medtimemovingperboutperperiod[number,x] = np.nanmedian(timebouts[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts)),:])


        avgspeedperboutperperiod[number,x] = np.nanmean(speedbouts[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts)),:])


        medspeedperboutperperiod[number,x] = np.nanmedian(speedbouts[number,x*int(nBlocks/nExpts):((x+1)*int(nBlocks/nExpts)),:])

        x=x+1
    
  
print("Saving tracking data arrays....")
np.savez(inDir + 'Results/TrackedVideos/' + Name + '/AnalyzedData.npz', medfiltcentY=medfiltcentY, medfiltcentX=medfiltcentX, disp=disp, filtdisp=filtdisp, thigmo=thigmo, filtthigmo=filtthigmo, 
         centeryesno=centeryesno, movementyesno=movementyesno, movementyesnofill=movementyesnofill,
         boutsperblock=boutsperblock, totdistperblock=totdistperblock, totdispperblock=totdispperblock, tottimemovingperblock=tottimemovingperblock,
         totavgspeedperblock=totavgspeedperblock, avgthigmoperblock=avgthigmoperblock, medthigmoperblock=medthigmoperblock, fractionouterperblock=fractionouterperblock,
         boutsperperiod=boutsperperiod, totdistperperiod=totdistperperiod, totdispperperiod=totdispperperiod, tottimemovingperperiod=tottimemovingperperiod, 
         totavgspeedperperiod=totavgspeedperperiod, avgthigmoperperiod=avgthigmoperperiod, medthigmoperperiod=medthigmoperperiod, fractionouterperperiod=fractionouterperperiod,
         avgdistperboutperblock=avgdistperboutperblock, avgdispperboutperblock=avgdispperboutperblock, avgtimemovingperboutperblock=avgtimemovingperboutperblock,
         avgspeedperboutperblock=avgspeedperboutperblock, meddistperboutperblock=meddistperboutperblock, meddispperboutperblock=meddispperboutperblock,
         medtimemovingperboutperblock=medtimemovingperboutperblock, medspeedperboutperblock=medspeedperboutperblock, avgdistperboutperperiod=avgdistperboutperperiod,
         avgdispperboutperperiod=avgdispperboutperperiod, avgtimemovingperboutperperiod=avgtimemovingperboutperperiod, avgspeedperboutperperiod=avgspeedperboutperperiod,
         meddistperboutperperiod=meddistperboutperperiod, meddispperboutperperiod=meddispperboutperperiod, medtimemovingperboutperperiod=medtimemovingperboutperperiod,
         medspeedperboutperperiod=medspeedperboutperperiod, distbouts=distbouts, dispbouts=dispbouts, timebouts=timebouts, speedbouts=speedbouts,
         fractiontopperblock=fractiontopperblock, fractiontopperperiod=fractiontopperperiod, topyesno=topyesno, distfrommiddle=distfrommiddle,
         avgdistfrommiddleperblock=avgdistfrommiddleperblock, meddistfrommiddleperblock=meddistfrommiddleperblock,
         avgdistfrommiddleperperiod=avgdistfrommiddleperperiod, meddistfrommiddleperperiod=meddistfrommiddleperperiod,
         totdisttopperblock=totdisttopperblock, fractiondisttopperblock=fractiondisttopperblock, fractiondisttopperperiod=fractiondisttopperperiod,
         transitionstotopperblock=transitionstotopperblock, transitionstotopperperiod=transitionstotopperperiod
         )


print("Setting up final arrays....")   
Gene = 'test'

#designate which fish are mut/het/wt
muts = np.array([])
hets = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
wts = np.array([])

#muts = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33])
#hets = np.array([34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66])
#wts = np.array([67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100])


print("Setting up final dictionaries....")
resultdict = {}

#setup dictionary with total results for all fish based on event
for number in range(nFish):
    geno = np.nan
    if (number+1) in muts:
        geno = "mut"
    if (number+1) in hets:
        geno = "het"
    if (number+1) in wts:
        geno = "wt"
    for i in range(nBlocks+nExpts):  
        if i<nBlocks:
            resultdict[str(number+1)+"-"+str(i+1)] = {
                'fish#':(number+1), 
                'genotype': geno,
                'block':(i+1),
                '#bouts': boutsperblock[number, i],
                'totdist': totdistperblock[number, i],
                'totdisp': totdispperblock[number, i],
                'tottimemvmt': tottimemovingperblock[number, i],
                'totavgspeed': totavgspeedperblock[number, i],
                'avgfromcenter': avgthigmoperblock[number, i],
                'medfromcenter': medthigmoperblock[number, i],
                'fractionouter': fractionouterperblock[number, i],
                'fractiontop' : fractiontopperblock[number,i],
                'fractiondisttop' : fractiondisttopperblock[number,i],
                'transitionstotop' : transitionstotopperblock[number,i],
                'avgdistfrommiddle' : avgdistfrommiddleperblock[number,i],
                'meddistfrommiddle' : meddistfrommiddleperblock[number,i],
                'avgdistperbout': avgdistperboutperblock[number, i],
                'meddistperbout': meddistperboutperblock[number, i],
                'avgdispperbout': avgdispperboutperblock[number, i],
                'meddispperbout': meddispperboutperblock[number, i],
                'avgtimeperbout': avgtimemovingperboutperblock[number, i],
                'medtimeperbout': medtimemovingperboutperblock[number, i],
                'avgspeedperbout': avgspeedperboutperblock[number, i],
                'medspeedperbout': medspeedperboutperblock[number, i],
                }
            
        #last columns will be totals in light then dark (ie per period)
        if i==nBlocks:
            resultdict[str(number+1)+"-"+str(i+1)] = {
                'fish#':(number+1), 
                'genotype': geno,
                'block':(i+1),
                '#bouts': boutsperperiod[number, 0],
                'totdist': totdistperperiod[number, 0],
                'totdisp': totdispperperiod[number, 0],
                'tottimemvmt': tottimemovingperperiod[number, 0],
                'totavgspeed': totavgspeedperperiod[number, 0],
                'avgfromcenter': avgthigmoperperiod[number, 0],
                'medfromcenter': medthigmoperperiod[number, 0],
                'fractionouter': fractionouterperperiod[number, 0],
                'fractiontop': fractiontopperperiod[number, 0],
                'fractiondisttop': fractiondisttopperperiod[number, 0],
                'avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 0],
                'meddistfrommiddle' : meddistfrommiddleperperiod[number, 0],
                'avgdistperbout': avgdistperboutperperiod[number, 0],
                'meddistperbout': meddistperboutperperiod[number, 0],
                'avgdispperbout': avgdispperboutperperiod[number, 0],
                'meddispperbout': meddispperboutperperiod[number, 0],
                'avgtimeperbout': avgtimemovingperboutperperiod[number, 0],
                'medtimeperbout': medtimemovingperboutperperiod[number, 0],
                'avgspeedperbout': avgspeedperboutperperiod[number, 0],
                'medspeedperbout': medspeedperboutperperiod[number, 0],
                }
  
        if i==nBlocks+1:
            resultdict[str(number+1)+"-"+str(i+1)] = {
                'fish#':(number+1), 
                'genotype': geno,
                'block':(i+1),
                '#bouts': boutsperperiod[number, 1],
                'totdist': totdistperperiod[number, 1],
                'totdisp': totdispperperiod[number, 1],
                'tottimemvmt': tottimemovingperperiod[number, 1],
                'totavgspeed': totavgspeedperperiod[number, 1],
                'avgfromcenter': avgthigmoperperiod[number, 1],
                'medfromcenter': medthigmoperperiod[number, 1],
                'fractionouter': fractionouterperperiod[number, 1],
                'fractiontop': fractiontopperperiod[number, 1],
                'fractiondisttop': fractiondisttopperperiod[number, 1],
                'avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 1],
                'meddistfrommiddle' : meddistfrommiddleperperiod[number, 1],
                'avgdistperbout': avgdistperboutperperiod[number, 1],
                'meddistperbout': meddistperboutperperiod[number, 1],
                'avgdispperbout': avgdispperboutperperiod[number, 1],
                'meddispperbout': meddispperboutperperiod[number, 1],
                'avgtimeperbout': avgtimemovingperboutperperiod[number, 1],
                'medtimeperbout': medtimemovingperboutperperiod[number, 1],
                'avgspeedperbout': avgspeedperboutperperiod[number, 1],
                'medspeedperbout': medspeedperboutperperiod[number, 1],
                }        
 
        if i==nBlocks+2:
            resultdict[str(number+1)+"-"+str(i+1)] = {
                'fish#':(number+1), 
                'genotype': geno,
                'block':(i+1),
                '#bouts': boutsperperiod[number, 2],
                'totdist': totdistperperiod[number, 2],
                'totdisp': totdispperperiod[number, 2],
                'tottimemvmt': tottimemovingperperiod[number, 2],
                'totavgspeed': totavgspeedperperiod[number, 2],
                'avgfromcenter': avgthigmoperperiod[number, 2],
                'medfromcenter': medthigmoperperiod[number, 2],
                'fractionouter': fractionouterperperiod[number, 2],
                'fractiontop': fractiontopperperiod[number, 2],
                'fractiondisttop': fractiondisttopperperiod[number, 2],
                'avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 2],
                'meddistfrommiddle' : meddistfrommiddleperperiod[number, 2],
                'avgdistperbout': avgdistperboutperperiod[number, 2],
                'meddistperbout': meddistperboutperperiod[number, 2],
                'avgdispperbout': avgdispperboutperperiod[number, 2],
                'meddispperbout': meddispperboutperperiod[number, 2],
                'avgtimeperbout': avgtimemovingperboutperperiod[number, 2],
                'medtimeperbout': medtimemovingperboutperperiod[number, 2],
                'avgspeedperbout': avgspeedperboutperperiod[number, 2],
                'medspeedperbout': medspeedperboutperperiod[number, 2],
                }             

        if i==nBlocks+3:
            resultdict[str(number+1)+"-"+str(i+1)] = {
                'fish#':(number+1), 
                'genotype': geno,
                'block':(i+1),
                '#bouts': boutsperperiod[number, 3],
                'totdist': totdistperperiod[number, 3],
                'totdisp': totdispperperiod[number, 3],
                'tottimemvmt': tottimemovingperperiod[number, 3],
                'totavgspeed': totavgspeedperperiod[number, 3],
                'avgfromcenter': avgthigmoperperiod[number, 3],
                'medfromcenter': medthigmoperperiod[number, 3],
                'fractionouter': fractionouterperperiod[number, 3],
                'fractiontop': fractiontopperperiod[number, 3],
                'fractiondisttop': fractiondisttopperperiod[number, 3],
                'avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 3],
                'meddistfrommiddle' : meddistfrommiddleperperiod[number, 3],
                'avgdistperbout': avgdistperboutperperiod[number, 3],
                'meddistperbout': meddistperboutperperiod[number, 3],
                'avgdispperbout': avgdispperboutperperiod[number, 3],
                'meddispperbout': meddispperboutperperiod[number, 3],
                'avgtimeperbout': avgtimemovingperboutperperiod[number, 3],
                'medtimeperbout': medtimemovingperboutperperiod[number, 3],
                'avgspeedperbout': avgspeedperboutperperiod[number, 3],
                'medspeedperbout': medspeedperboutperperiod[number, 3],
                }

        if i==nBlocks+4:
            resultdict[str(number+1)+"-"+str(i+1)] = {
                'fish#':(number+1), 
                'genotype': geno,
                'block':(i+1),
                '#bouts': boutsperperiod[number, 4],
                'totdist': totdistperperiod[number, 4],
                'totdisp': totdispperperiod[number, 4],
                'tottimemvmt': tottimemovingperperiod[number, 4],
                'totavgspeed': totavgspeedperperiod[number, 4],
                'avgfromcenter': avgthigmoperperiod[number, 4],
                'medfromcenter': medthigmoperperiod[number, 4],
                'fractionouter': fractionouterperperiod[number, 4],
                'fractiontop': fractiontopperperiod[number, 4],
                'fractiondisttop': fractiondisttopperperiod[number, 4],
                'avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 4],
                'meddistfrommiddle' : meddistfrommiddleperperiod[number, 4],
                'avgdistperbout': avgdistperboutperperiod[number, 4],
                'meddistperbout': meddistperboutperperiod[number, 4],
                'avgdispperbout': avgdispperboutperperiod[number, 4],
                'meddispperbout': meddispperboutperperiod[number, 4],
                'avgtimeperbout': avgtimemovingperboutperperiod[number, 4],
                'medtimeperbout': medtimemovingperboutperperiod[number, 4],
                'avgspeedperbout': avgspeedperboutperperiod[number, 4],
                'medspeedperbout': medspeedperboutperperiod[number, 4],
                }

        if i==nBlocks+5:
            resultdict[str(number+1)+"-"+str(i+1)] = {
                'fish#':(number+1), 
                'genotype': geno,
                'block':(i+1),
                '#bouts': boutsperperiod[number, 5],
                'totdist': totdistperperiod[number, 5],
                'totdisp': totdispperperiod[number, 5],
                'tottimemvmt': tottimemovingperperiod[number, 5],
                'totavgspeed': totavgspeedperperiod[number, 5],
                'avgfromcenter': avgthigmoperperiod[number, 5],
                'medfromcenter': medthigmoperperiod[number, 5],
                'fractionouter': fractionouterperperiod[number, 5],
                'fractiontop': fractiontopperperiod[number, 5],
                'fractiondisttop': fractiondisttopperperiod[number, 5],
                'avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 5],
                'meddistfrommiddle' : meddistfrommiddleperperiod[number, 5],
                'avgdistperbout': avgdistperboutperperiod[number, 5],
                'meddistperbout': meddistperboutperperiod[number, 5],
                'avgdispperbout': avgdispperboutperperiod[number, 5],
                'meddispperbout': meddispperboutperperiod[number, 5],
                'avgtimeperbout': avgtimemovingperboutperperiod[number, 5],
                'medtimeperbout': medtimemovingperboutperperiod[number, 5],
                'avgspeedperbout': avgspeedperboutperperiod[number, 5],
                'medspeedperbout': medspeedperboutperperiod[number, 5],
                }

dfresult = pd.DataFrame(data=resultdict)
dfresultT = dfresult.T


#dictionary for each individual fish
resultdictindiv = {}

#setup dictionary with total results for all fish into dictionary/dataframe
for number in range(nFish):
    geno = np.nan
    if (number+1) in muts:
        geno = "mut"
    if (number+1) in hets:
        geno = "het"
    if (number+1) in wts:
        geno = "wt"       
    for i in range(nBlocks+nExpts):  
        if i==0:
            resultdictindiv[str(number+1)] = {
            'fish#':(number+1), 
            'genotype': geno,
            str(i+1)+'-#bouts': boutsperblock[number, i],
            str(i+1)+'-totdist': totdistperblock[number, i],
            str(i+1)+'-totdisp': totdispperblock[number, i],
            str(i+1)+'-tottimemvmt': tottimemovingperblock[number, i],
            str(i+1)+'-totavgspeed': totavgspeedperblock[number, i],
            str(i+1)+'-avgfromcenter': avgthigmoperblock[number, i],
            str(i+1)+'-medfromcenter': medthigmoperblock[number, i],
            str(i+1)+'-fractionouter': fractionouterperblock[number, i],
            str(i+1)+'-fractiontop': fractiontopperblock[number, i],
            str(i+1)+'-fractiondisttop' : fractiondisttopperblock[number,i],
            str(i+1)+'-avgdistfrommiddle' : avgdistfrommiddleperblock[number,i],
            str(i+1)+'-meddistfrommiddle' : meddistfrommiddleperblock[number,i],
            str(i+1)+'-avgdistperbout': avgdistperboutperblock[number, i],
            str(i+1)+'-meddistperbout': meddistperboutperblock[number, i],
            str(i+1)+'-avgdispperbout': avgdispperboutperblock[number, i],
            str(i+1)+'-meddispperbout': meddispperboutperblock[number, i],
            str(i+1)+'-avgtimeperbout': avgtimemovingperboutperblock[number, i],
            str(i+1)+'-medtimeperbout': medtimemovingperboutperblock[number, i],
            str(i+1)+'-avgspeedperbout': avgspeedperboutperblock[number, i],
            str(i+1)+'-medspeedperbout': medspeedperboutperblock[number, i],
            }            
        
        if i>0 and i<nBlocks:
            resultdictindiv[str(number+1)].update( {
            'fish#':(number+1), 
            'genotype': geno,
            str(i+1)+'-#bouts': boutsperblock[number, i],
            str(i+1)+'-totdist': totdistperblock[number, i],
            str(i+1)+'-totdisp': totdispperblock[number, i],
            str(i+1)+'-tottimemvmt': tottimemovingperblock[number, i],
            str(i+1)+'-totavgspeed': totavgspeedperblock[number, i],
            str(i+1)+'-avgfromcenter': avgthigmoperblock[number, i],
            str(i+1)+'-medfromcenter': medthigmoperblock[number, i],
            str(i+1)+'-fractionouter': fractionouterperblock[number, i],
            str(i+1)+'-fractiontop': fractiontopperblock[number, i],
            str(i+1)+'-fractiondisttop' : fractiondisttopperblock[number,i],
            str(i+1)+'-avgdistfrommiddle' : avgdistfrommiddleperblock[number,i],
            str(i+1)+'-meddistfrommiddle' : meddistfrommiddleperblock[number,i],
            str(i+1)+'-avgdistperbout': avgdistperboutperblock[number, i],
            str(i+1)+'-meddistperbout': meddistperboutperblock[number, i],
            str(i+1)+'-avgdispperbout': avgdispperboutperblock[number, i],
            str(i+1)+'-meddispperbout': meddispperboutperblock[number, i],
            str(i+1)+'-avgtimeperbout': avgtimemovingperboutperblock[number, i],
            str(i+1)+'-medtimeperbout': medtimemovingperboutperblock[number, i],
            str(i+1)+'-avgspeedperbout': avgspeedperboutperblock[number, i],
            str(i+1)+'-medspeedperbout': medspeedperboutperblock[number, i],
            })            
            
        #last columns will be totals in light then dark (ie per period)
        if i==nBlocks:            
            resultdictindiv[str(number+1)].update( {
            'fish#':(number+1), 
            'genotype': geno,
            str(i+1)+'-#bouts': boutsperperiod[number, 0],
            str(i+1)+'-totdist': totdistperperiod[number, 0],
            str(i+1)+'-totdisp': totdispperperiod[number, 0],
            str(i+1)+'-tottimemvmt': tottimemovingperperiod[number, 0],
            str(i+1)+'-totavgspeed': totavgspeedperperiod[number, 0],
            str(i+1)+'-avgfromcenter': avgthigmoperperiod[number, 0],
            str(i+1)+'-medfromcenter': medthigmoperperiod[number, 0],
            str(i+1)+'-fractionouter': fractionouterperperiod[number, 0],
            str(i+1)+'-fractiontop': fractiontopperperiod[number, 0],
            str(i+1)+'-fractiondisttop': fractiondisttopperperiod[number, 0],
            str(i+1)+'-avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 0],
            str(i+1)+'-meddistfrommiddle' : meddistfrommiddleperperiod[number, 0],
            str(i+1)+'-avgdistperbout': avgdistperboutperperiod[number, 0],
            str(i+1)+'-meddistperbout': meddistperboutperperiod[number, 0],
            str(i+1)+'-avgdispperbout': avgdispperboutperperiod[number, 0],
            str(i+1)+'-meddispperbout': meddispperboutperperiod[number, 0],
            str(i+1)+'-avgtimeperbout': avgtimemovingperboutperperiod[number, 0],
            str(i+1)+'-medtimeperbout': medtimemovingperboutperperiod[number, 0],
            str(i+1)+'-avgspeedperbout': avgspeedperboutperperiod[number, 0],
            str(i+1)+'-medspeedperbout': medspeedperboutperperiod[number, 0],
            })

        if i==nBlocks+1:            
            resultdictindiv[str(number+1)].update( {
            'fish#':(number+1), 
            'genotype': geno,
            str(i+1)+'-#bouts': boutsperperiod[number, 1],
            str(i+1)+'-totdist': totdistperperiod[number, 1],
            str(i+1)+'-totdisp': totdispperperiod[number, 1],
            str(i+1)+'-tottimemvmt': tottimemovingperperiod[number, 1],
            str(i+1)+'-totavgspeed': totavgspeedperperiod[number, 1],
            str(i+1)+'-avgfromcenter': avgthigmoperperiod[number, 1],
            str(i+1)+'-medfromcenter': medthigmoperperiod[number, 1],
            str(i+1)+'-fractionouter': fractionouterperperiod[number, 1],
            str(i+1)+'-fractiontop': fractiontopperperiod[number, 1],
            str(i+1)+'-fractiondisttop': fractiondisttopperperiod[number, 1],
            str(i+1)+'-avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 1],
            str(i+1)+'-meddistfrommiddle' : meddistfrommiddleperperiod[number, 1],
            str(i+1)+'-avgdistperbout': avgdistperboutperperiod[number, 1],
            str(i+1)+'-meddistperbout': meddistperboutperperiod[number, 1],
            str(i+1)+'-avgdispperbout': avgdispperboutperperiod[number, 1],
            str(i+1)+'-meddispperbout': meddispperboutperperiod[number, 1],
            str(i+1)+'-avgtimeperbout': avgtimemovingperboutperperiod[number, 1],
            str(i+1)+'-medtimeperbout': medtimemovingperboutperperiod[number, 1],
            str(i+1)+'-avgspeedperbout': avgspeedperboutperperiod[number, 1],
            str(i+1)+'-medspeedperbout': medspeedperboutperperiod[number, 1],
            })

        if i==nBlocks+2:            
            resultdictindiv[str(number+1)].update( {
            'fish#':(number+1), 
            'genotype': geno,
            str(i+1)+'-#bouts': boutsperperiod[number, 2],
            str(i+1)+'-totdist': totdistperperiod[number, 2],
            str(i+1)+'-totdisp': totdispperperiod[number, 2],
            str(i+1)+'-tottimemvmt': tottimemovingperperiod[number, 2],
            str(i+1)+'-totavgspeed': totavgspeedperperiod[number, 2],
            str(i+1)+'-avgfromcenter': avgthigmoperperiod[number, 2],
            str(i+1)+'-medfromcenter': medthigmoperperiod[number, 2],
            str(i+1)+'-fractionouter': fractionouterperperiod[number, 2],
            str(i+1)+'-fractiontop': fractiontopperperiod[number, 2],
            str(i+1)+'-fractiondisttop': fractiondisttopperperiod[number, 2],
            str(i+1)+'-avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 2],
            str(i+1)+'-meddistfrommiddle' : meddistfrommiddleperperiod[number, 2],
            str(i+1)+'-avgdistperbout': avgdistperboutperperiod[number, 2],
            str(i+1)+'-meddistperbout': meddistperboutperperiod[number, 2],
            str(i+1)+'-avgdispperbout': avgdispperboutperperiod[number, 2],
            str(i+1)+'-meddispperbout': meddispperboutperperiod[number, 2],
            str(i+1)+'-avgtimeperbout': avgtimemovingperboutperperiod[number, 2],
            str(i+1)+'-medtimeperbout': medtimemovingperboutperperiod[number, 2],
            str(i+1)+'-avgspeedperbout': avgspeedperboutperperiod[number, 2],
            str(i+1)+'-medspeedperbout': medspeedperboutperperiod[number, 2],
            })

        if i==nBlocks+3:            
            resultdictindiv[str(number+1)].update( {
            'fish#':(number+1), 
            'genotype': geno,
            str(i+1)+'-#bouts': boutsperperiod[number, 3],
            str(i+1)+'-totdist': totdistperperiod[number, 3],
            str(i+1)+'-totdisp': totdispperperiod[number, 3],
            str(i+1)+'-tottimemvmt': tottimemovingperperiod[number, 3],
            str(i+1)+'-totavgspeed': totavgspeedperperiod[number, 3],
            str(i+1)+'-avgfromcenter': avgthigmoperperiod[number, 3],
            str(i+1)+'-medfromcenter': medthigmoperperiod[number, 3],
            str(i+1)+'-fractionouter': fractionouterperperiod[number, 3],
            str(i+1)+'-fractiontop': fractiontopperperiod[number, 3],
            str(i+1)+'-fractiondisttop': fractiondisttopperperiod[number, 3],
            str(i+1)+'-avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 3],
            str(i+1)+'-meddistfrommiddle' : meddistfrommiddleperperiod[number, 3],
            str(i+1)+'-avgdistperbout': avgdistperboutperperiod[number, 3],
            str(i+1)+'-meddistperbout': meddistperboutperperiod[number, 3],
            str(i+1)+'-avgdispperbout': avgdispperboutperperiod[number, 3],
            str(i+1)+'-meddispperbout': meddispperboutperperiod[number, 3],
            str(i+1)+'-avgtimeperbout': avgtimemovingperboutperperiod[number, 3],
            str(i+1)+'-medtimeperbout': medtimemovingperboutperperiod[number, 3],
            str(i+1)+'-avgspeedperbout': avgspeedperboutperperiod[number, 3],
            str(i+1)+'-medspeedperbout': medspeedperboutperperiod[number, 3],
            })

        if i==nBlocks+4:            
            resultdictindiv[str(number+1)].update( {
            'fish#':(number+1), 
            'genotype': geno,
            str(i+1)+'-#bouts': boutsperperiod[number, 4],
            str(i+1)+'-totdist': totdistperperiod[number, 4],
            str(i+1)+'-totdisp': totdispperperiod[number, 4],
            str(i+1)+'-tottimemvmt': tottimemovingperperiod[number, 4],
            str(i+1)+'-totavgspeed': totavgspeedperperiod[number, 4],
            str(i+1)+'-avgfromcenter': avgthigmoperperiod[number, 4],
            str(i+1)+'-medfromcenter': medthigmoperperiod[number, 4],
            str(i+1)+'-fractionouter': fractionouterperperiod[number, 4],
            str(i+1)+'-fractiontop': fractiontopperperiod[number, 4],
            str(i+1)+'-fractiondisttop': fractiondisttopperperiod[number, 4],
            str(i+1)+'-avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 4],
            str(i+1)+'-meddistfrommiddle' : meddistfrommiddleperperiod[number, 4],
            str(i+1)+'-avgdistperbout': avgdistperboutperperiod[number, 4],
            str(i+1)+'-meddistperbout': meddistperboutperperiod[number, 4],
            str(i+1)+'-avgdispperbout': avgdispperboutperperiod[number, 4],
            str(i+1)+'-meddispperbout': meddispperboutperperiod[number, 4],
            str(i+1)+'-avgtimeperbout': avgtimemovingperboutperperiod[number, 4],
            str(i+1)+'-medtimeperbout': medtimemovingperboutperperiod[number, 4],
            str(i+1)+'-avgspeedperbout': avgspeedperboutperperiod[number, 4],
            str(i+1)+'-medspeedperbout': medspeedperboutperperiod[number, 4],
            })

        if i==nBlocks+5:            
            resultdictindiv[str(number+1)].update( {
            'fish#':(number+1), 
            'genotype': geno,
            str(i+1)+'-#bouts': boutsperperiod[number, 5],
            str(i+1)+'-totdist': totdistperperiod[number, 5],
            str(i+1)+'-totdisp': totdispperperiod[number, 5],
            str(i+1)+'-tottimemvmt': tottimemovingperperiod[number, 5],
            str(i+1)+'-totavgspeed': totavgspeedperperiod[number, 5],
            str(i+1)+'-avgfromcenter': avgthigmoperperiod[number, 5],
            str(i+1)+'-medfromcenter': medthigmoperperiod[number, 5],
            str(i+1)+'-fractionouter': fractionouterperperiod[number, 5],
            str(i+1)+'-fractiontop': fractiontopperperiod[number, 5],
            str(i+1)+'-fractiondisttop': fractiondisttopperperiod[number, 5],
            str(i+1)+'-avgdistfrommiddle' : avgdistfrommiddleperperiod[number, 5],
            str(i+1)+'-meddistfrommiddle' : meddistfrommiddleperperiod[number, 5],
            str(i+1)+'-avgdistperbout': avgdistperboutperperiod[number, 5],
            str(i+1)+'-meddistperbout': meddistperboutperperiod[number, 5],
            str(i+1)+'-avgdispperbout': avgdispperboutperperiod[number, 5],
            str(i+1)+'-meddispperbout': meddispperboutperperiod[number, 5],
            str(i+1)+'-avgtimeperbout': avgtimemovingperboutperperiod[number, 5],
            str(i+1)+'-medtimeperbout': medtimemovingperboutperperiod[number, 5],
            str(i+1)+'-avgspeedperbout': avgspeedperboutperperiod[number, 5],
            str(i+1)+'-medspeedperbout': medspeedperboutperperiod[number, 5],
            })
            
dfresultindiv = pd.DataFrame(data=resultdictindiv)
dfresultindivT = dfresultindiv.T



variables = ['#bouts','totdist','totdisp','tottimemvmt','totavgspeed','avgfromcenter','medfromcenter','fractionouter','fractiontop', 'avgdistperbout','fractiondisttop',
             'avgdistfrommiddle','meddistfrommiddle','meddistperbout','avgdispperbout','meddispperbout','avgtimeperbout','medtimeperbout','avgspeedperbout','medspeedperbout'
             ]


genoresultdict = {}
statsdict = {}


#setup dictionary with results averaged over genotype into dictionary/dataframe
for number in range(nBlocks+2):
    for i,element in enumerate(variables):
        if float(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].std(axis=0)) == 0:
            genoresultdict[str(number+1)+"-"+str(element)+"-"+"mut"] = {
            'block':(number+1),
            'value': str(element),
            'genotype': "mut",
            'count': float(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].count()),
            'avg':float(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].mean(axis=0)),
            'median':float(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].median(axis=0)),
            'stdev':float(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].std(axis=0)),
            '(avg-wt)/wtstd': np.nan
            }
            genoresultdict[str(number+1)+"-"+str(element)+"-"+"het"] = {
            'block':(number+1),
            'value': str(element),
            'genotype': "het",
            'count': float(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].count()),
            'avg':float(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].mean(axis=0)),
            'median':float(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].median(axis=0)),
            'stdev':float(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].std(axis=0)),
            '(avg-wt)/wtstd': np.nan
            }
            
        else:    
            genoresultdict[str(number+1)+"-"+str(element)+"-"+"mut"] = {
            'block':(number+1),
            'value': str(element),
            'genotype': "mut",
            'count': float(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].count()),
            'avg':float(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].mean(axis=0)),
            'median':float(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].median(axis=0)),
            'stdev':float(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].std(axis=0)),
            '(avg-wt)/wtstd': (float(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].mean(axis=0)) - float(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].mean(axis=0)))/float(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].std(axis=0))
            }
            genoresultdict[str(number+1)+"-"+str(element)+"-"+"het"] = {
            'block':(number+1),
            'value': str(element),
            'genotype': "het",
            'count': float(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].count()),
            'avg':float(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].mean(axis=0)),
            'median':float(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].median(axis=0)),
            'stdev':float(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].std(axis=0)),
            '(avg-wt)/wtstd': (float(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].mean(axis=0)) - float(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].mean(axis=0)))/float(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].std(axis=0))
            }
        genoresultdict[str(number+1)+"-"+str(element)+"-"+"wt"] = {
            'block':(number+1),
            'value': str(element),
            'genotype': "wt",
            'count': float(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].count()),
            'avg':float(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].mean(axis=0)),
            'median':float(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].median(axis=0)),
            'stdev':float(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].std(axis=0))
            }

#setup dictionary with statistics for each event and property, removing those with <5 tracked objects in block 
        anova = np.array(stats.f_oneway(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].dropna(), 
            dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].dropna(),
            dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].dropna()))
        
        mut = np.array(dfresultT.loc[(dfresultT['genotype']=='mut') & (dfresultT['block']==(number+1)),[str(element)]].dropna())
        het = np.array(dfresultT.loc[(dfresultT['genotype']=='het') & (dfresultT['block']==(number+1)),[str(element)]].dropna())
        wt = np.array(dfresultT.loc[(dfresultT['genotype']=='wt') & (dfresultT['block']==(number+1)),[str(element)]].dropna())
 
        if anova[1]<0.05:
            sig = 'YES'
        else:
            sig = 'NO'
       
        ttestmutvhet = np.array(stats.ttest_ind(mut.astype(float), het.astype(float), equal_var=False))
       
        ttestmutvwt = np.array(stats.ttest_ind(mut.astype(float), wt.astype(float), equal_var=False))
        
        ttesthetvwt = np.array(stats.ttest_ind(het.astype(float), wt.astype(float), equal_var=False))
        
        statsdict[str("Baseline-")+str(number+1)+"-"+str(element)] = {
            'ANOVA F-value': float(anova[0]),
            'ANOVA p-value': float(anova[1]),
            'Significance?': str(sig),
            'mut v het p value': float(ttestmutvhet[1]),
            'mut v wt p value': float(ttestmutvwt[1]),
            'het v wt p value': float(ttesthetvwt[1])
            }




dfgenoresult = pd.DataFrame(data=genoresultdict)
dfgenoresultT = dfgenoresult.T

dfgenoresultT['block+value'] = dfgenoresultT['block'].apply(str)+"-"+dfgenoresultT['value'].apply(str)

dfstats = pd.DataFrame(data=statsdict)
dfstatsT = dfstats.T
          

print("Saving results in spreadsheet....")  
dfresultT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResultsByFishbyEvent.csv')
dfresultindivT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResultsByIndividualFish.csv')
dfgenoresultT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResultsByGenotype.csv')
dfstatsT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResults_STATS.csv')


print("Plotting results....")  
#All Blocks
plt.figure(figsize=(25,50))
plt.subplot(7,3,1)
sns.boxplot(x='block', y='#bouts', data=dfresultT, hue='genotype').set_title("Number of bouts")

plt.subplot(7,3,2)
sns.boxplot(x='block', y='totdist', data=dfresultT, hue='genotype').set_title("Total distance moved")

plt.subplot(7,3,3)
sns.boxplot(x='block', y='totdisp', data=dfresultT, hue='genotype').set_title("Total displacement moved")

plt.subplot(7,3,4)
sns.boxplot(x='block', y='tottimemvmt', data=dfresultT, hue='genotype').set_title("Total time moved")

plt.subplot(7,3,5)
sns.boxplot(x='block', y='totavgspeed', data=dfresultT, hue='genotype').set_title("Total average speed")

plt.subplot(7,3,6)
sns.boxplot(x='block', y='avgfromcenter', data=dfresultT, hue='genotype').set_title("Average distance from center")

plt.subplot(7,3,7)
sns.boxplot(x='block', y='medfromcenter', data=dfresultT, hue='genotype').set_title("Median distance from center")

plt.subplot(7,3,8)
sns.boxplot(x='block', y='fractionouter', data=dfresultT, hue='genotype').set_title("Fraction of time in outer rim")

plt.subplot(7,3,9)
sns.boxplot(x='block', y='fractiontop', data=dfresultT, hue='genotype').set_title("Fraction of time in top half")

plt.subplot(7,3,10)
sns.boxplot(x='block', y='fractiondisttop', data=dfresultT, hue='genotype').set_title("Fraction of distance traveled in top half")

plt.subplot(7,3,11)
sns.boxplot(x='block', y='avgdistfrommiddle', data=dfresultT, hue='genotype').set_title("Average distance from middle")

plt.subplot(7,3,12)
sns.boxplot(x='block', y='meddistfrommiddle', data=dfresultT, hue='genotype').set_title("Median distance from middle")

plt.subplot(7,3,13)
sns.boxplot(x='block', y='avgdistperbout', data=dfresultT, hue='genotype').set_title("Average distance per bout")

plt.subplot(7,3,14)
sns.boxplot(x='block', y='meddistperbout', data=dfresultT, hue='genotype').set_title("Median distance per bout")

plt.subplot(7,3,15)
sns.boxplot(x='block', y='avgdispperbout', data=dfresultT, hue='genotype').set_title("Average displacement per bout")

plt.subplot(7,3,16)
sns.boxplot(x='block', y='meddispperbout', data=dfresultT, hue='genotype').set_title("Median displacement per bout")

plt.subplot(7,3,17)
sns.boxplot(x='block', y='avgtimeperbout', data=dfresultT, hue='genotype').set_title("Average time per bout")

plt.subplot(7,3,18)
sns.boxplot(x='block', y='medtimeperbout', data=dfresultT, hue='genotype').set_title("Median time per bout")

plt.subplot(7,3,19)
sns.boxplot(x='block', y='avgspeedperbout', data=dfresultT, hue='genotype').set_title("Average speed per bout")

plt.subplot(7,3,20)
sns.boxplot(x='block', y='medspeedperbout', data=dfresultT, hue='genotype').set_title("Median speed per bout")




# plt.figure(figsize=(25,50))
# plt.subplot(7,3,1)
# sns.boxplot(x='block', y='#bouts', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Number of bouts")

# plt.subplot(7,3,2)
# sns.boxplot(x='block', y='totdist', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Total distance moved")

# plt.subplot(7,3,3)
# sns.boxplot(x='block', y='totdisp', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Total displacement moved")

# plt.subplot(7,3,4)
# sns.boxplot(x='block', y='tottimemvmt', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Total time moved")

# plt.subplot(7,3,5)
# sns.boxplot(x='block', y='totavgspeed', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Total average speed")

# plt.subplot(7,3,6)
# sns.boxplot(x='block', y='avgfromcenter', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Average distance from center")

# plt.subplot(7,3,7)
# sns.boxplot(x='block', y='medfromcenter', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Median distance from center")

# plt.subplot(7,3,8)
# sns.boxplot(x='block', y='fractionouter', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Fraction of time in outer rim")

# plt.subplot(7,3,9)
# sns.boxplot(x='block', y='fractiontop', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Fraction of time in top half")

# plt.subplot(7,3,10)
# sns.boxplot(x='block', y='fractiondisttop', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Fraction of distance traveled in top half")

# plt.subplot(7,3,11)
# sns.boxplot(x='block', y='avgdistfrommiddle', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Average distance from middle")

# plt.subplot(7,3,12)
# sns.boxplot(x='block', y='meddistfrommiddle', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Median distance from middle")

# plt.subplot(7,3,13)
# sns.boxplot(x='block', y='avgdistperbout', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Average distance per bout")

# plt.subplot(7,3,14)
# sns.boxplot(x='block', y='meddistperbout', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Median distance per bout")

# plt.subplot(7,3,15)
# sns.boxplot(x='block', y='avgdispperbout', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Average displacement per bout")

# plt.subplot(7,3,16)
# sns.boxplot(x='block', y='meddispperbout', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Median displacement per bout")

# plt.subplot(7,3,17)
# sns.boxplot(x='block', y='avgtimeperbout', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Average time per bout")

# plt.subplot(7,3,18)
# sns.boxplot(x='block', y='medtimeperbout', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Median time per bout")

# plt.subplot(7,3,19)
# sns.boxplot(x='block', y='avgspeedperbout', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Average speed per bout")

# plt.subplot(7,3,20)
# sns.boxplot(x='block', y='medspeedperbout', data=dfresultT.loc[(dfresultT['fish#']!=5)], hue='genotype').set_title("Median speed per bout")


plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_all.pdf', bbox_inches='tight')

plt.clf()



#Heatmaps
plt.figure(figsize=(25,10))
plt.subplot(6,3,1)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==1)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,2)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==2)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,3)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==3)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,4)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==4)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,5)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==5)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,6)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==6)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,7)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==7)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,8)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==8)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,9)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==9)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,10)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==10)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,11)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==11)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,12)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==12)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,13)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==13)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,14)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==14)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,15)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==15)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')

plt.subplot(6,3,16)
grid = dfgenoresultT.loc[(dfgenoresultT['block']==16)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5, vmax=2.5, cmap='RdBu')


plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_all_Heatmaps.pdf', bbox_inches='tight')

plt.clf()



# correlation heatmap for every value for each fish
totcorrheatmap = dfresultT.copy()

del totcorrheatmap['genotype']
del totcorrheatmap['event']
del totcorrheatmap['fish#']
del totcorrheatmap['totbendev']
del totcorrheatmap['totdispev']
del totcorrheatmap['totdistev']
del totcorrheatmap['totangvelev']

del totcorrheatmap['totlatev']
del totcorrheatmap['totorientev']
del totcorrheatmap['totslcev']

plt.figure(figsize=(25,10))
sns.heatmap(totcorrheatmap.astype(float).corr(method='pearson'), vmin=-0.6, vmax=0.6, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_TotalCorrelation matrix.pdf', bbox_inches='tight')


propcorrheatmap = totcorrheatmap.copy()

del propcorrheatmap['habavgbend']
del propcorrheatmap['habavgdisp']
del propcorrheatmap['habavgdist']
del propcorrheatmap['habavglat']
del propcorrheatmap['habavgorient']
del propcorrheatmap['habavgduration']
del propcorrheatmap['habmedbend']
del propcorrheatmap['habavgangvel']
del propcorrheatmap['habmedangvel']
del propcorrheatmap['habmeddisp']
del propcorrheatmap['habmeddist']
del propcorrheatmap['habmedlat']
del propcorrheatmap['habmedorient']
del propcorrheatmap['habmedduration']
del propcorrheatmap['habslc']
                
del propcorrheatmap['ppiavgbend']
del propcorrheatmap['ppiavgdisp']
del propcorrheatmap['ppiavgdist']
del propcorrheatmap['ppiavglat']
del propcorrheatmap['ppiavgorient']
del propcorrheatmap['ppimedbend']
del propcorrheatmap['ppimeddisp']
del propcorrheatmap['ppimeddist']
del propcorrheatmap['ppimedlat']
del propcorrheatmap['ppimedorient']
del propcorrheatmap['ppislc']   
del propcorrheatmap['ppiavgduration']
del propcorrheatmap['ppimedduration']
del propcorrheatmap['ppiavgangvel']
del propcorrheatmap['ppimedangvel']
                   
plt.figure(figsize=(15,5))
sns.heatmap(propcorrheatmap.astype(float).corr(method='pearson'), vmin=-0.6, vmax=0.6, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_PropsCorrelation matrix.pdf', bbox_inches='tight')


#correlation heat map for ppi components
corrppiheatmap = dfresultT.loc[(dfresultT['event']>=7) & (dfresultT['event']<=10)].copy()

del corrppiheatmap['genotype']
del corrppiheatmap['%llc']
del corrppiheatmap['%nomovmt']
del corrppiheatmap['%react']
del corrppiheatmap['%slc']
del corrppiheatmap['AUC']
del corrppiheatmap['avgbend']
del corrppiheatmap['avgdisp']
del corrppiheatmap['avgdist']
del corrppiheatmap['avgduration']

del corrppiheatmap['avglat']
del corrppiheatmap['avgorient']
del corrppiheatmap['event']
del corrppiheatmap['fish#']
del corrppiheatmap['habavgbend']
del corrppiheatmap['habavgdisp']

del corrppiheatmap['habavgduration']
del corrppiheatmap['habmedduration']
del corrppiheatmap['habavgdist']
del corrppiheatmap['habavglat']
del corrppiheatmap['habavgorient']
del corrppiheatmap['habmedbend']
del corrppiheatmap['habmeddisp']
del corrppiheatmap['habmeddist']

del corrppiheatmap['habmedlat']
del corrppiheatmap['habmedorient']
del corrppiheatmap['habslc']
del corrppiheatmap['medianbend']
del corrppiheatmap['mediandisp']
del corrppiheatmap['mediandist']
del corrppiheatmap['medianduration']

del corrppiheatmap['medianlat']
del corrppiheatmap['medianorient']
del corrppiheatmap['slc/llc ratio']
del corrppiheatmap['totbendev']
del corrppiheatmap['totdispev']
del corrppiheatmap['totdistev']

del corrppiheatmap['totlatev']
del corrppiheatmap['totorientev']
del corrppiheatmap['totslcev']

plt.figure(figsize=(15,5))
sns.heatmap(corrppiheatmap.astype(float).corr(method='pearson'), vmin=-0.6, vmax=0.6, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_PPICorrelation matrix.pdf', bbox_inches='tight')

#correlation heat map for hab components
corrhabheatmap = dfresultT.loc[(dfresultT['event']>=12)].copy()

del corrhabheatmap['genotype']
del corrhabheatmap['%llc']
del corrhabheatmap['%nomovmt']
del corrhabheatmap['%react']
del corrhabheatmap['%slc']
del corrhabheatmap['AUC']
del corrhabheatmap['avgbend']
del corrhabheatmap['avgdisp']
del corrhabheatmap['avgdist']

del corrhabheatmap['avglat']
del corrhabheatmap['avgorient']
del corrhabheatmap['avgduration']
del corrhabheatmap['medianduration']
del corrhabheatmap['event']
del corrhabheatmap['fish#']
del corrhabheatmap['ppiavgbend']
del corrhabheatmap['ppiavgdisp']

del corrhabheatmap['ppiavgdist']
del corrhabheatmap['ppiavglat']
del corrhabheatmap['ppiavgorient']
del corrhabheatmap['ppiavgduration']
del corrhabheatmap['ppimedbend']
del corrhabheatmap['ppimeddisp']
del corrhabheatmap['ppimeddist']
del corrhabheatmap['ppimedduration']

del corrhabheatmap['ppimedlat']
del corrhabheatmap['ppimedorient']
del corrhabheatmap['ppislc']
del corrhabheatmap['medianbend']
del corrhabheatmap['mediandisp']
del corrhabheatmap['mediandist']

del corrhabheatmap['medianlat']
del corrhabheatmap['medianorient']
del corrhabheatmap['slc/llc ratio']
del corrhabheatmap['totbendev']
del corrhabheatmap['totdispev']
del corrhabheatmap['totdistev']

del corrhabheatmap['totlatev']
del corrhabheatmap['totorientev']
del corrhabheatmap['totslcev']

plt.figure(figsize=(15,5))
sns.heatmap(corrhabheatmap.astype(float).corr(method='pearson'), vmin=-0.6, vmax=0.6, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_HabCorrelation matrix.pdf', bbox_inches='tight')


# to do: time of movement --> how to categorize this? total with a 1 over total frames or certain number of frames ==stopped
# categorize movements into SLC/LLC? or other things?? Or is there any reason to do this rather than just using latency?
# should there be another category for subthreshold movement <1radian?                            
                            
# also, different programs for tracking + data processing?

# will this work if we then have more than one of these 10 events blocks








#disp = np.sqrt(np.square(np.diff(headY[5,:]))+ np.square(np.diff(headX[5,:])))
#avg_speed = np.mean(speed/pixframe)
#avg_speed1 = np.mean(speed1/pixframe)
#print(avg_speed)
#print(avg_speed1)



    #fig = plt.figure(figsize=(20,10))
    #plt.plot(speed)
    #plt.ylabel('Fish speed (px/frame)')
    #plt.xlabel('time')
    #fig.savefig(inDir+topName+'FishSpeed.png', dpi=100)

#total distance moved
#totdist = np.sum(speed[0:118])
#print(totdist)

#print(speed)
#print(speed1)

#time to first movement
    
