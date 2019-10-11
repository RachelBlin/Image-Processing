#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 09:10:51 2019

@author: rblin
"""

# This script will be used to test the values returned by my functions to 
# compute the polarimetric parameters. I'll use a pacth of a polarimetric image 
# and compute the intensities, the Stokes vector and verify all the constraints 
# by hand to compare those results with my program.

import numpy as np
import imageio

# Loading the image I'll use to get the ground truth
image = imageio.imread("ground_truth.tiff")
#image = imageio.imread("/media/rblin/EC42-B858/test_polar_2/Raw/file_1000x1000_001000.tiff")
# Verifying the type of the values is uint16
if type(image[0,0]) != np.uint16:
    print("Image read not uint16")

# Getting a portion of the image. We want it to be structed in a way to respect 
# the superpixels constitution. For a reminder, a superpixels is composed of 
# the intensities in the following order:
#  45 | 0
# ---------
# 90  | 135
# To get a patch of that image that can be read this way, we have to take a 
# portion of it, starting with a even index and ending with an even index 
# because Python if we take a list from i:j in Python it'll return the list 
# from the ith to the (j-i)th value.
# For the ground truth, I choosed to work on image(400:407, 140:147)
# The values read at those indexes should return the following matrix:
# ground_truth = 
#9056	13120	9360	14448	9776	14736	9584	11360
#10192	19088	9888	19088	10768	18576	9712	14704
#9712	14448	9632	14336	10192	15408	9568	11152
#10448	19648	9648	19376	10352	18368	9344	14976
#9504	14544	9584	15472	10032	16320	9200	12864
#10048	18208	10432	20032	10352	18912	10272	15424
#9792	13616	9824	15568	9872	15440	9248	12672
#9904	17424	10464	20112	10416	18464	9360	14864

check_reading_patch = np.array([
        [9056, 13120, 9360, 14448, 9776, 14736, 9584, 11360],
        [10192, 19088, 9888, 19088, 10768, 18576, 9712, 14704],
        [9712, 14448, 9632, 14336, 10192, 15408, 9568, 11152],
        [10448, 19648, 9648, 19376, 10352, 18368, 9344, 14976],
        [9504, 14544, 9584, 15472, 10032, 16320, 9200, 12864],
        [10048, 18208, 10432, 20032, 10352, 18912, 10272, 15424],
        [9792, 13616, 9824, 15568, 9872, 15440, 9248, 12672],
        [9904, 17424, 10464, 20112, 10416, 18464, 9360, 14864]], 
dtype=np.uint16)

# Checking the manually entered pixels are in uint16
if type(check_reading_patch[0,0]) != np.uint16:
    print("Manually entered ground truth not uint16")

# Checking if the ground truth read from the image is in uint16
ground_truth = image[400:408, 140:148]
if type(ground_truth[0,0]) != np.uint16:
    print("Read ground truth not uint16")
    
# Checking if the ground truth read from the image is the same than the 
# manually entered pixelwise
    
diff = check_reading_patch - ground_truth
if np.any(diff != np.zeros((check_reading_patch.shape[0], check_reading_patch.shape[1]), dtype=np.uint16)):
    print("Manually entered ground truth not equal to read ground truth")

# Saving patch image to get the ground truth in .tiff so that it can be used by
# eveyone for tests

imageio.imwrite('patch_ground_truth.tiff', ground_truth)

# Verifying that the save image is in uint16

final_image = imageio.imread("patch_ground_truth.tiff")
if type(final_image[0,0]) != np.uint16:
    print("Image not saved in uint16")

# Getting the intensities. The ground truth we should get manually are the 
# followings:
# I0 =
#13120	14448	14736	11360
#14448	14336	15408	11152
#14544	15472	16320	12864
#13616	15568	15440	12672
# I45 = 
#9056	9360	9776	9584
#9712	9632	10192	9568
#9504	9584	10032	9200
#9792	9824	9872	9248
# I90 =
#10192	9888	10768	9712
#10448	9648	10352	9344
#10048	10432	10352	10272
#9904	10464	10416	9360
# I135 =
#19088	19088	18576	14704
#19648	19376	18368	14976
#18208	20032	18912	15424
#17424	20112	18464	14864

I0 = final_image[0:final_image.shape[0]:2, 1:final_image.shape[0]:2] # I0
I45 = final_image[0:final_image.shape[0]:2, 0:final_image.shape[0]:2] # I45
I90 = final_image[1:final_image.shape[0]:2, 0:final_image.shape[0]:2] # I90
I135 = final_image[1:final_image.shape[0]:2, 1:final_image.shape[0]:2] # I135

I0_truth = np.array([[13120, 14448, 14736, 11360],
[14448, 14336, 15408, 11152],
[14544, 15472, 16320, 12864],
[13616, 15568, 15440, 12672]], dtype=np.uint16)

I45_truth = np.array([[9056, 9360, 9776, 9584],
[9712, 9632, 10192, 9568],
[9504, 9584, 10032, 9200],
[9792, 9824, 9872, 9248]], dtype=np.uint16)
    
I90_truth = np.array([[10192, 9888, 10768, 9712],
[10448, 9648, 10352, 9344],
[10048, 10432, 10352, 10272],
[9904, 10464, 10416, 9360]], dtype=np.uint16)
    
I135_truth = np.array([[19088, 19088, 18576, 14704],
[19648, 19376, 18368, 14976],
[18208, 20032, 18912, 15424],
[17424, 20112, 18464, 14864]], dtype=np.uint16)

# Checking if the truth intensities are the same than the read intensities
    
# I0
if np.any(I0 != I0_truth):
    print("Manually entered I0 not equal to read I0")
# I45
if np.any(I45 != I45_truth):
    print("Manually entered I45 not equal to read I45")
# I90
if np.any(I90 != I90_truth):
    print("Manually entered I90 not equal to read I90")
# I135
if np.any(I135 != I135_truth):
    print("Manually entered I135 not equal to read I135")
    
# Computing the matrices I0 + I90 and I45 + 135. The ground truth we manually 
# get are the followings:
# I0 + I90 =
#23312	24336	25504	21072
#24896	23984	25760	20496
#24592	25904	26672	23136
#23520	26032	25856	22032
# I45 + I135 =
#28144	28448	28352	24288
#29360	29008	28560	24544
#27712	29616	28944	24624
#27216	29936	28336	24112

# I0 + I90  manual
Stokes_0_first_truth = np.array([[23312, 24336, 25504, 21072],
[24896, 23984, 25760, 20496],
[24592, 25904, 26672, 23136],
[23520, 26032, 25856, 22032]], dtype=np.uint16)

# I45 + I135 manual
Stokes_0_second_truth = np.array([[28144, 28448, 28352, 24288],
[29360, 29008, 28560, 24544],
[27712, 29616, 28944, 24624],
[27216, 29936, 28336, 24112]], dtype=np.uint16)

# I0 + I90
Stokes_0_first = I0 + I90
# I45 + I135
Stokes_0_second = I45 + I135

# Checking if the computed I0 + I90 and manual I0 + I90 are equal:
if np.any(Stokes_0_first_truth != Stokes_0_first):
    print("Manually computed I0 + I90 not equal to computed I0 + I90")
    
# Checking if the computed I45 + I135 and manual I45 + I135 are equal:
if np.any(Stokes_0_second_truth != Stokes_0_second):
    print("Manually computed I45 + I135 not equal to computed I45 + I135")
    
# Computing the difference (I0 + I90) - (I45 + I135). The ground truth we 
# manually get is the following:
#I0 + I90 - (I45 + I135) =
#-4832 -4112 -2848 -3216
#-4464 -5024 -2800 -4048
#-3120 -3712 -2272 -1488
#-3696 -3904 -2480 -2080
    
diff_intensities_truth = np.array([[-4832, -4112, -2848, -3216],
[-4464, -5024, -2800, -4048],
[-3120, -3712, -2272, -1488],
[-3696, -3904, -2480, -2080]], dtype=np.uint16)
    
diff_intensities = Stokes_0_first - Stokes_0_second

# Checking if (I0 + I90) - (I45 + I135) manually computed and 
# (I0 + I90) - (I45 + I135) computed bu this program are the sam:
if np.any(diff_intensities != diff_intensities_truth):
    print("Manually computed (I0 + I90) - (I45 + I135) not equal to computed (I0 + I90) - (I45 + I135)")