# detector.py
# by bradley mackey
# computer vision assignment
# SSAIII 2018/19


import main
import cv2
import os
import numpy as np

#####################################################################

# [[[fragments adapted from stereo_disparity.py]]]
# useful because already deals with differences between the left and right images, useful code for cycling through all images
# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

# the path to the image data
master_path_to_dataset = main.master_path_to_dataset
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns
skip_forward_file_pattern = "" # set to timestamp to skip forward to

crop_disparity = False # display full or cropped disparity image
pause_playback = False # pause until key press after each image

#####################################################################

# resolve full directory location of data set for left / right images
full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right)

# get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left))

for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)
    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = ""

    # from the left image filename get the corresponding right image
    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);

    # for sanity print out these filenames
    print(full_path_filename_left)
    print(full_path_filename_right)
    print()

    # check the file is a PNG file (left) and check a corresponding right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        cv2.imshow('left image',imgL)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        cv2.imshow('right image',imgR)

        print("-- files loaded successfully");
        print()

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')



# close all windows, we are done.
cv2.destroyAllWindows()

