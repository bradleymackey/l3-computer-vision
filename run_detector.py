# detector.py
# by bradley mackey
# computer vision assignment
# SSAIII 2018/19

import main
import cv2
import os
import numpy as np
import time
import math
import params
from utils import *
from sliding_window import *
from pre_process import *

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

# load SVM from file
try:
    print("loading SVM from file")
    svm = cv2.ml.SVM_load(params.HOG_SVM_PATH)
except:
    print("Missing files - SVM!")
    print("-- have you performed training to produce these files ?")
    exit()

# print some checks
print("svm size : ", len(svm.getSupportVectors()))
print("svm var count : ", svm.getVarCount())

show_scan_window_process = False

print()
print("--- beginning detections ---")
print()


for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = ""

    # from the left image filename get the corresponding right image
    filename_right = filename_left.replace("_L", "_R")
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # print out the filenames, as per the project requirements
    print(filename_left)
    print(filename_right,": nearest detected scene object: (X.Xm)")
    print()

    # check the file is a PNG file (left) and check a corresponding right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        # images are shown after person detections are made!

        
        left_copy = imgL.copy()

        # pre-process the left image
        left_processor = ImagePreprocessor(left_copy)
        left_processor.correct_gamma(gamma=1.7)
        left_processor.fix_luminance()
        left_processor.reduce_brightness()
        left_copy = left_processor.image


        imgray = cv2.cvtColor(left_copy,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        left_copy, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #left_copy = cv2.Canny(left_copy, 255,1)


        print("-------- files loaded successfully --------")
        print()

        # for a range of different image scales in an image pyramid

        current_scale = -1
        detections = []
        rescaling_factor = 1.25

        ################################ for each re-scale of the image

        for resized in pyramid(left_copy, scale=rescaling_factor):

            # at the start our scale = 1, because we catch the flag value -1

            if current_scale == -1:
                current_scale = 1

            # after this rescale downwards each time (division by re-scale factor)

            else:
                current_scale /= rescaling_factor

            rect_img = resized.copy()

            # if we want to see progress show each scale

            if (show_scan_window_process):
                cv2.imshow('current scale',rect_img)
                cv2.waitKey(10)

            # loop over the sliding window for each layer of the pyramid (re-sized image)

            window_size = params.DATA_WINDOW_SIZE
            step = math.floor(resized.shape[0] / 16)

            if step > 0:

                ############################# for each scan window

                for (x, y, window) in sliding_window(resized, window_size, step_size=step):

                    # if we want to see progress show each scan window

                    if (show_scan_window_process):
                        cv2.imshow('current window',window)
                        key = cv2.waitKey(10) # wait 10ms

                    # for each window region get the HoG feature point descriptors

                    img_data = ImageData(window)
                    img_data.compute_hog_descriptor()

                    # generate and classify each window by constructing a BoW
                    # histogram and passing it through the SVM classifier

                    if img_data.hog_descriptor is not None:

                        #print("detecting with SVM ...")
                        retval, [result] = svm.predict(np.float32([img_data.hog_descriptor]))
                        #print(result)

                        # if we get a detection, then record it

                        if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:

                            # store rect as (x1, y1) (x2,y2) pair

                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])

                            # if we want to see progress show each detection, at each scale

                            if (show_scan_window_process):
                                cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                                cv2.imshow('current scale',rect_img)
                                cv2.waitKey(40)

                            rect *= (1.0 / current_scale)
                            detections.append(rect)

                ########################################################


        # For the overall set of detections (over all scales) perform
        # non maximal suppression (i.e. remove overlapping boxes etc).

        detections = non_max_suppression_fast(np.int32(detections), 0.4)

        # finally draw all the detection on the original LEFT image
        for rect in detections:
            
            cv2.rectangle(imgL, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

        # draw the contours on the left copy image for the moment
        cv2.drawContours(left_copy, contours, -1, (0,255,0), 3)
        cv2.imshow('left image',imgL)
        cv2.imshow('left image copy',left_copy)
        cv2.imshow('right image',imgR)

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')

        key = cv2.waitKey(140 * (not(pause_playback))) & 0xFF # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("left.png", imgL)
            cv2.imwrite("right.png", imgR)
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity)
        elif (key == ord('p')):     # pause (on next frame)
            pause_playback = not(pause_playback)
    else:
            print("-- files skipped (perhaps one is missing or not PNG)")
            print()

# close all windows, we are done.
cv2.destroyAllWindows()

