# detector.py
# by bradley mackey
# computer vision assignment
# SSAIII 2018/19

import cv2
import os
import numpy as np
import time
import math
import params
import random
import csv
from utils import *
from sliding_window import *
from pre_process import *
from shape_detector import ShapeDetector
from stereo_to_3d import *

#####################################################################

# [[[fragments adapted from stereo_disparity.py]]]
# useful because already deals with differences between the left and right images, useful code for cycling through all images
# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Department of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

max_disparity = 128
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

# the path to the image data
master_path_to_dataset = params.master_path_to_dataset
directory_to_cycle_left = "left-images"
directory_to_cycle_right = "right-images"

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

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

def pre_process_image(image):
    """
    pre-processes image for better image detections
    """
    left_processor = ImagePreprocessor(image)
    left_processor.correct_gamma(gamma=1.7)
    left_processor.fix_luminance()
    left_processor.reduce_brightness()
    image = left_processor.image
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def detection_region(left,right,top,bottom):
    """
    creates a detection region rect
    """
    interest_ur = (left,top)
    interest_ll = (right, bottom)
    region_rect = [interest_ur[0],interest_ur[1],interest_ll[0],interest_ll[1]]
    return region_rect



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

    # check the file is a PNG file (left) and check a corresponding right image
    # actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        
        left_copy = imgL.copy()

        # pre-process the left image
        pre_process_image(left_copy)

        # imgray = cv2.cvtColor(left_copy,cv2.COLOR_BGR2GRAY)
        # ret,thresh = cv2.threshold(imgray,127,255,0)
        # kernel = np.ones((4,4),np.uint8)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
        # left_copy, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #left_copy = cv2.Canny(left_copy, 255,1)

        print("-------- files loaded successfully --------")
        print()

        detections = []

        ################ selective search ######################

        # ss.setBaseImage(left_copy)

        # # Switch to fast but low recall Selective Search method
        # # ss.switchToSelectiveSearchFast()

        # # Switch to high recall but slow Selective Search method (slower)
        # ss.switchToSelectiveSearchQuality()

        # # run selective search segmentation on input image
        # rects = ss.process()
        # print('Total Number of Region Proposals: {}'.format(len(rects)))

        # # number of region proposals to show
        # numShowRects = 100

        # # iterate over all the region proposals
        # for i, rect in enumerate(rects):
        #     x, y, w, h = rect
        #     detected_rect = [x,y,x+w,y+w]
        #     img_region = left_copy[y:y+h,x:x+w]


        #     img_data = ImageData(img_region)
        #     img_data.compute_hog_descriptor()

        #     # generate and classify each window by constructing a BoW
        #     # histogram and passing it through the SVM classifier

        #     if img_data.hog_descriptor is not None:

        #         #print("detecting with SVM ...")
        #         retval, [result] = svm.predict(np.float32([img_data.hog_descriptor]))
        #         #print(result)

        #         # if we get a detection, then record it

        #         if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:

        #             cv2.imshow("thing", img_region)

        #             # store rect as (x1, y1) (x2,y2) pair
        #             rect = np.float32([x, y, x + w, y + h])
        #             detections.append(rect)
        #     else:
        #         continue

        ################################
        


        ######## sliding window ##############


        # for a range of different image scales in an image pyramid

        current_scale = -1
        
        rescaling_factor = 1.32

        # for each re-scale of the image

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
        ################################



        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

        # perform preprocessing - raise to the power, as this subjectively appears
        # to improve subsequent disparity calculation

        grayL = np.power(grayL, 0.75).astype('uint8')
        grayR = np.power(grayR, 0.75).astype('uint8')


        # compute disparity image from undistorted and rectified stereo images
        # that we have loaded
        # (which for reasons best known to the OpenCV developers is returned scaled by 16)

        disparity = stereoProcessor.compute(grayL,grayR);

        # filter out noise and speckles (adjust parameters as needed)

        dispNoiseFilter = 5; # increase for more agressive filtering
        cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

        # scale the disparity to 8-bit for viewing
        # divide by 16 and convert to 8-bit image (then range of values should
        # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
        # so we fix this also using a initial threshold between 0 and max_disparity
        # as disparity=-1 means no disparity available

        _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO)
        disparity_scaled = (disparity / 16.).astype(np.uint8)

        # crop disparity to chop out left part where there are with no disparity
        # as this area is not seen by both cameras and also
        # chop out the bottom area (where we see the front of car bonnet)

        if (crop_disparity):
            width = np.size(disparity_scaled, 1)
            disparity_scaled = disparity_scaled[0:390,135:width]

        # display image (scaling it to the full 0->255 range based on the number
        # of disparities in use for the stereo part)

        cv2.imshow("disparity", (disparity_scaled * (256. / max_disparity)).astype(np.uint8))



        # For the overall set of detections (over all scales) perform
        # non maximal suppression (i.e. remove overlapping boxes etc).

        detections = non_max_suppression_fast(np.int32(detections), 0.8)

        height, width = imgL.shape[:2]

        # left pedestrian region for detections
        interest_1_ur = (0,int(height/7))
        interest_1_ll = (int(width/2.4), int(4*height/5))
        region_1_rect = [interest_1_ur[0],interest_1_ur[1],interest_1_ll[0],interest_1_ll[1]]

        # right pedestrian region for detections
        interest_2_ur = (int(width/2),int(height/7))
        interest_2_ll = (width, int(4*height/5))
        region_2_rect = [interest_2_ur[0],interest_2_ur[1],interest_2_ll[0],interest_2_ll[1]]

        # non-exclusive bottom region for detections
        partial_reg_1_ur = (0,int(height/2))
        partial_reg_1_ll = (width,height)
        partial_region_1_rect = [partial_reg_1_ur[0],partial_reg_1_ur[1],partial_reg_1_ll[0],partial_reg_1_ll[1]]

        # TESTING - draw these detection regions
        cv2.rectangle(imgL, interest_1_ur, interest_1_ll, (0, 255, 255), 2)
        cv2.rectangle(imgL, interest_2_ur, interest_2_ll, (0, 255, 255), 2)
        cv2.rectangle(imgL, partial_reg_1_ur, partial_reg_1_ll, (0, 255,0), 2)

        # finally draw all the detection on the original LEFT image
        valid_rects = []
        for rect in detections:
            # ensure fully enclosed by one of the compulsory regions
            if (rectContainsRectFully(region_1_rect,rect)) or (rectContainsRectFully(region_2_rect,rect)):
                # ensure not fully contained in a non-exclusive region
                if rectContainsRectFully(partial_region_1_rect,rect) == False:
                    # ensure that rect is not too small and roughly the correct shape
                    if rectGoodSize(width, height, rect):
                        valid_rects.append(rect)
        
        # draw all valid rects after compression
        valid_rects = compressDetections(valid_rects)
        for rect in valid_rects:
            point1 = (rect[0], rect[1])
            point2 = (rect[2], rect[3])
            cv2.rectangle(imgL, point1, point2, (0, 0, 255), 3)


        #points = project_disparity_to_3d(disparity_scaled, max_disparity, imgL)
        detected_distances = []
        for rect in valid_rects:
            centre = rect_centre(rect)
            dist = avg_dist_for_points_surrounding(centre,disparity_scaled,max_disparity)
            cv2.putText(imgL, "P: {:.2f}m".format(dist), (rect[0],rect[3]+20),cv2.FONT_HERSHEY_DUPLEX, 0.75, (0,0,255), thickness = 1)

        min_distance_string = None
        try:
            min_d = min(detected_distances)
            min_distance_string = ": nearest detected scene object: ({:.2f}m)".format(min_d)
        except:
            min_distance_string = ": no objects detected"
        # print out the filenames, as per the project requirements
        print(filename_left)
        print(filename_right,min_distance_string)
        print()


        #cv2.drawContours(imgL, contours, -1, (0,255,0), 1)
        cv2.imshow('left image',imgL)
        cv2.imshow('left image copy',left_copy)
        cv2.imshow('right image',imgR)


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

