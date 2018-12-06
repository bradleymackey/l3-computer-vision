#####################################################################

# Example : project SGBM disparity to 3D points for am example pair
# of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Deparment of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
import random
import csv
import params

master_path_to_dataset = params.master_path_to_dataset # dataset populated from global parameters file
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed

#####################################################################

# fixed camera parameters for this stereo setup (from calibration)
# DO NOT CHANGE

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0
image_centre_w = 474.5

#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, max_disparity, rgb=[]):

    points = []

    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    height, width = disparity.shape[:2]

    print("disparity height:",height)
    print("disparity width:",width)

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then we get reasonable scaling in X and Y output if we change
    # Z to Zmax in the lines X = ....; Y = ...; below

    # Zmax = ((f * B) / 2);

    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = (f * B) / disparity[y,x]

                X = ((x - image_centre_w) * Z) / f
                Y = ((y - image_centre_h) * Z) / f

                # add to points

                if(rgb.size > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]])
                else:
                    points.append([X,Y,Z])

    return points


def avg_dist_for_points_surrounding(point,disparity,max_disparity):
    """
    gets the approximate distance from the camera for a given point in the disparity image
    (similar in operation to project_disparity_to_3d, but only returns a single DEPTH value)
    """

    # potential points stored here
    points = []

    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    height, width = disparity.shape[:2]

    # define the regions that we will fetch disparity values for
    min_x = max(0,point[0]-200)
    max_x = min(width,point[0]+200)
    min_y = max(0,point[1]-200)
    max_y = min(height,point[1]+200)

    print("y range is:",min_y,max_y)
    print("x range is:",min_x,max_x)

    for y in range(min_y,max_y): # 0 - height is the y axis index
        for x in range(min_x,max_x): # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = (f * B) / disparity[y,x]

                # add to points
                points.append(Z)


    print("the points:",points)

    if points == []:
        return 5.134
    else:
        return np.percentile(points,17)




#####################################################################

# project a set of 3D points back the 2D image domain

def project_3D_points_to_2D_image_points(points):

    points2 = []

    # calc. Zmax as per above

    # Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2;

    for i1 in range(len(points)):

        # reverse earlier projection for X and Y to get x and y again

        x = ((points[i1][0] * camera_focal_length_px) / points[i1][2]) + image_centre_w
        y = ((points[i1][1] * camera_focal_length_px) / points[i1][2]) + image_centre_h
        points2.append([x,y])

    return points2

#####################################################################

