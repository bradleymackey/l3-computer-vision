# pre-process.py
# by bradley mackey
# computer vision assignment
# SSAIII 2018/19

"""
handlers for the pre-processing of images, in order to ensure the most accurate detections can be made
"""

import cv2
import numpy as np

# preprocessing pipeline:
# - gamma correct
# 

class ImagePreprocessor(object):
    """
    image preprocessor that will work on a specific image
    """

    def __init__(self, image):
        self.image = image

    def correct_gamma(self, gamma=1.0):
        # gamma correction code adapted from: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(self.image, table)

    def reduce_noise(self):
        pass
