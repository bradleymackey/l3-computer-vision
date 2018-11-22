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
        self.image = cv2.LUT(self.image, table)

    def smooth(self):
        """
        performs a simple smoothing convolution to the image
        code from: https://docs.opencv.org/3.1.0/d4/d13/tutorial_py_filtering.html
        """
        kernel = np.ones((3,3),np.float32)/9
        self.image = cv2.filter2D(self.image,-1,kernel)

    def fix_luminance(self):
        """
        fixes illumination in images, giving a smoother image overall
        code from: https://stackoverflow.com/a/39744436/3261161
        """
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        self.image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def reduce_brightness(self):
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 0
        v[v < lim] = 0
        v[v <= lim] -= 50

        final_hsv = cv2.merge((h, s, v))
        self.image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
