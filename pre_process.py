# pre-process.py
# by bradley mackey
# computer vision assignment
# SSAIII 2018/19

"""
handlers for the pre-processing of images, in order to ensure the most accurate detections can be made
"""

import cv2

# preprocessing pipeline:
# - gamma correct
# 

class ImagePreprocessor(object):
    """
    image preprocessor that will work on a specific image
    """

    def __init__(self, image):
        self.image = image

    def correct_gamma(self):
        pass

    def reduce_noise(self):
        pass

