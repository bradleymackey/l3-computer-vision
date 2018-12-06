

# from; https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
# also information from: https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

import cv2


class ShapeDetector(object):
    def __init__(self):
        pass
    
    def detect(self, c):
        # initialize the shape name and approximate the contour
        peri = cv2.arcLength(c, True)
        shape = "unidentified"
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
        
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape


    def is_shape_upright(self, c, img):
        rows,cols = img.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(img, cv2.DIST_L2,0.01,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        return (cols, lefty, righty)



