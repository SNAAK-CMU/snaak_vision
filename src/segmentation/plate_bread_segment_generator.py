"""
Script to define class for segmenting the pickup point on the acrylic sheet
"""
import cv2
import numpy as np

class PlateBreadSegementGenerator():
    def __init__(self):
        self.lower_blue = np.array([100,40,40])
        self.upper_blue = np.array([150,255,255]) 
        self.center_distance_thresh = 10

    def get_bread_pickup_point(self, image):
        '''
        Function to get the pickup point of the bread slice on the plate

        Inputs:
            image: RGB image of the plate
        Outputs:
            cX: x coordinate of the pickup point
            cY: y coordinate of the pickup point
            bottom_y: y coordinate of the bottom of the acrylic sheet (as seen by the camera, in arm frame the left most point of the acrylic sheet)
        '''
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue) 
        res = cv2.bitwise_and(image,image, mask= mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,10,255,0)
        
        # crop image
        # Crop locations
        top_left_crop = (390, 65)
        #top_right_crop = (490, 65)
        #bottom_left_crop = (390, 285)
        bottom_right_crop = (490, 265)
        crop_mask = np.zeros_like(thresh)
        crop_x_start, crop_y_start = top_left_crop
        crop_x_end, crop_y_end = bottom_right_crop

        crop_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end] = 255
        thresh = cv2.bitwise_and(thresh, crop_mask)

        min_area = 3000
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
                M = cv2.moments(cnt)
                # m00: Total area of the contour.
                # m10: Sum of the x-coordinates weighted by pixel intensities.
                # m01: Sum of the y-coordinates weighted by pixel intensities.
                # centroid is given by m10 / m00
                cX = int(M["m10"] / M["m00"]) 
                cY = int(M["m01"] / M["m00"])                
                #cv2.circle(image, (cX, cY), 5, (255, 0, 255), -1)

                cv2.drawContours(image, [approx], -1, (0, 255, 0), 3) 
                # bottom_y = -1
                # for point in cnt: 
                #     x, y = point[0]
                #     if abs(x - cX) <= self.center_distance_thresh:
                #         bottom_y = max(y, bottom_y)
                # if bottom_y == -1: 
                bottom_y = np.max(cnt[:, 0, 1])

                top_y = -1

                # Because of arcs in bread, center y point can be unreliable
                # To fix, find lowest y point near midline of plate that is not on the bottom line, this will correspond
                # to the lowest point of the bread
                for point in cnt: 
                    x, y = point[0]
                    if abs(x - cX) <= self.center_distance_thresh and (y < bottom_y - (bottom_y - cY) / 4):
                        top_y = max(y, top_y)
                cY = int(top_y + (bottom_y - top_y) / 2)

                if top_y  == -1: 
                    cY - (bottom_y - cY) / 2

                return cX, cY, bottom_y
        raise Exception("No suitable bread pickup point found")
    