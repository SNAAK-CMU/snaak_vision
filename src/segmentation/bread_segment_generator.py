"""
Script to define class for segmenting the first bread slice in the assembly area
"""

import numpy as np
import cv2
import heapq
from segmentation.segment_utils import convert_mask_to_orig_dims, segment_from_hsv, contour_segmentation

############# Parameters ################

TRAY_BOX_PIX = (200, 20, 630, 350)  # xmin, ymin, xmax, ymax
HSV_LOWER_BOUND = (10, 50, 100)
HSV_UPPER_BOUND = (40, 255, 255)

#########################################


class BreadSegmentGenerator:

    def __init__(self):
        self.crop_xmin, self.crop_ymin, self.crop_xmax, self.crop_ymax = TRAY_BOX_PIX
        self.hsv_lower_bound = np.array(HSV_LOWER_BOUND)
        self.hsv_upper_bound = np.array(HSV_UPPER_BOUND)

    def get_bread_placement_mask(self, image):
        """
        Called from the vision node during ingredient placement.
        Returns the segmentation mask for the base bread slice in assembly area.
        """
        # Crop out tray region
        cropped_image = image[
            self.crop_ymin : self.crop_ymax, self.crop_xmin : self.crop_xmax
        ]
        # cropped_image = cv2.GaussianBlur(cropped_image, (5, 5), 0)

        mask = segment_from_hsv(
            cropped_image, self.hsv_lower_bound, self.hsv_upper_bound
        )

        cv2.imwrite(
            "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/hsv_mask.jpg",
            mask,
        )

        kernel = np.ones((5, 5), np.uint8)
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed_mask = cv2.morphologyEx(
            opened_mask, cv2.MORPH_CLOSE, kernel, iterations=1
        )

        orig_mask = convert_mask_to_orig_dims(
            closed_mask,
            image,
            self.crop_xmin,
            self.crop_ymin,
            self.crop_xmax,
            self.crop_ymax,
        )

        # cv2.imshow("bread_hsv_mask", orig_mask)
        # cv2.imshow("Orig image", image)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return orig_mask

    def get_bread_pickup_point(self, image):
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0) 
        top_left_crop = (390, 65)
        bottom_right_crop = (510, 280)
        crop_mask = np.zeros_like(image)
        crop_x_start, crop_y_start = top_left_crop
        crop_x_end, crop_y_end = bottom_right_crop

        crop_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end] = 255
        crop = cv2.bitwise_and(blurred_image, crop_mask)

        contours, heirarchy = contour_segmentation(crop, binary_threshold=150, show_image=False, show_separate_contours=False, show_steps=False, close_kernel_size=7, open_kernel_size=7, segment_type='edges', edges_thresholds=(30, 50))
        min_area = 6000
        max_area = 17000

        contour_heap = [] # min heap
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            print(f"Contour area: {contour_area}")
            if min_area <= contour_area <= max_area:
                heapq.heappush(contour_heap, (contour_area, contour))

        if len(contour_heap) == 0:
            raise Exception("No contours found, could not find bread pickup point")
        else:
            bread_contour = heapq.heappop(contour_heap)[1]

        M = cv2.moments(bread_contour)
        cX = int(M["m10"] / M["m00"]) 
        cY = int(M["m01"] / M["m00"])  
        cv2.drawContours(image, [bread_contour], -1, (0, 255, 0), 3) 
        cv2.circle(image, (cX, cY), 5, (0, 255, 255), -1)
        return cX, cY