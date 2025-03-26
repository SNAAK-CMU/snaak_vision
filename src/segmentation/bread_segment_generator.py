"""
Script to define class for segmenting the first bread slice in the assembly area
"""

import numpy as np
import cv2

from segmentation.segment_utils import convert_mask_to_orig_dims, segment_from_hsv

############# Parameters ################

TRAY_BOX_PIX = (250, 60, 630, 300)  # xmin, ymin, xmax, ymax
HSV_LOWER_BOUND = (13, 70, 175)
HSV_UPPER_BOUND = (120, 255, 255)

#########################################


class BreadSegmentGenerator:

    TRAY_BOX_PIX = TRAY_BOX_PIX

    def __init__(self):
        self.crop_xmin, self.crop_ymin, self.crop_xmax, self.crop_ymax = TRAY_BOX_PIX
        self.hsv_lower_bound = np.array(HSV_LOWER_BOUND)
        self.hsv_upper_bound = np.array(HSV_UPPER_BOUND)

    def get_bread_mask(self, image):
        """
        Called from the vision node during ingredient placement.
        Returns the segmentation mask for the base bread slice in assembly area.
        """
        # Crop out tray region
        cropped_image = image[
            self.crop_ymin : self.crop_ymax, self.crop_xmin : self.crop_xmax
        ]
        cropped_image = cv2.GaussianBlur(cropped_image, (5, 5), 0)

        mask = segment_from_hsv(
            cropped_image, self.hsv_lower_bound, self.hsv_upper_bound
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

        return orig_mask
