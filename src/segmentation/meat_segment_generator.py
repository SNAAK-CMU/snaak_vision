"""
This file contains definition of a class for segmenting the top slice of meat
"""

import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from segmentation.segment_utils import (
    get_hsv_range,
    segment_from_hsv,
    calc_bbox_from_mask,
    convert_mask_to_orig_dims,
    keep_largest_blob,
)

######## Parameters ##########

CROP_XMIN = 300
CROP_YMIN = 50
CROP_XMAX = 430
CROP_YMAX = 280

CROP_XMIN_HSV = 330
CROP_YMIN_HSV = 100
CROP_XMAX_HSV = 400
CROP_YMAX_HSV = 250

MEAT_W = 100
MEAT_H = 100

LOWER_HSV_WHITE = np.array([0, 0, 66])
UPPER_HSV_WHITE = np.array([176, 53, 255])

###############################


class MeatSegmentGenerator:

    def __init__(self):
        """
        Initializes and loads SAM2 model for meat segmentation
        """
        self.sam2_checkpoint = "/home/snaak/Documents/manipulation_ws/src/sam2/checkpoints/sam2.1_hiera_small.pt"
        # self.sam2_checkpoint = "/home/user/sam2/checkpoints/sam2.1_hiera_small.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        self.__create_sam_predictor()

        # Set up
        self.got_first_img = False

        self.meat_w = MEAT_W
        self.meat_h = MEAT_H

        self.lower_hsv = None
        self.upper_hsv = None

        self.lower_hsv_white = LOWER_HSV_WHITE
        self.upper_hsv_white = UPPER_HSV_WHITE

        self.crop_xmin_hsv = CROP_XMIN_HSV
        self.crop_ymin_hsv = CROP_YMIN_HSV
        self.crop_xmax_hsv = CROP_XMAX_HSV
        self.crop_ymax_hsv = CROP_YMAX_HSV

        self.crop_xmin = CROP_XMIN
        self.crop_ymin = CROP_YMIN
        self.crop_xmax = CROP_XMAX
        self.crop_ymax = CROP_YMAX

    def __create_sam_predictor(self):
        """
        Initializes and loads SAM2 model
        """
        sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(sam2_model)

    def __segment_top_meat_e2e(self, image):
        """
        Input: image (numpy array): The input image to segment
        Output: mask (numpy array): The segmented mask of the top slice of meat

        """
        # Mask to remove white noise
        cropped_image = image[
            self.crop_ymin : self.crop_ymax, self.crop_xmin : self.crop_xmax
        ]
        white_mask = segment_from_hsv(
            cropped_image, self.lower_hsv_white, self.upper_hsv_white
        )
        not_white_mask = ~white_mask

        # Get HSV Mask
        hsv_mask = segment_from_hsv(cropped_image, self.lower_hsv, self.upper_hsv)
        kernel = np.ones((5, 5), np.uint8)
        hsv_mask = cv2.erode(hsv_mask, kernel, iterations=1)

        # Remove noise from hsv mask
        final_mask = cv2.bitwise_and(hsv_mask, not_white_mask)
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.erode(final_mask, kernel, iterations=1)
        final_mask = keep_largest_blob(final_mask)

        # Get bbox for all meat
        bounding_box = calc_bbox_from_mask(final_mask)

        # Run sam using all meat bbox
        self.predictor.set_image(cropped_image)
        masks, scores, logits = self.predictor.predict(
            box=bounding_box[None, :], multimask_output=True
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        all_meat_mask = masks[np.argmax(scores)]

        # Refine SAM mask
        all_meat_mask_orig = convert_mask_to_orig_dims(
            all_meat_mask,
            image,
            self.crop_xmin,
            self.crop_ymin,
            self.crop_xmax,
            self.crop_ymax,
        )
        kernel = np.ones((5, 5), np.uint8)
        all_meat_mask_orig = cv2.erode(all_meat_mask_orig, kernel, iterations=1)

        y_indices, x_indices = np.where(all_meat_mask_orig * 255 == 255)

        # Ensure there are white pixels
        if y_indices.size == 0 or x_indices.size == 0:
            print("No white object found!")

        # Find bounding box of the vertical rectangle
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)

        # Find coordinates of top meat box
        meat_bottom_centre_x = (xmin + xmax) // 2
        meat_bottom_y = ymax
        meat_top_y = ymax - self.meat_h
        meat_right_x = meat_bottom_centre_x + (self.meat_w // 2)
        meat_left_x = meat_bottom_centre_x - (self.meat_w // 2)
        meat_box = np.array([meat_left_x, meat_top_y, meat_right_x, meat_bottom_y])

        center_x = (meat_left_x + meat_right_x) // 2
        center_y = (meat_top_y + meat_bottom_y) // 2

        return center_x, center_y

    def __calibrate_color_thresholding(self, image):
        """
        Calculates hsv range for segmenting meat by sampling the image
        """
        # Get HSV range for the first image
        cropped_image = image[
            self.crop_ymin_hsv : self.crop_ymax_hsv,
            self.crop_xmin_hsv : self.crop_xmax_hsv,
        ]

        # cv2.imwrite("/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/meat_hsv_crop.jpg", cropped_image)

        self.lower_hsv, self.upper_hsv = get_hsv_range(cropped_image)

    def get_top_meat_slice_xy(self, image):
        """
        Input: image (numpy array): The input image to segment
        Output: mask (numpy array): The segmented mask of the top slice of meat
        """

        # If first image, get HSV range
        if not self.got_first_img:
            self.got_first_img = True
            self.__calibrate_color_thresholding(image)

        # Get middle of top meat slice
        x, y = self.__segment_top_meat_e2e(image)

        return x, y
