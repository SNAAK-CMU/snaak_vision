"""
This file contains definition of a class for segmenting the top slice of cheese
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
)

######## Parameters ##########

CROP_XMIN_HSV = 350
CROP_YMIN_HSV = 100
CROP_XMAX_HSV = 400
CROP_YMAX_HSV = 250

CROP_XMIN = 320
CROP_YMIN = 60
CROP_XMAX = 430
CROP_YMAX = 280

###############################


class CheeseSegmentGenerator:

    def __init__(self):

        # TODO: Change paths according deployment environment
        self.sam2_checkpoint = "/home/snaak/Documents/manipulation_ws/src/sam2/checkpoints/sam2.1_hiera_small.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        self.__create_sam_predictor()

        # Set up
        self.got_first_img = False
        self.lower_hsv = None
        self.upper_hsv = None

        self.crop_xmin_hsv = CROP_XMIN_HSV
        self.crop_ymin_hsv = CROP_YMIN_HSV
        self.crop_xmax_hsv = CROP_XMAX_HSV
        self.crop_ymax_hsv = CROP_YMAX_HSV

    def __create_sam_predictor(self):
        """
        Initializes and loads SAM2 model
        """

        sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(sam2_model)

    def __segment_top_cheese_2e2(self, image):
        """
        Input: RGB image
        Output: Binary mask of the top cheese slice
        Uses CV pipeline to get top slice of cheese from the rgb image
        """
        # Crop out the inner borders of the bin
        crop_xmin = CROP_XMIN
        crop_ymin = CROP_YMIN
        crop_xmax = CROP_XMAX
        crop_ymax = CROP_YMAX
        cropped_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        # Construct prompt using hsv mask
        mask = segment_from_hsv(cropped_image, self.lower_hsv, self.upper_hsv)
        bounding_box = calc_bbox_from_mask(mask)

        # Run SAM
        self.predictor.set_image(cropped_image)
        masks, scores, logits = self.predictor.predict(
            box=bounding_box[None, :], multimask_output=True
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        all_cheese_mask = masks[np.argmax(scores)]

        # Process mask from SAM
        all_cheese_mask_orig = convert_mask_to_orig_dims(
            all_cheese_mask, image, crop_xmin, crop_ymin, crop_xmax, crop_ymax
        )
        kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel
        all_cheese_mask_orig = cv2.erode(all_cheese_mask_orig, kernel, iterations=1)

        # Get top cheese box
        # Cheese dims in pix
        cheese_w = 90
        cheese_h = 90
        y_indices, x_indices = np.where(all_cheese_mask_orig * 255 == 255)

        # Ensure there are white pixels
        if y_indices.size == 0 or x_indices.size == 0:
            print("No white object found!")

        # Find bounding box of the vertical rectangle
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        cheese_bottom_centre_x = (xmin + xmax) // 2
        cheese_bottom_y = ymax
        cheese_top_y = ymax - cheese_h
        cheese_right_x = cheese_bottom_centre_x + (cheese_w // 2)
        cheese_left_x = cheese_bottom_centre_x - (cheese_w // 2)
        cheese_box = np.array(
            [cheese_left_x, cheese_top_y, cheese_right_x, cheese_bottom_y]
        )

        # Run SAM on top cheese box
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            box=cheese_box[None, :], multimask_output=True
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        top_cheese_mask = masks[np.argmax(scores)]
        kernel = np.ones((7, 7), np.uint8)
        top_cheese_mask = cv2.erode(top_cheese_mask, kernel, iterations=1)

        return top_cheese_mask

    def __calibrate_color_thresholding(self, image):
        """
        Calculates hsv range for segmenting cheese by sampling the image
        """
        cropped2_image = image[
            self.crop_ymin_hsv : self.crop_ymax_hsv,
            self.crop_xmin_hsv : self.crop_xmax_hsv,
        ]
        self.lower_hsv, self.upper_hsv = get_hsv_range(cropped2_image)

    def get_top_cheese_slice(self, image):
        """
        Called from the vision node during cheese pickup. Return the segmented top cheese slice
        """

        if not self.got_first_img:
            self.got_first_img = True
            self.__calibrate_color_thresholding(image)

        top_cheese_mask = self.__segment_top_cheese_2e2(image)
        return top_cheese_mask
