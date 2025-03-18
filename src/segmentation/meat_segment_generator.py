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
)

######## Parameters ##########

CROP_XMIN = 300
CROP_YMIN = 50
CROP_XMAX = 430
CROP_YMAX = 280

CROP_XMIN_HSV = 320
CROP_YMIN_HSV = 100
CROP_XMAX_HSV = 390
CROP_YMAX_HSV = 250

###############################

class MeatSegmentGenerator():
    
    def __init__(self):
        """
        Initializes and loads SAM2 model for meat segmentation
        """
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
        
        # Crop parameters for meat
    
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
        
        # crop the bin
        crop_xmin = CROP_XMIN
        crop_ymin = CROP_YMIN
        crop_xmax = CROP_XMAX
        crop_ymax = CROP_YMAX
        cropped_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        
        # Construct bounding box prompt using HSV mask
        mask = segment_from_hsv(image, self.lower_hsv, self.upper_hsv)
        kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel
        mask = cv2.erode(mask, kernel, iterations=1)

        bounding_box = calc_bbox_from_mask(mask)
        meat_right_x,  meat_top_y, meat_left_x, meat_bottom_y = bounding_box

        negative_points = [
            (meat_left_x, meat_top_y),  # Top-left
            (meat_right_x, meat_top_y),  # Top-right
            (meat_left_x, meat_bottom_y),  # Bottom-left
            (meat_right_x, meat_bottom_y)   # Bottom-right
        ]
        point_labels = np.array([0, 0, 0, 0])  # All are negative prompts
        
        # Predict mask using SAM2
        self.predictor.set_image(cropped_image)
        masks, scores, logits = self.predictor.predict(
            point_coords=negative_points, 
            point_labels=point_labels,
            multimask_output=True,
            box=bounding_box[None, :]
        )
        
        # Find the mask with the highest score
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        max_index = np.argmax(scores)
        all_meat_mask = masks[max_index]
        
        # Convert mask to original dimensions
        all_meat_mask_orig = convert_mask_to_orig_dims(all_meat_mask, image, crop_xmin, crop_ymin, crop_xmax, crop_ymax)

        # Erode the mask to remove noise
        kernel = np.ones((5, 5), np.uint8)
        all_meat_mask_orig = cv2.erode(all_meat_mask_orig, kernel, iterations=1)
        
        # Meat dims in pixels
        meat_w = 100
        meat_h = 100
        
        y_indices, x_indices = np.where(all_meat_mask_orig * 255 == 255)
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            print("No white object found!")
        
        # Design prompt box around the meat
        # Find bounding box of the vertical rectangle
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        
        meat_bottom_centre_x = (xmin + xmax) // 2
        meat_bottom_y = ymax

        meat_top_y = ymax - meat_h
        meat_right_x = meat_bottom_centre_x + (meat_w // 2)
        meat_left_x = meat_bottom_centre_x - (meat_w // 2)
        
        # positive prompt box for SAM
        meat_box = np.array(
            [meat_left_x, meat_top_y, meat_right_x, meat_bottom_y]
        )
        
        # Run SAM2 
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            box=meat_box[None, :], multimask_output=True)
        
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        
        top_meat_mask = masks[np.argmax(scores)]
        kernel = np.ones((7, 7), np.uint8)
        top_meat_mask = cv2.erode(top_meat_mask, kernel, iterations=1)
        
        return top_meat_mask
    
    def __calibrate_color_thresholding(self, image):
        """
        Calculates hsv range for segmenting meat by sampling the image
        """
        # Get HSV range for the first image
        cropped_image = image[
            self.crop_ymin_hsv:self.crop_ymax_hsv,
            self.crop_xmin_hsv:self.crop_xmax_hsv
        ]
        self.lower_hsv, self.upper_hsv = get_hsv_range(cropped_image)
    
    def get_top_meat_slice(self, image):
        """
        Input: image (numpy array): The input image to segment
        Output: mask (numpy array): The segmented mask of the top slice of meat
        """
        
        # If first image, get HSV range
        if not self.got_first_img:
            self.got_first_img = True
            self.__calibrate_color_thresholding(image)
        
        # Segment top meat slice using SAM2
        top_meat_mask = self.__segment_top_meat_e2e(image)
        
        return top_meat_mask        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        

        