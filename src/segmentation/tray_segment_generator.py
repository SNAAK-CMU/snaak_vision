"""
Defines class to generate segment of the tray in assembly area
"""

import numpy as np
import cv2

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

TRAY_AREA_GT = 120000


class TraySegmentGenerator:

    AREA_GT = TRAY_AREA_GT
    VARIANCE = 50
    IMG_H_OFFSET = 50

    def __init__(self):

        # TODO: Change paths according deployment environment
        # self.sam2_checkpoint = (
        #     "/home/snaak/Documents/vision_ws/checkpoints/sam2.1_hiera_small.pt"
        # )
        # self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        self.sam2_checkpoint = (
            "/home/user/sam2/checkpoints/sam2.1_hiera_small.pt"
        )
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        self.__create_sam_predictor()

    def __create_sam_predictor(self):
        """
        Initializes and loads SAM2 model
        """
        sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(sam2_model)

    def __update_prompts(self, image_shape):
        """
        Updates prompts for SAM based on image size
        """
        img_h, img_w, c = image_shape
        img_h -= self.IMG_H_OFFSET

        p1 = [img_w // 2 - self.VARIANCE, img_h // 2 - self.VARIANCE]
        p2 = [img_w // 2 + self.VARIANCE, img_h // 2 - self.VARIANCE]
        p3 = [img_w // 2 - self.VARIANCE, img_h // 2 + self.VARIANCE]
        p4 = [img_w // 2 + self.VARIANCE, img_h // 2 + self.VARIANCE]

        self.input_points = np.array([p1, p2, p3, p4])
        self.input_labels = np.array([1, 1, 1, 1])

    def __select_tray_mask(self, masks):
        """
        Selects a single mask based on area based heuristics
        """
        min_area_diff = 100000000
        selected_mask = None
        for mask in masks:
            area_mask = np.sum(mask)
            area_diff = abs(area_mask - self.AREA_GT)
            if area_diff < min_area_diff:
                selected_mask = mask
                min_area_diff = area_diff
        return selected_mask

    def __process_mask(self, mask):
        """
        Removes noise from the tray mask
        """
        mask_new = mask.copy()
        kernel = np.ones((5, 5), np.uint8)
        mask_new = cv2.erode(mask_new, kernel, iterations=1)
        mask_new = cv2.dilate(mask_new, kernel, iterations=1)
        return mask_new

    def get_tray_mask(self, image):
        """
        Called from the vision node during bread placement.
        Returns the segmentation mask for the tray in assembly area.
        """
        # Run SAM
        self.__update_prompts(image.shape)
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=self.input_points,
            point_labels=self.input_labels,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # Get final mask
        mask = self.__select_tray_mask(masks)
        mask = self.__process_mask(mask)

        return mask
