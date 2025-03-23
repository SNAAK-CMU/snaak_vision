"""
Script to define class for segmenting the first bread slice in the assembly area
"""

import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
from pycocotools.mask import decode

from segmentation.segment_utils import convert_mask_to_orig_dims

############# Parameters ################

BREAD_AREA_GT = 11500
TRAY_BOX_PIX = (295, 103, 530, 280)

#########################################


class BreadSegmentGenerator:

    AREA_GT = BREAD_AREA_GT
    TRAY_BOX_PIX = TRAY_BOX_PIX

    def __init__(self):
        self.sam2_checkpoint = "/home/snaak/Documents/manipulation_ws/src/sam2/checkpoints/sam2.1_hiera_small.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        # self.sam2_checkpoint = "/home/user/sam2/checkpoints/sam2.1_hiera_small.pt"
        # self.model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

        self.__create_sam_predictor()

    def __create_sam_predictor(self):
        """
        Initializes and loads SAM2 model
        """
        self.mode = "coco_rle"
        self.sam2 = build_sam2(
            self.model_cfg,
            self.sam2_checkpoint,
            device="cuda",
            apply_postprocessing=False,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2,
            points_per_side=32,
            points_per_batch=128,
            pred_iou_thresh=0.5,  # TODO: Change paths according deployment environment
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_nms_thresh=0.7,
            crop_n_layers=4,
            box_nms_thresh=0.4,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=5.0,
            crop_overlap_ratio=1000 / 1500,
            # use_m2m=True,
            output_mode=self.mode,
        )

    def select_bread_mask(self, masks):
        """
        Selects mask corresponding to bread slice based on area heuristics
        """
        area_diff_min = 1000000
        selected_mask = None

        for i in range(len(masks)):
            rle_annotation = masks[i]["segmentation"]
            area = masks[i]["area"]

            area_diff = abs(area - self.AREA_GT)
            if area_diff < area_diff_min:
                selected_mask = decode(rle_annotation)
                area_diff_min = area_diff

        return selected_mask

    def get_bread_mask(self, image):
        """
        Called from the vision node during ingredient placement.
        Returns the segmentation mask for the base bread slice in assembly area.
        """
        # Crop out tray region
        crop_xmin, crop_ymin, crop_xmax, crop_ymax = self.TRAY_BOX_PIX
        cropped_image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        masks = self.mask_generator.generate(cropped_image)
        cropped_mask = self.select_bread_mask(masks)
        orig_mask = convert_mask_to_orig_dims(
            cropped_mask, image, crop_xmin, crop_ymin, crop_xmax, crop_ymax
        )
        return orig_mask
