# class to check ingredient placement

import numpy as np
import cv2
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import segmentation.segment_utils as seg_utils
from segmentation.segment_utils import segment_from_hsv, convert_mask_to_orig_dims
from segmentation.UNet.ingredients_UNet import Ingredients_UNet


########## Parameters ##########

BREAD_HSV_LOWER_BOUND = (10, 50, 100)
BREAD_HSV_UPPER_BOUND = (40, 255, 255)

BREAD_HSV_LOWER_BOUND_STRICT = (10, 60, 100)
BREAD_HSV_UPPER_BOUND_STRICT = (20, 110, 220)

TRAY_HSV_LOWER_BOUND = (85, 40, 20)
TRAY_HSV_UPPER_BOUND = (130, 255, 255)

TRAY_BOX_PIX = (
    250,
    20,
    630,
    300,
)  # (x1, y1, x2, y2) coordinates of the tray box in the image

CHEESE_W = 103  # width of the cheese slice in pixels

SAM2_CHECKPOINT = (
    "/home/snaak/Documents/manipulation_ws/src/sam2/checkpoints/sam2.1_hiera_small.pt"
    # "/home/parth/snaak/projects/sam2/checkpoints/sam2.1_hiera_small.pt"
)
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

################################


class SandwichChecker:
    # this class checks everything in pixel coords
    def __init__(
        self,
        fov_width=0.775,
        fov_height=0.435,
        threshold_in_cm=3,
        image_width=848,
        image_height=480,
        tray_dims_m=[0.305, 0.220],
        bread_dims_m=[0.11, 0.08],
        cheese_dims_m=[0.090, 0.095],
        node_logger=None,
        tray_center=None,
        cheese_UNet=None,
        bologna_UNet=None,
        use_unet=False,
        ham_radius=0.05,
    ):

        self.tray_hsv_lower_bound = TRAY_HSV_LOWER_BOUND
        self.tray_hsv_upper_bound = TRAY_HSV_UPPER_BOUND
        self.bread_hsv_lower_bound = BREAD_HSV_LOWER_BOUND
        self.bread_hsv_upper_bound = BREAD_HSV_UPPER_BOUND
        self.crop_xmin, self.crop_ymin, self.crop_xmax, self.crop_ymax = TRAY_BOX_PIX

        self.fov_width = fov_width
        self.fov_height = fov_height
        self.threshold_in_cm = threshold_in_cm
        self.image_height = image_height
        self.image_width = image_width
        self.tray_center = tray_center

        self.pix_per_m = (
            (self.image_width / self.fov_width) + (self.image_height / self.fov_height)
        ) / 2
       
        self.pass_threshold = self.pix_per_m * (self.threshold_in_cm / 100)
        self.__calc_area_thresholds(tray_dims_m, bread_dims_m, cheese_dims_m)
        self.ham_radius_pix = ham_radius * self.pix_per_m

        # Initialize localization lists
        self.tray_contours = []
        self.bread_contours = []
        self.bread_centers = []
        self.cheese_contours = []
        self.cheese_centers = []
        self.ham_contours = []
        self.ham_centers = []
        self.place_images = []

        # Initialize SAM2
        self.sam2_checkpoint = SAM2_CHECKPOINT
        self.model_cfg = SAM2_MODEL_CFG
        self.__create_sam_predictor()

        # Initialize UNet
        self.cheese_UNet = cheese_UNet
        self.bologna_UNet = bologna_UNet
        self.use_unet = use_unet

        self.node_logger = node_logger
        if self.node_logger is not None:
            self.node_logger.info(
                f"Initialized SandwichChecker with parameters: fov_width={fov_width}, fov_height={fov_height}, threshold_in_cm={threshold_in_cm}, image_width={image_width}, image_height={image_height}"
            )
            self.node_logger.info(
                f"Calculated pix_per_m={self.pix_per_m}, pass_threshold={self.pass_threshold}"
            )
            self.node_logger.info(
                f"Calculated tray_area={self.tray_area}, bread_area={self.bread_area}, cheese_area={self.cheese_area}"
            )

    def __calc_area_thresholds(self, tray_dims_m, bread_dims_m, cheese_dims_m):
        """
        Calculate the area thresholds for tray, bread and cheese
        """
        self.tray_area = tray_dims_m[0] * tray_dims_m[1] * self.pix_per_m**2
        self.bread_area = bread_dims_m[0] * bread_dims_m[1] * self.pix_per_m**2
        self.cheese_area = cheese_dims_m[0] * cheese_dims_m[1] * self.pix_per_m**2
        self.min_tray_area = 0.9 * self.tray_area
        self.max_tray_area = 1.2 * self.tray_area
        self.min_bread_area = 0.9 * self.bread_area
        self.max_bread_area = 2.0 * self.bread_area
        self.min_cheese_area = 0.7 * self.cheese_area
        self.max_cheese_area = 1.4 * self.cheese_area

    def __create_sam_predictor(self):
        """
        Initializes and loads SAM2 model
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def reset(self):
        self.tray_contour = []
        self.tray_center = None

        self.cheese_contours = []
        self.cheese_centers = []

        self.bread_centers = []
        self.bread_contours = []

        self.ham_contours = []
        self.ham_centers = []

        self.place_images = []

    def get_bread_placement_mask_bottom(self, image):
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
            cropped_image, self.bread_hsv_lower_bound, self.bread_hsv_upper_bound
        )

        cv2.imwrite(
            "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/assembly_bread_hsv_mask.jpg",
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

        # Select the largest contour and black out everything else
        contours, _ = cv2.findContours(
            orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(orig_mask)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            orig_mask = cv2.bitwise_and(orig_mask, mask)
        else:
            orig_mask = np.zeros_like(orig_mask)
            self.node_logger.info("No contours found in the image.")

        # cv2.imshow("bread_hsv_mask", orig_mask)
        # cv2.imshow("Orig image", image)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return orig_mask

    def get_bread_top_placement_xy(self, image):
        second_crop = image[
            TRAY_BOX_PIX[1] : TRAY_BOX_PIX[3], TRAY_BOX_PIX[0] : TRAY_BOX_PIX[2]
        ]

        second_hsv = cv2.cvtColor(second_crop, cv2.COLOR_BGR2HSV)
        bread_hsv_lower_bound = np.array(BREAD_HSV_LOWER_BOUND_STRICT, dtype=np.uint8)
        bread_hsv_upper_bound = np.array(BREAD_HSV_UPPER_BOUND_STRICT, dtype=np.uint8)
        bread_mask = cv2.inRange(
            second_hsv, bread_hsv_lower_bound, bread_hsv_upper_bound
        )

        bread_mask = cv2.dilate(bread_mask, None, iterations=1)
        bread_mask = cv2.erode(bread_mask, None, iterations=2)

        contours, _ = cv2.findContours(
            bread_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            bread_mask = np.zeros_like(bread_mask)
            cv2.drawContours(bread_mask, [largest_contour], -1, 255, -1)

        # Apply closing to fill in small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bread_mask = cv2.morphologyEx(bread_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            bread_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        plot_image = second_crop.copy()
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)

            M = cv2.moments(hull)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # cv2.drawContours(plot_image, [hull], -1, (255, 0, 0), 3)
            # cv2.circle(plot_image, (cX, cY), 7, (0, 0, 255), -1)
            # cv2.imshow("Plot Image", plot_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Use SAM2 to localize bread
        input_points = [[cX, cY]]

        # Add two more points for prompting SAM
        input_points.append([cX, cY + 15])
        input_points.append([cX, cY - 15])

        input_points = np.array(input_points, dtype=np.float32)
        input_labels = np.array([1, 1, 1], dtype=np.int32)

        self.predictor.set_image(second_crop)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        # Select mask using area heuristic
        area_gt = 12500
        area_diff_min = 100000000
        selected_mask = masks[0]
        for i, mask in enumerate(masks):
            area_mask = np.sum(mask)
            area_diff = abs(area_mask - area_gt)
            if area_diff < area_diff_min:
                selected_mask = mask
                area_diff_min = area_diff

        # Convert mask to compatible format
        selected_mask = selected_mask * 255
        selected_mask = selected_mask.astype(np.uint8)

        # Remove noise from mask
        kernel = np.ones((5, 5), np.uint8)
        selected_mask = cv2.erode(selected_mask, kernel, iterations=1)
        selected_mask = cv2.dilate(selected_mask, kernel, iterations=1)

        # Select the largest contour and blacken out the rest
        contours, _ = cv2.findContours(
            selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            selected_mask = np.zeros_like(selected_mask)
            cv2.drawContours(selected_mask, [largest_contour], -1, 255, -1)

        # Convert mask to original image dimensions
        orig_mask = convert_mask_to_orig_dims(
            selected_mask,
            image,
            TRAY_BOX_PIX[0],
            TRAY_BOX_PIX[1],
            TRAY_BOX_PIX[2],
            TRAY_BOX_PIX[3],
        )

        # Find countours in the original mask
        contours, _ = cv2.findContours(
            orig_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        plot_image = image.copy()
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(plot_image, [largest_contour], -1, (0, 255, 0), 3)

            # Find the center of the convex hull
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

        cv2.circle(plot_image, (cX, cY), 7, (0, 0, 255), -1)

        return (cX, cY), plot_image

    def check_bread_bottom(self, image):
        """
        Extract tray contours, bread contours and their centers from the image.
        Check if the bread is placed in the tray.
        Args:
            image (opencv): image from assembly area after placing bread
        Returns:
            bool: True if bread is placed in tray, False otherwise
            image (opencv) : image with tray and bread contours drawn on it
        """

        # TODO: handle case when bread or tray are not detected in image

        # contours, heirarchy = seg_utils.contour_segmentation(
        #     image,
        #     show_image=True,
        #     show_separate_contours=True,
        #     show_steps=True,
        #     segment_type="edges",
        #     edges_thresholds=(15, 30),
        # )

        # # Extract tray ROIs
        # tray_contour_indices = [
        #     i
        #     for i, contour in enumerate(contours)
        #     if self.min_tray_area < cv2.contourArea(contour) < self.max_tray_area
        # ]
        # self.tray_contours = [contours[i] for i in tray_contour_indices]
        # self.node_logger.info(f"Number of tray contours: {len(self.tray_contours)}")

        # # Extract bread ROIs
        # bread_contour_indices = [
        #     i
        #     for i, contour in enumerate(contours)
        #     if 0.7*self.bread_area < cv2.contourArea(contour) < 2.0*self.bread_area
        # ]
        # # get mask of bread ROIs
        # bread_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # for i in bread_contour_indices:
        #     cv2.fillPoly(bread_mask, [contours[i]], 255)

        # # ROIs image
        # masked_image = cv2.bitwise_and(image, image, mask=bread_mask)

        # # get contours of ROIs image
        # contours, heirarchy = seg_utils.contour_segmentation(
        #     masked_image,
        #     show_image=True,
        #     show_separate_contours=True,
        #     show_steps=True,
        #     segment_type="binary",
        # )

        # # Extract bread contours
        # bread_contour_indices = [
        #     i
        #     for i, contour in enumerate(contours)
        #     if 0.7*self.bread_area < cv2.contourArea(contour) < 1.6*self.bread_area
        # ]

        # self.bread_contours = [contours[i] for i in bread_contour_indices]
        # self.node_logger.info(f"Number of bread contours: {len(self.tray_contours)}")

        # for bread_contour in self.bread_contours:
        #     # Calculate the center of the bread contour
        #     M = cv2.moments(bread_contour)
        #     if M["m00"] != 0:
        #         cx = int(M["m10"] / M["m00"])
        #         cy = int(M["m01"] / M["m00"])
        #     else:
        #         cx, cy = 0, 0

        #     self.bread_centers.append((cx, cy))

        bread_mask = self.get_bread_placement_mask_bottom(image)
        y_coords, x_coords = np.where(bread_mask == 255)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        bread_center = (center_x, center_y)
        self.bread_centers.append(bread_center)

        # Check if bread is placed in tray
        bread_on_tray = True

        # # measure distance between bread center and tray center
        distance = -1
        if self.tray_center is not None:
            distance = (
                (self.tray_center[0] - bread_center[0]) ** 2
                + (self.tray_center[1] - bread_center[1]) ** 2
            ) ** 0.5
            self.node_logger.info(
                f"Distance between bread center {bread_center} and tray center {self.tray_center}: {distance}"
            )
            if distance > self.pass_threshold:
                bread_on_tray = False

        plot_image = image.copy()

        # draw centers on image
        for center in self.bread_centers:
            cv2.circle(plot_image, center, 5, (0, 0, 255), -1)
        if self.tray_center is not None:
            cv2.circle(
                plot_image,
                (int(self.tray_center[0]), int(self.tray_center[1])),
                5,
                (0, 255, 0),
                -1,
            )

        # Write distance and pass threshold on the image
        cv2.putText(
            plot_image,
            f"Distance: {distance:.2f} px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            plot_image,
            f"Pass threshold: {self.pass_threshold:.2f} px",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return bread_on_tray, plot_image

    def check_bread_top(self, image):
        bread_center, plot_image = self.get_bread_top_placement_xy(image)
        bottom_bread_center = self.bread_centers[-1]
        self.bread_centers.append(bread_center)

        # Check if bread is placed in tray
        bread_on_tray = True

        # measure distance between bread center and tray center
        distance = -1
        if bottom_bread_center is not None:
            distance = (
                (bottom_bread_center[0] - bread_center[0]) ** 2
                + (bottom_bread_center[1] - bread_center[1]) ** 2
            ) ** 0.5
            self.node_logger.info(
                f"Distance between top bread center {bread_center} and bottom bread center {bottom_bread_center}: {distance}"
            )
            if distance > self.pass_threshold:
                bread_on_tray = False

        if bottom_bread_center is not None:
            cv2.circle(
                image,
                (int(bottom_bread_center[0]), int(bottom_bread_center[1])),
                5,
                (0, 255, 0),
                -1,
            )

        # Write distance and pass threshold on the image
        cv2.putText(
            plot_image,
            f"Distance: {distance:.2f} px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            plot_image,
            f"Pass threshold: {self.pass_threshold:.2f} px",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return bread_on_tray, plot_image

    def set_tray_center(self, tray_center):
        """
        Set the tray center coordinates.
        Args:
            tray_center (tuple): (x, y) coordinates of the tray center in pixels
        """
        if tray_center is None:
            raise ValueError(
                "Tray center cannot be set to None. Please set it before checking ingredients."
            )
        if self.tray_center is None:
            self.tray_center = tray_center
        self.node_logger.info(f"Tray center set to: {self.tray_center}")

    def check_cheese_single(self, image):
        """
        Extract cheese contours and their centers from the image.
        Threshold distace between cheese and bread centers.

        Args:
            image (opencv): image from assembly area after placing cheese
        Returns:
            bool: True if cheese center is within threshold distance from bread center, False otherwise
            image (opencv) : image with cheese contours drawn on it
        """

        # TODO: handle case when no cheese is detected in image

        if self.use_unet:
            pass
            cheese_center = (0,0)
        else:

            # get images
            total_images = len(self.place_images)
            first_image = self.place_images[total_images - 2]
            second_image = self.place_images[total_images - 1]

            # Crop the tray box from the bread image
            first_crop = first_image[
                TRAY_BOX_PIX[1] : TRAY_BOX_PIX[3], TRAY_BOX_PIX[0] : TRAY_BOX_PIX[2]
            ]

            # Crop the tray box from the first ham image
            second_crop = second_image[
                TRAY_BOX_PIX[1] : TRAY_BOX_PIX[3], TRAY_BOX_PIX[0] : TRAY_BOX_PIX[2]
            ]

            # Segment tray from the second image using tray HSV values
            # Convert the image to HSV color space
            second_hsv = cv2.cvtColor(second_crop, cv2.COLOR_BGR2HSV)
            tray_hsv_lower_bound = np.array(TRAY_HSV_LOWER_BOUND, dtype=np.uint8)
            tray_hsv_upper_bound = np.array(TRAY_HSV_UPPER_BOUND, dtype=np.uint8)
            tray_mask = cv2.inRange(second_hsv, tray_hsv_lower_bound, tray_hsv_upper_bound)

            # Invert the tray mask
            tray_mask_inv = cv2.bitwise_not(tray_mask)

            # Find the bounding box of the tray
            contours, _ = cv2.findContours(
                tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                tray_x, tray_y, tray_w, tray_h = cv2.boundingRect(largest_contour)

            # Black out everything outside the tray box coordinate in tray_mask_inv
            tray_mask_inv[0 : tray_y + 5, :] = 0
            tray_mask_inv[tray_y + tray_h - 5 :, :] = 0
            tray_mask_inv[:, 0 : tray_x + 5] = 0
            tray_mask_inv[:, tray_x + tray_w - 5 :] = 0

            # cv2.imshow("tray_mask_inv", tray_mask_inv)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # Segment bread using strict hsv bounds (to differentiate between bread and cheese)
            second_hsv = cv2.cvtColor(second_crop, cv2.COLOR_BGR2HSV)
            bread_hsv_lower_bound = np.array(BREAD_HSV_LOWER_BOUND_STRICT, dtype=np.uint8)
            bread_hsv_upper_bound = np.array(BREAD_HSV_UPPER_BOUND_STRICT, dtype=np.uint8)
            bread_mask = cv2.inRange(
                second_hsv, bread_hsv_lower_bound, bread_hsv_upper_bound
            )
            bread_mask_inv = cv2.bitwise_not(bread_mask)

            # Calculate the difference between the two images
            diff = cv2.absdiff(first_crop, second_crop)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # And operate the difference image with bread mask and tray mask
            gray_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=tray_mask_inv)
            gray_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=bread_mask_inv)
            gray_diff = cv2.GaussianBlur(gray_diff, (7, 7), 0)

            # Sliding window to find cheese bounding box
            cheese_box = [
                0,
                0,
                CHEESE_W,
                CHEESE_W,
            ]  # (x1, y1, x2, y2) coordinates of the cheese box in the image

            max_sum = 0
            best_cheese_box = None

            # Slide the cheese box over the image
            for col in range(0, gray_diff.shape[1] - CHEESE_W):  # iterate along image width
                cheese_box[0] = col
                cheese_box[2] = col + CHEESE_W

                for row in range(
                    0, gray_diff.shape[0] - CHEESE_W
                ):  # iterate along image height
                    cheese_box[1] = row
                    cheese_box[3] = row + CHEESE_W

                    cheese_crop = gray_diff[
                        cheese_box[1] : cheese_box[3], cheese_box[0] : cheese_box[2]
                    ]
                    cheese_crop_sum = np.sum(cheese_crop)
                    if cheese_crop_sum > max_sum:
                        max_sum = cheese_crop_sum
                        best_cheese_box = cheese_box.copy()

            cheese_center = (
                int((best_cheese_box[0] + best_cheese_box[2]) / 2) + TRAY_BOX_PIX[0],
                int((best_cheese_box[1] + best_cheese_box[3]) / 2) + TRAY_BOX_PIX[1],
            )

        # Check if cheese is placed within threshold distance from bread
        valid_cheese = False
        for bread_center in self.bread_centers:
            distance = (
                (bread_center[0] - cheese_center[0]) ** 2
                + (bread_center[1] - cheese_center[1]) ** 2
            ) ** 0.5
            self.node_logger.info(
                f"Distance between cheese center {cheese_center} and bread center {bread_center}: {distance}"
            )

            if distance < self.pass_threshold:
                valid_cheese = True
                break

        # Convert cheese box to orig image coordinates
        best_cheese_box[0] += TRAY_BOX_PIX[0]
        best_cheese_box[1] += TRAY_BOX_PIX[1]
        best_cheese_box[2] += TRAY_BOX_PIX[0]
        best_cheese_box[3] += TRAY_BOX_PIX[1]

        # Visualize cheese localization
        plot_image = image.copy()
        cv2.circle(plot_image, cheese_center, 5, (255, 0, 255), -1)
        cv2.rectangle(
            plot_image,
            (best_cheese_box[0], best_cheese_box[1]),
            (best_cheese_box[2], best_cheese_box[3]),
            (255, 0, 255),
            2,
        )

        # Visualize bread localization
        for contour in self.bread_contours:
            cv2.drawContours(plot_image, contour, -1, (255, 0, 0), 3)
        for center in self.bread_centers:
            cv2.circle(plot_image, center, 5, (0, 0, 255), -1)

        # Write distance and pass threshold on the image
        cv2.putText(
            plot_image,
            f"Distance: {distance:.2f} px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            plot_image,
            f"Pass threshold: {self.pass_threshold:.2f} px",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return (
            valid_cheese,
            plot_image,
        )  # if no cheese is placed in tray or cheese is not close to bread

    def check_cheese_multi(self, image, ingredient_count):
        # get images
        total_images = len(self.place_images)
        first_image = self.place_images[total_images - 2]
        second_image = self.place_images[total_images - 1]

        # Crop the tray box from the bread image
        first_crop = first_image[
            TRAY_BOX_PIX[1] : TRAY_BOX_PIX[3], TRAY_BOX_PIX[0] : TRAY_BOX_PIX[2]
        ]

        # Crop the tray box from the first ham image
        second_crop = second_image[
            TRAY_BOX_PIX[1] : TRAY_BOX_PIX[3], TRAY_BOX_PIX[0] : TRAY_BOX_PIX[2]
        ]

        # Segment tray from the second image using tray HSV values
        # Convert the image to HSV color space
        second_hsv = cv2.cvtColor(second_crop, cv2.COLOR_BGR2HSV)
        tray_hsv_lower_bound = np.array(TRAY_HSV_LOWER_BOUND, dtype=np.uint8)
        tray_hsv_upper_bound = np.array(TRAY_HSV_UPPER_BOUND, dtype=np.uint8)
        tray_mask = cv2.inRange(second_hsv, tray_hsv_lower_bound, tray_hsv_upper_bound)

        # Invert the tray mask
        tray_mask_inv = cv2.bitwise_not(tray_mask)

        # Find the bounding box of the tray
        contours, _ = cv2.findContours(
            tray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            tray_x, tray_y, tray_w, tray_h = cv2.boundingRect(largest_contour)

        # Black out everything outside the tray box coordinate in tray_mask_inv
        tray_mask_inv[0 : tray_y + 5, :] = 0
        tray_mask_inv[tray_y + tray_h - 5 :, :] = 0
        tray_mask_inv[:, 0 : tray_x + 5] = 0
        tray_mask_inv[:, tray_x + tray_w - 5 :] = 0

        # cv2.imshow("tray_mask_inv", tray_mask_inv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Segment bread using strict hsv bounds (to differentiate between bread and cheese)
        second_hsv = cv2.cvtColor(second_crop, cv2.COLOR_BGR2HSV)
        bread_hsv_lower_bound = np.array(BREAD_HSV_LOWER_BOUND_STRICT, dtype=np.uint8)
        bread_hsv_upper_bound = np.array(BREAD_HSV_UPPER_BOUND_STRICT, dtype=np.uint8)
        bread_mask = cv2.inRange(
            second_hsv, bread_hsv_lower_bound, bread_hsv_upper_bound
        )
        bread_mask_inv = cv2.bitwise_not(bread_mask)

        # Calculate the difference between the two images
        diff = cv2.absdiff(first_crop, second_crop)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # And operate the difference image with bread mask and tray mask
        gray_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=tray_mask_inv)
        gray_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=bread_mask_inv)
        gray_diff = cv2.GaussianBlur(gray_diff, (9, 9), 0)

        # Initialize window as per cheese count
        search_cheese_w = int(CHEESE_W + 0.4 * CHEESE_W * (ingredient_count - 1))
        cheese_box = [
            0,
            0,
            search_cheese_w,
            search_cheese_w,
        ]  # (x1, y1, x2, y2) coordinates of the cheese box in the image

        # Slide the cheese box over the image
        max_sum = 0
        best_cheese_box = None
        for col in range(
            0, gray_diff.shape[1] - search_cheese_w
        ):  # iterate along image width
            cheese_box[0] = col
            cheese_box[2] = col + search_cheese_w

            for row in range(
                0, gray_diff.shape[0] - search_cheese_w
            ):  # iterate along image height
                cheese_box[1] = row
                cheese_box[3] = row + search_cheese_w

                cheese_crop = gray_diff[
                    cheese_box[1] : cheese_box[3], cheese_box[0] : cheese_box[2]
                ]
                cheese_crop_sum = np.sum(cheese_crop)
                if cheese_crop_sum > max_sum:
                    max_sum = cheese_crop_sum
                    best_cheese_box = cheese_box.copy()

        # Blacken out everything outside the cheese box
        gray_diff_new = np.zeros_like(gray_diff)
        gray_diff_new[
            best_cheese_box[1] : best_cheese_box[3],
            best_cheese_box[0] : best_cheese_box[2],
        ] = gray_diff[
            best_cheese_box[1] : best_cheese_box[3],
            best_cheese_box[0] : best_cheese_box[2],
        ]
        gray_diff = gray_diff_new

        # Apply edge detection to the difference image
        edges = cv2.Canny(gray_diff, 20, 30)

        # Find a box that can encompass all the white pixels in the edge image using np.where
        multi_cheese_box = None
        y_indices, x_indices = np.where(edges > 0)
        if len(x_indices) > 0 and len(y_indices) > 0:
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            multi_cheese_box = (x1, y1, x2, y2)

        # Convert box to orig image coordinates
        multi_cheese_box = (
            multi_cheese_box[0] + TRAY_BOX_PIX[0],
            multi_cheese_box[1] + TRAY_BOX_PIX[1],
            multi_cheese_box[2] + TRAY_BOX_PIX[0],
            multi_cheese_box[3] + TRAY_BOX_PIX[1],
        )
        multi_cheese_center = (
            int((multi_cheese_box[0] + multi_cheese_box[2]) / 2),
            int((multi_cheese_box[1] + multi_cheese_box[3]) / 2),
        )

        # Check if cheese is placed within threshold distance from bread
        valid_cheese = False
        distance = -1
        for bread_center in self.bread_centers:
            distance = (
                (bread_center[0] - multi_cheese_center[0]) ** 2
                + (bread_center[1] - multi_cheese_center[1]) ** 2
            ) ** 0.5
            self.node_logger.info(
                f"Distance between cheese center {multi_cheese_center} and bread center {bread_center}: {distance}"
            )

            if distance < self.pass_threshold:
                valid_cheese = True
                break

        # Visualize cheese localization
        plot_image = image.copy()
        cv2.circle(plot_image, multi_cheese_center, 5, (255, 0, 255), -1)
        cv2.rectangle(
            plot_image,
            (multi_cheese_box[0], multi_cheese_box[1]),
            (multi_cheese_box[2], multi_cheese_box[3]),
            (255, 0, 255),
            2,
        )

        # Visualize bread localization
        for contour in self.bread_contours:
            cv2.drawContours(plot_image, contour, -1, (255, 0, 0), 3)
        for center in self.bread_centers:
            cv2.circle(plot_image, center, 5, (0, 0, 255), -1)

        # Write distance and pass threshold on the image
        cv2.putText(
            plot_image,
            f"Distance: {distance:.2f} px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            plot_image,
            f"Pass threshold: {self.pass_threshold:.2f} px",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return (
            valid_cheese,
            plot_image,
        )

    def check_cheese(self, image, ingredient_count):
        """
        Check if cheese is placed in tray and close to bread.
        Args:
            image (opencv): image from assembly area after placing cheese
            ingredient_count (int): number of cheese slices placed in tray
        Returns:
            bool: True if cheese is placed in tray and close to bread, False otherwise
            image (opencv) : image with cheese contours drawn on it
        """

        valid_cheese = None
        plot_image = None
        if ingredient_count == 1:
            valid_cheese, plot_image = self.check_cheese_single(image)
        else:
            valid_cheese, plot_image = self.check_cheese_multi(image, ingredient_count)

        return valid_cheese, plot_image

    def check_ham(self, image):
        # Extract previous and current place images
        total_images = len(self.place_images)
        first_image = self.place_images[total_images - 2]
        second_image = self.place_images[total_images - 1]

        # Crop the tray box from the bread image
        first_image = first_image[
            TRAY_BOX_PIX[1] : TRAY_BOX_PIX[3], TRAY_BOX_PIX[0] : TRAY_BOX_PIX[2]
        ]

        # Crop the tray box from the first ham image
        second_image = second_image[
            TRAY_BOX_PIX[1] : TRAY_BOX_PIX[3], TRAY_BOX_PIX[0] : TRAY_BOX_PIX[2]
        ]

        # Segment bread from the second image using bread HSV values
        second_hsv = cv2.cvtColor(second_image, cv2.COLOR_BGR2HSV)
        bread_hsv_lower_bound = np.array(BREAD_HSV_LOWER_BOUND, dtype=np.uint8)
        bread_hsv_upper_bound = np.array(BREAD_HSV_UPPER_BOUND, dtype=np.uint8)
        bread_mask = cv2.inRange(
            second_hsv, bread_hsv_lower_bound, bread_hsv_upper_bound
        )

        # Segment tray from the first image using tray HSV values
        tray_hsv_lower_bound = np.array(TRAY_HSV_LOWER_BOUND, dtype=np.uint8)
        tray_hsv_upper_bound = np.array(TRAY_HSV_UPPER_BOUND, dtype=np.uint8)
        tray_mask = cv2.inRange(second_hsv, tray_hsv_lower_bound, tray_hsv_upper_bound)

        # Invert bread and tray masks for noise removal
        bread_mask_inv = cv2.bitwise_not(bread_mask)
        tray_mask_inv = cv2.bitwise_not(tray_mask)

        # Calculate difference image
        diff = cv2.absdiff(first_image, second_image)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Denoise difference mask
        gray_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=bread_mask_inv)
        gray_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=tray_mask_inv)
        gray_diff = cv2.GaussianBlur(gray_diff, (7, 7), 0)

        cv2.imwrite("diff.jpg", gray_diff)

        # Fit circle to the top bologna slice
        circles = cv2.HoughCircles(
            gray_diff,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=15,
            minRadius=45, #self.ham_radius * 0.90 
            maxRadius=60, #self.ham_radius * 1.10
        )
        circles = np.uint16(np.around(circles))
        circle_scores = []
        for i in circles[0, :]:
            circle_filter = np.zeros_like(gray_diff)
            cv2.circle(circle_filter, (i[0], i[1]), i[2], (255, 255, 255), -1)
            masked_img = cv2.bitwise_and(gray_diff, circle_filter)
            score = np.sum(masked_img)
            circle_scores.append(score)
        max_index = np.argmax(circle_scores)
        best_circle = circles[0, max_index]
        best_circle_x, best_circle_y, best_circle_radius = (
            int(best_circle[0]),
            int(best_circle[1]),
            int(best_circle[2]),
        )

        # Transform coordinates of circle to original image
        best_circle_x += TRAY_BOX_PIX[0]
        best_circle_y += TRAY_BOX_PIX[1]

        # Check if bologna is placed accurately
        bread_center_x, bread_center_y = self.bread_centers[0]
        distance = (
            (best_circle_x - bread_center_x) ** 2
            + (best_circle_y - bread_center_y) ** 2
        ) ** 0.5
        is_ham_correct = distance < self.pass_threshold

        # Plot results
        best_circle_img = self.place_images[-1].copy()
        cv2.circle(
            best_circle_img,
            (best_circle_x, best_circle_y),
            best_circle_radius,
            (0, 255, 0),
            2,
        )

        cv2.circle(best_circle_img, (best_circle_x, best_circle_y), 5, (0, 255, 0), 3)
        for center in self.bread_centers:
            cv2.circle(best_circle_img, center, 5, (0, 0, 255), -1)

        # Write distance and pass threshold on the image
        cv2.putText(
            best_circle_img,
            f"Distance: {distance:.2f} px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            best_circle_img,
            f"Pass threshold: {self.pass_threshold:.2f} px",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return is_ham_correct, best_circle_img

    def check_ingredient(self, image, ingredient_name, ingredient_count=1):
        self.place_images.append(image)

        if ingredient_name == "bread_bottom":
            return self.check_bread_bottom(image)
        elif ingredient_name == "cheese":
            return self.check_cheese(image, ingredient_count)
        elif ingredient_name == "ham":
            return self.check_ham(image)
        elif ingredient_name == "bread_top":
            return self.check_bread_top(image)
        else:
            raise ValueError(f"Unknown ingredient: {ingredient_name}")


if __name__ == "__main__":
    # Testing
    fov_width = 0.775
    fov_height = 0.435
    threshold_in_cm = 3
    image_width = 848
    image_height = 480

    # Initialize the SandwichChecker with the specified parameters
    sandwich_checker = SandwichChecker(
        fov_width=fov_width,
        fov_height=fov_height,
        threshold_in_cm=threshold_in_cm,
        image_width=image_width,
        image_height=image_height,
    )

    # Place and check bread
    bread_place_image = cv2.imread(
        "/home/parth/snaak/data/SCH_images_041125/cheese_check_1/image_20250411-150101.png"
    )
    bread_place_image = cv2.resize(bread_place_image, (848, 480))
    bread_check, bread_check_image = sandwich_checker.check_ingredient(
        bread_place_image, "bread_bottom"
    )
    print(f"Is bread placed correctly? {bread_check}")
    cv2.imshow("Bread Check", bread_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Place and check cheese
    # cheese_place_image = cv2.imread(
    #     "/home/parth/snaak/data/SCH_images_041125/cheese_check_1/image_20250411-150115.png"
    # )
    # cheese_place_image = cv2.resize(cheese_place_image, (848, 480))
    # cheese_check, cheese_check_image = sandwich_checker.check_ingredient(
    #     cheese_place_image, "cheese"
    # )
    # print(f"Is cheese placed correctly? {cheese_check}")
    # cv2.imshow("Cheese Check", cheese_check_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Place and check cheese
    cheese_place_image = cv2.imread(
        "/home/parth/snaak/data/SCH_images_041125/cheese_check_1/image_20250411-150206.png"
    )
    cheese_place_image = cv2.resize(cheese_place_image, (848, 480))
    cheese_check, cheese_check_image = sandwich_checker.check_ingredient(
        cheese_place_image, "cheese", ingredient_count=2
    )
    print(f"Is cheese placed correctly? {cheese_check}")
    cv2.imshow("Cheese Check", cheese_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Place and check ham
    ham_place_image = cv2.imread(
        "/home/parth/snaak/data/SCH_images_041125/cheese_check_1/image_20250411-150241.png"
    )
    ham_place_image = cv2.resize(ham_place_image, (848, 480))
    ham_check, ham_check_image = sandwich_checker.check_ingredient(
        ham_place_image, "ham"
    )
    print(f"Is ham placed correctly? {ham_check}")
    cv2.imshow("Ham Check", ham_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Place and check cheese
    cheese_place_image = cv2.imread(
        "/home/parth/snaak/data/SCH_images_041125/cheese_check_1/image_20250411-150254.png"
    )
    cheese_place_image = cv2.resize(cheese_place_image, (848, 480))
    cheese_check, cheese_check_image = sandwich_checker.check_ingredient(
        cheese_place_image, "cheese"
    )
    print(f"Is cheese placed correctly? {cheese_check}")
    cv2.imshow("cheese Check", cheese_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Place and check bread
    bread_place_image = cv2.imread(
        "/home/parth/snaak/data/SCH_images_041125/cheese_check_1/image_20250411-150310.png"
    )
    bread_place_image = cv2.resize(bread_place_image, (848, 480))
    bread_check, bread_check_image = sandwich_checker.check_ingredient(
        bread_place_image, "bread_top"
    )
    print(f"Is top bread placed correctly? {bread_check}")
    cv2.imshow("Bread Check", bread_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
