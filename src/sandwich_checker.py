# class to check ingredient placement

import numpy as np
import cv2
import segmentation.segment_utils as seg_utils
from segmentation.segment_utils import segment_from_hsv, convert_mask_to_orig_dims


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

CHEESE_W = 97  # width of the cheese slice in pixels

################################


class SandwichChecker:
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

        # Calculate the area of the tray and bread in pixels
        self.tray_area = tray_dims_m[0] * tray_dims_m[1] * self.pix_per_m**2
        self.bread_area = bread_dims_m[0] * bread_dims_m[1] * self.pix_per_m**2
        self.cheese_area = cheese_dims_m[0] * cheese_dims_m[1] * self.pix_per_m**2
        self.min_tray_area = 0.9 * self.tray_area
        self.max_tray_area = 1.2 * self.tray_area
        self.min_bread_area = 0.9 * self.bread_area
        self.max_bread_area = 2.0 * self.bread_area
        self.min_cheese_area = 0.7 * self.cheese_area
        self.max_cheese_area = 1.4 * self.cheese_area

        self.tray_contours = []

        self.bread_contours = []
        self.bread_centers = []

        self.cheese_contours = []
        self.cheese_centers = []

        self.ham_contours = []
        self.ham_centers = []

        self.place_images = []

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

        # cv2.imshow("bread_hsv_mask", orig_mask)
        # cv2.imshow("Orig image", image)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return orig_mask

    def check_bread(self, image):
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

        bread_mask = self.get_bread_placement_mask(image)
        y_coords, x_coords = np.where(bread_mask == 255)
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        bread_center = (center_x, center_y)
        self.bread_centers.append(bread_center)

        # Check if bread is placed in tray
        bread_on_tray = True

        # # measure distance between bread center and tray center
        if self.tray_center is not None:
            for bread_center in self.bread_centers:
                distance = (
                    (self.tray_center[0] - bread_center[0]) ** 2
                    + (self.tray_center[1] - bread_center[1]) ** 2
                ) ** 0.5
                self.node_logger.info(
                    f"Distance between bread center {bread_center} and tray center {self.tray_center}: {distance}"
                )
                if distance > self.pass_threshold:
                    bread_on_tray = False
                    break

        # draw contours on image
        # for contour in self.tray_contours:
        #     cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
        # for contour in self.bread_contours:
        #     cv2.drawContours(image, contour, -1, (255, 0, 0), 3)
        # draw centers on image
        for center in self.bread_centers:
            cv2.circle(image, center, 5, (0, 0, 255), -1)
        if self.tray_center is not None:
            cv2.circle(
                image,
                (int(self.tray_center[0]), int(self.tray_center[1])),
                5,
                (0, 255, 0),
                -1,
            )

        # TODO: save detection image to self.place_images

        return bread_on_tray, image

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

    def check_cheese(self, image):
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
        # second_hsv = cv2.cvtColor(second_image, cv2.COLOR_BGR2HSV)
        # tray_hsv_lower_bound = np.array(TRAY_HSV_LOWER_BOUND, dtype=np.uint8)
        # tray_hsv_upper_bound = np.array(TRAY_HSV_UPPER_BOUND, dtype=np.uint8)
        # tray_mask = cv2.inRange(second_hsv, tray_hsv_lower_bound, tray_hsv_upper_bound)
        # tray_mask_inv = cv2.bitwise_not(tray_mask)

        # # Calculate difference image
        # diff = cv2.absdiff(first_image, second_image)
        # gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # # Denoise difference mask
        # gray_diff = cv2.bitwise_and(gray_diff, gray_diff, mask=tray_mask_inv)
        # gray_diff = cv2.GaussianBlur(gray_diff, (7, 7), 0)

        # cv2.imshow("diff mask", diff)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # contours, hierarchy = seg_utils.contour_segmentation(
        #     gray_diff,
        #     show_image=True,
        #     show_steps=True,
        #     segment_type="edges",
        #     show_separate_contours=True,
        #     edges_thresholds=(10, 20),
        # )

        # # Extract cheese contours
        # cheese_contour_indices = [
        #     i
        #     for i, contour in enumerate(contours)
        #     if 0.7*self.cheese_area < cv2.contourArea(contour) < 1.2 * self.cheese_area
        # ]
        # self.cheese_contours = [contours[i] for i in cheese_contour_indices]
        # self.node_logger.info(
        #     f"Number of cheese contours: {len(self.cheese_contours)}"
        # )

        # for cheese_contour in self.cheese_contours:
        #     # Calculate the center of the cheese contour
        #     M = cv2.moments(cheese_contour)
        #     if M["m00"] != 0:
        #         cx = int(M["m10"] / M["m00"])
        #         cy = int(M["m01"] / M["m00"])
        #     else:
        #         cx, cy = 0, 0

        #     self.cheese_centers.append((cx, cy))

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
            # self.node_logger.info(
            #     f"Distance between cheese center {cheese_center} and bread center {bread_center}: {distance}"
            # )

            if distance < self.pass_threshold:
                valid_cheese = True
                break

        # Visualize cheese localization
        plot_image = image.copy()
        cv2.circle(plot_image, cheese_center, 5, (255, 0, 255), -1)

        # Visualize bread localization
        for contour in self.bread_contours:
            cv2.drawContours(plot_image, contour, -1, (255, 0, 0), 3)
        for center in self.bread_centers:
            cv2.circle(plot_image, center, 5, (0, 0, 255), -1)

        return (
            valid_cheese,
            plot_image,
        )  # if no cheese is placed in tray or cheese is not close to bread

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
            minRadius=45,
            maxRadius=60,
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
        cv2.circle(best_circle_img, (best_circle[0], best_circle[1]), 2, (0, 0, 255), 3)
        for center in self.bread_centers:
            cv2.circle(best_circle_img, center, 10, (255, 0, 0), -1)

        return is_ham_correct, best_circle_img

    def check_ingredient(self, image, ingredient_name):
        self.place_images.append(image)

        if ingredient_name == "bread":
            return self.check_bread(image)
        elif ingredient_name == "cheese":
            return self.check_cheese(image)
        elif ingredient_name == "ham":
            return self.check_ham(image)
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

    # Load the image
    bread_place_image = cv2.imread(
        "/home/parth/snaak/data/SCH_images_041125/cheese_check_1/image_20250411-150101.png"
    )
    bread_place_image = cv2.resize(bread_place_image, (848, 480))

    # Check bread placement
    bread_check, bread_check_image = sandwich_checker.check_ingredient(
        bread_place_image, "bread"
    )

    print(f"Is bread placed correctly? {bread_check}")

    # visualize bread placement
    cv2.imshow("Bread Check", bread_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cheese_place_image = cv2.imread(
        "/home/parth/snaak/data/SCH_images_041125/cheese_check_1/image_20250411-150115.png"
    )
    cheese_place_image = cv2.resize(cheese_place_image, (848, 480))

    # Check cheese placement
    cheese_check, cheese_check_image = sandwich_checker.check_ingredient(
        cheese_place_image, "cheese"
    )

    print(f"Is cheese placed correctly? {cheese_check}")

    # visualize cheese placement
    cv2.imshow("Two Cheese Check", cheese_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cheese_place_image = cv2.imread(
        "/home/parth/snaak/data/SCH_images_041125/cheese_check_1/image_20250411-150206.png"
    )

    # resize image to (848, 480)
    cheese_place_image = cv2.resize(cheese_place_image, (848, 480))

    # Check ham placement
    cheese_check, cheese_check_image = sandwich_checker.check_ingredient(
        cheese_place_image, "cheese"
    )

    print(f"Is cheese placed correctly? {cheese_check}")

    # visualize ham placement
    cv2.imshow("Ham Check", cheese_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
