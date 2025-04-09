# class to check ingredient placement

import numpy as np
import cv2
import segmentation.segment_utils as seg_utils


########## Parameters ##########

BREAD_HSV_LOWER_BOUND = (10, 30, 100)
BREAD_HSV_UPPER_BOUND = (40, 255, 255)

TRAY_HSV_LOWER_BOUND = (85, 50, 20)
TRAY_HSV_UPPER_BOUND = (100, 255, 255)

TRAY_BOX_PIX = (
    250,
    20,
    630,
    300,
)  # (x1, y1, x2, y2) coordinates of the tray box in the image

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
        node_logger=None
    ):

        self.tray_hsv_lower_bound = TRAY_HSV_LOWER_BOUND
        self.tray_hsv_upper_bound = TRAY_HSV_UPPER_BOUND
        self.bread_hsv_lower_bound = BREAD_HSV_LOWER_BOUND
        self.bread_hsv_upper_bound = BREAD_HSV_UPPER_BOUND


        self.fov_width = fov_width
        self.fov_height = fov_height
        self.threshold_in_cm = threshold_in_cm
        self.image_height = image_height
        self.image_width = image_width

        self.pix_per_m = (
            (self.image_width / self.fov_width) + (self.image_height / self.fov_height)
        ) / 2
        self.pass_threshold = self.pix_per_m * (self.threshold_in_cm / 100)
        
        # Calculate the area of the tray and bread in pixels
        self.tray_area = tray_dims_m[0] * tray_dims_m[1] * self.pix_per_m ** 2
        self.bread_area = bread_dims_m[0] * bread_dims_m[1] * self.pix_per_m ** 2
        self.cheese_area = cheese_dims_m[0] * cheese_dims_m[1] * self.pix_per_m ** 2
        self.min_tray_area = 0.9 * self.tray_area
        self.max_tray_area = 1.4 * self.tray_area
        self.min_bread_area = 0.9 * self.bread_area
        self.max_bread_area = 1.1 * self.bread_area
        self.min_cheese_area = 0.9 * self.cheese_area
        self.max_cheese_area = 1.1 * self.cheese_area
        
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
        self.tray_contours = []
        self.tray_centers = []

        self.cheese_contours = []
        self.cheese_centers = []

        self.bread_centers = []
        self.bread_contours = []

        self.ham_contours = []
        self.ham_centers = []

        self.place_images = []

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

        contours, hierarchy = seg_utils.contour_segmentation(
            image, show_image=False, segment_type="binary", show_separate_contours=False, binary_threshold=170, show_steps=False
        )

        # Extract tray contours
        tray_contour_indices = [
            i
            for i, contour in enumerate(contours)
            if self.min_tray_area < cv2.contourArea(contour) < self.max_tray_area
        ]
        self.tray_contours = [contours[i] for i in tray_contour_indices]
        self.node_logger.info(
            f"Number of tray contours: {len(self.tray_contours)}"
        )

        # Extract bread contours
        bread_contour_indices = [
            i
            for i, contour in enumerate(contours)
            if self.min_bread_area < cv2.contourArea(contour) < self.max_bread_area
        ]
        self.bread_contours = [contours[i] for i in bread_contour_indices]
        self.node_logger.info(
            f"Number of bread contours: {len(self.tray_contours)}"
        )

        for bread_contour in self.bread_contours:
            # Calculate the center of the bread contour
            M = cv2.moments(bread_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            self.bread_centers.append((cx, cy))

        # Check if bread is placed in tray
        bread_on_tray = True
        #TODO measure distance between bread and tray centers

        # draw contours on image
        for contour in self.tray_contours:
            cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
        for contour in self.bread_contours:
            cv2.drawContours(image, contour, -1, (255, 0, 0), 3)
        # draw centers on image
        for center in self.bread_centers:
            cv2.circle(image, center, 5, (0, 0, 255), -1)

        # TODO: save detection image to self.place_images

        return bread_on_tray, image

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

        contours, hierarchy = seg_utils.contour_segmentation(
            image,
            show_image=False,
            show_steps=False,
            segment_type="edges",
            show_separate_contours=False,
            edges_thresholds=(30, 50),
        )

        # Extract cheese contours
        cheese_contour_indices = [
            i
            for i, contour in enumerate(contours)
            if self.min_cheese_area < cv2.contourArea(contour) < self.max_cheese_area
        ]
        self.cheese_contours = [contours[i] for i in cheese_contour_indices]
        self.node_logger.info(
            f"Number of cheese contours: {len(self.cheese_contours)}"
        )

        for cheese_contour in self.cheese_contours:
            # Calculate the center of the cheese contour
            M = cv2.moments(cheese_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            self.cheese_centers.append((cx, cy))

        # threshold distance between all cheese centers and bread centers
        valid_cheese = False
        for bread_center in self.bread_centers:
            for cheese_center in self.cheese_centers:
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

        # draw contours on image
        for contour in self.cheese_contours:
            cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
        for contour in self.bread_contours:
            cv2.drawContours(image, contour, -1, (255, 0, 0), 3)
        for contour in self.tray_contours:
            cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
        # draw centers on image
        for center in self.cheese_centers:
            cv2.circle(image, center, 5, (255, 0, 255), -1)
        for center in self.bread_centers:
            cv2.circle(image, center, 5, (0, 0, 255), -1)

        return (
            valid_cheese,
            image,
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
        "/home/parth/snaak/data/SCH_images_032335/cheese_ham_assembly/image_20250323-162330.png"
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
        "/home/parth/snaak/data/SCH_images_032335/cheese_ham_assembly/image_20250323-162333.png"
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

    ham_place_image = cv2.imread(
        "/home/parth/snaak/data/SCH_images_032335/cheese_ham_assembly/image_20250323-162337.png"
    )

    # resize image to (848, 480)
    ham_place_image = cv2.resize(ham_place_image, (848, 480))

    # Check ham placement
    ham_check, ham_check_image = sandwich_checker.check_ingredient(
        ham_place_image, "ham"
    )

    print(f"Is ham placed correctly? {ham_check}")

    # visualize ham placement
    cv2.imshow("Ham Check", ham_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
