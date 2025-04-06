# class to check ingredient placement

import segmentation.segment_utils as seg_utils
import cv2


class SandwichChecker:
    def __init__(self):

        self.min_tray_area = 125000
        self.max_tray_area = 127000
        self.min_bread_area = 27000
        self.max_bread_area = 38000
        self.min_cheese_area = 25000
        self.max_cheese_area = 28000

        self.tray_contours = []

        self.bread_contours = []
        self.bread_centers = []

        self.cheese_contours = []
        self.cheese_centers = []

        self.ham_contours = []
        self.ham_centers = []

        self.place_images = []

    def reset(self):
        self.tray_contours = []
        self.tray_centers = []

        self.cheese_contours = []
        self.cheese_centers = []

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
            image, show_image=False, segment_type="binary"
        )

        # Extract tray contours
        tray_contour_indices = [
            i
            for i, contour in enumerate(contours)
            if self.min_tray_area < cv2.contourArea(contour) < self.max_tray_area
        ]
        self.tray_contours = [contours[i] for i in tray_contour_indices]
        print(f"Number of tray contours: {len(self.tray_contours)}")

        # Extract bread contours
        bread_contour_indices = [
            i
            for i, contour in enumerate(contours)
            if self.min_bread_area < cv2.contourArea(contour) < self.max_bread_area
        ]
        self.bread_contours = [contours[i] for i in bread_contour_indices]
        print(f"Number of bread contours: {len(self.bread_contours)}")

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
        bread_on_tray = False
        for tray_index in tray_contour_indices:
            for bread_index in bread_contour_indices:
                if hierarchy[0][bread_index][3] == tray_index:
                    bread_on_tray = True
                    break
            if bread_on_tray:
                break
        
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

    def check_cheese(self, image, threshold):
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
        print(f"Number of cheese contours: {len(self.cheese_contours)}")

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
                if distance < threshold:
                    print(
                        f"Euclidean distance between bread and cheese centers: {distance}"
                    )
                    valid_cheese = True
                else:
                    print(
                        f"Euclidean distance between bread and cheese centers: {distance}"
                    )
        
        # draw contours on image
        for contour in self.cheese_contours:
            cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
        # draw centers on image
        for center in self.cheese_centers:
            cv2.circle(image, center, 5, (255, 0, 0), -1)
        for center in self.bread_centers:
            cv2.circle(image, center, 5, (0, 0, 255), -1)
        
        return valid_cheese, image  # if no cheese is placed in tray or cheese is not close to bread

    def check_ingredient(self, image, ingredient_name, threshold=0):
        if ingredient_name == "bread":
            return self.check_bread(image)
        elif ingredient_name == "cheese":
            return self.check_cheese(image, threshold)
        elif ingredient_name == "ham":
            # TODO: implement ham check
            pass
        else:
            raise ValueError(f"Unknown ingredient: {ingredient_name}")


if __name__ == "__main__":    
    SandwichChecker = SandwichChecker()
    
    bread_place_image = cv2.imread(
        "/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/sandwich_check_data/cheese_assembly/image_20250323-161448.png"
    )
    
    # Check bread placement
    bread_check, bread_check_image = SandwichChecker.check_bread(bread_place_image)
    
    # visualize bread placement
    cv2.imshow("Bread Check", bread_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # cheese_place_image = cv2.imread(
    #     "/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/sandwich_check_data/cheese_ham_assembly/image_20250323-162333.png"
    # )
    
    # Check cheese placement
    # cheese_check, cheese_check_image = SandwichChecker.check_cheese(cheese_place_image, threshold=50)
    
    # # visualize cheese placement
    # cv2.imshow("Cheese Check", cheese_check_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    two_cheese_place_image = cv2.imread(
        "/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/sandwich_check_data/cheese_assembly/image_20250323-161745.png"
    )
    
    # Check cheese placement
    cheese_check, cheese_check_image = SandwichChecker.check_cheese(two_cheese_place_image, threshold=50)
    
    # visualize cheese placement
    cv2.imshow("Two Cheese Check", cheese_check_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    

