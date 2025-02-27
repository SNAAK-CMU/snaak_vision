#!/usr/bin/python3

import cv2
import numpy as np

class ImageUtils:
    def __init__(self) -> None:
        pass

    def binarize_image(self, masked_img):
        gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        retval, im_bn = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )  # change 3rd param to invert binary image

        return im_bn

    def find_edges_in_binary_image(self, binary_image):
        black_img = np.zeros((np.shape(binary_image)[0], np.shape(binary_image)[1]))  # change to width height from image
        cont, heirarchy = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        im_cont = cv2.drawContours(black_img, cont, -1, (255, 255, 255), 1)
        return im_cont, cont

    def get_contour_center(self, contours):
        for contour in contours:
            M = cv2.moments(contour)
            
            # Calculate the center of the contour (centroid)
            if M["m00"] != 0:  # Avoid division by zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                print(f"Center of contour: ({cx}, {cy})")
                return (cx, cy)  # Return the center
            else:
                print("Contour area is zero, skipping.")
                return (None, None)
