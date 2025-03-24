import segmentation.segment_utils as seg_utils
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load bread image
bread_place_image = cv2.imread(
    "/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/sandwich_check_data/cheese_assembly/image_20250323-161448.png"
)

# get bread mask
contours = seg_utils.contour_segmentation(bread_place_image, show_image=False)
# to classify contours, several features can be used such as:
# shape, area, color, etc.
# using area for now
min_area = 30000  # Minimum area threshold for the bread slice
max_area = 35000  # Maximum area threshold for the bread slice
filtered_contours = [
    cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area
]
print(f"Number of filtered contours (potential bread slices): {len(filtered_contours)}")
# Visualize Filtered Contours
contour_image_filtered = bread_place_image.copy()
cv2.drawContours(
    contour_image_filtered, filtered_contours, -1, (0, 255, 0), thickness=2
)
plt.figure(figsize=(6, 6))
plt.title("Filtered Contours (Potential Bread Slices)")
plt.imshow(cv2.cvtColor(contour_image_filtered, cv2.COLOR_BGR2RGB))
plt.show()

# Load ingredient placed image
ingredient_place_image = cv2.imread(
    "/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/sandwich_check_data/cheese_assembly/image_20250323-161501.png"
)

# get difference mask
diff_mask = seg_utils.difference_mask(
    bread_place_image, ingredient_place_image, thresh=10
)

# add mask to ingredient placed image to highlight changed pixels
changed_pixels_image = cv2.bitwise_and(
    ingredient_place_image, ingredient_place_image, mask=diff_mask
)

# plt.figure(figsize=(6, 6))
# plt.title("Difference Mask")
# plt.imshow(cv2.cvtColor(changed_pixels_image, cv2.COLOR_BGR2RGB))
# plt.show()

# get contours from diff mask
contours = seg_utils.contour_segmentation(changed_pixels_image, show_image=False)

# get contours from tray area of ingredient placed image

# # get all contours from ingredient placed image
# contours = seg_utils.contour_segmentation(ingredient_place_image, show_image=False)

# # get tray contour by area
# min_area = 125000  # Minimum area threshold for the tray
# max_area = 127000  # Maximum area threshold for the tray
# filtered_contours = [
#     cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area
# ]
# print(f"Number of filtered contours (potential trays): {len(filtered_contours)}")
# # Visualize Filtered Contours
# contour_image_filtered = ingredient_place_image.copy()
# cv2.drawContours(
#     contour_image_filtered, filtered_contours, -1, (0, 255, 0), thickness=2
# )
# plt.figure(figsize=(6, 6))
# plt.title("Filtered Contours (Potential Trays)")
# plt.imshow(cv2.cvtColor(contour_image_filtered, cv2.COLOR_BGR2RGB))
# plt.show()

# segment out tray area from ingredient placed image
# tray_mask = np.zeros(ingredient_place_image.shape[:2], dtype=np.uint8)
# cv2.drawContours(tray_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)
# tray_area_image = cv2.bitwise_and(
#     ingredient_place_image, ingredient_place_image, mask=tray_mask
# )
# plt.figure(figsize=(6, 6))
# plt.title("Tray Area")
# plt.imshow(cv2.cvtColor(tray_area_image, cv2.COLOR_BGR2RGB))
# plt.show()

# # get contours inside tray
# contours = seg_utils.contour_segmentation(
#     tray_area_image, show_image=False, threshold=100, show_steps=True
# )
# # print number of contours detected
# print(f"Number of contours detected inside tray: {len(contours)}")
# # Visualize All Detected Contours
# contour_image_all = tray_area_image.copy()
# cv2.drawContours(contour_image_all, contours, -1, (0, 255, 0), thickness=2)
# plt.figure(figsize=(6, 6))
# plt.title("All Detected Contours Inside Tray")
# plt.imshow(cv2.cvtColor(contour_image_all, cv2.COLOR_BGR2RGB))
# plt.show()


# filter cheese contours by area (to isolate the cheese slice)
min_area = 25000  # Minimum area threshold for the cheese slice
max_area = 28000  # Maximum area threshold for the cheese slice
filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
print(f"Number of filtered contours (potential cheese slices): {len(filtered_contours)}")
# Visualize Filtered Contours
contour_image_filtered = ingredient_place_image.copy()
cv2.drawContours(contour_image_filtered, filtered_contours, -1, (0, 255, 0), thickness=2)
plt.figure(figsize=(6, 6))
plt.title("Filtered Contours (Potential Cheese Slices)")
plt.imshow(cv2.cvtColor(contour_image_filtered, cv2.COLOR_BGR2RGB))
plt.show()
