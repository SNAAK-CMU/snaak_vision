import segmentation.segment_utils as seg_utils
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys

# STEP 1: BREAD PLACED AND ARM MOVED TO CHECK POSITION - Capture Image from here

# Load bread image
bread_place_image = cv2.imread(
    "/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/sandwich_check_data/cheese_assembly/image_20250323-161907.png"
)

# STEP 2: CHECK BREAD PLACEMENT ON TRAY 

# get all contours from placed image
contours, hierarchy = seg_utils.contour_segmentation(bread_place_image, show_image=False)

# NOTE: to classify contours, several features can be used such as:
        # shape, area, color, etc.
        # using area for now

# get tray contour indices by area
min_tray_area = 125000  # Minimum area threshold for the tray
max_tray_area = 127000  # Maximum area threshold for the tray
tray_contour_indices = [
    i for i, cnt in enumerate(contours) if min_tray_area < cv2.contourArea(cnt) < max_tray_area
]
print(f"Number of filtered contours (potential trays): {len(tray_contour_indices)}")

# Visualize Filtered Tray Contours
contour_image_filtered = bread_place_image.copy()
cv2.drawContours(
    contour_image_filtered, [contours[i] for i in tray_contour_indices], -1, (0, 255, 0), thickness=2
)
plt.figure(figsize=(6, 6))
plt.title("Filtered Contours (Potential Trays)")
plt.imshow(cv2.cvtColor(contour_image_filtered, cv2.COLOR_BGR2RGB))
plt.show()

# get bread contour indices by area
min_bread_area = 30000  # Minimum area threshold for the bread slice
max_bread_area = 35000  # Maximum area threshold for the bread slice
bread_contour_indices = [
    i for i, cnt in enumerate(contours) if min_bread_area < cv2.contourArea(cnt) < max_bread_area
]
print(f"Number of filtered contours (potential bread slices): {len(bread_contour_indices)}")

# Visualize Filtered Bread Contours
contour_image_filtered = bread_place_image.copy()
cv2.drawContours(
    contour_image_filtered, [contours[i] for i in bread_contour_indices], -1, (0, 255, 0), thickness=2
)
plt.figure(figsize=(6, 6))
plt.title("Filtered Contours (Potential Bread Slices)")
plt.imshow(cv2.cvtColor(contour_image_filtered, cv2.COLOR_BGR2RGB))
plt.show()

# check if bread is placed on tray
bread_on_tray = False

# does the bread slice contour fall completely inside the tray contour?
for tray_index in tray_contour_indices:
    for bread_index in bread_contour_indices:
        if hierarchy[0][bread_index][3] == tray_index:
            bread_on_tray = True
            break
    if bread_on_tray:
        break

if not bread_on_tray:
    print("Bread slice is not placed on the tray.")
    sys.exit(1)
else:
    print("Bread slice is placed on the tray!")
    # get bread slice contour pixels
    bread_contour_pixels = np.zeros_like(bread_place_image)
    cv2.drawContours(bread_contour_pixels, [contours[bread_contour_indices[0]]], -1, (255), thickness=cv2.FILLED)

# STEP 3: PLACE INGREDIENT ON BREAD SLICE    

# Load ingredient placed image
ingredient_place_image = cv2.imread(
    "/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/sandwich_check_data/cheese_assembly/image_20250323-161850.png"
)

# STEP 4: CHECK INGREDIENT PLACEMENT ON BREAD SLICE

# get difference mask
diff_mask = seg_utils.difference_mask(
    bread_place_image, ingredient_place_image, thresh=18
) 
# need to tune the "difference threshold" accordingly (10-20 seems to work well for now)

# add mask to ingredient placed image to highlight changed pixels
changed_pixels_image = cv2.bitwise_and(
    ingredient_place_image, ingredient_place_image, mask=diff_mask
)

plt.figure(figsize=(6, 6))
plt.title("Difference Mask")
plt.imshow(cv2.cvtColor(changed_pixels_image, cv2.COLOR_BGR2RGB))
plt.show()

# get contours from diff mask
diff_image_contours, diff_image_hierarchy = seg_utils.contour_segmentation(changed_pixels_image, show_image=False)

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
filtered_contours = [cnt for cnt in diff_image_contours if min_area < cv2.contourArea(cnt) < max_area]
print(f"Number of filtered contours (potential cheese slices): {len(filtered_contours)}")
# Visualize Filtered Contours
contour_image_filtered = ingredient_place_image.copy()
cv2.drawContours(contour_image_filtered, filtered_contours, -1, (0, 255, 0), thickness=2)
plt.figure(figsize=(6, 6))
plt.title("Filtered Contours (Potential Cheese Slices)")
plt.imshow(cv2.cvtColor(contour_image_filtered, cv2.COLOR_BGR2RGB))
plt.show()

# calculate overlap between bread and cheese slices as percentage of cheese slice area
bread_cheese_overlap = 0

# get cheese slice contour pixels
cheese_contour_pixels = np.zeros_like(ingredient_place_image)
cv2.drawContours(cheese_contour_pixels, filtered_contours, -1, (255), thickness=cv2.FILLED)

# calculate overlap
# NOTE: The bread is not a rectangle, so the overlap currently computes the raw pixel overlap - improve this by fitting a rectangle around the bread slice?
overlap_pixels = cv2.bitwise_and(bread_contour_pixels, cheese_contour_pixels)

# visualize overlap
plt.figure(figsize=(6, 6))
plt.title("Overlap between Bread and Cheese Slices")
plt.imshow(overlap_pixels, cmap="gray")
plt.show()

# calculate overlap as percentage of cheese slice area
cheese_area = cv2.contourArea(filtered_contours[0])
overlap_area = np.count_nonzero(overlap_pixels)
bread_cheese_overlap = (overlap_area / cheese_area) * 100

prev_placement_scene = ingredient_place_image.copy

# STEP 6: PLACE NEXT INGREDIENT

ingredient_place_image = cv2.imread("/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/sandwich_check_data/cheese_assembly/image_20250323-161857.png")

# get difference mask
diff_mask = seg_utils.difference_mask(
    bread_place_image, ingredient_place_image, thresh=10
) 
# need to tune the "difference threshold" accordingly (10-20 seems to work well for now)

# add mask to ingredient placed image to highlight changed pixels
changed_pixels_image = cv2.bitwise_and(
    ingredient_place_image, ingredient_place_image, mask=diff_mask
)

plt.figure(figsize=(6, 6))
plt.title("Difference Mask")
plt.imshow(cv2.cvtColor(changed_pixels_image, cv2.COLOR_BGR2RGB))
plt.show()



