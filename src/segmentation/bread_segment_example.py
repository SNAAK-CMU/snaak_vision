# to segment a slice of bread from an image 

# TODO make this into a class so it can be used in vision node

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from img_utils import ImageUtils

ImageUtils = ImageUtils()

# Load the image
image = cv2.imread("/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/sandwich_check_data/cheese_assembly/image_20250323-161448.png")
original_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

# Step 1: Preprocessing (convert to grayscale and apply Gaussian blur)
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Visualize Grayscale and Blurred Image
# plt.figure(figsize=(12, 8))
# plt.subplot(1, 3, 1)
# plt.title("Original Image")
# plt.imshow(image)

# plt.subplot(1, 3, 2)
# plt.title("Grayscale Image")
# plt.imshow(gray, cmap="gray")

# plt.subplot(1, 3, 3)
# plt.title("Blurred Image")
# plt.imshow(blurred, cmap="gray")
# plt.show()

# Step 2: Thresholding
_, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

# Visualize Binary Image
# plt.figure(figsize=(6, 6))
# plt.title("Binary Image (Thresholded)")
# plt.imshow(binary, cmap="gray")
# plt.show()

# Step 3: Morphological Operations
kernel = np.ones((5, 5), np.uint8)
binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Close gaps
binary_opened = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel)   # Remove noise

# Visualize Morphological Operations
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("After Closing (Fill Gaps)")
plt.imshow(binary_closed, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("After Opening (Remove Noise)")
plt.imshow(binary_opened, cmap="gray")
plt.show()

# Step 4: Contour Detection
# contour_image, contours = ImageUtils.find_edges_in_binary_image(binary_opened)
contours, heirarchy = cv2.findContours(binary_opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"Number of contours detected: {len(contours)}")

# Visualize All Detected Contours
contour_image_all = original_image.copy()
cv2.drawContours(contour_image_all, contours, -1, (0, 255, 0), thickness=2)

plt.figure(figsize=(6, 6))
plt.title("All Detected Contours")
plt.imshow(cv2.cvtColor(contour_image_all, cv2.COLOR_BGR2RGB))
plt.show()

# Save contours image for debugging and info collection - need contour "description" for bread to identify it, so using area as the descriptor now

# # Create directory if it doesn't exist
# output_dir = "/Users/abhi/Documents/CMU/2024-25/Projects/SNAAK/Vision/contour_detection_bins"
# os.makedirs(output_dir, exist_ok=True)

num_contours = len(contours)
cols = 3
rows = (num_contours // cols) + 1

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print(f"Contour {i}: Area = {area}")

    # Create a mask for the current contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Apply mask to the original image to segment the contour
    segmented_contour = cv2.bitwise_and(original_image, original_image, mask=mask)
    
    # Save the segmented contour image
    # segmented_contour_bgr = cv2.cvtColor(segmented_contour, cv2.COLOR_RGB2BGR)
    #cv2.imwrite(os.path.join(output_dir, f"segmented_contour_{i}.png"), segmented_contour)

    # Add the segmented contour
    axes[i].imshow(cv2.cvtColor(segmented_contour, cv2.COLOR_BGR2RGB))
    axes[i].set_title(f"Contour {i} (Area = {area})")
    axes[i].axis('off')

    # Visualize the segmented contour
    plt.figure(figsize=(6, 6))
    plt.title(f"Segmented Contour {i} (Area = {area})")
    plt.imshow(cv2.cvtColor(segmented_contour, cv2.COLOR_BGR2RGB))
# # plt.show()

# # Hide any unused subplots
# for j in range(i + 1, len(axes)):
#     fig.delaxes(axes[j])

# plt.tight_layout()
# plt.show()

# Step 5: Filter Contours by Area (to isolate the bread slice)
min_area = 30000  # Minimum area threshold for the bread slice
max_area = 35000  # Maximum area threshold for the bread slice
filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
print(f"Number of filtered contours (potential bread slices): {len(filtered_contours)}")
# Visualize Filtered Contours
contour_image_filtered = original_image.copy()
cv2.drawContours(contour_image_filtered, filtered_contours, -1, (0, 255, 0), thickness=2)
plt.figure(figsize=(6, 6))
plt.title("Filtered Contours (Potential Bread Slices)")
plt.imshow(cv2.cvtColor(contour_image_filtered, cv2.COLOR_BGR2RGB))
plt.show()

