"""
Script with helper functions for segmentation related tasks
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

############### Parameters #################

SUCTION_CUP_RADIUS = 0.03 * 1.1  # * 1.5 for buffer
# bottom left and top right point of bin in arm link0 frame
BIN1_PICKUP_AREA = [(0.562, -0.24), (0.69, -0.48)]
BIN2_PICKUP_AREA = [(0.372, -0.24), (0.5, -0.48)]
BIN3_PICKUP_AREA = [(0.177, -0.24), (0.307, -0.48)]

############################################


def convert_mask_to_orig_dims(
    cropped_mask, orig_img, crop_xmin, crop_ymin, crop_xmax, crop_ymax
):
    """
    Extend mask to match the original image dimensions by padding with zeros
    """
    orig_mask = np.zeros_like(orig_img[:, :, 0], dtype=np.uint8)
    orig_mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = cropped_mask
    return orig_mask


def show_mask(mask, ax, random_color=False, borders=True):
    """
    Show mask with matplotlib
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=150):
    """
    Plot points on an image using matplotlib
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="o",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="o",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    """
    Plot box on an image using matplotlib
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    """
    Displays all masks generated by SAM using matplotlib
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


def get_hsv_range(roi):
    """
    Get lower and upper hsv range for all pixels in the region of interest (roi)
    """

    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    h_min = np.min(roi_hsv[:, :, 0])
    h_max = np.max(roi_hsv[:, :, 0])
    s_min = np.min(roi_hsv[:, :, 1])
    s_max = np.max(roi_hsv[:, :, 1])
    v_min = np.min(roi_hsv[:, :, 2])
    v_max = np.max(roi_hsv[:, :, 2])

    # Define HSV lower and upper bounds
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    return lower_hsv, upper_hsv


def get_hsv_range_from_image(image):
    """
    Get lower and upper hsv range for all pixels in the image, ignoring black pixels
    """

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask to ignore black pixels
    mask = (
        (hsv_image[:, :, 1] > 0) & (hsv_image[:, :, 2] > 0) & (hsv_image[:, :, 0] > 0)
    )

    # Get the non-black pixels in the HSV image
    non_black_pixels = hsv_image[mask]

    # Calculate the min and max values for each channel
    h_min, s_min, v_min = np.min(non_black_pixels, axis=0)
    h_max, s_max, v_max = np.max(non_black_pixels, axis=0)

    # Define HSV lower and upper bounds
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])

    return lower_hsv, upper_hsv


def segment_from_hsv(image, lower_hsv, upper_hsv):
    """
    Segment given image using HSV segmentation
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    return mask


def calc_bbox_from_mask(mask):
    """
    Calculate a single box to bound all the white regions int he mask
    """
    bbox = None
    y_indices, x_indices = np.where(mask == 255)  # Get row (y) and column (x) indices

    # Compute bounding box coordinates
    if y_indices.size > 0 and x_indices.size > 0:  # Ensure there are white pixels
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        bbox = np.array([xmin, ymin, xmax, ymax])

    return bbox


def keep_largest_blob(binary_image):
    """
    Remove all white blobs from the mask except the largest blob
    """
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return binary_image

    largest_contour = max(contours, key=cv2.contourArea)
    largest_blob_image = np.zeros_like(binary_image)
    cv2.drawContours(
        largest_blob_image, [largest_contour], -1, 255, thickness=cv2.FILLED
    )
    return largest_blob_image


def is_valid_pickup_point(X_pickup, Y_pickup, bin_id, bread_bin_id):
    if bin_id == 1:
        pickup_area = BIN1_PICKUP_AREA
    elif bin_id == 2:
        pickup_area = BIN2_PICKUP_AREA
    elif bin_id == 3:
        pickup_area = BIN3_PICKUP_AREA
    else:
        raise Exception("Not a valid bin id")

    # bl -> bottom left, tr -> top right
    bl_X, bl_Y = pickup_area[0]
    tr_X, tr_Y = pickup_area[1]

    r_cup = SUCTION_CUP_RADIUS
    if bin_id == bread_bin_id:
        tr_Y += 0.1
        # need to account for camera, but not cup on far right side (this offsets addition below)
        tr_Y -= r_cup

    # Two y conditions since we can have negative y values, just want to make sure we are
    # in between the bounds
    if bl_X + r_cup <= X_pickup <= tr_X - r_cup and (
        bl_Y - r_cup <= Y_pickup <= tr_Y + r_cup
        or tr_Y + r_cup <= Y_pickup <= bl_Y - r_cup
    ):
        return True
    else:
        return False


def contour_segmentation(
    image,
    binary_threshold=150,
    show_image=True,
    show_separate_contours=False,
    show_steps=False,
    close_kernel_size=7,
    open_kernel_size=7,
    segment_type="binary",
    edges_thresholds=(30, 50),
):
    # can adjust the threshold values based on the image characteristics
    """
    Erosion and Dilation:

    Erosion: A larger kernel size will erode more of the foreground object, making it smaller. This is useful for removing small noise but can also remove small parts of the object.
    Dilation: A larger kernel size will dilate more of the foreground object, making it larger. This can fill in small holes and gaps but can also cause small objects to merge.
    Opening and Closing:

    Opening (Erosion followed by Dilation): A larger kernel size will remove larger noise and small objects from the foreground. It is useful for separating objects that are close to each other.
    Closing (Dilation followed by Erosion): A larger kernel size will close larger gaps and holes within the foreground object. It is useful for connecting disjointed parts of an object.

    """
    original_image = image.copy()

    if segment_type == "binary":

        # Step 1: Preprocessing (convert to grayscale and apply Gaussian blur)
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 2: Binary Thresholding
        _, binary = cv2.threshold(blurred, binary_threshold, 255, cv2.THRESH_BINARY_INV)

        # Step 3: Morphological Operations
        close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        binary_closed = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, close_kernel
        )  # Close gaps
        binary_opened = cv2.morphologyEx(
            binary_closed, cv2.MORPH_OPEN, open_kernel
        )  # Remove noise

        thresholded_image = binary_opened

        if show_steps:
            cv2.imshow("Binary Image", binary)
            cv2.imshow("Closed Image", binary_closed)
            cv2.imshow("Opened Image", binary_opened)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif segment_type == "edges":
        # Step 1: Preprocessing (convert to grayscale and apply Gaussian blur)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # # Apply histogram equalization
        # equalized = cv2.equalizeHist(gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # # Apply bilateral filtering
        # bilateral_filtered = cv2.bilateralFilter(equalized, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 5: Morphological Operations
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        # dilated = cv2.dilate(closed, kernel, iterations=2)

        # Step 2: Edge Detection
        edges = cv2.Canny(closed, edges_thresholds[0], edges_thresholds[1])

        # Step 3: Morphological Operations
        # dilate edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        # erode edges
        edges = cv2.erode(edges, kernel, iterations=1)

        thresholded_image = edges

        if show_steps:
            cv2.imshow("Adaptive Threshold", adaptive_thresh)
            cv2.imshow("Closed Image", closed)
            # cv2.imshow("Dilated Image", dilated)
            cv2.imshow("Edges", edges)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    elif segment_type == "hsv":
        # Step 1: Convert to HSV
        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Step 2: Thresholding
        lower_hsv = np.array([0, 50, 50])
        upper_hsv = np.array([10, 255, 255])
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

        # Step 3: Morphological Operations
        close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask_opened = cv2.morphologyEx(
            mask_closed, cv2.MORPH_OPEN, open_kernel
        )  # Remove noise

        thresholded_image = mask_opened

    # Step 4: Contour Detection
    contours, hierarchy = cv2.findContours(
        thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    # print(f"Number of contours detected: {len(contours)}")

    # Visualize All Detected Contours
    if show_image:
        contour_image_all = original_image.copy()
        cv2.drawContours(contour_image_all, contours, -1, (0, 255, 0), thickness=2)
        cv2.imshow("All Contours", contour_image_all)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Visualize each separate contour and its area
    if show_separate_contours:
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            print(f"Contour {i}: Area = {area}")

            # show only if area is above a certain threshold
            if area < 5000:
                continue

            # Create a mask for the current contour
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

            # Apply mask to the original image to segment the contour
            segmented_contour = cv2.bitwise_and(
                original_image, original_image, mask=mask
            )

            # Visualize the segmented contour
            cv2.imshow(f"Segmented Contour {i}, Area = {area}", segmented_contour)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return contours, hierarchy


def difference_mask(image1, image2, thresh):
    """
    Segment out those pixels that changed from the previous image
    """
    # convert images to grayscale for comparison
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # compute Absolute Difference
    difference = cv2.absdiff(gray1, gray2)

    # threshold the Difference to Isolate Changed Pixels
    _, thresholded_diff = cv2.threshold(
        difference, thresh, 255, cv2.THRESH_BINARY
    )  # 10 is the pixel difference threshold value - adjust according to your needs

    # morphological Operations
    kernel = np.ones((5, 5), np.uint8)
    thresholded_diff = cv2.morphologyEx(
        thresholded_diff, cv2.MORPH_CLOSE, kernel
    )  # Fill gaps

    # apply Mask to Original Image (Highlight Changes)
    # changed_pixels_image = cv2.bitwise_and(image2, image2, mask=mask)

    # # Visualization of Results
    # plt.figure(figsize=(12, 8))

    # plt.subplot(1, 3, 1)
    # plt.title("Image 1")
    # plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

    # plt.subplot(1, 3, 2)
    # plt.title("Image 2")
    # plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    # plt.subplot(1, 3, 3)
    # plt.title("Changed Pixels Highlighted")
    # plt.imshow(cv2.cvtColor(changed_pixels_image, cv2.COLOR_BGR2RGB))

    # plt.tight_layout()
    # plt.show()

    return thresholded_diff


def is_point_within_bounds(img, x, y):
    if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
        return False
    else:
        return True


def get_averaged_depth(depth_image, x, y, kernel_size=3):
    # Do an average of the kernel_sizexkernel_size window around x, y
    y_min = max(0, y - kernel_size // 2)
    y_max = min(depth_image.shape[0], y + kernel_size // 2)
    x_min = max(0, x - kernel_size // 2)
    x_max = min(depth_image.shape[1], x + kernel_size // 2)

    depth_window = depth_image[y_min:y_max, x_min:x_max]

    # Only use nonzero (valid, points)
    valid_window = (depth_window > 0).astype(int)
    total_sum = np.sum(depth_window)
    num_valid_points = np.sum(valid_window)

    if num_valid_points == 0:
        raise Exception("No valid depth value found")

    depth = float((total_sum / num_valid_points) / 1000)
    return depth
