"""
Script with helper functions for segmentation related tasks
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

############### Parameters #################

SUCTION_CUP_RADIUS = 0.03 * 1.5 # * 1.5 for buffer
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
    if (bin_id == 1):
        pickup_area = BIN1_PICKUP_AREA
    elif (bin_id == 2):
        pickup_area = BIN2_PICKUP_AREA
    elif (bin_id == 3):
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
    if bl_X + r_cup <= X_pickup <= tr_X - r_cup and (bl_Y - r_cup <= Y_pickup <= tr_Y + r_cup or tr_Y + r_cup <= Y_pickup <= bl_Y - r_cup):
        return True
    else:
        return False

def is_point_within_bounds(img, x, y):
    if (
            x < 0
            or x >= img.shape[1]
            or y < 0
            or y >= img.shape[0]
    ):
        return False
    else:
        return True
    
def get_averaged_depth(depth_image, x, y, kernel_size = 3):
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