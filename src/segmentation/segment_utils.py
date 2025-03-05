import numpy as np
import matplotlib.pyplot as plt
import cv2



def convert_mask_to_orig_dims(cropped_mask, orig_img, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
    orig_mask = np.zeros_like(orig_img[:, :, 0], dtype=np.uint8)
    orig_mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = cropped_mask
    return orig_mask

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=150):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
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
        plt.axis('off')
        plt.show()

def get_hsv_range(roi):
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
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    return mask

def calc_bbox_from_mask(mask):
    bbox = None
    y_indices, x_indices = np.where(mask == 255)  # Get row (y) and column (x) indices

    # Compute bounding box coordinates
    if y_indices.size > 0 and x_indices.size > 0:  # Ensure there are white pixels
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        bbox = np.array([xmin, ymin, xmax, ymax])
    
    return bbox


def convert_mask_to_orig_dims(cropped_mask, orig_img, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
    orig_mask = np.zeros_like(orig_img[:, :, 0], dtype=np.uint8)
    orig_mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = cropped_mask
    return orig_mask


def get_top_from_all_cheese(cheese_w, chees_h, all_cheese_mask):
    kernel = np.ones((5, 5), np.uint8) 
    mask = cv2.erode(all_cheese_mask, kernel, iterations=1)
    

def get_hsv_range(roi):
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
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    return mask

def calc_bbox_from_mask(mask):
    bbox = None
    y_indices, x_indices = np.where(mask == 255)  # Get row (y) and column (x) indices

    # Compute bounding box coordinates
    if y_indices.size > 0 and x_indices.size > 0:  # Ensure there are white pixels
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        bbox = np.array([xmin, ymin, xmax, ymax])
    
    return bbox


def convert_mask_to_orig_dims(cropped_mask, orig_img, crop_xmin, crop_ymin, crop_xmax, crop_ymax):
    orig_mask = np.zeros_like(orig_img[:, :, 0], dtype=np.uint8)
    orig_mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = cropped_mask
    return orig_mask


def get_top_from_all_cheese(cheese_w, chees_h, all_cheese_mask):
    kernel = np.ones((5, 5), np.uint8) 
    mask = cv2.erode(all_cheese_mask, kernel, iterations=1)
    

