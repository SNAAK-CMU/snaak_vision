import os
import numpy as np
import cv2
from segmentation.bread_segment_generator import BreadSegmentGenerator

dir_path = "/home/user/data_collection/assembly_bread/"
image_names = sorted(os.listdir(dir_path))

segmentor = BreadSegmentGenerator()

for img_name in image_names:

    img_path = os.path.join(dir_path, img_name)
    image = cv2.imread(img_path)
    mask = segmentor.get_bread_mask(image)

    cv2.imwrite(f"{img_name}_result_mask.jpg", mask)

    y_coords, x_coords = np.where(mask == 255)
    x = int(np.mean(x_coords))
    y = int(np.mean(y_coords))

    cv2.circle(image, (x, y), 15, (255, 0, 0), -1)
    cv2.imwrite(f"{img_name}_result.jpg", image)
