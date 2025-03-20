import os
import cv2
from segmentation.meat_segment_generator import MeatSegmentGenerator

dir_path = "/home/user/data_collection/pr3_test/BAL_images_031925_3"
image_names = sorted(os.listdir(dir_path))

segmentor = MeatSegmentGenerator()

for img_name in image_names:

    img_path = os.path.join(dir_path, img_name)
    image = cv2.imread(img_path)
    x, y = segmentor.get_top_meat_slice_xy(image)

    cv2.circle(image, (x,y), 15, (255, 0, 0), -1)
    cv2.imwrite(f"{img_name}_result.jpg", image)
