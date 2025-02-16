#!/usr/bin/python3

from unet.ingredients_UNet import Ingredients_UNet
from post_processing.image_utlis import ImageUtils

from PIL import Image
import cv2
import numpy as np



if __name__ == "__main__":

    # create UNet object
    # take image, run inference, get mask
    # take top_layer mask, binarize, get edges
    # take edges, get x, y
    # take x,y get z

    # Test initialisation for cheese
    Cheese_UNet = Ingredients_UNet(count=False, classes=["background","top_cheese","other_cheese"], model_path="logs/cheese/top_and_other/best_epoch_weights.pth")
    img_utils = ImageUtils()

    image = Image.open("test_image.png")
    mask = Cheese_UNet.detect_image(image)
    # mask.save("image_mask.png")
    # print(np.array(mask).shape)
    top_layer_mask = Cheese_UNet.get_top_layer(mask, [250, 106, 77])
    # top_layer_mask.show("Top Layer")

    binary_mask = Image.fromarray(
        img_utils.binarize_image(masked_img=np.array(top_layer_mask))
    )

    binary_mask_edges, cont = img_utils.find_edges_in_binary_image(np.array(binary_mask))
    # print(cont)
    center = img_utils.get_contour_center(cont)
    # draw center
    cv2.circle(binary_mask_edges, center, 2, (255, 255, 255), 1)

    binary_mask_edges = Image.fromarray(binary_mask_edges)
    # binary_mask_edges.show("top layer edges")
    binary_mask_edges = binary_mask_edges.convert('RGB')
    binary_mask_edges.save("top_layer_edges_center.png")

    # use center and get z #TODO
