#!/usr/bin/python3

from unet.predict_cheese import Cheese_UNet

# create UNet object
# take image, run inference, get mask
# take top_layer mask, binarize, get edges
# take edges, get x, y
# take x,y get z
