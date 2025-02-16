#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from coordinates.srv import GetDepthAtPoint, GetXYZFromImage

from unet.Ingredients_UNet import Ingredients_UNet
from post_processing.image_utlis import ImageUtils


#Make these config
CHEESE_BIN_ID = 1
HAM_BIN_ID = 2
BREAD_BIN_ID = 3


class VisionNode(Node):
    def __init__(self):
        super().__init__('depth_service')
        self.bridge = CvBridge()
        self.depth_image = None
        self.rgb_image = None

        # Start UNet 
        self.cheese_unet = Ingredients_UNet(count=False, classes=["background","top_cheese","other_cheese"], model_path="logs/cheese/top_and_other/best_epoch_weights.pth") #TODO make these config 

        # post processing stuff
        self.img_utils = ImageUtils()

        # Subscribe to depth image topic (adjust topic name as needed)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10)

        # Subscribe to rgb image topic
        self.RGB_subscription = self.create_subscription(
            Image, 
            '', #TODO!
            self.rgb_callback,
            10
        )

        # Create the service server
        self.service = self.create_service(GetDepthAtPoint, 'get_depth_at_point', self.handle_get_depth)

        # Create the pickup point service server
        self.pickup_point_service = self.create_service(GetXYZFromImage, 'get_pickup_point', self.handle_pickup_point)

    def depth_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def handle_get_depth(self, request, response): #separate this from service callback stuff so the same function can be used for pickup point service
        if self.depth_image is None:
            self.get_logger().warn("Depth image not available yet!")
            response.depth = float('nan')
            return response

        try:
            # Ensure coordinates are within bounds of the image dimensions
            if request.x < 0 or request.x >= self.depth_image.shape[1] or \
               request.y < 0 or request.y >= self.depth_image.shape[0]:
                raise ValueError("Coordinates out of bounds")

            # Retrieve depth value at (x, y)
            response.depth = float(self.depth_image[request.y, request.x]) / 1000.0  # Convert mm to meters
            self.get_logger().info(f"Depth at ({request.x}, {request.y}): {response.depth} meters")
        except Exception as e:
            self.get_logger().error(f"Error retrieving depth: {e}")
            response.depth = float('nan')

        return response
    
    def handle_pickup_point(self, request, response):
        bin_ID = request.bin_ID
        timestamp = request.timestamp #use this to sync

        if bin_ID == CHEESE_BIN_ID: 
            # Cheese
            try:
                # Get X, Y
                mask = self.cheese_unet.detect_image(image)
                top_layer_mask = self.cheese_unet.get_top_layer(mask, [250, 106, 77]) #TODO make color a config
                binary_mask = Image.fromarray(img_utils.binarize_image(masked_img=np.array(top_layer_mask)))
                binary_mask_edges, cont = img_utils.find_edges_in_binary_image(np.array(binary_mask))
                (response.x, response.y) = img_utils.get_contour_center(cont)

                # Get Z
                # Ensure coordinates are within bounds of the image dimensions
                if response.x < 0 or response.x >= self.depth_image.shape[1] or \
                response.y < 0 or response.y >= self.depth_image.shape[0]:
                    raise ValueError("Coordinates out of bounds")

                # Retrieve depth value at (x, y)
                response.depth = float(self.depth_image[response.y, response.x]) / 1000.0  # Convert mm to meters
                self.get_logger().info(f"Got pickup point {response.x}, {response.y} and depth {response.depth:.2f} in bin {bin_ID} at {timestamp}")
            except Exception as e:
                self.get_logger().error(f"Error while calculating pickup point: {e}")
                response.x = -1
                response.y = -1
                response.depth = float('nan')
        
        return response        

def main(args=None):
    rclpy.init(args=args)
    node = DepthService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
