#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from coordinates.srv import GetDepthAtPoint, GetXYZFromImage

from rclpy.qos import QoSProfile, DurabilityPolicy
from unet.Ingredients_UNet import Ingredients_UNet
from post_processing.image_utlis import ImageUtils
from tf2_msgs.msg import TFMessage
import numpy as np
from sensor_msgs.msg import CameraInfo
import cv2

#Make these config
CHEESE_BIN_ID = 1
HAM_BIN_ID = 2
BREAD_BIN_ID = 3


qos_profile = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)


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

        self.subscription_tf = self.create_subscription(TFMessage, '/tf', self.tf_listener_callback_tf, 10)
        self.subscription_intrinsics = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_intrinsics_callback, 10) # TODO fix this
        self.subscription_tf_static = self.create_subscription(TFMessage,'/tf_static', self.static_tf_listener_callback_tf_static, qos_profile)
        
        self.transformations = {}
        self.K = np.eye(3)
        self.distortion_coefficients = np.zeros((1, 5))
        self.width = 0
        self.height = 0

    def tf_listener_callback_tf(self, msg):
        """ Handle incoming transform messages. """
        for transform in msg.transforms:
            if transform.child_frame_id and transform.header.frame_id:
                self.transformations[(transform.header.frame_id, transform.child_frame_id)] = transform

    def tf_static_listener_callback_tf_static(self, msg):
        """ Handle incoming transform messages. """
        for transform in msg.transforms:
            if transform.child_frame_id and transform.header.frame_id:
                self.transformations[(transform.header.frame_id, transform.child_frame_id)] = transform

    def depth_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def camera_intriniscs_callback(self, msg):
        intrinsic_matrix = msg.k 
        self.K = np.array(intrinsic_matrix).reshape((3, 3))
        self.distortion_coefficients = msg.d
        self.width = msg.width
        self.height = msg.height

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

    def transform_location(self, x, y, depth):
        '''
        Modified Version of code from handeye_calibration_ros2
        '''
        T = np.eye(4)
        link_order = [
            ('panda_link0','panda_hand'), ('panda_hand','camera_color_optical_frame'),
        ]
        transform_matrices = {}
        for (frame_id, child_frame_id) in link_order:
            if (frame_id, child_frame_id) in self.transformations:
                trans = self.transformations[(frame_id, child_frame_id)].transform
                translation = [trans.translation.x, trans.translation.y, trans.translation.z]
                rotation = [trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w]
                T_local = np.eye(4)
                T_local[:3, :3] = self.quaternion_to_rotation_matrix(*rotation)
                T_local[:3, 3] = translation
                transform_matrices[(frame_id, child_frame_id)] = T_local

        T_link0_camera = transform_matrices[('panda_link0', 'panda_hand')]@transform_matrices[('panda_hand', 'camera_color_optical_frame')]

        distorted_point = np.array([[x, y]], dtype=np.float32)
        undistorted_point = cv2.undistortPoints(distorted_point, self.K, self.distortion_coefficients)
        x_undistorted, y_undistorted = undistorted_point[0][0]

        x = (x_undistorted - self.K[0, 2]) * depth / self.K[0, 0]
        y = (y_undistorted - self.K[1, 2]) * depth / self.K[1, 1]
        z = depth
        point = np.array([x, y, z])  
        return T_link0_camera@point    


    def handle_pickup_point(self, request, response):
        bin_ID = request.bin_ID
        timestamp = request.timestamp #use this to sync

        if bin_ID == CHEESE_BIN_ID: 
            # Cheese
            try:
                # Get X, Y
                mask = self.cheese_unet.detect_image(self.rgb_image)
                top_layer_mask = self.cheese_unet.get_top_layer(mask, [250, 106, 77]) #TODO make color a config
                binary_mask = Image.fromarray(self.img_utils.binarize_image(masked_img=np.array(top_layer_mask)))
                binary_mask_edges, cont = self.img_utils.find_edges_in_binary_image(np.array(binary_mask))
                (response.x, response.y) = self.img_utils.get_contour_center(cont)

                # Get Z
                # Ensure coordinates are within bounds of the image dimensions
                if response.x < 0 or response.x >= self.depth_image.shape[1] or \
                response.y < 0 or response.y >= self.depth_image.shape[0]:
                    raise ValueError("Coordinates out of bounds")

                # Retrieve depth value at (x, y)
                response.depth = float(self.depth_image[response.y, response.x]) / 1000.0  # Convert mm to meters
                self.get_logger().info(f"Got pickup point {response.x}, {response.y} and depth {response.depth:.2f} in bin {bin_ID} at {timestamp}")

                response_transformed = self.transform_location(response.x, response.y, response.z)
                response.x = response_transformed[0]
                response.y = response_transformed[1]
                response.z = response_transformed[2]

            except Exception as e:
                self.get_logger().error(f"Error while calculating pickup point: {e}")
                response.x = -1
                response.y = -1
                response.depth = float('nan')
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
