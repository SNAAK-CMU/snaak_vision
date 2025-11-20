#!/home/snaak/Documents/manipulation_ws/src/snaak_vision/.venv/bin/python3

# Change above line if chaning .venv location
# author: Oliver
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from snaak_vision.srv import GetDepthAtPoint
from snaak_vision.srv import GetXYZFromImage
from snaak_vision.srv import CheckIngredientPlace
from std_srvs.srv import Trigger
from std_srvs.srv import Trigger
import traceback
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from PIL import Image as Im
import os
from collections import deque


from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_msgs.msg import TFMessage
import numpy as np
from sensor_msgs.msg import CameraInfo
import cv2
from scipy.spatial.transform import Rotation as R
import copy
from segmentation.cheese_segment_generator import CheeseSegmentGenerator
from segmentation.tray_segment_generator import TraySegmentGenerator
from segmentation.bread_segment_generator import BreadSegmentGenerator
from segmentation.meat_segment_generator import MeatSegmentGenerator
from segmentation.segment_utils import (
    calc_bbox_from_mask,
    is_valid_pickup_point,
    is_point_within_bounds,
    get_averaged_depth,
)

from segmentation.UNet.ingredients_UNet import Ingredients_UNet

from sandwich_checker import SandwichChecker
from datetime import datetime

############### Parameters #################

USE_UNET = True  # Set to True to use UNet, False to use SAM
USE_UNET_FOR_CHECK = True

# Assembly location IDs
ASSEMBLY_TRAY_ID = 4
ASSEMBLY_BREAD_ID = 5

FOV_WIDTH = 0.775  # metres
FOV_HEIGHT = 0.435  # metres
SW_CHECKER_THRESHOLD = 3  # cm

IMG_WIDTH = 848
IMG_HEIGHT = 480

TRAY_CENTER = [0.48, 0.0, 0.29]  # in arm frame
TRAY_CENTER_PIXELS = [426, 202]  # in image pixels

PICKUP_Z_OFFSET = 0.004

# Bin coordinates in wrist camera image from pregrasp position (used for masking the image)
# Bin2 coords
BIN2_XMIN = 250
BIN2_YMIN = 0
BIN2_XMAX = 470
BIN2_YMAX = 340

# Bin1 coords
BIN1_XMIN = 240
BIN1_YMIN = 0
BIN1_XMAX = 450
BIN1_YMAX = 330

# Bin3 coords
BIN3_XMIN = 355
BIN3_YMIN = 0
BIN3_XMAX = 565
BIN3_YMAX = 330

# Bin 4 coords
BIN4_XMIN = 81
BIN4_YMIN = 66
BIN4_XMAX = 440
BIN4_YMAX = 297

# Bin 5 coords
BIN5_XMIN = 67
BIN5_YMIN = 72
BIN5_XMAX = 434
BIN5_YMAX = 305

# Bin 6 coords
BIN6_XMIN = 69
BIN6_YMIN = 94
BIN6_XMAX = 433
BIN6_YMAX = 316

# BIN_COORDS
BIN_COORDS = [
    [BIN1_XMIN, BIN1_YMIN, BIN1_XMAX, BIN1_YMAX],
    [BIN2_XMIN, BIN2_YMIN, BIN2_XMAX, BIN2_YMAX],
    [BIN3_XMIN, BIN3_YMIN, BIN3_XMAX, BIN3_YMAX],
    [BIN4_XMIN, BIN4_YMIN, BIN4_XMAX, BIN4_YMAX],
    [BIN5_XMIN, BIN5_YMIN, BIN5_XMAX, BIN5_YMAX],
    [BIN6_XMIN, BIN6_YMIN, BIN6_XMAX, BIN6_YMAX],
]

# Cheese Dimensions in metres
CHEESE_WIDTH_MOZARELLA = 0.090
CHEESE_HEIGHT_MOZARELLA = 0.095

CHEESE_TOP_SLICE_PNG = [250, 106, 77]  # the mask color for top cheese slice

# Bread Dimensions in metres
BREAD_WIDTH = 0.15
BREAD_HEIGHT = 0.10
BREAD_TOP_SLICE_PNG = [250, 106, 77]  # the mask color for top cheese slice

# Tray Dimensions in metres
TRAY_WIDTH = 0.305
TRAY_HEIGHT = 0.220

# Ham Dimensions in metres
# 1098 pix/m ; ham_radius = 52 pix
BOLOGNA_RADIUS = 0.05  # metres
MEAT_TOP_SLICE_PNG = [61, 61, 245]  # the mask color for top bologna slice

FAILURE_IMAGES_PATH = "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/failure_images/"

#########################################################################

class VisionNode(Node):
    def __init__(self):
        super().__init__("snaak_vision")
        self.bridge = CvBridge()
        self.depth_image = None
        self.depth_queue = deque()
        self.rgb_image = None

        # camera FOV over assembly area
        self.fov_w = FOV_WIDTH  # metres
        self.fov_h = FOV_HEIGHT  # metres
        self.threshold_in_cm = SW_CHECKER_THRESHOLD

        # meters/pixel
        self.pixels_to_m = (
            (FOV_WIDTH / IMG_WIDTH) + (FOV_HEIGHT / IMG_HEIGHT)
        ) / 2  # how many metres per pixel

        # tray dimensions
        self.tray_width = TRAY_WIDTH
        self.tray_height = TRAY_HEIGHT

        # ingredient dimensions
        self.bread_width = BREAD_WIDTH
        self.bread_height = BREAD_HEIGHT
        self.cheese_width = CHEESE_WIDTH_MOZARELLA
        self.cheese_height = CHEESE_HEIGHT_MOZARELLA
        self.meat_radius = BOLOGNA_RADIUS
        self.cheese_area_pixels = (
            self.cheese_width * self.cheese_height * (1 / self.pixels_to_m**2)
        )
        self.meat_area_pixels = (
            np.pi * (self.meat_radius**2) * (1 / self.pixels_to_m**2)
        )

        # image size
        self.image_width = IMG_WIDTH
        self.image_height = IMG_HEIGHT
        self.detection_image = None
        self.detection_image_count = 0

        # init SAM segment generators
        self.use_SAM = not USE_UNET
        if self.use_SAM:
            self.get_logger().info("Using SAM for segmentation")
            try:
                self.cheese_segment_generator = (CheeseSegmentGenerator() if self.use_SAM else None)
                self.tray_segment_generator = TraySegmentGenerator() if self.use_SAM else None
                self.bread_segment_generator = BreadSegmentGenerator() if self.use_SAM else None
                self.meat_segment_generator = MeatSegmentGenerator() if self.use_SAM else None
            except Exception as e:
                self.get_logger().error(f"Error initializing segment generators: {e}")
                raise e
        
        # init UNets
        self.use_UNet = USE_UNET
        if self.use_UNet:
            self.get_logger().info("Using UNet for segmentation")

            try:
                self.Cheese_UNet = Ingredients_UNet(
                    count=False,
                    classes=["background", "top_cheese", "other_cheese"],
                    model_path="logs/cheese/UNet_CHE_000/best_epoch_weights.pth",  # choose weights
                    mix_type=1,
                    num_classes=3,
                )

                self.Bologna_UNet = Ingredients_UNet(
                    count=False,
                    classes=["background", "", "", "top_bologna", "other_bologna"],
                    model_path="logs/ham/UNet_BOL_000/best_epoch_weights.pth",
                    mix_type=1,
                    num_classes=5,
                )
                self.Bread_UNet = Ingredients_UNet(
                    count=False,
                    classes=["background", "top_bread", "other_bread"],
                    mix_type=1,
                    num_classes=3,
                    model_path="logs/bread/UNet_BRE_001/best_epoch_weights.pth",
                )
            except Exception as e:
                self.get_logger().error(f"Error initializing UNet models: {e}")
                raise e

        # init sandwich checker
        self.use_UNet_for_check = USE_UNET_FOR_CHECK
        self.sandwich_checker = SandwichChecker(
            fov_height=self.fov_h,
            fov_width=self.fov_w,
            threshold_in_cm=self.threshold_in_cm,
            image_width=self.image_width,
            image_height=self.image_height,
            node_logger=self.get_logger(),
            cheese_dims_m=[self.cheese_width, self.cheese_height],
            tray_dims_m=[self.tray_width, self.tray_height],
            bread_dims_m=[self.bread_width, self.bread_height],
            ham_radius_m=self.meat_radius,
            cheese_UNet=self.Cheese_UNet,
            bologna_UNet=self.Bologna_UNet,
            bread_UNet=self.Bread_UNet,
            use_unet=self.use_UNet_for_check,
            tray_center=TRAY_CENTER_PIXELS,
        )

        # init control variables
        self.assembly_tray_box = None
        self.assembly_bread_xyz_base = None

        # Subscribe to depth image topic (adjust topic name as needed)
        self.depth_subscription = self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.depth_callback, 10
        )

        # Subscribe to rgb image topic
        self.RGB_subscription = self.create_subscription(
            Image, "/camera/camera/color/image_rect_raw", self.rgb_callback, 10
        )

        # Create the service server
        self.service = self.create_service(
            GetDepthAtPoint,
            self.get_name() + "/get_depth_at_point",
            self.handle_get_depth,
        )

        # Create the pickup point service server
        self.pickup_point_service = self.create_service(
            GetXYZFromImage,
            self.get_name() + "/get_pickup_point",
            self.handle_pickup_point,
        )

        # Create the place point service server
        self.place_point_service = self.create_service(
            GetXYZFromImage,
            self.get_name() + "/get_place_point",
            self.handle_place_point,
        )

        # Create sandwich checker service server
        self.sandwich_checker_service = self.create_service(
            CheckIngredientPlace,
            self.get_name() + "/check_ingredient",
            self.handle_sandwich_check,
        )

        self.sandwich_check_reset_service = self.create_service(
            Trigger,
            self.get_name() + "/reset_sandwich_checker",
            self.handle_sandwich_check_reset,
        )

        self.subscription_tf = self.create_subscription(
            TFMessage, "/tf", self.tf_listener_callback_tf, 10
        )
        self.subscription_intrinsics = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_intrinsics_callback,
            10,
        )

        self.save_image_service = self.create_service(
            Trigger,
            self.get_name() + "/save_detection_image",
            self.handle_save_detection_image,
        )

        # Initialize image and transformation variables
        self.transformations = {}
        self.K = np.eye(3)
        self.distortion_coefficients = np.zeros((1, 5))
        self.width = 0
        self.height = 0

        # To publish pickup point to RViz
        self.marker_pub = self.create_publisher(Marker, "visualization_marker", 10)

        # for visualizing pickup point
        self.marker = Marker()
        self.marker.header.frame_id = (
            "panda_link0"  # Frame of reference - base link of the robot arm
        )
        self.marker.id = 0
        self.marker.type = Marker.SPHERE  # Marker type is a sphere
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.1  # Size of the sphere
        self.marker.scale.y = 0.1
        self.marker.scale.z = 0.1
        self.marker.color.a = 1.0  # Full opacity
        self.marker.color.r = 1.0  # Red color
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0

        self.get_logger().info("--- Node initialization completed successfully. ---")

    def tf_listener_callback_tf(self, msg):
        """Handle incoming transform messages."""
        try:
            for transform in msg.transforms:
                if transform.child_frame_id and transform.header.frame_id:
                    self.transformations[
                        (transform.header.frame_id, transform.child_frame_id)
                    ] = transform
        except Exception as e:
            self.get_logger().error(f"Failed to get transform message: {e}.")
            self.get_logger().error(f"Using previous transform")

    def depth_callback(self, msg):
        """Convert ROS Image message to OpenCV format"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            self.depth_queue.append(self.depth_image)
            if len(self.depth_queue) > 5:
                self.depth_queue.popleft()

        except CvBridgeError as e:
            self.get_logger().error(
                f"Failed to convert depth message to OpenCV format: {e}"
            )
            self.depth_image = None

    def rgb_callback(self, msg):
        """Convert ROS Image message to OpenCV format"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except CvBridgeError as e:
            self.get_logger().error(
                f"Failed to convert image message to OpenCV format: {e}"
            )
            self.rgb_image = None

    def camera_intrinsics_callback(self, msg):
        """Handle incoming camera info messages."""
        intrinsic_matrix = msg.k
        self.K = np.array(intrinsic_matrix).reshape((3, 3))
        self.distortion_coefficients = np.array(msg.d)
        self.width = msg.width
        self.height = msg.height

    def quaternion_to_rotation_matrix(self, x, y, z, w):
        """Convert a quaternion into a full three-dimensional rotation matrix."""
        return R.from_quat([x, y, z, w]).as_matrix()

    def get_depth(self, x, y):
        """
        Get the depth value at the specified pixel coordinates (x, y) in the depth image.
        :param x: X coordinate in the image
        :param y: Y coordinate in the image
        :return: Depth value in meters
        """
        # Check if depth image is available
        n = len(self.depth_queue)
        if n == 0:
            self.get_logger().warn("Depth image not available yet!")
            raise ValueError("Depth image not available yet")

        # Ensure coordinates are within bounds of the image dimensions
        if not is_point_within_bounds(self.depth_image, x, y):
            raise ValueError("Coordinates out of bounds")

        # Retrieve depth value at (x, y)
        self.get_logger().info(f"Getting Depth at ({x}, {y})")

        depth_sum = 0
        num_valid_depths = 0
        for img in self.depth_queue:
            res = get_averaged_depth(img, x, y)
            if res != -1:
                depth_sum += res
                num_valid_depths += 1

        if num_valid_depths == 0:
            raise ValueError("No Valid Depth Points")
        depth = depth_sum / num_valid_depths

        self.get_logger().info(f"Depth at ({x}, {y}): {depth} meters")
        return depth

    def handle_sandwich_check(self, request, response):
        try:
            ingredient_name = request.ingredient_name
            ingredient_count = request.ingredient_count
            if self.rgb_image is None:
                raise Exception("No RGB Image found!")
            image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
            self.get_logger().info(f"Checking ingredient: {ingredient_name}")
            self.get_logger().info(f"Ingredient count: {ingredient_count}")

            ingredient_check, check_image = self.sandwich_checker.check_ingredient(
                image=image,
                ingredient_name=ingredient_name,
                ingredient_count=ingredient_count,
            )

            self.get_logger().info(
                f"{ingredient_name} check result: {ingredient_check}"
            )
            response.is_placed = ingredient_check
            response.is_error = False

            # save check_image for debugging
            cv2.imwrite(
                f"/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/{ingredient_name}_check_image.jpg",
                check_image,
            )
        except Exception as e:
            self.get_logger().error(f"Error while checking ingredient: {e}")
            self.get_logger().error(traceback.print_exc())
            ingredient_check = False
            response.is_placed = ingredient_check
            response.is_error = True
        return response

    def handle_sandwich_check_reset(self, request, response):
        try:
            self.sandwich_checker.reset()
            response.success = True
            response.message = "Sandwich checker reset successfully"
        except Exception as e:
            self.get_logger().error(f"Error while resetting sandwich checker: {e}")
            response.success = False
            response.message = "Failed to reset sandwich checker"
        return response

    def handle_get_depth(self, request, response):
        """
        Handle the service request to get depth at a specific point in the image.
        :param request: Service request containing x and y coordinates
        :param response: Service response containing the depth value
        :return: Response with the depth value
        """
        try:
            x = request.x
            y = request.y
            response.depth = self.get_depth(x, y)
        except Exception as e:
            response.depth = float("nan")
            self.get_logger().error(f"Error while getting depth: {e}")
        return response

    def dehomogenize(self, point_h):
        """Dehomogenize by dividing x, y, z by w"""
        # Check if the point is in homogeneous coordinates
        if len(point_h) != 4:
            raise ValueError("Point must be in homogeneous coordinates (4D vector)")
        # Dehomogenize the point
        x, y, z, w = point_h
        if w != 0:
            return (x / w, y / w, z / w)
        else:
            raise ValueError("Homogeneous coordinate w cannot be zero")

    def transform_location_cam2base(self, x, y, depth):
        """
        Modified Version of code from handeye_calibration_ros2

        :param x: X coordinate in the image
        :param y: Y coordinate in the image
        :param depth: Depth value in meters
        :return: Transformed coordinates in base link frame

        """
        # apply intrinsic transform:
        point_img_frame = np.array([x, y, 1])
        point_cam = np.linalg.inv(self.K) @ point_img_frame
        point_cam = depth * point_cam  # Scale the normalized point by depth (Z)
        point_cam = np.concatenate([point_cam, np.array([1])])  # Homogenize the point

        # apply extrinsic transform
        T = np.eye(4)
        link_order = [
            ("panda_link0", "panda_hand"),
            ("panda_hand", "camera_color_optical_frame"),
        ]
        transform_matrices = {}

        for frame_id, child_frame_id in link_order:
            if (frame_id, child_frame_id) in self.transformations:
                transform = copy.deepcopy(
                    self.transformations[(frame_id, child_frame_id)].transform
                )
                translation = [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ]
                rotation = [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ]
                T = np.eye(4)
                T[:3, :3] = self.quaternion_to_rotation_matrix(*rotation)
                T[:3, 3] = translation
                transform_matrices[(frame_id, child_frame_id)] = T

        T_link0_camera = (
            transform_matrices[("panda_link0", "panda_hand")]
            @ transform_matrices[("panda_hand", "camera_color_optical_frame")]
        )

        self.get_logger().info("Got extrinsic transform, applying it to the point")

        point_base_link = T_link0_camera @ point_cam

        point_base_link = self.dehomogenize(point_base_link)

        return point_base_link

    def transform_location_base2cam(self, x, y, z):
        """
        transform from arm base frame to camera optical frame

        """

        # extrinsic transform
        T = np.eye(4)
        T = np.eye(4)
        link_order = [
            ("panda_link0", "panda_hand"),
            ("panda_hand", "camera_color_optical_frame"),
        ]
        transform_matrices = {}

        for frame_id, child_frame_id in link_order:
            if (frame_id, child_frame_id) in self.transformations:
                transform = copy.deepcopy(
                    self.transformations[(frame_id, child_frame_id)].transform
                )
                translation = [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                ]
                rotation = [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ]
                T = np.eye(4)
                T[:3, :3] = self.quaternion_to_rotation_matrix(*rotation)
                T[:3, 3] = translation
                transform_matrices[(frame_id, child_frame_id)] = T

        T_link0_camera = (
            transform_matrices[("panda_link0", "panda_hand")]
            @ transform_matrices[("panda_hand", "camera_color_optical_frame")]
        )

        T_camera_link0 = np.linalg.inv(T_link0_camera)

        point_camera_frame = T_camera_link0 @ np.array([x, y, z, 1])

        # apply intrinsic transform:
        point_camera_frame = point_camera_frame[:3] / point_camera_frame[3]
        point_img_frame = self.K @ point_camera_frame
        point_img_frame = (
            point_img_frame[:2] / point_img_frame[2]
        )  # Homogenize the point
        return point_img_frame

    def handle_pickup_point(self, request, response):
        """
        Handle the service request to get pickup point in the image.
        :param request: Service request containing location ID and timestamp
        :param response: Service response containing the pickup point coordinates
        :return: Response with the pickup point coordinates
        """

        try:
            bin_id = request.location_id
            ingredient_name = request.ingredient_name
            timestamp = request.timestamp  # TODO: use this to sync
            if self.rgb_image is None:
                raise Exception("No RGB image found!")

            image = np.copy(self.rgb_image)
            # self.get_logger().info(f"{image.shape}")
            self.detection_image = np.copy(image)

            self.get_logger().info(f"Got request for pickup point in bin ID: {bin_id}")
            # Bin Cropping Logic
            bin_coords = BIN_COORDS[bin_id - 1]
            x_min, y_min, x_max, y_max = bin_coords
            self.get_logger().info(
                f"Cropping with X coords: [{x_min}, {x_max}] and Y coords: [{y_min}, {y_max}]"
            )
            bin_mask = np.zeros_like(image, dtype=np.uint8)
            bin_mask[y_min:y_max, x_min:x_max] = 1
            masked_image = image * bin_mask
            masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)

            # Ingredient Segmentation Logic
            if ingredient_name == "cheese":
                self.get_logger().info(f"Segmenting cheese...")
                # Save images for debugging
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/cheese_pickup_source_image.jpg",
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/cheese_pickup_masked_image.jpg",
                    masked_image_bgr,
                )
                # Get binary mask
                if self.use_SAM:
                    if self.cheese_segment_generator:
                        mask = self.cheese_segment_generator.get_top_cheese_slice(
                            masked_image_bgr
                        )
                        self.get_logger().info("Got mask from SAM")
                    else:
                        raise Exception("No SAM model created for cheese!")
                elif self.use_UNet:
                    mask, max_contour_mask, max_contour_area = (
                        self.Cheese_UNet.get_top_layer_binary(
                            Im.fromarray(masked_image), CHEESE_TOP_SLICE_PNG
                        )
                    )
                    # If area is too large, crop out bottom 50% and try again
                    if max_contour_area > 1.2 * self.cheese_area_pixels:
                        self.get_logger().info(
                            f"Cheese area is too large: {max_contour_area} > {self.cheese_area_pixels}, cropping out bottom 50% of bin and trying again..."
                        )
                        # Mask out the bottom 50% along the longer axis (x or y)
                        bin_width = x_max - x_min
                        bin_height = y_max - y_min
                        half_bin_mask = np.zeros_like(image, dtype=np.uint8)
                        if bin_height >= bin_width:
                            # keep top half (smaller y values)
                            half_height = bin_height // 2
                            half_bin_mask[y_min : y_min + half_height, x_min:x_max] = 1
                        else:
                            # keep left half (smaller x values)
                            half_width = bin_width // 2
                            half_bin_mask[y_min:y_max, x_min : x_min + half_width] = 1
                        masked_image_half = image * half_bin_mask
                        mask, max_contour_mask, max_contour_area = (
                            self.Cheese_UNet.get_top_layer_binary(
                                Im.fromarray(masked_image_half), CHEESE_TOP_SLICE_PNG
                            )
                        )
                        if max_contour_area > 1.2 * self.cheese_area_pixels:
                            self.get_logger().info(
                                f"Cheese area is still too large: {max_contour_area} > {self.cheese_area_pixels}, skipping this image..."
                            )
                            raise Exception(
                                f"Cheese area after 50% crop is still too large: {max_contour_area} > {self.cheese_area_pixels}"
                            )
                    cv2.imwrite(
                        "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/cheese_pickup_unet_mask.jpg",
                        cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR),
                    )
                    self.get_logger().info(
                        f"Got mask from UNet, Cheese Area: {max_contour_area}"
                    )
                    mask = max_contour_mask  # choose the largest contour
                else:
                    self.get_logger().info("Neither SAM nor UNet were chosen")
                    raise Exception("No segmentation method chosen")

                mask_truth_value = np.max(np.array(mask))

                if mask_truth_value == 0:
                    raise Exception(
                        "UNet did not detect any cheese in bin. Please check the image"
                    )

                # Average the true pixels in binary mask to get center X, Y
                y_coords, x_coords = np.where(mask == mask_truth_value)
                cam_x = int(np.mean(x_coords))
                cam_y = int(np.mean(y_coords))

                # self.get_logger().info(f"Cheese pickup point {cam_x}, {cam_y}")
                cv2.circle(
                    masked_image_bgr,
                    (int(cam_x), int(cam_y)),
                    10,
                    color=(255, 0, 0),
                    thickness=-1,
                )

                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/cheese_pickup_point.jpg",
                    masked_image_bgr,
                )

            elif ingredient_name == "meat":
                # Meat
                self.get_logger().info(f"Segmenting Meat...")
                # Save images for debugging
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/meat_pickup_source_image.jpg",
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/meat_pickup_masked_image.jpg",
                    masked_image_bgr,
                )
                if self.use_SAM:
                    # Get X, Y using SAM
                    if self.meat_segment_generator:
                        cam_x, cam_y = (
                            self.meat_segment_generator.get_top_meat_slice_xy(
                                masked_image_bgr
                            )
                        )
                    else:
                        raise Exception("No SAM Model created for meat!")
                    self.get_logger().info("Got mask from SAM")
                    mask = None

                elif self.use_UNet:
                    # Use the already cropped image for UNet segmentation
                    mask, max_contour_mask, max_contour_area = (
                        self.Bologna_UNet.get_top_layer_binary(
                            Im.fromarray(masked_image), MEAT_TOP_SLICE_PNG
                        )
                    )

                    if max_contour_area > 1.2 * self.meat_area_pixels:
                        self.get_logger().info(
                            f"Meat area is too large: {max_contour_area} > {self.meat_area_pixels}, cropping out bottom 50% of bin and trying again..."
                        )
                        # Mask out the bottom 50% along the longer axis (x or y)
                        bin_width = x_max - x_min
                        bin_height = y_max - y_min
                        half_bin_mask = np.zeros_like(image, dtype=np.uint8)
                        if bin_height >= bin_width:
                            half_height = bin_height // 2
                            half_bin_mask[y_min : y_min + half_height, x_min:x_max] = 1
                        else:
                            half_width = bin_width // 2
                            half_bin_mask[y_min:y_max, x_min : x_min + half_width] = 1
                        masked_image_half = image * half_bin_mask
                        mask, max_contour_mask, max_contour_area = (
                            self.Bologna_UNet.get_top_layer_binary(
                                Im.fromarray(masked_image_half), MEAT_TOP_SLICE_PNG
                            )
                        )

                        if max_contour_area > 1.5 * self.meat_area_pixels:
                            self.get_logger().info(
                                f"Meat area is still too large: {max_contour_area} > {self.meat_area_pixels}, skipping this image..."
                            )
                            raise Exception(
                                f"Meat area after 75% crop is still too large: {max_contour_area} > {self.meat_area_pixels}"
                            )

                    cv2.imwrite(
                        "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/meat_pickup_unet_mask.jpg",
                        cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR),
                    )

                    self.get_logger().info(
                        f"Got mask from UNet, Meat Area: {max_contour_area}"
                    )
                    mask = max_contour_mask
                    mask_truth_value = np.max(mask)

                    if mask_truth_value == 0:
                        raise Exception("Bologna mask is empty")

                    self.get_logger().info(f"Max value in mask {mask_truth_value}")
                    # Average the true pixels in binary mask to get center X, Y
                    y_coords, x_coords = np.where(mask == mask_truth_value)
                    cam_x = int(np.mean(x_coords))
                    cam_y = int(np.mean(y_coords))
                else:
                    self.get_logger().info("Neither SAM nor UNet were chosen")
                    raise Exception("No segmentation method chosen")

                self.get_logger().info(f"Mid point {cam_x}, {cam_y}")

                cv2.circle(
                    masked_image_bgr,
                    (int(cam_x), int(cam_y)),
                    10,
                    color=(255, 0, 0),
                    thickness=-1,
                )
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/meat_pickup_point.jpg",
                    masked_image_bgr,
                )

            elif ingredient_name == "bread":
                # Bread
                print("Segmenting Bread...")
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bread_pickup_source_image.jpg",
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bread_pickup_masked_image.jpg",
                    masked_image_bgr,
                )
                if self.use_UNet:
                    # Use the UNet model for bread segmentation
                    mask, max_contour_mask, max_contour_area = (
                        self.Bread_UNet.get_top_layer_binary(
                            Im.fromarray(masked_image), BREAD_TOP_SLICE_PNG
                        )
                    )
                    # If area is too large, crop out bottom 50% and try again
                    if max_contour_area > 1.2 * (
                        self.bread_width * self.bread_height * (1 / self.pixels_to_m**2)
                    ):
                        self.get_logger().info(
                            f"Bread area is too large, cropping out bottom 50% of bin and trying again..."
                        )
                        # Mask out the bottom 50% along the longer axis (x or y)
                        bin_width = x_max - x_min
                        bin_height = y_max - y_min
                        half_bin_mask = np.zeros_like(image, dtype=np.uint8)
                        if bin_height >= bin_width:
                            half_height = bin_height // 2
                            half_bin_mask[y_min : y_min + half_height, x_min:x_max] = 1
                        else:
                            half_width = bin_width // 2
                            half_bin_mask[y_min:y_max, x_min : x_min + half_width] = 1
                        masked_image_half = image * half_bin_mask
                        mask, max_contour_mask, max_contour_area = (
                            self.Bread_UNet.get_top_layer_binary(
                                Im.fromarray(masked_image_half), BREAD_TOP_SLICE_PNG
                            )
                        )

                    cv2.imwrite(
                        "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bread_pickup_unet_mask.jpg",
                        cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR),
                    )
                    self.get_logger().info(f"Got mask from UNet for bread")
                    mask = max_contour_mask  # choose the largest contour

                    # Average the true pixels in binary mask to get center X, Y
                    mask_truth_value = np.max(mask)
                    if mask_truth_value == 0:
                        raise Exception(
                            "UNet did not detect any bread in bin. Please check the image"
                        )

                    y_coords, x_coords = np.where(mask == mask_truth_value)
                    cam_x = int(np.mean(x_coords))
                    cam_y = int(np.mean(y_coords))
                else:
                    cam_x, cam_y = self.bread_segment_generator.get_bread_pickup_point(
                        masked_image_bgr
                    )
                    mask = None

                cv2.circle(
                    masked_image_bgr,
                    (cam_x, cam_y),
                    10,
                    color=(255, 0, 0),
                    thickness=-1,
                )
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bread_pickup_point.jpg",
                    masked_image_bgr,
                )

            else:
                raise Exception("Incorrect Bin ID")

            if self.depth_image is None:
                raise Exception("No depth image found!")

            if (
                cam_x < 0
                or cam_x >= self.depth_image.shape[1]
                or cam_y < 0
                or cam_y >= self.depth_image.shape[0]
            ):
                raise ValueError("Coordinates out of bounds")

            # Retrieve depth value at (x, y)
            cam_z = self.get_depth(cam_x, cam_y)

            # sanity check
            if cam_z == 0:
                raise Exception("Invalid Z")
            self.get_logger().info(
                f"Got pickup point in optical frame: {cam_x}, {cam_y} and depth: {cam_z:.2f} in bin {bin_id} at {timestamp}"
            )

            self.get_logger().info("Transforming pickup point from camera optical frame to base frame...")

            response_transformed = self.transform_location_cam2base(cam_x, cam_y, cam_z)
            self.get_logger().info("Got transform, applying it to point...")
            response.x = response_transformed[0]
            response.y = response_transformed[1]
            response.z = (
                response_transformed[2] - PICKUP_Z_OFFSET
            ) 

            self.get_logger().info(
                f"Transformed coords: X: {response.x}, Y: {response.y}, Z:{response.z}"
            )
            is_reachable = is_valid_pickup_point(
                response.x, response.y, bin_id, 3, self.get_logger()
            )

            if not is_reachable:
                raise Exception("Pickup points not within bin")
            self.get_logger().info("Pickup point is valid!")

        except Exception as e:
            self.get_logger().error(f"Error while calculating pickup point: {e}")
            self.get_logger().error(traceback.print_exc())
            response.x = -1.0
            response.y = -1.0
            response.z = float("nan")
        finally:
            return response

    def handle_save_detection_image(self, request, response):
        """
        Handle the service request to save the detection image.
        :param request: Service request
        :param response: Service response indicating success or failure
        :return: Response with success or failure message
        """

        try:
            if self.detection_image is None:
                self.get_logger().error("No detection image available to save.")
                response.success = False
                response.message = "No detection image available."
                return response

            # Create the directory if it doesn't exist
            if os.path.exists(FAILURE_IMAGES_PATH) is False:
                os.makedirs(FAILURE_IMAGES_PATH)
                self.get_logger().info(f"Created directory: {FAILURE_IMAGES_PATH}")

            # Save the detection image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                FAILURE_IMAGES_PATH,
                f"failure_image_{self.detection_image_count}_{timestamp}.jpg",
            )
            cv2.imwrite(filename, cv2.cvtColor(self.detection_image, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f"Saved detection image to {filename}")
            self.detection_image_count += 1

            response.success = True
            response.message = f"Detection image saved to {filename} "
            self.detection_image = None
            self.get_logger().info("Detection image reset to None")
        except Exception as e:
            self.get_logger().error(f"Failed to save detection image: {e}")
            response.success = False
            response.message = str(e)

        return response

    def handle_place_point(self, request, response):
        """
        Handle the service request to get place point in the image.
        :param request: Service request containing location ID and timestamp
        :param response: Service response containing the place point coordinates
        :return: Response with the place point coordinates
        """

        try:
            location_id = request.location_id
            timestamp = request.timestamp  # use this to sync

            if self.rgb_image is None:
                raise Exception("No RGB image available!")

            image = np.copy(self.rgb_image)
            self.get_logger().info(
                f"Handle Place Point Called with location ID: {location_id}"
            )
            self.get_logger().info(
                f"Tray ID: {ASSEMBLY_TRAY_ID}, Bread ID: {ASSEMBLY_BREAD_ID}"
            )

            if location_id == ASSEMBLY_TRAY_ID:

                if self.tray_segment_generator: # this will be None if SAM model not loaded, i.e., if use_UNet is True
                    self.get_logger().info(f"Segmenting tray")

                    mask = self.tray_segment_generator.get_tray_mask(image)

                    self.assembly_tray_box = calc_bbox_from_mask(mask * 255)

                    # Average the positions of white points to get center
                    y_coords, x_coords = np.where(mask == 1)
                    cam_x = int(np.mean(x_coords))
                    cam_y = int(np.mean(y_coords))

                    # Save images for debugging
                    cv2.circle(
                        np.array(image), (cam_x, cam_y), 10, color=(255, 0, 0), thickness=-1
                    )
                    cv2.imwrite(
                        "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/tray_mask.jpg",
                        mask * 255,
                    )
                    cv2.imwrite(
                        "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/tray_img.jpg",
                        np.array(image),
                    )
                else:
                    self.get_logger().info(f"Using predefined tray center")
                    cam_x = int(TRAY_CENTER_PIXELS[0])
                    cam_y = int(TRAY_CENTER_PIXELS[1])

            elif location_id == ASSEMBLY_BREAD_ID:
                # Just get the bottom bread center from sandwich checker
                self.get_logger().info(f"Getting bottom bread center for placing")
                bottom_bread_center = self.sandwich_checker.bread_centers[0]
                cam_x = int(bottom_bread_center[0])
                cam_y = int(bottom_bread_center[1])

            else:
                raise Exception(f"Got unknown place location ID:{location_id})")

            # get depth
            cam_z = self.get_depth(cam_x, cam_y)

            # transform coordinates
            response_transformed = self.transform_location_cam2base(cam_x, cam_y, cam_z)

            self.get_logger().info("got transform, applying it to point...")

            response.x = response_transformed[0]
            response.y = response_transformed[1]
            response.z = response_transformed[2]

            self.get_logger().info(
                f"Transformed coords: X: {response.x}, Y: {response.y}, Z:{response.z}"
            )

            if not (response.z > 0.28 and response.z < 0.35):
                raise Exception("Invalid Depth")

            # publish point to topic
            self.marker.pose.position = Point(x=response.x, y=response.y, z=response.z)

            # Get the current time and set it in the header
            self.marker.header.stamp = self.get_clock().now().to_msg()

            # Publish the marker
            self.marker_pub.publish(self.marker)
            self.get_logger().info("Published point to RViz")

        except Exception as e:
            self.get_logger().error(f"Error while calculating place point: {e}")
            self.get_logger().error(traceback.print_exc())
            response.x = -1.0
            response.y = -1.0
            response.z = float("nan")

        return response


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
