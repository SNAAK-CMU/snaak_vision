#!/home/snaak/Documents/manipulation_ws/src/snaak_vision/.venv/bin/python3

# Change above line if chaning .venv location
# author: Oliver
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from snaak_vision.srv import GetDepthAtPoint
from snaak_vision.srv import GetXYZFromImage
from snaak_vision.srv import CheckIngredientPlace
from std_srvs.srv import Trigger
import traceback
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from PIL import Image as Im


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
from segmentation.plate_bread_segment_generator import PlateBreadSegementGenerator
from segmentation.meat_segment_generator import MeatSegmentGenerator
from segmentation.segment_utils import (
    calc_bbox_from_mask,
    is_valid_pickup_point,
    is_point_within_bounds,
    get_averaged_depth,
)

from segmentation.UNet.ingredients_UNet import Ingredients_UNet

from sandwich_checker import SandwichChecker

############### Parameters #################

# Make these config
HAM_BIN_ID = 1
CHEESE_BIN_ID = 2
BREAD_BIN_ID = 3
ASSEMBLY_TRAY_ID = 4
ASSEMBLY_BREAD_ID = 5

FOV_WIDTH = 0.775  # metres
FOV_HEIGHT = 0.435  # metres
SW_CHECKER_THRESHOLD = 3  # cm

IMG_WIDTH = 848
IMG_HEIGHT = 480

############################################


class VisionNode(Node):
    def __init__(self):
        super().__init__("snaak_vision")
        self.bridge = CvBridge()
        self.depth_image = None
        self.rgb_image = None

        # camera FOV over assembly area
        self.fov_w = FOV_WIDTH  # metres
        self.fov_h = FOV_HEIGHT  # metres
        self.threshold_in_cm = SW_CHECKER_THRESHOLD

        # image size
        self.image_width = IMG_WIDTH
        self.image_height = IMG_HEIGHT

        # init segmentation objects
        self.use_SAM = False
        self.cheese_segment_generator = CheeseSegmentGenerator()
        self.tray_segment_generator = TraySegmentGenerator()
        self.bread_segment_generator = BreadSegmentGenerator()
        self.plate_bread_segment_generator = PlateBreadSegementGenerator()
        self.meat_segment_generator = MeatSegmentGenerator()

        # init UNet
        self.use_UNet = True
        self.Cheese_UNet = Ingredients_UNet(
            count=False,
            classes=["background", "top_cheese", "other_cheese"],
            model_path="logs/cheese/best_epoch_weights.pth",  # choose weights
            mix_type=1,
        )

        # init sandwich checker
        self.sandwich_checker = SandwichChecker(
            fov_height=self.fov_h,
            fov_width=self.fov_w,
            threshold_in_cm=self.threshold_in_cm,
            image_width=self.image_width,
            image_height=self.image_height,
            node_logger=self.get_logger(),
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
        )  # TODO fix this

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
        # Convert ROS Image message to OpenCV format
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except CvBridgeError as e:
            self.get_logger().error(
                f"Failed to convert depth message to OpenCV format: {e}"
            )
            self.rgb_image = None

    def rgb_callback(self, msg):
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
        intrinsic_matrix = msg.k
        self.K = np.array(intrinsic_matrix).reshape((3, 3))
        self.distortion_coefficients = np.array(msg.d)
        self.width = msg.width
        self.height = msg.height

    def quaternion_to_rotation_matrix(self, x, y, z, w):
        """Convert a quaternion into a full three-dimensional rotation matrix."""
        return R.from_quat([x, y, z, w]).as_matrix()

    def get_depth(self, x, y):
        if self.depth_image is None:
            self.get_logger().warn("Depth image not available yet!")
            return float("nan")

        # Ensure coordinates are within bounds of the image dimensions
        if not is_point_within_bounds(self.depth_image, x, y):
            raise ValueError("Coordinates out of bounds")

        # Retrieve depth value at (x, y)
        self.get_logger().info(f"Getting Depth at ({x}, {y})")

        depth = get_averaged_depth(self.depth_image, x, y)
        self.get_logger().info(f"Depth at ({x}, {y}): {depth} meters")
        return depth

    def handle_sandwich_check(self, request, response):
        try:
            ingredient_name = request.ingredient_name
            image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
            self.get_logger().info(f"Checking ingredient: {ingredient_name}")
            ingredient_check, check_image = self.sandwich_checker.check_ingredient(
                image=image, ingredient_name=ingredient_name
            )
            self.get_logger().info(
                f"{ingredient_name} check result: {ingredient_check}"
            )
            response.is_placed = ingredient_check
            # TODO: do something with check_image
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
        try:
            x = request.x
            y = request.y
            response.depth = self.get_depth(x, y)
        except Exception as e:
            response.depth = float("nan")
            self.get_logger().error(f"Error while getting depth: {e}")
        return response

    def dehomogenize(self, point_h):
        # Dehomogenize by dividing x, y, z by w
        x, y, z, w = point_h
        if w != 0:
            return (x / w, y / w, z / w)
        else:
            raise ValueError("Homogeneous coordinate w cannot be zero")

    def transform_location(self, x, y, depth):
        """
        Modified Version of code from handeye_calibration_ros2
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

        point_base_link = T_link0_camera @ point_cam  # why not inverse??????

        point_base_link = self.dehomogenize(point_base_link)

        return point_base_link

    def handle_pickup_point(self, request, response):
        try:
            bin_id = request.location_id
            timestamp = request.timestamp  # use this to sync
            image = self.rgb_image

            self.get_logger().info(f"{image.shape}")
            image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)

            self.get_logger().info(f"Bin ID: {bin_id}")
            # self.get_logger().info(f"Cheese Bin ID: {CHEESE_BIN_ID}")
            if bin_id == CHEESE_BIN_ID:
                # Cheese

                self.get_logger().info(f"Segmenting cheese")

                # Get binary mask
                if self.use_SAM:
                    self.get_logger().info("Saved Input Image")
                    mask = self.cheese_segment_generator.get_top_cheese_slice(image)
                    self.get_logger().info("Got mask from SAM")
                elif self.use_UNet:
                    # PIL stores images as RGB, OpenCV stores as BGR
                    # TODO: change UNet to work with cv2 images
                    unet_input_image = self.rgb_image
                    mask, max_contour_mask = np.array(
                        self.Cheese_UNet.get_top_layer_binary(
                            Im.fromarray(unet_input_image), [250, 250, 55]
                        )
                    )
                    # TODO: handle case where there is a top slice outside the bin
                    self.get_logger().info("Got mask from UNet")
                else:
                    self.get_logger().info("Neither SAM nor UNet were chosen")
                    raise Exception("No segmentation method chosen")

                # Save images for debugging
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/cheese_input_image.jpg",
                    image,
                )
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/cheese_mask.jpg",
                    mask,
                )
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/max_cheese_mask.jpg",
                    max_contour_mask,
                )

                mask_truth_value = np.max(
                    mask
                )  # TODO: this would cause the center of mask to be center of image if entire image is true / false add check here to return invalid pickup point if max value of mask is 0
                self.get_logger().info(f"Max value in mask {mask_truth_value}")
                # Average the true pixels in binary mask to get center X, Y
                y_coords, x_coords = np.where(mask == mask_truth_value)
                cam_x = int(np.mean(x_coords))
                cam_y = int(np.mean(y_coords))

                self.get_logger().info(f"Mid point {cam_x}, {cam_y}")
                cv2.circle(image, (cam_x, cam_y), 10, color=(255, 0, 0), thickness=-1)

                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/cheese_pickup_point.jpg",
                    image,
                )

            elif bin_id == HAM_BIN_ID:
                # Ham

                self.get_logger().info(f"Segmenting ham")

                # Get X, Y using SAM
                cam_x, cam_y = self.meat_segment_generator.get_top_meat_slice_xy(image)

                self.get_logger().info(f"Mid point {cam_x}, {cam_y}")
                cv2.circle(image, (cam_x, cam_y), 10, color=(255, 0, 0), thickness=-1)

                # Save images for debugging
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/meat_img.jpg",
                    image,
                )

            elif bin_id == BREAD_BIN_ID:
                # Bread

                self.get_logger().info(f"Segmenting bread")

                cam_x, cam_y, lower_y = (
                    self.bread_segment_generator.get_bread_pickup_point(image)
                )

                # Save images for debugging
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bread_pickup_img.jpg",
                    image,
                )

            else:
                raise "Incorrect Bin ID"

            if (
                cam_x < 0
                or cam_x >= self.depth_image.shape[1]
                or cam_y < 0
                or cam_y >= self.depth_image.shape[0]
            ):
                raise ValueError("Coordinates out of bounds")

            # Retrieve depth value at (x, y)
            cam_z = self.get_depth(cam_x, cam_y)

            # These adjustments need to be removed and the detection should be adjusted to account for the end effector size
            # cam_z += 0.05  # now the end effector just touches the cheese, we need it to go a little lower to actually make a seal
            # cam_x += 0.02  # the x is a little off - either the end effector is incorrectly described or the detection needs to be adjusted

            if cam_z == 0:
                raise Exception("Invalid Z")
            # self.get_logger().info(f"Got pickup point {response.x}, {response.y} and depth {response.depth:.2f} in bin {bin_ID} at {timestamp}")

            self.get_logger().info("transforming coordinates...")
            response_transformed = self.transform_location(cam_x, cam_y, cam_z)

            self.get_logger().info("got transform, applying it to point...")
            z_offset = 0.02 if bin_id == BREAD_BIN_ID else 0.01
            response.x = response_transformed[0]
            response.y = response_transformed[1]
            response.z = (
                response_transformed[2] - z_offset
            )  # now the end effector just touches the cheese, we need it to go a little lower to actually make a seal

            self.get_logger().info(
                f"Transformed coords: X: {response.x}, Y: {response.y}, Z:{response.z}"
            )
            is_reachable = is_valid_pickup_point(
                response.x, response.y, bin_id, BREAD_BIN_ID
            )
            if not is_reachable:
                raise Exception("Pickup point not within bin")

        except Exception as e:
            self.get_logger().error(f"Error while calculating pickup point: {e}")
            self.get_logger().error(traceback.print_exc())
            response.x = -1.0
            response.y = -1.0
            response.z = float("nan")

        return response

    def handle_place_point(self, request, response):
        try:
            location_id = request.location_id
            timestamp = request.timestamp  # use this to sync
            depth_image = self.depth_image

            image = self.rgb_image
            self.get_logger().info(
                f"Handle Place Point Called with location ID: {location_id}"
            )
            self.get_logger().info(
                f"Tray ID: {ASSEMBLY_TRAY_ID}, Bread ID: {ASSEMBLY_BREAD_ID}"
            )

            if location_id == ASSEMBLY_TRAY_ID:
                self.get_logger().info(f"Segmenting tray")

                mask = self.tray_segment_generator.get_tray_mask(image)

                self.assembly_tray_box = calc_bbox_from_mask(mask * 255)

                # Average the positions of white points to get center
                y_coords, x_coords = np.where(mask == 1)
                cam_x = int(np.mean(x_coords))
                cam_y = int(np.mean(y_coords))

                # Save images for debugging
                cv2.circle(image, (cam_x, cam_y), 10, color=(255, 0, 0), thickness=-1)
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/tray_mask.jpg",
                    mask * 255,
                )
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/tray_img.jpg",
                    image,
                )

            if location_id == ASSEMBLY_BREAD_ID:
                image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
                self.get_logger().info(f"Segmenting Bread")
                mask = self.bread_segment_generator.get_bread_placement_mask(image)
                self.get_logger().info(f"Bread segmentation completed")

                # Average the positions of white points to get center
                y_coords, x_coords = np.where(mask == 255)
                cam_x = int(np.mean(x_coords))
                cam_y = int(np.mean(y_coords))

                # Save images for debugging
                cv2.circle(image, (cam_x, cam_y), 10, color=(255, 0, 0), thickness=-1)
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bread_place_mask.jpg",
                    mask,
                )
                cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bread__place_img.jpg",
                    image,
                )

            # get depth
            cam_z = self.get_depth(cam_x, cam_y)

            # transform coordinates
            response_transformed = self.transform_location(cam_x, cam_y, cam_z)

            self.get_logger().info("got transform, applying it to point...")

            response.x = response_transformed[0]
            response.y = response_transformed[1]
            response.z = response_transformed[2]

            self.get_logger().info(
                f"Transformed coords: X: {response.x}, Y: {response.y}, Z:{response.z}"
            )

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
