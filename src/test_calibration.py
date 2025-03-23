#!/usr/bin/env python3
# author: Oliver
# Script to validate camera calibration

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from autolab_core import RigidTransform
from snaak_vision.srv import GetXYZFromImage
import numpy as np
from sensor_msgs.msg import CameraInfo
import cv2
import copy
from frankapy import FrankaArm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys
from scipy.spatial.transform import Rotation as R


class TestCalibration(Node):
    '''
    ROS Node with service to test calibration of camera. Detects bolt on the left center of the 
    assembly area (to the left of the scale next to bin 2).
    Returns the difference between the camera localized point and point measured using the arm.
    '''
    def __init__(self):
        super().__init__('test_calibration')

        self._test_calibration_service = self.create_service(
            GetXYZFromImage,
            "snaak_vision/test_calibration",
            self.test_calibration_callback,
        )
        self.subscription_tf = self.create_subscription(
            TFMessage, "/tf", self.tf_listener_callback_tf, 10
        )
        self.transformations = {}
        self.subscription_intrinsics = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_intrinsics_callback,
            10,
        ) 
        # Subscribe to depth image topic (adjust topic name as needed)
        self.depth_subscription = self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.depth_callback, 10
        )

        # Subscribe to rgb image topic
        self.RGB_subscription = self.create_subscription(
            Image, "/camera/camera/color/image_rect_raw", self.rgb_callback, 10
        )
        self.bridge = CvBridge()
        self.fa = FrankaArm(init_rclpy=False)
        #self.fa.reset_joints()
        self.K = np.eye(3)
        self.distortion_coefficients = np.zeros((1, 5))
        self.depth_image = None
        self.rgb_image = None
        self.bolt_translation = np.array([0.33965218, -0.18945675, 0.26170138]) # Gathered with print pose method in manipulation test scripts


    def test_calibration_callback(self, request, response):
        default_rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        pose = RigidTransform(from_frame='franka_tool', to_frame='world')
        pose.translation = self.fa.get_pose().translation #np.array([0.32584863, -0.11785498, 0.44802292]) #TODO Define Pose
        pose.translation = np.array([0.35584863, -0.15785498, 0.44802292])
        pose.rotation = default_rotation
        self.get_logger().info(f"Moving to photo pose")
        self.fa.goto_pose(pose)
        
        image = self.rgb_image
        image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)

        height, width = image.shape[:2]
        crop_height = 50
        crop_width = 70
        crop_y_start = (height - crop_height) // 2
        crop_y_end = crop_y_start + crop_height
        crop_x_start = (width - crop_width) // 2
        crop_x_end = crop_x_start + crop_width
        crop_mask = np.zeros_like(image)
        crop_mask[crop_y_start:crop_y_end, crop_x_start:crop_x_end] = 255

        # With HSV, create a mask that only shows the bolt which is black
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        res = cv2.bitwise_and(image, image, mask=mask)
        res = cv2.bitwise_and(res, crop_mask)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,10,255,0)
        edges = cv2.Canny(thresh, 50, 150) 

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 30
        bolt_center_pixels = None

        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
                M = cv2.moments(cnt)

                cX = int(M["m10"] / M["m00"]) 
                cY = int(M["m01"] / M["m00"])
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 3) 
                bolt_center_pixels = (cX, cY)
                cv2.circle(image, (cX, cY), 5, color=(255, 0, 0), thickness=-1)

        if bolt_center_pixels is None:
            response.x = -1.0
            response.y = -1.0
            response.z = float("nan") 
            self.get_logger().info(f"No bolt found")
            return response
        else:
            self.get_logger().info(f"Bolt center pixels: {bolt_center_pixels}")
            cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bolt_img.jpg",
                        image,
                    )
            cv2.imwrite(
                    "/home/snaak/Documents/manipulation_ws/src/snaak_vision/src/segmentation/bolt_mask.jpg",
                        edges,
                    )
            cZ = float(self.depth_image[int(cY / 2.0), int(cX / 2.0)]) / 1000.0

            response_transformed = self.transform_location(cX, cY, cZ)

            self.get_logger().info("got transform, applying it to point...")

            response.x = response_transformed[0] - self.bolt_translation[0]
            response.y = response_transformed[1] - self.bolt_translation[1]
            response.z = response_transformed[2] - self.bolt_translation[2]
            return response
    
    def depth_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="passthrough"
        )

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def tf_listener_callback_tf(self, msg):
        """Handle incoming transform messages."""
        for transform in msg.transforms:
            if transform.child_frame_id and transform.header.frame_id:
                self.transformations[
                    (transform.header.frame_id, transform.child_frame_id)
                ] = transform
    def quaternion_to_rotation_matrix(self, x, y, z, w):
        """Convert a quaternion into a full three-dimensional rotation matrix."""
        return R.from_quat([x, y, z, w]).as_matrix()
    
    def camera_intrinsics_callback(self, msg):
        intrinsic_matrix = msg.k
        self.K = np.array(intrinsic_matrix).reshape((3, 3))
        self.distortion_coefficients = np.array(msg.d)
        self.width = msg.width
        self.height = msg.height
        

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
                transform = copy.deepcopy(self.transformations[(frame_id, child_frame_id)].transform)
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
    
def main(args=None):
    # TODO add proper shutdown with FrankaPy
    rclpy.init(args=args)
    manipulation_action_server = TestCalibration()
    try:
        rclpy.spin(manipulation_action_server)
    except Exception as e:
        manipulation_action_server.get_logger().error(f'Error occurred: {e}')
    except KeyboardInterrupt:
        manipulation_action_server.get_logger().info('Keyboard interrupt received, shutting down...')
    finally:
        manipulation_action_server.fa.stop_robot_immediately()
        manipulation_action_server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping Gracefully...")
        sys.exit(0)