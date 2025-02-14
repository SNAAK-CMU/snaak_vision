#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from coordinates.srv import GetDepthAtPoint

class DepthService(Node):
    def __init__(self):
        super().__init__('depth_service')
        self.bridge = CvBridge()
        self.depth_image = None

        # Subscribe to depth image topic (adjust topic name as needed)
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10)

        # Create the service server
        self.service = self.create_service(GetDepthAtPoint, 'get_depth_at_point', self.handle_get_depth)

    def depth_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def handle_get_depth(self, request, response):
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

def main(args=None):
    rclpy.init(args=args)
    node = DepthService()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
