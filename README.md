# ROS2 Node for Vision Subsystem

The ROS2 node for the vision subsystem of SNAAK.

## Setup and Running

1. Setup a Python3 virtual environment and install requirements

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

2. Remember to activate the environment before running the code

    ```bash
    source .venv/bin/activate
    ros2 run snaak_vision vision_node.py
    ```

## Topic Subscriptions

1. /camera/camera/color/image_rect_raw
    - The raw RGB image topic
    - Type: sensor_msgs/Image
2. /camera/camera/depth/image_rect_raw
    - The raw depth map topic
    - Type: sensor_msgs/Image

## Services Provided

1. /snaak_vision/get_depth_at_point
    - Returns depth value from depth map at given XY point
    - Type: snaak_vision/GetDepthAtPoint
        - Request: int32 x; int32 y
        - Response: float32 depth

2. /snaak_vision/get_pickup_point
    - Returns pickup point for the ingredient in the bin defined by location ID
    - Type: snaak_vision/GetXYZFromImage
        - Request: int32 location_id; float32 timestamp
        - Current Location ID definitions:
            - HAM_BIN_ID = 1
            - CHEESE_BIN_ID = 2
            - BREAD_BIN_ID = 3
        - Response: float32 x; float32 y; float32 z
3. /snaak_vision/get_place_point
    - Returns place point, which is currently defined as the center of the bread on the tray
    - Type: snaak_vision/GetXYZFromImage
        - Request: int32 location_id; float32 timestamp
        - Current Location ID definitions:
            - ASSEMBLY_TRAY_ID = 4
            - ASSEMBLY_BREAD_ID = 5
        - Response: float32 x; float32 y; float32 z
