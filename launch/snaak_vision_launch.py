import launch
import launch_ros.actions
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Include the RealSense launch file
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ]),
        launch_arguments={
            'camera_name': 'camera',
            'device_type': 'd405',
            'enable_color': 'true',
            'enable_depth': 'true'
        }.items()
    )

    return launch.LaunchDescription([
        realsense_launch,
        launch_ros.actions.Node(
            package='handeye_realsense',
            executable='eye2hand',
            name='handeye_publisher',
        ),
        launch_ros.actions.Node(
            package='snaak_vision',
            executable='vision_node.py',
            name='snaak_vision',
        ),
    ])
