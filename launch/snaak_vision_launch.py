import launch
import launch_ros.actions
import os

def generate_launch_description():

    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(
            'virtualenv_python_executable',
            default_value='/home/snaak/Documents/manipulation_ws/src/snaak_vision/.venv/bin/python3',
            description='Full path to the virtualenv python executable'
        ),
        launch_ros.actions.Node(
            package='snaak_vision',
            executable='vision_node.py',
            name='snaak_vision',
            prefix=launch.substitutions.LaunchConfiguration('virtualenv_python_executable')
        ),
        launch_ros.actions.Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            name="realsense2_camera_node"
        ),
        
        launch_ros.actions.Node(
            package='handeye_realsense',
            executable='eye2hand',
            name='handeye_publisher',
        )
    ])