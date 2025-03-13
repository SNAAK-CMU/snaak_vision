import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='snaak_vision',
            executable='vision_node.py',
            name='snaak_vision')
    ])