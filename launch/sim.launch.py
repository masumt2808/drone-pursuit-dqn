from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='drone_pursuit',
            executable='evader_node',
            name='evader_node',
            parameters=[{'speed': 0.5}],
            output='screen',
        ),
    ])
