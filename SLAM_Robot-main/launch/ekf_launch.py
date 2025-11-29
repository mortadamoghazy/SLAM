from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    ekf_config = os.path.join(
        get_package_share_directory('articubot_one'),
        'config', 'ekf.yaml'
    )

    return LaunchDescription([
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=[ekf_config],
            remappings=[
                ('/odometry/filtered', '/odom_fused')
            ]
        )
    ])
