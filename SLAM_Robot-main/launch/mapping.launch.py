from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_dir = os.path.join(
        get_package_share_directory('articubot_one'), 'config'
    )

    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='mapping_node',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[
            '-configuration_directory', config_dir,
            '-configuration_basename', 'turtlebot3_lds_2d.lua'
        ],
        remappings=[
            ('scan', '/scan'),
            # Odom remapping not needed if use_odometry = false
        ]
    )

    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='occupancy_node',
        output='screen',
        parameters=[
            {"use_sim_time": True},
            {"resolution": 0.05},
            {"publish_period_sec": 1.0}
        ]
    )

    pose_publisher_node = Node(
        package='articubot_one',
        executable='cartographer_pose_publisher.py',
        name='cartographer_pose_pub',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'parent_frame': 'map'}  # <- changed from 'odom'
        ]
    )

    slam_eval_node = Node(
        package='articubot_one',
        executable='slam_evaluator.py',
        name='slam_evaluator',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'duration_sec': 60.0},
            {'world_frame': 'map'},       # <- changed from 'odom'
            {'base_frame': 'base_link'},
            {'est_topic': '/cartographer_pose'},
            {'output_dir': os.path.expanduser('~/slam_eval')}
        ]
    )

    return LaunchDescription([
        TimerAction(period=5.0, actions=[cartographer_node]),
        TimerAction(period=6.0, actions=[occupancy_grid_node]),
        TimerAction(period=7.0, actions=[pose_publisher_node]),
        TimerAction(period=8.0, actions=[slam_eval_node])
    ])
