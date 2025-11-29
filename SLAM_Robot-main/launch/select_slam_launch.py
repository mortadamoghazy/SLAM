import os
import sys


from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def choose_slam_method(context, *args, **kwargs):
    slam_choice = input(
        "\nChoose SLAM type:\n"
        "1. slam_toolbox\n"
        "2. cartographer\n"
        "3. hector_slam\n"
        "Enter number: "
    ).strip()

    launch_actions = []

    if slam_choice == '1':
        print("[INFO] Launching SLAM Toolbox")
        launch_actions.append(
            Node(
                package='slam_toolbox',
                executable='async_slam_toolbox_node',
                name='slam_toolbox',
                output='screen',
                parameters=[os.path.join(
                    get_package_share_directory('articubot_one'),
                    'config', 'mapper_params_online_async.yaml')],
                arguments=['--ros-args', '--params-file',
                           os.path.join(get_package_share_directory('articubot_one'),
                                        'config', 'mapper_params_online_async.yaml')],
            )
        )

    elif slam_choice == '2':
        print("[INFO] Launching Cartographer")
        launch_actions.append(
            Node(
                package='cartographer_ros',
                executable='cartographer_node',
                name='cartographer_node',
                output='screen',
                parameters=[os.path.join(
                    get_package_share_directory('articubot_one'),
                    'config', 'cartographer_config.lua')],
                arguments=['-configuration_directory', os.path.join(
                    get_package_share_directory('articubot_one'), 'config'),
                    '-configuration_basename', 'cartographer_config.lua']
            )
        )

    elif slam_choice == '3':
        print("[INFO] Launching Hector SLAM")
        launch_actions.append(
            Node(
                package='hector_mapping',
                executable='hector_mapping',
                name='hector_mapping',
                output='screen',
                parameters=[{'use_sim_time': True}]
            )
        )
    else:
        print("[ERROR] Invalid choice. Exiting.")
        sys.exit(1)

    return launch_actions

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=choose_slam_method)
    ])