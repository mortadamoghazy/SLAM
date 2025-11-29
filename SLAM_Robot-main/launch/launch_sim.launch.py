#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


def generate_launch_description():
    package_name = 'articubot_one'
    pkg_share = os.path.join(os.getenv('HOME'), 'dev_ws', 'src', package_name)

    # Robot State Publisher
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(pkg_share, 'launch', 'rsp.launch.py')]),
        launch_arguments={
            'use_sim_time': 'true',
            'use_ros2_control': 'true'
        }.items()
    )

    # Gazebo Launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join('/opt/ros/humble/share/gazebo_ros/launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'extra_gazebo_args': '--ros-args --params-file ' + os.path.join(pkg_share, 'config', 'gazebo_params.yaml'),
            'world': os.path.join(pkg_share, 'worlds', 'cones.world')
        }.items()
    )

    # Spawn the robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'my_bot'],
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Controller spawners
    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_cont"],
        parameters=[{'use_sim_time': True}]
    )

    joint_broad_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_broad"],
        parameters=[{'use_sim_time': True}]
    )

    # Joystick
    joystick_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(pkg_share, 'launch', 'joystick.launch.py')]),
        launch_arguments={'use_sim_time': 'true'}.items()
    )

    # Twist mux
    twist_mux = Node(
        package="twist_mux",
        executable="twist_mux",
        parameters=[os.path.join(pkg_share, 'config', 'twist_mux.yaml'), {'use_sim_time': True}],
        remappings=[('/cmd_vel_out', '/diff_cont/cmd_vel_unstamped')]
    )

    # IMU covariance override
    imu_cov_override_node = Node(
        package=package_name,
        executable='imu_covariance_override_node.py',
        name='imu_covariance_override',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # EKF fusion
    ekf_fusion_node = Node(
        package=package_name,
        executable='ekf_fusion_node.py',
        name='ekf_fusion',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )


    # SLAM Evaluator (optional utility)
    slam_evaluator = Node(
        package=package_name,
        executable='toolbox_eval.py',
        name='slam_evaluator',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'robot_name': 'my_bot',
            'est_topic': '/pose',
            'output_dir': os.path.expanduser('~/slam_eval'),
            'world_frame': 'odom',
            'base_frame': 'base_link'
        }]
    )

    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity,
        diff_drive_spawner,
        joint_broad_spawner,
        joystick_launch,
        twist_mux,
        imu_cov_override_node,
        ekf_fusion_node,
        slam_evaluator,
        #cartographer_pose_pub
    ])
