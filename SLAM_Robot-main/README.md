## Robot Package Template

slam_toolbox in slam mode: ros2 launch articubot_one online_async_launch.py 
with the mapper_params



slam_toolbox in slam mode: ros2 launch articubot_one online_async_launch.py 
with the localization_params



nav2: ros2 launch articubot_one online_async_launch.py 
with the localization_params


    ros2 run twist_mux twist_mux --ros-args --params-file ./src/articubot_one/config/twist_mux.yaml -r cmd_vel_out:=/diff_cont/cmd_vel_unstamped

    ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true
