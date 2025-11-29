cmake_minimum_required(VERSION 3.5)
project(articubot_one)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

# Testing support (optional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# Install resource files
install(
  DIRECTORY config description launch worlds
  DESTINATION share/${PROJECT_NAME}
)

# ✅ Install the Python IMU override script
install(
  PROGRAMS
    imu_covariance_override/imu_covariance_override_node.py
    imu_covariance_override/ekf_fusion_node.py
    imu_covariance_override/slam_evaluator.py
    imu_covariance_override/cartographer_pose_publisher.py
    imu_covariance_override/toolbox_eval.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
