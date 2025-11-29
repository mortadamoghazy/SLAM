include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  map_frame = "map",
  tracking_frame = "base_link",           -- Robot base for internal tracking
  published_frame = "base_link",          -- Reported pose frame
  odom_frame = "odom",                    -- EKF-fused odometry frame
  provide_odom_frame = true,            -- Don't publish tf from Cartographer

  publish_frame_projected_to_2d = true,  -- Flatten pose (needed for Nav2)

  use_odometry = false,                   -- Use odom from EKF (odom_fused)
  use_nav_sat = false,
  use_landmarks = false,

  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,   -- Use full scan (good for LD06)
  num_point_clouds = 0,

  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,       -- Map update frequency
  pose_publish_period_sec = 5e-3,        -- Robot pose publishing freq
  trajectory_publish_period_sec = 30e-3, -- Full trajectory topic publishing

  rangefinder_sampling_ratio = 1.0,
  odometry_sampling_ratio = 1.0,
  fixed_frame_pose_sampling_ratio = 1.0,
  imu_sampling_ratio = 1.0,
  landmarks_sampling_ratio = 1.0,
}

MAP_BUILDER.use_trajectory_builder_2d = true

-- Trajectory builder tuning for tight indoor spaces
TRAJECTORY_BUILDER_2D.min_range = 0.1                 -- LD06 safe min
TRAJECTORY_BUILDER_2D.max_range = 5.0                 -- 5m max for apartments
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 2.0   -- For areas with no return
TRAJECTORY_BUILDER_2D.use_imu_data = false            -- Already fused in EKF

-- Improve scan matching in small spaces
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.01)  -- Lower = more sensitive to rotation
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.01          -- More frequent updates in small rooms
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 1.0

-- Increase scan matching accuracy
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 40.0

-- Submap resolution (default is good for indoor)
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 35     -- Smaller = faster submap creation

-- Pose graph optimization
POSE_GRAPH.optimize_every_n_nodes = 20
POSE_GRAPH.constraint_builder.min_score = 0.55        -- Allow slightly weaker matches
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.6

return options