#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
import numpy as np
import math

class EKFOdometryFusionNode(Node):
    def __init__(self):
        super().__init__('ekf_odom_fusion_node')

        self.sub_odom = self.create_subscription(Odometry, '/diff_cont/odom', self.odom_callback, 10)
        self.sub_imu = self.create_subscription(Imu, '/imu_corrected', self.imu_callback, 10)
        self.pub_fused = self.create_publisher(Odometry, '/odom_fused', 10)

        self.x = np.zeros((6, 1))  # [x, y, theta, vx, vy, omega]
        self.P = np.eye(6) * 0.1
        self.Q = np.diag([0.01, 0.01, 0.05, 0.1, 0.1, 0.2])
        self.R = np.diag([0.02, 0.02, 0.01])

        self.last_time = None
        self.latest_imu_orientation = None
        self.latest_imu_angular_velocity = None

        self.filtered_yaw = None
        self.yaw_filter_alpha = 0.8

    def imu_callback(self, msg):
        self.latest_imu_orientation = msg.orientation
        self.latest_imu_angular_velocity = msg.angular_velocity.z

        imu_yaw = self.quaternion_to_yaw(msg.orientation)
        if self.filtered_yaw is None:
            self.filtered_yaw = imu_yaw
        else:
            delta = self.normalize_angle(imu_yaw - self.filtered_yaw)
            self.filtered_yaw = self.normalize_angle(
                self.filtered_yaw + (1 - self.yaw_filter_alpha) * delta)

    def odom_callback(self, msg):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            self.x[0, 0] = msg.pose.pose.position.x
            self.x[1, 0] = msg.pose.pose.position.y
            init_yaw = self.quaternion_to_yaw(
                self.latest_imu_orientation or msg.pose.pose.orientation)
            self.filtered_yaw = init_yaw
            self.x[2, 0] = init_yaw
            self.x[3, 0] = msg.twist.twist.linear.x
            self.x[4, 0] = msg.twist.twist.linear.y
            self.x[5, 0] = self.latest_imu_angular_velocity or msg.twist.twist.angular.z
            return

        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0 or dt > 1.0:
            self.last_time = now
            return
        self.last_time = now

        theta = self.x[2, 0]
        v = self.x[3, 0]
        omega = self.latest_imu_angular_velocity or self.x[5, 0]

        self.x[0, 0] += v * math.cos(theta) * dt
        self.x[1, 0] += v * math.sin(theta) * dt
        self.x[2, 0] += omega * dt
        self.x[2, 0] = self.normalize_angle(self.x[2, 0])
        self.x[5, 0] = omega

        F = np.eye(6)
        F[0, 2] = -v * math.sin(theta) * dt
        F[0, 3] = math.cos(theta) * dt
        F[1, 2] = v * math.cos(theta) * dt
        F[1, 3] = math.sin(theta) * dt
        F[2, 5] = dt

        self.P = F @ self.P @ F.T + self.Q

        z = np.array([
            [msg.pose.pose.position.x],
            [msg.pose.pose.position.y],
            [self.filtered_yaw]
        ])

        H = np.zeros((3, 6))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        y = z - H @ self.x
        y[2, 0] = self.normalize_angle(y[2, 0])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        fused_msg = Odometry()
        fused_msg.header.stamp = now.to_msg()
        fused_msg.header.frame_id = 'odom'
        fused_msg.child_frame_id = 'base_link'

        fused_msg.pose.pose.position.x = float(self.x[0])
        fused_msg.pose.pose.position.y = float(self.x[1])
        fused_msg.pose.pose.orientation = self.yaw_to_quaternion(float(self.x[2]))
        fused_msg.twist.twist.linear.x = float(self.x[3])
        fused_msg.twist.twist.linear.y = float(self.x[4])
        fused_msg.twist.twist.angular.z = float(self.x[5])

        fused_msg.pose.covariance[0] = 0.02
        fused_msg.pose.covariance[7] = 0.02
        fused_msg.pose.covariance[35] = 0.01

        fused_msg.twist.covariance[0] = 0.05
        fused_msg.twist.covariance[7] = 0.05
        fused_msg.twist.covariance[35] = 0.02

        self.pub_fused.publish(fused_msg)

    def quaternion_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    node = EKFOdometryFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
