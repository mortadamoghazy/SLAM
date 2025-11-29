#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

class ImuCovarianceOverrideNode(Node):
    def __init__(self):
        super().__init__('imu_covariance_override_node')

        self.sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.pub = self.create_publisher(Imu, '/imu_corrected', 10)

        self.orientation_cov = [0.0004, 0.0, 0.0,
                                0.0, 0.0004, 0.0,
                                0.0, 0.0, 0.0004]
        self.angular_vel_cov = [0.0001, 0.0, 0.0,
                                0.0, 0.0001, 0.0,
                                0.0, 0.0, 0.0001]
        self.linear_acc_cov = [0.05, 0.0, 0.0,
                               0.0, 0.05, 0.0,
                               0.0, 0.0, 0.05]

    def imu_callback(self, msg):
        msg.orientation_covariance = self.orientation_cov
        msg.angular_velocity_covariance = self.angular_vel_cov
        msg.linear_acceleration_covariance = self.linear_acc_cov
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImuCovarianceOverrideNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
