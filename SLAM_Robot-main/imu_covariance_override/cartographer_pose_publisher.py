#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from rclpy.duration import Duration
from builtin_interfaces.msg import Time as BuiltinTime


class CartographerPosePublisher(Node):
    def __init__(self):
        super().__init__('cartographer_pose_pub')

        # Publisher to send estimated pose
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/cartographer_pose', 10)

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer callback at 20Hz
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        try:
            now = self.get_clock().now()
            # Lookup the transform from map → base_link
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())


            # Prepare the PoseWithCovarianceStamped message
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = now.to_msg()
            pose_msg.header.frame_id = 'map'  # Correct frame (not 'odom')

            pose_msg.pose.pose.position.x = trans.transform.translation.x
            pose_msg.pose.pose.position.y = trans.transform.translation.y
            pose_msg.pose.pose.position.z = trans.transform.translation.z
            pose_msg.pose.pose.orientation = trans.transform.rotation

            # Small constant covariance
            pose_msg.pose.covariance = [
                1e-4, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1e-4, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1e-4, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1e-4, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 1e-4, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 1e-4
            ]

            # Publish the estimated pose
            self.pose_pub.publish(pose_msg)
            self.get_logger().info(
                f"✅ Published Pose at ({pose_msg.pose.pose.position.x:.3f}, {pose_msg.pose.pose.position.y:.3f})"
            )

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'⚠️ TF lookup failed for map → base_link: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = CartographerPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
