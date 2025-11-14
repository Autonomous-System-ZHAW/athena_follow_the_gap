import rclpy
from rclpy.node import Node

import rclpy.time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import numpy as np
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path
from follow_the_gap.calculation import CalculationFollowTheGap
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool

# heading_diff is 20° (≈ 0.349 rad). With 0.33 rad, we are slightly below this value.
MAX_STEERING_ANGLE_RADIANS = 0.33
BASIC_SPEED = 1.7825
CAR_LENGTH = 0.26


class FollowTheGap(Node):
    """
    ROS2 node that implements the Follow-The-Gap algorithm for autonomous
    navigation. It subscribes to LiDAR scans, computes the optimal steering
    angle and speed, and publishes control commands for the vehicle.
    """

    def __init__(self):
        super().__init__("follow_the_gap")
        self.get_logger().info("Follow the Gap Node has been started.")

        self.calculate = CalculationFollowTheGap(self.get_logger)
        self.tf_buffer = Buffer()
        self.emergency = False
        # curvature calculation lower steering
        self.lower_steering_limit = np.tan(-MAX_STEERING_ANGLE_RADIANS) / CAR_LENGTH
        # curvature calculation upper steering
        self.upper_steering_limit = np.tan(MAX_STEERING_ANGLE_RADIANS) / CAR_LENGTH

        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, qos_policy
        )

        self.ackermann_pub = self.create_publisher(
            AckermannDriveStamped, "/ackermann_cmd", qos_policy
        )

        self.sub_emergency = self.create_subscription(
            Bool, "/emergency", self.emergency_callback, 10
        )

    def get_transform(self):
        """
        Lookup the transformation between `odom` and `base_link` frames.

        Returns:
            tuple[float, float]: The x and y translation (in meters) of
                                 `base_link` relative to `odom`.
        """

        now = rclpy.time.Time()
        trans: TransformStamped = self.tf_buffer.lookup_transform(
            "odom", "base_link", now  # parent frame  # child frame
        )

        x = trans.transform.translation.x
        y = trans.transform.translation.y
        z = trans.transform.translation.z

        self.get_logger().info(
            f"base_link Position relativ zu odom: x={x:.2f}, y={y:.2f}, z={z:.2f}"
        )
        return x, y

    def lidar_callback(self, msg: LaserScan):
        """
        Callback for incoming LiDAR data. Processes the scan using the
        Follow-The-Gap algorithm, calculates steering and speed, applies
        emergency handling if necessary, and actuates the car.

        Args:
            msg (LaserScan): Incoming LiDAR scan message.
        """

        steering, speed_factor = self.calculate.calculate_steering(msg)
        steering = np.clip(
            steering,
            BASIC_SPEED * self.lower_steering_limit,
            BASIC_SPEED * self.upper_steering_limit,
        )
        speed = (
            BASIC_SPEED  # * speed_factor # + ((np.pi / 2) - np.abs(steering)) / np.pi
        )

        if self.emergency:
            self.actuate_car(0.0, 0.0)
        else:
            self.actuate_car(speed, steering)

    def emergency_callback(self, msg: Bool) -> None:
        """
        Callback for the emergency stop signal. Updates the emergency
        state of the vehicle.

        Args:
            msg (Bool): True if emergency stop is active, False otherwise.
        """

        self.emergency = msg.data

    def actuate_car(self, speed, steering):
        """
        Publish velocity and steering commands to the car.

        Args:
            speed (float): Linear velocity command (m/s).
            steering (float): Steering angle command (radians).
        """

        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()

        ackermann_msg.drive.speed = speed
        ackermann_msg.drive.steering_angle = steering

        self.ackermann_pub.publish(ackermann_msg)


def main():
    rclpy.init()

    node = FollowTheGap()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
