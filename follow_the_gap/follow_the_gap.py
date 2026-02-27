import rclpy

import rclpy.time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import numpy as np
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path, Odometry
from follow_the_gap.calculation import CalculationFollowTheGap
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn


# heading_diff is 20° (≈ 0.349 rad). With 0.33 rad, we are slightly below this value.
MAX_STEERING_ANGLE_RADIANS = 0.33
BASIC_SPEED = 0.8
CAR_LENGTH = 0.26


class FollowTheGap(LifecycleNode):
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
        # self.lower_steering_limit = np.tan(-MAX_STEERING_ANGLE_RADIANS) / CAR_LENGTH
        self.lower_steering_limit = -MAX_STEERING_ANGLE_RADIANS

        # curvature calculation upper steering
        # self.upper_steering_limit = np.tan(MAX_STEERING_ANGLE_RADIANS) / CAR_LENGTH
        self.upper_steering_limit = MAX_STEERING_ANGLE_RADIANS

        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.scan_sub = None
        self.ackermann_pub = None
        self.emergency_sub = None

    def on_configure(self, state: State):
        sensor_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        control_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, sensor_qos
        )

        """
        self.odom_sub = self.create_subscription(
            Odometry, "/sensors/imu", self.odom_callback, sensor_qos
        )
        """

        self.ackermann_pub = self.create_lifecycle_publisher(
            AckermannDriveStamped, "/ackermann_cmd", 10
        )

        self.emergency_sub = self.create_subscription(
            Bool, "/emergency_stop", self.emergency_callback, 10
        )

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State):
        self.get_logger().info("Activating...")
        self.ackermann_pub.on_activate(state)

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State):
        self.get_logger().info("Deactivating...")
        self.actuate_car(0.0, 0.0)
        self.ackermann_pub.on_deactivate(state)

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State):
        self.get_logger().info("Cleaning up...")

        self.destroy_publisher(self.ackermann_pub)
        self.destroy_subscription(self.scan_sub)
        self.destroy_subscription(self.emergency_sub)

        return TransitionCallbackReturn.SUCCESS

    def lidar_callback(self, msg_scan: LaserScan):
        """
        Callback for incoming LiDAR. Processes the scan using the
        Follow-The-Gap algorithm, calculates steering and speed, applies
        emergency handling if necessary, and actuates the car.

        The last received odometry message is used together with the LiDAR data
        to compute the steering. Since odometry is published at a higher rate
        than the LiDAR scans, the algorithm always uses the most recent odometry
        data whenever a new LiDAR message arrives.

        Args:
            msg_scan (LaserScan): Incoming LiDAR scan message.
        """

        # check if odom is already available
        # if self.last_odom is None:
        # return

        steering = self.calculate.calculate_steering(msg_scan)
        self.get_logger().info(f"Steering before: {steering}")
        """
        steering = np.clip(
            steering,
            self.lower_steering_limit,
            self.upper_steering_limit,
        )
        """
        steering = self.steering_mapping(steering)
        self.get_logger().info(f"Steering after: {steering}")

        # self.get_logger().info("Steering after: %s", steering)

        speed = (
            BASIC_SPEED  # * speed_factor # + ((np.pi / 2) - np.abs(steering)) / np.pi
        )

        if self.emergency:
            self.actuate_car(0.0, 0.0)
        else:
            self.actuate_car(speed, steering)

    def steering_mapping(self, steering_value):
        # return 0.4/3.14 * steering_value + 0.5
        # return 3.14/0.4 * steering_value - 3.925

        return -3.14 / 0.4 * steering_value + 3.925
        # return -5 * steering_value + 2.5

    def odom_callback(self, msg_odom: Odometry):
        """
        Save the latest odometry data received from the VESC controller.

        Args:
            msg_odom (Odometry): Incoming Odometry message.
        """

        self.last_odom = msg_odom

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

        if not self.ackermann_pub.is_activated:
            return

        ackermann_msg = AckermannDriveStamped()
        ackermann_msg.header.stamp = self.get_clock().now().to_msg()

        ackermann_msg.drive.speed = speed
        ackermann_msg.drive.steering_angle = steering

        self.ackermann_pub.publish(ackermann_msg)


def main():
    rclpy.init()
    node = FollowTheGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
