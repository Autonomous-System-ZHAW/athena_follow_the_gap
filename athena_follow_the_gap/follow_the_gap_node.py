import rclpy

# import rerun as rr
import time

import rclpy.time
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import numpy as np
from geometry_msgs.msg import PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header
from nav_msgs.msg import Path, Odometry
from athena_follow_the_gap.calculation import CalculationFollowTheGap
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker
from rclpy.lifecycle import LifecycleNode, State, TransitionCallbackReturn
from scipy.spatial.transform import Rotation as R

# heading_diff is 20° (≈ 0.349 rad). With 0.26 rad, we are slightly below this value.
MAX_STEERING_ANGLE_RADIANS = 0.25
BASIC_SPEED = 1.0
CAR_LENGTH = 0.26
WHEELBASE = 0.84


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

        self.last_steering = 0.0
        self.max_delta = 0.2
        self.lookahead = 5.0

        # rr.init("Athena Simulation", recording_id="athena")
        # rr.connect_grpc("rerun+http://localhost:9876/proxy")
        # rr.serve_grpc()

        # rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self.goal_points = []

        self.scan_sub = None
        self.ackermann_pub = None
        self.emergency_sub = None
        self.goal_pub = None
        self.points_pub = None
        self.arc_pub = None

    def on_configure(self, state: State):
        sensor_qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
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

        # goal point for visualisation in Foxglove (normal publisher, also works while inactive)
        self.goal_pub = self.create_publisher(Marker, "/follow_the_gap/goal", 10)
        # lidar points the algorithm uses (front half), for visualisation in Foxglove
        self.points_pub = self.create_publisher(
            Marker, "/follow_the_gap/relevant_points", 10
        )
        # pure pursuit arc from the car to the goal point, for visualisation in Foxglove
        self.arc_pub = self.create_publisher(Marker, "/follow_the_gap/curvature", 10)

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
        self.destroy_publisher(self.goal_pub)
        self.destroy_publisher(self.points_pub)
        self.destroy_publisher(self.arc_pub)

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

        steering, speed, goal, relevant_lidar_points, lidar_points_with_safety = (
            self.calculate.calculate_steering(msg_scan)
        )

        self.get_logger().info(f"Steering 1: {steering}, Speed 1: {speed}")

        """
        steering = np.clip(
            steering,
            self.lower_steering_limit,
            self.upper_steering_limit,
        )
        """

        indices = np.array([p[0] for p in lidar_points_with_safety])
        distances = np.array([p[1] for p in lidar_points_with_safety])

        angles = indices * msg_scan.angle_increment

        x = distances * np.cos(angles)
        y = distances * np.sin(angles)
        z = np.zeros_like(x)

        points_disparity = np.stack([x, y, z], axis=1)

        indices = np.array([p[0] for p in relevant_lidar_points])
        distances = np.array([p[1] for p in relevant_lidar_points])

        angles = indices * msg_scan.angle_increment

        x = distances * np.cos(angles)
        y = distances * np.sin(angles)
        z = np.zeros_like(x)

        points = np.stack([x, y, z], axis=1)

        # publish the relevant lidar points as magenta dots in the lidar frame
        # (display only: rotated by angle_min, so they lie on the scan in Foxglove)
        points_marker = Marker()
        points_marker.header = msg_scan.header  # same frame and time as the scan
        points_marker.ns = "follow_the_gap"
        points_marker.id = 1
        points_marker.type = Marker.POINTS
        points_marker.action = Marker.ADD
        points_marker.pose.orientation.w = 1.0
        points_marker.scale.x = 0.04  # point width [m]
        points_marker.scale.y = 0.04  # point height [m]
        points_marker.color.r = 1.0
        points_marker.color.b = 1.0
        points_marker.color.a = 1.0
        for px, py in zip(x, y):
            dx, dy = self.to_display(px, py, msg_scan.angle_min)
            points_marker.points.append(Point(x=dx, y=dy, z=0.0))
        self.points_pub.publish(points_marker)

        # rr.set_time("sim_time", timestamp=time.time())
        """
        rr.log(
            "car",
            rr.Ellipsoids3D(
                centers=[0, 0, 0],
                half_sizes=[0.1, 0.1, 0.1],
                colors=[255, 200, 10],
                fill_mode="solid",
            ),
        )

        # purple
        rr.log(
            "car/lidar",
            rr.Points3D(
                points,
                radii=[0.02] * len(points),
                colors=[[255, 0, 255]] * len(points),
            ),
        )

        # green
                 
        rr.log(
            "car/lidar_disparity",
            rr.Points3D(
                points_disparity,
                radii=[0.02] * len(points),
                colors=[[51, 255, 51]] * len(points),
            ),
        )
        """

        goal_index, goal_dist = goal[0], goal[1]
        goal_angle = goal_index * msg_scan.angle_increment

        goal_x = goal_dist * np.cos(goal_angle)
        goal_y = goal_dist * np.sin(goal_angle)

        # steering direction from the goal side: goal_x > 0 = goal on the right
        # (algorithm coordinates), ROS steering: + = left, - = right
        sign = 1 if goal_x < 0 else -1

        # publish the goal point as a red sphere in the lidar frame
        marker = Marker()
        marker.header = msg_scan.header  # same frame and time as the scan
        marker.ns = "follow_the_gap"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        goal_dx, goal_dy = self.to_display(goal_x, goal_y, msg_scan.angle_min)
        marker.pose.position.x = goal_dx
        marker.pose.position.y = goal_dy
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color.r = 1.0
        marker.color.a = 1.0
        self.goal_pub.publish(marker)

        # publish the pure pursuit arc from the car (origin) to the goal point
        arc_marker = Marker()
        arc_marker.header = msg_scan.header  # same frame and time as the scan
        arc_marker.ns = "follow_the_gap"
        arc_marker.id = 2
        arc_marker.type = Marker.LINE_STRIP
        arc_marker.action = Marker.ADD
        arc_marker.pose.orientation.w = 1.0
        arc_marker.scale.x = 0.03  # line width [m]
        arc_marker.color.r = 1.0
        arc_marker.color.g = 0.6
        arc_marker.color.a = 1.0
        for ax, ay in self.pursuit_arc(goal_x, goal_y):
            dx, dy = self.to_display(ax, ay, msg_scan.angle_min)
            arc_marker.points.append(Point(x=dx, y=dy, z=0.0))
        self.arc_pub.publish(arc_marker)

        """
        rr.log(
            "car/goal",
            rr.Points3D(
                positions=[[goal_x, goal_y, 0.0]],
                radii=[0.1],
                colors=[[255, 0, 0]],
            ),
        )
        """

        steering = self.pure_pursuit(goal_dist, goal_x) * sign
        """
        rr.log(
            "car/curvature",
            rr.Ellipsoids3D(
                centers=[-1 / steering, 0, 0],
                half_sizes=[1 / steering, 1 / steering, 0.0],
                colors=[255, 200, 10],
            ),
        )
        """
        speed = (
            BASIC_SPEED
            * 4  # * speed_factor # + ((np.pi / 2) - np.abs(steering)) / np.pi
        )

        if self.emergency:
            self.actuate_car(0.0, 0.0)
        else:
            self.actuate_car(speed, steering)

    def to_display(self, x, y, angle_min):
        """Algorithm coordinates -> lidar frame, for the display only.

        The algorithm uses angle = index * angle_increment, the scan's real angle is
        angle_min + index * angle_increment. Rotating by angle_min puts the point where
        it really is in the lidar frame, so it lies on the scan in Foxglove.
        """
        c, s = np.cos(angle_min), np.sin(angle_min)
        return float(c * x - s * y), float(s * x + c * y)

    def pursuit_arc(self, goal_x, goal_y, n=40):
        """Points of the circular arc from the car to the goal (algorithm coordinates).

        In the algorithm coordinates goal_y points forward and goal_x sideways. The arc
        starts at the car (0, 0) heading forward and ends at the goal; its centre lies on
        the x axis at R = (goal_x^2 + goal_y^2) / (2 * goal_x), the curvature is 1 / R
        (same geometry as pure pursuit: 2 * |goal_x| / distance^2).
        """
        if abs(goal_x) < 1e-6:  # goal straight ahead: straight line
            return [(0.0, 0.0), (0.0, float(goal_y))]
        R = (goal_x**2 + goal_y**2) / (2 * goal_x)  # signed radius
        t_goal = np.arctan2(goal_y / R, 1 - goal_x / R)  # arc angle at the goal
        t = np.linspace(0.0, t_goal, n)
        return list(zip(R * (1 - np.cos(t)), R * np.sin(t)))

    def pure_pursuit(self, L, y):
        if abs(y) < 1e-6:
            return 0.001
        curvature = (2 * abs(y)) / (L**2)
        return curvature

    def soft_clip_delta(self, new_steering):
        delta = new_steering - self.last_steering

        # Soft clipping mit tanh
        delta_clipped = self.max_delta * np.tanh(delta / self.max_delta)

        steering_out = self.last_steering + delta_clipped
        self.last_steering = steering_out

        return steering_out

    def steering_mapping(self, steering_value):
        # return 0.4/3.14 * steering_value + 0.5
        # return 3.14/0.4 * steering_value - 3.925
        # return steering_value
        return float(-3.14 / 0.4 * steering_value + 3.925)
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

        ackermann_msg.drive.speed = 1.5  # float(speed)

        self.get_logger().info(f"Steering 2: {steering}, Speed 2: {speed}")

        ackermann_msg.drive.steering_angle = float(steering)

        """
        rr.log(
            "steering",
            rr.SeriesLines(colors=[255, 0, 0], names="steering", widths=2),
            static=True,
        )
        rr.log("steering", rr.Scalars(steering))
        """

        self.ackermann_pub.publish(ackermann_msg)


def main():
    rclpy.init()
    node = FollowTheGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
