import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.impl.rcutils_logger import RcutilsLogger

CAR_WIDTH = 0.21
MINIMAL_DISPARITY_GAP = 1.4
DISPARITY_DISTANCE = 3.5
DEFAULT_GAP = 0.8
DISTANCE_10_METER = 10
DISTANCE_5_METER = 5
DISTANCE_2_METER = 2
DISTANCE_1_METER = 1


class CalculationFollowTheGap:
    """
    This class implements the Follow-The-Gap algorithm for autonomous
    driving based on LiDAR input. It processes laser scan data,
    detects gaps between obstacles, and computes the optimal steering
    angle and speed for safe navigation.
    """

    def __init__(self, logger: RcutilsLogger):
        self.get_logger = logger

    def calculate_steering(self, msg_scan: LaserScan):
        """
        Calculate the steering angle and driving speed using the
        Follow-The-Gap algorithm.

        Args:
            msg_scan (LaserScan): LiDAR scan message containing distance
                                measurements around the vehicle.

        Returns:
            tuple[float, float]: Steering angle (radians) and driving speed.
        """

        # filter only relevant lidar points
        relevant_lidar_points = self.__process_lidar_data(msg_scan)

        # calculate disparity of the relevant lidar points
        lidar_points_with_safety = self.__calculate_disparities(relevant_lidar_points)

        # calculate gaps
        all_possible_gaps = self.__possible_gaps(lidar_points_with_safety)

        # evaluate the best gap and goal target
        goal_target = self.__max_gap(all_possible_gaps)

        # calculate steering according to the goal target
        steering, speed = self.__best_goal_point(goal_target)

        # return steering/goal_target[1]*1.7825, speed
        return steering, speed

    def __process_lidar_data(self, msg: LaserScan) -> list:
        self.angle_increment = msg.angle_increment
        lidar_range = len(msg.ranges)
        self.angle_increment_value_180_degree = (lidar_range / 2) * self.angle_increment
        relevant_lidar_points = []

        # only between 0° and 180°
        for index, distance in enumerate(msg.ranges):
            if (
                index * self.angle_increment >= 0
                and index * self.angle_increment
                <= self.angle_increment_value_180_degree
            ):
                if np.isnan(distance):
                    distance = (
                        relevant_lidar_points[-1][1]
                        if len(relevant_lidar_points) != 0
                        else DEFAULT_GAP
                    )
                if distance < CAR_WIDTH:
                    distance = CAR_WIDTH

                relevant_lidar_points.append((index, distance))

        return relevant_lidar_points

    def __calculate_disparities(self, relevant_lidar_points: list) -> list:
        disparities = []
        for i in range(len(relevant_lidar_points) - 1):
            current_index, current_distance = relevant_lidar_points[i]
            _, next_distance = relevant_lidar_points[i + 1]
            disparity = abs(next_distance - current_distance)

            if (
                MINIMAL_DISPARITY_GAP <= disparity
                and current_distance <= DISPARITY_DISTANCE
            ):
                disparities.append(current_index)

        lidar_points_with_safety = self.__safety_bubble(
            relevant_lidar_points, disparities
        )

        return lidar_points_with_safety

    def __possible_gaps(self, lidar_points_with_safety: list) -> list:
        possible_gaps = []
        gap = []

        for point in lidar_points_with_safety:
            number = point[1]
            if number > 0:
                gap.append((point[0], number))
            else:
                if gap:
                    possible_gaps.append(gap)
                    gap = []

        if gap:
            possible_gaps.append(gap)

        return possible_gaps

    def __max_gap(self, all_possible_gaps_with_safety: list) -> list:
        deepest_gap_sum = 0
        index_deepest_gap = []

        for i in all_possible_gaps_with_safety:
            sum_current_gap = 0

            for index, dist in i:
                sum_current_gap += dist

            sum_current_gap = sum_current_gap / len(i)

            if deepest_gap_sum < sum_current_gap:
                deepest_gap_sum = sum_current_gap
                index_deepest_gap = i

        best_point = max(index_deepest_gap, key=lambda p: p[1])

        return best_point

    def __safety_bubble(self, relevant_lidar_points: list, disparities: list) -> list:
        # set safety bubble and all points inside the bubbel to zero
        filtered_points = relevant_lidar_points.copy()
        for index_dis in disparities:
            _, distance = filtered_points[index_dis]
            if distance <= 0:
                continue
            alpha = 2 * np.arcsin(CAR_WIDTH / (2 * distance))
            bubble_indices = int(alpha / self.angle_increment)

            for i in range(len(filtered_points)):
                index, _ = filtered_points[i]
                if abs(index - index_dis) <= bubble_indices:
                    filtered_points[i] = (index, 0)

        return filtered_points

    def __best_goal_point(self, goal_target):
        speed_factor = 0
        if goal_target[1] > DISTANCE_10_METER:
            speed_factor = 3.0
        elif goal_target[1] > DISTANCE_5_METER:
            speed_factor = 2.0
        elif goal_target[1] > DISTANCE_2_METER:
            speed_factor = 1.0
        elif goal_target[1] > DISTANCE_1_METER:
            speed_factor = 0.6
        else:
            speed_factor = 0.4
        steering = goal_target[0] * self.angle_increment
        return (
            steering - (self.angle_increment_value_180_degree / 2)
        ) * speed_factor, speed_factor
