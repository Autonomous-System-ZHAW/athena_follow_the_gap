from launch import LaunchDescription
from launch_ros.actions import LifecycleNode


def generate_launch_description():

    follow_the_gap_node = LifecycleNode(
        package="follow_the_gap",
        executable="follow_the_gap",
        name="follow_the_gap_controller",
        output="screen",
    )

    return LaunchDescription([follow_the_gap_node])
