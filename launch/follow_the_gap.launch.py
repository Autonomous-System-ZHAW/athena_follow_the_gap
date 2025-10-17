from launch import LaunchDescription
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch import events
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from launch.actions import Shutdown, EmitEvent


def generate_launch_description():
    follow_the_gap_node = LifecycleNode(
        package="follow_the_gap",
        executable="follow_the_gap",
        name="follow_the_gap_controller",
        namespace="",
        output="screen",
        on_exit=Shutdown(),
        #        parameters=['config.yaml']
    )

    # configure node
    configure_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=events.matches_action(follow_the_gap_node),
            transition_id=Transition.TRANSITION_CONFIGURE,
        )
    )

    # activate node
    activate_event = EmitEvent(
        event=ChangeState(
            lifecycle_node_matcher=events.matches_action(follow_the_gap_node),
            transition_id=Transition.TRANSITION_ACTIVATE,
        )
    )

    return LaunchDescription([follow_the_gap_node, configure_event, activate_event])
