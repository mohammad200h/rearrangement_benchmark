#!/usr/bin/env python3
"""Basic cube stacking demo."""


import copy
import time
import rclpy

# message libraries
from geometry_msgs.msg import PoseStamped

# moveit_py
from moveit.planning import (
    MoveItPy,
    PlanRequestParameters,
)

# gripper client
from gripper_action_client import GripperClient

# config file libraries
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory


# we need to specify our moveit_py config at the top of each notebook we use.
# this is since we will start spinning a moveit_py node within this notebook.

moveit_config = (
    MoveItConfigsBuilder(robot_name="panda", package_name="franka_robotiq_moveit_config")
    .robot_description(
        file_path=get_package_share_directory("franka_robotiq_description") + "/urdf/robot.urdf.xacro",
        mappings={"robot_ip": "192.168.106.39", "robotiq_gripper": "true"},
    )
    .robot_description_semantic("config/panda.srdf.xacro")
    .trajectory_execution("config/moveit_controllers.yaml")
    .moveit_cpp(file_path=get_package_share_directory("panda_stacking_demo") + "/config/notebook.yaml")
    .to_moveit_configs()
).to_dict()


def go_to_configuration(panda, panda_arm, configuration_name):
    """Go to a configuration by name."""
    # set plan start state using predefined state
    panda_arm.set_start_state_to_current_state()

    # set goal using a pose message this time
    panda_arm.set_goal_state(configuration_name=configuration_name)

    # plan to goal
    planner_params = PlanRequestParameters(panda, "pilz_lin")
    planner_params.max_velocity_scaling_factor = 0.2
    planner_params.max_acceleration_scaling_factor = 0.2
    plan_result = panda_arm.plan(single_plan_parameters=planner_params)

    # execute the plan
    if plan_result:
        robot_traj = plan_result.trajectory
        panda.execute(robot_traj, blocking=True, controllers=[])


def go_to_pose(panda, panda_arm, pose, link):
    """Go to a pose."""
    # set plan start state to current state
    panda_arm.set_start_state_to_current_state()

    # set pose goal with PoseStamped message
    from geometry_msgs.msg import PoseStamped

    pose_goal = PoseStamped()
    pose_goal.header.frame_id = "panda_link0"
    pose_goal.pose.orientation.x = pose.pose.orientation.x
    pose_goal.pose.orientation.y = pose.pose.orientation.y
    pose_goal.pose.orientation.z = pose.pose.orientation.z
    pose_goal.pose.orientation.w = pose.pose.orientation.w
    pose_goal.pose.position.x = pose.pose.position.x
    pose_goal.pose.position.y = pose.pose.position.y
    pose_goal.pose.position.z = pose.pose.position.z
    panda_arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link=link)

    # plan to goal
    planner_params = PlanRequestParameters(panda, "pilz_lin")
    planner_params.max_velocity_scaling_factor = 0.1
    planner_params.max_acceleration_scaling_factor = 0.1
    plan_result = panda_arm.plan(single_plan_parameters=planner_params)

    # execute the plan
    if plan_result:
        robot_traj = plan_result.trajectory
        panda.execute(robot_traj, blocking=True, controllers=[])


if __name__ == "__main__":
    # initialise rclpy (only for logging purposes)
    rclpy.init()

    # instantiate moveit_py instance and a planning component for the panda_arm
    panda = MoveItPy(node_name="moveit_py", config_dict=moveit_config)
    panda_arm = panda.get_planning_component("panda_arm")
    gripper = GripperClient("/robotiq_gripper_controller/gripper_cmd")

    def open_gripper():
        """Open the gripper."""
        gripper.send_gripper_command(0.0, 0.0)

    def close_gripper():
        """Close the gripper."""
        gripper.send_gripper_command(1.0, 0.025)

    # hardcoded poses (until grasp pose estimation is completed)
    place_pose = PoseStamped()
    place_pose.header.frame_id = "panda_link0"
    place_pose.pose.position.x = 0.5038
    place_pose.pose.position.y = 0.0
    place_pose.pose.position.z = 0.1795
    place_pose.pose.orientation.x = 0.9972
    place_pose.pose.orientation.y = 0.0361
    place_pose.pose.orientation.z = 0.0
    place_pose.pose.orientation.w = 0.0

    pick_pose = PoseStamped()
    pick_pose.header.frame_id = "panda_link0"
    pick_pose.pose.position.x = 0.147
    pick_pose.pose.position.y = -0.3994
    pick_pose.pose.position.z = 0.1795
    pick_pose.pose.orientation.x = 0.9972
    pick_pose.pose.orientation.y = 0.0361
    pick_pose.pose.orientation.z = 0.0
    pick_pose.pose.orientation.w = 0.0

    # stack the blocks
    for i in range(5):

        if i != 0:
            pick_pose.pose.position.x += 0.1
            place_pose.pose.position.z += 0.05

        # start at ready configuration
        go_to_configuration(panda, panda_arm, "ready")
        open_gripper()  # ensure gripper starts open

        # approach pick pose
        approach_pose = copy.deepcopy(pick_pose)
        approach_pose.pose.position.z += 0.1
        go_to_pose(panda, panda_arm, approach_pose, "panda_link8")
        go_to_pose(panda, panda_arm, pick_pose, "panda_link8")

        # close gripper
        time.sleep(1)
        close_gripper()
        go_to_pose(panda, panda_arm, approach_pose, "panda_link8")

        # back to ready configuration
        go_to_configuration(panda, panda_arm, "ready")

        # approach place pose
        approach_pose = copy.deepcopy(place_pose)
        approach_pose.pose.position.z += 0.1
        go_to_pose(panda, panda_arm, approach_pose, "panda_link8")
        go_to_pose(panda, panda_arm, place_pose, "panda_link8")

        # open gripper
        time.sleep(1)
        open_gripper()

    # unstack the blocks
    for i in range(5):
        if i != 0:
            pick_pose.pose.position.x -= 0.1
            place_pose.pose.position.z -= 0.05

        # start at ready configuration
        go_to_configuration(panda, panda_arm, "ready")

        # Note: previous pick poses are now place poses and vice versa
        pick_pose_ = copy.deepcopy(place_pose)
        place_pose_ = copy.deepcopy(pick_pose)

        # approach pick pose
        approach_pose = copy.deepcopy(pick_pose_)
        approach_pose.pose.position.z += 0.1
        go_to_pose(panda, panda_arm, approach_pose, "panda_link8")
        go_to_pose(panda, panda_arm, pick_pose_, "panda_link8")

        # close gripper
        time.sleep(1)
        close_gripper()
        go_to_pose(panda, panda_arm, approach_pose, "panda_link8")

        # back to ready configuration
        go_to_configuration(panda, panda_arm, "ready")

        # approach place pose
        approach_pose = copy.deepcopy(place_pose_)
        approach_pose.pose.position.z += 0.1
        go_to_pose(panda, panda_arm, approach_pose, "panda_link8")
        go_to_pose(panda, panda_arm, place_pose_, "panda_link8")

        # open gripper
        time.sleep(1)
        open_gripper()
        go_to_pose(panda, panda_arm, approach_pose, "panda_link8")

    # back to ready configuration
    go_to_configuration(panda, panda_arm, "ready")

    # shutdown rclpy
    rclpy.shutdown()
