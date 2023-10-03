"""Launch motion planning prerequisites."""

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

from launch import LaunchDescription
from launch_ros.actions import Node


def load_yaml(package_name, file_path):
    """Load a yaml file from the specified package and file path."""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def generate_launch_description():
    """Launch motion planning."""
    # moveit configuration
    moveit_config = (
        MoveItConfigsBuilder(robot_name="panda", package_name="franka_robotiq_moveit_config")
        .robot_description(
            file_path=get_package_share_directory("franka_robotiq_description") + "/urdf/robot.urdf.xacro",
            mappings={"use_fake_hardware": "true", "robot_ip": "192.168.106.99", "robotiq_gripper": "true"},
        )
        .robot_description_semantic("config/panda.srdf.xacro")
        .trajectory_execution("config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    # publish static transforms
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        arguments=["0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "panda_link0"],
    )

    # publish robot state
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[moveit_config.robot_description],
    )

    # setup rviz
    rviz_config = os.path.join(get_package_share_directory("rearrangements"), "config", "motion_planning.rviz")

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription(
        [
            static_tf,
            robot_state_publisher,
            rviz_node,
        ]
    )
