"""Demonstrating Control Server."""

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

import launch_ros.actions
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch.actions import ExecuteProcess, Shutdown


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
    """Launch the moveit configs and controllers."""
    # moveit config (hardcode mappings for now, due to docker error)
    moveit_config = (
        MoveItConfigsBuilder(robot_name="panda", package_name="franka_robotiq_moveit_config")
        .robot_description(
            file_path=get_package_share_directory("franka_robotiq_description") + "/urdf/robot.urdf.xacro",
            mappings={"robot_ip": "192.168.106.39", "robotiq_gripper": "true"},
        )
        .robot_description_semantic("config/panda.srdf.xacro")
        .trajectory_execution("config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    # robotiq gripper
    robotiq_pkg_share = launch_ros.substitutions.FindPackageShare(package="robotiq_driver").find("robotiq_driver")

    robotiq_update_rate_config_file = PathJoinSubstitution(
        [
            robotiq_pkg_share,
            "config",
            "robotiq_update_rate.yaml",
        ]
    )

    robotiq_controllers_file = "robotiq_controllers.yaml"
    robotiq_controllers = PathJoinSubstitution([robotiq_pkg_share, "config", robotiq_controllers_file])

    # joint state publisher
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        parameters=[{"source_list": ["franka/joint_states"], "rate": 30}],
    )

    # ros 2 controllers
    ros2_controllers_path = os.path.join(
        get_package_share_directory("moveit_resources_franka_panda_moveit_config"),
        "config",
        "ros2_controllers.yaml",
    )
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            ros2_controllers_path,
            robotiq_update_rate_config_file,
            robotiq_controllers,
        ],
        remappings=[("joint_states", "franka/joint_states")],
        output={
            "stdout": "screen",
            "stderr": "screen",
        },
        on_exit=Shutdown(),
    )

    load_controllers = []
    for controller in [
        "panda_arm_controller",
        "joint_state_broadcaster",
        "robotiq_gripper_controller",
        "robotiq_activation_controller",
    ]:
        load_controllers += [
            ExecuteProcess(
                cmd=["ros2 run controller_manager spawner {}".format(controller)],
                shell=True,
                output="screen",
            )
        ]

    return LaunchDescription(
        [
            joint_state_publisher,
            ros2_control_node,
        ]
        + load_controllers
    )
