"""Launch known poses stacking demo."""

import os
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


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
    """Launch the stacking demo."""
    # robot_ip = LaunchConfiguration("robot_ip")
    # hand = LaunchConfiguration("hand")

    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.106.39",
        description="Robot IP address.",
    )

    hand_arg = DeclareLaunchArgument(
        "hand",
        default_value="true",
        description="Whether to use the hand.",
    )

    # moveit configuration
    moveit_config = (
        MoveItConfigsBuilder(robot_name="panda", package_name="franka_robotiq_moveit_config")
        .robot_description(
            file_path=get_package_share_directory("franka_robotiq_description") + "/urdf/realsense_static.urdf.xacro",
            mappings={"robot_ip": "192.168.106.39", "robotiq_gripper": "true"},
        )
        .robot_description_semantic("config/panda.srdf.xacro")
        .trajectory_execution("config/moveit_controllers.yaml")
        .to_moveit_configs()
    )

    # publish static transforms (camera calibration to be included here)
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
    rviz_config = os.path.join(get_package_share_directory("panda_stacking_demo"), "config", "motion_planning.rviz")

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

    # MoveIt stacking application
    stacking_app = Node(
        name="stacking_app",
        package="panda_stacking_demo",
        executable="hard_coded_stack.py",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
        ],
    )

    return LaunchDescription(
        [
            robot_ip_arg,
            hand_arg,
            static_tf,
            robot_state_publisher,
            rviz_node,
            stacking_app,
        ]
    )
