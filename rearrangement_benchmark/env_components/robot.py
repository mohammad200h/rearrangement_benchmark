"""Utilities for setting up robot."""

import numpy as np

from dm_robotics.moma.effectors import (
        arm_effector, 
        default_gripper_effector,
        cartesian_6d_velocity_effector,
        cartesian_4d_velocity_effector,
        )
from dm_robotics.moma.sensors import robot_arm_sensor, robotiq_gripper_sensor, robot_tcp_sensor
from dm_robotics.moma import robot


def setup_robot_manipulator(cfg, scene_components):
    """Set up robot manipulator."""
    # read robot and gripper from scene
    arm = scene_components["arm"]
    gripper = scene_components["gripper"]

    # set up robot control interfaces
    arm_hardware_interface = arm_effector.ArmEffector(
        arm,
        action_range_override=None,
        robot_name="franka_emika_panda",
    )

    gripper_hardware_interface = default_gripper_effector.DefaultGripperEffector(
        gripper,
        robot_name="robotiq_2f85",
    )

    # set up robot sensors
    arm_sensor = robot_arm_sensor.RobotArmSensor(
        arm,
        name="franka_emika_panda",
        have_torque_sensors=False,
    )

    gripper_sensor = robotiq_gripper_sensor.RobotiqGripperSensor(
        gripper,
        name="robotiq_2f85",
    )

    gripper_tcp_sensor = robot_tcp_sensor.RobotTCPSensor(gripper, name="robotiq_2f85")
    robot_sensors = [arm_sensor, gripper_sensor, gripper_tcp_sensor]

    # instantiate robot
    return robot.StandardRobot(
        arm=arm,
        arm_base_site_name="panda_link0",
        gripper=gripper,
        robot_sensors=robot_sensors,
        arm_effector=arm_hardware_interface,
        gripper_effector=gripper_hardware_interface,
    )
