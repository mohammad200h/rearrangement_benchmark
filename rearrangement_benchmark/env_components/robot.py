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


# TODO: overwrite actuators


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
    
    # define cartesian controller
    # Build the 4D cartesian controller, we use a 6D cartesian effector under the
    # hood.
    #effector_model = cartesian_6d_velocity_effector.ModelParams(
    #        element=arm.wrist_site, joints=arm.joints)
    #effector_control = cartesian_6d_velocity_effector.ControlParams(
    #      control_timestep_seconds=0.05,
    #      max_lin_vel=0.07,
    #      max_rot_vel=1.0,
    #      joint_velocity_limits=np.array([2.1750] * 7), # not actually uniform for all joints
    #      nullspace_gain=0.025,
    #      nullspace_joint_position_reference= np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
    #      regularization_weight=1e-2,
    #      enable_joint_position_limits=True,
    #      minimum_distance_from_joint_position_limit=0.01,
    #      joint_position_limit_velocity_scale=0.95,
    #      max_cartesian_velocity_control_iterations=300,
    #      max_nullspace_control_iterations=300)

    # Don't activate collision avoidance because we are restricted to the virtual
    # workspace in the center of the basket.
    #cart_effector_6d = cartesian_6d_velocity_effector.Cartesian6dVelocityEffector(
    #      robot_name="franka_emika_panda",
    #      joint_velocity_effector=arm_hardware_interface,
    #      model_params=effector_model,
    #      control_params=effector_control)
    #cart_effector_4d = cartesian_4d_velocity_effector.Cartesian4dVelocityEffector(
    #      effector_6d=cart_effector_6d,
    #      element=arm.wrist_site,
    #      effector_prefix=f'franka_emika_panda_cart_4d_vel')

    # Constrain the workspace of the robot.
    #cart_effector_4d = cartesian_4d_velocity_effector.limit_to_workspace(
    #      cartesian_effector=cart_effector_4d,
    #      element=gripper.tool_center_point,
    #      min_workspace_limits=WORKSPACE_CENTER - WORKSPACE_SIZE / 2,
    #      max_workspace_limits=WORKSPACE_CENTER + WORKSPACE_SIZE / 2,
    #      wrist_joint=arm.joints[-1],
    #      wrist_limits=WRIST_RANGE,
    #      reverse_wrist_range=True)

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
