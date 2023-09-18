""" Build a MuJoCo scene for robot manipulation tasks. """

import enum
from typing import Dict, Sequence, Tuple, Union, List
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

# arena models
from dm_robotics.moma.models.arenas import empty

# robot models
from models.arms import franka_emika
from dm_robotics.moma.models.end_effectors.robot_hands import robot_hand
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma import robot

# props
from dm_robotics.moma import prop

# robot hardware
from dm_robotics.moma import sensor
from dm_robotics.moma.sensors.camera_sensor import CameraImageSensor
from dm_robotics.moma.sensors import prop_pose_sensor
from dm_robotics.moma.sensors import robot_arm_sensor
from dm_robotics.moma.sensors import robot_tcp_sensor
from dm_robotics.moma.sensors import robotiq_gripper_sensor
from dm_robotics.moma import effector
from dm_robotics.moma.effectors import arm_effector, default_gripper_effector
from dm_robotics.moma.sensors import prop_pose_sensor
from dm_robotics.moma.sensors import robot_arm_sensor
from dm_robotics.moma.sensors import robotiq_gripper_sensor

# task logic
from dm_robotics.geometry import pose_distribution
from dm_robotics import agentflow as af
from dm_robotics.moma import base_task
from dm_robotics.moma import action_spaces
from dm_robotics.moma import entity_initializer

# physics
from dm_control import composer, mjcf, mujoco
from dm_control.composer.observation import observable

# visualization
import matplotlib.pyplot as plt
import PIL.Image

# custom props
from props import add_objects
from visualization import render_scene

import random
import numpy as np
import jax
import jax.numpy as jnp

def build_arena(name: str) -> composer.Arena:
    """Build a MuJoCo arena."""
    arena = empty.Arena(name=name)
    arena.mjcf_model.option.timestep = 0.001
    arena.mjcf_model.option.gravity = (0., 0., -1.0)
    arena.mjcf_model.size.nconmax = 1000
    arena.mjcf_model.size.njmax = 2000
    arena.mjcf_model.visual.__getattr__('global').offheight = 480
    arena.mjcf_model.visual.__getattr__('global').offwidth = 640
    arena.mjcf_model.visual.map.znear = 0.0005
    return arena

def add_robot_and_gripper(arena: composer.Arena) -> Tuple[composer.Entity, composer.Entity]:
    """Add a robot and gripper to the arena."""""
    # load the robot and gripper
    arm = franka_emika.FER()
    gripper = robotiq_2f85.Robotiq2F85() 

    # attach the gripper to the robot
    robot.standard_compose(arm=arm, gripper=gripper)

    # add the robot and gripper to the arena
    arena.attach(arm)
    
    return arm, gripper

if __name__=="__main__":
    
    ## Build base scene ##
    
    # TODO: read from config file
    MAX_OBJECTS = 3

    # build the base arena
    arena = build_arena("test")

    # add the robot and gripper to the arena
    arm, gripper = add_robot_and_gripper(arena)

    # add objects to the arena
    props, extra_sensors = add_objects(arena, ['block', 'cylinder', 'sphere'], MAX_OBJECTS)

    # build the physics
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    
    ## Robot hardware ##

    # interface for robot control
    arm_hardware_interface = arm_effector.ArmEffector(
            arm,
            action_range_override=None,
            robot_name = "franka_emika_panda",
            )
    gripper_hardware_interface = default_gripper_effector.DefaultGripperEffector(
            gripper,
            robot_name = "robotiq_2f85",
            )

    # robot sensors
    arm_sensor = robot_arm_sensor.RobotArmSensor(
                arm,
                name="franka_emika_panda",
                have_torque_sensors=True,
                )
    gripper_sensor = robotiq_gripper_sensor.RobotiqGripperSensor(
            gripper,
            name="robotiq_2f85",
            )
    gripper_tcp_sensor = robot_tcp_sensor.RobotTCPSensor(gripper, name="robotiq_2f85")
    

    ## Task Definition ##

    # try set up task logic
    robot_sensors = [
            arm_sensor,
            gripper_tcp_sensor,
            gripper_sensor,
    ]

    fer_and_gripper = robot.StandardRobot(
            arm=arm,
            arm_base_site_name="panda_link0",
            gripper=gripper,
            robot_sensors=robot_sensors,
            arm_effector=arm_hardware_interface,
            gripper_effector=gripper_hardware_interface,
            )


    task = base_task.BaseTask(
            task_name="test",
            arena=arena,
            robots=[fer_and_gripper],
            props=props,
            extra_sensors=extra_sensors,
            extra_effectors=[],
            control_timestep=0.1,
            scene_initializer=lambda _: None,
            episode_initializer=lambda _: None,
            )
    
    # get action spec
    parent_action_spec = task.effectors_action_spec(physics)
    
    # define action spaces
    joint_action_space = action_spaces.ArmJointActionSpace(
            af.prefix_slicer(parent_action_spec, arm_hardware_interface.prefix)
            )
    gripper_action_space = action_spaces.GripperActionSpace(
            af.prefix_slicer(parent_action_spec, gripper_hardware_interface.prefix)
            )
    combined_action_space = af.CompositeActionSpace(
            [joint_action_space, gripper_action_space]
            )

    initializers = []
    # robot intializer
    gripper_pose_dist = pose_distribution.UniformPoseDistribution(
            min_pose_bounds=np.array([0.5, -0.1, 0.1,
                              0.75 * np.pi, -0.25 * np.pi, -0.5 * np.pi]),
            max_pose_bounds=np.array([0.7, 0.1, 0.2,
                              1.25 * np.pi, 0.25 * np.pi, 0.5 * np.pi])
    )
    initialize_arm = entity_initializer.PoseInitializer(
            initializer_fn = fer_and_gripper.position_gripper,
            pose_sampler = gripper_pose_dist.sample_pose,
            )
    initializers.append(initialize_arm)

    # prop initializer

    # TODO: read workspace params from config
    for prop in props:
        prop_pose_dist = pose_distribution.UniformPoseDistribution(
                min_pose_bounds=np.array([0.3, -0.35, 0.05, 0.0, 0.0, -np.pi]),
                max_pose_bounds=np.array([0.9, 0.35, 0.05, 0.0, 0.0, np.pi]))
        initialize_prop = entity_initializer.PoseInitializer(
                initializer_fn = prop.set_pose, 
                pose_sampler = prop_pose_dist.sample_pose
                )
        initializers.append(initialize_prop)
    
    # task initializer
    entities_initializer = entity_initializer.TaskEntitiesInitializer(initializers)


    # Run the initializer and see what the scene looks like.
    before = render_scene(physics)
    entities_initializer(physics, np.random.RandomState())
    physics.step()  # propogate the changes from the initializer.
    after = render_scene(physics)
    
    before = PIL.Image.fromarray(before)
    after = PIL.Image.fromarray(after)
    before.show()
    after.show()

