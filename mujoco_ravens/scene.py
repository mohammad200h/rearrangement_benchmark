""" Build a MuJoCo scene for robot manipulation tasks. """

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

# arena models
from dm_robotics.moma.models.arenas import empty

# robot models
from models.arms import franka_emika
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma import robot

# props
from dm_robotics.moma import prop

# robot hardware
from dm_robotics.moma import sensor
from dm_robotics.moma.sensors.camera_sensor import CameraImageSensor
from dm_robotics.moma import effector
from dm_robotics.moma.effectors import arm_effector, default_gripper_effector

# physics
from dm_control import composer, mjcf, mujoco

# visualization
import matplotlib.pyplot as plt
import PIL.Image


def render_scene(physics: mjcf.Physics) -> np.ndarray:
  camera = mujoco.MovableCamera(physics, height=480, width=480)
  camera.set_pose([0, 0, 0.75], 2.5, 180, -30)
  return camera.render()

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

def _add_block(arena: composer.Arena) -> composer.Entity:
  block = prop.Block()
  frame = arena.add_free_entity(block)
  block.set_freejoint(frame.freejoint)
  return block

if __name__=="__main__":
    # build the base arena
    arena = build_arena("test")

    # add the robot and gripper to the arena
    arm, gripper = add_robot_and_gripper(arena)

    # add objects to the arena
    block = _add_block(arena)

    # visualize the current scene
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    image = PIL.Image.fromarray(render_scene(physics))
    image.show()

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

    # Try running the effectors in a loop.
    num_steps = int(1. / physics.timestep())  # run for a second
    min_control = arm_hardware_interface.action_spec(physics).minimum
    max_control = arm_hardware_interface.action_spec(physics).maximum
    key = random.PRNGKey(0)
    arm_command = jax.vmap(random.uniform, in_axes=(None, None, None, 0, 0), out_axes=0)(
            key,
            (1,),
            jnp.float32,
            min_control,
            max_control,
            ).squeeze().__array__()
    gripper_command = np.array([255.0], dtype=np.float32) # close the gripper

    for _ in range(num_steps):
      arm_hardware_interface.set_control(physics, arm_command)
      gripper_hardware_interface.set_control(physics, gripper_command)
      physics.step()
    
    # Visualize the new state of things.
    after = render_scene(physics)
    after = PIL.Image.fromarray(after)
    after.show()
