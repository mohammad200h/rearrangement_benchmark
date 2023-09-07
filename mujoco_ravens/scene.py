""" Build a MuJoCo scene for robot manipulation tasks. """

from typing import Dict, Sequence, Tuple, Union

import numpy as np
import mujoco

# robot models
from models.arms import franka_emika
from dm_robotics.moma import robot
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma.models.robots.robot_arms import robot_arm
from dm_robotics.moma.models.arenas import empty

# robot hardware
from dm_robotics.moma import sensor
from dm_robotics.moma import effector

# physics
from dm_control import composer, mjcf, mujoco

# visualization
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


if __name__=="__main__":
    # build the scene
    arena = build_arena("test")
    arm, gripper = add_robot_and_gripper(arena)

    # visualize the scene
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    image = PIL.Image.fromarray(render_scene(physics))
    image.show()
