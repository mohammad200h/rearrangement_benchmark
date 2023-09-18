"""Build a MuJoCo scene for robot manipulation tasks."""

from typing import Tuple
import numpy as np

# arena models
from dm_robotics.moma.models.arenas import empty

# robot models
from models.arms import franka_emika
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma import robot

# initializers
from dm_robotics.geometry import pose_distribution
from dm_robotics.moma import entity_initializer

# physics
from dm_control import composer, mjcf

# custom props
from props import add_objects
from visualization import render_scene

# config
import hydra
from omegaconf import DictConfig


def build_arena(name: str) -> composer.Arena:
    """Build a MuJoCo arena."""
    arena = empty.Arena(name=name)
    arena.mjcf_model.option.timestep = 0.001
    arena.mjcf_model.option.gravity = (0.0, 0.0, -1.0)
    arena.mjcf_model.size.nconmax = 1000
    arena.mjcf_model.size.njmax = 2000
    arena.mjcf_model.visual.__getattr__("global").offheight = 480
    arena.mjcf_model.visual.__getattr__("global").offwidth = 640
    arena.mjcf_model.visual.map.znear = 0.0005
    return arena


def add_robot_and_gripper(arena: composer.Arena, arm, gripper) -> Tuple[composer.Entity, composer.Entity]:
    """Add a robot and gripper to the arena."""
    # attach the gripper to the robot
    robot.standard_compose(arm=arm, gripper=gripper)

    # add the robot and gripper to the arena
    arena.attach(arm)

    return arm, gripper


@hydra.main(version_base=None, config_path="./config", config_name="scene")
def construct_base_scene(cfg: DictConfig) -> None:
    """Build a base scene for robot manipulation tasks."""
    MAX_OBJECTS = cfg.props.max_objects

    # build the base arena
    arena = build_arena("test")

    # add robot arm and gripper to the arena
    arm = franka_emika.FER()
    gripper = robotiq_2f85.Robotiq2F85()
    arm, gripper = add_robot_and_gripper(arena, arm, gripper)

    # add props to the arena
    props, extra_sensors = add_objects(arena, ["block", "cylinder", "sphere"], MAX_OBJECTS)

    # build the physics
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    # add basic prop initializer so they are visible in the scene
    initializers = []
    for prop in props:
        prop_pose_dist = pose_distribution.UniformPoseDistribution(
            min_pose_bounds=np.array([0.3, -0.35, 0.05, 0.0, 0.0, -np.pi]),
            max_pose_bounds=np.array([0.9, 0.35, 0.05, 0.0, 0.0, np.pi]),
        )
        initialize_prop = entity_initializer.PoseInitializer(
            initializer_fn=prop.set_pose, pose_sampler=prop_pose_dist.sample_pose
        )
        initializers.append(initialize_prop)

    entities_initializer = entity_initializer.TaskEntitiesInitializer(initializers)
    entities_initializer(physics, np.random.RandomState())
    physics.step()

    # visualize the scene
    if cfg.visualize_base_scene:
        render_scene(physics)

    return {
        "arena": arena,
        "physics": physics,
        "arm": arm,
        "gripper": gripper,
        "props": props,
        "extra_sensors": extra_sensors,
    }


if __name__ == "__main__":
    construct_base_scene()
