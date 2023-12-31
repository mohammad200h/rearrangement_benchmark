"""Build a MuJoCo scene for robot manipulation tasks."""

from typing import Tuple

# arena models
from dm_robotics.moma.models.arenas import empty

# robot
from dm_robotics.moma import robot

# physics
from dm_control import composer, mjcf

# custom props
from rearrangement_benchmark.env_components.props import add_objects, Rectangle
from rearrangement_benchmark.env_components.cameras import add_camera

# config
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def build_arena(name: str) -> composer.Arena:
    """Build a MuJoCo arena."""
    arena = empty.Arena(name=name)
    arena.mjcf_model.option.timestep = 0.0001
    arena.mjcf_model.option.gravity = (0.0, 0.0, -9.8)
    arena.mjcf_model.size.nconmax = 1000
    arena.mjcf_model.size.njmax = 2000
    arena.mjcf_model.visual.__getattr__("global").offheight = 640
    arena.mjcf_model.visual.__getattr__("global").offwidth = 640
    arena.mjcf_model.visual.map.znear = 0.0005
    return arena


def add_basic_table(arena: composer.Arena) -> Rectangle:
    """Add a basic table to the arena."""
    table = Rectangle(
        name="table",
        x_len=0.8,
        y_len=1.0,
        z_len=0.4,
        rgba=(0.5, 0.5, 0.5, 1.0),
    )

    # define attachment site
    attach_site = arena.mjcf_model.worldbody.add(
        "site",
        name="table_center",
        pos=(0.4, 0.0, 0.0),
    )

    arena.attach(table, attach_site)

    return table


def add_robot_and_gripper(arena: composer.Arena, arm, gripper) -> Tuple[composer.Entity, composer.Entity]:
    """Add a robot and gripper to the arena."""
    # attach the gripper to the robot
    robot.standard_compose(arm=arm, gripper=gripper)

    # define robot base site
    robot_base_site = arena.mjcf_model.worldbody.add(
        "site",
        name="robot_base",
        pos=(0.0, 0.0, 0.4),
    )

    # add the robot and gripper to the arena
    arena.attach(arm, robot_base_site)

    return arm, gripper


@hydra.main(version_base=None, config_path="../config", config_name="scene")
def construct_base_scene(cfg: DictConfig) -> None:
    """Build a base scene for robot manipulation tasks."""
    # build the base arena
    arena = build_arena("base_scene")

    # add a basic table to the arena
    add_basic_table(arena)

    # add robot arm and gripper to the arena
    arm = instantiate(cfg.robots.arm)
    gripper = instantiate(cfg.robots.gripper)
    # arm = franka_emika.FER()
    # gripper = robotiq_2f85.Robotiq2F85()
    arm, gripper = add_robot_and_gripper(arena, arm, gripper)

    # add props to the arena
    props, extra_sensors = add_objects(
        arena,
        shapes=cfg.props.shapes,
        colours=cfg.props.colours,
        min_object_size=cfg.props.min_object_size,
        max_object_size=cfg.props.max_object_size,
        min_objects=cfg.props.min_objects,
        max_objects=cfg.props.max_objects,
        sample_size=cfg.props.sample_size,
        sample_colour=cfg.props.sample_colour,
    )

    # add cameras to the arena
    for camera in cfg.cameras:
        camera, camera_sensor = add_camera(
            arena,
            name=camera.name,
            pos=camera.pos,
            quat=camera.quat,
            height=camera.height,
            width=camera.width,
            fovy=camera.fovy,
        )
        
        extra_sensors += camera_sensor

    # build the physics
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

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
