"""Build a MuJoCo scene for robot manipulation tasks."""

from typing import Tuple

# arena models
from dm_robotics.moma.models.arenas import empty

# robot models
from models.arms import franka_emika
from dm_robotics.moma.models.end_effectors.robot_hands import robotiq_2f85
from dm_robotics.moma import robot

# physics
from dm_control import composer, mjcf

# custom props
from props import add_objects, Rectangle
from cameras import add_camera

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
    arena.mjcf_model.visual.__getattr__("global").offheight = 640
    arena.mjcf_model.visual.__getattr__("global").offwidth = 640
    arena.mjcf_model.visual.map.znear = 0.0005
    return arena


def add_basic_table(arena: composer.Arena) -> Rectangle:
    """Add a basic table to the arena."""
    table = Rectangle(
        name="table",
        x_len=0.6,
        y_len=0.8,
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


@hydra.main(version_base=None, config_path="./config", config_name="scene")
def construct_base_scene(cfg: DictConfig) -> None:
    """Build a base scene for robot manipulation tasks."""

    # build the base arena
    arena = build_arena("test")

    # add a basic table to the arena
    add_basic_table(arena)

    # add robot arm and gripper to the arena
    arm = franka_emika.FER()
    gripper = robotiq_2f85.Robotiq2F85()
    arm, gripper = add_robot_and_gripper(arena, arm, gripper)

    # add props to the arena
    props, extra_sensors = add_objects(arena,
                                       shapes=cfg.props.shapes,
                                       colours=cfg.props.colours,
                                       min_object_size=cfg.props.min_object_size,
                                       max_object_size=cfg.props.max_object_size,
                                       min_objects=cfg.props.min_objects,
                                       max_objects=cfg.props.max_objects,
                                       sample_size=cfg.props.sample_size,
                                       sample_colour=cfg.props.sample_colour,)

    # add overhead camera to the arena
    overhead_camera, overhead_camera_sensor = add_camera(
        arena,
        "overhead_camera",
        pos=(0.4, 0.0, 2.0),
        quat=(0.7068252, 0, 0, 0.7073883),
        height=640,
        width=640,
        fovy=120,
    )

    extra_sensors += overhead_camera_sensor

    front_camera, front_camera_sensor = add_camera(
        arena,
        "front_camera",
        pos=(2.5, 0.0, 1.4),
        quat=(0.6133964, 0.3514872, 0.3512074, 0.6138851),
        height=640,
        width=640,
        fovy=120,
    )

    extra_sensors += front_camera_sensor

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
