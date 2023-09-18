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
from props import Block, Cylinder, Sphere

import random
import numpy as np
import jax
import jax.numpy as jnp


@dataclass
class Colours:
    """Colour values for the objects in the scene."""
    colour_map: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "red": (1.0, 0.0, 0.0, 1.0),
            "green": (0.0, 1.0, 0.0, 1.0),
            "blue": (0.0, 0.0, 1.0, 1.0),
            "yellow": (1.0, 1.0, 0.0, 1.0),
            "cyan": (0.0, 1.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0, 1.0),
            "white": (1.0, 1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0, 1.0),
            "grey": (0.5, 0.5, 0.5, 1.0),
            "orange": (1.0, 0.5, 0.0, 1.0),
            "purple": (0.5, 0.0, 0.5, 1.0),
            "brown": (0.5, 0.25, 0.0, 1.0),
            "pink": (1.0, 0.75, 0.8, 1.0),
            })

    @property
    def colour_names(self) -> Sequence[str]:
        """Return the names of the colours."""
        return list(self.colour_map.keys())

    def sample_colour(self,) -> Tuple[float, float, float, float]:
        """Sample a random colour."""
        return self.colour_map[random.choice(self.colour_names)]

def render_scene(physics: mjcf.Physics) -> np.ndarray:
  camera = mujoco.MovableCamera(physics, height=480, width=480)
  camera.set_pose([0.0, 0, 0.0], 2.5, 180, -30)
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

colours = Colours()
MIN_OBJECT_SIZE = 0.02
MAX_OBJECT_SIZE = 0.05

def _add_block(
        arena: composer.Arena,
        name: str = "block",    
        width: float = 0.1,
        height: float = 0.1,
        depth: float = 0.1,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample: bool = False,
        seed: int = 0,
        ) -> composer.Entity:
    if sample:
        # sample block dimensions
        width = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        height = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        depth = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)

        # sample block color
        rgba = colours.sample_colour()

    # create block and add to arena
    block = Block(
        name=name,
        width=width,
        height=height,
        depth=depth,
        rgba=rgba)
    frame = arena.add_free_entity(block)
    block.set_freejoint(frame.freejoint)
    
    return block

def _add_cylinder(
        arena: composer.Arena,
        name: str = "cylinder",
        radius: float = 0.01,
        half_height: float = 0.01,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample: bool = False,
        ) -> composer.Entity:
    if sample:
        # sample cylinder dimensions
        radius = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)
        half_height = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)

        # sample cylinder color
        rgba = colours.sample_colour()

    # create cylinder and add to arena
    cylinder = Cylinder(
        name=name,
        radius=radius,
        half_height=half_height,
        rgba=rgba)

    frame = arena.add_free_entity(cylinder)
    cylinder.set_freejoint(frame.freejoint)
    return cylinder

def _add_sphere(
        arena: composer.Arena,
        name: str = "sphere",
        radius: float = 0.01,
        rgba: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
        sample: bool = False,
        seed: int = 0,
        ) -> composer.Entity:
    if sample:
        # sample sphere dimensions
        radius = np.random.uniform(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE)

        # sample sphere color
        rgba = colours.sample_colour()
    
    # create sphere and add to arena
    sphere = Sphere(
            radius=radius,
            rgba=rgba)
    frame = arena.add_free_entity(sphere)
    sphere.set_freejoint(frame.freejoint)
    return sphere

def _add_object(area: composer.Arena, name: str, sample: bool = False) -> composer.Entity:
    """Add an object to the arena based on object_type."""

    if name == 'block':
        return _add_block(area, sample=sample)
    elif name == 'cylinder':
        return _add_cylinder(area, sample=sample)
    elif name == 'sphere':
        return _add_sphere(area, sample=sample)
    else:
        raise ValueError(f'Unknown object type {name}')

#TODO (add basic samplers for number of each object shape and their colours)

def add_objects(arena: composer.Arena, objects: List[str], max_objects: int) -> List[composer.Entity]:
    """Add objects to the arena."""
    extra_sensors = []
    props = []
    
    # randomly sample num_objects of each object type
    num_objects = np.random.randint(1, max_objects, size=len(objects))
    
    for object_type, amount in zip(objects, num_objects):
        for i in range(amount):
            obj = _add_object(arena, object_type, sample=True)
            props.append(obj)
            extra_sensors.append(prop_pose_sensor.PropPoseSensor(obj, name=f'{object_type}_{i}'))

    return props, extra_sensors 


if __name__=="__main__":
    
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)

    ## Build base scene ##
    MAX_OBJECTS = 3

    # build the base arena
    arena = build_arena("test")

    # add the robot and gripper to the arena
    arm, gripper = add_robot_and_gripper(arena)

    # add objects to the arena
    props, extra_sensors = add_objects(arena, ['block', 'cylinder', 'sphere'], MAX_OBJECTS)

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

