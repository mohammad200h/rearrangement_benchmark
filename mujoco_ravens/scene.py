""" Build a MuJoCo scene for robot manipulation tasks. """

import enum
from typing import Dict, Sequence, Tuple, Union

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

@enum.unique
class GripperPoseObservations(enum.Enum):
  """Observations exposed by this sensor.
  
  Typically, for each MoMa sensor we provide a corresponding enum that tracks
  all the available observations for the sensor. It helps with programatically
  fetching observations later when we get to the RL environment.
  """
  # The world x,y,z position of the gripper's tool-center-point.
  POS = '{}_pos'
  # The world orientation quaternion of the gripper's tool-center-point.
  QUAT = '{}_quat'

  def get_obs_key(self, name: str) -> str:
    """Returns the key to the observation in the observables dict."""
    return self.value.format(name)

class GripperPoseSensor(sensor.Sensor[GripperPoseObservations]):

  def __init__(self, gripper: robot_hand.RobotHand, name: str):
    self._gripper = gripper
    self._name = name

    # Mapping of observation names to the composer observables (callables that
    # will produce numpy arrays containing the observations).
    self._observables = {
        self.get_obs_key(GripperPoseObservations.POS):
            observable.Generic(self._pos),
        self.get_obs_key(GripperPoseObservations.QUAT):
            observable.Generic(self._quat),
    }
    for obs in self._observables.values():
      obs.enabled = True

  def initialize_episode(self, physics: mjcf.Physics,
                         random_state: np.random.RandomState) -> None:
    pass  # Nothing special needed each episode.

  @property
  def observables(self) -> Dict[str, observable.Observable]:
    return self._observables

  @property
  def name(self) -> str:
    return self._name

  def get_obs_key(self, obs: GripperPoseObservations) -> str:
    return obs.get_obs_key(self._name)

  def _pos(self, physics: mjcf.Physics) -> np.ndarray:
    return physics.bind(self._gripper.tool_center_point).xpos

  def _quat(self, physics: mjcf.Physics) -> np.ndarray:
    rmat = physics.bind(self._gripper.tool_center_point).xmat
    quat = transformations.mat_to_quat(np.reshape(rmat, [3, 3]))
    return transformations.positive_leading_quat(quat)

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
    #image.show()

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

    # try running control.
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
    
    # visualize the new state of things.
    after = render_scene(physics)
    after = PIL.Image.fromarray(after)
    #after.show()

    # robot sensors
    gripper_sensor = robotiq_gripper_sensor.RobotiqGripperSensor(
            gripper,
            name="robotiq_2f85",
            )
    gripper_pose_sensor = GripperPoseSensor(
            gripper_sensor,
            name="robotiq_2f85_pose",
            )

    # try set up task logic
    robot_sensors = [
            robot_arm_sensor.RobotArmSensor(
                arm,
                name="franka_emika_panda",
                have_torque_sensors=True,
                ),
            gripper_pose_sensor,
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

    block_pose_sensor = prop_pose_sensor.PropPoseSensor(block, name='block')

    task = base_task.BaseTask(
            task_name="test",
            arena=arena,
            robots=[fer_and_gripper],
            props=[block],
            extra_sensors=[block_pose_sensor],
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

    # robot intializer
    gripper_pose_dist = pose_distribution.UniformPoseDistribution(
            min_pose_bounds=np.array([0.5, -0.1, 0.1,
                              0.75 * np.pi, -0.25 * np.pi, -0.5 * np.pi]),
            max_pose_bounds=np.array([0.7, 0.1, 0.2,
                              1.25 * np.pi, 0.25 * np.pi, 0.5 * np.pi])
    )
    print(gripper_pose_dist.sample_pose(np.random.RandomState()))
    initialize_arm = entity_initializer.PoseInitializer(
            initializer_fn = fer_and_gripper.position_gripper,
            pose_sampler = gripper_pose_dist.sample_pose,
            )

    # block initializer
    block_pose_dist = pose_distribution.UniformPoseDistribution(
        min_pose_bounds=np.array([0.5, -0.1, 0.05, 0.0, 0.0, -np.pi]),
        max_pose_bounds=np.array([0.7, 0.1, 0.05, 0.0, 0.0, np.pi]))
    initialize_block = entity_initializer.PoseInitializer(
        initializer_fn = block.set_pose, 
        pose_sampler = block_pose_dist.sample_pose
        )
    
    # task initializer
    entities_initializer = entity_initializer.TaskEntitiesInitializer([
        initialize_arm,
        initialize_block,
    ])


    # Run the initializer and see what the scene looks like.
    before = render_scene(physics)
    entities_initializer(physics, np.random.RandomState())
    physics.step()  # propogate the changes from the initializer.
    after = render_scene(physics)
    
    before = PIL.Image.fromarray(before)
    after = PIL.Image.fromarray(after)
    before.show()
    after.show()

