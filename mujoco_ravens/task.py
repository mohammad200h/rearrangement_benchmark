"""A script to setup RAVENS tasks."""

# config
import hydra
from omegaconf import DictConfig

from base_scene import construct_base_scene
from robot import setup_robot_manipulator

from dm_robotics.moma import base_task
from dm_robotics.moma import action_spaces
from dm_robotics.moma import entity_initializer
from dm_robotics.geometry import pose_distribution
from dm_robotics.moma import subtask_env_builder
from dm_robotics import agentflow as af

from dm_robotics.agentflow.preprocessors import observation_transforms

import numpy as np


@hydra.main(version_base=None, config_path="./config", config_name="scene")
def construct_task_env(cfg: DictConfig):
    """Construct a RAVENS task environment."""
    # construct base scene
    scene_components = construct_base_scene(cfg)

    # set up robot
    robot = setup_robot_manipulator(cfg, scene_components)

    # create base task
    task = base_task.BaseTask(
        task_name="test",
        arena=scene_components["arena"],
        robots=[robot],
        props=scene_components["props"],
        extra_sensors=scene_components["extra_sensors"],
        extra_effectors=[],
        control_timestep=0.1,
        scene_initializer=lambda _: None,
        episode_initializer=lambda _: None,
    )

    # action space

    # define action spaces
    parent_action_spec = task.effectors_action_spec(scene_components["physics"])

    joint_action_space = action_spaces.ArmJointActionSpace(
        af.prefix_slicer(parent_action_spec, robot.arm_effector.prefix)
    )
    gripper_action_space = action_spaces.GripperActionSpace(
        af.prefix_slicer(parent_action_spec, robot.gripper_effector.prefix)
    )
    combined_action_space = af.CompositeActionSpace([joint_action_space, gripper_action_space])

    # initialization logic

    initializers = []

    # initialize robot based on position of end effector
    gripper_pose_dist = pose_distribution.UniformPoseDistribution(
        min_pose_bounds=np.array([0.5, -0.1, 0.1, 0.75 * np.pi, -0.25 * np.pi, -0.5 * np.pi]),
        max_pose_bounds=np.array([0.7, 0.1, 0.2, 1.25 * np.pi, 0.25 * np.pi, 0.5 * np.pi]),
    )
    initialize_arm = entity_initializer.PoseInitializer(
        initializer_fn=robot.position_gripper,
        pose_sampler=gripper_pose_dist.sample_pose,
    )
    initializers.append(initialize_arm)

    # prop initializers
    for prop in scene_components["props"]:
        prop_pose_dist = pose_distribution.UniformPoseDistribution(
            min_pose_bounds=np.array([0.3, -0.35, 0.05, 0.0, 0.0, -np.pi]),
            max_pose_bounds=np.array([0.9, 0.35, 0.05, 0.0, 0.0, np.pi]),
        )
        initialize_prop = entity_initializer.PoseInitializer(
            initializer_fn=prop.set_pose, pose_sampler=prop_pose_dist.sample_pose
        )
        initializers.append(initialize_prop)

    # combine robot and prop initializers
    entities_initializer = entity_initializer.TaskEntitiesInitializer(initializers)

    # timestep logic
    preprocessors = []

    # cast observations to float32
    preprocessors.append(observation_transforms.CastPreprocessor(dtype=np.float32))

    # add observables
    sensors = []
    for sensor in robot.sensors:
        sensors += sensor.observables.keys()

    for sensor in scene_components["extra_sensors"]:
        sensors += sensor.observables.keys()

    preprocessors.append(
        observation_transforms.RetainObservations(
            [
                "franka_emika_panda_joint_position",
                "franka_emika_panda_joint_velocity",
                "robotiq_2f85_tcp",
            ],
            raise_on_missing=True,
        )
    )

    # construct the environment
    task.set_episode_initializer(entities_initializer)

    env_builder = subtask_env_builder.SubtaskEnvBuilder()
    env_builder.set_task(task)
    env_builder.set_action_space(combined_action_space)
    for preprocessor in preprocessors:
        env_builder.add_preprocessor(preprocessor)
    task_environment = env_builder.build()

    return task_environment


if __name__ == "__main__":
    construct_task_env()
