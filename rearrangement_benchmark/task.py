"""A script to setup RAVENS tasks."""

# config
from hydra import compose, initialize
from omegaconf import DictConfig

from rearrangement_benchmark.env_components.base_scene import construct_base_scene
from rearrangement_benchmark.env_components.robot import setup_robot_manipulator

from dm_robotics.moma import action_spaces
from dm_robotics.moma import base_task
from dm_robotics.moma import entity_initializer
from dm_robotics.moma import moma_option
from dm_robotics.geometry import pose_distribution
from dm_robotics.moma import subtask_env_builder
from dm_robotics import agentflow as af

from dm_robotics.agentflow.preprocessors import observation_transforms

import numpy as np

initialize(version_base=None, config_path="./config", job_name="default_config")
DEFAULT_CONFIG = compose(config_name="scene")


def construct_task_env(cfg: DictConfig = DEFAULT_CONFIG):
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
        min_pose_bounds=np.array(cfg.task.initializers.gripper.min_pose),
        max_pose_bounds=np.array(cfg.task.initializers.gripper.max_pose),
    )
    initialize_arm = entity_initializer.PoseInitializer(
        initializer_fn=robot.position_gripper,
        pose_sampler=gripper_pose_dist.sample_pose,
    )
    initializers.append(initialize_arm)

    # prop initializers
    for prop in scene_components["props"]:
        prop_pose_dist = pose_distribution.UniformPoseDistribution(
            min_pose_bounds=np.array(cfg.task.initializers.workspace.min_pose),
            max_pose_bounds=np.array(cfg.task.initializers.workspace.max_pose),
        )

        def position_prop(physics, pos, quat, random_state):
            del random_state
            prop.set_pose(physics, pos, quat)
    
        initialize_prop = entity_initializer.PoseInitializer(
            initializer_fn=position_prop, pose_sampler=prop_pose_dist.sample_pose
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
            sensors,
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

    # for the reset option we need to ensure values are within ranges of robot joints
    base_env = env_builder.build_base_env()
    parent_action_spec = task.effectors_action_spec(physics=base_env.physics, effectors=task.effectors)

    # https://github.com/google-deepmind/dm_robotics/blob/e4631a91363b3f7b05bc848e818ad6485292f110/py/agentflow/options/basic_options.py#L103
    # TODO: fix this logic based on selected control interface (e.g. position, velocity, torque)
    # currently choosing zeros for position control results in a validation error
    if cfg.robots.arm.actuator_config.type == "general":  # in this case general is position control
        noop_action = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0], dtype=np.float32)
    elif cfg.robots.arm.actuator_config.type == "motor":
        noop_action = af.spec_utils.zeros(parent_action_spec)
    else:
        raise ValueError(f"Unsupported actuator type: {cfg.robots.arm.actuator_config.type}")

    delegate = af.FixedOp(noop_action, name="NoOp")
    reset_option = moma_option.MomaOption(
        physics_getter=lambda: base_env.physics,
        effectors=task.effectors,
        delegate=delegate,
    )
    env_builder.set_reset_option(reset_option)

    task_environment = env_builder.build()

    return task_environment, cfg


if __name__ == "__main__":
    import PIL

    # test the environment
    task_env, config = construct_task_env()
    obs = task_env.reset()
    front_camera = obs[3]["front_camera_rgb_img"].astype(np.uint8)
    PIL.Image.fromarray(front_camera).show()
    overhead_camera = obs[3]["overhead_camera_rgb_img"].astype(np.uint8)
    PIL.Image.fromarray(overhead_camera).show()
    task_env.close()
