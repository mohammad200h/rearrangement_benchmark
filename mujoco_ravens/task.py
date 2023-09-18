"""A script to setup RAVENS tasks."""

# config
import hydra
from omegaconf import DictConfig

from base_scene import construct_base_scene
from robot import setup_robot_manipulator

@hydra.main(version_base=None, config_path="./config", config_name="scene")
def construct_task_env(cfg: DictConfig):
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
   
    ## action space

    # define action spaces
    parent_action_spec = task.effectors_action_spec(physics)
   
    joint_action_space = action_spaces.ArmJointActionSpace(
           af.prefix_slicer(parent_action_spec, arm_hardware_interface.prefix)
           )
    gripper_action_space = action_spaces.GripperActionSpace(
           af.prefix_slicer(parent_action_spec, gripper_hardware_interface.prefix)
           )
    combined_action_space = af.CompositeActionSpace(
           [joint_action_space, gripper_action_space]
           )

    ## initialization logic 

    initializers = []
   
    # initialize robot based on position of end effector
    gripper_pose_dist = pose_distribution.UniformPoseDistribution(
           min_pose_bounds=np.array([0.5, -0.1, 0.1,
                             0.75 * np.pi, -0.25 * np.pi, -0.5 * np.pi]),
           max_pose_bounds=np.array([0.7, 0.1, 0.2,
                             1.25 * np.pi, 0.25 * np.pi, 0.5 * np.pi])
    )
    initialize_arm = entity_initializer.PoseInitializer(
           initializer_fn = robot.position_gripper,
           pose_sampler = gripper_pose_dist.sample_pose,
           )
    initializers.append(initialize_arm)

    # prop initializers
    for prop in props:
        prop_pose_dist = pose_distribution.UniformPoseDistribution(
               min_pose_bounds=np.array([0.3, -0.35, 0.05, 0.0, 0.0, -np.pi]),
               max_pose_bounds=np.array([0.9, 0.35, 0.05, 0.0, 0.0, np.pi]))
        initialize_prop = entity_initializer.PoseInitializer(
               initializer_fn = prop.set_pose, 
               pose_sampler = prop_pose_dist.sample_pose
               )
        initializers.append(initialize_prop)
   
    # combine robot and prop initializers
    entities_initializer = entity_initializer.TaskEntitiesInitializer(initializers)

   ## timestep logic


    
if __name__=="__main__":
    construct_task_env()
