# Overview

    ├──── config                                       # config to define task environment
    ├──── models                                       # MoMa model definitions
    ├──── mujoco_menagerie                             # high-quality models for MuJoCo physics engine
    ├──── ros2_robotics_research_toolkit               # ROS 2 workspace for both real and simulated experiments
    ├── base_scene.py                                  # definition of base scene (arena, robots, props)
    ├── cameras.py                                     # utilities for rendering scene
    ├── props.py                                       # prop creation and addition to scene
    ├── rearrangement_task.py                          # high-level API for rearrangement tasks
    ├── robot.py                                       # robot creation and addition to scene
    ├── robot.urdf                                     # URDF used by IK library (will be moved in future)
    ├── ros2_mujoco_client.py                          # ROS 2 based client for stepping MuJoCo sim environment
    ├── ros2_rearrangement.py                          # ROS 2 reliant high-level API for rearrangement tasks
    ├── ros2_start_docker.py                           # basic utilities for starting docker from Python
    ├── task.py                                        # task logic and creation (MoMa)
