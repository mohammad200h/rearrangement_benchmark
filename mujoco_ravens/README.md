# Overview

    ├──── config                # config to define task environment
    ├──── models                # MoMa model definitions
    ├──── mujoco_menagerie      # high-quality models for MuJoCo physics engine
    ├──── ros2_ws               # ROS 2 workspace for both real and simulated experiments
    ├── base_scene.py           # definition of base scene (arena, robots, props)
    ├── cameras.py              # utilities for rendering scene
    ├── props.py                # prop creation and addition to scene
    ├── robot.py                # robot creation and addition to scene
    ├── task.py                 # task logic and creation (MoMa)
    ├── rearrangement_task.py   # high-level API for rearrangement tasks
