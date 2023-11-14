"""A high-level task API for rearrangement tasks that leverage motion planning."""
import os
import sys
import subprocess

from ros_start_docker import (
    start_control_server,
    shutdown_control_server,
    start_motion_planning_prerequisites,
    shutdown_motion_planning_prerequisites,
)

from ament_index_python.packages import get_package_share_directory

# ros 2 client library
import rclpy
from rclpy.node import Node
from rclpy.logging import get_logger

# moveit python library
from moveit_configs_utils import MoveItConfigsBuilder

from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
    PlanRequestParameters,
    #    MultiPipelinePlanRequestParameters,
)


class RearrangementTask(Node):
    """A high-level task API for rearrangement tasks that leverage motion planning."""

    def __init__(self, config=None):
        """Initialize the rearrangement task."""
        super().__init__("rearrangement_task")

        self.config = config

    def __enter__(self):
        """Reset the task environment on entry of context."""
        # sets up ros 2 nodes
        self._start_ros()

        # reset the task environment

        # wait for the task environment to be ready
        import time

        time.sleep(15)

    def __exit__(self, type, value, traceback):
        """Shutdown docker containers on exit of context."""
        self._shutdown_ros()

    # def __del__(self):
    #    """Close the task environment on instance deletion."""

    def _start_ros(self):
        """Start ROS 2 client library."""
        self.logger = get_logger("rearrangement_task")
        # if self.config.task.use_simulation:
        try:
            # TODO: move to docker container
            # simulation client
            subprocess.Popen(["python", "ros_mujoco_client.py"])

            # control server
            start_control_server()

            # motion planning
            start_motion_planning_prerequisites()  # static transforms, rviz, etc.

            # moveit configuration
            self.moveit_config = (
                MoveItConfigsBuilder(
                    robot_name="franka_emika_panda",
                    package_name="franka_robotiq_moveit_config",
                )
                .robot_description(
                    file_path=get_package_share_directory("franka_robotiq_description") + "/urdf/robot.urdf.xacro",
                    mappings={"use_fake_hardware": "true", "robot_ip": "192.168.106.39", "robotiq_gripper": "true"},
                )
                .robot_description_semantic("config/panda.srdf.xacro")
                .trajectory_execution("config/moveit_controllers.yaml")
                .moveit_cpp(
                    file_path=os.path.join(
                        get_package_share_directory("rearrangements"), "config", "planning_pipelines.yaml"
                    )
                )
                .to_moveit_configs()
            ).to_dict()

            # moveit client
            self.ROBOT = MoveItPy(node_name="moveit_py", config_dict=self.moveit_config)
            self.ROBOT_ARM = self.ROBOT.get_planning_component("panda_arm")

        except Exception as e:
            print(e)
            sys.exit(1)

    def _shutdown_ros(self):
        """Shutdown ROS 2 client library."""
        # shutdown control server + simulation client
        shutdown_control_server()
        rclpy.shutdown()
        self.control_client_thread.join()
        self.control_client.destroy_node()

        # shutdown motion planning
        shutdown_motion_planning_prerequisites()

    def _plan_and_execute(
        self,
        robot,
        planning_component,
        logger,
        single_plan_parameters=None,
        multi_plan_parameters=None,
        sleep_time=0.0,
    ):
        """Plan and execute a motion."""
        # plan to goal
        logger.info("Planning trajectory")
        if multi_plan_parameters is not None:
            plan_result = planning_component.plan(multi_plan_parameters=multi_plan_parameters)
        elif single_plan_parameters is not None:
            plan_result = planning_component.plan(single_plan_parameters=single_plan_parameters)
        else:
            plan_result = planning_component.plan()

        # execute the plan
        if plan_result:
            logger.info("Executing plan")
            robot_trajectory = plan_result.trajectory
            robot.execute(robot_trajectory, controllers=[])
        else:
            logger.error("Planning failed")

    def reset_robot(self):
        """Reset the robot to a known configuration."""
        # fix starting joint configuration
        # start_joint_config = {
        #        "panda_joint1": 0.0,
        #        "panda_joint2": -0.785,
        #        "panda_joint3": 0.0,
        #        "panda_joint4": -2.356,
        #        "panda_joint5": 0.0,
        #        "panda_joint6": 1.571,
        #        "panda_joint7": 0.785,
        #        }
        # robot_state.joint_positions = start_joint_config

        # set start to current sim state (should be read from robot_state_publisher)
        self.ROBOT_ARM.set_start_state_to_current_state()
        start_state = self.ROBOT_ARM.get_start_state()
        print(f"start_state: {start_state.joint_positions}")

        # set random target
        robot_model = self.ROBOT.get_robot_model()
        robot_state = RobotState(robot_model)
        robot_state.set_to_random_positions()
        self.ROBOT_ARM.set_goal_state(robot_state=robot_state)

        print(f"robot_state: {robot_state.joint_positions}")

        # perform motion planning
        plan_params = PlanRequestParameters(self.ROBOT, "ompl_rrtc")
        self._plan_and_execute(
            self.ROBOT, self.ROBOT_ARM, self.logger, sleep_time=3.0, single_plan_parameters=plan_params
        )

    def transporter_pick(self, pixel_coords):
        """
        Pick up an object.

        This method leverages MoveIt to plan and execute a pick motion.
        ROS 2 control manages sending commands to hardware interfaces.
        """
        # map pixel coordinates to real world coordinates

        # plan and execute to pregrasp pose

        # plan and execute to grasp pose

        # grasp object

        # plan and execute to preplace pose
        raise NotImplementedError

    def transporter_place(self, pixel_coords):
        """
        Place an object.

        This method leverages MoveIt to plan and execute a place motion.
        ROS 2 control manages sending commands to hardware interfaces.
        """
        # map pixel coordinates to real world coordinates

        # plan and execute to place pose

        # release object

        # plan and execute to pregrasp pose
        raise NotImplementedError

    # def render(self):
    #    """Render the task environment."""
    #    pixels = self._task_env.physics.render()
    #    PIL.Image.fromarray(pixels).show()


if __name__ == "__main__":
    rclpy.init()
    task = RearrangementTask(None)
    with task:
        task.reset_robot()
        print("sleeping")
        import time

        time.sleep(3600)
