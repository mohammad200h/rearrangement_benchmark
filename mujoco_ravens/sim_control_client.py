"""Client for setting controls and stepping simulation environment."""
import numpy as np

from rclpy.node import Node
from sensor_msgs import JointState

from dm_robotics.moma.subtask_env import SubTaskEnvironment


class MuJoCoControlClient(Node):
    """A ROS 2 client library for sending control commands to mujoco simulation of robot."""

    def __init__(self, sim: SubTaskEnvironment):
        """Initialize the control client."""
        super().__init__("mujoco_control_client")
        self.sim = sim

        # create publisher for joint state
        self.state_publisher = self.create_publisher(
            JointState,
            "/topic_based_joint_states",
            10,
        )

        # create subscription for joint commands
        self.control = self.create_subscription(
            JointState,
            "/topic_based_joint_commands",
            self._set_joint_command,
            10,
        )

    def _set_joint_command(self, control_msg):
        """Send a joint command to the robot."""
        # convert control message to numpy array
        torque_command = np.array(control_msg.effort)
        gripper_command = np.array([0.0])  # for now set to zero
        control_command = np.concatenate((torque_command, gripper_command))

        # step simulation with control command
        self.sim.step(control_command)

        # publish the updated joint state
        self._publish_joint_state()

    def _publish_joint_state(self):
        """Publish joint state from mujoco simulation."""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.sim.joint_names
        joint_state_msg.position = self.sim.joint_positions
        joint_state_msg.velocity = self.sim.joint_velocities
        joint_state_msg.effort = self.sim.joint_torques
        self.state_publisher.publish(joint_state_msg)
