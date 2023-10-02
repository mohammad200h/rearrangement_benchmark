"""Client for setting controls and stepping simulation environment."""

import rclpy
from rclpy.node import Node


class ControlClient(Node):
    """A ROS 2 client library for sending control commands to mujoco simulation of robot."""

    def __init__(self):
        """Initialize the control client."""
        super().__init__("control_client")

        # subscribe to joint state topic to set joint commands
        self.control = self.create_subscription(
            # message_type,
            "/topic_based_joint_commands",
            self._set_joint_command,
            10,
        )

        # publish joint state from mujoco simulation
        self.state_publisher = self.create_publisher(
            # message_type,
            "/topic_based_joint_states",
            10,
        )
        self.timer = self.create_timer(0.1, self._publish_joint_state)

    def _set_joint_command(self, control_msg):
        """Send a joint command to the robot."""
        # convert control message to numpy array

        # set joint command in mujoco simulation

        raise NotImplementedError

    def _publish_joint_state(self):
        """Publish joint state from mujoco simulation."""
        # get joint state from mujoco simulation
        # joint_state_msg = message_type()

        # publish joint state
        # self.state_publisher.publish(joint_state_msg)
        raise NotImplementedError


if __name__ == "__main__":
    rclpy.init()
    control_client = ControlClient()
    rclpy.spin(control_client)
    control_client.destroy_node()
    rclpy.shutdown()
