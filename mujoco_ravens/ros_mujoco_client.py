"""Client for setting controls and stepping simulation environment."""
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from task import construct_task_env


class MuJoCoSimClient(Node):
    """A ROS 2 client for managing a mujoco simulation instance."""

    def __init__(self, config=None):
        """Initialize the control client."""
        super().__init__("mujoco_control_client")
        if config is None:
            self.sim, config = construct_task_env()
        else:
            self.sim, _ = construct_task_env(config)

        self.config = config

        # reset the simulation
        self.sim.reset()

        # create publisher for joint state
        self.state_publisher = self.create_publisher(
            JointState,
            "/topic_based_joint_states",
            10,
        )
        # publish initial joint state
        self._publish_joint_state()

        # create subscription for joint commands
        self.control = self.create_subscription(
            JointState,
            "/topic_based_joint_commands",
            self._set_joint_command,
            10,
        )

        # Define timer for publishing joint state
        timer_frequency = 100.0
        self.timer = self.create_timer(1.0 / timer_frequency, self.timer_callback)

    def timer_callback(self):
        """Publish joint state."""
        self._publish_joint_state()

    def _set_joint_command(self, control_msg):
        """Send a joint command to the robot."""
        # convert control message to numpy array
        position_command = np.array(control_msg.position)
        gripper_command = np.array([0.0])  # for now set to zero
        control_command = np.concatenate((position_command, gripper_command))

        # step simulation with control command
        if position_command.shape[0] == 7:
            self.get_logger().info("Stepping simulation with control command: {}".format(control_command))
            self.sim.step(control_command)

    def _publish_joint_state(self):
        """Publish joint state from mujoco simulation."""
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()

        # get data from simulation
        joint_names = [
            self.sim.physics.model.id2name(i, "joint")
            for i in range(self.sim.physics.model.njnt)
            if any(keyword in self.sim.physics.model.id2name(i, "joint") for keyword in ["nohand/joint"])
        ]
        joint_positions = self.sim.physics.named.data.qpos[joint_names]
        joint_velocities = self.sim.physics.named.data.qvel[joint_names]
        joint_names = ["panda_" + name.split("/")[-1] for name in joint_names]

        # publish data to topic
        joint_state_msg.name = joint_names
        joint_state_msg.position = list(joint_positions)
        joint_state_msg.velocity = list(joint_velocities)
        self.state_publisher.publish(joint_state_msg)

    # TODO: move to publisher
    # def props(self) -> dict:
    #    """
    #    Gets domain model.

    #    The domain model is a dictionary of objects and their properties.
    #    """
    #    # get prop object names
    #    prop_names = [
    #        self._task_env.physics.model.id2name(i, "geom")
    #        for i in range(self._task_env.physics.model.ngeom)
    #        if any(keyword in self._task_env.physics.model.id2name(i, "geom") for keyword in self.shapes)
    #    ]
    #    prop_ids = [self._task_env.physics.model.name2id(name, "geom") for name in prop_names]

    #    # get object information
    #    prop_positions = self._task_env.physics.named.data.geom_xpos[prop_names]
    #    prop_orientations = self._task_env.physics.named.data.geom_xmat[prop_names]
    #    prop_orientations = [mat_to_quat(mat.reshape((3, 3))) for mat in prop_orientations]
    #    prop_rgba = self._task_env.physics.named.model.geom_rgba[prop_names]
    #    prop_names = [name.split("/")[0] for name in prop_names]

    #    # get object bounding box information
    #    def get_bbox(prop_id, segmentation_map):
    #        """Get the bounding box of an object (PASCAL VOC)."""
    #        prop_coords = np.argwhere(segmentation_map[:, :, 0] == prop_id)
    #        bbox_corners = np.array(
    #            [
    #                np.min(prop_coords[:, 0]),
    #                np.min(prop_coords[:, 1]),
    #                np.max(prop_coords[:, 0]),
    #                np.max(prop_coords[:, 1]),
    #            ]
    #        )

    #        return bbox_corners

    #    # TODO: consider vectorizing this
    #    segmentation_map = self._task_env.physics.render(segmentation=True)
    #    prop_bbox = []
    #    for idx in prop_ids:
    #        bbox = get_bbox(idx, segmentation_map)
    #        prop_bbox.append(bbox)

    #    # create a dictionary with all the data
    #    prop_info = {
    #        prop_names[i]: {
    #            "position": prop_positions[i],
    #            "orientation": prop_orientations[i],
    #            "rgba": prop_rgba[i],
    #            "bbox": prop_bbox[i],
    #        }
    #        for i in range(len(prop_names))
    #    }

    #    return prop_info


if __name__ == "__main__":
    rclpy.init()
    mujoco_sim_client = MuJoCoSimClient()
    rclpy.spin(mujoco_sim_client)
    rclpy.shutdown()
