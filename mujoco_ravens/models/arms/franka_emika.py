"""Franka Emika Panda Robot Arm."""

from typing import List

import numpy as np

from dm_control import mjcf
from dm_robotics.moma.models import types
from dm_robotics.moma.models.robots.robot_arms import robot_arm


class FER(robot_arm.RobotArm):
    """Franka Emika Panda Robot Arm."""

    _joints: List[types.MjcfElement]
    _actuators: List[types.MjcfElement]
    _fer_root: mjcf.RootElement

    def _build(self):
        self._fer_root = mjcf.from_path("./mujoco_menagerie/franka_emika_panda/panda_nohand.xml")
        self._joints = self._fer_root.find_all("joint")
        self._actuators = self._fer_root.find_all("actuator")
        self._wrist_site = self._fer_root.find("site", "wrist_site")
        self._attachment_site = self._fer_root.find("site", "attachment_site")

    @property
    def joints(self):
        """Returns a list of joints in the robot."""
        return self._joints

    @property
    def actuators(self):
        """Returns a list of actuators in the robot."""
        return self._actuators

    @property
    def mjcf_model(self):
        """Returns the MJCF model for the robot."""
        return self._fer_root

    @property
    def name(self):
        """Returns the name of the robot."""
        return "franka_emika_panda"

    @property
    def wrist_site(self):
        """Returns the wrist site."""
        return self._wrist_site

    @property
    def attachment_site(self):
        """Returns the attachment site."""
        return self._attachment_site

    def set_joint_angles(self, physics: mjcf.Physics, qpos: np.ndarray) -> None:
        """Set the joint angles of the robot."""
        physics.bind(self._joints).qpos = qpos
