"""A high-level task API for rearrangement tasks that leverage motion planning."""

import numpy as np
from ikpy.chain import Chain

import PIL

from task import construct_task_env

URDF_PATH = "./robot.urdf"

class RearrangementTask(object):
    """A high-level API for performing rearrangement tasks."""

    def __init__(self):
        """Initializes a rearrangement task."""
        self._sim, self.config = construct_task_env()
        self.ee_chain = Chain.from_urdf_file(URDF_PATH, base_elements=["panda_link0"]) # TODO: read from config
        
        # current status of the robot
        self.gripper_status = "open"
        self.joint_angles = None 
        self.obs = None

    def __del__(self):
        """Cleans up the task."""
        self._sim.close()

    def update_internal_vars(self, obs):
        """Updates internal variables based on the observation."""
        self.joint_angles = obs[3]["franka_emika_panda_joint_pos"]
        self.obs = obs
    
    def reset(self):
        """Resets the task."""
        obs = self._sim.reset()
        self.update_internal_vars(obs)
        return obs

    def move_eef(self, target, max_iters=100):
        """Moves the end effector to the target position, while maintaining upright orientation.

        Args:
            target: A 3D target position.
        """
        # first get joint target using inverse kinematics
        joint_target = self.ee_chain.inverse_kinematics(
            target_position = target,
            target_orientation = [0, 0, -1],
            orientation_mode = "Z",
            initial_position = np.array([0.0, 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0]),
            )[1:-1]
    
        # include the gripper status in the command
        if self.gripper_status == "open":
            joint_target = np.concatenate([joint_target, np.array([-255.0])])
        elif self.gripper_status == "closed":
            joint_target = np.concatenate([joint_target, np.array([255.0])])
        else:
            raise ValueError("Invalid gripper status: {}".format(self.gripper_status))

        # step the sim env until within a certain threshold of the target position
        def check_target_reached(target, joint_vals, position_threshold=0.01, orientation_threshold=0.01):
            # perform fk with current joint values
            joint_vals = np.concatenate([np.zeros(1), joint_vals, np.zeros(1)]) # add zeros for dummy joints
            ee_pos = self.ee_chain.forward_kinematics(
                joints = joint_vals,
                full_kinematics = False,
                )[:3, 3]

            # check cartesian position
            if np.linalg.norm(ee_pos - target) > position_threshold:
                return False

            # TODO: add orientation check
            # check orientation
            
            return True

        iters = 0
        target_reached = False
        while (not target_reached) or (iters < max_iters):
            iters += 1
            obs = self._sim.step(joint_target)
            joint_vals = obs[3]["franka_emika_panda_joint_pos"]
            target_reached = check_target_reached(target, joint_vals) 
        
        self.update_internal_vars(obs)
        if target_reached:
            return True, obs

        return False, None
        

    def open_gripper(self):
        """Opens the gripper."""
        self.gripper_status = "open"
        joint_target = np.concatenate([self.joint_angles, np.array([-255.0])])
        for i in range(10):
            obs = self._sim.step(joint_target)
        self.update_internal_vars(obs)
        return obs

    def close_gripper(self):
        """Closes the gripper."""
        self.gripper_status = "closed"
        joint_target = np.concatenate([self.joint_angles, np.array([255.0])])
        for i in range(10):
            obs = self._sim.step(joint_target)
        self.update_internal_vars(obs)
        return obs

    def pick(self):
        """Picks up an object."""
        pass

    def place(self):
        """Places an object."""
        pass

    def transporter_pick(self):
        """Picks up an object using the transporter."""
        pass

    def transporter_place(self):
        """Places an object using the transporter."""
        pass
    
    def display_cameras(self):
        """Displays the current camera images."""
        # TODO: generalize this to any number of cameras
        front_camera = obs[3]["front_camera_rgb_img"].astype(np.uint8)
        PIL.Image.fromarray(front_camera).show()
        overhead_camera = obs[3]["overhead_camera_rgb_img"].astype(np.uint8)
        PIL.Image.fromarray(overhead_camera).show()
        left_camera = obs[3]["left_camera_rgb_img"].astype(np.uint8)
        PIL.Image.fromarray(left_camera).show()
        right_camera = obs[3]["right_camera_rgb_img"].astype(np.uint8)
        PIL.Image.fromarray(right_camera).show()


if __name__=="__main__":
    task = RearrangementTask()
    obs = task.reset()
    task.display_cameras()
    
    # use move_eef function
    status, obs = task.move_eef(np.array([0.6, 0.25, 0.4]))
    obs = task.open_gripper()
    obs = task.close_gripper()

    # show result of commands
    if status:
        task.display_cameras()
