"""A high-level task API for rearrangement tasks that leverage motion planning."""

import numpy as np
import mujoco
from mujoco import viewer
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
from dm_robotics.transformations import transformations as tr
from dm_robotics.transformations.transformations import mat_to_quat, quat_to_mat, quat_to_euler

import PIL

from rearrangement_benchmark.task import construct_task_env


URDF_PATH = "./models/arms/robot.urdf"

class RearrangementTask(object):
    """A high-level API for performing rearrangement tasks."""

    def __init__(self, cfg=None, viewer=True):
        """Initializes a rearrangement task."""
        # if a config is provided overwrite the default config
        if cfg is not None:
            self._sim, self.config = construct_task_env(cfg)
        else:
            self._sim, self.config = construct_task_env()
        self.viewer_flag = viewer
        if self.viewer_flag:
            self.viewer = None
        self.shapes = self.config.props.shapes
        self.ee_chain = Chain.from_urdf_file(URDF_PATH, base_elements=["panda_link0"]) # TODO: read from config

        # current status of the robot
        self.gripper_status = "open"
        self.joint_angles = None 
        self.joint_velocities = None
        self.obs = None

    def __del__(self):
        """Cleans up the task."""
        if self.viewer_flag:
            del self.viewer
        self._sim.close()

    def update_internal_vars(self, obs):
        """Updates internal variables based on the observation."""
        self.obs = obs
        self.joint_angles = self.obs[3]["franka_emika_panda_joint_pos"]
        self.joint_velocities = self.obs[3]["franka_emika_panda_joint_vel"]
    
    def reset(self):
        """Resets the task."""
        obs = self._sim.reset()
        self.update_internal_vars(obs)
        if self.viewer_flag:
            self.viewer = viewer.launch_passive(self._sim.physics.model._model, self._sim.physics.data._data)
        return obs
    
    def pixel_2_world(self, camera_name, coords):
        """Returns the world coordinates for a given pixel."""
        width, height = [480, 640] # TODO: read from config

        # get camera parameters
        # Note: MoMa employs opencv camera convention where +z faces the scene
        # this is different from mujoco where -z faces the scene
        # this was confusing on first pass over both APIs
        intrinsics = self.obs[3][camera_name + "_intrinsics"]
        
        # get camera position and orientation (from MoMa's perspective)
        pos = self.obs[3][camera_name + "_pos"]
        quat = self.obs[3][camera_name + "_quat"]
        depth_val = self.obs[3][camera_name + "_depth_img"][coords[1], coords[0]]
        
        # convert to camera frame coordinates
        image_coords = np.expand_dims(np.concatenate([coords, np.ones(1)]), axis=-1)
        camera_coords =  np.linalg.inv(intrinsics[:3,:3]) @ image_coords
        camera_coords = np.squeeze(camera_coords * depth_val)

        # convert camera coordinates to world coordinates
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = quat_to_mat(quat)[:3,:3].T
        world_to_camera[:3, 3] = -quat_to_mat(quat)[:3,:3].T @ pos
        world_coords = world_to_camera @ np.concatenate([np.squeeze(camera_coords), np.ones(1)])
        world_coords = world_coords[:3] / world_coords[3]

        return world_coords

    def world_2_pixel(self, camera_name, coords):
        """Returns the pixel coordinates for a given world coordinate."""
        intrinsics = self.obs[3][camera_name + "_intrinsics"]
        pos = self.obs[3][camera_name + "_pos"]
        quat = self.obs[3][camera_name + "_quat"]

        # convert world coordinates to camera coordinates
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = quat_to_mat(quat)[:3,:3]
        camera_to_world[:3, 3] = pos
        camera_coords = camera_to_world @ np.concatenate([coords, np.ones(1)])
        camera_coords = camera_coords[:3] / camera_coords[3]

        # convert camera coordinates to image coordinates
        image_coords = intrinsics[:3, :3] @ camera_coords
        image_coords = image_coords[:2] / image_coords[2]

        return image_coords

    def move_eef(self, target_pose, target_orientation, duration=5.0):
        """Moves the end effector to the target position, while maintaining upright orientation.

        Args:
            target: A 3D target position.
        """
        target_reached = False
       
        # Generate joint target via IK
        current_joint_angles = self.joint_angles
        current_joint_angles = np.concatenate([np.array([0.0]), current_joint_angles, np.array([0.0])]) # required format for ikpy

        target_quat = mat_to_quat(target_orientation)
        joint_target = self.ee_chain.inverse_kinematics(
            target_position = target_pose,
            target_orientation = target_orientation,
            orientation_mode = "all",
            initial_position = current_joint_angles,
            )[1:-1]

        start_time = self._sim.physics.data.time
        while (not target_reached) and (self._sim.physics.data.time - start_time < duration):
            # get current joint values
            current_joint_position = self._sim.physics.data.qpos[:7]
            current_joint_velocity = self._sim.physics.data.qvel[:7]

            # calculate joint torques
            torques = self.pd_control(
                target_joint_angles = joint_target,
                target_joint_velocities = np.zeros(7),
                current_joint_angles = current_joint_position,
                current_joint_velocities = current_joint_velocity,
                    )

            # include the gripper status in the command
            if self.gripper_status == "open":
                command = np.concatenate([torques, np.array([-255.0])])
            elif self.gripper_status == "closed":
                command = np.concatenate([torques, np.array([255.0])])
            else:
                raise ValueError("Invalid gripper status: {}".format(self.gripper_status))
            
            # step simulation env
            obs = self._sim.step(command)
            self.update_internal_vars(obs)
            target_reached = self.check_target_reached(target_pose, target_quat, self.joint_angles)
            if self.viewer_flag:
                self.viewer.sync()
       
        if target_reached:
            return True
        else:
            return False

    def check_target_reached(self, target_pose, target_quat, joint_vals, position_threshold=0.01, orientation_threshold=0.5):
        # calculate end effector pose
        joint_vals = np.concatenate([np.zeros(1), joint_vals, np.zeros(1)]) # add zeros for dummy joints
        ee_pose = self.ee_chain.forward_kinematics(
            joints = joint_vals,
            full_kinematics = False,
            )
        ee_pos = ee_pose[:3, 3]
        ee_quat = mat_to_quat(ee_pose[:3, :3])

        # calculate distance to target
        linear_dist = np.linalg.norm(ee_pos - target_pose)
        angular_dist = tr.quat_dist(target_quat/np.linalg.norm(target_quat), ee_quat/np.linalg.norm(ee_quat))
        
        # check if target reached
        if (linear_dist > position_threshold) or (angular_dist > orientation_threshold):
            return False
        else:
            return True
        
    # TODO: move to independent class
    def pd_control(self, target_joint_angles, target_joint_velocities, current_joint_angles, current_joint_velocities, kp=600, kd=400):
        # calculate target acceleration
        target_acceleration = kp * (target_joint_angles - current_joint_angles) + kd * (target_joint_velocities - current_joint_velocities)
        prev = current_joint_velocities.copy()
        
        # apply inverse dynamics to get joint torques
        self._sim.physics.data.qacc[:7] = target_acceleration
        mujoco.mj_inverse(self._sim.physics.model._model, self._sim.physics.data._data)
        sol = self._sim.physics.data.qfrc_inverse[:7].copy()
        self._sim.physics.data.qacc[:7] = prev

        return sol

    def open_gripper(self, duration=2.0):
        """Opens the gripper."""
        self.gripper_status = "open"
        joint_target = np.concatenate([self.joint_angles, np.array([-255.0])])
        start_time = self._sim.physics.data.time
        while self._sim.physics.data.time - start_time < duration:
            obs = self._sim.step(joint_target)
            self.viewer.sync()
        self.update_internal_vars(obs)
        return obs

    def close_gripper(self, duration=2.0):
        """Closes the gripper."""
        self.gripper_status = "closed"
        joint_target = self.joint_angles.copy()
        start_time = self._sim.physics.data.time
        while self._sim.physics.data.time - start_time < duration:
            command = self.pd_control(
                target_joint_angles = joint_target,
                target_joint_velocities = np.zeros(7),
                current_joint_angles = self.joint_angles,
                current_joint_velocities = self.joint_velocities,
                )
            command = np.concatenate([command, np.array([255.0])])
            obs = self._sim.step(command)
            self.update_internal_vars(obs)
            self.viewer.sync()
        return obs

    def pick(self, object_name):
        """Picks up an object."""
        obj_pose = self.props[object_name]["position"]
        obj_quat = self.props[object_name]["orientation"]
        
        # generate grasp poses for object
        pre_grasp_pose = np.array([obj_pose[0], obj_pose[1], 0.6])
        grasp_pose = np.copy(pre_grasp_pose)
        grasp_pose[2] = 0.2

        # generate grasp orientation
        obj_rot = R.from_quat(obj_quat)
        obj_rot_mat = obj_rot.as_matrix()
        obj_rot_z = np.rad2deg(np.arctan2(obj_rot_mat[1,0], obj_rot_mat[0,0]))
        # double check the -45 term
        grasp_mat = R.from_euler('xyz', [0, 180, obj_rot_z-45], degrees=True).as_matrix()


        # display side rgb
        rgb = self.obs[3]["left_camera_rgb_img"].astype(np.uint8)

        # pre-grasp pose
        status = self.move_eef(pre_grasp_pose, grasp_mat)
        # display side rgb
        rgb = self.obs[3]["left_camera_rgb_img"].astype(np.uint8)
        
        # grasp pose
        status = self.move_eef(grasp_pose, grasp_mat)
        # display side rgb
        rgb = self.obs[3]["left_camera_rgb_img"].astype(np.uint8)

        # close gripper
        self.close_gripper()

        # lift object
        status = self.move_eef(pre_grasp_pose, grasp_mat)
        # display side rgb
        rgb = self.obs[3]["left_camera_rgb_img"].astype(np.uint8)

    def place(self, coord):
        """Places an object."""
        place_pose = coord
        preplace_pose = np.array([coord[0], coord[1], 0.5])
        grasp_mat = R.from_euler('xyz', [0, 180, 0], degrees=True).as_matrix()

        # go to pre-place pose
        self.move_eef(preplace_pose, grasp_mat)
        rgb = self.obs[3]["left_camera_rgb_img"].astype(np.uint8)

        # go to place pose
        self.move_eef(place_pose, grasp_mat)
        rgb = self.obs[3]["left_camera_rgb_img"].astype(np.uint8)

        # open gripper
        self.open_gripper()
        rgb = self.obs[3]["left_camera_rgb_img"].astype(np.uint8)

        # go to pre-place pose
        self.move_eef(preplace_pose, grasp_mat)
        rgb = self.obs[3]["left_camera_rgb_img"].astype(np.uint8)

        # go to default pose
        
    def transporter_pick(self):
        """Picks up an object using the transporter."""
        pass

    def transporter_place(self):
        """Places an object using the transporter."""
        pass
    
    @property
    def props(self) -> dict:
       """
       Gets domain model.
       
       The domain model is a dictionary of objects and their properties.
       """
       # get prop object names
       prop_names = [
           self._sim.physics.model.id2name(i, "geom")
           for i in range(self._sim.physics.model.ngeom)
           if any(keyword in self._sim.physics.model.id2name(i, "geom") for keyword in self.shapes)
       ]
       prop_ids = [self._sim.physics.model.name2id(name, "geom") for name in prop_names]

       # get object information
       prop_positions = self._sim.physics.named.data.geom_xpos[prop_names]
       prop_orientations = self._sim.physics.named.data.geom_xmat[prop_names]
       prop_orientations = [R.from_matrix(mat.reshape((3, 3))).as_quat() for mat in prop_orientations]
       prop_rgba = self._sim.physics.named.model.geom_rgba[prop_names]
       prop_names = [name.split("/")[0] for name in prop_names]

       # get object bounding box information
       def get_bbox(prop_id, segmentation_map):
           """Get the bounding box of an object (PASCAL VOC)."""
           prop_coords = np.argwhere(segmentation_map[:, :, 0] == prop_id)
           bbox_corners = np.array(
               [
                   np.min(prop_coords[:, 0]),
                   np.min(prop_coords[:, 1]),
                   np.max(prop_coords[:, 0]),
                   np.max(prop_coords[:, 1]),
               ]
           )

           return bbox_corners

       # TODO: consider vectorizing this
       segmentation_map = self._sim.physics.render(segmentation=True)
       prop_bbox = []
       for idx in prop_ids:
           bbox = get_bbox(idx, segmentation_map)
           prop_bbox.append(bbox)

       # create a dictionary with all the data
       prop_info = {
           prop_names[i]: {
               "position": prop_positions[i],
               "orientation": prop_orientations[i],
               "rgba": prop_rgba[i],
               "bbox": prop_bbox[i],
           }
           for i in range(len(prop_names))
       }

       return prop_info

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
