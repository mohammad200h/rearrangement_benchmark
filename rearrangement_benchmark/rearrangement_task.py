"""A high-level task API for rearrangement tasks that leverage motion planning."""

import numpy as np
from ikpy.chain import Chain
from dm_robotics.transformations.transformations import mat_to_quat, quat_to_mat, quat_to_euler

import PIL

from rearrangement_benchmark.task import construct_task_env


URDF_PATH = "./models/arms/robot.urdf"

class RearrangementTask(object):
    """A high-level API for performing rearrangement tasks."""

    def __init__(self, cfg=None):
        """Initializes a rearrangement task."""
        # if a config is provided overwrite the default config
        if cfg is not None:
            self._sim, self.config = construct_task_env(cfg)
        else:
            self._sim, self.config = construct_task_env()

        self.shapes = self.config.props.shapes
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
    
    def pixel_2_world(self, camera_name, coords):
        """Returns the world coordinates for a given pixel in camera."""
        width, height = [480, 640] # TODO: read from config
        
        # Investigate the mjcf cause I didn't know what the heck was going on with cameras :)
        #from dm_control.mjcf import export_with_assets
        #mjcf_model = self._sim._env._task.root_entity.mjcf_model
        #export_with_assets(mjcf_model, ".", "test.xml")

        # get camera parameters
        # Note: MoMa employs opencv camera convention where +z faces the scene
        # this is different from mujoco where -z faces the scene
        # this was confusing on first pass over both APIs
        intrinsics = self.obs[3][camera_name + "_intrinsics"]
        
        # manually calculate the intrinsics as a sanity check!!!
        #mj_fovy = self._sim._env.physics.named.model.cam_fovy["overhead_camera/overhead_camera"]
        #half_angle_rad = np.deg2rad(mj_fovy / 2.0)
        #focal_length = (height / 2.0) / np.tan(half_angle_rad)
        #intrinsics = np.array([
        #    [focal_length, 0.0, (width-1) / 2.0, 0.0], 
        #    [0.0, focal_length, (height-1) / 2.0, 0.0], 
        #    [0.0, 0.0, 1.0, 0.0]
        #    ])
        #print("intrinsics: {}".format(intrinsics[:3,:3]))
        
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
       prop_orientations = [mat_to_quat(mat.reshape((3, 3))) for mat in prop_orientations]
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
