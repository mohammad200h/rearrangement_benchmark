"""Generating data for the transporter network."""

import numpy as np

import cv2
import matplotlib.pyplot as plt
import PIL.Image

from dm_robotics.transformations import transformations as tr
from scipy.spatial.transform import Rotation as R
from rearrangement_benchmark.rearrangement_task import RearrangementTask

from hydra import compose, initialize

TRANSPORTER_CONFIG = compose(config_name="transporter_data_collection")

if __name__=="__main__":
    task = RearrangementTask(cfg = TRANSPORTER_CONFIG)

    obs = task.reset()
    print(obs[3].keys())
    
    # get the pose of a random object
    objects = task.props.keys()
    obj = list(objects)[0]
    task.pick(obj)
    #obj_pose = task.props[obj]["position"]
    #obj_quat = task.props[obj]["orientation"]

    # get z component of obj_quat for gripper rotation
    #obj_rot = R.from_quat(obj_quat)
    #obj_rot_mat = obj_rot.as_matrix()
    #obj_rot_z = np.rad2deg(np.arctan2(obj_rot_mat[1,0], obj_rot_mat[0,0]))
    # double check the -45 term
    #grasp_mat = R.from_euler('xyz', [0, 180, obj_rot_z-45], degrees=True).as_matrix()



    # generate grasp pose for obj
    #pre_grasp_pose = np.array([obj_pose[0], obj_pose[1], 0.5])
    #grasp_pose = np.array([obj_pose[0], obj_pose[1], 0.2])
    
    # display side rgb
    #rgb = obs[3]["left_camera_rgb_img"].astype(np.uint8)
    #PIL.Image.fromarray(rgb).show()

    # display depth
    #depth = obs[3]["overhead_camera_depth_img"]
    #depth_map = cv2.applyColorMap(cv2.normalize(depth, None, 0, 100, cv2.NORM_MINMAX, cv2.CV_8UC1), cv2.COLORMAP_JET)
    #cv2.imshow("depth", depth_map)
    #cv2.waitKey(10)

    # debug pixel_se2
    #world_coords = task.pixel_2_world("overhead_camera", [280,320])
    #world_coords += np.array([0,0,0.3])
    

    # pre-grasp pose
    #status, obs = task.move_eef(pre_grasp_pose, grasp_mat)
    # display side rgb
    #rgb = obs[3]["left_camera_rgb_img"].astype(np.uint8)
    #PIL.Image.fromarray(rgb).show()
    
    # grasp pose
    #status, obs = task.move_eef(grasp_pose, grasp_mat)
    # display side rgb
    #rgb = obs[3]["left_camera_rgb_img"].astype(np.uint8)
    #PIL.Image.fromarray(rgb).show()

    # close gripper
    #task.close_gripper()

    # lift object
    #status, obs = task.move_eef(pre_grasp_pose, grasp_mat)
    # display side rgb
    #rgb = obs[3]["left_camera_rgb_img"].astype(np.uint8)
    #PIL.Image.fromarray(rgb).show()


