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
    task = RearrangementTask(cfg = TRANSPORTER_CONFIG, viewer=True)
    obs = task.reset()
    
    #import mujoco 
    #mujoco.viewer.launch(task._sim.physics.model._model, task._sim.physics.data._data)

    # get the pose of a random object
    objects = task.props.keys()
    obj = list(objects)[0]

    # debug pixel_se2
    #world_coords = task.pixel_2_world("overhead_camera", [280,320])
    #print("world_coords: ", world_coords)
    #image_coords = task.world_2_pixel("overhead_camera", world_coords)
    #print("image_coords: ", image_coords)

    # test pick action
    task.pick(obj)
    print("pick action done")

    # test place action
    task.place([0.6, 0.0, 0.3])
    print("place action done")
