"""Generating data for the transporter network."""

import numpy as np
# surpress numpy warnings
np.seterr(divide='ignore', invalid='ignore')

import cv2
import matplotlib.pyplot as plt
import PIL.Image

from rearrangement_benchmark.rearrangement_task import RearrangementTask

from hydra import compose, initialize

TRANSPORTER_CONFIG = compose(config_name="transporter_data_collection")

if __name__=="__main__":
    task = RearrangementTask(cfg = TRANSPORTER_CONFIG)

    obs = task.reset()
    print(obs[3].keys())


    # display rgb
    rgb = obs[3]["overhead_camera_rgb_img"].astype(np.uint8)
    # overlay a red circle centered at [240,320]
    cv2.circle(rgb, (280,320), 10, (0,0,255), -1)
    PIL.Image.fromarray(rgb).show()

    # display depth
    depth = obs[3]["overhead_camera_depth_img"]
    depth_map = cv2.applyColorMap(cv2.normalize(depth, None, 0, 100, cv2.NORM_MINMAX, cv2.CV_8UC1), cv2.COLORMAP_JET)
    cv2.imshow("depth", depth_map)
    cv2.waitKey(10)

    # debug pixel_se2
    world_coords = task.pixel_2_world("overhead_camera", [280,320])
    world_coords += np.array([0,0,0.3])

    status, obs = task.move_eef(world_coords)

    if status:
        # display rgb
        rgb = obs[3]["overhead_camera_rgb_img"].astype(np.uint8)
        # overlay a red circle centered at [240,320]
        cv2.circle(rgb, (0,0), 10, (0,0,255), -1)
        PIL.Image.fromarray(rgb).show()
        

